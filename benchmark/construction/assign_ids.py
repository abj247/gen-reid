#!/usr/bin/env python
"""
Baseline Stage-A (ReID-model identity assignment): turn a tracker's within-shot tracklets into
GLOBAL identities using a *dedicated ReID model* (instead of CLIP). For each tracked video:
  per-tracklet ReID feature (mean over sampled crops) -> agglomerative cosine cluster -> global id.
Writes results/ids_<reid>__<tracker>/<vid>.json = {"groups": {global_id: [tracklet_id,...]}}.
gen_conditioned_clips.py then consumes this via --ids_dir (CLIP only used for referent grounding).

Backends:
  osnet     -> boxmot ReID (osnet_x0_25_msmt17), run in the `track` conda env.
  clipreid  -> CLIP-ReID  (run in `reid` env; needs repo+weights)        [added later]
  transreid -> TransReID  (run in `reid` env; needs repo+weights)        [added later]
  solider   -> SOLIDER    (run in `reid` env; needs repo+weights)        [added later]

Run (tracking environment): python -m benchmark.construction.assign_ids \
        --reid osnet --tracks_dir results/tracks_botsort --out results/ids_osnet__botsort \
        --shard $SLURM_ARRAY_TASK_ID --nshards 16
"""
import argparse, glob, json, os
import numpy as np
import cv2

import os as _os
from pathlib import Path as _Path
ROOT = _os.environ.get("PERSISTQA_ROOT") or str(_Path(__file__).resolve().parents[2])
VIDEO_DIR = "/home/c3-0/datasets/moviechat1k-test"
CLUSTER_THRESH = 0.30   # same as the CLIP-linking baseline, for apples-to-apples
MAX_CROPS = 6           # crops sampled per tracklet to form its feature


def resolve(anon, a2r):
    for stem in (a2r.get(anon, anon), anon):
        for e in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
            p = os.path.join(VIDEO_DIR, f"{stem}{e}")
            if os.path.exists(p):
                return p
    return None


def spread(dets, k):
    if len(dets) <= k:
        return dets
    idx = np.linspace(0, len(dets) - 1, k).astype(int)
    return [dets[i] for i in idx]


# Upstream re-identification checkouts. See docs/EXTERNAL_MODELS.md.
_EXT = _os.environ.get("PERSISTQA_EXTERNAL") or _os.path.join(ROOT, "external")
REPO = {"clipreid": _os.path.join(_EXT, "CLIP-ReID"),
        "transreid": _os.path.join(_EXT, "TransReID"),
        "solider": _os.path.join(_EXT, "SOLIDER-REID")}
CKPT = {"clipreid": f"{ROOT}/weights/clipreid_market_vit.pth",
        "transreid": f"{ROOT}/weights/transreid_market_vit.pth",
        "solider": f"{ROOT}/weights/solider_market_swinbase.pth"}
INSIZE = {"clipreid": [256, 128], "transreid": [256, 128], "solider": [384, 128]}


def _crop_extractor(name):
    """Crop-based ReID models (CLIP-ReID / TransReID / SOLIDER): build model in `reid` env,
    return feats_for_frame(frame_bgr, boxes_xyxy) -> (N, D) raw features (loop L2-normalizes)."""
    import sys, types
    import torch
    import torchvision.transforms as T
    from PIL import Image
    sys.path.insert(0, REPO[name])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = INSIZE[name]
    tf = T.Compose([T.Resize([H, W]), T.ToTensor(),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    if name == "solider":
        # SOLIDER's swin backbone imports mmcv.runner.load_checkpoint at module load; only used by
        # init_weights (we load the full ReID checkpoint ourselves), so stub it to avoid the mmcv dep.
        if "mmcv" not in sys.modules:
            m = types.ModuleType("mmcv"); r = types.ModuleType("mmcv.runner")
            r.load_checkpoint = lambda *a, **k: None
            m.runner = r; sys.modules["mmcv"] = m; sys.modules["mmcv.runner"] = r

    from config import cfg          # each repo exposes its own default CfgNode
    if hasattr(cfg, "defrost"):
        cfg.defrost()
    cfg.INPUT.SIZE_TRAIN = [H, W]; cfg.INPUT.SIZE_TEST = [H, W]
    cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]; cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
    cfg.MODEL.PRETRAIN_CHOICE = "self"; cfg.MODEL.PRETRAIN_PATH = ""
    cfg.MODEL.SIE_CAMERA = False; cfg.MODEL.SIE_VIEW = False; cfg.MODEL.JPM = False
    cfg.TEST.FEAT_NORM = "yes"

    if name == "clipreid":
        from model.make_model_clipreid import make_model as mk
        cfg.MODEL.NAME = "ViT-B-16"; cfg.MODEL.STRIDE_SIZE = [16, 16]
        cfg.TEST.NECK_FEAT = "after"; cfg.DATASETS.NAMES = "market1501"
        model = mk(cfg, num_class=751, camera_num=1, view_num=1)
        fwd = lambda x: model(x)
    elif name == "transreid":
        from model.make_model import make_model as mk
        cfg.MODEL.NAME = "transformer"
        cfg.MODEL.TRANSFORMER_TYPE = "vit_base_patch16_224_TransReID"
        cfg.MODEL.STRIDE_SIZE = [16, 16]; cfg.MODEL.NECK = "bnneck"
        cfg.TEST.NECK_FEAT = "before"
        model = mk(cfg, num_class=751, camera_num=0, view_num=0)
        fwd = lambda x: model(x, cam_label=None, view_label=None)
    elif name == "solider":
        from model.make_model import make_model as mk
        cfg.MODEL.NAME = "transformer"
        cfg.MODEL.TRANSFORMER_TYPE = "swin_base_patch4_window7_224"
        cfg.MODEL.SEMANTIC_WEIGHT = 0.2; cfg.MODEL.NECK = "bnneck"
        cfg.TEST.NECK_FEAT = "before"
        model = mk(cfg, num_class=751, camera_num=0, view_num=0, semantic_weight=0.2)
        def fwd(x):
            out = model(x)
            return out[0] if isinstance(out, (tuple, list)) else out
    else:
        raise ValueError(name)

    model.load_param(CKPT[name])
    model.eval().to(dev)

    def feats_for_frame(frame_bgr, boxes_xyxy):
        ims = []
        for b in boxes_xyxy:
            x1, y1, x2, y2 = [max(0, int(v)) for v in b]
            c = frame_bgr[y1:y2, x1:x2]
            if c.size == 0:
                c = np.zeros((8, 8, 3), np.uint8)
            ims.append(tf(Image.fromarray(c[:, :, ::-1]).convert("RGB")))  # BGR->RGB
        with torch.no_grad():
            f = fwd(torch.stack(ims).to(dev))
        return np.asarray(f.detach().cpu().float().numpy(), dtype=np.float32)
    return feats_for_frame


def build_extractor(name):
    """Return feats_for_frame(frame_bgr, boxes_xyxy) -> (N, D) raw features (loop L2-normalizes)."""
    if name == "osnet":
        from boxmot.reid.core import ReID
        import torch
        dev = "0" if torch.cuda.is_available() else "cpu"
        reid = ReID(weights="osnet_x0_25_msmt17.pt", device=dev, half=False)
        def feats_for_frame(frame_bgr, boxes_xyxy):
            arr = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
            return np.asarray(reid.model.get_features(arr, frame_bgr), dtype=np.float32)
        return feats_for_frame
    if name in ("clipreid", "transreid", "solider"):
        return _crop_extractor(name)
    raise ValueError(f"reid backend {name} not wired")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reid", default="osnet")
    ap.add_argument("--tracks_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thresh", type=float, default=CLUSTER_THRESH,
                    help="agglomerative cosine-distance threshold; SOLIDER features are anisotropic "
                         "(~6x compressed) so use a smaller value (~0.05) than CLIP/OSNet/TransReID (0.30)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    a2r = json.load(open(f"{ROOT}/video_id_mapping.json")).get("anon_to_real", {})
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    tracked = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{ROOT}/{args.tracks_dir}/*.json"))
    if args.limit:
        tracked = tracked[:args.limit]
    tracked = [v for i, v in enumerate(tracked) if i % args.nshards == args.shard]
    print(f"[shard {args.shard}/{args.nshards}] reid={args.reid} {args.tracks_dir} -> {args.out} | {len(tracked)} videos", flush=True)

    feats_for_frame = build_extractor(args.reid)

    for vi, anon in enumerate(tracked):
        outp = f"{args.out}/{anon}.json"
        if os.path.exists(outp):
            continue
        tr = json.load(open(f"{ROOT}/{args.tracks_dir}/{anon}.json")).get("identities", {})
        vp = resolve(anon, a2r)
        if not vp or not tr:
            json.dump({"groups": {}}, open(outp, "w")); continue
        # collect, per frame, the (tracklet, box) crops we need -> read each frame once
        per_frame = {}   # frame_idx -> list of (tid, box)
        for tid, t in tr.items():
            for d in spread(t["dets"], MAX_CROPS):
                per_frame.setdefault(int(d["global_frame"]), []).append((tid, d["box"]))
        cap = cv2.VideoCapture(vp)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        tl_feats = {}    # tid -> list of feature vectors
        for fidx in sorted(per_frame):
            if nframes and fidx >= nframes:
                fidx_use = nframes - 1
            else:
                fidx_use = fidx
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx_use)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            items = per_frame[fidx]
            boxes = [it[1] for it in items]
            try:
                F = feats_for_frame(frame, boxes)   # (N, D)
            except Exception as e:
                continue
            for (tid, _), fv in zip(items, F):
                n = np.linalg.norm(fv) + 1e-8
                tl_feats.setdefault(tid, []).append(fv / n)
        cap.release()

        tl_key = [k for k in tr if tl_feats.get(k)]
        if not tl_key:
            json.dump({"groups": {}}, open(outp, "w")); continue
        Fmat = np.array([np.mean(tl_feats[k], 0) for k in tl_key])
        Fmat = Fmat / (np.linalg.norm(Fmat, axis=1, keepdims=True) + 1e-8)
        if len(Fmat) == 1:
            lab = np.array([1])
        else:
            lab = fcluster(linkage(pdist(Fmat, "cosine"), "average"), t=args.thresh, criterion="distance")
        groups = {}
        for k, l in zip(tl_key, lab):
            groups.setdefault(int(l), []).append(k)
        json.dump({"reid": args.reid, "tracker_dir": args.tracks_dir, "groups": groups}, open(outp, "w"))
        if (vi + 1) % 10 == 0:
            print(f"  [{vi+1}/{len(tracked)}] {anon}: {len(tl_key)} tracklets -> {len(groups)} ids", flush=True)
    print(f"[shard {args.shard}] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
