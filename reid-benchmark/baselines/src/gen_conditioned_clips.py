#!/usr/bin/env python
"""
Baseline Stage 2a (reid env): turn BoT-SORT (or any) tracklets into per-question identity-conditioned
MP4 clips. For each tracked video: CLIP-link tracklets -> global identities; for each question:
ground the referent to a global id, render that identity's marked frames, write a short MP4. These
clips are then answered by each model's STANDARD inference(video_path,...) path (uniform across all
17 models -> no per-model cache-format bugs).

Run: CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 /home/ab260989/.conda/envs/reid/bin/python \
        gen_conditioned_clips.py --tracks_dir results/tracks_botsort --out conditioned/botsort [--limit N]
Outputs: <out>/<vid>__<qid>.mp4  +  <out>/manifest.json  {"vid|qid": "clip path"}
"""
import argparse, glob, json, os
import numpy as np
import cv2
from PIL import Image
import probe_video_judge_v2 as P

ROOT = "/home/ab260989/gen-reid"
CLUSTER_THRESH = 0.30
N_CLIP_FRAMES = 16     # frames written per conditioned clip
CLIP_FPS = 4


def load_bench():
    raw = json.load(open(f"{ROOT}/combined_all_hard_v3_retagged.json")); out = {}
    for v in raw["videos"]:
        for q in v.get("questions", []):
            qid = q.get("question_id")
            if qid:
                out[f"{v['video_id']}|{qid}"] = {"qtext": q.get("question_text", "")}
    return out


def main():
    import torch, decord, open_clip
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks_dir", default="results/tracks_botsort")
    ap.add_argument("--out", default="conditioned/botsort")
    ap.add_argument("--ids_dir", default="", help="if set, use precomputed ReID global-id grouping "
                    "(<ids_dir>/<vid>.json {'groups':{gid:[tid,...]}}) instead of CLIP clustering")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    a2r = json.load(open(f"{ROOT}/video_id_mapping.json")).get("anon_to_real", {})
    bench = load_bench()
    tracked = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{ROOT}/{args.tracks_dir}/*.json"))
    if args.limit:
        tracked = tracked[:args.limit]
    print(f"[plan] {len(tracked)} tracked videos -> {args.out}", flush=True)

    clip, _, prep = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    clip = clip.eval().to(dev); ctok = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    def fimg(pils):
        with torch.no_grad():
            f = torch.nn.functional.normalize(clip.encode_image(torch.stack([prep(p) for p in pils]).to(dev)).float(), dim=-1)
        return f.cpu().numpy()

    def ftxt(t):
        with torch.no_grad():
            f = torch.nn.functional.normalize(clip.encode_text(ctok([t[:300]]).to(dev)).float(), dim=-1)
        return f.cpu().numpy()[0]

    manifest = {}
    for vi, anon in enumerate(tracked):
        trp = f"{ROOT}/{args.tracks_dir}/{anon}.json"
        tr = json.load(open(trp)).get("identities", {})
        vp = P.resolve_video(anon, a2r)
        if not vp or not tr:
            continue
        vr = decord.VideoReader(vp); N = len(vr)
        # per-tracklet CLIP feature
        tl_feat, tl_key = [], []
        for tid, t in tr.items():
            crops = []
            for d in P.spread(t["dets"], 6):
                gf = min(int(d["global_frame"]), N - 1); fr = vr[gf].asnumpy()
                x1, y1, x2, y2 = [max(0, int(v)) for v in d["box"]]
                cr = fr[y1:y2, x1:x2]
                if cr.size:
                    crops.append(Image.fromarray(cr).convert("RGB"))
            if crops:
                tl_feat.append(fimg(crops).mean(0)); tl_key.append(tid)
        if not tl_feat:
            del vr; continue
        F = np.array([f / (np.linalg.norm(f) + 1e-8) for f in tl_feat])
        if args.ids_dir:
            # use the dedicated-ReID global-id grouping; CLIP feats (F) only used for grounding below
            grp = json.load(open(f"{ROOT}/{args.ids_dir}/{anon}.json")).get("groups", {}) \
                  if os.path.exists(f"{ROOT}/{args.ids_dir}/{anon}.json") else {}
            tid2g = {tid: int(g) for g, tids in grp.items() for tid in tids}
            nxt = (max((int(g) for g in grp), default=0)) + 1
            lab = []
            for k in tl_key:
                if k in tid2g:
                    lab.append(tid2g[k])
                else:
                    lab.append(nxt); nxt += 1   # tracklet missing from grouping -> own identity
            lab = np.array(lab)
        else:
            lab = np.array([1]) if len(F) == 1 else fcluster(linkage(pdist(F, "cosine"), "average"), t=CLUSTER_THRESH, criterion="distance")
        gid_dets, gid_feat = {}, {}
        for k, l in zip(tl_key, lab):
            gid_dets.setdefault(int(l), []).extend(tr[k]["dets"])
            gid_feat.setdefault(int(l), []).append(F[tl_key.index(k)])
        gfeat = {l: np.mean(v, 0) for l, v in gid_feat.items()}
        # per question: ground -> render -> mp4
        for k in [kk for kk in bench if kk.split("|")[0] == anon]:
            qf = ftxt(bench[k]["qtext"])
            gl = max(gfeat, key=lambda l: float(np.dot(qf, gfeat[l] / (np.linalg.norm(gfeat[l]) + 1e-8))))
            dets = gid_dets[gl]
            frames = []
            for d in P.spread(dets, N_CLIP_FRAMES):
                gfi = min(int(d["global_frame"]), N - 1)
                frames.append(P.render(vr[gfi].asnumpy(), d["box"]))
            if not frames:
                continue
            clip_path = f"{args.out}/{anon}__{k.split('|')[1]}.mp4"
            h, w = frames[0].height, frames[0].width
            wtr = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS, (w, h))
            for fr in frames:
                wtr.write(cv2.cvtColor(np.array(fr), cv2.COLOR_RGB2BGR))
            wtr.release()
            manifest[k] = clip_path
        del vr
        if (vi + 1) % 10 == 0:
            print(f"  [{vi+1}/{len(tracked)}] clips so far: {len(manifest)}", flush=True)
    json.dump(manifest, open(f"{args.out}/manifest.json", "w"))
    print(f"\n[done] wrote {len(manifest)} conditioned clips + manifest -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
