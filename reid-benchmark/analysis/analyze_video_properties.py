#!/usr/bin/env python3
"""
MOT-proxy video-property analysis. For each video derive cheap structural
properties that proxy for re-ID difficulty WITHOUT ground-truth tracks:
  - duration (s), n_frames
  - n_scene_cuts  (PySceneDetect ContentDetector)
  - max_people    (max persons detected in any sampled frame, YOLOv8n)
  - mean_people   (mean persons across sampled frames)
  - id_switch_opportunity = n_scene_cuts * max_people  (cuts x crowding)

Output: results/video_properties.json  ({anon_video_id: {props}}).
Downstream we regress item difficulty (from per_question.csv) on these.

Runs in the `reid` env. Uses anon<->real id mapping so output is keyed by anon id.

Usage:
    python analyze_video_properties.py --video_dir /home/c3-0/datasets/moviechat1k-test \
        --mapping video_id_mapping.json --out results/video_properties.json
"""
import argparse, json, time
from pathlib import Path
import numpy as np


def count_scene_cuts(path):
    from scenedetect import detect, ContentDetector
    try:
        scenes = detect(path, ContentDetector())
        return max(len(scenes) - 1, 0)
    except Exception:
        return -1


def people_stats_and_variability(path, yolo, clip_model, clip_proc, device, n_sample=8):
    """Returns (max_people, mean_people, clip_variability).
    clip_variability = mean pairwise cosine distance between CLIP image
    embeddings of the sampled frames - a coarse, identity-AGNOSTIC proxy for
    how much the scene's appearance changes. (Identity-specific appearance
    change requires a tracker and is left to the method.)"""
    import decord, torch
    from PIL import Image
    try:
        vr = decord.VideoReader(path)
        idx = np.linspace(0, len(vr) - 1, n_sample).astype(int)
        frames = vr.get_batch(idx).asnumpy()  # (n,H,W,3) RGB
        counts = []
        for f in frames:
            res = yolo.predict(f[:, :, ::-1], classes=[0], verbose=False, conf=0.35)
            counts.append(int((res[0].boxes.cls == 0).sum()) if res else 0)
        # CLIP embeddings
        pil = [Image.fromarray(f) for f in frames]
        inp = clip_proc(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip_model.get_image_features(**inp)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        sim = (emb @ emb.T).cpu().numpy()
        n = sim.shape[0]
        off = sim[np.triu_indices(n, k=1)]
        variability = float(1.0 - off.mean()) if off.size else 0.0
        return max(counts), float(np.mean(counts)), round(variability, 4)
    except Exception:
        return -1, -1.0, -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", default="/home/c3-0/datasets/moviechat1k-test")
    ap.add_argument("--mapping", default="video_id_mapping.json")
    ap.add_argument("--bench", default="combined_all_hard_v3_real_ids.json")
    ap.add_argument("--out", default="results/video_properties.json")
    args = ap.parse_args()

    real_to_anon = json.load(open(args.mapping))["real_to_anon"]
    bench = json.load(open(args.bench))
    real_ids = [v["video_id"] for v in bench["videos"]]

    from ultralytics import YOLO
    import torch
    from transformers import CLIPModel, CLIPProcessor
    yolo = YOLO("yolov8n.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    vdir = Path(args.video_dir)
    out = {}
    t0 = time.time()
    for i, rid in enumerate(real_ids):
        vpath = None
        for ext in (".mp4", ".avi", ".mkv", ".mov", ".webm", ""):
            cand = vdir / f"{rid}{ext}"
            if cand.is_file():
                vpath = str(cand); break
        anon = real_to_anon.get(rid, rid)
        if vpath is None:
            out[anon] = {"error": "missing"}; continue
        import decord
        try:
            vr = decord.VideoReader(vpath)
            nfr = len(vr); fps = float(vr.get_avg_fps()) or 25.0
            dur = nfr / fps
        except Exception:
            dur = nfr = -1
        cuts = count_scene_cuts(vpath)
        mx, mn, varb = people_stats_and_variability(vpath, yolo, clip_model, clip_proc, device)
        out[anon] = {
            "real_id": rid, "duration_s": round(dur, 1), "n_frames": nfr,
            "n_scene_cuts": cuts, "max_people": mx, "mean_people": round(mn, 2),
            "clip_variability": varb,
            "id_switch_opportunity": (cuts * mx) if (cuts >= 0 and mx >= 0) else -1,
        }
        if (i + 1) % 25 == 0 or i == len(real_ids) - 1:
            print(f"  [{i+1}/{len(real_ids)}] {(i+1)/max(time.time()-t0,1e-3):.2f} vid/s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Wrote {args.out} for {len(out)} videos", flush=True)


if __name__ == "__main__":
    main()
