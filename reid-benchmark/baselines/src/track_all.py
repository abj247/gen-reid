#!/usr/bin/env python
"""
Stage 1 (track env), generalized: track ALL benchmark videos with one of 4 trackers, sharded for
SLURM arrays, resumable (skips videos already tracked). Output: results/tracks_<tracker>/<vid>.json
(same schema as track_videos.py).
Run: /home/.../track/bin/python track_all.py --tracker botsort --shard $SLURM_ARRAY_TASK_ID --nshards 16
Trackers: botsort | bytetrack | deepocsort | strongsort
"""
import argparse, json, os
import numpy as np
import cv2

ROOT = "/home/ab260989/gen-reid"
VIDEO_DIR = "/home/c3-0/datasets/moviechat1k-test"
TARGET_FPS = 3
MAX_FRAMES = 800
ENTITY_CLASSES = [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
COCO = {0: "person", 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
        19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe"}


def resolve(anon, a2r):
    for stem in (a2r.get(anon, anon), anon):
        for e in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
            p = os.path.join(VIDEO_DIR, f"{stem}{e}")
            if os.path.exists(p):
                return p
    return None


def make_tracker(name, reid, fps):
    from boxmot.trackers import BotSort, ByteTrack, DeepOcSort, StrongSort
    if name == "botsort":
        return BotSort(reid_model=reid.model, with_reid=True, frame_rate=fps)
    if name == "bytetrack":
        return ByteTrack(frame_rate=fps)
    if name == "deepocsort":
        return DeepOcSort(reid_model=reid.model, frame_rate=fps)
    if name == "strongsort":
        return StrongSort(reid_model=reid.model)
    raise ValueError(name)


def main():
    import torch
    from ultralytics import YOLO
    from boxmot.reid.core import ReID
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", required=True, choices=["botsort", "bytetrack", "deepocsort", "strongsort"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    out_dir = f"{ROOT}/results/tracks_{args.tracker}"
    os.makedirs(out_dir, exist_ok=True)
    dev = "0" if torch.cuda.is_available() else "cpu"
    a2r = json.load(open(f"{ROOT}/video_id_mapping.json")).get("anon_to_real", {})
    bench = json.load(open(f"{ROOT}/combined_all_hard_v3_retagged.json"))
    vids = [v["video_id"] for v in bench["videos"]]
    vids = [v for i, v in enumerate(vids) if i % args.nshards == args.shard]
    print(f"[shard {args.shard}/{args.nshards}] tracker={args.tracker} | {len(vids)} videos", flush=True)

    yolo = YOLO("yolov8x.pt")
    reid = None if args.tracker == "bytetrack" else ReID(weights="osnet_x0_25_msmt17.pt", device=dev, half=False)

    for vi, anon in enumerate(vids):
        outp = f"{out_dir}/{anon}.json"
        if os.path.exists(outp):
            continue  # resumable
        vp = resolve(anon, a2r)
        if not vp:
            continue
        try:
            cap = cv2.VideoCapture(vp)
            native = cap.get(cv2.CAP_PROP_FPS) or 24.0
            step = max(1, int(round(native / TARGET_FPS)))
            tracker = make_tracker(args.tracker, reid, TARGET_FPS)
            idents, labels = {}, {}
            fi = proc = 0
            while proc < MAX_FRAMES:
                if not cap.grab():
                    break
                if fi % step == 0:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    r = yolo(frame, classes=ENTITY_CLASSES, conf=0.4, verbose=False, device=dev)
                    b = r[0].boxes
                    dets = (np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy().reshape(-1, 1),
                            b.cls.cpu().numpy().reshape(-1, 1)], 1).astype(np.float32)
                            if b is not None and len(b) else np.empty((0, 6), np.float32))
                    a = np.asarray(tracker.update(dets, frame))
                    if a.ndim == 2 and a.shape[1] >= 7:
                        for row in a:
                            tid = int(row[4])
                            idents.setdefault(tid, []).append({"global_frame": fi,
                                "box": [float(row[0]), float(row[1]), float(row[2]), float(row[3])]})
                            labels.setdefault(tid, []).append(int(row[6]))
                    proc += 1
                fi += 1
            cap.release()
            out = {"anon_id": anon, "native_fps": float(native), "target_fps": TARGET_FPS, "identities": {}}
            for tid, dets in idents.items():
                if len(dets) < 3:
                    continue
                lab = COCO.get(max(set(labels[tid]), key=labels[tid].count), "entity")
                out["identities"][f"id_{tid}"] = {"label": lab, "dets": dets}
            json.dump(out, open(outp, "w"))
            if (vi + 1) % 10 == 0:
                print(f"  [{vi+1}/{len(vids)}] {anon}: {len(out['identities'])} tracklets", flush=True)
        except Exception as e:
            print(f"  [err] {anon}: {e}", flush=True)
    print(f"[shard {args.shard}] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
