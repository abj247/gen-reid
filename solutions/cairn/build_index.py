#!/usr/bin/env python
"""MEMORY BANK, write phase 1/2: a CLIP retrieval index over video chunks.

The design, and why it is not QuestMem again
--------------------------------------------
QuestMem (analysis3/questmem/run_questmem.py) chunked the video into 8, asked the ANSWERING VLM to
rate each chunk ("Score: N"), kept the top-2, and then THREW THE VISUAL ENCODINGS AWAY and
re-decoded 16 fresh frames. Two things killed it:

  * the scout hit the true evidence window only 37.1% of the time, against a 52% break-even, and
    top-2-by-score never beat top-2-at-random (+0.66, p=0.70) -- the scoring, i.e. the entire idea,
    was never shown to do anything;
  * it was not a memory system at all. It cached TEXT NOTES; nothing visual survived the write.

This module fixes the first half. `analysis3/memory/vtm_qcond.py` had already tried
similarity retrieval using the VLM's OWN embedding space -- mean LLM input-embedding of the
question against mean bank token per frame -- and it LOST to its random control (qcond 28.89 vs
qrand 29.31, MDE 0.79 on n=3,299). That is the expected outcome: post-projector visual tokens and
text-token embeddings share a numeric space but were never trained to be comparable across
modalities, so cosine similarity between them is close to noise.

CLIP's space WAS trained for exactly that. The one method in this project that has ever beaten its
baseline on multiple backbones is CLIP keyframe selection (+1.95 / +2.01 / +2.65 on Qwen2.5-VL-7B /
InternVL3-14B / Ovis2.5-9B), and the earlier retrieval gate measured free question-CLIP retrieval
at 55.5% chunk hit versus the VLM self-scout's 37.1%. So the retrieval signal here is CLIP, and the
VLM is only ever asked to answer -- never to rank.

What is stored
--------------
Per video: the CLIP image embedding of EVERY sampled frame, plus its chunk id and frame index.
Per question: nothing (query embeddings are computed at read time, which is microseconds).

Storing per-FRAME rather than per-chunk embeddings is deliberate: it lets the read phase try both
pooling rules without re-encoding a single video. `mean` is the smooth choice; `max` is the right
one if the evidence is a short burst inside a long chunk, which is exactly this benchmark's shape
(median evidence window 4.4 s inside a ~24 s chunk). Both are one line at read time.

Cost: 8 chunks x 8 frames x 512 floats x 4 B = 131 KB per video, ~59 MB for the corpus.

Run (reid env, 1 GPU):
  python analysis3/membank/build_clip_index.py --out analysis3/membank/index [--limit N]
"""
import argparse
import json
import os
import sys

import numpy as np

from persistqa.paths import BENCH_REAL, ROOT, VIDEO_DIR  # noqa: E402

N_CHUNKS = 8            # identical to run_questmem.py:56, so chunk ids mean the same thing
FRAMES_PER_CHUNK = 8    # 64 sampled frames per video, same pool size as gen_keyframe_clips.py
CLIP_MODEL = ("ViT-B-32-quickgelu", "openai")   # same weights as gen_keyframe_clips.py


def chunk_bounds(n_total, n_chunks=N_CHUNKS):
    """[(lo, hi)] inclusive frame bounds. Copy of run_questmem.py:109-112 / chunk_alloc.py."""
    b = np.linspace(0, n_total, n_chunks + 1).astype(int)
    return [(int(b[i]), int(max(b[i + 1] - 1, b[i]))) for i in range(n_chunks)]


def chunk_frame_indices(n_total, n_chunks=N_CHUNKS, per_chunk=FRAMES_PER_CHUNK):
    """Per chunk: `per_chunk` uniform frame indices inside it. Returns (indices, chunk_id) arrays."""
    idx, cid = [], []
    for c, (lo, hi) in enumerate(chunk_bounds(n_total, n_chunks)):
        f = np.linspace(lo, hi, per_chunk).astype(int)
        idx.extend(int(i) for i in f)
        cid.extend([c] * per_chunk)
    return np.array(idx), np.array(cid)


def main():
    import torch
    import decord
    import open_clip
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "solutions" / "cairn" / "index"))
    ap.add_argument("--bench", default=str(BENCH_REAL))
    ap.add_argument("--video_dir", default=str(VIDEO_DIR))
    ap.add_argument("--n_chunks", type=int, default=N_CHUNKS)
    ap.add_argument("--per_chunk", type=int, default=FRAMES_PER_CHUNK)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw = json.load(open(args.bench))
    vids = []
    for v in raw.get("videos", []):
        vp = v.get("video_path")
        if not vp:
            cand = os.path.join(args.video_dir, f"{v['video_id']}.mp4")
            vp = cand if os.path.isfile(cand) else None
        if vp:
            vids.append((v["video_id"], vp))
    # nine video_ids appear as 2-3 separate benchmark entries; the INDEX is per video, so dedupe
    # here or the same video is encoded twice and the second write silently wins.
    seen, uniq = set(), []
    for vid, vp in vids:
        if vid not in seen:
            seen.add(vid)
            uniq.append((vid, vp))
    vids = uniq
    if args.limit:
        vids = vids[: args.limit]
    print(f"[index] {len(vids)} unique videos | {args.n_chunks} chunks x {args.per_chunk} frames "
          f"-> {args.out}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clip, _, prep = open_clip.create_model_and_transforms(CLIP_MODEL[0], pretrained=CLIP_MODEL[1])
    clip = clip.eval().to(dev)

    n_done = n_skip = 0
    for i, (vid, vp) in enumerate(vids):
        out_p = os.path.join(args.out, f"{vid}.npz")
        if os.path.exists(out_p):
            n_skip += 1
            continue
        try:
            vr = decord.VideoReader(vp, num_threads=2)
            n_total = len(vr)
            fps = float(vr.get_avg_fps()) or 25.0
        except Exception as e:
            print(f"  [vid err] {vid}: {str(e)[:90]}", flush=True)
            continue
        idx, cid = chunk_frame_indices(n_total, args.n_chunks, args.per_chunk)
        try:
            arr = vr.get_batch([int(x) for x in idx]).asnumpy()
        except Exception as e:
            print(f"  [decode err] {vid}: {str(e)[:90]}", flush=True)
            del vr
            continue
        pil = [Image.fromarray(a).convert("RGB") for a in arr]
        with torch.no_grad():
            feats = torch.nn.functional.normalize(
                clip.encode_image(torch.stack([prep(p) for p in pil]).to(dev)).float(), dim=-1
            ).cpu().numpy().astype(np.float32)
        np.savez_compressed(out_p, feats=feats, frame_idx=idx.astype(np.int32),
                            chunk_id=cid.astype(np.int16), n_total=np.int64(n_total),
                            fps=np.float32(fps))
        del vr, pil, arr
        n_done += 1
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(vids)}] wrote {n_done}, skipped {n_skip}", flush=True)
    print(f"[index] done: wrote {n_done}, skipped {n_skip} already present -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
