#!/usr/bin/env python
"""
Stage-1 de-risk probe: QUERY-CONDITIONED KEYFRAME SELECTION (training-free, model-agnostic).
For each question, score a dense pool of candidate frames against the query text with CLIP and keep
the top-K (in temporal order) -> render a short MP4 + manifest, exactly like gen_conditioned_clips.py,
so eval_conditioned.py runs on it unchanged. Tests "smarter frames > uniform frames at equal budget".

Query modes:
  question          : question_text only
  question_options  : question_text + all options
  referent          : question_text + temporal_anchor   (the re-ID-aware variant: bias toward the anchor scene)

CONTROL modes -- neither uses CLIP or the query at all. Added 2026-08-11 because the kf arms had
been compared only against the published uniform-8 row, and that comparison cannot separate three
different explanations. Without both of these the +1.95/+2.14/+2.72 gains cannot be claimed.

  random   : topk sampled from the SAME candidate pool with a per-question seeded RNG.
             Isolates QUERY CONDITIONING. This project has already watched a question-conditioned
             selector lose to its own random control (qcond 28.89 vs qrand 29.31 on the full
             benchmark), so "top-k by CLIP" must be shown to beat "k at random from the same pool"
             before any of it means anything. `kf_referent - kf_random` is THE headline number.

  uniform  : the plain uniform-k frames, rendered through this identical MP4 pipeline.
             Isolates the PIPELINE. The kf arms are evaluated by eval_conditioned.py on rendered
             mp4v clips at CLIP_FPS; the published uniform-8 baseline is evaluated by
             evaluate_vlm_video_v2.py on the SOURCE video. Different decoder, container, and
             resolution history. `kf_uniform - published_uniform8` is the offset that must be
             subtracted from any headline gain.
             Exactness note: with the defaults (n_candidates=64, topk=8) the pool is
             linspace(0, N-1, 64) and 63/7 == 9, so linspace(0, N-1, 8) is an EXACT SUBSET of the
             pool at positions [0, 9, 18, ..., 63]. This mode therefore renders precisely the
             frames the uniform-8 baseline sees, not an approximation of them.

Run (reid env):
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python gen_keyframe_clips.py \
     --query_mode referent --out conditioned_keyframes/referent [--limit N]
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python gen_keyframe_clips.py \
     --query_mode random  --out conditioned_keyframes/random
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python gen_keyframe_clips.py \
     --query_mode uniform --out conditioned_keyframes/uniform8
Outputs: <out>/<vid>__<qid>.mp4 + <out>/manifest.json  {"vid|qid": clip_path}
"""
import argparse, json, os, zlib
import numpy as np
import cv2
from PIL import Image
from benchmark import video_io as P

from persistqa.paths import BENCH_ANON, ID_MAPPING, ROOT  # noqa: E402
N_CANDIDATES = 64     # dense pool sampled uniformly from the video, scored by CLIP
TOPK = 8              # keyframes kept -> matches the uniform-8 baseline budget
CLIP_FPS = 4
N_CHUNKS = 8          # chunk mode: contiguous segments over the candidate pool
CHUNK_TOPK = 2        # chunks retrieved; k frames are split evenly between them
CLIP_MODEL = ("ViT-B-32-quickgelu", "openai")


def load_bench():
    raw = json.load(open(BENCH_ANON)); out = {}
    for v in raw["videos"]:
        for q in v.get("questions", []):
            qid = q.get("question_id")
            if not qid:
                continue
            md = q.get("metadata", {})
            out[f"{v['video_id']}|{qid}"] = {
                "qtext": q.get("question_text", ""),
                "options": " ".join(q.get("options", {}).values()),
                "anchor": md.get("temporal_anchor", "") or "",
            }
    return out


CLIP_MODES = ("question", "question_options", "referent", "chunk")
CONTROL_MODES = ("random", "uniform")


def query_text(rec, mode):
    if mode == "question":
        return rec["qtext"]
    if mode == "question_options":
        return f"{rec['qtext']} {rec['options']}"
    if mode in ("referent", "chunk"):
        # `chunk` reuses the referent query verbatim; the two differ only in HOW the scored pool is
        # turned into a selection, so sharing the query is what makes them comparable.
        return f"{rec['qtext']} {rec['anchor']}".strip()
    raise ValueError(mode)


def select_pool_indices(mode, key, n_pool, topk, sims=None):
    """Which POOL positions to keep, always returned in temporal order.

    One function for all five modes so the three arms cannot drift apart in how many frames they
    keep or how they order them -- the whole comparison rests on every arm feeding exactly `topk`
    frames in ascending time.
    """
    k = min(topk, n_pool)
    if mode == "chunk":
        # CHUNK-RETRIEVAL: rank the N_CHUNKS contiguous segments, then sample evenly INSIDE the
        # winners -- the model-agnostic twin of analysis3/membank/run_membank.py, which splices
        # InternVL embeddings directly and therefore cannot run on any other backbone. Rendering
        # frames instead costs the "memory is never re-decoded" property but buys replication
        # across every backbone in the project, using the SAME kf_random / kf_uniform8 controls.
        #
        # Scored per chunk by MAX, not mean: the evidence window is a median 4.4 s inside a ~24 s
        # chunk, so a chunk earns its place because ONE frame matches. On a single-frame burst the
        # max margin is ~8x the mean margin (measured in the membank selftest).
        if sims is None:
            raise ValueError("chunk mode needs CLIP similarities")
        per = max(1, n_pool // N_CHUNKS)
        n_ch = min(N_CHUNKS, n_pool)
        cs = np.array([sims[c * per:(c + 1) * per].max() if len(sims[c * per:(c + 1) * per])
                       else -1e9 for c in range(n_ch)])
        top = sorted(int(c) for c in np.argsort(-cs)[:CHUNK_TOPK])
        per_ch = [k // len(top)] * len(top)
        for i in range(k - sum(per_ch)):          # spread the remainder, never drop a frame
            per_ch[i] += 1
        out = []
        for c, m in zip(top, per_ch):
            lo, hi = c * per, min((c + 1) * per, n_pool) - 1
            out.extend(int(x) for x in np.linspace(lo, hi, m).astype(int))
        out = sorted(set(out))
        # dedupe can only shrink the set at tiny pools; top up from the best unused positions
        if len(out) < k:
            for c in np.argsort(-sims):
                if int(c) not in out:
                    out.append(int(c))
                    if len(out) == k:
                        break
            out = sorted(out)
        return np.array(out[:k])
    if mode in CLIP_MODES:
        if sims is None:
            raise ValueError(f"{mode} needs CLIP similarities")
        return np.sort(np.argsort(-sims)[:k])
    if mode == "random":
        # Seeded on the QUESTION key, not on a global counter: the control must be reproducible
        # across re-runs and independent of iteration order, and two questions on the same video
        # must get different draws (otherwise this degenerates into a per-video control).
        rng = np.random.RandomState(zlib.crc32(key.encode()) & 0xFFFFFFFF)
        return np.sort(rng.choice(n_pool, size=k, replace=False))
    if mode == "uniform":
        # Exact subset of the pool when (n_pool - 1) % (k - 1) == 0; see the module docstring.
        return np.linspace(0, n_pool - 1, k).astype(int)
    raise ValueError(mode)


def main():
    import torch, decord, open_clip
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_mode", default="referent",
                    choices=list(CLIP_MODES) + list(CONTROL_MODES))
    ap.add_argument("--out", default="conditioned_keyframes/referent")
    ap.add_argument("--n_candidates", type=int, default=N_CANDIDATES)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # CPU fallback: the control modes need no CLIP at all, and even the CLIP modes are cheap
    # enough on CPU (~70 min for the corpus, measured by analysis3/membank/build_clip_index.py) to
    # be worth running there while the GPUs are busy. Hardcoding "cuda" made a CPU run die with
    # "No CUDA GPUs are available" AFTER decoding, which wastes the whole pass.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a2r = json.load(open(ID_MAPPING)).get("anon_to_real", {})
    bench = load_bench()
    vids = sorted({k.split("|")[0] for k in bench})
    if args.limit:
        vids = vids[:args.limit]
    print(f"[plan] {len(vids)} videos | mode={args.query_mode} | {args.n_candidates} candidates -> top{args.topk} -> {args.out}", flush=True)

    # The control modes consult neither CLIP nor the query, so the model is not loaded for them --
    # that also makes it impossible for a control to accidentally use query information.
    needs_clip = args.query_mode in CLIP_MODES
    if needs_clip:
        clip, _, prep = open_clip.create_model_and_transforms(CLIP_MODEL[0], pretrained=CLIP_MODEL[1])
        clip = clip.eval().to(dev); ctok = open_clip.get_tokenizer(CLIP_MODEL[0])
    else:
        clip = prep = ctok = None
        print(f"[control] mode={args.query_mode}: CLIP not loaded, query text never read", flush=True)

    def fimg(pils):
        with torch.no_grad():
            f = torch.nn.functional.normalize(clip.encode_image(torch.stack([prep(p) for p in pils]).to(dev)).float(), dim=-1)
        return f.cpu().numpy()

    def ftxt(t):
        with torch.no_grad():
            f = torch.nn.functional.normalize(clip.encode_text(ctok([t[:300]]).to(dev)).float(), dim=-1)
        return f.cpu().numpy()[0]

    manifest = {}
    for vi, anon in enumerate(vids):
        vp = P.resolve_video(anon, a2r)
        if not vp:
            continue
        try:
            vr = decord.VideoReader(vp); N = len(vr)
        except Exception as e:
            print(f"  [vid err] {anon}: {e}", flush=True); continue
        cand_idx = np.linspace(0, N - 1, min(args.n_candidates, N)).astype(int)
        pil = [Image.fromarray(vr[int(i)].asnumpy()).convert("RGB") for i in cand_idx]
        # Encoded once per video and reused across its questions -- but only when a CLIP mode
        # actually needs it.
        feats = fimg(pil) if needs_clip else None
        for k in [kk for kk in bench if kk.split("|")[0] == anon]:
            sims = (feats @ ftxt(query_text(bench[k], args.query_mode))) if needs_clip else None
            top = select_pool_indices(args.query_mode, k, len(pil), args.topk, sims)
            frames = [pil[i] for i in top]
            clip_path = f"{args.out}/{anon}__{k.split('|')[1]}.mp4"
            w, h = frames[0].width, frames[0].height
            wtr = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS, (w, h))
            for fr in frames:
                wtr.write(cv2.cvtColor(np.array(fr), cv2.COLOR_RGB2BGR))
            wtr.release()
            manifest[k] = clip_path
        del vr
        if (vi + 1) % 10 == 0:
            print(f"  [{vi+1}/{len(vids)}] clips: {len(manifest)}", flush=True)
    json.dump(manifest, open(f"{args.out}/manifest.json", "w"))
    print(f"[done] wrote {len(manifest)} keyframe clips + manifest -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
