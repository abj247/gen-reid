#!/usr/bin/env python
"""
Baseline Stage 2b: answer the MCQ on identity-conditioned clips with ANY model, via the model's
STANDARD inference(video_path,...) path (uniform across all create_model models -> no cache bugs).
Resumable. Writes results_baseline/<pipeline>/<model>/predictions.jsonl with the same schema as
results_video_v2 so raw-vs-conditioned comparison is a simple join.

Run (reid env example):
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python -m benchmark.evaluation.run_manifest \
      --model internvl3-8b --clips conditioned/botsort --pipeline botsort [--limit N]
Launch with the videochat-flash env python for videochat-flash-2b/7b.
"""
import argparse
import json
import os

from benchmark.models_registry import create_model
from persistqa.paths import BENCH_ANON, RESULTS_DIR, ROOT


def load_bench():
    raw = json.load(open(BENCH_ANON)); out = {}
    for v in raw["videos"]:
        for q in v.get("questions", []):
            qid = q.get("question_id")
            if qid:
                md = q.get("metadata", {})
                out[f"{v['video_id']}|{qid}"] = {"options": q.get("options", {}),
                    "gold": (q.get("correct_answer") or "").strip().upper()[:1],
                    "qtext": q.get("question_text", ""),
                    "capability": md.get("capability", "?"), "reid": md.get("reid_canonical", "?")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--clips", default="conditioned/botsort")
    ap.add_argument("--pipeline", default="botsort")
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    bench = load_bench()
    manifest = json.load(open(f"{ROOT}/{args.clips}/manifest.json"))
    keys = sorted(manifest)
    if args.limit:
        keys = keys[:args.limit]

    outdir = f"{RESULTS_DIR}/{args.pipeline}/{args.model}"
    os.makedirs(outdir, exist_ok=True)
    outp = f"{outdir}/predictions.jsonl"
    done = set()
    if os.path.exists(outp):
        for line in open(outp):
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                pass
    todo = [k for k in keys if k in bench and bench[k]["gold"] and k not in done]
    print(f"[plan] model={args.model} pipeline={args.pipeline} | {len(todo)} to do ({len(done)} done)", flush=True)

    quant = (args.model == "internvl3-14b")
    model = create_model(args.model, device="cuda", num_frames=args.num_frames,
                         max_pixels=(256, 256), quantize_4bit=quant)
    model.load_model()

    n = c = 0
    with open(outp, "a") as fh:
        for i, k in enumerate(todo):
            b = bench[k]; clip = manifest[k]
            if not os.path.exists(clip):
                continue
            try:
                pred = model.inference(clip, b["qtext"], b["options"])
            except Exception as e:
                print(f"  err {k}: {e}", flush=True); continue
            pred = (pred or "?").strip().upper()[:1]
            ok = (pred == b["gold"])
            n += 1; c += int(ok)
            fh.write(json.dumps({"key": k, "model": args.model, "pipeline": args.pipeline,
                                 "video_id": k.split("|")[0], "question_id": k.split("|")[1],
                                 "capability": b["capability"], "reid": b["reid"],
                                 "predicted": pred, "correct": b["gold"], "is_correct": ok}) + "\n")
            fh.flush()
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(todo)}] running acc={c/max(n,1):.3f}", flush=True)
    print(f"\n[done] {args.model}/{args.pipeline}: {c}/{n} = {c/max(n,1):.3f} (conditioned)  -> {outp}", flush=True)


if __name__ == "__main__":
    main()
