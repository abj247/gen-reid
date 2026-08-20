#!/usr/bin/env python3
"""
Video+Text VLM Evaluation v2 - Multi-Model, Per-Video Caching, Checkpointed.

Runs each VLM on the committee-debiased MovieChat-1k benchmark (combined_all_hard_v3
or equivalent) WITH video frames. Caches each video's frames once and runs all of
that video's questions against the cached frames before moving on - amortizes the
expensive video-decode step.

Per-question line in the log:
    [model] [Q i/N] running_acc=X.XX% vs baseline 12.5% (+Y.YYpp) | infer=Z.ZZs | ETA M.Mm

Per-model JSONL checkpoint (predictions.jsonl):
    one record per question; if the job dies, the next launch resumes the same
    model from the next un-processed question without re-decoding any videos.

Output schema is identical to evaluate_vlm_text_only_v2.py so downstream tooling
(compare_text_vs_video.py, analyze_bias_comprehensive.py) works on both.

Usage:
    python evaluate_vlm_video_v2.py \\
        --benchmark combined_all_hard_v3_real_ids.json \\
        --video_dir /home/c3-0/datasets/moviechat1k-test \\
        --output_dir results_video_v2 \\
        --models qwen2.5-vl-3b qwen2.5-vl-7b ...
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

from benchmark.models_registry import create_model


# All 12 models with video support after the recent additions.
DEFAULT_MODELS = [
    "ovis2.5-2b",
    "internvl3-2b",
    "qwen3-vl-real-2b",
    "qwen2.5-vl-3b",
    "qwen3-vl-real-4b",
    "gemma3-4b",
    "qwen2.5-vl-7b",
    "qwen3-vl-real-8b",
    "ovis2.5-9b",
    "internvl3-8b",
    "video-llava",
    "gemma3-12b",
    "internvl3-14b",
]

# 4-bit quant only for the very largest models (still 48GB GPU).
QUANTIZE_4BIT = {"internvl3-14b"}


def jsonl_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


def append_jsonl(path: Path, records: List[Dict[str, Any]]):
    with path.open("a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_benchmark_with_videos(bench_path: Path, video_dir: Path):
    """Return list of (video_id, video_path, [questions])."""
    with bench_path.open() as f:
        bench = json.load(f)
    items = []
    for v in bench.get("videos", []):
        vid = v.get("video_id", "")
        video_path = None
        for ext in (".mp4", ".avi", ".mkv", ".mov", ".webm", ""):
            cand = video_dir / f"{vid}{ext}"
            if cand.is_file():
                video_path = str(cand)
                break
        if video_path is None:
            print(f"WARN: video not found for {vid} in {video_dir}, skipping its questions")
            continue
        items.append((vid, video_path, v.get("questions", [])))
    return items


def resolve_correct_letter(q, options):
    raw = q.get("correct_answer") or q.get("answer") or ""
    letter = raw.strip().upper()[:1] if raw else ""
    if letter and letter in options:
        return letter
    keys = list(options.keys())
    return keys[0] if keys else "A"


def evaluate_one_model(model_name: str, items, output_dir: Path, force: bool,
                       num_frames: int):
    """Per-video frame caching + per-question inference + JSONL checkpoint."""
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_file = model_dir / "predictions.jsonl"

    flat_q = [(vid, vpath, q) for vid, vpath, qs in items for q in qs]
    n_questions = len(flat_q)
    have = jsonl_line_count(pred_file)

    if have == n_questions and not force:
        print(f"[{model_name}] already complete ({have}/{n_questions}) - skipping.",
              flush=True)
        with pred_file.open() as f:
            predictions = [json.loads(line) for line in f]
        display_name = predictions[0].get("model_name", model_name) if predictions else model_name
        return {"model_name": display_name, "predictions": predictions,
                "timestamp": datetime.now().isoformat()}

    if pred_file.exists() and (force or (have != 0 and have != n_questions)):
        if force:
            print(f"[{model_name}] --force: clearing existing predictions ({have} lines).",
                  flush=True)
        else:
            print(f"[{model_name}] partial run found ({have}/{n_questions}) "
                  f"- resuming from question {have}.", flush=True)
            # Resume: rewind index, don't unlink.
    elif have == 0 and pred_file.exists():
        pred_file.unlink()  # empty leftover file

    if force and pred_file.exists():
        pred_file.unlink()
        have = 0

    quant_4bit = model_name in QUANTIZE_4BIT or os.environ.get("REID_FORCE_4BIT") == "1"
    print(f"\n{'='*60}\n[{model_name}] loading (4bit={quant_4bit}, num_frames={num_frames})"
          f"\n{'='*60}", flush=True)
    t0 = time.time()
    try:
        model = create_model(model_name, device="cuda", num_frames=num_frames,
                             quantize_4bit=quant_4bit)
        model.load_model()
    except Exception as e:
        import traceback
        print(f"[{model_name}] LOAD FAILED: {e}", flush=True)
        traceback.print_exc()
        return {"model_name": model_name, "predictions": [], "load_error": str(e),
                "timestamp": datetime.now().isoformat()}

    print(f"[{model_name}] loaded in {time.time()-t0:.1f}s; running {n_questions - have} of "
          f"{n_questions} questions ({have} already done)...", flush=True)

    # Resume: replay disk-saved correct counts so running_acc continues smoothly.
    running_correct = 0
    running_total = 0
    if have > 0:
        with pred_file.open() as f:
            for line in f:
                rec = json.loads(line)
                running_total += 1
                if rec.get("is_correct"):
                    running_correct += 1

    n_opts = len(flat_q[0][2].get("options", {})) if flat_q else 8
    baseline = 100.0 / n_opts
    t1 = time.time()
    last_vid = None
    video_cache = None

    # Iterate per-video so we decode each mp4 only once
    cursor = 0  # global question index
    for vid, vpath, qs in items:
        # Skip ahead if these questions are already in the checkpoint
        if cursor + len(qs) <= have:
            cursor += len(qs)
            continue

        try:
            t_prep = time.time()
            video_cache = model.prepare_video(vpath)
            prep_t = time.time() - t_prep
        except Exception as e:
            print(f"[{model_name}] video prepare FAILED for {vid}: {e}", flush=True)
            # Mark all questions of this video as ERROR and move on
            records = []
            for q in qs:
                if cursor >= have:
                    records.append({
                        "model_name": model.model_name, "video_id": vid,
                        "question_id": q.get("question_id", ""),
                        "capability": q.get("metadata", {}).get("capability", "unknown"),
                        "referral_strategy": q.get("metadata", {}).get("referral_strategy", "unknown"),
                        "difficulty": q.get("metadata", {}).get("difficulty", "Unknown"),
                        "predicted": "ERROR",
                        "correct": resolve_correct_letter(q, q.get("options", {})),
                        "is_correct": False,
                        "error": str(e),
                    })
                cursor += 1
            if records:
                append_jsonl(pred_file, records)
            continue

        for q in qs:
            if cursor < have:
                cursor += 1
                continue

            options = q.get("options", {})
            correct_letter = resolve_correct_letter(q, options)

            t_inf = time.time()
            try:
                predicted = model.inference_with_cache(
                    video_cache, q.get("question_text") or q.get("question", ""), options
                )
            except torch.cuda.OutOfMemoryError as oom:
                torch.cuda.empty_cache()
                predicted = "OOM"
                print(f"[{model_name}] OOM at vid={vid} q={q.get('question_id')}",
                      flush=True)
            except Exception as e:
                predicted = "ERROR"
                print(f"[{model_name}] infer error at vid={vid} q={q.get('question_id')}: {e}",
                      flush=True)
            infer_t = time.time() - t_inf

            is_correct = predicted == correct_letter
            running_total += 1
            if is_correct:
                running_correct += 1

            record = {
                "model_name": model.model_name, "video_id": vid,
                "question_id": q.get("question_id", ""),
                "capability": q.get("metadata", {}).get("capability", "unknown"),
                "referral_strategy": q.get("metadata", {}).get("referral_strategy", "unknown"),
                "difficulty": q.get("metadata", {}).get("difficulty", "Unknown"),
                "predicted": predicted, "correct": correct_letter,
                "is_correct": is_correct,
            }
            append_jsonl(pred_file, [record])
            cursor += 1

            run_acc = 100.0 * running_correct / max(running_total, 1)
            delta = run_acc - baseline
            sign = "+" if delta >= 0 else ""
            elapsed_total = time.time() - t1
            done_since_resume = running_total - (have)
            rate = max(done_since_resume, 1) / max(elapsed_total, 1e-3)
            eta_min = (n_questions - cursor) / max(rate, 1e-6) / 60.0
            if cursor % 5 == 0 or cursor <= 5 or cursor == n_questions:
                print(
                    f"  [{model_name}] [{cursor}/{n_questions}] "
                    f"running_acc={run_acc:5.2f}% ({running_correct}/{running_total}) "
                    f"vs 12.5% ({sign}{delta:.2f}pp) "
                    f"| infer={infer_t:.2f}s | {rate:.2f} q/s | ETA {eta_min:.1f}m",
                    flush=True,
                )

        del video_cache
        torch.cuda.empty_cache()

    elapsed = time.time() - t1
    final_acc = 100.0 * running_correct / max(running_total, 1)
    print(
        f"[{model_name}] DONE in {elapsed:.1f}s "
        f"| FINAL acc={final_acc:.2f}% ({running_correct}/{running_total}) "
        f"vs baseline {baseline:.1f}%",
        flush=True,
    )

    with pred_file.open() as f:
        predictions = [json.loads(line) for line in f]

    display_name = model.model_name
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {"model_name": display_name, "predictions": predictions,
            "timestamp": datetime.now().isoformat()}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--video_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--limit_videos", type=int, default=None,
                   help="Cap to first N videos (smoke test)")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    if os.path.exists(hf_home):
        free_gb = shutil.disk_usage(hf_home).free / 1e9
        print(f"HF cache: {hf_home} ({free_gb:.0f} GB free)", flush=True)

    print(f"\n{'='*60}\nVIDEO+TEXT BIAS EVAL v2\n{'='*60}", flush=True)
    print(f"Benchmark: {args.benchmark}", flush=True)
    print(f"Video dir: {args.video_dir}", flush=True)

    items = load_benchmark_with_videos(Path(args.benchmark), Path(args.video_dir))
    if args.limit_videos:
        items = items[:args.limit_videos]
    n_videos = len(items)
    n_questions = sum(len(qs) for _, _, qs in items)
    print(f"Videos: {n_videos}, Questions: {n_questions}", flush=True)

    models = args.models or DEFAULT_MODELS
    print(f"Models ({len(models)}): {', '.join(models)}", flush=True)
    print(f"Output: {out}\n", flush=True)

    all_results: Dict[str, Dict[str, Any]] = {}
    n_opts = len(items[0][2][0].get("options", {})) if items and items[0][2] else 8
    baseline = 100.0 / n_opts

    for idx, mn in enumerate(models, start=1):
        print(f"\n>>> MODEL {idx}/{len(models)}: {mn} <<<", flush=True)
        result = evaluate_one_model(mn, items, out, args.force, args.num_frames)
        all_results[result["model_name"]] = result

        print(f"\n{'-'*60}\nLEADERBOARD after {idx}/{len(models)} "
              f"(random baseline {baseline:.1f}%):\n{'-'*60}", flush=True)
        rows = []
        for nm, r in all_results.items():
            preds = r.get("predictions", [])
            if not preds:
                rows.append((nm, 0.0, 0, 0)); continue
            corr = sum(1 for p in preds if p.get("is_correct"))
            rows.append((nm, 100.0 * corr / len(preds), corr, len(preds)))
        rows.sort(key=lambda r: -r[1])
        for nm, acc, corr, tot in rows:
            d = acc - baseline
            s = "+" if d >= 0 else ""
            print(f"  {nm:30s}  {acc:6.2f}%  ({corr}/{tot})  vs baseline ({s}{d:.2f}pp)",
                  flush=True)

    raw_path = out / "raw_results.json"
    with raw_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAggregated raw_results.json -> {raw_path}", flush=True)
    print(f"\n{'='*60}\nDONE\n{'='*60}", flush=True)


if __name__ == "__main__":
    main()
