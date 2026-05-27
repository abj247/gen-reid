#!/usr/bin/env python3
"""
Text-Only Bias Eval v2 — Multi-Model, Batched, Resumable.

Runs each VLM's language path on the (anonymized) MovieChat-1k merged
debiased benchmark with NO video/image input. Above-baseline accuracy
(12.5% for 8 options) flags textual/prior bias.

Differences vs v1 (evaluate_vlm_bias_text_only.py):
  - One script invocation evaluates the whole model matrix sequentially.
  - Per-model batched inference where the underlying processor supports it
    (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Gemma3). Ovis / InternVL3 stay batch_size=1.
  - JSONL checkpoint per model: crashes don't lose work.
  - Resume: model is skipped if its JSONL is already complete (line count matches).
  - max_new_tokens=8 across the board (MCQ answer is one letter) — large KV
    savings; 14B / 12B fit at small batch with 4-bit quantization on 16GB.
  - Final aggregate raw_results.json keeps the v1 schema for downstream
    consumers (compare_text_vs_video.py, analyze_bias_comprehensive.py).

Usage:
    python evaluate_vlm_text_only_v2.py \\
        --benchmark combined_all_text_only.json \\
        --output_dir results_text_only_v2

    # Smoke test with one model on a subset:
    python evaluate_vlm_text_only_v2.py \\
        --benchmark combined_all_text_only.json \\
        --models qwen2.5-vl-3b --limit 50 \\
        --output_dir results_text_only_smoke
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

from vlm_models import create_model


def load_benchmark_questions(benchmark_path):
    """Flatten the benchmark JSON into a list of normalized question dicts.

    Handles both the anonymized and original schema: question text under
    `question_text` or `question`, answer under `correct_answer` or `answer`,
    options as a dict (A/B/...) or a list (converted to A/B/...).
    """
    with open(benchmark_path) as f:
        data = json.load(f)

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    flat = []
    for video in data.get("videos", []):
        video_id = video.get("video_id", "")
        for q in video.get("questions", []):
            meta = q.get("metadata", {})
            raw_options = q.get("options", {})
            if isinstance(raw_options, list):
                options = {labels[i]: opt for i, opt in enumerate(raw_options) if i < len(labels)}
            else:
                options = raw_options
            flat.append({
                "video_id": video_id,
                "question_id": q.get("question_id", ""),
                "question_text": q.get("question_text", q.get("question", "")),
                "options": options,
                "correct_answer": (q.get("correct_answer") or q.get("answer", "")).strip().upper(),
                "capability": meta.get("capability", "unknown"),
                "referral_strategy": meta.get("referral_strategy", "unknown"),
                "difficulty": meta.get("difficulty", "Unknown"),
            })
    return flat


# Default 12-model matrix (4 scale tiers × multiple families).
# Override with --models. Order matters: small models first so smoke runs are quick.
DEFAULT_MODELS = [
    # ~2B tier
    "ovis2.5-2b",
    "internvl3-2b",
    "qwen3-vl-real-2b",
    # ~3-4B tier
    "qwen2.5-vl-3b",
    "qwen3-vl-real-4b",
    "gemma3-4b",
    # ~7-9B tier
    "qwen2.5-vl-7b",
    "qwen3-vl-real-8b",
    "ovis2.5-9b",
    "internvl3-8b",
    "video-llava",
    # ~12-15B tier (4-bit required to fit on 16GB)
    "gemma3-12b",
    "internvl3-14b",
]


# Per-model batch sizes. Conservative for a single 16GB GPU.
# 1 = no batching (model API not safely batchable, or too large).
PER_MODEL_BATCH_SIZE = {
    "qwen2-vl-2b": 16,
    "qwen2.5-vl-3b": 16,
    "qwen2.5-vl-7b": 8,
    "qwen2.5-vl": 8,
    "qwen3-vl-real-2b": 16,
    "qwen3-vl-real-4b": 12,
    "qwen3-vl-real-8b": 8,
    "gemma3-4b": 12,
    "gemma3-12b": 2,
    "ovis2.5-2b": 1,
    "ovis2.5": 1,
    "ovis2.5-9b": 1,
    "internvl3-2b": 1,
    "internvl3-8b": 1,
    "internvl3-14b": 1,
    "video-llava": 4,
}

# Models that require 4-bit quantization to fit on 16GB.
QUANTIZE_4BIT = {"gemma3-12b", "internvl3-14b"}


def jsonl_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


def append_jsonl(path: Path, records: List[Dict[str, Any]]):
    with path.open("a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def evaluate_one_model(model_name: str, questions: List[Dict[str, Any]],
                       output_dir: Path, device: str, force: bool) -> Dict[str, Any]:
    """Load model, run batched text-only inference, write JSONL checkpoint, return result dict."""
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_file = model_dir / "predictions.jsonl"

    n_questions = len(questions)
    have = jsonl_line_count(pred_file)

    if have == n_questions and not force:
        print(f"[{model_name}] already complete ({have}/{n_questions}) - skipping. "
              f"(Use --force to redo.)")
        with pred_file.open() as f:
            predictions = [json.loads(line) for line in f]
        # Try to recover the model's display name from the first prediction, else fall back.
        display_name = predictions[0].get("model_name", model_name) if predictions else model_name
        return {"model_name": display_name, "predictions": predictions,
                "timestamp": datetime.now().isoformat()}

    if pred_file.exists() and (force or (have != 0 and have != n_questions)):
        if force:
            print(f"[{model_name}] --force: clearing existing predictions ({have} lines).")
        else:
            print(f"[{model_name}] partial run found ({have}/{n_questions}) - restarting from scratch.")
        pred_file.unlink()

    bs = PER_MODEL_BATCH_SIZE.get(model_name, 1)
    quant_4bit = model_name in QUANTIZE_4BIT
    print(f"\n{'='*60}\n[{model_name}] loading "
          f"(batch_size={bs}, 4bit={quant_4bit})\n{'='*60}")

    t0 = time.time()
    try:
        model = create_model(model_name, device=device, quantize_4bit=quant_4bit)
        model.load_model()
    except Exception as e:
        import traceback
        print(f"[{model_name}] LOAD FAILED: {e}")
        traceback.print_exc()
        return {"model_name": model_name, "predictions": [], "load_error": str(e),
                "timestamp": datetime.now().isoformat()}

    print(f"[{model_name}] loaded in {time.time()-t0:.1f}s; "
          f"running {n_questions} questions...")

    t1 = time.time()
    pending = list(range(n_questions))
    batches_done = 0
    running_correct = 0
    running_total = 0
    n_opts = len(questions[0]["options"]) if questions else 8
    random_baseline = 100.0 / n_opts
    for batch_idx_chunk in chunks(pending, bs):
        batch_q = [questions[i] for i in batch_idx_chunk]
        q_texts = [q["question_text"] for q in batch_q]
        opts = [q["options"] for q in batch_q]
        t_batch = time.time()
        try:
            predicted = model.batch_inference_text_only(q_texts, opts)
        except torch.cuda.OutOfMemoryError:
            print(f"[{model_name}] OOM at batch_size={bs}; falling back to batch_size=1 for this batch")
            torch.cuda.empty_cache()
            predicted = []
            for qt, op in zip(q_texts, opts):
                try:
                    predicted.append(model.inference_text_only(qt, op))
                except Exception as e:
                    print(f"  per-item error: {e}")
                    predicted.append("ERROR")
        except Exception as e:
            import traceback
            print(f"[{model_name}] batch error: {e}; falling back per-item")
            traceback.print_exc()
            predicted = []
            for qt, op in zip(q_texts, opts):
                try:
                    predicted.append(model.inference_text_only(qt, op))
                except Exception as e2:
                    print(f"  per-item error: {e2}")
                    predicted.append("ERROR")

        records = []
        for q, pred in zip(batch_q, predicted):
            correct_letter = (q["correct_answer"] or "X")[0]
            if correct_letter not in q["options"]:
                option_keys = list(q["options"].keys()) if isinstance(q["options"], dict) else []
                correct_letter = option_keys[0] if option_keys else "A"
            is_correct = pred == correct_letter
            running_total += 1
            if is_correct:
                running_correct += 1
            records.append({
                "model_name": model.model_name,
                "video_id": q["video_id"],
                "question_id": q["question_id"],
                "capability": q["capability"],
                "referral_strategy": q["referral_strategy"],
                "difficulty": q["difficulty"],
                "predicted": pred,
                "correct": correct_letter,
                "is_correct": is_correct,
            })
        append_jsonl(pred_file, records)
        batches_done += 1
        done = min(batches_done * bs, n_questions)
        elapsed_total = time.time() - t1
        rate = done / max(elapsed_total, 1e-3)
        running_acc = 100.0 * running_correct / max(running_total, 1)
        delta = running_acc - random_baseline
        sign = "+" if delta >= 0 else ""
        eta_s = (n_questions - done) / max(rate, 1e-6)
        print(
            f"  [{model_name}] [{done}/{n_questions}] "
            f"running_acc={running_acc:5.2f}% ({running_correct}/{running_total}) "
            f"vs baseline {random_baseline:.1f}% ({sign}{delta:.2f}pp) "
            f"| batch_t={time.time()-t_batch:.2f}s "
            f"| {rate:.2f} q/s | ETA {eta_s/60:.1f}m",
            flush=True,
        )

    elapsed = time.time() - t1
    final_acc = 100.0 * running_correct / max(running_total, 1)
    print(
        f"[{model_name}] DONE in {elapsed:.1f}s ({n_questions/elapsed:.2f} q/s) "
        f"| FINAL acc={final_acc:.2f}% ({running_correct}/{running_total}) "
        f"vs baseline {random_baseline:.1f}%",
        flush=True,
    )

    # Reload predictions from disk to ensure we report exactly what's persisted.
    with pred_file.open() as f:
        predictions = [json.loads(line) for line in f]

    display_name = model.model_name
    # Free GPU memory before next model.
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {"model_name": display_name, "predictions": predictions,
            "timestamp": datetime.now().isoformat()}


def parse_args():
    p = argparse.ArgumentParser(
        description="Text-only bias eval v2 (multi-model, batched, resumable).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    p.add_argument("--benchmark", required=True, help="Path to anonymized benchmark JSON")
    p.add_argument("--models", nargs="+", default=None,
                   help=f"Models to run (default: {len(DEFAULT_MODELS)}-model matrix)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap to first N questions (smoke test)")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if model's JSONL is already complete")
    p.add_argument("--no_metrics", action="store_true", help="Skip writing metrics.csv")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # HF cache space sanity check
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    if os.path.exists(hf_home):
        free_gb = shutil.disk_usage(hf_home).free / 1e9
        print(f"HF cache: {hf_home} ({free_gb:.0f} GB free)")
        if free_gb < 30:
            print(f"WARNING: <30 GB free on HF cache disk; large models may fail to download.")

    print(f"\n{'='*60}\nTEXT-ONLY BIAS EVAL v2\n{'='*60}")
    print(f"Benchmark: {args.benchmark}")

    questions = load_benchmark_questions(args.benchmark)
    if args.limit:
        questions = questions[:args.limit]
    print(f"Questions: {len(questions)}")

    models = args.models or DEFAULT_MODELS
    print(f"Models ({len(models)}): {', '.join(models)}")
    print(f"Output: {out}\n")

    n_opts = len(questions[0]["options"]) if questions else 8
    baseline = 100.0 / n_opts

    all_results: Dict[str, Dict[str, Any]] = {}
    for idx, model_name in enumerate(models, start=1):
        print(f"\n>>> MODEL {idx}/{len(models)}: {model_name} <<<", flush=True)
        result = evaluate_one_model(model_name, questions, out, args.device, args.force)
        all_results[result["model_name"]] = result

        # Cumulative leaderboard after every model finishes
        print(f"\n{'-'*60}\nLEADERBOARD after {idx}/{len(models)} models "
              f"(random baseline {baseline:.1f}%):\n{'-'*60}", flush=True)
        rows = []
        for mn, r in all_results.items():
            preds = r.get("predictions", [])
            if not preds:
                rows.append((mn, 0.0, 0, 0))
                continue
            corr = sum(1 for p in preds if p.get("is_correct"))
            rows.append((mn, 100.0 * corr / len(preds), corr, len(preds)))
        rows.sort(key=lambda r: -r[1])
        for mn, acc, corr, tot in rows:
            delta = acc - baseline
            sign = "+" if delta >= 0 else ""
            print(f"  {mn:30s}  {acc:6.2f}%  ({corr}/{tot})  "
                  f"vs baseline ({sign}{delta:.2f}pp)", flush=True)

    raw_path = out / "raw_results.json"
    with raw_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAggregated raw_results.json -> {raw_path}")

    if not args.no_metrics:
        # Write a compact metrics.csv (overall + per-capability accuracy).
        # Plotting lives in analysis/ (plot_text_only.py) and reads raw_results.json.
        import csv
        n_opts = len(questions[0]["options"]) if questions else 8
        rows = []
        all_caps = set()
        for mname, res in all_results.items():
            preds = res.get("predictions", [])
            if not preds:
                continue
            total = len(preds)
            correct = sum(1 for p in preds if p.get("is_correct"))
            cap_acc = {}
            for p in preds:
                c = p.get("capability", "unknown")
                cap_acc.setdefault(c, [0, 0])
                cap_acc[c][1] += 1
                if p.get("is_correct"):
                    cap_acc[c][0] += 1
            all_caps.update(cap_acc)
            row = {
                "model": mname,
                "overall_accuracy": 100.0 * correct / max(total, 1),
                "random_baseline": 100.0 / n_opts,
                "total_correct": correct,
                "total_questions": total,
            }
            for c, (cc, tt) in cap_acc.items():
                row[f"cap_{c}"] = 100.0 * cc / max(tt, 1)
            rows.append(row)
        cols = ["model", "overall_accuracy", "random_baseline",
                "total_correct", "total_questions"] + [f"cap_{c}" for c in sorted(all_caps)]
        with (out / "metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"metrics.csv -> {out/'metrics.csv'}")

    print(f"\n{'='*60}\nDONE\n{'='*60}")


if __name__ == "__main__":
    main()
