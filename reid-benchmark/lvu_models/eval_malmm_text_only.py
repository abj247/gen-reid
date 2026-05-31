#!/usr/bin/env python3
"""Standalone MA-LMM text-only evaluator.

Runs in the `ma-lmm` conda env (transformers 4.33.3 + LAVIS). Loads
MA-LMM (memory-augmented InstructBLIP, vicuna7b backbone) and runs MCQ
text-only inference with a dummy black video, emitting predictions in
the same JSONL schema as our other models.

Usage:
    PYTHONPATH=/home/ab260989/third_party/MA-LMM python eval_malmm_text_only.py \\
        --benchmark combined_all_hard_v3.json \\
        --output_dir results_text_only_v2 \\
        --models ma-lmm-vicuna7b
"""
import argparse, gc, json, re, time
from pathlib import Path
import numpy as np
import torch


MODELS = {
    "ma-lmm-vicuna7b":  ("vicuna7b",  "MA-LMM-Vicuna-7B"),
    "ma-lmm-vicuna13b": ("vicuna13b", "MA-LMM-Vicuna-13B"),
}

NUM_FRAMES = 20            # MA-LMM default; works with the released checkpoints
MEMORY_BANK_LENGTH = 10    # MA-LMM default


def load_benchmark(path):
    with open(path) as f:
        data = json.load(f)
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    flat = []
    for v in data.get("videos", []):
        vid = v.get("video_id", "")
        for q in v.get("questions", []):
            meta = q.get("metadata", {})
            raw_opts = q.get("options", {})
            if isinstance(raw_opts, list):
                opts = {labels[i]: o for i, o in enumerate(raw_opts) if i < len(labels)}
            else:
                opts = raw_opts
            flat.append({
                "video_id": vid,
                "question_id": q.get("question_id", ""),
                "question_text": q.get("question_text", q.get("question", "")),
                "options": opts,
                "correct_answer": (q.get("correct_answer") or q.get("answer", "")).strip().upper(),
                "capability": meta.get("capability", "unknown"),
                "referral_strategy": meta.get("referral_strategy", "unknown"),
                "difficulty": meta.get("difficulty", "Unknown"),
            })
    return flat


def format_mcq_prompt(question, options):
    keys = sorted(options.keys())
    body = "\n".join(f"{k}. {options[k]}" for k in keys)
    letter_list = ", ".join(keys[:-1]) + f", or {keys[-1]}"
    # InstructBLIP uses "Question: ... Answer:" framing in its training data.
    return (f"Question: {question}\nOptions:\n{body}\n"
            f"Answer with only the letter ({letter_list}) of the correct option.\nAnswer:")


def extract_answer(response, num_options=8):
    response = response.strip().upper()
    max_letter = chr(ord("A") + num_options - 1)
    rng = f"A-{max_letter}"
    valid = [chr(ord("A") + i) for i in range(num_options)]
    if response in valid:
        return response
    for p in (rf"^([{rng}])\.", rf"^([{rng}])\)", rf"^\(([{rng}])\)",
              rf"^answer[:\s]*([{rng}])", rf"^the answer is[:\s]*([{rng}])"):
        m = re.search(p, response, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = re.search(rf"\b([{rng}])\b", response)
    return m.group(1).upper() if m else "INVALID"


def jsonl_line_count(path):
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


def evaluate_one_model(model_key, questions, output_dir, force):
    from lavis.models import load_model_and_preprocess
    model_type, display = MODELS[model_key]
    model_dir = output_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_file = model_dir / "predictions.jsonl"
    n_q = len(questions)
    have = jsonl_line_count(pred_file)
    if have == n_q and not force:
        print(f"[{model_key}] already complete ({have}/{n_q}) - skipping.", flush=True)
        return
    if pred_file.exists() and (force or (have != 0 and have != n_q)):
        print(f"[{model_key}] clearing existing predictions ({have} lines).", flush=True)
        pred_file.unlink()

    device = torch.device("cuda")
    print(f"[{model_key}] loading MA-LMM ({model_type}) ...", flush=True)
    t0 = time.time()
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_vicuna_instruct_malmm", model_type=model_type, is_eval=True,
        device=device, memory_bank_length=MEMORY_BANK_LENGTH, num_frames=NUM_FRAMES,
    )
    model.eval()
    print(f"[{model_key}] loaded in {time.time()-t0:.1f}s", flush=True)

    # Build a dummy black-video tensor once: shape (C=3, T=NUM_FRAMES, H=224, W=224)
    # vis_processors expect (C, T, H, W) per the demo (transposes from get_batch).
    dummy = torch.zeros(3, NUM_FRAMES, 224, 224, dtype=torch.float32)
    dummy_proc = vis_processors["eval"](dummy).to(device).unsqueeze(0)

    correct = total = 0
    t_start = time.time()
    f_out = pred_file.open("a")
    try:
        for i, q in enumerate(questions):
            prompt = format_mcq_prompt(q["question_text"], q["options"])
            try:
                with torch.no_grad():
                    out = model.generate({"image": dummy_proc, "prompt": prompt}, max_length=16)
                resp = out[0] if isinstance(out, (list, tuple)) else str(out)
                predicted = extract_answer(resp, num_options=len(q["options"]))
            except Exception as e:
                predicted = "ERROR"
                if i < 3:
                    print(f"  err on {q['question_id']}: {e}", flush=True)

            correct_letter = (q["correct_answer"] or "A")[0]
            if correct_letter not in q["options"]:
                correct_letter = list(q["options"].keys())[0] if q["options"] else "A"
            is_correct = predicted == correct_letter
            total += 1
            correct += int(is_correct)
            f_out.write(json.dumps({
                "model_name": display, "video_id": q["video_id"],
                "question_id": q["question_id"], "capability": q["capability"],
                "referral_strategy": q["referral_strategy"], "difficulty": q["difficulty"],
                "predicted": predicted, "correct": correct_letter, "is_correct": is_correct,
            }) + "\n")
            f_out.flush()
            if (i + 1) % 50 == 0 or i == 0 or i == n_q - 1:
                elapsed = time.time() - t_start
                rate = (i + 1) / max(elapsed, 1e-3)
                acc = 100.0 * correct / total
                eta = (n_q - i - 1) / max(rate, 1e-6) / 60.0
                print(
                    f"  [{model_key}] [{i+1}/{n_q}] acc={acc:5.2f}% ({correct}/{total}) | "
                    f"{rate:.2f} q/s | ETA {eta:.1f}m", flush=True,
                )
    finally:
        f_out.close()

    print(f"[{model_key}] DONE total acc={100.0*correct/max(total,1):.2f}% "
          f"({correct}/{total}) in {(time.time()-t_start)/60:.1f}m", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--models", nargs="+", default=["ma-lmm-vicuna7b"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    questions = load_benchmark(args.benchmark)
    if args.limit:
        questions = questions[:args.limit]
    print(f"Loaded {len(questions)} questions from {args.benchmark}", flush=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for key in args.models:
        if key not in MODELS:
            print(f"Unknown model key: {key}. Skipping.", flush=True); continue
        print(f"\n>>> {key} <<<", flush=True)
        evaluate_one_model(key, questions, out, args.force)


if __name__ == "__main__":
    main()
