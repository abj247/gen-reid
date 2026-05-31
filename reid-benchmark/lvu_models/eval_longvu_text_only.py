#!/usr/bin/env python3
"""Standalone LongVU text-only evaluator.

Runs in the `longvu` conda env (torch 2.1.2 + transformers 4.42.4). Loads a
LongVU checkpoint, feeds question + options + a single dummy black frame, and
writes predictions in the same JSONL schema as our other models so the
existing aggregator and analysis scripts work unchanged.

Output: <output_dir>/<model_key>/predictions.jsonl (resumable; lines are appended).

Usage:
    PYTHONPATH=/home/ab260989/third_party/LongVU python eval_longvu_text_only.py \\
        --benchmark combined_all_hard_v3.json \\
        --output_dir results_text_only_v2 \\
        --models longvu-qwen2-7b longvu-llama3-3b
"""
import argparse, gc, json, os, re, sys, time
from pathlib import Path
import numpy as np
import torch


# Model registry: key -> (HF repo id, model_name_arg, conv_template, display_name)
MODELS = {
    "longvu-qwen2-7b":   ("Vision-CAIR/LongVU_Qwen2_7B",    "cambrian_qwen",   "qwen",   "LongVU-Qwen2-7B"),
    "longvu-llama3-3b":  ("Vision-CAIR/LongVU_Llama3_2_3B", "cambrian_llama3", "llama3", "LongVU-Llama3.2-3B"),
}


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
    prompt = f"{question}\n\nOptions:\n"
    for k in keys:
        prompt += f"{k}. {options[k]}\n"
    letter_list = ", ".join(keys[:-1]) + f", or {keys[-1]}"
    prompt += f"\nAnswer with only the letter ({letter_list}) of the correct option."
    return prompt


def extract_answer(response, num_options=8):
    response = response.strip().upper()
    max_letter = chr(ord("A") + num_options - 1)
    rng = f"A-{max_letter}"
    valid = [chr(ord("A") + i) for i in range(num_options)]
    if response in valid:
        return response
    patterns = [
        rf"^([{rng}])\.", rf"^([{rng}])\)", rf"^\(([{rng}])\)",
        rf"^answer[:\s]*([{rng}])", rf"^the answer is[:\s]*([{rng}])",
        rf"^([{rng}])\s*[-:]",
    ]
    for p in patterns:
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
    from longvu.builder import load_pretrained_model
    from longvu.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from longvu.conversation import conv_templates, SeparatorStyle
    from longvu.mm_datautils import KeywordsStoppingCriteria, process_images, tokenizer_image_token

    repo_id, model_name_arg, conv_key, display = MODELS[model_key]
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

    print(f"[{model_key}] loading {repo_id} ...", flush=True)
    t0 = time.time()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        repo_id, None, model_name_arg, device="cuda")
    model.eval()
    print(f"[{model_key}] loaded in {time.time()-t0:.1f}s", flush=True)

    # Pre-build dummy frame once: a single 448x448 black frame
    dummy_np = np.zeros((1, 448, 448, 3), dtype=np.uint8)
    dummy_video = process_images(dummy_np, image_processor, model.config)
    dummy_video = [item.unsqueeze(0) for item in dummy_video]
    image_sizes = [(448, 448)]

    correct = total = 0
    t_start = time.time()
    f_out = pred_file.open("a")
    try:
        for i, q in enumerate(questions):
            prompt_text = DEFAULT_IMAGE_TOKEN + "\n" + format_mcq_prompt(
                q["question_text"], q["options"])
            conv = conv_templates[conv_key].copy()
            conv.append_message(conv.roles[0], prompt_text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
            try:
                with torch.inference_mode():
                    out_ids = model.generate(
                        input_ids, images=dummy_video, image_sizes=image_sizes,
                        do_sample=False, max_new_tokens=8, use_cache=True,
                        stopping_criteria=[stopping],
                    )
                resp = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
                predicted = extract_answer(resp, num_options=len(q["options"]))
            except Exception as e:
                predicted = "ERROR"
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
            if (i + 1) % 25 == 0 or i == 0 or i == n_q - 1:
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
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Cap to first N questions (smoke)")
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
