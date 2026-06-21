#!/usr/bin/env python3
"""LongVU video+text evaluator (real frames). Runs in the `longvu` conda env."""
import argparse, gc, json, os, re, time
from pathlib import Path
import numpy as np
import torch
from decord import VideoReader, cpu

MODELS = {
    "longvu-qwen2-7b": ("Vision-CAIR/LongVU_Qwen2_7B", "cambrian_qwen", "qwen", "LongVU-Qwen2-7B"),
}
NUM_FRAMES = 8


def load_benchmark_with_videos(path, video_dir):
    with open(path) as f:
        bench = json.load(f)
    items = []
    for v in bench.get("videos", []):
        vid = v.get("video_id", "")
        vpath = None
        for ext in (".mp4", ".avi", ".mkv", ".mov", ".webm", ""):
            cand = Path(video_dir) / f"{vid}{ext}"
            if cand.is_file():
                vpath = str(cand); break
        if vpath is None:
            print(f"WARN no mp4 for {vid}, skipping", flush=True); continue
        items.append((vid, vpath, v.get("questions", [])))
    return items


def load_conditioned(manifest_path, bench_path="/home/ab260989/gen-reid/combined_all_hard_v3_retagged.json"):
    """One item per manifested question: (real_vid, clip_path, [question_dict])."""
    manifest = json.load(open(manifest_path)); bench = json.load(open(bench_path))
    qmap = {}
    for v in bench.get("videos", []):
        for q in v.get("questions", []):
            qmap[f"{v['video_id']}|{q.get('question_id')}"] = q
    items = []
    for key, clip in manifest.items():
        if key in qmap and Path(clip).is_file():
            items.append((key.split("|")[0], clip, [qmap[key]]))
    return items


def format_mcq_prompt(question, options):
    keys = sorted(options.keys())
    body = "\n".join(f"{k}. {options[k]}" for k in keys)
    return f"{question}\n\nOptions:\n{body}\n\nAnswer with only the letter ({', '.join(keys[:-1])}, or {keys[-1]}) of the correct option."


def extract_answer(response, num_options=8):
    response = response.strip().upper()
    max_letter = chr(ord("A") + num_options - 1)
    rng = f"A-{max_letter}"
    valid = [chr(ord("A") + i) for i in range(num_options)]
    if response in valid: return response
    for p in (rf"^([{rng}])\.", rf"^([{rng}])\)", rf"^\(([{rng}])\)",
              rf"^answer[:\s]*([{rng}])", rf"^the answer is[:\s]*([{rng}])"):
        m = re.search(p, response, re.IGNORECASE)
        if m: return m.group(1).upper()
    m = re.search(rf"\b([{rng}])\b", response)
    return m.group(1).upper() if m else "INVALID"


def jsonl_line_count(p):
    if not p.exists(): return 0
    with p.open() as f: return sum(1 for _ in f)


def sample_frames(video_path, n=NUM_FRAMES):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    idx = np.linspace(0, max(total-1, 0), n).astype(int)
    frames = vr.get_batch(idx).asnumpy()  # (n, H, W, 3) uint8
    h, w = frames.shape[1], frames.shape[2]
    return frames, (h, w)


def evaluate_one_model(model_key, items, output_dir, force):
    from longvu.builder import load_pretrained_model
    from longvu.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from longvu.conversation import conv_templates, SeparatorStyle
    from longvu.mm_datautils import KeywordsStoppingCriteria, process_images, tokenizer_image_token

    repo_id, model_name_arg, conv_key, display = MODELS[model_key]
    model_dir = output_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_file = model_dir / "predictions.jsonl"
    flat = [(vid, vpath, q) for vid, vpath, qs in items for q in qs]
    n_q = len(flat)
    have = jsonl_line_count(pred_file)
    if have == n_q and not force:
        print(f"[{model_key}] already complete ({have}/{n_q}) - skipping.", flush=True); return
    if pred_file.exists() and (force or (have != 0 and have != n_q)):
        print(f"[{model_key}] clearing {have} lines.", flush=True); pred_file.unlink()

    print(f"[{model_key}] loading {repo_id} ...", flush=True)
    t0 = time.time()
    tokenizer, model, image_processor, _ = load_pretrained_model(repo_id, None, model_name_arg, device="cuda")
    model.eval()
    print(f"[{model_key}] loaded in {time.time()-t0:.1f}s; running {n_q} questions...", flush=True)

    # Per-video frame caching
    last_vpath = None
    cached_video = None
    cached_sizes = None

    correct = total = 0
    f_out = pred_file.open("a")
    t_start = time.time()
    try:
        for i, (vid, vpath, q) in enumerate(flat):
            if vpath != last_vpath:
                try:
                    frames_np, hw = sample_frames(vpath, NUM_FRAMES)
                    video = process_images(frames_np, image_processor, model.config)
                    video = [item.unsqueeze(0) for item in video]
                    cached_video = video
                    cached_sizes = [hw]
                    last_vpath = vpath
                except Exception as e:
                    print(f"  video err {vid}: {e}", flush=True)
                    cached_video = None
            if cached_video is None:
                predicted = "ERROR"
            else:
                prompt_text = DEFAULT_IMAGE_TOKEN + "\n" + format_mcq_prompt(q["question_text"], q["options"])
                conv = conv_templates[conv_key].copy()
                conv.append_message(conv.roles[0], prompt_text)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
                try:
                    with torch.inference_mode():
                        out_ids = model.generate(
                            input_ids, images=cached_video, image_sizes=cached_sizes,
                            do_sample=False, max_new_tokens=8, use_cache=True,
                            stopping_criteria=[stopping],
                        )
                    resp = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
                    predicted = extract_answer(resp, num_options=len(q["options"]))
                except Exception as e:
                    predicted = "ERROR"

            correct_letter = (q.get("correct_answer") or "A")[0]
            if correct_letter not in q["options"]:
                correct_letter = list(q["options"].keys())[0] if q["options"] else "A"
            is_correct = predicted == correct_letter
            total += 1
            correct += int(is_correct)
            f_out.write(json.dumps({
                "model_name": display, "video_id": vid, "question_id": q.get("question_id",""),
                "capability": q.get("metadata", {}).get("capability","unknown"),
                "referral_strategy": q.get("metadata", {}).get("referral_strategy","unknown"),
                "difficulty": q.get("metadata", {}).get("difficulty","Unknown"),
                "predicted": predicted, "correct": correct_letter, "is_correct": is_correct,
            }) + "\n")
            f_out.flush()
            if (i+1) % 25 == 0 or i == 0 or i == n_q-1:
                elapsed = time.time()-t_start
                rate = (i+1)/max(elapsed,1e-3)
                eta = (n_q-i-1)/max(rate,1e-6)/60.0
                print(f"  [{model_key}] [{i+1}/{n_q}] acc={100*correct/total:5.2f}% ({correct}/{total}) | {rate:.2f} q/s | ETA {eta:.1f}m", flush=True)
    finally:
        f_out.close()
    print(f"[{model_key}] DONE acc={100*correct/max(total,1):.2f}% ({correct}/{total}) in {(time.time()-t_start)/60:.1f}m", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--video_dir", default=None)
    ap.add_argument("--manifest", default=None, help="conditioned-clip manifest; if set, answer each question on its clip")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit_videos", type=int, default=None)
    args = ap.parse_args()
    if args.manifest:
        items = load_conditioned(args.manifest)
    else:
        items = load_benchmark_with_videos(args.benchmark, args.video_dir)
    if args.limit_videos: items = items[:args.limit_videos]
    print(f"Loaded {len(items)} videos, {sum(len(qs) for _,_,qs in items)} questions", flush=True)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for key in args.models:
        if key not in MODELS:
            print(f"Unknown model {key}, skipping", flush=True); continue
        print(f"\n>>> {key} <<<", flush=True)
        evaluate_one_model(key, items, out, args.force)


if __name__ == "__main__":
    main()
