#!/usr/bin/env python3
"""MA-LMM video+text evaluator (real frames). Runs in the `ma-lmm` conda env."""
import argparse, gc, json, os, re, time
from pathlib import Path
import numpy as np
import torch
from decord import VideoReader, cpu

MODELS = {"ma-lmm-vicuna7b": ("vicuna7b", "MA-LMM-Vicuna-7B")}
NUM_FRAMES = 20            # MA-LMM default
MEMORY_BANK_LENGTH = 10


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
            print(f"WARN no mp4 for {vid}", flush=True); continue
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


def format_mcq_prompt(q, opts):
    keys = sorted(opts.keys())
    body = "\n".join(f"{k}. {opts[k]}" for k in keys)
    return (f"Question: {q}\nOptions:\n{body}\n"
            f"Answer with only the letter ({', '.join(keys[:-1])}, or {keys[-1]}) of the correct option.\nAnswer:")


def extract_answer(response, num_options=8):
    response = response.strip().upper()
    rng = f"A-{chr(ord('A')+num_options-1)}"
    valid = [chr(ord('A')+i) for i in range(num_options)]
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


def sample_video_tensor(video_path, n=NUM_FRAMES):
    """Returns a torch.Tensor of shape (C=3, T=n, H, W) in float32 [0,255].
    MA-LMM's lavis sets decord.bridge='torch' at import time, so get_batch
    already returns a torch tensor; handle both bridge configurations."""
    import decord
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    idx = list(np.linspace(0, max(total-1, 0), n).astype(int))
    batch = vr.get_batch(idx)
    if hasattr(batch, "asnumpy"):
        frames_np = batch.asnumpy()
        frames = torch.from_numpy(frames_np)
    else:
        frames = batch  # already a torch tensor
    return frames.permute(3, 0, 1, 2).to(torch.float32)  # (3, n, H, W)


def evaluate_one_model(model_key, items, output_dir, force):
    from lavis.models import load_model_and_preprocess
    model_type, display = MODELS[model_key]
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

    device = torch.device("cuda")
    print(f"[{model_key}] loading MA-LMM ({model_type}) ...", flush=True)
    t0 = time.time()
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_vicuna_instruct_malmm", model_type=model_type, is_eval=True,
        device=device, memory_bank_length=MEMORY_BANK_LENGTH, num_frames=NUM_FRAMES,
    )
    model.eval()
    print(f"[{model_key}] loaded in {time.time()-t0:.1f}s; {n_q} questions to run", flush=True)

    last_vpath = None
    cached = None
    correct = total = 0
    f_out = pred_file.open("a")
    t_start = time.time()
    try:
        for i, (vid, vpath, q) in enumerate(flat):
            if vpath != last_vpath:
                try:
                    raw = sample_video_tensor(vpath, NUM_FRAMES)
                    cached = vis_processors["eval"](raw).to(device).unsqueeze(0)
                    last_vpath = vpath
                except Exception as e:
                    print(f"  video err {vid}: {e}", flush=True)
                    cached = None
            if cached is None:
                predicted = "ERROR"
            else:
                prompt = format_mcq_prompt(q["question_text"], q["options"])
                try:
                    with torch.no_grad():
                        out = model.generate({"image": cached, "prompt": prompt}, max_length=16)
                    resp = out[0] if isinstance(out, (list, tuple)) else str(out)
                    predicted = extract_answer(resp, num_options=len(q["options"]))
                except Exception as e:
                    predicted = "ERROR"

            correct_letter = (q.get("correct_answer") or "A")[0]
            if correct_letter not in q["options"]:
                correct_letter = list(q["options"].keys())[0] if q["options"] else "A"
            is_correct = predicted == correct_letter
            total += 1; correct += int(is_correct)
            f_out.write(json.dumps({
                "model_name": display, "video_id": vid, "question_id": q.get("question_id",""),
                "capability": q.get("metadata",{}).get("capability","unknown"),
                "referral_strategy": q.get("metadata",{}).get("referral_strategy","unknown"),
                "difficulty": q.get("metadata",{}).get("difficulty","Unknown"),
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
    ap.add_argument("--manifest", default=None, help="conditioned-clip manifest; answer each question on its clip")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--models", nargs="+", default=["ma-lmm-vicuna7b"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit_videos", type=int, default=None)
    args = ap.parse_args()
    items = load_conditioned(args.manifest) if args.manifest else load_benchmark_with_videos(args.benchmark, args.video_dir)
    if args.limit_videos: items = items[:args.limit_videos]
    print(f"Loaded {len(items)} videos, {sum(len(qs) for _,_,qs in items)} questions", flush=True)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for key in args.models:
        if key not in MODELS:
            print(f"Unknown {key}, skipping", flush=True); continue
        print(f"\n>>> {key} <<<", flush=True)
        evaluate_one_model(key, items, out, args.force)


if __name__ == "__main__":
    main()
