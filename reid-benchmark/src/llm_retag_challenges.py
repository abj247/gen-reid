#!/usr/bin/env python3
"""
LLM-based re-tagging of the Re-ID challenge type for every benchmark question.

The original `re-id_challenge` metadata is near-free-text (3476 distinct strings
for 3666 questions), which makes failure-mode grouping unreliable. Here we use a
local instruct LLM to classify each question into ONE canonical category from a
fixed taxonomy, reading the question text + options (the most informative signal)
plus the original free-text hint.

Output:
  - reid_retags.json : {"video_id|question_id": {"label": <canon>, "raw": <model output>}}
  - combined_all_hard_v3_retagged.json : benchmark copy with metadata["reid_canonical"]

Default model: google/gemma-3-12b-it (strong text classifier, cached locally).
Override with --model (any HF instruct model id) and --quantize_4bit for 12B on small GPUs.

Usage:
    python llm_retag_reid.py --bench combined_all_hard_v3.json \\
        --model google/gemma-3-12b-it --batch_size 8 --quantize_4bit
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch

# ---- Canonical taxonomy ----------------------------------------------------
# (label, one-line definition shown to the LLM)
TAXONOMY = [
    ("cross_scene_reid",
     "Re-identify the same person/object across a scene change or camera cut "
     "(different location, angle, or shot)."),
    ("long_term_tracking",
     "Track an identity across most of the video or through many cuts / a long "
     "time gap."),
    ("multi_hop_tracking",
     "Follow an identity through a CHAIN of intermediate events/people to reach "
     "the answer (A->B->C reasoning)."),
    ("occlusion_recovery",
     "The target leaves the frame / is occluded / disappears and must be "
     "re-identified when it returns."),
    ("disambiguation_similar",
     "Distinguish between two or more SIMILAR-LOOKING people/objects (same "
     "outfit, look-alikes)."),
    ("role_position_swap",
     "People or objects swap roles, positions, or possessions and must be told "
     "apart afterward."),
    ("action_sequence",
     "Identify by a SEQUENCE of actions over time (what someone does before/"
     "after another action)."),
    ("spatial_relationship",
     "Reason about relative position, direction, or movement to a location."),
    ("appearance_change",
     "Track despite a change in appearance (clothing, pose, lighting) of the "
     "same identity."),
    ("single_shot",
     "Answerable from a single moment/frame; no real cross-time tracking "
     "needed."),
]
LABELS = [t[0] for t in TAXONOMY]
LABEL_SET = set(LABELS)


def build_prompt(question, options, raw_hint):
    opts = "\n".join(f"  {k}. {v}" for k, v in options.items()) if isinstance(options, dict) else str(options)
    taxo = "\n".join(f"{i+1}. {lab}: {desc}" for i, (lab, desc) in enumerate(TAXONOMY))
    return (
        "You are labelling a video question-answering benchmark about "
        "re-identifying people/objects in a video.\n"
        "Classify the QUESTION into exactly ONE of these categories:\n\n"
        f"{taxo}\n\n"
        f"QUESTION: {question}\n"
        f"OPTIONS:\n{opts}\n"
        f"ORIGINAL HINT: {raw_hint}\n\n"
        "Reply with ONLY the category label (e.g. cross_scene_reid). "
        "No explanation.\n"
        "Label:"
    )


def parse_label(text):
    t = text.strip().lower()
    # direct contains
    for lab in LABELS:
        if lab in t:
            return lab
    # keyword fallback
    kw = {
        "cross": "cross_scene_reid", "scene": "cross_scene_reid", "cut": "cross_scene_reid",
        "long": "long_term_tracking", "multi": "multi_hop_tracking", "hop": "multi_hop_tracking",
        "occl": "occlusion_recovery", "disambig": "disambiguation_similar",
        "similar": "disambiguation_similar", "swap": "role_position_swap",
        "role": "role_position_swap", "action": "action_sequence",
        "spatial": "spatial_relationship", "appearance": "appearance_change",
        "single": "single_shot",
    }
    for k, lab in kw.items():
        if k in t:
            return lab
    return "single_shot"


def load_model(model_id, quantize_4bit, device):
    from transformers import AutoProcessor, AutoTokenizer
    load_kwargs = dict(dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True)
    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    is_gemma = "gemma-3" in model_id.lower()
    if is_gemma:
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(model_id, **load_kwargs).eval()
        proc = AutoProcessor.from_pretrained(model_id)
        tok = proc.tokenizer
        return model, proc, tok, "gemma"
    # generic causal LM path
    from transformers import AutoModelForCausalLM
    load_kwargs["trust_remote_code"] = True
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).eval()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tok, tok, "causal"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default="combined_all_hard_v3.json")
    ap.add_argument("--model", default="google/gemma-3-12b-it")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--quantize_4bit", action="store_true")
    ap.add_argument("--out_tags", default="reid_retags.json")
    ap.add_argument("--out_bench", default="combined_all_hard_v3_retagged.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    bench = json.load(open(args.bench))
    items = []
    for v in bench["videos"]:
        vid = v.get("video_id")
        for q in v.get("questions", []):
            items.append((vid, q.get("question_id"),
                          q.get("question_text", q.get("question", "")),
                          q.get("options", {}),
                          q.get("metadata", {}).get("re-id_challenge", "")))
    if args.limit:
        items = items[:args.limit]
    print(f"Questions to tag: {len(items)}")

    device = "cuda"
    print(f"Loading {args.model} (4bit={args.quantize_4bit}) ...")
    model, proc, tok, kind = load_model(args.model, args.quantize_4bit, device)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print("Loaded.")

    tags = {}
    t0 = time.time()
    bs = args.batch_size
    for start in range(0, len(items), bs):
        batch = items[start:start + bs]
        prompts = [build_prompt(q, o, h) for (_, _, q, o, h) in batch]
        if kind == "gemma":
            msgs = [[{"role": "user", "content": [{"type": "text", "text": p}]}] for p in prompts]
            inputs = proc.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True).to(device)
        else:
            texts = [tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                for p in prompts]
            inputs = tok(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=12, do_sample=False)
        plen = inputs["input_ids"].shape[1]
        outs = tok.batch_decode(gen[:, plen:], skip_special_tokens=True)
        for (vid, qid, *_), raw in zip(batch, outs):
            tags[f"{vid}|{qid}"] = {"label": parse_label(raw), "raw": raw.strip()[:40]}
        done = start + len(batch)
        if (start // bs) % 10 == 0 or done >= len(items):
            rate = done / max(time.time() - t0, 1e-3)
            print(f"  [{done}/{len(items)}] {rate:.1f} q/s "
                  f"ETA {(len(items)-done)/max(rate,1e-6)/60:.1f}m", flush=True)

    json.dump(tags, open(args.out_tags, "w"), indent=2)
    print(f"Wrote {args.out_tags}")

    # label distribution
    from collections import Counter
    dist = Counter(t["label"] for t in tags.values())
    print("\nLabel distribution:")
    for lab in LABELS:
        print(f"  {lab:24s} {dist.get(lab,0):5d}")

    # write retagged benchmark
    for v in bench["videos"]:
        vid = v.get("video_id")
        for q in v.get("questions", []):
            key = f"{vid}|{q.get('question_id')}"
            if key in tags:
                q.setdefault("metadata", {})["reid_canonical"] = tags[key]["label"]
    json.dump(bench, open(args.out_bench, "w"), indent=2)
    print(f"Wrote {args.out_bench}")


if __name__ == "__main__":
    main()
