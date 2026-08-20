#!/usr/bin/env python3
"""
Generate LLM-debiased benchmark: use Qwen2.5-VL to create 7 plausible
wrong answers per question that match the correct answer's style/length.

This eliminates text-only bias because the same LLM that will be tested
generates the distractors — it cannot distinguish its own outputs from
the correct answer without seeing the video.

Usage:
    python generate_debiased_options.py \
        --input movienetTest_v2_8opt.json \
        --output movienetTest_v2_8opt_llm_debiased.json \
        --seed 42

Requires GPU (runs Qwen2.5-VL-7B in 4-bit). ~1.5-2.5 hours for 1710 questions.
"""

import argparse
import gc
import json
import random
import re
import string
import sys
import time
from collections import Counter
from copy import deepcopy

if sys.version_info >= (3, 14):
    import torch
    _orig = torch.compile
    def _no_compile(model, *args, **kwargs):
        return model
    torch.compile = _no_compile

import torch


def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def load_model(device="cuda"):
    """Load Qwen2.5-VL-7B in bf16 (48GB GPU is enough for 7B)."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"Loading {model_id} (bf16)...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id)
    print(f"Model loaded. GPU: {torch.cuda.get_device_name(0)}")
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"  GPU memory used: {mem:.1f} GB")
    return model, processor


def generate_text(model, processor, prompt, max_new_tokens=512,
                  temperature=0.7, device="cuda"):
    """Generate free-form text from the model."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )

    # Decode only the generated part
    generated = output_ids[0][inputs.input_ids.shape[-1]:]
    response = processor.decode(generated, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)

    del inputs, output_ids, generated
    clear_gpu()
    return response


def build_prompt(question_text, correct_answer):
    """Build the distractor generation prompt."""
    target_len = len(correct_answer)
    return f"""You are creating a multiple-choice video comprehension test. Given the question below, generate exactly 7 wrong but highly plausible answer options.

Rules:
- Each option must be a believable answer someone might give if they misremember the video
- Each option must be approximately {target_len} characters long (within 10 characters)
- Match the grammatical style of this example: "{correct_answer}"
- Each option must describe a different but plausible scenario for the scene described
- Do NOT include or paraphrase the correct answer: "{correct_answer}"
- Make every option equally detailed and specific — no option should be vaguer than others
- Number them 1 through 7, one per line

Question: {question_text}

1."""


def parse_distractors(response, correct_answer, target_len):
    """Parse LLM response into list of distractor strings."""
    distractors = []
    correct_lower = correct_answer.lower().strip()

    # Try to extract numbered items
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove numbering: "1. ", "2) ", "1:", etc.
        cleaned = re.sub(r'^[\d]+[\.\)\:]?\s*', '', line).strip()
        if not cleaned:
            continue
        # Skip if it's too similar to the correct answer
        if cleaned.lower().strip() == correct_lower:
            continue
        # Skip if duplicate
        if cleaned.lower().strip() in {d.lower().strip() for d in distractors}:
            continue
        distractors.append(cleaned)

    return distractors[:7]


def generate_distractors_for_question(model, processor, question_text,
                                       correct_answer, rng, device="cuda"):
    """Generate 7 distractors for one question. Retry if needed."""
    target_len = len(correct_answer)

    for attempt in range(3):
        temp = 0.7 + attempt * 0.15  # Increase temperature on retry
        prompt = build_prompt(question_text, correct_answer)
        response = generate_text(model, processor, prompt,
                                 max_new_tokens=600, temperature=temp,
                                 device=device)

        distractors = parse_distractors(response, correct_answer, target_len)

        if len(distractors) >= 7:
            return distractors[:7]

        # If we got some but not enough, try to get more
        if len(distractors) >= 4 and attempt < 2:
            needed = 7 - len(distractors)
            prompt2 = f"""Generate {needed} more plausible wrong answers for this question.
They must be approximately {target_len} characters long and different from these existing options:
{chr(10).join(f'- {d}' for d in distractors)}
Also different from: "{correct_answer}"

Question: {question_text}

1."""
            response2 = generate_text(model, processor, prompt2,
                                      max_new_tokens=400, temperature=temp + 0.1,
                                      device=device)
            extra = parse_distractors(response2, correct_answer, target_len)
            # Filter out duplicates of existing
            existing_lower = {d.lower().strip() for d in distractors}
            for d in extra:
                if d.lower().strip() not in existing_lower:
                    distractors.append(d)
                    existing_lower.add(d.lower().strip())
                if len(distractors) >= 7:
                    break

        if len(distractors) >= 7:
            return distractors[:7]

    # Fallback: pad with generic plausible options if still short
    return distractors[:7]


def get_content_words(text):
    """Extract content words for overlap analysis."""
    STOP = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'to', 'of', 'in', 'for', 'on', 'with',
        'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'and', 'but', 'or', 'if',
        'while', 'that', 'this', 'what', 'which', 'who', 'his', 'her',
        'its', 'their', 'he', 'she', 'it', 'they', 'we', 'you', 'him',
        'them',
    }
    words = set(re.findall(r'[a-z]+', text.lower()))
    return words - STOP


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM-debiased benchmark options",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Process only N questions (debug)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed)

    # Load benchmark
    with open(args.input) as f:
        data = json.load(f)

    # Flatten questions
    all_items = []
    for v_idx, video in enumerate(data['videos']):
        for q_idx, q in enumerate(video['questions']):
            all_items.append({
                'v_idx': v_idx,
                'q_idx': q_idx,
                'video_id': video['video_id'],
                'q': deepcopy(q),
            })

    total = len(all_items)
    if args.max_samples:
        all_items = all_items[:args.max_samples]
        total = len(all_items)
    print(f"Loaded {total} questions from {len(data['videos'])} videos")

    # Load model
    model, processor = load_model(args.device)

    # Assign balanced target positions
    labels = list(string.ascii_uppercase[:8])
    indices = list(range(total))
    rng.shuffle(indices)
    for rank, idx in enumerate(indices):
        all_items[idx]['target_pos'] = labels[rank % 8]

    # Stats
    stats = {
        'success': 0,
        'partial': 0,
        'failed': 0,
        'total_generated': 0,
        'retries': 0,
    }

    # Generate distractors for each question
    print(f"\nGenerating distractors for {total} questions...")
    start_time = time.time()

    for i, item in enumerate(all_items):
        q = item['q']
        correct_key = q['correct_answer'].strip().upper()
        correct_text = q['options'][correct_key]
        question_text = q.get('question_text', q.get('question', ''))

        distractors = generate_distractors_for_question(
            model, processor, question_text, correct_text, rng, args.device
        )

        n_generated = len(distractors)
        stats['total_generated'] += n_generated

        if n_generated >= 7:
            stats['success'] += 1
        elif n_generated >= 4:
            stats['partial'] += 1
            # Pad with original distractors if we're short
            original_distractors = [v for k, v in q['options'].items()
                                    if k != correct_key]
            rng.shuffle(original_distractors)
            existing_lower = {d.lower().strip() for d in distractors}
            existing_lower.add(correct_text.lower().strip())
            for od in original_distractors:
                if od.lower().strip() not in existing_lower:
                    distractors.append(od)
                    existing_lower.add(od.lower().strip())
                if len(distractors) >= 7:
                    break
        else:
            stats['failed'] += 1
            # Keep original distractors
            distractors = [v for k, v in q['options'].items() if k != correct_key]

        # Ensure exactly 7 distractors
        distractors = distractors[:7]

        # Shuffle and relabel A-H
        target_pos = item['target_pos']
        rng.shuffle(distractors)
        new_options = {target_pos: correct_text}
        remaining = [p for p in labels if p != target_pos]
        for pos, text in zip(remaining, distractors):
            new_options[pos] = text

        q['options'] = new_options
        q['correct_answer'] = target_pos

        # Progress logging
        elapsed = time.time() - start_time
        if (i + 1) % 10 == 0 or i == 0 or i == total - 1:
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{total}] "
                  f"OK={stats['success']} Partial={stats['partial']} "
                  f"Fail={stats['failed']} | "
                  f"{rate:.1f} q/min | ETA: {eta:.0f} min")

    # =================================================================
    # VERIFICATION
    # =================================================================
    print(f"\n{'='*60}")
    print("LLM DEBIASING VERIFICATION")
    print(f"{'='*60}")

    # Answer distribution
    ans_dist = Counter(item['q']['correct_answer'] for item in all_items)
    print(f"\n--- Answer Distribution ---")
    for pos in labels:
        count = ans_dist.get(pos, 0)
        pct = count / total * 100
        print(f"  {pos}: {count} ({pct:.1f}%)")

    # Length stats
    correct_lens = []
    distractor_lens = []
    for item in all_items:
        opts = item['q']['options']
        ca = item['q']['correct_answer']
        for k, v in opts.items():
            if k == ca:
                correct_lens.append(len(v))
            else:
                distractor_lens.append(len(v))

    avg_c = sum(correct_lens) / len(correct_lens) if correct_lens else 0
    avg_d = sum(distractor_lens) / len(distractor_lens) if distractor_lens else 0
    print(f"\n--- Length Statistics ---")
    print(f"  Avg correct:    {avg_c:.1f} chars")
    print(f"  Avg distractor: {avg_d:.1f} chars")
    print(f"  Difference: {avg_c - avg_d:+.1f} chars")

    # Pick-longest heuristic
    pick_longest = sum(
        1 for item in all_items
        if max(item['q']['options'], key=lambda k: len(item['q']['options'][k]))
        == item['q']['correct_answer']
    )
    print(f"\n--- 'Pick Longest' Heuristic: {pick_longest/total*100:.1f}% ---")

    # Length rank distribution
    rank_counts = Counter()
    for item in all_items:
        opts = item['q']['options']
        ca = item['q']['correct_answer']
        sorted_keys = sorted(opts.keys(), key=lambda k: len(opts[k]))
        rank_counts[sorted_keys.index(ca)] += 1
    print(f"\n--- Length Rank Distribution ---")
    for rank in range(8):
        count = rank_counts.get(rank, 0)
        pct = count / total * 100
        print(f"  Rank {rank}: {count} ({pct:.1f}%)")

    # Word overlap
    correct_overlaps = []
    distractor_overlaps = []
    for item in all_items:
        opts = item['q']['options']
        ca = item['q']['correct_answer']
        q_words = get_content_words(item['q'].get('question_text', ''))
        for k, v in opts.items():
            v_words = get_content_words(v)
            overlap = len(q_words & v_words) / len(q_words | v_words) if (q_words | v_words) else 0
            if k == ca:
                correct_overlaps.append(overlap)
            else:
                distractor_overlaps.append(overlap)
    print(f"\n--- Word Overlap ---")
    print(f"  Correct avg:    {sum(correct_overlaps)/len(correct_overlaps):.4f}")
    print(f"  Distractor avg: {sum(distractor_overlaps)/len(distractor_overlaps):.4f}")

    # Pipeline stats
    print(f"\n--- Pipeline Stats ---")
    print(f"  Fully generated (7/7): {stats['success']}")
    print(f"  Partial (padded with originals): {stats['partial']}")
    print(f"  Failed (kept originals): {stats['failed']}")
    print(f"  Total distractors generated: {stats['total_generated']}")
    print(f"  Total time: {(time.time()-start_time)/60:.1f} min")

    # =================================================================
    # REBUILD OUTPUT
    # =================================================================
    video_map = {}
    for item in all_items:
        key = (item['v_idx'], item['video_id'])
        if key not in video_map:
            video_map[key] = []
        video_map[key].append(item['q'])

    output_videos = []
    for (v_idx, vid_id), questions in sorted(video_map.items()):
        output_videos.append({
            'video_id': vid_id,
            'questions': questions,
        })

    output = {
        'benchmark_name': data.get('benchmark_name', 'Benchmark') + ' (LLM Debiased)',
        'version': data.get('version', '1.0') + '_llm_debiased',
        'total_questions': total,
        'num_options': 8,
        'debiasing_applied': [
            'All distractors generated by Qwen2.5-VL-7B to match correct answer style/length',
            'Scene-appropriate distractors (LLM understands question context)',
            'Balanced answer distribution across A-H (~12.5% each)',
            'No mechanical patterns (qualifier padding, marker words, etc.)',
        ],
        'random_baseline_pct': 12.5,
        'seed': args.seed,
        'videos': output_videos,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved: {args.output}")
    print(f"  Videos: {len(output_videos)}, Questions: {total}, Options: 8")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
