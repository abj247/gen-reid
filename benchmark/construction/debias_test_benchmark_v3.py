#!/usr/bin/env python3
"""
Nuclear debiasing: replace ALL distractors with correct answers from other questions.

Why this works:
  - Every option in the MCQ is a "correct answer to some question"
  - Identical quality, length, detail, style — no statistical signal
  - LLM can't use length, specificity, connector count, or quality as cues
  - Only remaining signal: semantic coherence between question and its answer

Distractor selection priority:
  1. Same video + same capability (max 1-3, most contextually relevant)
  2. Same capability, different video (length-matched, format-matched)

Additional debiasing:
  - For each question, ensure at least 2 distractors share keywords with the question
    (reduces word-overlap signal where only the correct answer echoes question words)
  - Balanced answer distribution across A-H (~12.5% each)

Usage:
    python debias_test_benchmark_v3.py \\
        --input movienetTest_v2_8opt.json \\
        --output movienetTest_v2_8opt_nuclear.json \\
        --seed 42
"""

import argparse
import json
import random
import re
import string
from collections import Counter, defaultdict
from copy import deepcopy


# =============================================================================
# HELPERS
# =============================================================================

def classify_answer_format(text):
    """Classify answer by grammatical structure."""
    t = text.strip()
    if re.match(r'^(he|she|they|it|the man|the woman|the boy|the girl|the person)\s',
                t, re.IGNORECASE):
        return "pronoun_subject"
    if re.match(r'^[A-Z][a-z]+ing\s', t):
        return "gerund"
    if re.match(r'^(in|on|at|near|behind|under|inside|outside|beside|between|next to|'
                r'across|above|below|standing|sitting|lying|leaning)\s',
                t, re.IGNORECASE):
        return "locative"
    if re.match(r'^(a|an|the)\s', t, re.IGNORECASE):
        return "article_noun"
    if re.match(r'^(it|its)\s', t, re.IGNORECASE):
        return "it_subject"
    if len(t.split()) <= 4:
        return "short_phrase"
    return "other"


def get_content_words(text):
    """Extract content words (nouns, verbs, adjectives) from text.
    Excludes common stop words."""
    STOP = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'and', 'but', 'or', 'if', 'while',
        'that', 'this', 'these', 'those', 'what', 'which', 'who', 'whom',
        'his', 'her', 'its', 'their', 'my', 'your', 'he', 'she', 'it',
        'they', 'we', 'you', 'him', 'them', 'us', 'me',
    }
    words = set(re.findall(r'[a-z]+', text.lower()))
    return words - STOP


def compute_word_overlap(text1, text2):
    """Compute Jaccard overlap of content words."""
    w1 = get_content_words(text1)
    w2 = get_content_words(text2)
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


# =============================================================================
# BUILD CORRECT-ANSWER POOLS
# =============================================================================

def build_correct_answer_pools(data):
    """Build pools of correct answers indexed by capability and video.

    Returns:
        video_cap: {(video_id, capability): [(text, format, length, qid)]}
        cap_pool:  {capability: [(text, format, length, video_id, qid)]}
    """
    video_cap = defaultdict(list)
    cap_pool = defaultdict(list)

    for video in data['videos']:
        vid = video['video_id']
        for q in video['questions']:
            correct_key = q['correct_answer'].strip().upper()
            text = q['options'][correct_key]
            cap = q.get('metadata', {}).get('capability', 'unknown')
            qid = q.get('question_id', '')
            fmt = classify_answer_format(text)
            length = len(text)

            entry_local = (text, fmt, length, qid)
            entry_global = (text, fmt, length, vid, qid)

            video_cap[(vid, cap)].append(entry_local)
            cap_pool[cap].append(entry_global)

    return video_cap, cap_pool


def select_distractors(correct_text, question_text, capability, video_id,
                       question_id, video_cap, cap_pool, rng, n_distractors=7):
    """Select 7 distractors from correct-answer pools.

    Strategy:
      1. Same video + same capability (most plausible, 1-3 available)
      2. Same capability cross-video (length/format matched)
      3. Prefer distractors that share some keywords with the question
         (reduces the signal where ONLY the correct answer echoes question words)
    """
    correct_len = len(correct_text)
    correct_fmt = classify_answer_format(correct_text)
    question_words = get_content_words(question_text)

    selected = []
    selected_lower = {correct_text.lower().strip()}

    # ── Priority 1: Same video + same capability ─────────────────────
    local_pool = video_cap.get((video_id, capability), [])
    for text, fmt, length, qid in local_pool:
        if qid == question_id:
            continue
        if text.lower().strip() in selected_lower:
            continue
        selected.append(text)
        selected_lower.add(text.lower().strip())

    # ── Priority 2: Same capability, cross-video ─────────────────────
    if len(selected) < n_distractors:
        global_pool = cap_pool.get(capability, [])

        # Score candidates by: format match + length proximity + keyword overlap
        candidates = []
        for text, fmt, length, vid, qid in global_pool:
            if vid == video_id and qid == question_id:
                continue
            if text.lower().strip() in selected_lower:
                continue
            # Length filter: within ±40% of correct answer length
            if correct_len > 0:
                ratio = length / correct_len
                if ratio < 0.6 or ratio > 1.4:
                    continue

            # Score components
            fmt_penalty = 0 if fmt == correct_fmt else 300
            len_dist = abs(length - correct_len)

            # Bonus for sharing keywords with the question
            # (this reduces the signal where only correct echoes question words)
            distractor_words = get_content_words(text)
            shared_with_q = len(question_words & distractor_words)
            keyword_bonus = -shared_with_q * 50  # lower score = better

            score = fmt_penalty + len_dist + keyword_bonus
            candidates.append((score, text))

        candidates.sort(key=lambda x: x[0])

        needed = n_distractors - len(selected)
        # Pick from top candidates with some randomness
        top_n = min(needed * 3, len(candidates))
        pool = candidates[:top_n]
        rng.shuffle(pool)

        for _, text in pool:
            if text.lower().strip() in selected_lower:
                continue
            selected.append(text)
            selected_lower.add(text.lower().strip())
            if len(selected) >= n_distractors:
                break

    # ── Fallback: relax length constraint ─────────────────────────────
    if len(selected) < n_distractors:
        global_pool = cap_pool.get(capability, [])
        for text, fmt, length, vid, qid in global_pool:
            if text.lower().strip() in selected_lower:
                continue
            selected.append(text)
            selected_lower.add(text.lower().strip())
            if len(selected) >= n_distractors:
                break

    return selected[:n_distractors]


# =============================================================================
# SHUFFLE & RELABEL
# =============================================================================

def shuffle_and_relabel(correct_text, distractors, target_position, rng):
    """Assign correct to target_position, shuffle distractors into rest."""
    labels = list(string.ascii_uppercase[:8])
    rng.shuffle(distractors)

    new_options = {}
    new_options[target_position] = correct_text
    remaining = [p for p in labels if p != target_position]
    for pos, text in zip(remaining, distractors[:len(remaining)]):
        new_options[pos] = text

    return new_options


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Nuclear debiasing: correct answers as distractors",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.input) as f:
        data = json.load(f)

    total_q = sum(len(v['questions']) for v in data['videos'])
    print(f"Loaded {total_q} questions from {len(data['videos'])} videos")

    # Build correct-answer pools
    print("Building correct-answer pools...")
    video_cap, cap_pool = build_correct_answer_pools(data)
    for cap, pool in sorted(cap_pool.items()):
        print(f"  {cap}: {len(pool)} correct answers available")

    # Flatten items
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

    # Balanced target positions
    labels = list(string.ascii_uppercase[:8])
    indices = list(range(total))
    rng.shuffle(indices)
    for rank, idx in enumerate(indices):
        all_items[idx]['target_pos'] = labels[rank % 8]

    # Stats
    stats = {
        'original_length_ranks': [],
        'final_length_ranks': [],
        'same_video_used': 0,
        'cross_video_used': 0,
        'short_distractors': 0,
    }

    # Process each question
    print("Replacing all distractors...")
    for i, item in enumerate(all_items):
        q = item['q']
        correct_key = q['correct_answer'].strip().upper()
        correct_text = q['options'][correct_key]
        question_text = q.get('question_text', q.get('question', ''))
        cap = q.get('metadata', {}).get('capability', 'unknown')
        vid = item['video_id']
        qid = q.get('question_id', '')

        # Record original length rank
        orig_sorted = sorted(q['options'].keys(), key=lambda k: len(q['options'][k]))
        stats['original_length_ranks'].append(orig_sorted.index(correct_key))

        # Select 7 new distractors (all correct answers from other questions)
        distractors = select_distractors(
            correct_text, question_text, cap, vid, qid,
            video_cap, cap_pool, rng, n_distractors=7
        )

        # Count same-video vs cross-video
        local_texts = {e[0] for e in video_cap.get((vid, cap), []) if e[3] != qid}
        for d in distractors:
            if d in local_texts:
                stats['same_video_used'] += 1
            else:
                stats['cross_video_used'] += 1

        # Shuffle and relabel
        new_options = shuffle_and_relabel(
            correct_text, distractors, item['target_pos'], rng
        )
        q['options'] = new_options
        q['correct_answer'] = item['target_pos']

        # Record final length rank
        final_sorted = sorted(new_options.keys(), key=lambda k: len(new_options[k]))
        stats['final_length_ranks'].append(final_sorted.index(item['target_pos']))

        if len(distractors) < 7:
            stats['short_distractors'] += 1

    # =================================================================
    # VERIFICATION
    # =================================================================
    print(f"\n{'='*60}")
    print("NUCLEAR DEBIASING VERIFICATION")
    print(f"{'='*60}")

    # 1. Answer distribution
    ans_dist = Counter(item['q']['correct_answer'] for item in all_items)
    print(f"\n--- Answer Distribution ---")
    for pos in labels:
        count = ans_dist.get(pos, 0)
        pct = count / total * 100
        print(f"  {pos}: {count} ({pct:.1f}%)")

    # 2. Length rank BEFORE
    orig_ranks = Counter(stats['original_length_ranks'])
    print(f"\n--- Length Rank BEFORE ---")
    for rank in range(8):
        count = orig_ranks.get(rank, 0)
        pct = count / total * 100
        print(f"  Rank {rank}: {count} ({pct:.1f}%)")

    # 3. Length rank AFTER
    final_ranks = Counter(stats['final_length_ranks'])
    print(f"\n--- Length Rank AFTER ---")
    max_rank_pct = 0
    for rank in range(8):
        count = final_ranks.get(rank, 0)
        pct = count / total * 100
        flag = " *" if pct > 18 else ""
        print(f"  Rank {rank}: {count} ({pct:.1f}%){flag}")
        max_rank_pct = max(max_rank_pct, pct)

    # 4. Pick-longest heuristic
    pick_longest = 0
    for item in all_items:
        opts = item['q']['options']
        ca = item['q']['correct_answer']
        longest = max(opts.keys(), key=lambda k: len(opts[k]))
        if longest == ca:
            pick_longest += 1
    print(f"\n--- 'Pick Longest' Heuristic ---")
    print(f"  BEFORE: {sum(1 for r in stats['original_length_ranks'] if r==7)/total*100:.1f}%")
    print(f"  AFTER:  {pick_longest/total*100:.1f}%")

    # 5. Average correct vs distractor length
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
    avg_c = sum(correct_lens) / len(correct_lens)
    avg_d = sum(distractor_lens) / len(distractor_lens)
    print(f"\n--- Length Statistics ---")
    print(f"  Avg correct length:    {avg_c:.1f} chars")
    print(f"  Avg distractor length: {avg_d:.1f} chars")
    print(f"  Difference: {avg_c - avg_d:+.1f} chars ({(avg_c-avg_d)/avg_d*100:+.1f}%)")

    # 6. Word overlap (correct vs distractor with question)
    correct_overlaps = []
    distractor_overlaps = []
    for item in all_items:
        opts = item['q']['options']
        ca = item['q']['correct_answer']
        q_text = item['q'].get('question_text', item['q'].get('question', ''))
        for k, v in opts.items():
            overlap = compute_word_overlap(q_text, v)
            if k == ca:
                correct_overlaps.append(overlap)
            else:
                distractor_overlaps.append(overlap)
    avg_co = sum(correct_overlaps) / len(correct_overlaps)
    avg_do = sum(distractor_overlaps) / len(distractor_overlaps)
    print(f"\n--- Question-Answer Word Overlap ---")
    print(f"  Correct avg:    {avg_co:.4f}")
    print(f"  Distractor avg: {avg_do:.4f}")
    print(f"  Gap: {avg_co - avg_do:+.4f}")

    # 7. Pipeline stats
    print(f"\n--- Pipeline Stats ---")
    print(f"  Same-video distractors used: {stats['same_video_used']}")
    print(f"  Cross-video distractors used: {stats['cross_video_used']}")
    print(f"  Questions with <7 distractors: {stats['short_distractors']}")

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
        'benchmark_name': data.get('benchmark_name', 'Benchmark') + ' (Nuclear Debiased)',
        'version': data.get('version', '1.0') + '_nuclear_debiased',
        'total_questions': total,
        'num_options': 8,
        'debiasing_applied': [
            'All 7 distractors replaced with correct answers from other questions',
            'Same-capability matching (format + length matched)',
            'Same-video priority for contextual relevance',
            'Keyword overlap balancing (distractors share question words too)',
            'Balanced answer distribution across A-H (~12.5% each)',
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
