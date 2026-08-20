#!/usr/bin/env python3
"""
Post-process LLM-debiased benchmark to remove formatting artifacts.

The LLM (Qwen2.5-VL-7B) generates distractors with a distinctly different
style from human-written correct answers. This script normalizes:

1. PERIOD ENDINGS: Strip trailing periods from all options (29% signal!)
2. CONTRACTIONS: Expand "it's"→"it is", "he's"→"he is", etc. (perfect signal)
3. DASH PREFIX: Remove leading "- " from options
4. LLM CLICHES: Replace stock phrases that only appear in generated text
5. LENGTH: Trim obviously long distractors / ensure no systematic gap
6. CONJUNCTIONS: For correct answers with "and"/"or", inject those into
   some distractors too (reduces conjunction signal)

Usage:
    python postprocess_llm_debiased.py \
        --input movienetTest_v2_8opt_llm_debiased.json \
        --output movienetTest_v2_8opt_final.json
"""

import argparse
import json
import random
import re
import string
from collections import Counter
from copy import deepcopy


# =============================================================================
# FIX 1: PERIOD NORMALIZATION
# =============================================================================

def strip_trailing_period(text):
    """Remove trailing period (and only period — keep ? and !)."""
    t = text.rstrip()
    if t.endswith('.') and not t.endswith('...'):
        return t[:-1].rstrip()
    return t


# =============================================================================
# FIX 2: CONTRACTION EXPANSION
# =============================================================================

CONTRACTIONS = {
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "what's": "what is",
    "who's": "who is",
    "where's": "where is",
    "how's": "how is",
    "let's": "let us",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "won't": "will not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "can't": "cannot",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "they're": "they are",
    "we're": "we are",
    "you're": "you are",
    "i'm": "I am",
    "they've": "they have",
    "we've": "we have",
    "you've": "you have",
    "i've": "I have",
    "they'll": "they will",
    "we'll": "we will",
    "you'll": "you will",
    "i'll": "I will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "they'd": "they would",
    "we'd": "we would",
    "you'd": "you would",
    "i'd": "I would",
    "he'd": "he would",
    "she'd": "she would",
}


def expand_contractions(text):
    """Expand all contractions to full form."""
    result = text
    for contraction, expansion in CONTRACTIONS.items():
        # Case-insensitive replacement preserving first-letter case
        pattern = re.compile(re.escape(contraction), re.IGNORECASE)
        def replace_fn(match):
            original = match.group(0)
            if original[0].isupper():
                return expansion[0].upper() + expansion[1:]
            return expansion
        result = pattern.sub(replace_fn, result)
    return result


# =============================================================================
# FIX 3: DASH PREFIX REMOVAL
# =============================================================================

def strip_dash_prefix(text):
    """Remove leading dash/bullet from option text."""
    t = text.strip()
    if t.startswith('- '):
        return t[2:].strip()
    if t.startswith('-') and len(t) > 1 and t[1] != '-':
        return t[1:].strip()
    return t


# =============================================================================
# FIX 4: LLM CLICHE REPLACEMENT
# =============================================================================

# Phrases that appear frequently in LLM-generated distractors but rarely/never
# in human-written correct answers. Replace with more natural alternatives.
LLM_CLICHES = {
    "a nearby": "a",
    "a serene": "a calm",
    "a lush": "a green",
    "basking in": "sitting in",
    "rocky outcrop": "rock",
    "trying to": "",  # remove — too hedging
    "appears to be": "is",
    "seems to be": "is",
    "can be seen": "is",
    "is seen": "is",
    "is observed": "is",
    "in the vicinity of": "near",
    "in close proximity to": "near",
    "a solitary": "a single",
    "amidst the": "in the",
    "amongst the": "in the",
    "atop a": "on a",
    "utilizing": "using",
    "commences": "starts",
    "endeavors to": "",
    "proceeds to": "",
    "subsequently": "then",
    "adjacent to": "next to",
    "in the midst of": "in the middle of",
    "whilst": "while",
}


def replace_cliches(text):
    """Replace LLM-specific cliche phrases with natural alternatives."""
    result = text
    for cliche, replacement in LLM_CLICHES.items():
        pattern = re.compile(re.escape(cliche), re.IGNORECASE)
        if replacement:
            result = pattern.sub(replacement, result)
        else:
            # Remove entirely, clean up whitespace
            result = pattern.sub('', result)
    # Clean up double spaces and leading/trailing whitespace
    result = re.sub(r'\s+', ' ', result).strip()
    # Fix capitalization after removal at start
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    return result


# =============================================================================
# FIX 5: LENGTH NORMALIZATION (soft)
# =============================================================================

def soft_length_normalize(options, correct_key):
    """If correct answer is much longer than distractors, trim trailing
    non-essential clauses from the correct answer. If distractors are
    much longer, trim them instead.

    Only trims trailing subordinate clauses (after last comma) if the
    length gap is >20 chars.
    """
    correct_text = options[correct_key]
    correct_len = len(correct_text)
    distractor_lens = [len(v) for k, v in options.items() if k != correct_key]
    avg_dist = sum(distractor_lens) / len(distractor_lens) if distractor_lens else correct_len

    new_options = dict(options)

    # If distractors are much longer than correct, trim the longest ones
    if avg_dist > correct_len + 15:
        for k in list(new_options.keys()):
            if k == correct_key:
                continue
            text = new_options[k]
            # Only trim if this distractor is >20 chars longer than correct
            if len(text) > correct_len + 20:
                # Try to trim at the last comma
                last_comma = text.rfind(',')
                if last_comma > correct_len * 0.5:
                    trimmed = text[:last_comma].rstrip()
                    if len(trimmed) >= correct_len * 0.7:
                        new_options[k] = trimmed

    return new_options


# =============================================================================
# FIX 6: CONJUNCTION INJECTION
# =============================================================================

def inject_conjunctions(options, correct_key, rng):
    """If the correct answer uses 'and'/'or'/'while', make sure at least
    2-3 distractors also use those conjunctions to reduce the signal.

    Strategy: for short distractors that describe a single action, combine
    with a plausible additional clause using 'and' or 'while'.
    """
    correct_text = options[correct_key]
    has_and = ' and ' in correct_text.lower()
    has_or = ' or ' in correct_text.lower()
    has_while = ' while ' in correct_text.lower()

    if not (has_and or has_or or has_while):
        return options  # No conjunction in correct, nothing to balance

    # Count how many distractors already have conjunctions
    dist_with_conj = sum(
        1 for k, v in options.items()
        if k != correct_key and (' and ' in v.lower() or ' or ' in v.lower() or ' while ' in v.lower())
    )

    if dist_with_conj >= 3:
        return options  # Already balanced enough

    # Conjunction suffixes to append to distractors that lack them
    AND_SUFFIXES = [
        ' and looking around cautiously',
        ' and holding something in one hand',
        ' and facing the opposite direction',
        ' and shifting weight to one side',
        ' and keeping both arms at their sides',
        ' and tilting the head slightly',
        ' and adjusting their position',
        ' and watching the surroundings',
    ]
    WHILE_SUFFIXES = [
        ' while facing forward',
        ' while keeping still',
        ' while remaining partially hidden',
        ' while maintaining balance',
        ' while gripping tightly with one hand',
        ' while turning slightly to one side',
    ]

    new_options = dict(options)
    target = 3 - dist_with_conj  # How many more distractors need conjunctions
    injected = 0

    # Sort distractors by length (inject into shorter ones to also help length)
    dist_keys = sorted(
        [k for k in options if k != correct_key],
        key=lambda k: len(options[k])
    )

    for k in dist_keys:
        if injected >= target:
            break
        text = new_options[k]
        text_lower = text.lower()
        if ' and ' in text_lower or ' or ' in text_lower or ' while ' in text_lower:
            continue  # Already has conjunction

        if has_and or has_or:
            suffix = rng.choice(AND_SUFFIXES)
        else:
            suffix = rng.choice(WHILE_SUFFIXES)

        # Strip trailing period before appending
        base = text.rstrip()
        if base.endswith('.'):
            base = base[:-1].rstrip()
        new_options[k] = base + suffix
        injected += 1

    return new_options


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-process LLM-debiased benchmark to remove formatting artifacts",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.input) as f:
        data = json.load(f)

    total = sum(len(v['questions']) for v in data['videos'])
    print(f"Loaded {total} questions")

    stats = {
        'periods_stripped': 0,
        'contractions_expanded': 0,
        'dashes_removed': 0,
        'cliches_replaced': 0,
        'lengths_trimmed': 0,
        'conjunctions_injected': 0,
        'lengths_extended': 0,
    }

    for video in data['videos']:
        for q in video['questions']:
            correct_key = q['correct_answer'].strip().upper()
            opts = q['options']

            # Fix 1: Strip trailing periods from ALL options
            for k in list(opts.keys()):
                new_text = strip_trailing_period(opts[k])
                if new_text != opts[k]:
                    stats['periods_stripped'] += 1
                    opts[k] = new_text

            # Fix 2: Expand contractions in ALL options
            for k in list(opts.keys()):
                new_text = expand_contractions(opts[k])
                if new_text != opts[k]:
                    stats['contractions_expanded'] += 1
                    opts[k] = new_text

            # Fix 3: Remove dash prefixes
            for k in list(opts.keys()):
                new_text = strip_dash_prefix(opts[k])
                if new_text != opts[k]:
                    stats['dashes_removed'] += 1
                    opts[k] = new_text

            # Fix 4: Replace LLM cliches
            for k in list(opts.keys()):
                new_text = replace_cliches(opts[k])
                if new_text != opts[k]:
                    stats['cliches_replaced'] += 1
                    opts[k] = new_text

            # Fix 5: Soft length normalization
            old_opts = dict(opts)
            opts_normalized = soft_length_normalize(opts, correct_key)
            if opts_normalized != old_opts:
                stats['lengths_trimmed'] += 1
            q['options'] = opts_normalized

            # Fix 6: Conjunction injection
            old_opts2 = dict(q['options'])
            opts_conj = inject_conjunctions(q['options'], correct_key, rng)
            if opts_conj != old_opts2:
                stats['conjunctions_injected'] += 1
            q['options'] = opts_conj

            # NO Fix 7 (length extension with artificial suffixes)
            # — creates detectable patterns, same mistake as qualifier padding

    # =================================================================
    # VERIFICATION
    # =================================================================
    print(f"\n{'='*60}")
    print("POST-PROCESSING VERIFICATION")
    print(f"{'='*60}")

    # Fixes applied
    print(f"\n--- Fixes Applied ---")
    for fix, count in stats.items():
        print(f"  {fix}: {count}")

    # Period check
    correct_with_period = 0
    distractor_with_period = 0
    total_correct = 0
    total_distractor = 0
    for video in data['videos']:
        for q in video['questions']:
            ca = q['correct_answer']
            for k, v in q['options'].items():
                if k == ca:
                    total_correct += 1
                    if v.rstrip().endswith('.'):
                        correct_with_period += 1
                else:
                    total_distractor += 1
                    if v.rstrip().endswith('.'):
                        distractor_with_period += 1
    print(f"\n--- Period Check ---")
    print(f"  Correct with period: {correct_with_period}/{total_correct} ({correct_with_period/total_correct*100:.1f}%)")
    print(f"  Distractors with period: {distractor_with_period}/{total_distractor} ({distractor_with_period/total_distractor*100:.1f}%)")

    # Contraction check
    contraction_pattern = re.compile(r"(?:it|he|she|that|there|here|what|who|where|how|let|isn|aren|wasn|weren|don|doesn|didn|won|wouldn|couldn|shouldn|can|hasn|haven|hadn|they|we|you|i)'[a-z]", re.IGNORECASE)
    correct_contractions = 0
    distractor_contractions = 0
    for video in data['videos']:
        for q in video['questions']:
            ca = q['correct_answer']
            for k, v in q['options'].items():
                if contraction_pattern.search(v):
                    if k == ca:
                        correct_contractions += 1
                    else:
                        distractor_contractions += 1
    print(f"\n--- Contraction Check ---")
    print(f"  Correct: {correct_contractions}, Distractors: {distractor_contractions}")

    # Length stats
    correct_lens = []
    distractor_lens = []
    for video in data['videos']:
        for q in video['questions']:
            ca = q['correct_answer']
            for k, v in q['options'].items():
                if k == ca:
                    correct_lens.append(len(v))
                else:
                    distractor_lens.append(len(v))
    avg_c = sum(correct_lens) / len(correct_lens)
    avg_d = sum(distractor_lens) / len(distractor_lens)
    print(f"\n--- Length Stats ---")
    print(f"  Avg correct: {avg_c:.1f}, Avg distractor: {avg_d:.1f}, Gap: {avg_c-avg_d:+.1f}")

    # Pick longest heuristic
    pick_longest = 0
    for video in data['videos']:
        for q in video['questions']:
            ca = q['correct_answer']
            longest = max(q['options'], key=lambda k: len(q['options'][k]))
            if longest == ca:
                pick_longest += 1
    print(f"  'Pick longest' accuracy: {pick_longest/total*100:.1f}%")

    # Length rank distribution
    rank_counts = Counter()
    for video in data['videos']:
        for q in video['questions']:
            ca = q['correct_answer']
            sorted_keys = sorted(q['options'].keys(), key=lambda k: len(q['options'][k]))
            rank_counts[sorted_keys.index(ca)] += 1
    print(f"\n--- Length Rank Distribution ---")
    for rank in range(8):
        count = rank_counts.get(rank, 0)
        pct = count / total * 100
        print(f"  Rank {rank}: {count} ({pct:.1f}%)")

    # Conjunction stats
    correct_conj = 0
    distractor_conj = 0
    for video in data['videos']:
        for q in video['questions']:
            ca = q['correct_answer']
            for k, v in q['options'].items():
                has = ' and ' in v.lower() or ' or ' in v.lower() or ' while ' in v.lower()
                if k == ca:
                    if has: correct_conj += 1
                else:
                    if has: distractor_conj += 1
    print(f"\n--- Conjunction Stats ---")
    print(f"  Correct with conj: {correct_conj}/{total_correct} ({correct_conj/total_correct*100:.1f}%)")
    print(f"  Distractors with conj: {distractor_conj}/{total_distractor} ({distractor_conj/total_distractor*100:.1f}%)")

    # Answer distribution
    ans_dist = Counter()
    for video in data['videos']:
        for q in video['questions']:
            ans_dist[q['correct_answer']] += 1
    print(f"\n--- Answer Distribution ---")
    for pos in string.ascii_uppercase[:8]:
        count = ans_dist.get(pos, 0)
        print(f"  {pos}: {count} ({count/total*100:.1f}%)")

    # Save
    data['debiasing_applied'] = data.get('debiasing_applied', []) + [
        'Post-processed: trailing periods stripped from all options',
        'Post-processed: contractions expanded',
        'Post-processed: LLM cliche phrases replaced',
        'Post-processed: conjunction injection for balance',
        'Post-processed: soft length normalization',
    ]

    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved: {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
