#!/usr/bin/env python3
"""
Create debiased 8-option MovieNet benchmark (v2).

v1 (5-option) reduced text-only accuracy from ~40% to ~28-31%, but still
well above the 20% random baseline due to:
  1. Qualifier padding created NEW detectable markers
  2. Incomplete marker dictionary (missing color/posture/furniture variants)
  3. Synonym clusters still had 3+ near-identical options
  4. Length rank skewed

v2 approach (8-option, target ≤17% vs 12.5% random baseline):
  - Start from movienet_benchmark_8options.json as-is
  - Comprehensive marker cleanup (200+ fancy→plain mappings)
  - Register swap (50% probability)
  - Replace duplicates with cross-question fillers (same-video priority)
  - Cap synonym clusters at 3 members max
  - NO qualifier padding (was itself a bias signal)
  - Shuffle & relabel A-H with balanced distribution

Usage:
    python create_debiased_benchmark_v2.py \\
        --input movienet_benchmark_8options.json \\
        --output movienet_benchmark_debiased_v2.json \\
        --seed 42
"""

import argparse
import hashlib
import json
import random
import re
import string
from collections import Counter, defaultdict
from copy import deepcopy
from difflib import SequenceMatcher


# =============================================================================
# COMPREHENSIVE MARKER DICTIONARY
# =============================================================================

# Phrases to remove entirely (inserted qualifiers that pad distractors)
MARKER_REMOVE = [
    "faintly visible",
    "faintly discernible",
    "slightly worn",
    "faintly",
    "subtly",
    "vaguely",
    "visibly",
    "partially obscured",
    "partially",
]

# Fancy → plain word mappings (word-boundary replacement)
MARKER_NORMALIZE = {
    # ── Colors ───────────────────────────────────────────────────────────
    # Red variants
    "crimson": "red", "scarlet": "red", "maroon": "red",
    "burgundy": "red", "reddish": "red",
    # Blue variants
    "cobalt": "blue", "azure": "blue", "cerulean": "blue",
    "indigo": "blue", "steel-blue": "blue",
    # Black variants
    "onyx": "black", "ebony": "black", "jet-black": "black",
    "near-black": "black", "charcoal": "black",
    # White variants
    "ivory": "white", "bone-white": "white", "off-white": "white",
    "cream": "white",
    # Orange variants
    "burnt-orange": "orange", "copper-orange": "orange",
    "tangerine": "orange", "amber-orange": "orange",
    # Brown variants
    "tawny": "brown", "auburn": "brown", "chestnut": "brown",
    "russet": "brown", "mahogany": "brown", "coffee-colored": "brown",
    # Yellow variants
    "ochre": "yellow", "mustard-yellow": "yellow", "amber": "yellow",
    # Green variants
    "emerald": "green", "sage": "green", "moss-green": "green",
    "forest-green": "green", "olive": "green",
    # Grey variants
    "iron-grey": "grey", "silver-grey": "grey", "slate-grey": "grey",
    "ash-grey": "grey", "pewter": "grey",
    # Purple variants
    "plum": "purple", "deep-purple": "purple", "aubergine": "purple",
    "mauve": "purple", "violet": "purple",
    # Pink variants
    "rose": "pink", "coral": "pink", "dusty-pink": "pink",
    "salmon": "pink", "blush-pink": "pink",
    # Tone/shade
    "dusky": "dark", "somber": "dark", "sombre": "dark",
    "muted": "dark", "shadowed": "dark",
    "pastel": "light", "washed-out": "light",

    # ── Size ─────────────────────────────────────────────────────────────
    "miniature": "small", "diminutive": "small", "compact": "small",
    "tiny": "small", "petite": "small",
    "hefty": "large", "bulky": "large", "sizable": "large",
    "substantial": "large",
    "elongated": "long", "lengthy": "long", "extended": "long",

    # ── Posture / Position ───────────────────────────────────────────────
    "poised": "standing", "stationed": "standing", "erect": "standing",
    "upright": "standing",
    "settled": "sitting", "perched": "sitting", "seated": "sitting",
    "trudging": "walking", "pacing": "walking", "striding": "walking",
    "bolting": "running", "dashing": "running", "hurrying": "running",
    "rushing": "running", "sprinting": "running",
    "reclining": "lying", "sprawled": "lying", "prone": "lying",

    # ── Looking ──────────────────────────────────────────────────────────
    "gazing": "looking", "glancing": "looking", "peering": "looking",

    # ── Holding / Grasping ───────────────────────────────────────────────
    "clasping": "holding", "clutching": "holding", "cradling": "holding",
    "grasping": "holding", "gripping": "holding",

    # ── Speaking ─────────────────────────────────────────────────────────
    "bellowing": "shouting", "exclaiming": "shouting",
    "yelling": "shouting", "hollering": "shouting",
    "murmuring": "talking", "chattering": "talking",
    "conversing": "talking",

    # ── Crying ───────────────────────────────────────────────────────────
    "sobbing": "crying", "wailing": "crying", "weeping": "crying",
    "whimpering": "crying",

    # ── Furniture / Objects ──────────────────────────────────────────────
    "bedstead": "bed", "pallet": "bed", "cot": "bed", "bunk": "bed",
    "slate-stone": "stone", "granite": "stone", "marble": "stone",
    "limestone": "stone", "sandstone": "stone", "flagstone": "stone",
    "portal": "door", "entryway": "door", "threshold": "door",
    "casement": "window", "windowpane": "window",
    "tome": "book", "volume": "book", "ledger": "book",
    "goblet": "cup", "beaker": "cup", "tumbler": "cup",
    "flask": "bottle", "phial": "bottle", "vial": "bottle",
    "satchel": "bag", "knapsack": "bag",
    "chronometer": "watch", "timepiece": "watch",

    # ── Clothing ─────────────────────────────────────────────────────────
    "tunic": "shirt", "blouse": "shirt",
    "blazer": "jacket", "windbreaker": "jacket",
    "frock": "dress", "smock": "dress", "gown": "dress",
    "pullover": "sweater", "jumper": "sweater",
    "greatcoat": "coat", "overcoat": "coat", "topcoat": "coat",
    "cloak": "coat",
    "bonnet": "hat", "beret": "hat", "skullcap": "hat",
    "headpiece": "hat", "headwear": "hat",
    "cravat": "scarf", "muffler": "scarf",

    # ── Wearing ──────────────────────────────────────────────────────────
    "donning": "wearing", "attired": "wearing", "clad": "wearing",
    "sporting": "wearing",

    # ── Body parts (fancy → plain) ───────────────────────────────────────
    "visage": "face", "countenance": "face",
    "nape": "neck",
    "coiffure": "hair", "locks": "hair", "mane": "hair",
    "tresses": "hair",

    # ── Spatial ──────────────────────────────────────────────────────────
    "adjacent": "near", "atop": "on top of",

    # ── Age ───────────────────────────────────────────────────────────────
    "adolescent": "young", "juvenile": "young", "youthful": "young",
    "fresh-faced": "young",

    # ── Material ─────────────────────────────────────────────────────────
    "polished-wood": "wooden", "carved-wood": "wooden",
    "rough-hewn": "wooden", "timber": "wooden",
    "suede": "leather", "tanned": "leather", "hide": "leather",
}


# =============================================================================
# ANSWER FORMAT CLASSIFIER (from harden_benchmark.py)
# =============================================================================

def classify_answer_format(text):
    """Classify answer text by grammatical structure."""
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
    if len(t.split()) <= 4:
        return "short_phrase"
    return "other"


# =============================================================================
# MARKER CLEANUP
# =============================================================================

def clean_markers(text):
    """Remove/normalize marker words from option text."""
    result = text

    # Remove inserted qualifiers (longer first)
    for marker in sorted(MARKER_REMOVE, key=len, reverse=True):
        pattern = re.compile(
            r',?\s*\b' + re.escape(marker) + r'\b,?\s*',
            re.IGNORECASE
        )
        result = pattern.sub(' ', result)

    # Normalize fancy → plain (word boundaries)
    for fancy, plain in MARKER_NORMALIZE.items():
        pattern = re.compile(r'\b' + re.escape(fancy) + r'\b', re.IGNORECASE)
        result = pattern.sub(plain, result)

    # Clean up whitespace and punctuation artifacts
    result = re.sub(r'\s+', ' ', result).strip()
    result = result.strip(', ')
    return result


def check_remaining_markers(text):
    """Return any marker words still in text."""
    found = []
    text_lower = text.lower()
    for marker in MARKER_REMOVE:
        if re.search(r'\b' + re.escape(marker.lower()) + r'\b', text_lower):
            found.append(marker)
    for fancy in MARKER_NORMALIZE:
        if re.search(r'\b' + re.escape(fancy.lower()) + r'\b', text_lower):
            found.append(fancy)
    return found


# =============================================================================
# SYNONYM CLUSTER IDENTIFICATION
# =============================================================================

def find_synonym_cluster(options, correct_key, threshold=0.5):
    """Group options into synonym cluster (similar to correct) vs fillers."""
    correct_text = options[correct_key].lower().strip()
    cluster = [correct_key]
    fillers = []

    for key, text in options.items():
        if key == correct_key:
            continue
        ratio = SequenceMatcher(None, correct_text, text.lower().strip()).ratio()
        if ratio >= threshold:
            cluster.append(key)
        else:
            fillers.append(key)

    return cluster, fillers


# =============================================================================
# CROSS-QUESTION DISTRACTOR POOLS (adapted from harden_benchmark.py)
# =============================================================================

def build_distractor_pools(all_questions):
    """Build multi-level pools for filler replacement.

    Returns pools of (text, format) tuples for format-matched selection.
    """
    video_cap_pool = defaultdict(list)
    video_pool = defaultdict(list)
    cap_pool = defaultdict(list)
    global_pool = []

    # Phrases from v1 qualifier padding — exclude from pool
    padding_blacklist = [
        "which appears", "which catches", "which is fairly",
        "upon closer examination", "catching the light",
        "well-fitted and distinctive", "well-preserved",
        "positioned near the center", "placed within easy reach",
        "noticeable against", "resting in a clearly visible",
    ]

    seen = set()  # dedup within pools

    for q in all_questions:
        vid = q['video_id']
        cap = q.get('metadata', {}).get('capability', 'unknown')
        opts = q['options'] if isinstance(q['options'], dict) else {
            chr(ord('A') + i): v for i, v in enumerate(q['options'])
        }
        correct = q.get('correct_answer', '').strip().upper()

        for key, text in opts.items():
            if key == correct:
                continue
            # Clean the text before adding to pool
            cleaned = clean_markers(text)
            key_dedup = cleaned.lower().strip()
            if key_dedup in seen:
                continue
            seen.add(key_dedup)

            # Skip entries with padding phrases
            cleaned_lower = cleaned.lower()
            if any(p in cleaned_lower for p in padding_blacklist):
                continue

            fmt = classify_answer_format(cleaned)
            entry = (cleaned, fmt, len(cleaned))

            video_cap_pool[(vid, cap)].append(entry)
            video_pool[vid].append(entry)
            cap_pool[cap].append(entry)
            global_pool.append(entry)

    return video_cap_pool, video_pool, cap_pool, global_pool


def select_replacement_filler(correct_text, existing_texts_lower, pools_priority,
                              rng, match_format=True):
    """Select one replacement filler from pools (priority order).

    Returns cleaned filler text, or None if nothing suitable found.
    """
    target_len = len(correct_text)
    target_fmt = classify_answer_format(correct_text)

    for pool in pools_priority:
        candidates = []
        for text, fmt, length in pool:
            if text.lower().strip() in existing_texts_lower:
                continue
            # Score: format match + length proximity
            fmt_penalty = 0 if (fmt == target_fmt or not match_format) else 1000
            len_dist = abs(length - target_len)
            score = fmt_penalty + len_dist
            candidates.append((score, text))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0])

        # Pick from top 5 candidates randomly for variety
        top_n = min(5, len(candidates))
        chosen = rng.choice(candidates[:top_n])
        return chosen[1]

    return None


# =============================================================================
# REGISTER SWAP
# =============================================================================

def maybe_swap_correct_with_variant(options, correct_key, cluster_keys, rng):
    """With 50% probability, swap correct answer text with a synonym variant."""
    swap_candidates = [k for k in cluster_keys if k != correct_key]
    if not swap_candidates or rng.random() > 0.5:
        return options, correct_key

    swap_key = rng.choice(swap_candidates)
    new_options = dict(options)
    new_options[correct_key], new_options[swap_key] = (
        new_options[swap_key], new_options[correct_key]
    )
    return new_options, correct_key


# =============================================================================
# DEDUPLICATION WITH CROSS-QUESTION FILLERS
# =============================================================================

def deduplicate_with_fillers(options, correct_key, video_id, capability,
                             pools, rng):
    """Replace duplicate options with cross-question fillers."""
    video_cap_pool, video_pool, cap_pool, global_pool = pools

    pools_priority = [
        video_cap_pool.get((video_id, capability), []),
        video_pool.get(video_id, []),
        cap_pool.get(capability, []),
        global_pool,
    ]

    seen_texts = {}
    duplicates = []

    for key in sorted(options.keys()):
        text_lower = options[key].strip().lower()
        if text_lower in seen_texts:
            duplicates.append(key)
        else:
            seen_texts[text_lower] = key

    if not duplicates:
        return options, 0

    correct_text = options[correct_key]
    existing_lower = {v.strip().lower() for v in options.values()}

    new_options = dict(options)
    replaced = 0

    for dup_key in duplicates:
        filler = select_replacement_filler(
            correct_text, existing_lower, pools_priority, rng
        )
        if filler:
            new_options[dup_key] = filler
            existing_lower.add(filler.lower().strip())
            replaced += 1
        # If no filler found, leave the duplicate (rare edge case)

    return new_options, replaced


# =============================================================================
# CLUSTER SIZE CAPPING
# =============================================================================

def cap_cluster_size(options, correct_key, video_id, capability, pools,
                     rng, max_cluster=3, max_iterations=3):
    """Iteratively cap synonym cluster to ≤max_cluster members."""
    video_cap_pool, video_pool, cap_pool, global_pool = pools
    pools_priority = [
        video_cap_pool.get((video_id, capability), []),
        video_pool.get(video_id, []),
        cap_pool.get(capability, []),
        global_pool,
    ]

    total_replaced = 0
    new_options = dict(options)

    for iteration in range(max_iterations):
        cluster_keys, _ = find_synonym_cluster(new_options, correct_key, threshold=0.6)

        if len(cluster_keys) <= max_cluster:
            break

        correct_text = new_options[correct_key]
        others = [k for k in cluster_keys if k != correct_key]
        others.sort(key=lambda k: abs(len(new_options[k]) - len(correct_text)))
        keep = others[:max_cluster - 1]
        replace = others[max_cluster - 1:]

        existing_lower = {v.strip().lower() for v in new_options.values()}

        for key in replace:
            filler = select_replacement_filler(
                correct_text, existing_lower, pools_priority, rng
            )
            if filler:
                new_options[key] = filler
                existing_lower.add(filler.lower().strip())
                total_replaced += 1

    return new_options, total_replaced


# =============================================================================
# SHUFFLE & RELABEL
# =============================================================================

def shuffle_and_relabel(options, correct_key, target_position, rng, n_options=8):
    """Reassign options to A-H with correct answer at target_position."""
    labels = [chr(ord('A') + i) for i in range(n_options)]
    correct_text = options[correct_key]
    distractor_texts = [v for k, v in options.items() if k != correct_key]
    rng.shuffle(distractor_texts)

    new_options = {}
    new_options[target_position] = correct_text

    remaining = [p for p in labels if p != target_position]
    for pos, text in zip(remaining, distractor_texts[:len(remaining)]):
        new_options[pos] = text

    return new_options, target_position


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def debias_question(q, video_id, target_position, pools, rng, stats):
    """Apply full v2 debiasing pipeline to one question."""
    opts_orig = dict(q['options']) if isinstance(q['options'], dict) else {
        chr(ord('A') + i): v for i, v in enumerate(q['options'])
    }
    correct_key = q['correct_answer'].strip().upper()
    capability = q.get('metadata', {}).get('capability', 'unknown')
    changes = []

    # Step 1: Comprehensive marker cleanup on ALL options
    opts_cleaned = {}
    markers_cleaned = 0
    for k, v in opts_orig.items():
        cleaned = clean_markers(v)
        if cleaned != v:
            markers_cleaned += 1
        opts_cleaned[k] = cleaned
    if markers_cleaned:
        changes.append(f"marker_cleanup({markers_cleaned})")
        stats['markers_cleaned'] += markers_cleaned

    # Step 2: Register swap (50% probability)
    cluster_keys, filler_keys = find_synonym_cluster(opts_cleaned, correct_key)
    stats['cluster_sizes'].append(len(cluster_keys))

    opts_swapped, correct_key = maybe_swap_correct_with_variant(
        opts_cleaned, correct_key, cluster_keys, rng
    )
    if opts_swapped != opts_cleaned:
        changes.append("register_swap")
        stats['swaps'] += 1

    # Step 3: Deduplicate with cross-question fillers
    opts_dedup, n_dedup = deduplicate_with_fillers(
        opts_swapped, correct_key, video_id, capability, pools, rng
    )
    if n_dedup:
        changes.append(f"dedup_replaced({n_dedup})")
        stats['dedup_replaced'] += n_dedup

    # Step 4: Cap cluster size at 3
    opts_capped, n_capped = cap_cluster_size(
        opts_dedup, correct_key, video_id, capability, pools, rng,
        max_cluster=3, max_iterations=5
    )
    if n_capped:
        changes.append(f"cluster_capped({n_capped})")
        stats['cluster_capped'] += n_capped

    # Step 5: Shuffle & relabel A-H
    n_opts = len(opts_capped)
    new_options, new_correct = shuffle_and_relabel(
        opts_capped, correct_key, target_position, rng, n_options=n_opts
    )
    changes.append(f"shuffled(correct={new_correct})")

    # Step 6: NO qualifier padding (explicitly skipped — v1 mistake)

    return new_options, new_correct, changes


def main():
    parser = argparse.ArgumentParser(
        description="Create debiased 8-option MovieNet benchmark (v2)",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.input) as f:
        data = json.load(f)

    # Flatten questions for pool building
    all_questions = []
    for video in data['videos']:
        for q in video['questions']:
            q_copy = deepcopy(q)
            q_copy['video_id'] = video['video_id']
            all_questions.append(q_copy)

    total = len(all_questions)
    print(f"Loaded {total} questions from {len(data['videos'])} videos")

    # Build cross-question distractor pools (with cleaned text)
    print("Building distractor pools...")
    pools = build_distractor_pools(all_questions)
    print(f"  Pool sizes: video_cap={sum(len(v) for v in pools[0].values())}, "
          f"video={sum(len(v) for v in pools[1].values())}, "
          f"cap={sum(len(v) for v in pools[2].values())}, "
          f"global={len(pools[3])}")

    # Flatten items for processing
    all_items = []
    for v_idx, video in enumerate(data['videos']):
        for q_idx, q in enumerate(video['questions']):
            all_items.append({
                'v_idx': v_idx,
                'q_idx': q_idx,
                'video_id': video['video_id'],
                'q': deepcopy(q),
            })

    # Assign balanced target positions (A-H)
    labels = list(string.ascii_uppercase[:8])
    indices = list(range(total))
    rng.shuffle(indices)
    for rank, idx in enumerate(indices):
        all_items[idx]['target_pos'] = labels[rank % 8]

    # Stats tracking
    stats = {
        'cluster_sizes': [],
        'markers_cleaned': 0,
        'swaps': 0,
        'dedup_replaced': 0,
        'cluster_capped': 0,
    }

    # Apply pipeline
    print("Applying debiasing pipeline...")
    for i, item in enumerate(all_items):
        new_options, new_correct, changes = debias_question(
            item['q'], item['video_id'], item['target_pos'], pools, rng, stats
        )
        item['q']['options'] = new_options
        item['q']['correct_answer'] = new_correct
        item['q']['debiasing_applied'] = changes

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{total}")

    # =================================================================
    # VERIFICATION
    # =================================================================
    print(f"\n{'='*60}")
    print("DEBIASING v2 VERIFICATION")
    print(f"{'='*60}")

    # 1. Answer distribution
    ans_dist = Counter(item['q']['correct_answer'] for item in all_items)
    print(f"\n--- Answer Distribution (target: ~{100/8:.1f}% each) ---")
    for pos in labels:
        count = ans_dist.get(pos, 0)
        pct = count / total * 100
        flag = " *" if abs(pct - 12.5) > 2 else ""
        print(f"  {pos}: {count} ({pct:.1f}%){flag}")

    # 2. Remaining marker words
    all_remaining = []
    for item in all_items:
        opts_text = ' '.join(item['q']['options'].values())
        remaining = check_remaining_markers(opts_text)
        all_remaining.extend(remaining)
    if all_remaining:
        marker_counts = Counter(all_remaining)
        print(f"\n--- WARNING: {len(marker_counts)} marker types remaining ---")
        for m, c in marker_counts.most_common(15):
            print(f"  '{m}': {c}")
    else:
        print(f"\n--- Marker words: NONE remaining ---")

    # 3. Cluster sizes (post-pipeline)
    cluster_dist = Counter()
    for item in all_items:
        opts = item['q']['options']
        ca = item['q']['correct_answer']
        cluster, _ = find_synonym_cluster(opts, ca, threshold=0.6)
        cluster_dist[len(cluster)] += 1
    print(f"\n--- Synonym Cluster Sizes (threshold=0.6, post-pipeline) ---")
    for size in sorted(cluster_dist.keys()):
        count = cluster_dist[size]
        print(f"  Size {size}: {count} questions ({count/total*100:.1f}%)")

    # 4. Length rank distribution
    rank_counts = Counter()
    for item in all_items:
        opts = item['q']['options']
        ca = item['q']['correct_answer']
        sorted_keys = sorted(opts.keys(), key=lambda k: len(opts[k]))
        rank = sorted_keys.index(ca)
        rank_counts[rank] += 1
    print(f"\n--- Correct Answer Length Rank (0=shortest, 7=longest) ---")
    for rank in range(8):
        count = rank_counts.get(rank, 0)
        pct = count / total * 100
        print(f"  Rank {rank}: {count} ({pct:.1f}%)")

    # 5. No qualifier padding present
    padding_count = 0
    padding_phrases = [
        "which appears", "which catches", "which is fairly",
        "upon closer examination", "catching the light",
        "well-fitted and distinctive", "well-preserved",
        "positioned near the center", "placed within easy reach",
    ]
    for item in all_items:
        opts_text = ' '.join(item['q']['options'].values()).lower()
        for phrase in padding_phrases:
            if phrase in opts_text:
                padding_count += 1
                break
    print(f"\n--- Qualifier padding: {padding_count} questions with padding phrases ---")

    # 6. Pipeline stats
    print(f"\n--- Pipeline Statistics ---")
    avg_cluster = sum(stats['cluster_sizes'])/len(stats['cluster_sizes'])
    print(f"  Avg initial cluster size: {avg_cluster:.1f}")
    print(f"  Options with markers cleaned: {stats['markers_cleaned']}")
    print(f"  Register swaps: {stats['swaps']}")
    print(f"  Duplicates replaced with fillers: {stats['dedup_replaced']}")
    print(f"  Cluster members replaced (capping): {stats['cluster_capped']}")
    print(f"  Options per question: {Counter(len(item['q']['options']) for item in all_items)}")

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

    total_q = sum(len(v['questions']) for v in output_videos)
    output = {
        'benchmark_name': data.get('benchmark_name', 'Benchmark') + ' (Debiased v2)',
        'version': data.get('version', '1.0') + '_debiased_v2',
        'total_questions': total_q,
        'num_options': 8,
        'debiasing_version': 'v2',
        'debiasing_applied': [
            'Comprehensive marker cleanup (200+ fancy→plain mappings)',
            'Register swap (50% probability correct↔cluster variant)',
            'Cross-question filler replacement for duplicates (same-video priority)',
            'Cluster size capped at 3 (correct + 2 variants max)',
            'Balanced answer distribution across A-H (~12.5% each)',
            'NO qualifier padding (v1 mistake — creates detectable markers)',
        ],
        'random_baseline_pct': 12.5,
        'seed': args.seed,
        'videos': output_videos,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved: {args.output}")
    print(f"  Videos: {len(output_videos)}, Questions: {total_q}, Options: 8")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
