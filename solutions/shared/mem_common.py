#!/usr/bin/env python
"""
mem_common.py — shared infrastructure for the P3 memory storage-vs-retrieval analysis
(PersistQA). Shared helpers for evidence windows, frame grids and prompt construction.

Provides:
  - load_evidence()        localized evidence windows (dict "video_id|question_id" -> rec)
  - load_benchmark()       benchmark questions, dict video_id -> {question_id -> question}
  - uniform_grid()         np.linspace integer frame grid
  - evidence_mask()        per-grid-frame bool "inside evidence window" + sampled_coverage
  - index builders         idx_uniform / idx_evidence_last / idx_evidence_only /
                           idx_random_window / idx_guaranteed
  - hint_text() / wrong_hint_text()   "The relevant moment occurs between MM:SS and MM:SS."
  - survival metrics       best_match_cos / survival_gap / rank_auc
  - probe_cv()             LogisticRegression 8-way letter probe, GroupKFold(5) by video_id
  - __main__ --make_subsets  builds mem_subset_core.json (~800 keys, 150 videos stratified
                           by length tercile) and mem_subset_ctrl.json (~400 random of core)

Run (build the subsets; CPU only):
    python solutions/shared/mem_common.py --make_subsets


Outputs are written next to this file (analysis3/mem/). Deterministic under --seed.
"""

import argparse
import json
import math
import os
import re
import sys
import zlib

import numpy as np

import os as _os
from pathlib import Path as _Path
ROOT = _os.environ.get("PERSISTQA_ROOT") or str(_Path(__file__).resolve().parents[2])
A3 = os.path.join(ROOT, "analysis3")
MEM_DIR = os.path.join(A3, "mem")
BENCH_PATH = os.path.join(ROOT, "combined_all_hard_v3_real_ids.json")
EVIDENCE_JSON = os.path.join(A3, "evidence_windows.json")
EVIDENCE_JSONL = os.path.join(A3, "evidence_windows.jsonl")
VIDEO_DIR = "/home/c3-0/datasets/moviechat1k-test"
RESULTS_MEM = os.path.join(ROOT, "results_mem")
SUBSET_CORE_PATH = os.path.join(MEM_DIR, "mem_subset_core.json")
SUBSET_CTRL_PATH = os.path.join(MEM_DIR, "mem_subset_ctrl.json")

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_evidence(path=None, max_span_s=60.0):
    """Load evidence windows; return {key: rec} for LOCALIZED records only.

    A record is localized iff t0 is not null. If max_span_s is not None,
    additionally drop records whose window span (t1 - t0) exceeds it
    (P3_DESIGN: filter windows <= 60s). Pass max_span_s=None to disable.

    Accepts either the finalized dict JSON (evidence_windows.json) or the
    streaming JSONL (evidence_windows.jsonl, one {"key": ..., ...} per line —
    the file may still be being written; partial trailing lines are skipped).
    If path is None, prefers the .json and falls back to the .jsonl.
    """
    if path is None:
        path = EVIDENCE_JSON if os.path.exists(EVIDENCE_JSON) else EVIDENCE_JSONL
    recs = {}
    if path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial trailing line while file is being written
                recs[r["key"]] = r
    else:
        with open(path) as f:
            raw = json.load(f)
        for k, r in raw.items():
            r = dict(r)
            r.setdefault("key", k)
            recs[k] = r
    out = {}
    for k, r in recs.items():
        if k.endswith("|None"):
            continue  # upstream artifact of a question missing question_id
        if r.get("t0") is None or r.get("t1") is None:
            continue
        if max_span_s is not None and (r["t1"] - r["t0"]) > max_span_s:
            continue
        out[k] = r
    return out


def load_benchmark(path=BENCH_PATH):
    """Return dict video_id -> {question_id -> question dict}.

    Quirks handled: (a) one question (-fRY1b6WAx4 q9) stores its id under
    "push_id" instead of "question_id" — fall back to it; (b) 9 video_ids
    appear in 2-3 separate entries with different question sets — MERGE them,
    first-seen wins on (video_id, question_id) collisions, matching the
    done-set semantics of gen_evidence_windows.py so evidence keys join to
    the same question the window was generated for.
    """
    with open(path) as f:
        bench = json.load(f)
    by_vid = {}
    for v in bench["videos"]:
        qs = by_vid.setdefault(v["video_id"], {})
        for q in v["questions"]:
            qid = q.get("question_id") or q.get("push_id")
            if qid is None or qid in qs:
                continue
            qs[qid] = q
    return by_vid


def video_path(video_id):
    return os.path.join(VIDEO_DIR, f"{video_id}.mp4")


# ---------------------------------------------------------------------------
# Frame grids and evidence masks
# ---------------------------------------------------------------------------

def uniform_grid(n_total_frames, n):
    """n uniformly spaced integer frame indices over [0, n_total_frames-1]."""
    n = min(int(n), int(n_total_frames))
    return np.linspace(0, n_total_frames - 1, n).astype(int).tolist()


def _window_bounds(rec):
    """(lo, hi) frame bounds of the evidence window, padded by half-width
    half = max(1, N // (2 * n_dense))  (half the dense-sampling stride)."""
    N = int(rec["video_nframes"])
    half = max(1, N // (2 * int(rec["n_dense"])))
    vf = rec["video_frames"]
    lo = max(0, min(vf) - half)
    hi = min(N - 1, max(vf) + half)
    return lo, hi


def evidence_mask(grid, rec):
    """Bool mask per grid frame: inside the padded evidence window.

    Returns (mask: np.ndarray[bool] len(grid), sampled_coverage: bool).
    sampled_coverage is False when the grid misses the window entirely
    (the SAMPLING-bound bucket in the decision matrix).
    """
    lo, hi = _window_bounds(rec)
    g = np.asarray(grid, dtype=int)
    mask = (g >= lo) & (g <= hi)
    return mask, bool(mask.any())


# ---------------------------------------------------------------------------
# Index builders (all take total frame count N, evidence rec, budget n)
# ---------------------------------------------------------------------------

def idx_uniform(N, rec, n):
    """Plain uniform grid (rec unused; kept for a uniform signature). Sorted."""
    return uniform_grid(N, n)


def idx_evidence_last(N, rec, n):
    """Uniform grid REORDERED so in-window frames come LAST.

    Same frame SET as idx_uniform; order = out-of-window frames in temporal
    order, then in-window frames in temporal order. Tests recency/retrieval:
    content identical to uniform, only the position in the sequence changes.
    Returns an ORDERED (not sorted) int list.
    """
    grid = uniform_grid(N, n)
    mask, _ = evidence_mask(grid, rec)
    out_frames = [f for f, m in zip(grid, mask) if not m]
    in_frames = [f for f, m in zip(grid, mask) if m]
    return out_frames + in_frames


def idx_evidence_only(N, rec, n):
    """n frames linspaced INSIDE the (padded) evidence window. Sorted."""
    lo, hi = _window_bounds(rec)
    return np.linspace(lo, hi, int(n)).astype(int).tolist()


def _random_nonoverlapping_window(N_units, lo, hi, seed):
    """Random same-length window [s, s+span] within [0, N_units-1] that does
    not overlap [lo, hi]. Units are arbitrary (frames or seconds).
    Returns (s, s+span) or None if no non-overlapping placement exists."""
    span = hi - lo
    rng = np.random.RandomState(seed)
    ranges = []  # valid start ranges (inclusive)
    left_max = lo - span - 1
    if left_max >= 0:
        ranges.append((0, left_max))
    right_min = hi + 1
    right_max = N_units - 1 - span
    if right_max >= right_min:
        ranges.append((right_min, right_max))
    if not ranges:
        return None
    sizes = np.array([b - a + 1 for a, b in ranges], dtype=float)
    which = rng.choice(len(ranges), p=sizes / sizes.sum())
    a, b = ranges[which]
    s = int(rng.randint(a, b + 1))
    return s, s + span


def idx_random_window(N, rec, n, seed=0):
    """Control for idx_evidence_only: n frames linspaced inside a RANDOM
    same-duration window that does NOT overlap the evidence window.
    Sorted int list, or None if the video is too short to place one."""
    lo, hi = _window_bounds(rec)
    win = _random_nonoverlapping_window(N, lo, hi, seed)
    if win is None:
        return None
    s, e = win
    return np.linspace(s, e, int(n)).astype(int).tolist()


def idx_guaranteed(N, rec, n):
    """Uniform grid, but if no grid frame falls in the evidence window, swap
    the grid frame nearest to the window center for the window center itself.
    Quantifies the sampling-bound confound (>=1 in-window frame guaranteed).
    Sorted int list."""
    grid = uniform_grid(N, n)
    mask, covered = evidence_mask(grid, rec)
    if covered:
        return grid
    lo, hi = _window_bounds(rec)
    center = int((lo + hi) // 2)
    g = np.asarray(grid, dtype=int)
    nearest = int(np.argmin(np.abs(g - center)))
    g[nearest] = center
    return sorted(set(g.tolist()))


# ---------------------------------------------------------------------------
# Hints
# ---------------------------------------------------------------------------

def mmss(seconds):
    seconds = max(0, int(round(float(seconds))))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def hint_text(rec):
    """Correct temporal hint sentence for the evidence window."""
    return (f"The relevant moment occurs between {mmss(rec['t0'])} "
            f"and {mmss(rec['t1'])}.")


def wrong_hint_text(rec, seed=0):
    """Same-format hint pointing at a random same-duration window that does
    NOT overlap the true one. None if the video is too short to place one."""
    dur = float(rec["video_nframes"]) / float(rec["fps"])
    # work in whole seconds so the MM:SS strings are exact
    lo = int(math.floor(rec["t0"]))
    hi = int(math.ceil(rec["t1"]))
    win = _random_nonoverlapping_window(int(dur), lo, hi, seed)
    if win is None:
        return None
    s, e = win
    return (f"The relevant moment occurs between {mmss(s)} "
            f"and {mmss(e)}.")


# ---------------------------------------------------------------------------
# Summarize-then-answer ablation (shared prompt builders + tail parser)
# ---------------------------------------------------------------------------

def _options_block(options):
    """Options dict -> 'A. ...\nB. ...' block (sorted letters)."""
    keys = sorted(options.keys())
    return "\n".join(f"{k}. {options[k]}" for k in keys)


def summarize_prompt(question, options):
    """Summarize-then-answer prompt (uniform across models). The model first
    writes a short summary, then the MCQ answer as 'Answer: <letter>'."""
    return ("First, briefly summarize what happens in this video in 2-4 "
            "sentences, focusing on the people and what they do. "
            "Then answer this question:\n"
            f"{question}\n{_options_block(options)}\n"
            "End your reply with 'Answer: <letter>' on the last line.")


def preamble_prompt(question, options):
    """Neutral non-summary control: same structure, but the preamble asks for
    visual style/setting instead of a people-focused summary."""
    return ("First, briefly describe the visual style and setting of this "
            "video in 2-3 sentences. Then answer this question:\n"
            f"{question}\n{_options_block(options)}\n"
            "End your reply with 'Answer: <letter>' on the last line.")


def parse_tail_letter(text, n_options=8, options=None):
    """Parse the FINAL answer letter from the TAIL of a free-form response.

    Primary: LAST occurrence of 'Answer: X' (tolerates 'answer is', bold
    markers, parentheses, trailing punctuation, lowercase letter).
    Fallback 1: last option-echo pattern 'C. some text' / '(C) ...' anywhere
    (models that skip the summary often reply exactly like this — possibly
    mid-line after echoing the question — and the option text may itself
    contain the article 'A', so plain standalone matching mis-fires).
    Fallback 2: last standalone UPPERCASE letter in range (case-sensitive so
    the article 'a' in prose does not match).
    Fallback 3 (only if an options dict is passed): if the FULL text of
    exactly ONE option (normalized, len>=4) appears verbatim in the response,
    return its letter — catches prose answers that quote the option without
    any letter. Deterministic; no fuzzy matching. 'INVALID' otherwise.
    """
    if not text:
        return "INVALID"
    t = str(text).strip()
    max_letter = chr(ord("A") + int(n_options) - 1)
    ms = re.findall(
        rf"answer\s*(?:is)?\s*[:\-]?\s*\**\(?([A-{max_letter}])(?![A-Za-z])",
        t, re.IGNORECASE)
    if ms:
        return ms[-1].upper()
    ms = re.findall(rf"(?:^|[\s(])([A-{max_letter}])[.):]\s", t + " ")
    if ms:
        return ms[-1]
    ms = re.findall(rf"\b([A-{max_letter}])\b", t)
    if ms:
        return ms[-1]
    if options:
        low = re.sub(r"\s+", " ", t.lower())
        hits = []
        for k in sorted(options):
            ot = re.sub(r"\s+", " ", str(options[k]).strip().lower()).rstrip(".")
            if len(ot) >= 4 and ot in low:
                hits.append(k)
        if len(hits) == 1:
            return hits[0]
        # Fallback 4: word-overlap of each option against the response TAIL
        # (last 2 sentences). Catches prose answers that paraphrase a short
        # option ("...the seats in the stadium is red." -> option "Red") which
        # the verbatim rule above misses (len>=4 excludes short options).
        # Deterministic: unique argmax with coverage >= 0.6 and a clear margin.
        STOP = {"a", "an", "the", "of", "in", "on", "at", "is", "it", "and", "or", "to", "his", "her"}
        sents = re.split(r"[.!?]\s+", t.strip())
        tail = " ".join(sents[-2:]).lower()
        tail_words = set(re.findall(r"[a-z0-9']+", tail)) - STOP
        scored = []
        for k in sorted(options):
            ow = set(re.findall(r"[a-z0-9']+", str(options[k]).lower())) - STOP
            if not ow:
                continue
            scored.append((len(ow & tail_words) / len(ow), k))
        scored.sort(reverse=True)
        if scored and scored[0][0] >= 0.6 and (len(scored) == 1 or scored[0][0] > scored[1][0] + 1e-9):
            return scored[0][1]
    return "INVALID"


# ---------------------------------------------------------------------------
# Survival metrics
# ---------------------------------------------------------------------------

def best_match_cos(pre, post):
    """Per-pre-frame survival: max cosine similarity to any post row.

    pre: (T, C) pre-compression frame features; post: (S, C) post-compression.
    Returns np.ndarray (T,). If post is empty, returns zeros.
    """
    pre = np.asarray(pre, dtype=np.float32)
    post = np.asarray(post, dtype=np.float32)
    if post.ndim != 2 or post.shape[0] == 0:
        return np.zeros(pre.shape[0], dtype=np.float32)
    pre_n = pre / (np.linalg.norm(pre, axis=1, keepdims=True) + 1e-8)
    post_n = post / (np.linalg.norm(post, axis=1, keepdims=True) + 1e-8)
    return (pre_n @ post_n.T).max(axis=1)


def survival_gap(survival, mask):
    """mean(survival in-window) - mean(survival out-of-window).
    NaN if either side is empty."""
    survival = np.asarray(survival, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0 or (~mask).sum() == 0:
        return float("nan")
    return float(survival[mask].mean() - survival[~mask].mean())


def rank_auc(survival, mask):
    """Mann-Whitney AUC: P(random in-window frame outranks random out-of-window
    frame) under the survival score, with tie correction. NaN if degenerate."""
    survival = np.asarray(survival, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    n_pos = int(mask.sum())
    n_neg = int((~mask).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Tie-averaged ranks, numpy-only (scipy is not installed in the model envs
    # -- videoxl/longvu/ma-lmm/videochat-flash -- so no scipy.stats.rankdata).
    # Equivalent to scipy.stats.rankdata(survival, method="average").
    order = np.argsort(survival, kind="mergesort")
    sorted_s = survival[order]
    inv = np.empty(len(order), dtype=np.intp)
    inv[order] = np.arange(len(order))
    obs = np.r_[True, sorted_s[1:] != sorted_s[:-1]]  # start of each tie group
    dense = obs.cumsum()[inv]                          # 1-based tie-group id per element
    counts = np.r_[np.nonzero(obs)[0], len(sorted_s)]  # cumulative counts at group edges
    ranks = 0.5 * (counts[dense] + counts[dense - 1] + 1)
    auc = (ranks[mask].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# ---------------------------------------------------------------------------
# Probe harness
# ---------------------------------------------------------------------------

def probe_cv(features, keys, gold_letters, option_texts=None, seed=0, C=1.0):
    """Linear decodability probe: multinomial LogisticRegression over the 8
    answer-letter classes, GroupKFold(5) grouped by video_id (extracted from
    keys "video_id|question_id") so no video leaks across folds.

    Run identically on pre- vs post-compression features; the top-1 drop is
    the decodability loss (P3_DESIGN threshold: >10pp).

    features: (N, D) array. keys: N keys. gold_letters: N letters in A..H.
    option_texts is accepted for signature compatibility but unused — the
    logistic path IS the primary protocol (no sentence-transformers needed).

    Returns {"fold_acc": [...], "mean_acc": float, "n": int, "n_folds": int,
             "class_counts": {letter: count}}.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray([str(g).strip().upper() for g in gold_letters])
    groups = np.asarray([k.split("|")[0] for k in keys])
    n_folds = min(5, len(np.unique(groups)))
    if n_folds < 2:
        raise ValueError("probe_cv needs >=2 distinct video_id groups")
    gkf = GroupKFold(n_splits=n_folds)
    fold_acc = []
    for tr, te in gkf.split(X, y, groups):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=C, random_state=seed),
        )
        clf.fit(X[tr], y[tr])
        fold_acc.append(float((clf.predict(X[te]) == y[te]).mean()))
    letters, counts = np.unique(y, return_counts=True)
    return {
        "fold_acc": fold_acc,
        "mean_acc": float(np.mean(fold_acc)),
        "n": int(len(y)),
        "n_folds": n_folds,
        "class_counts": {l: int(c) for l, c in zip(letters, counts)},
    }


# ---------------------------------------------------------------------------
# Prediction jsonl helpers (shared schema, resumable)
# ---------------------------------------------------------------------------

def load_done_keys(pred_path):
    """Keys ("video_id|question_id") already present in a predictions jsonl."""
    done = set()
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add(f"{r['video_id']}|{r['question_id']}")
    return done


def append_prediction(pred_path, model_name, video_id, question_id,
                      predicted, correct, extra=None):
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    rec = {
        "model_name": model_name,
        "video_id": video_id,
        "question_id": question_id,
        "predicted": predicted,
        "correct": correct,
        "is_correct": bool(predicted == correct),
    }
    if extra:
        rec.update(extra)
    with open(pred_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
    return rec


def load_subset(path=SUBSET_CORE_PATH):
    """List of subset keys, or None if the subset file does not exist yet
    (callers then fall back to --limit_videos style capping)."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Subset construction
# ---------------------------------------------------------------------------

def make_subsets(seed=0, n_videos=150, target_core=800, n_ctrl=400,
                 max_span_s=60.0, evidence_path=None, force=False):
    """Build mem_subset_core.json / mem_subset_ctrl.json.

    Core: all localized-evidence questions of n_videos videos, stratified by
    video-length terciles (durations from video_nframes/fps in the evidence
    records). If that exceeds target_core, apply the smallest deterministic
    per-video cap that brings the total to <= target_core (per-video keys are
    shuffled under the seed before capping, then the final list is sorted).
    Ctrl: n_ctrl random keys of core. Deterministic under seed.
    """
    if (os.path.exists(SUBSET_CORE_PATH) and os.path.exists(SUBSET_CTRL_PATH)
            and not force):
        core = load_subset(SUBSET_CORE_PATH)
        ctrl = load_subset(SUBSET_CTRL_PATH)
        print(f"[make_subsets] already exist (use --force to rebuild): "
              f"core={len(core)} ctrl={len(ctrl)}")
        return core, ctrl

    recs = load_evidence(evidence_path, max_span_s=max_span_s)
    print(f"[make_subsets] localized evidence records "
          f"(span<={max_span_s}s): {len(recs)}")

    by_vid = {}
    dur = {}
    for k, r in sorted(recs.items()):
        vid = k.split("|")[0]
        by_vid.setdefault(vid, []).append(k)
        dur[vid] = float(r["video_nframes"]) / float(r["fps"])
    vids = sorted(by_vid)
    print(f"[make_subsets] videos with localized evidence: {len(vids)}")

    rng = np.random.RandomState(seed)

    # stratify by video-length terciles
    durations = np.array([dur[v] for v in vids])
    t1, t2 = np.percentile(durations, [100.0 / 3, 200.0 / 3])
    terciles = {0: [], 1: [], 2: []}
    for v in vids:
        b = 0 if dur[v] <= t1 else (1 if dur[v] <= t2 else 2)
        terciles[b].append(v)
    print(f"[make_subsets] tercile cuts: {t1:.1f}s / {t2:.1f}s; sizes: "
          f"{[len(terciles[b]) for b in range(3)]}")

    per_bucket = n_videos // 3
    chosen = []
    for b in range(3):
        pool = terciles[b]
        take = min(per_bucket, len(pool))
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    # top up from the remainder if any bucket was short
    if len(chosen) < n_videos:
        rest = [v for v in vids if v not in set(chosen)]
        extra = min(n_videos - len(chosen), len(rest))
        chosen.extend(rng.choice(rest, size=extra, replace=False).tolist())
    chosen = sorted(chosen)
    print(f"[make_subsets] chosen videos: {len(chosen)}")

    # per-video keys, shuffled deterministically, then the smallest cap that
    # brings the total to <= target_core
    shuffled = {}
    for v in chosen:
        ks = sorted(by_vid[v])
        rng2 = np.random.RandomState(seed + (zlib.crc32(v.encode()) % 100000))
        rng2.shuffle(ks)
        shuffled[v] = ks
    total_all = sum(len(shuffled[v]) for v in chosen)
    if total_all <= target_core:
        cap = max(len(shuffled[v]) for v in chosen)
    else:
        cap = 1
        while sum(min(len(shuffled[v]), cap + 1) for v in chosen) <= target_core:
            cap += 1
    core = sorted(k for v in chosen for k in shuffled[v][:cap])
    print(f"[make_subsets] total available {total_all}, per-video cap {cap} "
          f"-> core {len(core)}")

    n_ctrl = min(n_ctrl, len(core))
    ctrl = sorted(rng.choice(core, size=n_ctrl, replace=False).tolist())

    os.makedirs(MEM_DIR, exist_ok=True)
    with open(SUBSET_CORE_PATH, "w") as f:
        json.dump(core, f, indent=1)
    with open(SUBSET_CTRL_PATH, "w") as f:
        json.dump(ctrl, f, indent=1)
    print(f"[make_subsets] wrote {SUBSET_CORE_PATH} ({len(core)} keys)")
    print(f"[make_subsets] wrote {SUBSET_CTRL_PATH} ({len(ctrl)} keys)")
    return core, ctrl


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--make_subsets", action="store_true",
                    help="build mem_subset_core.json / mem_subset_ctrl.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_videos", type=int, default=150)
    ap.add_argument("--target_core", type=int, default=800)
    ap.add_argument("--n_ctrl", type=int, default=400)
    ap.add_argument("--max_span_s", type=float, default=60.0,
                    help="drop evidence windows longer than this (<=0: keep all)")
    ap.add_argument("--evidence", type=str, default=None,
                    help="evidence windows path (.json or .jsonl); default: "
                         "evidence_windows.json, falling back to the .jsonl")
    ap.add_argument("--limit", type=int, default=0,
                    help="unused here; accepted for interface uniformity")
    ap.add_argument("--force", action="store_true",
                    help="rebuild subsets even if the files exist")
    args = ap.parse_args()

    if args.make_subsets:
        max_span = args.max_span_s if args.max_span_s > 0 else None
        make_subsets(seed=args.seed, n_videos=args.n_videos,
                     target_core=args.target_core, n_ctrl=args.n_ctrl,
                     max_span_s=max_span, evidence_path=args.evidence,
                     force=args.force)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
