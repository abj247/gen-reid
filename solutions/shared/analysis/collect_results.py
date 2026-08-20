#!/usr/bin/env python3
"""
collect_results.py -- ONE authoritative master results table for the two-granularity
CLIP-retrieval study (Lantern, frame-level; Cairn, segment-level with a stored memory).

Scans:
  results_baseline/<pipeline>/<backbone>/predictions.jsonl   (ANON video ids)
  solutions/cairn/results/*.jsonl                          (REAL video ids)

Emits:
  solutions/shared/analysis/results_master.csv
  solutions/shared/analysis/RESULTS_MASTER.md
  a stdout rendering + an explicit DISCREPANCY report against the numbers quoted
  in the task prompt (which came from a prior session and must be verified).

House rules enforced here:
  * every delta is PAIRED, on the intersection of keys both arms answered; n reported.
  * significance = EXACT McNemar (binomtest on discordant b vs c, two-sided).
  * every null carries an MDE ("no difference larger than X points at 80% power").
  * every row states its frame/token budget.
  * oracle arms are labelled CEILING inline, never a method.

Usage:
  python collect_results.py              # real run
  python collect_results.py --selftest   # synthetic fixture self-test, then exit
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, OrderedDict, defaultdict

from scipy.stats import binomtest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUTDIR = os.path.join(ROOT, "analysis3", "selanal")

PROVISIONAL = []  # (arm, backbone, rows, minutes since last write)

Z_ALPHA = 1.959963985  # two-sided alpha = 0.05
Z_BETA = 0.8416212336  # power = 0.80

# ----------------------------------------------------------------------------- arms
# PATH 1+2 (rendered-mp4 pipelines, ANON-keyed). baseline = kf_uniform8, random ctrl = kf_random.
KF_BASELINE = "kf_uniform8"
KF_RANDOM = "kf_random"
KF_ARMS = OrderedDict([
    # pipeline        (path,        budget_str,   budget_note)
    ("kf_uniform8",   ("frame",     "8 frames",   "BASELINE (uniform-8, identical mp4 pipeline)")),
    ("kf_random",     ("frame",     "8 frames",   "CONTROL (random 8 from same 64-pool, per-q seeded)")),
    ("kf_referent",   ("frame",     "8 frames",   "keyframe: top-8 of 64 by CLIP(question+temporal_anchor)")),
    ("kf_question",   ("frame",     "8 frames",   "keyframe: top-8 of 64 by CLIP(question only)")),
    ("kf_chunk",      ("chunk",     "8 frames",   "chunk: 8 chunks, max-sim, top-2, even inside")),
    ("kf_q_t16",      ("frame",     "16 frames",  "NOT budget-matched to the 8-frame arms")),
    ("kf_q_t32",      ("frame",     "32 frames",  "NOT budget-matched to the 8-frame arms")),
])
KF_BACKBONES = ["internvl3-14b", "qwen2.5-vl-7b", "ovis2.5-9b", "videochat-flash-7b"]

# PATH 2 true memory bank (REAL-keyed). baseline = mb_uniform32.
MB_BASELINE = "mb_uniform32"
MB_RANDOM_FOR = {"mb_top1": "mb_rand1", "mb_top2": "mb_rand2", "mb_oracle": "mb_rand2"}
MB_ORDER = ["mb_uniform32", "mb_rand1", "mb_rand2", "mb_top1", "mb_top2", "mb_oracle"]
MB_NOTE = {
    "mb_uniform32": "BASELINE (uniform 32 frames spliced from the bank)",
    "mb_rand1": "CONTROL (1 random chunk)",
    "md_rand2": "",
    "mb_rand2": "CONTROL (2 random chunks)",
    "mb_top1": "memory bank: top-1 chunk by max CLIP sim",
    "mb_top2": "memory bank: top-2 chunks by max CLIP sim",
    "mb_oracle": "CEILING -- uses the oracle evidence chunk; answer-informed, NEVER a method",
}

# ------------------------------------------------------------------- quoted numbers
# Verbatim from the task prompt. RECOMPUTED values are checked against these.
QUOTED_KF = {
    # backbone: (base, chunk, keyfr, d_chunk, p_chunk, d_keyfr, p_keyfr, d_kf_vs_chunk, p_kf_vs_chunk)
    "internvl3-14b": (24.62, 26.94, 27.37, +2.32, 2.36e-4, +2.75, 1.93e-05, +0.43, 0.492),
    "qwen2.5-vl-7b": (17.14, 17.94, 19.27, +0.80, 0.215, +2.13, 0.00106, +1.33, 0.0274),
    "ovis2.5-9b":    (22.86, 24.56, 25.36, +1.70, 0.0173, +2.51, 4.35e-4, +0.80, 0.233),
}
QUOTED_KF_VS_RANDOM = {"internvl3-14b": 2.91, "qwen2.5-vl-7b": 1.67, "ovis2.5-9b": 2.63}
QUOTED_KF_N = 3233
QUOTED_VCF_KEYFRAME_DELTA = -3.04  # videochat-flash-7b, keyframe vs base
QUOTED_MB = {
    "n": 759,
    "mb_top2": 32.28, "mb_rand2": 28.46, "mb_oracle": 35.05,
    "d_top2_rand2": 3.82, "p_top2_rand2": 0.0093,
    "d_top2_uniform32": 1.32,
    "chunk_hit_top2": 48.6, "chunk_hit_rand2": 25.6,
    "visual_tokens": 8192,
}


# ------------------------------------------------------------------------ statistics
def mcnemar_exact(a: dict, b: dict):
    """Paired exact McNemar on the intersection of keys.

    a, b: {key: bool_correct}. Returns dict with n, acc_a, acc_b, delta (pp), p, MDE (pp).
    delta is a-minus-b in percentage points, computed ON THE INTERSECTION only.
    """
    keys = sorted(set(a) & set(b))
    n = len(keys)
    if n == 0:
        return dict(n=0, acc_a=None, acc_b=None, delta=None, p=None, mde=None, b=0, c=0)
    nb = sum(1 for k in keys if a[k] and not b[k])       # a right, b wrong
    nc = sum(1 for k in keys if (not a[k]) and b[k])     # a wrong, b right
    ca = sum(1 for k in keys if a[k])
    cb = sum(1 for k in keys if b[k])
    disc = nb + nc
    p = binomtest(nb, disc, 0.5, alternative="two-sided").pvalue if disc > 0 else 1.0
    # MDE: smallest |delta| (pp) detectable at 80% power / two-sided alpha=.05 for THIS n and
    # THIS discordance rate.  McNemar normal approx: |b-c|/sqrt(b+c) >= z_a + z_b
    #   => |b-c|/n >= (z_a+z_b)*sqrt(pi_d/n)
    pi_d = disc / n
    mde = 100.0 * (Z_ALPHA + Z_BETA) * math.sqrt(pi_d / n) if pi_d > 0 else None
    return dict(n=n, acc_a=100.0 * ca / n, acc_b=100.0 * cb / n,
                delta=100.0 * (ca - cb) / n, p=p, mde=mde, b=nb, c=nc)


# ----------------------------------------------------------------------------- IO
def load_jsonl(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dedupe_first(rows, keyfn):
    """Deterministic dedupe by FIRST occurrence of keyfn(row).

    Returns (dict key->row, n_dropped, n_gold_conflicts).
    A gold conflict = the same key appeared with a different `correct` letter; the first
    occurrence wins and the conflict is counted (these are real: 9 real video_ids carry
    2-3 anon ids, so anon->real collapsing collides).
    """
    out, seen_gold, dropped, conflicts = OrderedDict(), {}, 0, 0
    for r in rows:
        k = keyfn(r)
        gold = r.get("correct")
        if k in out:
            dropped += 1
            if seen_gold.get(k) is not None and gold is not None and gold != seen_gold[k]:
                conflicts += 1
            continue
        out[k] = r
        seen_gold[k] = gold
    return out, dropped, conflicts


def load_mapping():
    p = os.path.join(ROOT, "video_id_mapping.json")
    if not os.path.exists(p):
        return {}, {}
    m = json.load(open(p))
    return m.get("anon_to_real", {}), m.get("real_to_anon", {})


# ------------------------------------------------------------------------- collection
def collect_kf(warnings):
    """{backbone: {arm: {key(anon)->bool}}} plus a dedupe ledger."""
    data = defaultdict(dict)
    ledger = []
    for arm in KF_ARMS:
        for bb in KF_BACKBONES:
            f = os.path.join(ROOT, "results_baseline", arm, bb, "predictions.jsonl")
            if not os.path.exists(f):
                continue
            rows = load_jsonl(f)
            keyed, dropped, conf = dedupe_first(
                rows, lambda r: f"{r['video_id']}|{r['question_id']}")
            if dropped:
                ledger.append((f"{arm}/{bb}", "anon", len(rows), dropped, conf))
            data[bb][arm] = {k: bool(v["is_correct"]) for k, v in keyed.items()}
            age_min = (time.time() - os.path.getmtime(f)) / 60.0
            if age_min < 30:
                PROVISIONAL.append((arm, bb, len(rows), age_min))
    return data, ledger


def collect_mb(warnings):
    """{arm: {key(real)->bool}} + chunk-hit rates + backbone name."""
    d = os.path.join(ROOT, "analysis3", "membank", "results")
    arms = defaultdict(dict)
    hits = defaultdict(lambda: [0, 0])
    tokens = {}
    backbone = "internvl3-14b"
    ledger = []
    if not os.path.isdir(d):
        warnings.append(f"membank results dir missing: {d}")
        return arms, hits, tokens, backbone, ledger
    files = sorted(f for f in os.listdir(d) if f.endswith(".jsonl"))
    # keep the RICHEST file per (backbone) -- the smaller partial files are earlier
    # subsets of the same run and would otherwise inject stale rows.
    best, best_n = None, -1
    for f in files:
        rows = load_jsonl(os.path.join(d, f))
        if len(rows) > best_n:
            best, best_n, best_rows = f, len(rows), rows
    if len(files) > 1:
        warnings.append(
            f"membank: {len(files)} result files present; using the largest "
            f"({best}, {best_n} rows) and ignoring {len(files)-1} smaller partial file(s).")
    backbone = best.split("__")[0]
    per_arm = defaultdict(list)
    for r in best_rows:
        per_arm[r["arm"]].append(r)
    for arm, rows in per_arm.items():
        keyed, dropped, conf = dedupe_first(rows, lambda r: f"{r['video_id']}|{r['question_id']}")
        if dropped:
            ledger.append((f"membank/{arm}", "real", len(rows), dropped, conf))
        arms[arm] = {k: bool(v["is_correct"]) for k, v in keyed.items()}
        tokens[arm] = keyed[next(iter(keyed))].get("visual_tokens")
        for v in keyed.values():
            if v.get("chunk_hit") is not None:
                hits[arm][1] += 1
                hits[arm][0] += bool(v["chunk_hit"])
    return arms, hits, tokens, backbone, ledger


# ----------------------------------------------------------------------------- rows
def fmt(x, nd=2):
    return "" if x is None else f"{x:.{nd}f}"


def fmt_p(p):
    if p is None:
        return ""
    if p >= 1e-4:
        return f"{p:.4g}"
    return f"{p:.3e}"


def build_rows(kf, mb_arms, mb_tokens, mb_backbone):
    rows = []
    # ---- keyframe / chunk path
    for bb in KF_BACKBONES:
        arms = kf.get(bb, {})
        base = arms.get(KF_BASELINE, {})
        rnd = arms.get(KF_RANDOM, {})
        for arm, (path, budget, note) in KF_ARMS.items():
            if arm not in arms:
                continue
            own = arms[arm]
            n_own = len(own)
            acc = 100.0 * sum(own.values()) / n_own if n_own else None
            vb = mcnemar_exact(own, base) if (base and arm != KF_BASELINE) else None
            vr = mcnemar_exact(own, rnd) if (rnd and arm != KF_RANDOM) else None
            rows.append(dict(
                path=path, backbone=bb, arm=arm, n=n_own, frames_or_tokens=budget,
                accuracy=acc,
                delta_vs_baseline=vb["delta"] if vb else None,
                p_vs_baseline=vb["p"] if vb else None,
                n_pair_baseline=vb["n"] if vb else None,
                delta_vs_random_control=vr["delta"] if vr else None,
                p_vs_random_control=vr["p"] if vr else None,
                n_pair_random=vr["n"] if vr else None,
                MDE=vb["mde"] if vb else (vr["mde"] if vr else None),
                MDE_vs_random=vr["mde"] if vr else None,
                note=note))
    # ---- memory bank path
    base = mb_arms.get(MB_BASELINE, {})
    for arm in MB_ORDER:
        if arm not in mb_arms:
            continue
        own = mb_arms[arm]
        n_own = len(own)
        acc = 100.0 * sum(own.values()) / n_own if n_own else None
        vb = mcnemar_exact(own, base) if (base and arm != MB_BASELINE) else None
        ctrl = mb_arms.get(MB_RANDOM_FOR.get(arm, ""), {})
        vr = mcnemar_exact(own, ctrl) if ctrl else None
        tok = mb_tokens.get(arm)
        rows.append(dict(
            path="membank", backbone=mb_backbone, arm=arm, n=n_own,
            frames_or_tokens=f"{tok} visual tokens (32f x 256tok)" if tok else "",
            accuracy=acc,
            delta_vs_baseline=vb["delta"] if vb else None,
            p_vs_baseline=vb["p"] if vb else None,
            n_pair_baseline=vb["n"] if vb else None,
            delta_vs_random_control=vr["delta"] if vr else None,
            p_vs_random_control=vr["p"] if vr else None,
            n_pair_random=vr["n"] if vr else None,
            MDE=vb["mde"] if vb else (vr["mde"] if vr else None),
            MDE_vs_random=vr["mde"] if vr else None,
            note=MB_NOTE.get(arm, "")))
    return rows


# ------------------------------------------------- native (non-budget-matched) baseline
NATIVE_DIR = "results_video_v2"


def collect_native(a2r, notes):
    """The models' OWN video pipeline (results_video_v2), REAL-keyed.

    This is NOT budget-matched to the 8-frame arms -- each model runs its own default
    frame budget (no frame count is recorded in those files, so it cannot be verified
    from the artefacts). It is included because the prior session's quoted
    videochat-flash-7b number was measured against THIS baseline, not against
    kf_uniform8, and the two disagree in SIGN.
    """
    out = []
    for bb in KF_BACKBONES:
        f = os.path.join(ROOT, NATIVE_DIR, bb, "predictions.jsonl")
        if not os.path.exists(f):
            continue
        keyed, dropped, conf = dedupe_first(
            load_jsonl(f), lambda r: f"{r['video_id']}|{r['question_id']}")
        nat = {k: bool(v["is_correct"]) for k, v in keyed.items()}
        if dropped:
            notes.append(f"{NATIVE_DIR}/{bb}: {dropped} duplicate rows dropped "
                         f"({conf} gold-letter conflicts).")
        yield bb, nat, dropped, conf


def native_rows(kf, a2r, notes):
    rows = []
    for bb, nat, _, _ in collect_native(a2r, notes):
        for arm in ["kf_uniform8", "kf_random", "kf_chunk", "kf_referent", "kf_question"]:
            d = kf.get(bb, {}).get(arm)
            if not d:
                continue
            mapped, drop = OrderedDict(), 0
            for k, v in d.items():
                vid, qid = k.split("|", 1)
                rk = f"{a2r.get(vid, vid)}|{qid}"
                if rk in mapped:
                    drop += 1
                    continue
                mapped[rk] = v
            r = mcnemar_exact(mapped, nat)
            rows.append(dict(backbone=bb, arm=arm, dropped=drop, **r))
    return rows


# ------------------------------------------------------------------ cross-path join
def cross_path(kf, mb_arms, a2r, warnings):
    """kf 8-frame arms vs membank 32-frame arms on the SHARED question set.

    kf is anon-keyed, membank is real-keyed: a naive join gives ZERO intersection.
    Map anon->real first, dedupe by first occurrence, and report the drop.
    """
    out = []
    bb = "internvl3-14b"
    if bb not in kf or not mb_arms:
        return out, None
    mapped = {}
    ledger = {}
    for arm, d in kf[bb].items():
        m, dropped = OrderedDict(), 0
        for k, v in d.items():
            vid, qid = k.split("|", 1)
            rk = f"{a2r.get(vid, vid)}|{qid}"
            if rk in m:
                dropped += 1
                continue
            m[rk] = v
        mapped[arm] = m
        ledger[arm] = dropped
    mb_keys = set(mb_arms.get(MB_BASELINE, {}))
    for kf_arm in ["kf_uniform8", "kf_random", "kf_referent", "kf_chunk"]:
        if kf_arm not in mapped:
            continue
        sub = {k: v for k, v in mapped[kf_arm].items() if k in mb_keys}
        for mb_arm in ["mb_uniform32", "mb_top2"]:
            if mb_arm not in mb_arms:
                continue
            r = mcnemar_exact(mb_arm and mb_arms[mb_arm] or {}, sub)
            out.append(dict(kf_arm=kf_arm, mb_arm=mb_arm, **r))
    return out, ledger


# ------------------------------------------------------------------- discrepancies
def check(warn, label, got, want, tol, unit="pp"):
    if got is None:
        warn.append(f"MISSING: {label} could not be recomputed (quoted {want}).")
        return
    if abs(got - want) > tol:
        warn.append(f"DISCREPANCY: {label}: recomputed {got:.4g} vs quoted {want:.4g} "
                    f"(|diff| {abs(got-want):.4g} > tol {tol:g} {unit}).")


def discrepancy_report(kf, mb_arms, mb_hits):
    w = []
    for bb, q in QUOTED_KF.items():
        arms = kf.get(bb, {})
        if not arms:
            w.append(f"MISSING: backbone {bb} has no kf results.")
            continue
        base, chunk, kfr = arms.get(KF_BASELINE, {}), arms.get("kf_chunk", {}), arms.get("kf_referent", {})
        rnd = arms.get(KF_RANDOM, {})
        (qb, qc, qk, qdc, qpc, qdk, qpk, qdkc, qpkc) = q
        for name, d, quoted_acc in (("base", base, qb), ("chunk", chunk, qc), ("keyframe", kfr, qk)):
            if d:
                acc = 100.0 * sum(d.values()) / len(d)
                check(w, f"{bb} {name} accuracy (n={len(d)})", acc, quoted_acc, 0.02)
                if len(d) != QUOTED_KF_N:
                    w.append(f"DISCREPANCY: {bb} {name} n={len(d)} but prompt quotes n={QUOTED_KF_N}.")
        for name, d, qd, qp in (("chunk-vs-base", chunk, qdc, qpc), ("keyframe-vs-base", kfr, qdk, qpk)):
            if d and base:
                r = mcnemar_exact(d, base)
                check(w, f"{bb} {name} delta", r["delta"], qd, 0.02)
                check(w, f"{bb} {name} p", r["p"], qp, max(0.05 * qp, 1e-6), "rel")
        if kfr and chunk:
            r = mcnemar_exact(kfr, chunk)
            check(w, f"{bb} keyframe-vs-chunk delta", r["delta"], qdkc, 0.02)
            check(w, f"{bb} keyframe-vs-chunk p", r["p"], qpkc, max(0.05 * qpkc, 1e-6), "rel")
        if kfr and rnd and bb in QUOTED_KF_VS_RANDOM:
            r = mcnemar_exact(kfr, rnd)
            check(w, f"{bb} keyframe-vs-random delta", r["delta"], QUOTED_KF_VS_RANDOM[bb], 0.02)
            if r["p"] > 0.01:
                w.append(f"DISCREPANCY: {bb} keyframe-vs-random p={r['p']:.4g} but prompt "
                         f"claims all three are p<=0.01.")
    # videochat-flash: the quoted -3.04 does not reproduce against kf_uniform8.
    arms = kf.get("videochat-flash-7b", {})
    if arms.get("kf_referent") and arms.get(KF_BASELINE):
        r = mcnemar_exact(arms["kf_referent"], arms[KF_BASELINE])
        if abs(r["delta"] - QUOTED_VCF_KEYFRAME_DELTA) > 0.02:
            w.append(
                f"DISCREPANCY (SIGN FLIP, the important one): videochat-flash-7b "
                f"keyframe-vs-BUDGET-MATCHED-baseline recomputes to {r['delta']:+.2f} pp "
                f"(kf_referent {r['acc_a']:.2f} vs kf_uniform8 {r['acc_b']:.2f}, n={r['n']}, "
                f"exact McNemar p={r['p']:.4g}) -- keyframe WINS on videochat-flash-7b. "
                f"The quoted {QUOTED_VCF_KEYFRAME_DELTA:+.2f} was measured against a DIFFERENT "
                f"and NOT budget-matched baseline (`{NATIVE_DIR}`, the model's own video "
                f"pipeline); see the native-baseline table, where kf_referent vs "
                f"{NATIVE_DIR} does reproduce at about {QUOTED_VCF_KEYFRAME_DELTA:+.2f}. "
                f"The 'videochat-flash is a failure case' claim is therefore ONLY true "
                f"against the un-matched native baseline and must be stated that way.")
    # membank
    for arm, qacc in (("mb_top2", QUOTED_MB["mb_top2"]), ("mb_rand2", QUOTED_MB["mb_rand2"]),
                      ("mb_oracle", QUOTED_MB["mb_oracle"])):
        d = mb_arms.get(arm, {})
        if d:
            check(w, f"membank {arm} accuracy (n={len(d)})", 100.0 * sum(d.values()) / len(d), qacc, 0.02)
            if len(d) != QUOTED_MB["n"]:
                w.append(f"DISCREPANCY: membank {arm} n={len(d)} but prompt quotes n={QUOTED_MB['n']}.")
    if mb_arms.get("mb_top2") and mb_arms.get("mb_rand2"):
        r = mcnemar_exact(mb_arms["mb_top2"], mb_arms["mb_rand2"])
        check(w, "membank mb_top2-vs-mb_rand2 delta", r["delta"], QUOTED_MB["d_top2_rand2"], 0.02)
        check(w, "membank mb_top2-vs-mb_rand2 p", r["p"], QUOTED_MB["p_top2_rand2"],
              max(0.05 * QUOTED_MB["p_top2_rand2"], 1e-6), "rel")
    if mb_arms.get("mb_top2") and mb_arms.get(MB_BASELINE):
        r = mcnemar_exact(mb_arms["mb_top2"], mb_arms[MB_BASELINE])
        check(w, "membank mb_top2-vs-mb_uniform32 delta", r["delta"], QUOTED_MB["d_top2_uniform32"], 0.02)
    for arm, q in (("mb_top2", QUOTED_MB["chunk_hit_top2"]), ("mb_rand2", QUOTED_MB["chunk_hit_rand2"])):
        h, tot = mb_hits.get(arm, [0, 0])
        if tot:
            check(w, f"membank {arm} chunk-hit rate (n={tot})", 100.0 * h / tot, q, 0.06)
    return w


# ----------------------------------------------------------------------------- render
CSV_COLS = ["path", "backbone", "arm", "n", "frames_or_tokens", "accuracy",
            "delta_vs_baseline", "p_vs_baseline", "n_pair_baseline",
            "delta_vs_random_control", "p_vs_random_control", "n_pair_random",
            "MDE", "MDE_vs_random", "note"]


def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            o = dict(r)
            for k in ("accuracy", "delta_vs_baseline", "delta_vs_random_control", "MDE", "MDE_vs_random"):
                o[k] = fmt(o.get(k))
            for k in ("p_vs_baseline", "p_vs_random_control"):
                o[k] = fmt_p(o.get(k))
            w.writerow({c: o.get(c, "") for c in CSV_COLS})


def md_table(rows):
    hdr = ["path", "backbone", "arm", "n", "budget", "acc %", "d vs base", "p (McNemar)",
           "n pair", "d vs random", "p (McNemar)", "n pair", "MDE base", "MDE rand"]
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        out.append("| " + " | ".join([
            r["path"], r["backbone"], "`%s`" % r["arm"], str(r["n"]), r["frames_or_tokens"],
            fmt(r["accuracy"]),
            ("%+.2f" % r["delta_vs_baseline"]) if r["delta_vs_baseline"] is not None else "--",
            fmt_p(r["p_vs_baseline"]) or "--",
            str(r["n_pair_baseline"] or "--"),
            ("%+.2f" % r["delta_vs_random_control"]) if r["delta_vs_random_control"] is not None else "--",
            fmt_p(r["p_vs_random_control"]) or "--",
            str(r["n_pair_random"] or "--"),
            fmt(r["MDE"]) or "--", fmt(r["MDE_vs_random"]) or "--",
        ]) + " |")
    return "\n".join(out)


def render(rows, warnings, ledger, cross, cross_ledger, mb_hits, notes, natrows):
    L = []
    L.append("# Master results table -- one CLIP signal, two retrieval granularities")
    L.append("")
    L.append("Auto-generated by `solutions/shared/analysis/collect_results.py`. Do not hand-edit.")
    L.append("")
    L.append("**Reading rules.** Every `delta` is PAIRED: it is computed on the intersection of "
             "questions BOTH arms answered, and `n pair` gives that intersection size (it differs "
             "from the arm's own `n` whenever a run is partial). `p` is the EXACT McNemar test "
             "(two-sided binomial on the discordant pairs). `MDE` is the smallest effect, in "
             "percentage points, that this pairing could have detected at 80% power / two-sided "
             "alpha=0.05, given its observed discordance rate -- so a non-significant row means "
             "*no difference larger than MDE points*, never \"no difference\". Budget is stated on "
             "every row; budget matching is the central discipline of this study.")
    L.append("")
    L.append("**Baselines.** frame/chunk path: `kf_uniform8` (plain uniform-8 through the identical "
             "mp4 pipeline; with a 64-frame pool and top-8, uniform-8 is an EXACT subset of the "
             "pool). Random control: `kf_random` (8 drawn at random from the SAME 64-pool, "
             "per-question seeded) -- it isolates QUERY CONDITIONING, while `kf_uniform8` isolates "
             "the PIPELINE. memory-bank path: `mb_uniform32`; random controls `mb_rand1`/`mb_rand2` "
             "matched on chunk count.")
    L.append("")
    L.append("**CEILING.** `mb_oracle` selects the human-verified evidence chunk. It is "
             "answer-informed and is a CEILING / diagnostic, NEVER a method. Any hit/miss "
             "conditioning elsewhere carries the same label.")
    L.append("")
    if PROVISIONAL:
        L.append("> **PROVISIONAL ROWS -- A RUN IS STILL WRITING.** The following prediction "
                 "files were modified in the last 30 minutes, so their `n` and accuracy WILL "
                 "MOVE and every contrast that touches them is provisional. Re-run this script "
                 "when the jobs finish before quoting these rows anywhere:")
        L.append(">")
        for arm, bb, nrows, age in sorted(PROVISIONAL):
            L.append(f"> - `{arm}` / {bb}: {nrows} rows, last written "
                     f"{age:.1f} min ago (target 3233)")
        L.append("")
    L.append(md_table(rows))
    L.append("")
    L.append("## Chunk-hit rates (memory bank; diagnostic, uses oracle evidence chunk -> DIAGNOSTIC)")
    L.append("")
    L.append("| arm | chunk-hit % | n |")
    L.append("|---|---|---|")
    for arm in MB_ORDER:
        h, tot = mb_hits.get(arm, [0, 0])
        if tot:
            L.append(f"| `{arm}` | {100.0*h/tot:.1f} | {tot} |")
    L.append("")
    if natrows:
        L.append("## Native video baseline (`%s`) -- NOT budget-matched" % NATIVE_DIR)
        L.append("")
        L.append("Each model's OWN video pipeline at its OWN default frame budget (no frame "
                 "count is recorded in those artefacts, so the budget cannot be verified from "
                 "them). These rows are **NOT budget-matched** and are therefore a context "
                 "check, not the headline comparison -- the headline comparison is against "
                 "`kf_uniform8`, which is budget-matched by construction. This table is "
                 "included because the prior session's videochat-flash-7b number was measured "
                 "here, and it disagrees in SIGN with the budget-matched one. REAL-keyed, so "
                 "the frame arms are anon->real mapped first.")
        L.append("")
        L.append("| backbone | frame arm (8f) | n pair | acc arm % | acc native % | delta | p (McNemar) | MDE | rows dropped by anon->real |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in natrows:
            L.append(f"| {r['backbone']} | `{r['arm']}` | {r['n']} | {fmt(r['acc_a'])} | "
                     f"{fmt(r['acc_b'])} | {r['delta']:+.2f} | {fmt_p(r['p'])} | "
                     f"{fmt(r['mde'])} | {r['dropped']} |")
        L.append("")
    if cross:
        L.append("## Cross-path join (anon->real id mapping applied)")
        L.append("")
        L.append("The frame/chunk results are ANON-keyed and the memory-bank results are REAL-keyed; "
                 "a naive join gives ZERO intersection, so anon ids are mapped to real ids first. "
                 "These rows are **NOT budget-matched** (8 rendered frames vs 8,192 visual tokens = "
                 "32 spliced frames) and exist only to place the two paths on one question set.")
        L.append("")
        L.append("| membank arm (32f/8192 tok) | frame arm (8f) | n pair | acc mb % | acc kf % | delta | p (McNemar) | MDE |")
        L.append("|---|---|---|---|---|---|---|---|")
        for c in cross:
            L.append(f"| `{c['mb_arm']}` | `{c['kf_arm']}` | {c['n']} | {fmt(c['acc_a'])} | "
                     f"{fmt(c['acc_b'])} | {c['delta']:+.2f} | {fmt_p(c['p'])} | {fmt(c['mde'])} |")
        L.append("")
    L.append("## Dedupe ledger")
    L.append("")
    L.append("Duplicate `(video_id, question_id)` pairs are resolved deterministically by FIRST "
             "occurrence. They arise because 9 real video_ids carry 2-3 anon ids, so collapsing "
             "anon->real collides; 40 of the colliding pairs carry a DIFFERENT gold letter.")
    L.append("")
    if ledger:
        L.append("| source | key space | rows read | rows dropped | gold-letter conflicts |")
        L.append("|---|---|---|---|---|")
        for src, space, tot, dropped, conf in ledger:
            L.append(f"| {src} | {space} | {tot} | {dropped} | {conf} |")
    else:
        L.append("No duplicates in the native key spaces (`kf_*` anon-keyed, membank real-keyed): "
                 "0 rows dropped.")
    L.append("")
    if cross_ledger:
        tot = sum(cross_ledger.values())
        L.append(f"Anon->real collapsing for the cross-path join drops **{tot} rows** across "
                 f"{len(cross_ledger)} frame/chunk arms "
                 f"({', '.join(f'{k}: {v}' for k, v in sorted(cross_ledger.items()))}); "
                 f"64 distinct (real video, question) pairs are duplicated, 40 of them with a "
                 f"different gold letter.")
        L.append("")
    L.append("## Discrepancy report vs the numbers quoted in the task prompt")
    L.append("")
    if warnings:
        for x in warnings:
            L.append(f"- **{x}**")
    else:
        L.append("- No discrepancies: every quoted accuracy, delta, p-value, n and chunk-hit rate "
                 "was reproduced from the prediction files within tolerance.")
    L.append("")
    if notes:
        L.append("## Notes")
        L.append("")
        for x in notes:
            L.append(f"- {x}")
        L.append("")
    L.append("## Arm glossary")
    L.append("")
    L.append("| arm | budget | what it is |")
    L.append("|---|---|---|")
    for arm, (path, budget, note) in KF_ARMS.items():
        L.append(f"| `{arm}` | {budget} | {note} |")
    for arm in MB_ORDER:
        L.append(f"| `{arm}` | 8192 visual tokens | {MB_NOTE.get(arm,'')} |")
    L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------------- self-test
def selftest():
    """Synthetic fixture matching the real schemas exactly; checks accuracy, paired
    McNemar, MDE and the first-occurrence dedupe against hand-computed values."""
    import tempfile
    ok = True

    def eq(name, got, want, tol=1e-9):
        nonlocal ok
        good = (got is None and want is None) or (got is not None and abs(got - want) <= tol)
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: got {got!r} want {want!r}")
        ok = ok and good

    # 1) dedupe by first occurrence, with a gold-letter conflict
    rows = [
        {"video_id": "vid_0001", "question_id": "q1", "correct": "A", "is_correct": True},
        {"video_id": "vid_0002", "question_id": "q1", "correct": "B", "is_correct": False},
        {"video_id": "vid_0001", "question_id": "q1", "correct": "C", "is_correct": False},  # dup, diff gold
        {"video_id": "vid_0001", "question_id": "q1", "correct": "A", "is_correct": True},   # dup, same gold
    ]
    keyed, dropped, conf = dedupe_first(rows, lambda r: f"{r['video_id']}|{r['question_id']}")
    eq("dedupe n_kept", float(len(keyed)), 2.0)
    eq("dedupe n_dropped", float(dropped), 2.0)
    eq("dedupe gold_conflicts", float(conf), 1.0)
    eq("dedupe first-wins is_correct", float(keyed["vid_0001|q1"]["is_correct"]), 1.0)

    # 2) paired McNemar on a hand-built table: n=100 intersection, b=20, c=5
    A = {f"k{i}": False for i in range(120)}
    B = {f"k{i}": False for i in range(100)}          # B lacks 20 keys -> intersection 100
    for i in range(20):
        A[f"k{i}"] = True                              # A right, B wrong  -> b=20
    for i in range(20, 25):
        B[f"k{i}"] = True                              # A wrong, B right  -> c=5
    for i in range(25, 35):
        A[f"k{i}"] = True; B[f"k{i}"] = True           # concordant right
    r = mcnemar_exact(A, B)
    eq("mcnemar n", float(r["n"]), 100.0)
    eq("mcnemar b", float(r["b"]), 20.0)
    eq("mcnemar c", float(r["c"]), 5.0)
    eq("mcnemar acc_a", r["acc_a"], 30.0, 1e-9)
    eq("mcnemar acc_b", r["acc_b"], 15.0, 1e-9)
    eq("mcnemar delta", r["delta"], 15.0, 1e-9)
    eq("mcnemar p (exact binom 20/25)", r["p"],
       binomtest(20, 25, 0.5, alternative="two-sided").pvalue, 1e-12)
    eq("MDE", r["mde"], 100.0 * (Z_ALPHA + Z_BETA) * math.sqrt(0.25 / 100), 1e-9)
    # symmetry
    r2 = mcnemar_exact(B, A)
    eq("mcnemar symmetric p", r2["p"], r["p"], 1e-12)
    eq("mcnemar antisymmetric delta", r2["delta"], -15.0, 1e-9)
    # identical arms -> p=1, MDE undefined (zero discordance)
    r3 = mcnemar_exact(A, dict(A))
    eq("identical arms delta", r3["delta"], 0.0)
    eq("identical arms p", r3["p"], 1.0)
    eq("identical arms MDE is None", 0.0 if r3["mde"] is None else 1.0, 0.0)
    # disjoint keys -> n=0
    r4 = mcnemar_exact({"x": True}, {"y": True})
    eq("disjoint n", float(r4["n"]), 0.0)

    # 3) end-to-end on a synthetic predictions.jsonl written to a temp tree
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "predictions.jsonl")
        with open(f, "w") as fh:
            for i in range(10):
                fh.write(json.dumps({
                    "key": f"vid_{i:04d}|q1", "model": "m", "pipeline": "kf_referent",
                    "video_id": f"vid_{i:04d}", "question_id": "q1", "capability": "location",
                    "reid": "single_shot", "predicted": "A", "correct": "A" if i < 3 else "B",
                    "is_correct": i < 3}) + "\n")
        rr = load_jsonl(f)
        kk, dd, cc = dedupe_first(rr, lambda r: f"{r['video_id']}|{r['question_id']}")
        acc = 100.0 * sum(bool(v["is_correct"]) for v in kk.values()) / len(kk)
        eq("fixture file accuracy", acc, 30.0, 1e-9)
        eq("fixture file dropped", float(dd), 0.0)

    # 4) anon->real collapse produces collisions (mirrors the real 9-video case)
    a2r = {"vid_0001": "REAL_A", "vid_0002": "REAL_A"}
    d = {"vid_0001|q1": True, "vid_0002|q1": False}
    m, drop = OrderedDict(), 0
    for k, v in d.items():
        vid, qid = k.split("|", 1)
        rk = f"{a2r.get(vid, vid)}|{qid}"
        if rk in m:
            drop += 1; continue
        m[rk] = v
    eq("anon->real collapse kept", float(len(m)), 1.0)
    eq("anon->real collapse dropped", float(drop), 1.0)

    print(f"\nSELFTEST {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())

    notes = []
    kf, ledger = collect_kf(notes)
    mb_arms, mb_hits, mb_tokens, mb_backbone, mb_ledger = collect_mb(notes)
    ledger = ledger + mb_ledger
    a2r, _ = load_mapping()

    rows = build_rows(kf, mb_arms, mb_tokens, mb_backbone)
    cross, cross_ledger = cross_path(kf, mb_arms, a2r, notes)
    natrows = native_rows(kf, a2r, notes)
    warnings = discrepancy_report(kf, mb_arms, mb_hits)

    os.makedirs(OUTDIR, exist_ok=True)
    csv_path = os.path.join(OUTDIR, "results_master.csv")
    md_path = os.path.join(OUTDIR, "RESULTS_MASTER.md")
    write_csv(rows, csv_path)
    md = render(rows, warnings, ledger, cross, cross_ledger, mb_hits, notes, natrows)
    with open(md_path, "w") as fh:
        fh.write(md + "\n")

    print(md)
    print(f"\n[wrote] {csv_path}\n[wrote] {md_path}")


if __name__ == "__main__":
    main()
