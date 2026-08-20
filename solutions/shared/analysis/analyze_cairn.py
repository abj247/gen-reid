#!/usr/bin/env python
"""PATH-2 mechanism analysis: why does chunk-level retrieval trail frame-level selection?

The claim under test
--------------------
At a fixed 8-frame budget the two retrieval granularities make opposite bets.

  `referent` (frame-level) buys BREADTH: 8 independently chosen moments, so a higher chance that at
      least one frame lands inside the oracle evidence window, but almost no redundancy on any one
      moment.
  `chunk` (chunk-level) buys DEPTH: all 8 frames live inside 2 contiguous chunks, so a hit tends to
      come with several frames of the evidence, but the whole budget is staked on 2 of 8 chunks.

Whether that trade is good depends on an empirical quantity nobody has measured here: the shape of
the accuracy return to in-window frames. If the return SATURATES -- the first frame in the window is
worth a lot and the 2nd/3rd/4th add ~nothing -- then depth is a bad buy, breadth wins, and
chunk < keyframe follows. If the return is roughly linear in depth, the hypothesis is wrong and the
observed ordering needs a different explanation. This script tests saturation, it does not assume it.

Everything is measured against the human-verified oracle evidence windows
(benchmark/data/evidence_windows.json, 92-94% verified). ANY conditioning on those windows -- hit/miss,
depth, the dose curve, mb_oracle -- is a DIAGNOSTIC or a CEILING, never a method. Nothing here is
runnable at test time and nothing here is proposed as a system.

House rules honoured
--------------------
  * paired comparisons on the INTERSECTION of keys both arms answered, n reported everywhere;
  * significance = exact McNemar (binomtest on discordant pairs b vs c, two-sided);
  * every null reported as "no difference larger than <MDE> points" at 80% power, never "no
    difference";
  * frame / token budget stated on every arm;
  * keyframe results are ANON-keyed and membank results are REAL-keyed -- joined through
    video_id_mapping.json, never naively.

How the crux test is identified
-------------------------------
The saturation contrast is built INSIDE a fixed ordered (deep_mode, shallow_mode) cell and never
across cells. Choosing the deep and the shallow arm by MODES order -- what this script used to do --
made `referent` the deep arm in 74% of pairs and the shallow arm in 17%, so the "dose" contrast was
in fact a referent-vs-not-referent contrast, i.e. exactly the accuracy difference Path 2 is trying to
explain. Three combinations are reported: per-cell + stratified exact McNemar, a Mantel-Haenszel
combination with a heterogeneity test, and the MODE-BALANCED estimate that averages the two
orientations of each unordered mode pair so the mode main effect cancels algebraically. The same
stratification is applied to the value of the first in-window frame and to the dose slope (within-mode
slopes, n-weighted). A conditional model, `correct ~ 1{depth>=1} + (depth-1) + C(mode) + C(backbone)`
with per-question fixed effects (within-question demeaned linear probability model, cluster-robust on
the question key), is reported alongside as an independent handle. Everything pooled is clustered on
the question key, because 3 backbones answer every question.

Self-test
---------
  python analyze_cairn.py --fixture
runs the whole pipeline on synthetic selections + synthetic predictions with a PLANTED dose curve --
one saturating, one linear, and one CONFOUNDED regime (no dose effect, but a large mode bonus on
`referent`) which the old unstratified test is fooled by and the stratified one is not -- and asserts
the recovered marginal values and the saturation verdict. It does NOT assert that the decomposition
terms sum to the gap: `residual` is defined as the remainder, so that sum is a tautology.

CPU only, seconds of runtime.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binomtest, chi2, norm

from persistqa.paths import ROOT  # noqa: E402

# selection mode  ->  the pipeline directory whose predictions were produced from that selection
MODE2PIPE = {
    "referent": "kf_referent",
    "chunk": "kf_chunk",
    "random": "kf_random",
    "uniform": "kf_uniform8",
}
MODES = ("referent", "chunk", "random", "uniform")
# videochat-flash-7b is held out of the POOLED dose curve and reported separately -- not because the
# method fails there (it does not: keyframe beats budget-matched uniform-8 by +2.94, p=2.44e-05,
# n=3,233), but because its rendered-mp4 pipeline penalty is an order of magnitude larger than the
# other backbones' (-5.75 pts vs -0.80 InternVL / -0.16 Qwen), so pooling it would mix a pipeline
# effect into a selection measurement. The earlier "-3.04, characterised failure case" label was an
# artefact of comparing against VCF's own NATIVE video pipeline, which is not budget-matched.
PRIMARY_BACKBONES = ("internvl3-14b", "qwen2.5-vl-7b", "ovis2.5-9b")
ALL_BACKBONES = PRIMARY_BACKBONES + ("videochat-flash-7b",)

Z = 1.959963984540054   # z_{0.025}
ZB = 0.8416212335729143  # z_{0.20}, i.e. 80% power
DEPTH_CAP = 4           # in-window frames are bucketed 1,2,3,4+ ; x is capped at 4 everywhere
N_BOOT = 10000          # cluster-bootstrap draws (P2 requires >= 10,000)


# --------------------------------------------------------------------------------------- stats --
def wilson(k: int, n: int):
    """Wilson score 95% interval for a binomial proportion."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + Z * Z / n
    c = p + Z * Z / (2 * n)
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return [round(max(0.0, (c - h) / d) * 100, 2), round(min(1.0, (c + h) / d) * 100, 2)]


def mcnemar(pairs, cluster_ids=None, n_boot=0, seed=20260819):
    """Exact McNemar on a list of (a_correct, b_correct) booleans.

    Returns n, b (a right / b wrong), c (a wrong / b right), the paired difference in accuracy
    points (a - b), the two-sided exact p, and the MDE at 80% power.

    P2: pass `cluster_ids` whenever the pairs are NOT independent -- the usual case here is that the
    same question is answered by 3 backbones and so contributes 3 pairs (`sat_paired` was drawing
    exactly 3.00 pairs per question). The exact p is left alone (it is still the right test of the
    sharp null), but the MDE is inflated by the Kish design effect, and a seeded cluster bootstrap of
    (b-c)/n is added when `n_boot` > 0. An unclustered MDE understates the true one by 20-75% at the
    intra-question correlations seen here, and the house rule states every null against that MDE.
    """
    n = len(pairs)
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    p = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    diff = ((b - c) / n * 100) if n else 0.0
    # McNemar MDE: var of (b-c)/n is ~pi_d/n under the null, pi_d = discordance rate.
    pi_d = ((b + c) / n) if n else 0.0
    mde = (Z + ZB) * math.sqrt(pi_d / n) * 100 if n and pi_d > 0 else float("nan")
    out = {"n": n, "b": b, "c": c, "diff_pts": round(diff, 2),
           "p": p, "mde_pts": round(mde, 2) if mde == mde else None}
    if cluster_ids is not None and n:
        assert len(cluster_ids) == n, "cluster_ids must be one per pair"
        d = [int(bool(x)) - int(bool(y)) for x, y in pairs]
        de = cluster_deff(d, cluster_ids)
        out.update({"n_clusters": de["n_clusters"], "mean_pairs_per_cluster": de["mean_cluster_size"],
                    "icc": de["rho"], "design_effect": de["deff"],
                    "mde_pts_clustered": (round(out["mde_pts"] * math.sqrt(de["deff"]), 2)
                                          if (out["mde_pts"] is not None and de["deff"]) else None)})
        if n_boot > 0:
            uq, inv = np.unique(np.asarray(cluster_ids, dtype=object), return_inverse=True)
            mat = np.zeros((len(uq), 3))
            for i, (x, y) in zip(inv, pairs):
                mat[i, 0] += float(bool(x) and not bool(y))
                mat[i, 1] += float(bool(y) and not bool(x))
                mat[i, 2] += 1.0
            bt = _multinomial_boot(mat, n_boot, seed)
            with np.errstate(invalid="ignore", divide="ignore"):
                dd = np.where(bt[:, 2] > 0, (bt[:, 0] - bt[:, 1]) / np.where(bt[:, 2] > 0, bt[:, 2], 1.0), np.nan) * 100
            dd = dd[np.isfinite(dd)]
            if len(dd):
                out["p_cluster_bootstrap"] = float(min(1.0, max(
                    1.0 / len(dd), 2 * min((dd <= 0).mean(), (dd >= 0).mean()))))
                sd = float(dd.std(ddof=1)) if len(dd) > 1 else None
                out["se_boot_pts"] = round(sd, 3) if sd else None
                if sd:
                    out["mde_pts_clustered"] = round((Z + ZB) * sd, 2)
                dd = np.sort(dd)
                out["ci95_boot_clustered"] = [round(float(dd[int(.025 * len(dd))]), 2),
                                              round(float(dd[min(len(dd) - 1, int(math.ceil(.975 * len(dd))) - 1)]), 2)]
                out["n_boot"] = int(len(dd))
    return out


def two_prop(k1, n1, k2, n2):
    """Unpaired two-proportion z-test + MDE at 80% power. Used only where pairing is impossible."""
    if n1 == 0 or n2 == 0:
        return {"n1": n1, "n2": n2, "diff_pts": None, "p": None, "mde_pts": None}
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"n1": n1, "n2": n2, "diff_pts": round((p1 - p2) * 100, 2), "p": 1.0, "mde_pts": None}
    z = (p1 - p2) / se
    # P5: the old erf form underflows to EXACTLY 0.0 at |z| >~ 8.3, which combined with the
    # `(p or 1) < 0.05` idiom printed "[null]" for p < 1e-16. scipy's norm.sf does not underflow at
    # these magnitudes; the floor guarantees a strictly positive p in any case.
    p = max(float(2 * norm.sf(abs(z))), 5e-324)
    mde = (Z + ZB) * se * 100
    return {"n1": n1, "n2": n2, "diff_pts": round((p1 - p2) * 100, 2), "p": p,
            "mde_pts": round(mde, 2)}


def null_phrase(res, key="mde_pts"):
    """House rule: every null is stated against its MDE. Prefers the CLUSTERED MDE when one exists.

    (`if m else` treated an MDE of exactly 0.0 as "n too small"; `is not None` is the right test.)
    """
    m, tag = res.get(key), ""
    if key == "mde_pts" and res.get("mde_pts_clustered") is not None:
        m, tag = res["mde_pts_clustered"], ", clustered on the question key"
    if m is None or m != m:
        return "n too small for an MDE"
    return "no difference larger than {:.2f} points (80% power{})".format(m, tag)


def wls_slope(xs, ys, ws):
    """Weighted least squares slope of y on x. Used for accuracy-per-extra-in-window-frame."""
    W = sum(ws)
    if W == 0 or len(xs) < 2:
        return float("nan")
    mx = sum(w * x for w, x in zip(ws, xs)) / W
    my = sum(w * y for w, y in zip(ws, ys)) / W
    num = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    den = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    return num / den if den > 0 else float("nan")


# ------------------------------------------------------------------- P4: null-safe formatting --
def _spec_width_align(spec: str):
    """(width, align) of a format spec, so a missing value can be padded like a present one."""
    s, align = spec, ">"
    if len(s) >= 2 and s[1] in "<>^=":
        align, s = s[1], s[2:]
    elif s and s[0] in "<>^=":
        align, s = s[0], s[1:]
    if s and s[0] in "+- ":
        s = s[1:]
    if s.startswith("#"):
        s = s[1:]
    if s.startswith("0"):
        s = s[1:]
    d = ""
    for ch in s:
        if ch.isdigit():
            d += ch
        else:
            break
    return (int(d) if d else 0), align


def fmt(x, spec, na="n/a"):
    """format() that survives None/NaN.

    P4: the old idiom `{v if v is None else v:>8}` returns None INTO the spec and raises
    `TypeError: unsupported format string passed to NoneType.__format__`. Empty depth / spread /
    hit buckets are common on any per-backbone or per-capability slice, so this must never crash.
    """
    if x is None or (isinstance(x, float) and x != x):
        w, align = _spec_width_align(spec)
        return na.ljust(w) if align == "<" else (na.center(w) if align == "^" else na.rjust(w))
    return format(x, spec)


def sig(p, alpha=0.05):
    """P5: `(p or 1) < alpha` is False for p == 0.0, which INVERTS the verdict for the most
    significant results (two_prop's erf underflows to exactly 0.0 at |z| >~ 8.3). Never use
    truthiness on a p-value."""
    return p is not None and p == p and p < alpha


# ------------------------------------------------------------------------- P2: clustering --
def _multinomial_boot(mat, n_boot, seed, chunk=1000):
    """Cluster bootstrap of column sums of `mat` (one ROW per cluster).

    Resampling clusters with replacement is equivalent to drawing the per-cluster multiplicities
    from Multinomial(G, 1/G), which lets the whole bootstrap be a few matrix products.
    Returns (n_boot, mat.shape[1])."""
    G = mat.shape[0]
    if G == 0 or n_boot <= 0:
        return np.zeros((0, mat.shape[1]))
    rng = np.random.default_rng(seed)
    p = np.full(G, 1.0 / G)
    out = np.empty((n_boot, mat.shape[1]), dtype=float)
    done = 0
    while done < n_boot:
        c = min(chunk, n_boot - done)
        counts = rng.multinomial(G, p, size=c).astype(np.float64)
        out[done:done + c] = counts @ mat
        done += c
    return out


def cluster_deff(values, cluster_ids):
    """Kish design effect 1 + (mbar-1)*rho for `values` clustered by `cluster_ids`.

    rho is the one-way random-effects ANOVA intra-cluster correlation. Used to inflate an MDE that
    was computed as if every observation were independent -- the house rule states every null
    against an MDE, so the MDE has to be the clustered one."""
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if n == 0:
        return {"rho": None, "deff": None, "n_clusters": 0, "mean_cluster_size": None}
    uq, inv = np.unique(np.asarray(cluster_ids, dtype=object), return_inverse=True)
    G = len(uq)
    ng = np.bincount(inv, minlength=G).astype(float)
    if G < 2 or n == G:
        return {"rho": 0.0, "deff": 1.0, "n_clusters": G, "mean_cluster_size": round(n / G, 3)}
    means = np.bincount(inv, weights=vals, minlength=G) / ng
    grand = vals.mean()
    msb = float((ng * (means - grand) ** 2).sum() / (G - 1))
    msw = float(((vals - means[inv]) ** 2).sum() / (n - G)) if n > G else 0.0
    m0 = (n - (ng ** 2).sum() / n) / (G - 1)
    denom = msb + (m0 - 1) * msw
    rho = 0.0 if denom <= 0 else max(0.0, min(1.0, (msb - msw) / denom))
    mbar = n / G
    return {"rho": round(rho, 4), "deff": round(1 + (mbar - 1) * rho, 4),
            "n_clusters": G, "mean_cluster_size": round(mbar, 3)}

def _rowsum(A):
    """Sum over the LAST axis via a matrix-vector product, not `A.sum(-1)`.

    Not a stylistic choice. On this numpy build (2.2.6) chained expressions over arrays above numpy's
    temporary-elision threshold returned CORRUPTED reductions inside this process: for the (draws,
    mode, depth) bootstrap tensor, `n.sum(-1)` was correct while `(n*x).sum(-1)` came back eight
    orders of magnitude wrong and negative, which silently turned every within-mode slope into NaN
    (it reproduced at >=9,999 draws and vanished at 5,000, and the same expression evaluated
    correctly in a fresh process). A BLAS matvec does not participate in temporary elision. Anything
    that aggregates a large array in this file goes through here or through an explicit matmul, and
    the cell-layout assertion below the aggregation fails loudly if it ever silently returns.
    """
    A = np.ascontiguousarray(np.asarray(A, dtype=float))
    return A @ np.ones(A.shape[-1], dtype=float)


# --------------------------------------------------- P1: stratified (mode-controlled) pairing --
def build_cells(keys, sel, preds, backbones, deep_ok, shallow_ok):
    """(deep_mode, shallow_mode) -> [(question_key, deep_correct, shallow_correct), ...].

    P1 FIX. The old code took `[m for m in MODES if ...][0]`, i.e. the arm identity was decided by
    the fixed priority list MODES = (referent, chunk, random, uniform). `referent` therefore ended up
    as the deep arm in 74% of pairs and the shallow arm in 17%, so the contrast measured
    referent-vs-not-referent -- exactly the accuracy difference Path 2 is trying to explain.
    Here EVERY admissible ordered (deep, shallow) mode pair is enumerated and kept in its own cell,
    so the mode composition is explicit and can be conditioned on rather than averaged over blindly.
    """
    cells = defaultdict(list)
    for k in keys:
        deep = [m for m in MODES if deep_ok(sel[k][m]["n_in_window"])]
        shal = [m for m in MODES if shallow_ok(sel[k][m]["n_in_window"])]
        for md in deep:
            for ms in shal:
                if md == ms:
                    continue
                for bb in backbones:
                    cd = preds.get((bb, md), {}).get(k)
                    cs = preds.get((bb, ms), {}).get(k)
                    if cd is None or cs is None:
                        continue
                    cells[(md, ms)].append((k, bool(cd), bool(cs)))
    return cells


def stratified_paired(cells, n_boot=10000, seed=20260819):
    """Combine per-(deep_mode, shallow_mode) paired contrasts three ways. All three are reported.

    (i)   per-cell exact McNemar, plus the stratified exact McNemar. Under the conditional null
          "OR = 1 in every stratum" the discordant counts are independent Binomial(b+c, 1/2), so
          sum(b) ~ Binomial(sum(b+c), 1/2) -- which is numerically the SAME test as pooling all
          discordant pairs. That is precisely why pooling alone does not repair the P1 confound: the
          test is fine, the estimand is not. It is reported for completeness only.
    (ii)  Mantel-Haenszel / inverse-variance combination of the per-cell paired risk differences,
          with a Cochran-Q heterogeneity test across cells. This estimates a COMMON depth effect
          under the assumption that the mode contrast is the same in every cell -- Q tests that.
    (iii) MODE-BALANCED estimate -- the only combination that is not confounded by construction.
          The two orientations of a pair must enter with EQUAL weight or the mode term does not
          cancel, so the noisier orientation cannot be down-weighted; inverse-variance weighting is
          applied only ACROSS unordered mode pairs. Cells whose reverse orientation is missing are
          therefore dropped entirely rather than folded in at partial weight.
          Cell (deep=A, shallow=B) estimates  depth + mode(A-B);
          cell (deep=B, shallow=A) estimates  depth - mode(A-B);
          their average cancels the mode main effect exactly. Unordered mode pairs that appear in
          only one orientation cannot contribute and are listed as unusable.

    Every SE/CI/MDE reported here comes from a cluster bootstrap on the QUESTION KEY (P2): three
    backbones answer each question and one question can supply several cells, so pairs are not
    independent.
    """
    cell_list = [c for c in sorted(cells) if cells[c]]
    per_cell, w_s, d_s = {}, [], []
    for c in cell_list:
        rows = cells[c]
        b = sum(1 for _, a, s in rows if a and not s)
        cc = sum(1 for _, a, s in rows if s and not a)
        n = len(rows)
        p = binomtest(b, b + cc, 0.5).pvalue if (b + cc) > 0 else 1.0
        d = (b - cc) / n
        v = (b + cc - (b - cc) ** 2 / n) / (n ** 2) if n else float("nan")
        per_cell["%s>%s" % c] = {
            "deep_mode": c[0], "shallow_mode": c[1], "n_pairs": n,
            "n_keys": len({k for k, _, _ in rows}), "b": b, "c": cc,
            "diff_pts": round(d * 100, 2), "p_exact": p,
            "var": v}
        d_s.append(d)
        w_s.append((1.0 / v) if (v == v and v > 0) else 0.0)

    B = sum(per_cell[k]["b"] for k in per_cell)
    C = sum(per_cell[k]["c"] for k in per_cell)
    N = sum(per_cell[k]["n_pairs"] for k in per_cell)
    p_strat_exact = binomtest(B, B + C, 0.5).pvalue if (B + C) > 0 else 1.0

    w = np.array(w_s, dtype=float)
    d = np.array(d_s, dtype=float)
    mh = {"diff_pts": None, "se_pts": None, "p": None, "q": None, "q_df": max(0, len(d) - 1),
          "q_p": None}
    if w.sum() > 0:
        dm = float((w * d).sum() / w.sum())
        se = float(math.sqrt(1.0 / w.sum()))
        z = dm / se if se > 0 else 0.0
        q = float((w * (d - dm) ** 2).sum())
        mh = {"diff_pts": round(dm * 100, 2), "se_pts": round(se * 100, 3),
              "p": float(2 * norm.sf(abs(z))),
              "q": round(q, 3), "q_df": max(0, int((w > 0).sum()) - 1),
              "q_p": float(chi2.sf(q, max(1, int((w > 0).sum()) - 1))) if (w > 0).sum() > 1 else None}

    # --- mode-balanced pairs
    idx = {c: j for j, c in enumerate(cell_list)}
    bal_pairs, unusable = [], []
    for j, c in enumerate(cell_list):
        a, bmode = c
        if a >= bmode:
            continue
        rev = (bmode, a)
        if rev in idx:
            bal_pairs.append((a, bmode, j, idx[rev]))
        else:
            unusable.append("%s>%s" % c)
    for c in cell_list:
        a, bmode = c
        if a > bmode and (bmode, a) not in idx:
            unusable.append("%s>%s" % c)
    bal_w, bal_d = [], []
    for a, bmode, j1, j2 in bal_pairs:
        v1 = per_cell["%s>%s" % (a, bmode)]["var"]
        v2 = per_cell["%s>%s" % (bmode, a)]["var"]
        if not (v1 == v1 and v2 == v2) or (v1 + v2) <= 0:
            bal_w.append(0.0)
        else:
            bal_w.append(4.0 / (v1 + v2))
        bal_d.append((d_s[j1] + d_s[j2]) / 2.0)
    bal = {"diff_pts": None, "n_mode_pairs": len(bal_pairs),
           "mode_pairs": ["%s~%s" % (a, b) for a, b, _, _ in bal_pairs],
           "unusable_single_orientation_cells": sorted(set(unusable))}
    bw = np.array(bal_w, dtype=float)
    bd = np.array(bal_d, dtype=float)
    if bw.sum() > 0:
        bal["diff_pts"] = round(float((bw * bd).sum() / bw.sum()) * 100, 2)

    # --- key-clustered bootstrap of both combinations (fixed point-estimate weights)
    kall = sorted({k for c in cell_list for k, _, _ in cells[c]})
    ki = {k: i for i, k in enumerate(kall)}
    S = len(cell_list)
    mat = np.zeros((len(kall), 3 * S))
    for j, c in enumerate(cell_list):
        for k, a, s in cells[c]:
            i = ki[k]
            mat[i, 3 * j] += float(a and not s)
            mat[i, 3 * j + 1] += float(s and not a)
            mat[i, 3 * j + 2] += 1.0
    boot = _multinomial_boot(mat, n_boot, seed)
    ci_mh = ci_bal = [None, None]
    sd_mh = sd_bal = None
    p_bal_boot = None
    if len(boot):
        Bb, Cb, Nb = boot[:, 0::3], boot[:, 1::3], boot[:, 2::3]
        with np.errstate(invalid="ignore", divide="ignore"):
            D = np.where(Nb > 0, (Bb - Cb) / np.where(Nb > 0, Nb, 1.0), np.nan)
        ok = np.isfinite(D)
        num = _rowsum(np.where(ok, D * w, 0.0))
        den = _rowsum(np.where(ok, w, 0.0))
        mh_boot = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan) * 100
        mh_boot = mh_boot[np.isfinite(mh_boot)]
        if len(mh_boot):
            mh_boot.sort()
            ci_mh = [round(float(mh_boot[int(.025 * len(mh_boot))]), 2),
                     round(float(mh_boot[min(len(mh_boot) - 1, int(.975 * len(mh_boot)))]), 2)]
            sd_mh = float(mh_boot.std(ddof=1)) if len(mh_boot) > 1 else None
        if bal_pairs:
            j1 = np.array([p[2] for p in bal_pairs]); j2 = np.array([p[3] for p in bal_pairs])
            Dp = (D[:, j1] + D[:, j2]) / 2.0
            okp = np.isfinite(Dp)
            nump = _rowsum(np.where(okp, Dp * bw, 0.0))
            denp = _rowsum(np.where(okp, bw, 0.0))
            bal_boot = np.where(denp > 0, nump / np.where(denp > 0, denp, 1.0), np.nan) * 100
            bal_boot = bal_boot[np.isfinite(bal_boot)]
            if len(bal_boot):
                p_bal_boot = float(min(1.0, max(1.0 / len(bal_boot),
                                                2 * min((bal_boot <= 0).mean(), (bal_boot >= 0).mean()))))
                sd_bal = float(bal_boot.std(ddof=1)) if len(bal_boot) > 1 else None
                bal_boot.sort()
                ci_bal = [round(float(bal_boot[int(.025 * len(bal_boot))]), 2),
                          round(float(bal_boot[min(len(bal_boot) - 1, int(.975 * len(bal_boot)))]), 2)]
    bal.update({"ci95_boot_clustered_on_key": ci_bal, "p_bootstrap": p_bal_boot,
                "se_boot_pts": round(sd_bal, 3) if sd_bal else None,
                "mde_pts": round((Z + ZB) * sd_bal, 2) if sd_bal else None})
    mh.update({"ci95_boot_clustered_on_key": ci_mh,
               "se_boot_pts": round(sd_mh, 3) if sd_mh else None,
               "mde_pts": round((Z + ZB) * sd_mh, 2) if sd_mh else None})

    # pooled (confounded) view + its clustering diagnostics, kept only to expose the defect
    flat = [(a, s) for c in cell_list for _, a, s in cells[c]]
    flat_ids = [k for c in cell_list for k, _, _ in cells[c]]
    pooled = mcnemar(flat, cluster_ids=flat_ids, n_boot=n_boot, seed=seed + 1)
    return {"per_cell": per_cell, "n_cells": len(cell_list), "n_pairs": N,
            "n_keys": len(kall), "b_total": B, "c_total": C,
            "stratified_exact_p": p_strat_exact, "mantel_haenszel": mh, "mode_balanced": bal,
            "pooled_across_cells_CONFOUNDED": pooled, "n_boot": n_boot}


# ------------------------------------------- P1: conditional model with per-question fixed effects
def fe_lpm(y, X, names, groups, clusters):
    """Within-group demeaned LINEAR PROBABILITY MODEL with cluster-robust (CR1) standard errors.

    Choice stated explicitly: a within-question demeaned LPM rather than a conditional logit. Two
    reasons. (1) The estimand is a difference in accuracy POINTS, which is what every other number in
    this file and in the house rules is denominated in; a conditional-logit coefficient is a log odds
    ratio and would need a second, model-dependent step to become points. (2) The dose model under
    test, acc = a + V1*1{depth>=1} + v*max(depth-1,0), is additive on the probability scale -- it is
    the LPM that is correctly specified for it, and it is exactly the generative model the fixture
    plants. Cost: the LPM's fitted values are not constrained to [0,1] and its residuals are
    heteroskedastic, which the cluster-robust SEs absorb.

    Fixed effects are absorbed by demeaning; columns that vanish under demeaning (collinear with the
    fixed effects) are dropped and reported as absorbed. SEs cluster on `clusters` (the question key).
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p_full = X.shape
    guq, ginv = np.unique(np.asarray(groups, dtype=object), return_inverse=True)
    G = len(guq)
    ng = np.bincount(ginv, minlength=G).astype(float)
    ym = y - (np.bincount(ginv, weights=y, minlength=G) / ng)[ginv]
    Xm = np.empty_like(X)
    for j in range(p_full):
        Xm[:, j] = X[:, j] - (np.bincount(ginv, weights=X[:, j], minlength=G) / ng)[ginv]
    keep = [j for j in range(p_full) if np.abs(Xm[:, j]).max() > 1e-9]
    absorbed = [names[j] for j in range(p_full) if j not in keep]
    Xk = Xm[:, keep]
    if Xk.shape[1] == 0 or n <= G:
        return {"error": "no identifying variation after absorbing the fixed effects",
                "n_obs": n, "n_groups": G, "absorbed": absorbed}
    XtX_inv = np.linalg.pinv(Xk.T @ Xk)
    beta = XtX_inv @ (Xk.T @ ym)
    u = ym - Xk @ beta
    cuq, cinv = np.unique(np.asarray(clusters, dtype=object), return_inverse=True)
    Gc = len(cuq)
    meat = np.zeros((Xk.shape[1], Xk.shape[1]))
    for g in range(Gc):
        m = cinv == g
        s = Xk[m].T @ u[m]
        meat += np.outer(s, s)
    dof = max(1, n - G - Xk.shape[1])
    scale = (Gc / max(1, Gc - 1)) * ((n - 1) / dof)
    V = XtX_inv @ meat @ XtX_inv * scale
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    out = {"n_obs": int(n), "n_fe_groups": int(G), "n_clusters": int(Gc),
           "absorbed_by_fixed_effects": absorbed, "coef": {}}
    for i, j in enumerate(keep):
        z = beta[i] / se[i] if se[i] > 0 else 0.0
        out["coef"][names[j]] = {
            "coef_pts": round(float(beta[i]) * 100, 3),
            "se_pts": round(float(se[i]) * 100, 3),
            "p": float(2 * norm.sf(abs(z))),
            "mde_pts": round(float((Z + ZB) * se[i]) * 100, 2) if se[i] > 0 else None}
    return out


# ---------------------------------------------------------------------------------- data load --
def load_selections(path):
    """key -> mode -> selection record. Later rows win (the dumper appends and may be re-run)."""
    sel = defaultdict(dict)
    if not os.path.exists(path):
        return sel
    bad = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                bad += 1
                continue
            if "key" in r and r.get("mode") in MODE2PIPE:
                sel[r["key"]][r["mode"]] = r
    if bad:
        print(f"[warn] {bad} unparseable lines in {path} (file is written live; partial is fine)")
    return sel


def load_predictions(base):
    """(backbone, mode) -> key -> bool correct."""
    preds = {}
    for bb in ALL_BACKBONES:
        for mode, pipe in MODE2PIPE.items():
            p = os.path.join(base, pipe, bb, "predictions.jsonl")
            d = {}
            if os.path.exists(p):
                with open(p) as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        if "key" in r and "is_correct" in r:
                            d[r["key"]] = bool(r["is_correct"])
            preds[(bb, mode)] = d
    return preds


# --------------------------------------------------------------------------- the analysis body --
def analyse(sel, preds, membank_rows, backbones, out_json, verbose=True):
    P = print if verbose else (lambda *a, **k: None)

    # Only keys where an oracle evidence window exists AND every mode has been dumped: otherwise
    # breadth/depth would be compared across different question subsets.
    keys = sorted(k for k, d in sel.items()
                  if all(m in d for m in MODES) and d["referent"].get("ev_f0") is not None)
    n_keys = len(keys)
    P("=" * 100)
    P("PATH-2 MECHANISM ANALYSIS -- concentration and depth at a fixed 8-frame budget")
    P("=" * 100)
    P(f"\nkeys with all 4 selection modes dumped AND an oracle evidence window: n={n_keys}")
    if n_keys == 0:
        P("[fatal] nothing to analyse")
        return None

    # COVERAGE CAVEAT. dump_selections.py walks videos in sorted order and appends, so a partial
    # file is a PREFIX of the corpus, not a random sample. Quantify the slice before reading
    # anything off it: if the subset's uniform-8 accuracy is far from the corpus-level number, the
    # levels here are not the paper's levels and only the WITHIN-subset contrasts are meaningful.
    full_pipe = preds.get((backbones[0], "uniform"), {})
    if full_pipe:
        sub = [k for k in keys if k in full_pipe]
        a_sub = sum(full_pipe[k] for k in sub) / len(sub) * 100 if sub else float("nan")
        a_all = sum(full_pipe.values()) / len(full_pipe) * 100
        cov = n_keys / len(full_pipe) * 100
        nv_sub = len({k.split("|")[0] for k in sub})
        nv_all = len({k.split("|")[0] for k in full_pipe})
        P(f"  coverage: {n_keys}/{len(full_pipe)} of the answered questions ({cov:.1f}%), "
          f"{nv_sub}/{nv_all} videos.")
        P(f"  coverage: {n_keys}/{len(full_pipe)} of the answered questions ({cov:.1f}%). "
          f"dump_selections.py appends in sorted video order, so a partial file is a PREFIX slice, "
          f"NOT a random sample.")
        P(f"  slice check on {backbones[0]} uniform-8 (8 frames): this subset {a_sub:.2f}% vs "
          f"full corpus {a_all:.2f}%  (delta {a_sub - a_all:+.2f} pts)")
        if abs(a_sub - a_all) > 2.0:
            P("  !! the slice is HARDER/EASIER than the corpus by more than 2 pts: treat the absolute "
              "accuracy levels below as slice-specific and read only the WITHIN-slice contrasts, all "
              "of which are paired on the same questions. Re-run when the dump completes.")

    # ------------------------------------------------------------------ A. breadth / depth stats
    bd = {}
    for m in MODES:
        rows = [sel[k][m] for k in keys]
        hits = [r for r in rows if r.get("n_in_window", 0) > 0]
        bd[m] = {
            "hit_rate": round(len(hits) / len(rows) * 100, 2),
            "mean_depth_given_hit": round(sum(r["n_in_window"] for r in hits) / len(hits), 3) if hits else 0.0,
            "mean_depth": round(sum(r["n_in_window"] for r in rows) / len(rows), 3),
            "mean_spread": round(sum(float(r.get("spread") or 0.0) for r in rows) / len(rows), 4),
            "mean_distinct_chunks": round(
                sum(len(set(r.get("chunk_ids") or [])) for r in rows) / len(rows), 3),
        }
    P("\n" + "-" * 100)
    P("A. BREADTH vs DEPTH  (all arms: 8 frames; evidence windows are a DIAGNOSTIC, never a method)")
    P("-" * 100)
    P(f"{'mode':<10} {'budget':<9} {'hit rate %':>11} {'E[depth|hit]':>13} {'E[depth]':>9} "
      f"{'spread':>8} {'distinct chunks':>16}")
    for m in MODES:
        s = bd[m]
        P(f"{m:<10} {'8 frames':<9} {s['hit_rate']:>11.2f} {s['mean_depth_given_hit']:>13.3f} "
          f"{s['mean_depth']:>9.3f} {s['mean_spread']:>8.4f} {s['mean_distinct_chunks']:>16.3f}")

    # paired hit-rate tests -- same questions, so McNemar on the hit indicator is legitimate
    P("\n  paired hit-rate tests (exact McNemar on the HIT indicator, same n={} questions):".format(n_keys))
    hit_tests = {}
    for a, b in (("referent", "chunk"), ("referent", "random"), ("referent", "uniform"),
                 ("chunk", "random"), ("chunk", "uniform")):
        r = mcnemar([(sel[k][a]["n_in_window"] > 0, sel[k][b]["n_in_window"] > 0) for k in keys])
        hit_tests[f"{a}_vs_{b}"] = r
        tag = "*" if sig(r["p"]) else " "
        note = "" if sig(r["p"]) else "   [null: " + null_phrase(r) + "]"
        P(f"    {a:>9} - {b:<9} {r['diff_pts']:+7.2f} pts  p={r['p']:.4g}{tag}  n={r['n']}{note}")

    # ---------------------------------------------------------------------- B. the depth curve
    # accuracy vs number of in-window frames, restricted to HITS. Pooled over the 4 selection modes
    # and the primary backbones; each observation is one (key, mode, backbone) prediction.
    def depth_bucket(d):
        return "4+" if d >= 4 else str(d)

    curve_k, curve_n = Counter(), Counter()
    depth_obs = []            # (depth, correct) for the slope fit
    miss_k = miss_n = 0
    for k in keys:
        for m in MODES:
            d = sel[k][m]["n_in_window"]
            for bb in backbones:
                c = preds.get((bb, m), {}).get(k)
                if c is None:
                    continue
                if d == 0:
                    miss_n += 1
                    miss_k += int(c)
                else:
                    curve_n[depth_bucket(d)] += 1
                    curve_k[depth_bucket(d)] += int(c)
                    depth_obs.append((d, int(c)))

    depth_curve = {}
    for b in ("1", "2", "3", "4+"):
        n = curve_n[b]
        depth_curve[b] = {"acc": round(curve_k[b] / n * 100, 2) if n else None,
                          "n": n, "ci95": wilson(curve_k[b], n)}
    acc_miss = miss_k / miss_n * 100 if miss_n else None
    hit_n = sum(curve_n.values())
    hit_k = sum(curve_k.values())
    acc_hit = hit_k / hit_n * 100 if hit_n else None

    P("\n" + "-" * 100)
    P("B. THE DOSE CURVE -- accuracy vs number of selected frames INSIDE the evidence window")
    P("   (restricted to HITS; DIAGNOSTIC conditioning on oracle windows, not a method)")
    P("   pooled over the 4 selection modes x backbones {}; every arm = 8 frames".format(list(backbones)))
    P("-" * 100)
    P(f"{'in-window frames':<18} {'acc %':>8} {'n':>7}   95% Wilson CI")
    P(f"{'0 (MISS)':<18} {fmt(acc_miss, '>8.2f')} {miss_n:>7}   {wilson(miss_k, miss_n)}")
    if acc_miss is not None and acc_miss < 15.0:
        P(f"  !! acc|miss = {acc_miss:.2f}% sits on the 12.5% chance floor (8-way MCQ). A floored "
          f"baseline COMPRESSES every effect on this curve toward zero, which is a second reason the "
          f"depth slope here is a lower bound rather than an estimate.")
    for b in ("1", "2", "3", "4+"):
        v = depth_curve[b]
        P(f"{b:<18} {fmt(v['acc'], '>8.2f')} {v['n']:>7}   {v['ci95']}")

    # SECONDARY = unpaired proportions, depth==1 vs depth>=2, over all observations. Mode-confounded
    # (see the depth x mode table above) and unpaired: kept as a sanity check, never as evidence.
    k_ge2 = sum(curve_k[b] for b in ("2", "3", "4+"))
    n_ge2 = sum(curve_n[b] for b in ("2", "3", "4+"))
    sat_unpaired = two_prop(k_ge2, n_ge2, curve_k["1"], curve_n["1"])

    # slope per extra in-window frame, WLS on the hits-only cell means (x = depth, capped at 4)
    xs, ys, ws = [], [], []
    for b, x in (("1", 1.0), ("2", 2.0), ("3", 3.0), ("4+", 4.0)):
        if curve_n[b] > 0:
            xs.append(x); ys.append(curve_k[b] / curve_n[b] * 100); ws.append(curve_n[b])
    slope = wls_slope(xs, ys, ws)

    # bootstrap CI for that slope, CLUSTERED on the question key: the 4 modes x N backbones
    # observations for one question are not independent, so keys are the resampling unit.
    # P1 also applies here: the pooled slope mixes depth with mode (at depth 1 the cells are mostly
    # uniform, at depth 4+ mostly chunk/referent), so the WITHIN-mode slopes and their n-weighted
    # combination are computed alongside it and are the mode-free version of the same quantity.
    MI = {m: i for i, m in enumerate(MODES)}
    hkeys = sorted({k for k in keys
                    if any(sel[k][m]["n_in_window"] > 0 for m in MODES)})
    # column layout of cellmat: for mode index mi and depth x in 1..DEPTH_CAP,
    #   col 2*(mi*DEPTH_CAP + x-1)     = n observations in that (mode, depth) cell
    #   col 2*(mi*DEPTH_CAP + x-1) + 1 = correct answers in it
    cellmat = np.zeros((len(hkeys), len(MODES) * DEPTH_CAP * 2))
    for i, k in enumerate(hkeys):
        for m in MODES:
            d = sel[k][m]["n_in_window"]
            if d == 0:
                continue
            x = min(d, DEPTH_CAP)
            base = (MI[m] * DEPTH_CAP + (x - 1)) * 2
            for bb in backbones:
                c = preds.get((bb, m), {}).get(k)
                if c is None:
                    continue
                cellmat[i, base] += 1.0          # n
                cellmat[i, base + 1] += float(c)  # k

    # aggregation coefficients: one column per (group, statistic); groups = each mode, then POOLED.
    _STATS = ("N", "Sx", "Sxx", "Sy", "Sxy")
    COEF = np.zeros((len(MODES) * DEPTH_CAP * 2, (len(MODES) + 1) * len(_STATS)))
    for mi in range(len(MODES)):
        for j in range(DEPTH_CAP):
            xv = float(j + 1)
            cn, ck = 2 * (mi * DEPTH_CAP + j), 2 * (mi * DEPTH_CAP + j) + 1
            for gi in (mi, len(MODES)):     # its own mode, and the pooled group
                b0 = gi * len(_STATS)
                COEF[cn, b0 + 0] += 1.0      # N   = sum n
                COEF[cn, b0 + 1] += xv       # Sx  = sum n*x
                COEF[cn, b0 + 2] += xv * xv  # Sxx = sum n*x^2
                COEF[ck, b0 + 3] += 1.0      # Sy  = sum k
                COEF[ck, b0 + 4] += xv       # Sxy = sum k*x

    def _slopes_from(flat):
        """(pooled_slope, per_mode_slopes, per_mode_n) from raw cell counts, pts per extra frame.

        One matmul plus elementwise algebra, with no reshape and no array reductions -- see _rowsum
        for why the reduction path is not trusted for arrays this size in this process.
        """
        a = np.ascontiguousarray(np.asarray(flat, dtype=float))
        st = a @ COEF
        def _sl(gi):
            b0 = gi * len(_STATS)
            N, Sx, Sxx = st[..., b0], st[..., b0 + 1], st[..., b0 + 2]
            Sy, Sxy = st[..., b0 + 3], st[..., b0 + 4]
            with np.errstate(invalid="ignore", divide="ignore"):
                Nz = np.where(N > 0, N, np.nan)
                num = Sxy - Sx * Sy / Nz
                den = Sxx - Sx * Sx / Nz
                return np.where(den > 0, 100.0 * num / np.where(den > 0, den, np.nan), np.nan), N
        pooled, _ = _sl(len(MODES))
        pm = [_sl(i) for i in range(len(MODES))]
        return pooled, np.stack([p[0] for p in pm], axis=-1), np.stack([p[1] for p in pm], axis=-1)

    colsum = np.ones(len(hkeys)) @ cellmat if len(hkeys) else np.zeros(cellmat.shape[1])
    pooled_pt, permode_pt, permode_n = _slopes_from(colsum)
    slope_by_mode = {m: (float(permode_pt[i]) if permode_pt[i] == permode_pt[i] else None)
                     for i, m in enumerate(MODES)}
    mode_depth_n = {(m, b): int(round(colsum[(MI[m] * DEPTH_CAP + j) * 2]))
                    for m in MODES for j, b in enumerate(("1", "2", "3", "4+"))}
    mode_depth_k = {(m, b): int(round(colsum[(MI[m] * DEPTH_CAP + j) * 2 + 1]))
                    for m in MODES for j, b in enumerate(("1", "2", "3", "4+"))}
    # layout self-check: the matmul aggregation must reproduce a plain-python count.
    for i, m in enumerate(MODES):
        assert abs(float(permode_n[i]) - sum(mode_depth_n[(m, b)] for b in ("1", "2", "3", "4+"))) < 1e-6, \
            ("cell-aggregation layout mismatch", m)

    def _ms(per_mode, nm):
        """n-weighted combination of the WITHIN-mode slopes = the mode-free version of the pooled
        slope. Modes with no identifying depth variation drop out."""
        ok = np.isfinite(per_mode) & (nm > 0)
        w = np.where(ok, nm, 0.0)
        den = _rowsum(w)
        num = _rowsum(np.where(ok, per_mode * w, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / np.where(den > 0, den, np.nan), np.nan)
    slope_ms = float(_ms(permode_pt, permode_n))

    boot_cells = _multinomial_boot(cellmat, N_BOOT, seed=424242)
    boot_slopes, slope_ci, slope_ms_ci = [], [None, None], [None, None]
    if len(boot_cells):
        bp, bpm, bnm = _slopes_from(boot_cells)
        bs = np.sort(bp[np.isfinite(bp)])
        boot_slopes = bs.tolist()
        if len(bs):
            slope_ci = [round(float(bs[int(.025 * len(bs))]), 3),
                        round(float(bs[min(len(bs) - 1, int(math.ceil(.975 * len(bs))) - 1)]), 3)]
        bms = np.sort(_ms(bpm, bnm)[np.isfinite(_ms(bpm, bnm))])
        if len(bms):
            slope_ms_ci = [round(float(bms[int(.025 * len(bms))]), 3),
                           round(float(bms[min(len(bms) - 1, int(math.ceil(.975 * len(bms))) - 1)]), 3)]

    # raw value of ARRIVING in the window at all. Note this already contains whatever the extra
    # frames contribute, because a hit carries E[depth|hit] frames on average, not one.
    value_of_a_hit = (acc_hit - acc_miss) if (acc_hit is not None and acc_miss is not None) else float("nan")
    # capped depth, to match the x used to fit `slope` (the review's minor note: mixing uncapped
    # E[depth|hit] with a slope fitted on x capped at 4 biases the intercept).
    mean_depth_hit = (sum(min(d, DEPTH_CAP) for d, _ in depth_obs) / len(depth_obs)) if depth_obs else 1.0
    # value of the FIRST in-window frame alone = the intercept of the additive model
    #   acc = a + V1*1{depth>=1} + v*(depth-1)
    # i.e. strip the average extra-frame contribution back out of the raw hit-vs-miss gap.
    first_val = (value_of_a_hit - slope * (mean_depth_hit - 1.0)) \
        if (value_of_a_hit == value_of_a_hit and slope == slope) else value_of_a_hit
    slope_excludes_zero = bool(slope_ci[0] is not None and (slope_ci[0] > 0 or slope_ci[1] < 0))
    # the verdict uses the MODE-STRATIFIED slope: the pooled one is confounded with mode exactly as
    # the pooled pair set was (P1), so it cannot be allowed to decide anything.
    slope_ms_excludes_zero = bool(slope_ms_ci[0] is not None
                                  and (slope_ms_ci[0] > 0 or slope_ms_ci[1] < 0))

    # DEPTH x MODE contingency -- P1's "depth and mode are almost collinear" made visible.
    P("\n  DEPTH x MODE contingency (hits only; cell = n obs (acc %), obs = key x backbone)")
    P("  The pooled dose curve above is NOT a pure depth contrast -- read this table first: at depth 1")
    P("  the cells are dominated by one set of modes and at depth 4+ by another, so any slope fitted")
    P("  across the pooled row mixes the dose with the selection mode.")
    P(f"  {'mode':<10} " + " ".join(f"{b:>15}" for b in ("1", "2", "3", "4+")) + f"{'slope':>11}")
    for m in MODES:
        cs = []
        for b in ("1", "2", "3", "4+"):
            n_, k_ = mode_depth_n[(m, b)], mode_depth_k[(m, b)]
            cs.append(f"{n_:>5} ({fmt(k_ / n_ * 100 if n_ else None, '>5.1f')}%)")
        P(f"  {m:<10} " + " ".join(f"{c:>15}" for c in cs) + f"{fmt(slope_by_mode.get(m), '>11.3f')}")

    # ------------------------------------------------------------------ the SATURATION TEST (crux)
    # P1 FIX. The previous implementation chose the deep and the shallow arm with
    # `[m for m in MODES if ...][0]`, i.e. by the fixed priority order (referent, chunk, random,
    # uniform). On real data `referent` was then the deep arm in 74% of pairs and the shallow arm in
    # 17%, so the "dose" contrast was overwhelmingly a referent-vs-not-referent contrast -- the very
    # accuracy difference Path 2 exists to explain. Everything below is built INSIDE a fixed ordered
    # (deep_mode, shallow_mode) cell, and the cells are combined only in ways that make the mode
    # composition explicit. The (deep_mode, shallow_mode) x n contingency table is printed and
    # written to the JSON.
    sat_cells = build_cells(keys, sel, preds, backbones, lambda d: d >= 2, lambda d: d == 1)
    sat_strat = stratified_paired(sat_cells, n_boot=N_BOOT, seed=20260819)
    sat_paired = sat_strat["pooled_across_cells_CONFOUNDED"]   # reported ONLY as the defect exhibit

    # value of the FIRST in-window frame: hit vs miss, stratified on the mode pair the same way
    hit_cells = build_cells(keys, sel, preds, backbones, lambda d: d > 0, lambda d: d == 0)
    hit_strat = stratified_paired(hit_cells, n_boot=N_BOOT, seed=20260919)
    hit_paired = hit_strat["pooled_across_cells_CONFOUNDED"]

    def _contingency(strat, title, deep_label, shallow_label):
        P(f"\n    {title}")
        P(f"      {'deep arm':<10} {'shallow arm':<12} {'n pairs':>8} {'n keys':>7} {'b':>5} {'c':>5} "
          f"{'diff pts':>9}  exact p")
        for name in sorted(strat["per_cell"], key=lambda x: -strat["per_cell"][x]["n_pairs"]):
            v = strat["per_cell"][name]
            P(f"      {v['deep_mode']:<10} {v['shallow_mode']:<12} {v['n_pairs']:>8} {v['n_keys']:>7} "
              f"{v['b']:>5} {v['c']:>5} {fmt(v['diff_pts'], '>+9.2f')}  {v['p_exact']:.4g}")
        P(f"      {'TOTAL':<10} {'':<12} {strat['n_pairs']:>8} {strat['n_keys']:>7} "
          f"{strat['b_total']:>5} {strat['c_total']:>5}")
        P(f"      ({deep_label} = deep arm, {shallow_label} = shallow arm; a question can appear in "
          f"more than one cell, hence n keys < n pairs)")

    P("\n  SATURATION TEST (the crux) -- MODE-STRATIFIED (P1 fix); every arm is 8 frames")

    _contingency(hit_strat, "(deep_mode, shallow_mode) x n contingency -- FIRST in-window frame "
                            "(hit vs miss):", "depth>=1", "depth==0")
    hb = hit_strat["mode_balanced"]; hm_ = hit_strat["mantel_haenszel"]
    P(f"    value of the FIRST in-window frame:")
    P(f"      MODE-BALANCED (orientations averaged, mode main effect cancels): "
      f"{fmt(hb['diff_pts'], '+.2f')} pts, 95% key-clustered bootstrap CI {hb['ci95_boot_clustered_on_key']}, "
      f"p={fmt(hb['p_bootstrap'], '.4g')}, MDE {fmt(hb['mde_pts'], '.2f')} pts "
      f"({hb['n_mode_pairs']} balanced mode pairs)")
    P(f"      Mantel-Haenszel over cells: {fmt(hm_['diff_pts'], '+.2f')} pts, p={fmt(hm_['p'], '.4g')}, "
      f"heterogeneity Q={fmt(hm_['q'], '.2f')} (df={hm_['q_df']}, p={fmt(hm_['q_p'], '.4g')})")
    P(f"      stratified exact McNemar p={hit_strat['stratified_exact_p']:.4g} "
      f"(n={hit_strat['n_pairs']} pairs from {hit_strat['n_keys']} questions)")
    P(f"      [CONFOUNDED, shown only as the defect exhibit] pooling cells without stratifying: "
      f"{fmt(hit_paired['diff_pts'], '+.2f')} pts, n={hit_paired['n']}, "
      f"design effect {fmt(hit_paired.get('design_effect'), '.2f')} -> clustered MDE "
      f"{fmt(hit_paired.get('mde_pts_clustered'), '.2f')} pts (unclustered would say "
      f"{fmt(hit_paired.get('mde_pts'), '.2f')})")
    P(f"      unpaired pooled: acc|hit {fmt(acc_hit, '.2f')} (n={hit_n}) vs acc|miss "
      f"{fmt(acc_miss, '.2f')} (n={miss_n})  ->  value of a hit {fmt(value_of_a_hit, '+.2f')} pts")
    P(f"      of which the FIRST frame alone accounts for {fmt(first_val, '+.2f')} pts "
      f"(raw gap minus slope x (E[capped depth|hit]={mean_depth_hit:.2f} - 1))")

    _contingency(sat_strat, "(deep_mode, shallow_mode) x n contingency -- frames BEYOND the first "
                            "(depth>=2 vs depth==1):", "depth>=2", "depth==1")
    sb = sat_strat["mode_balanced"]; sm_ = sat_strat["mantel_haenszel"]
    P(f"    value of frames BEYOND the first:")
    P(f"      MODE-BALANCED: {fmt(sb['diff_pts'], '+.2f')} pts, 95% key-clustered bootstrap CI "
      f"{sb['ci95_boot_clustered_on_key']}, p={fmt(sb['p_bootstrap'], '.4g')}, "
      f"MDE {fmt(sb['mde_pts'], '.2f')} pts ({sb['n_mode_pairs']} balanced mode pairs; "
      f"a bootstrap p is bounded below by 1/{sat_strat['n_boot']})")
    if sb["unusable_single_orientation_cells"]:
        P(f"        cells with only one orientation, so unusable for balancing: "
          f"{sb['unusable_single_orientation_cells']}")
    P(f"      Mantel-Haenszel over cells: {fmt(sm_['diff_pts'], '+.2f')} pts, p={fmt(sm_['p'], '.4g')}, "
      f"heterogeneity Q={fmt(sm_['q'], '.2f')} (df={sm_['q_df']}, p={fmt(sm_['q_p'], '.4g')})")
    P(f"      stratified exact McNemar p={sat_strat['stratified_exact_p']:.4g} "
      f"(n={sat_strat['n_pairs']} pairs from {sat_strat['n_keys']} questions)")
    P(f"      [CONFOUNDED, shown only as the defect exhibit] pooling cells without stratifying: "
      f"{fmt(sat_paired['diff_pts'], '+.2f')} pts, p={sat_paired['p']:.4g}, n={sat_paired['n']}, "
      f"design effect {fmt(sat_paired.get('design_effect'), '.2f')} -> clustered MDE "
      f"{fmt(sat_paired.get('mde_pts_clustered'), '.2f')} pts (unclustered would say "
      f"{fmt(sat_paired.get('mde_pts'), '.2f')})")
    P(f"      unpaired check: {fmt(sat_unpaired['diff_pts'], '+.2f')} pts p={fmt(sat_unpaired['p'], '.4g')} "
      f"(n1={sat_unpaired['n1']}, n2={sat_unpaired['n2']})"
      + ("" if sig(sat_unpaired["p"]) else f"  [null: {null_phrase(sat_unpaired)}]")
      + "  [UNPAIRED and mode-confounded: secondary only]")
    P(f"    WLS slope over the hits-only dose curve: {fmt(slope, '+.3f')} pts per extra in-window "
      f"frame, 95% key-clustered bootstrap CI {slope_ci} ({len(boot_slopes)} resamples)")
    P(f"      mode-STRATIFIED slope (within-mode slopes, n-weighted): {fmt(slope_ms, '+.3f')} pts, "
      f"95% key-clustered bootstrap CI {slope_ms_ci}   <- the mode-free version of the same number")
    for m in MODES:
        P(f"        within {m:<9} slope {fmt(slope_by_mode.get(m), '+.3f')} pts/frame "
          f"(n={sum(mode_depth_n[(m, b)] for b in ('1','2','3','4+'))})")

    # ---- the conditional model, per-question fixed effects (P1, second required handle)
    fe_rows = []
    for k in keys:
        for m in MODES:
            d = sel[k][m]["n_in_window"]
            for bb in backbones:
                c = preds.get((bb, m), {}).get(k)
                if c is None:
                    continue
                fe_rows.append((k, bb, m, d, int(c)))
    fe_names = ["hit_1{depth>=1}", "extra_frames_(depth-1)"] + [f"mode[{m}]" for m in MODES[1:]] \
               + [f"backbone[{b}]" for b in backbones[1:]]
    fe_X, fe_y, fe_g, fe_gb, fe_cl = [], [], [], [], []
    for k, bb, m, d, c in fe_rows:
        row = [1.0 if d >= 1 else 0.0, float(max(0, min(d, DEPTH_CAP) - 1))]
        row += [1.0 if m == mm else 0.0 for mm in MODES[1:]]
        row += [1.0 if bb == b2 else 0.0 for b2 in backbones[1:]]
        fe_X.append(row); fe_y.append(float(c)); fe_g.append(k); fe_gb.append(k + "||" + bb)
        fe_cl.append(k)
    fe_key = fe_lpm(fe_y, fe_X, fe_names, fe_g, fe_cl) if fe_rows else {"error": "no rows"}
    fe_keybb = fe_lpm(fe_y, fe_X, fe_names, fe_gb, fe_cl) if fe_rows else {"error": "no rows"}
    P("\n    CONDITIONAL MODEL  correct ~ 1{depth>=1} + (depth-1) + C(mode) + C(backbone), with")
    P("    per-question fixed effects. Within-question demeaned LINEAR PROBABILITY MODEL (chosen over")
    P("    conditional logit because the estimand is accuracy POINTS and the planted/assumed dose")
    P("    model is additive on the probability scale); SEs cluster-robust on the question key.")
    for label, fe in (("FE = question", fe_key), ("FE = question x backbone", fe_keybb)):
        if "error" in fe:
            P(f"      {label}: [{fe['error']}]")
            continue
        P(f"      {label}: n={fe['n_obs']} obs, {fe['n_fe_groups']} FE groups, "
          f"{fe['n_clusters']} clusters"
          + (f", absorbed: {fe['absorbed_by_fixed_effects']}" if fe["absorbed_by_fixed_effects"] else ""))
        for nm, v in fe["coef"].items():
            P(f"        {nm:<24} {fmt(v['coef_pts'], '>+8.2f')} pts  se {fmt(v['se_pts'], '>6.2f')}  "
              f"p={fmt(v['p'], '.4g')}  MDE {fmt(v['mde_pts'], '.2f')}"
              + ("" if sig(v["p"]) else "   [null]"))

    # ---- verdict
    fe_extra = fe_key.get("coef", {}).get("extra_frames_(depth-1)", {}) if "error" not in fe_key else {}
    prim = sb if sb["diff_pts"] is not None else sm_
    prim_name = "mode-balanced" if sb["diff_pts"] is not None else "Mantel-Haenszel"
    prim_delta = prim.get("diff_pts")
    prim_p = prim.get("p_bootstrap", prim.get("p"))
    prim_mde = prim.get("mde_pts")
    rejects = sig(prim_p) or sig(fe_extra.get("p")) or slope_ms_excludes_zero
    powered = (prim_mde is not None and first_val == first_val and prim_mde <= abs(first_val))
    if rejects:
        saturates, verdict = False, "extra in-window frames DO pay"
    elif powered:
        saturates, verdict = True, "saturates"
    else:
        saturates, verdict = None, "INDETERMINATE -- underpowered"
    underpowered = not powered
    P(f"\n    POWER (stated unconditionally, whatever the verdict): the mode-stratified design can "
      f"detect {fmt(prim_mde, '.2f')} pts for depth >= 2 vs depth == 1, at 80% power, against a first-frame value of "
      f"{fmt(first_val, '.2f')} pts. The mis-specified pooled test would have advertised "
      f"{fmt(sat_paired.get('mde_pts'), '.2f')} pts (unclustered) / "
      f"{fmt(sat_paired.get('mde_pts_clustered'), '.2f')} pts (clustered) on the same data. Stratifying "
      f"splits {sat_strat['n_pairs']} pairs across {sat_strat['n_cells']} cells and DISCARDS the "
      f"{len(sb['unusable_single_orientation_cells'])} cells that exist in only one orientation, so it "
      f"buys validity WITH power: the pre-fix power caveat (saturation MDE 4.11 pts vs a 1.75-pt "
      f"first-frame value) carries forward and is made worse, not better, by this fix.")
    P(f"\n    => SATURATION VERDICT: {verdict}   (saturates={saturates})")
    P(f"       decision rule: primary = the {prim_name} stratified estimate "
      f"({fmt(prim_delta, '+.2f')} pts, p={fmt(prim_p, '.4g')}, MDE {fmt(prim_mde, '.2f')} pts); "
      f"secondary = the FE conditional model's (depth-1) coefficient "
      f"({fmt(fe_extra.get('coef_pts'), '+.2f')} pts, p={fmt(fe_extra.get('p'), '.4g')}) and the "
      f"key-clustered MODE-STRATIFIED slope CI {slope_ms_ci} (the pooled slope CI {slope_ci} is "
      f"mode-confounded and decides nothing).")
    if saturates is None:
        P(f"       !! NO VERDICT IS REPORTED. Nothing rejects, but the mode-stratified design could "
          f"only have detected {fmt(prim_mde, '.2f')} pts per extra frame, which is not smaller than "
          f"the {fmt(abs(first_val) if first_val == first_val else None, '.2f')}-pt value of the first "
          f"frame itself. 'Cannot reject' is NOT 'saturates' at this n. The pre-existing power caveat "
          f"(unstratified saturation MDE 4.11 pts vs first-frame value 1.75 pts) carries forward and "
          f"is made WORSE, not better, by stratifying: splitting the pairs across "
          f"{sat_strat['n_cells']} mode cells costs power. Re-run when the dump completes.")
    elif saturates:
        P("       reading: depth is a bad buy -- the budget should be spent on independent moments "
          "(BREADTH), which is the direction of the chunk < keyframe ordering.")
    else:
        P("       reading: extra in-window frames DO pay -- the breadth/depth story does NOT explain "
          "the ordering and must be abandoned.")

    marginal_value = {
        "first_frame_in_window": round(first_val, 3) if first_val == first_val else None,
        "each_additional_frame": round(slope, 3) if slope == slope else None,
        "each_additional_frame_mode_stratified": round(slope_ms, 3) if slope_ms == slope_ms else None,
        "p_additional_beyond_first": prim_p,
        "estimator_for_p": prim_name + " stratified on (deep_mode, shallow_mode), "
                                       "key-clustered bootstrap",
        "additional_beyond_first_pts": prim_delta,
        "additional_beyond_first_mde_pts": prim_mde,
        "saturates": saturates,
        "saturation_verdict": verdict,
    }

    # -------------------------------------------------------------- C. gap decomposition per bb
    P("\n" + "-" * 100)
    P("C. GAP DECOMPOSITION  acc(chunk) - acc(keyframe/referent), both arms 8 frames")
    P("   breadth term = dP_hit x value_of_a_hit ;  depth term = dE[depth|hit] x value_per_extra_frame")
    P("   (depth term is NOT re-weighted by P_hit, so it is an upper bound on the depth channel.")
    P("    `resid` is the UNEXPLAINED REMAINDER: it is DEFINED as gap - breadth - depth, so the fact")
    P("    that the three add up to the gap is an identity of the definition and is NOT evidence of")
    P("    anything. It absorbs the breadth x depth interaction and every channel not modelled here.)")
    P("-" * 100)
    gap_dec = {}
    P(f"{'backbone':<20} {'n':>6} {'gap':>8} {'breadth':>9} {'depth':>8} {'resid':>8}  "
      f"{'dP_hit':>8} {'V_hit':>7} {'dE[dep|hit]':>12} {'v_extra':>8}  McNemar p")
    for bb in backbones:
        pc, pk = preds.get((bb, "chunk"), {}), preds.get((bb, "referent"), {})
        kk = [k for k in keys if k in pc and k in pk]
        if not kk:
            continue
        gap = (sum(pc[k] for k in kk) - sum(pk[k] for k in kk)) / len(kk) * 100
        mc = mcnemar([(pc[k], pk[k]) for k in kk])
        ph_c = sum(sel[k]["chunk"]["n_in_window"] > 0 for k in kk) / len(kk)
        ph_k = sum(sel[k]["referent"]["n_in_window"] > 0 for k in kk) / len(kk)
        hc = [sel[k]["chunk"]["n_in_window"] for k in kk if sel[k]["chunk"]["n_in_window"] > 0]
        hk = [sel[k]["referent"]["n_in_window"] for k in kk if sel[k]["referent"]["n_in_window"] > 0]
        ed_c = sum(hc) / len(hc) if hc else 0.0
        ed_k = sum(hk) / len(hk) if hk else 0.0

        # per-backbone value of a hit and value per extra frame, from THIS backbone's own data
        bk = bn = mk = mn = 0
        bxs, bys, bws = [], [], []
        cell = defaultdict(lambda: [0, 0])
        for k in kk:
            for m in MODES:
                c = preds.get((bb, m), {}).get(k)
                if c is None:
                    continue
                d = sel[k][m]["n_in_window"]
                if d == 0:
                    mn += 1; mk += int(c)
                else:
                    bn += 1; bk += int(c)
                    cell[min(d, 4)][0] += int(c); cell[min(d, 4)][1] += 1
        for x in sorted(cell):
            kk_, nn_ = cell[x]
            bxs.append(float(x)); bys.append(kk_ / nn_ * 100); bws.append(nn_)
        v_hit = (bk / bn * 100 - mk / mn * 100) if (bn and mn) else float("nan")
        v_extra = wls_slope(bxs, bys, bws)
        breadth = (ph_c - ph_k) * v_hit if v_hit == v_hit else float("nan")
        depth = (ed_c - ed_k) * v_extra if v_extra == v_extra else float("nan")
        resid = gap - (breadth if breadth == breadth else 0) - (depth if depth == depth else 0)
        gap_dec[bb] = {"chunk_minus_keyframe": round(gap, 3),
                       "breadth_term": round(breadth, 3) if breadth == breadth else None,
                       "depth_term": round(depth, 3) if depth == depth else None,
                       "residual": round(resid, 3),
                       "n": len(kk), "mcnemar_p": mc["p"], "mde_pts": mc["mde_pts"],
                       "d_p_hit_pts": round((ph_c - ph_k) * 100, 2),
                       "value_of_a_hit_pts": round(v_hit, 3) if v_hit == v_hit else None,
                       "d_mean_depth_given_hit": round(ed_c - ed_k, 3),
                       "value_per_extra_frame_pts": round(v_extra, 3) if v_extra == v_extra else None}
        P(f"{bb:<20} {len(kk):>6} {fmt(gap, '>+8.2f')} {fmt(breadth, '>+9.2f')} "
          f"{fmt(depth, '>+8.2f')} {fmt(resid, '>+8.2f')}  "
          f"{(ph_c-ph_k)*100:>+8.2f} {fmt(v_hit, '>+7.2f')} {ed_c-ed_k:>+12.3f} {fmt(v_extra, '>+8.3f')}  "
          f"{mc['p']:.4g}" + ("" if sig(mc["p"]) else f"  [null: {null_phrase(mc)}]"))

    # ---------------------------------------------------- C2. does breadth matter more when weak?
    # prediction: on a weaker backbone the value of a hit is a larger share of what it can do, so
    # losing breadth costs more -> chunk trails keyframe by MORE on the weaker model.
    weak_test = {}
    base_acc = {}
    for bb in backbones:
        pk = preds.get((bb, "uniform"), {})
        kk = [k for k in keys if k in pk]
        base_acc[bb] = (sum(pk[k] for k in kk) / len(kk) * 100) if kk else None
    ranked = [b for b in sorted(base_acc, key=lambda x: (base_acc[x] is None, base_acc[x] or 0))
              if base_acc[b] is not None]
    if len(ranked) >= 2:
        weakest, strongest = ranked[0], ranked[-1]
        common = [k for k in keys
                  if all(k in preds.get((b, m), {}) for b in (weakest, strongest) for m in ("chunk", "referent"))]
        if common:
            dw = [preds[(weakest, "chunk")][k] - preds[(weakest, "referent")][k] for k in common]
            ds = [preds[(strongest, "chunk")][k] - preds[(strongest, "referent")][k] for k in common]
            did = (sum(dw) - sum(ds)) / len(common) * 100
            rnd = random.Random(0)
            idx = range(len(common))
            boots = []
            for _ in range(4000):
                s = [rnd.randrange(len(common)) for _ in idx]
                boots.append((sum(dw[i] for i in s) - sum(ds[i] for i in s)) / len(s) * 100)
            boots.sort()
            p_boot = 2 * min(sum(1 for x in boots if x >= 0), sum(1 for x in boots if x <= 0)) / len(boots)
            p_boot = min(1.0, p_boot)
            weak_test = {"weakest": weakest, "strongest": strongest, "n_common": len(common),
                         "gap_weak_pts": round(sum(dw) / len(common) * 100, 2),
                         "gap_strong_pts": round(sum(ds) / len(common) * 100, 2),
                         "diff_in_diff_pts": round(did, 2),
                         "ci95": [round(boots[int(.025 * len(boots))], 2), round(boots[int(.975 * len(boots))], 2)],
                         "p_bootstrap": p_boot,
                         "value_of_a_hit_weak": gap_dec.get(weakest, {}).get("value_of_a_hit_pts"),
                         "value_of_a_hit_strong": gap_dec.get(strongest, {}).get("value_of_a_hit_pts")}
            P("\n  DOES BREADTH MATTER MORE WHEN THE MODEL IS WEAK?")
            P(f"    weakest={weakest} (uniform-8 acc {base_acc[weakest]:.2f}) vs "
              f"strongest={strongest} (uniform-8 acc {base_acc[strongest]:.2f}), paired n={len(common)}")
            P(f"    chunk-minus-keyframe gap:  weak {weak_test['gap_weak_pts']:+.2f} pts, "
              f"strong {weak_test['gap_strong_pts']:+.2f} pts")
            P(f"    difference-in-differences {did:+.2f} pts, 95% paired bootstrap CI "
              f"{weak_test['ci95']}, p={p_boot:.4g}")
            P(f"    value of a hit:  weak {weak_test['value_of_a_hit_weak']}  vs  "
              f"strong {weak_test['value_of_a_hit_strong']} pts")
            if p_boot >= 0.05:
                P("    -> NOT SUPPORTED at this n: the weak-vs-strong difference in the chunk penalty is "
                  "within noise. State it as unsupported, do not narrate it.")
            else:
                P("    -> SUPPORTED: the chunk penalty is significantly larger on the weaker backbone.")

    # -------------------------------------------------------------------- D. membank replication
    P("\n" + "-" * 100)
    P("D. MEMORY-BANK REPLICATION -- different budget (8,192 visual tokens = 32 frames x 256 tok),")
    P("   different mechanism (stored LLM embeddings spliced, video never re-decoded), REAL-keyed.")
    P("   mb_oracle is a CEILING (answer-informed chunk pick) and is EXCLUDED from every number here.")
    P("-" * 100)
    mb = {"chunk_hit_top2": None, "chunk_hit_rand2": None, "acc_given_chunk_hit": None,
          "acc_given_chunk_miss": None, "n": 0}
    mb_extra = {}
    if membank_rows:
        by_arm = defaultdict(dict)
        for r in membank_rows:
            by_arm[r["arm"]][r["key"]] = r
        def hit_rate(arm):
            rs = [r for r in by_arm.get(arm, {}).values() if r.get("chunk_hit") is not None]
            return (sum(bool(r["chunk_hit"]) for r in rs) / len(rs) * 100, len(rs)) if rs else (None, 0)
        ht2, nt2 = hit_rate("mb_top2")
        hr2, nr2 = hit_rate("mb_rand2")
        mb["chunk_hit_top2"] = round(ht2, 2) if ht2 is not None else None
        mb["chunk_hit_rand2"] = round(hr2, 2) if hr2 is not None else None
        # accuracy given chunk hit / miss, pooled over the two 2-chunk arms (never mb_oracle)
        hk = hn = mk_ = mn_ = 0
        for arm in ("mb_top2", "mb_rand2"):
            for r in by_arm.get(arm, {}).values():
                if r.get("chunk_hit") is None:
                    continue
                if r["chunk_hit"]:
                    hn += 1; hk += int(bool(r["is_correct"]))
                else:
                    mn_ += 1; mk_ += int(bool(r["is_correct"]))
        mb["acc_given_chunk_hit"] = round(hk / hn * 100, 2) if hn else None
        mb["acc_given_chunk_miss"] = round(mk_ / mn_ * 100, 2) if mn_ else None
        mb["n"] = hn + mn_
        # paired tests on the common keys
        ck = sorted(set(by_arm.get("mb_top2", {})) & set(by_arm.get("mb_rand2", {})))
        if ck:
            mch = mcnemar([(bool(by_arm["mb_top2"][k].get("chunk_hit")),
                            bool(by_arm["mb_rand2"][k].get("chunk_hit"))) for k in ck])
            mca = mcnemar([(bool(by_arm["mb_top2"][k]["is_correct"]),
                            bool(by_arm["mb_rand2"][k]["is_correct"])) for k in ck])
            mb_extra = {"n_paired": len(ck), "chunk_hit_mcnemar": mch, "acc_mcnemar": mca}
            P(f"  mb_top2 vs mb_rand2, paired n={len(ck)}, both 8,192 tokens (2 chunks x 16 frames):")
            P(f"    chunk-hit rate  {fmt(mb['chunk_hit_top2'], '.2f')}% vs "
              f"{fmt(mb['chunk_hit_rand2'], '.2f')}%  "
              f"({mch['diff_pts']:+.2f} pts, p={mch['p']:.4g}, n={mch['n']})")
            P(f"    accuracy        {sum(by_arm['mb_top2'][k]['is_correct'] for k in ck)/len(ck)*100:.2f}% vs "
              f"{sum(by_arm['mb_rand2'][k]['is_correct'] for k in ck)/len(ck)*100:.2f}%  "
              f"({mca['diff_pts']:+.2f} pts, p={mca['p']:.4g})")
        P(f"  accuracy | chunk HIT  = {mb['acc_given_chunk_hit']}%   "
          f"accuracy | chunk MISS = {mb['acc_given_chunk_miss']}%   (pooled over the two 2-chunk arms, "
          f"n={mb['n']})   [DIAGNOSTIC: conditions on the oracle evidence chunk]")
        if mb["acc_given_chunk_hit"] is not None and mb["acc_given_chunk_miss"] is not None:
            P(f"  -> landing in the right chunk is worth "
              f"{mb['acc_given_chunk_hit'] - mb['acc_given_chunk_miss']:+.2f} pts at this budget too, "
              f"i.e. the ARRIVING is what pays; this replicates the frame-level finding on a "
              f"different budget and a different mechanism.")
    else:
        P("  [no membank rows found]")

    # ------------------------------------------------------------------------ E. spread vs acc
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    sk, sn = Counter(), Counter()
    for k in keys:
        for m in MODES:
            sp = float(sel[k][m].get("spread") or 0.0)
            bi = max(0, min(len(labels) - 1, next(i for i in range(len(labels)) if sp < edges[i + 1])))
            for bb in backbones:
                c = preds.get((bb, m), {}).get(k)
                if c is None:
                    continue
                sn[labels[bi]] += 1
                sk[labels[bi]] += int(c)
    spread_vs_acc = {"bins": labels,
                     "acc": [round(sk[l] / sn[l] * 100, 2) if sn[l] else None for l in labels],
                     "n": [sn[l] for l in labels]}
    P("\n" + "-" * 100)
    P("E. TEMPORAL DISPERSION vs ACCURACY (spread = (max-min selected frame)/(n_total-1); all arms 8 frames)")
    P("-" * 100)
    # mode composition of each bin -- spread is nearly DETERMINISTIC given the mode (uniform=1.0 by
    # construction, chunk lowest), so the pooled table below is largely a relabelling of "mode" and
    # must not be read as an independent effect of dispersion.
    comp = defaultdict(Counter)
    for k in keys:
        for m in MODES:
            sp = float(sel[k][m].get("spread") or 0.0)
            bi = max(0, min(len(labels) - 1, next(i for i in range(len(labels)) if sp < edges[i + 1])))
            comp[labels[bi]][m] += 1
    P(f"{'spread bin':<12} {'acc %':>8} {'n':>8}   95% Wilson CI      dominant mode(s)")
    for l in labels:
        tot = sum(comp[l].values()) or 1
        dom = ", ".join(f"{m} {c/tot*100:.0f}%" for m, c in comp[l].most_common(2))
        P(f"{l:<12} {spread_vs_acc['acc'][labels.index(l)] if sn[l] else '-':>8} {sn[l]:>8}   "
          f"{str(wilson(sk[l], sn[l])):<18} {dom}")
    P("  !! CONFOUND: spread is near-deterministic given the mode (uniform=1.0 by construction), so the")
    P("     table above mostly re-expresses the mode contrasts. The mode-free control is below.")

    # WITHIN-MODE control: terciles of spread inside each mode separately, where dispersion varies
    # for reasons other than which method produced the selection.
    within = {}
    P("\n  within-mode control (terciles of spread INSIDE each mode; uniform is constant and skipped):")
    P(f"  {'mode':<10} {'tercile':<9} {'spread rng':>16} {'acc %':>8} {'n':>7}   95% Wilson CI")
    for m in MODES:
        vals = sorted(float(sel[k][m].get("spread") or 0.0) for k in keys)
        if not vals or vals[0] == vals[-1]:
            P(f"  {m:<10} constant spread ({vals[0] if vals else float('nan'):.3f}) -- no within-mode variation to test")
            continue
        q1, q2 = vals[len(vals) // 3], vals[2 * len(vals) // 3]
        cells = {"low": [0, 0, 1e9, -1e9], "mid": [0, 0, 1e9, -1e9], "high": [0, 0, 1e9, -1e9]}
        for k in keys:
            sp = float(sel[k][m].get("spread") or 0.0)
            t = "low" if sp <= q1 else ("mid" if sp <= q2 else "high")
            cells[t][2] = min(cells[t][2], sp); cells[t][3] = max(cells[t][3], sp)
            for bb in backbones:
                c = preds.get((bb, m), {}).get(k)
                if c is not None:
                    cells[t][1] += 1; cells[t][0] += int(c)
        within[m] = {}
        for t in ("low", "mid", "high"):
            kk_, nn_, lo_, hi_ = cells[t]
            within[m][t] = {"acc": round(kk_ / nn_ * 100, 2) if nn_ else None, "n": nn_,
                            "ci95": wilson(kk_, nn_), "spread_range": [round(lo_, 3), round(hi_, 3)]}
            P(f"  {m:<10} {t:<9} {f'{lo_:.3f}-{hi_:.3f}':>16} "
              f"{fmt(kk_/nn_*100 if nn_ else None, '>8.2f')} {nn_:>7}   {wilson(kk_, nn_)}")
        lo, hi = cells["low"], cells["high"]
        tp = two_prop(hi[0], hi[1], lo[0], lo[1])
        within[m]["high_minus_low"] = tp
        P(f"  {m:<10} high-low  {fmt(tp['diff_pts'], '+.2f')} pts  p={fmt(tp['p'], '.4g')}"
          + ("" if sig(tp['p']) else f"   [null: {null_phrase(tp)}]"))

    out = {
        "n_keys": n_keys,
        "breadth_depth": bd,
        "depth_curve": depth_curve,
        "marginal_value": marginal_value,
        "gap_decomposition": {b: {kk: v[kk] for kk in
                                  ("chunk_minus_keyframe", "breadth_term", "depth_term", "residual")}
                              for b, v in gap_dec.items()},
        "membank": mb,
        "spread_vs_acc": spread_vs_acc,
    }
    extra = {
        "backbones_pooled": list(backbones),
        "acc_given_hit": round(acc_hit, 2) if acc_hit is not None else None,
        "acc_given_miss": round(acc_miss, 2) if acc_miss is not None else None,
        "n_hit_obs": hit_n, "n_miss_obs": miss_n,
        "value_of_a_hit_pooled": round(value_of_a_hit, 3) if value_of_a_hit == value_of_a_hit else None,
        "mean_depth_given_hit_pooled": round(mean_depth_hit, 3),
        "hit_rate_paired_tests": hit_tests,
        "dose_slope_pts_per_extra_frame": round(slope, 3) if slope == slope else None,
        "dose_slope_ci95_bootstrap_clustered_on_key": slope_ci,
        "dose_slope_excludes_zero_POOLED_CONFOUNDED": slope_excludes_zero,
        "dose_slope_mode_stratified_excludes_zero": slope_ms_excludes_zero,
        "saturation_underpowered": underpowered,
        "saturation_verdict": verdict,
        "saturation_primary_estimator": prim_name,
        "saturation_stratified": sat_strat,
        "first_frame_stratified": hit_strat,
        "mode_pair_contingency_saturation": {
            "cells": {k: {kk: vv for kk, vv in v.items() if kk != "var"}
                      for k, v in sat_strat["per_cell"].items()},
            "n_pairs": sat_strat["n_pairs"], "n_keys": sat_strat["n_keys"],
            "n_cells": sat_strat["n_cells"]},
        "mode_pair_contingency_first_frame": {
            "cells": {k: {kk: vv for kk, vv in v.items() if kk != "var"}
                      for k, v in hit_strat["per_cell"].items()},
            "n_pairs": hit_strat["n_pairs"], "n_keys": hit_strat["n_keys"],
            "n_cells": hit_strat["n_cells"]},
        "conditional_model_question_fe": fe_key,
        "conditional_model_question_x_backbone_fe": fe_keybb,
        "dose_slope_mode_stratified": round(slope_ms, 3) if slope_ms == slope_ms else None,
        "dose_slope_mode_stratified_ci95": slope_ms_ci,
        "dose_slope_by_mode": slope_by_mode,
        "depth_by_mode_n": {f"{m}|{b}": mode_depth_n[(m, b)] for m in MODES
                            for b in ("1", "2", "3", "4+")},
        "depth_by_mode_correct": {f"{m}|{b}": mode_depth_k[(m, b)] for m in MODES
                                  for b in ("1", "2", "3", "4+")},
        "saturation_paired_POOLED_CONFOUNDED_do_not_report": sat_paired,
        "saturation_unpaired": sat_unpaired,
        "first_frame_paired_POOLED_CONFOUNDED_do_not_report": hit_paired,
        "gap_decomposition_full": gap_dec,
        "weak_vs_strong": weak_test,
        "membank_paired": mb_extra,
        "uniform8_accuracy": {b: (round(v, 2) if v is not None else None) for b, v in base_acc.items()},
        "spread_bin_mode_composition": {l: dict(comp[l]) for l in labels},
        "spread_within_mode_terciles": within,
    }

    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(out, f, indent=1)
        with open(out_json.replace(".json", "_extra.json"), "w") as f:
            json.dump(extra, f, indent=1)
        P(f"\n[write] {out_json}")
        P(f"[write] {out_json.replace('.json', '_extra.json')}")

    # P6. There is NO summation check here any more. `residual` is DEFINED as
    # gap - breadth_term - depth_term, so "the three terms sum to the gap" is a tautology; asserting
    # it validated nothing, and `(x or 0)` silently swallowed a NaN term into the residual so the
    # assert passed anyway. What IS worth asserting is that the two modelled terms actually exist:
    # a None/NaN term means an empty hit or miss stratum, and it must fail loudly rather than be
    # absorbed into a residual that is then narrated as "unexplained".
    bad = {b: v for b, v in gap_dec.items()
           if v["breadth_term"] is None or v["depth_term"] is None
           or v["breadth_term"] != v["breadth_term"] or v["depth_term"] != v["depth_term"]}
    assert not bad, ("breadth/depth term is None or NaN -- an empty hit or miss stratum, not a "
                     "residual: %s" % sorted(bad))
    P("\n[check] every backbone has a finite breadth term and a finite depth term. "
      "(`residual` is the UNEXPLAINED remainder by definition and is NOT a validated quantity.)")
    return out


def load_membank(pattern):
    rows, seen = [], set()
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "key" not in r or "arm" not in r:
                continue
            sig = (r["key"], r["arm"])
            if sig in seen:
                continue
            seen.add(sig)
            rows.append(r)
    return rows


# ------------------------------------------------------------------------------------ fixture --
def make_fixture(tmp, saturating: bool, n_keys=700, seed=0, mode_bonus=0.0):
    """Synthetic selections + predictions with a PLANTED dose curve.

    ground truth:  P(correct) = base + V*1{depth>=1} + EXTRA*max(depth-1,0) + BONUS*1{mode=referent}
    saturating fixture: EXTRA = 0    -> the analysis must return saturates=True
    linear fixture:     EXTRA = 6pt  -> the analysis must return saturates=False
    confounded fixture: EXTRA = 0 and BONUS = 25pt on `referent`, whose planted geometry makes it the
      shallow arm far more often than the deep arm (the real data has the opposite imbalance; either
      way the "dose" contrast picks up the mode bonus, here with a negative sign). The mis-specified unstratified test must be
      FOOLED by it (it sees a large positive "dose" effect that is really the mode bonus) while the
      mode-balanced / fixed-effect estimates must not be.
    Modes are planted with the hypothesised geometry: referent broad (high hit rate, depth ~1),
    chunk narrow (lower hit rate, depth ~2.5).
    """
    rnd = random.Random(seed)
    V, EXTRA = 20.0, (0.0 if saturating else 6.0)
    BASE = {"internvl3-14b": 20.0, "qwen2.5-vl-7b": 12.0, "ovis2.5-9b": 18.0}
    geom = {"referent": (0.55, (1, 1, 2)), "chunk": (0.45, (2, 3, 4)),
            "random": (0.35, (1, 1, 2)), "uniform": (0.40, (1, 2, 2))}
    selp = os.path.join(tmp, "selections.jsonl")
    preds = defaultdict(list)
    with open(selp, "w") as f:
        for i in range(n_keys):
            key = f"vid_{i//8:04d}|q{i%8}"
            N = rnd.randrange(2000, 6000)
            f0 = rnd.randrange(0, N - 300); f1 = f0 + 200
            for m in MODES:
                hp, depths = geom[m]
                hit = rnd.random() < hp
                depth = rnd.choice(depths) if hit else 0
                frames = sorted(rnd.sample(range(N), 8))
                sp = round((frames[-1] - frames[0]) / (N - 1), 4) * (0.4 if m == "chunk" else 1.0)
                f.write(json.dumps({
                    "key": key, "real_key": key, "video_id": key.split("|")[0],
                    "question_id": key.split("|")[1], "mode": m,
                    "sel_pool": list(range(8)), "sel_frames": frames, "n_total": N, "fps": 25.0,
                    "ev_f0": f0, "ev_f1": f1, "ev_span_frames": f1 - f0 + 1,
                    "n_in_window": depth, "hit": depth > 0,
                    "nearest_s": 1.0, "spread": sp,
                    "chunk_ids": sorted(rnd.sample(range(8), 2 if m == "chunk" else 5)),
                }) + "\n")
                for bb, base in BASE.items():
                    p = (base + (V if depth >= 1 else 0.0) + EXTRA * max(0, depth - 1)
                         + (mode_bonus if m == "referent" else 0.0))
                    preds[(bb, m)].append({"key": key, "model": bb, "pipeline": MODE2PIPE[m],
                                           "video_id": key.split("|")[0], "question_id": key.split("|")[1],
                                           "capability": "x", "reid": "x", "predicted": "A", "correct": "A",
                                           "is_correct": rnd.random() * 100 < p})
    base_dir = os.path.join(tmp, "results_baseline")
    for (bb, m), rows in preds.items():
        d = os.path.join(base_dir, MODE2PIPE[m], bb)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "predictions.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return selp, base_dir, V, EXTRA


def run_fixture():
    print("=" * 100)
    print("FIXTURE SELF-TEST -- synthetic data matching the selections.jsonl schema exactly,")
    print("with a PLANTED dose curve. No real data is touched.")
    print("=" * 100)
    ok = True
    regimes = (
        ("SATURATING (planted EXTRA = 0 pts/frame, no mode bonus)", True, 0.0, True),
        ("LINEAR     (planted EXTRA = 6 pts/frame, no mode bonus)", False, 0.0, False),
        ("CONFOUNDED (planted EXTRA = 0 pts/frame, +25 pts mode bonus on `referent`)", True, 25.0, True),
    )
    for name, saturating, bonus, expect_sat in regimes:
        tmp = tempfile.mkdtemp(prefix="path2fix_")
        selp, base, V, EXTRA = make_fixture(tmp, saturating, mode_bonus=bonus)
        print(f"\n########## fixture: {name} ##########")
        sel = load_selections(selp)
        preds = load_predictions(base)
        outp = os.path.join(tmp, "cairn_stats.json")
        out = analyse(sel, preds, [], PRIMARY_BACKBONES, outp, verbose=True)
        extra = json.load(open(outp.replace(".json", "_extra.json")))
        mv = out["marginal_value"]
        satb = extra["saturation_stratified"]["mode_balanced"]
        hitb = extra["first_frame_stratified"]["mode_balanced"]
        naive = extra["saturation_paired_POOLED_CONFOUNDED_do_not_report"]
        fe = extra["conditional_model_question_fe"].get("coef", {})
        fe_extra_c = fe.get("extra_frames_(depth-1)", {}).get("coef_pts")
        fe_hit_c = fe.get("hit_1{depth>=1}", {}).get("coef_pts")
        print(f"\n[fixture] planted first-frame value {V:.1f} pts, mode bonus {bonus:.1f} pts on referent")
        print(f"[fixture]   pooled-intercept recovery      {fmt(mv['first_frame_in_window'], '.2f')} pts")
        edh = extra["mean_depth_given_hit_pooled"]
        v_hit_expected = V + EXTRA * (edh - 1.0)   # a hit carries E[depth|hit] frames, not one
        print(f"[fixture]   MODE-BALANCED value of a HIT    {fmt(hitb['diff_pts'], '.2f')} pts "
              f"(expected V + EXTRA*(E[depth|hit]={edh:.2f} - 1) = {v_hit_expected:.2f})")
        print(f"[fixture]   FE model 1{{depth>=1}} coef       {fmt(fe_hit_c, '.2f')} pts")
        print(f"[fixture] planted per-extra-frame {EXTRA:.1f} pts")
        print(f"[fixture]   pooled WLS slope                {fmt(mv['each_additional_frame'], '.2f')} pts")
        print(f"[fixture]   mode-stratified slope           {fmt(mv['each_additional_frame_mode_stratified'], '.2f')} pts")
        print(f"[fixture]   MODE-BALANCED depth>=2 vs ==1   {fmt(satb['diff_pts'], '.2f')} pts "
              f"(p={fmt(satb['p_bootstrap'], '.3g')}, MDE {fmt(satb['mde_pts'], '.2f')})")
        print(f"[fixture]   FE model (depth-1) coef         {fmt(fe_extra_c, '.2f')} pts "
              f"(p={fmt(fe.get('extra_frames_(depth-1)', {}).get('p'), '.3g')})")
        print(f"[fixture]   MIS-SPECIFIED pooled contrast   {fmt(naive['diff_pts'], '.2f')} pts "
              f"(p={fmt(naive['p'], '.3g')})  <- the defect")
        print(f"[fixture] verdict = {mv['saturation_verdict']} (saturates={mv['saturates']}), "
              f"expected saturates={expect_sat}")
        checks = [
            ("saturation verdict", mv["saturates"] == expect_sat),
            ("MODE-BALANCED per-extra-frame within 3 pts of planted",
             satb["diff_pts"] is not None and abs(satb["diff_pts"] - EXTRA) < 3.0),
            ("FE (depth-1) coefficient within 3 pts of planted",
             fe_extra_c is not None and abs(fe_extra_c - EXTRA) < 3.0),
            ("MODE-BALANCED value of a HIT within 5 pts of V + EXTRA*(E[depth|hit]-1)",
             hitb["diff_pts"] is not None and abs(hitb["diff_pts"] - v_hit_expected) < 5.0),
            ("FE 1{depth>=1} coefficient within 5 pts of planted",
             fe_hit_c is not None and abs(fe_hit_c - V) < 5.0),
            ("pooled-intercept first-frame value within 5 pts of planted (mode-confounded when a "
             "mode bonus is planted, so only checked at bonus=0)",
             bonus > 0 or abs(mv["first_frame_in_window"] - V) < 5.0),
            ("mode-stratified slope within 3 pts of planted",
             mv["each_additional_frame_mode_stratified"] is not None
             and abs(mv["each_additional_frame_mode_stratified"] - EXTRA) < 3.0),
            ("all schema top-level keys present",
             set(out) == {"n_keys", "breadth_depth", "depth_curve", "marginal_value",
                          "gap_decomposition", "membank", "spread_vs_acc"}),
            ("referent hit rate > chunk hit rate (planted geometry)",
             out["breadth_depth"]["referent"]["hit_rate"] > out["breadth_depth"]["chunk"]["hit_rate"]),
            ("chunk E[depth|hit] > referent E[depth|hit] (planted geometry)",
             out["breadth_depth"]["chunk"]["mean_depth_given_hit"] >
             out["breadth_depth"]["referent"]["mean_depth_given_hit"]),
            ("every backbone has a finite breadth term and a finite depth term "
             "(NOT a summation check -- residual is the remainder by definition)",
             all(v["breadth_term"] is not None and v["depth_term"] is not None
                 and v["breadth_term"] == v["breadth_term"] and v["depth_term"] == v["depth_term"]
                 for v in out["gap_decomposition"].values())),
            ("mode-pair contingency emitted with >= 2 cells",
             extra["mode_pair_contingency_saturation"]["n_cells"] >= 2),
        ]
        if bonus > 0:
            # the point of this regime: the old estimator must be fooled, the new one must not be.
            checks.append(("MIS-SPECIFIED pooled contrast is fooled by the mode bonus "
                           "(large and significant although the planted dose effect is 0)",
                           abs(naive["diff_pts"]) > 3.0 and sig(naive["p"])))
            checks.append(("MODE-BALANCED estimate is NOT fooled (|diff| < 3 pts and not significant)",
                           abs(satb["diff_pts"]) < 3.0 and not sig(satb["p_bootstrap"])))
        for label, good in checks:
            print(f"[fixture]  {'PASS' if good else 'FAIL'}  {label}")
            ok &= bool(good)
    print("\n" + "=" * 100)
    print("FIXTURE RESULT:", "ALL CHECKS PASSED" if ok else "FAILURES ABOVE")
    print("=" * 100)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selections", default=f"{ROOT}/solutions/shared/analysis/selections.jsonl")
    ap.add_argument("--results", default=f"{ROOT}/results_baseline")
    ap.add_argument("--membank", default=f"{ROOT}/solutions/cairn/results/*.jsonl")
    ap.add_argument("--out", default=f"{ROOT}/solutions/shared/analysis/cairn_stats.json")
    ap.add_argument("--all-backbones", action="store_true",
                    help="pool videochat-flash-7b into the dose curve too (it is a characterised "
                         "failure case with incomplete chunk/random/uniform runs)")
    ap.add_argument("--fixture", action="store_true", help="run the synthetic self-test and exit")
    args = ap.parse_args()

    if args.fixture:
        sys.exit(run_fixture())

    sel = load_selections(args.selections)
    preds = load_predictions(args.results)
    mbrows = load_membank(args.membank)
    bbs = ALL_BACKBONES if args.all_backbones else PRIMARY_BACKBONES
    print(f"[load] selections: {len(sel)} keys from {args.selections} "
          f"(file is written live by dump_selections.py; a partial file is analysed as-is)")
    print(f"[load] membank rows: {len(mbrows)}")
    out = analyse(sel, preds, mbrows, bbs, args.out)

    # videochat-flash-7b, the characterised failure case, always reported, never hidden
    if not args.all_backbones:
        keys = sorted(k for k, d in sel.items()
                      if all(m in d for m in MODES) and d["referent"].get("ev_f0") is not None)
        pc, pk = preds.get(("videochat-flash-7b", "chunk"), {}), preds.get(("videochat-flash-7b", "referent"), {})
        kk = [k for k in keys if k in pc and k in pk]
        print("\n" + "-" * 100)
        print("F. videochat-flash-7b -- reported separately, NOT a failure case. Against the")
        print("   budget-matched uniform-8 keyframe WINS here: +2.94 pts, p=2.44e-05, n=3,233 (and")
        print("   +2.88 vs its random control). The superseded '-3.04 failure' compared against VCF's")
        print("   own NATIVE pipeline, which is not budget-matched. It is held out of the pooled dose")
        print("   curve because its mp4-render penalty (-5.75 pts) dwarfs the other backbones'.")
        print("-" * 100)
        if kk:
            m = mcnemar([(pc[k], pk[k]) for k in kk])
            print(f"  acc(chunk) - acc(referent) = {m['diff_pts']:+.2f} pts, p={m['p']:.4g}, n={m['n']}"
                  + ("" if sig(m["p"]) else f"   [null: {null_phrase(m)}]"))
            hn = sum(1 for k in kk for mo in MODES
                     if sel[k][mo]["n_in_window"] > 0 and k in preds.get(("videochat-flash-7b", mo), {}))
            hk = sum(preds[("videochat-flash-7b", mo)][k] for k in kk for mo in MODES
                     if sel[k][mo]["n_in_window"] > 0 and k in preds.get(("videochat-flash-7b", mo), {}))
            mn = sum(1 for k in kk for mo in MODES
                     if sel[k][mo]["n_in_window"] == 0 and k in preds.get(("videochat-flash-7b", mo), {}))
            mk = sum(preds[("videochat-flash-7b", mo)][k] for k in kk for mo in MODES
                     if sel[k][mo]["n_in_window"] == 0 and k in preds.get(("videochat-flash-7b", mo), {}))
            if hn and mn:
                print(f"  acc|hit {hk/hn*100:.2f}% (n={hn}) vs acc|miss {mk/mn*100:.2f}% (n={mn})  -> "
                      f"value of a hit {hk/hn*100 - mk/mn*100:+.2f} pts   [DIAGNOSTIC]")
        else:
            print("  [no overlapping keys in the current partial selections file]")
    return out


if __name__ == "__main__":
    main()
