#!/usr/bin/env python3
"""
PATH 1 MECHANISM ANALYSIS -- does CLIP keyframe selection help *because* it lands
on evidence?

===============================================================================
WHAT THIS IS, AND WHAT IT IS NOT
===============================================================================
Every hit/miss quantity in this file is computed against the ORACLE EVIDENCE
WINDOWS in benchmark/data/evidence_windows.json (human-verified 92-94%).  The oracle
window is answer-informed.  Therefore EVERY hit-conditioned number below --
P_hit, acc|hit, acc|miss, the dose-response curve, the mediation split -- is a
DIAGNOSTIC that explains an already-measured method, and is NEVER itself a
method and NEVER a reportable accuracy for a system.  The only numbers here
that are method-legal are the unconditional accuracies acc(referent),
acc(random), acc(uniform), acc(chunk), which are exactly the headline numbers.

===============================================================================
THE CLAIM UNDER TEST (stated so it can fail)
===============================================================================
H_med: the accuracy gain of `referent` (CLIP top-8 of a 64-frame pool) over its
       `random` control (8 drawn at random from the SAME 64-pool) is MEDIATED by
       evidence-window hit rate.
H_med predicts BOTH of:
  (a) P_hit(referent) >> P_hit(random) ~ P_hit(uniform);   and
  (b) acc | hit is approximately EQUAL across modes -- a frame inside the window
      is worth the same no matter how it was chosen.
(b) is the falsifier.  If acc|hit(referent) > acc|hit(random) by a material
amount, then hitting is not the whole story and the residual is UNEXPLAINED by
the mediator.  We report that residual explicitly rather than absorbing it.

===============================================================================
THE DECOMPOSITION (derived, exact, no leftover)
===============================================================================
For mode m write, by the law of total probability over the binary event
H = "at least one selected frame falls inside the oracle window":

    acc(m) = P_m * h_m + (1 - P_m) * m_m
        P_m = P(H | m)              hit rate of mode m
        h_m = P(correct | H,  m)    accuracy conditioned on hitting
        m_m = P(correct | ~H, m)    accuracy conditioned on missing

Let r = referent, c = random control, and
    dP = P_r - P_c,  dh = h_r - h_c,  dm = m_r - m_c,
    Pbar = (P_r + P_c)/2,  hbar = (h_r + h_c)/2,  mbar = (m_r + m_c)/2.

Apply the exact bilinear identity  x*y - x'*y' = (x-x')*(y+y')/2 + (x+x')*(y-y')/2
to each of the two products:

    P_r*h_r - P_c*h_c             =  dP*hbar + Pbar*dh
    (1-P_r)*m_r - (1-P_c)*m_c     = -dP*mbar + (1-Pbar)*dm

Adding:

    total_gain = acc(r) - acc(c)
               = dP * (hbar - mbar)                  <- explained_by_hitrate
               + [ Pbar*dh + (1-Pbar)*dm ]           <- explained_by_cond_acc

This is an IDENTITY (symmetric Kitagawa form): the two terms sum to total_gain
with zero interaction term left over, which is why we use the mode-averaged
conditionals hbar/mbar/Pbar rather than one mode's.  The residual is computed
and reported anyway as a numerical check; it must be ~1e-12.

  explained_by_hitrate  = the part of the gain you get purely from moving
                          probability mass from the miss stratum (value mbar)
                          into the hit stratum (value hbar).  This is the
                          mediated part.
  explained_by_cond_acc = the part of the gain that survives *after* fixing the
                          composition, i.e. the frames referent picks are worth
                          more than the frames random picks AT EQUAL hit status.
                          This is the part NOT explained by the mediator.
  pct_mediated          = 100 * explained_by_hitrate / total_gain, and ONLY
                          when total_gain is itself significant.  A percentage
                          of a denominator that is statistically zero is not a
                          quantity; when the paired McNemar on total_gain does
                          not reject at 0.05, or |total_gain| < 1e-4, this
                          field is emitted as null and is never printed.  When
                          it is emitted it carries a seeded key-clustered
                          bootstrap CI (`pct_mediated_ci95`).

===============================================================================
STATISTICS (house rules)
===============================================================================
* Paired only, on the INTERSECTION of keys both arms answered; n reported every
  time.  Significance = EXACT McNemar (scipy binomtest on discordant b vs c,
  two-sided).  No chi-square anywhere.
* CLUSTERING.  `mcnemar_exact` REQUIRES a `cluster_ids` argument -- there is no
  default, so the unclustered path cannot be taken by accident.  When the ids
  are all distinct (one row per question, which is the case everywhere in this
  file, because every McNemar here is run inside a single backbone) the design
  effect is exactly 1 and the reported p stays the EXACT McNemar p.  When ids
  repeat -- i.e. the same question is stacked across backbones/modes -- the p
  and the CI come from a seeded CLUSTER bootstrap that resamples whole
  questions (n_boot >= 10,000), the exact p is retained only as
  `p_exact_unclustered` and labelled INVALID, and the reported MDE is computed
  at the effective sample size n/deff, so a pooled null can never be stated
  against an n that was inflated by backbone reuse.
* THE (b) TEST -- acc|hit(referent) vs acc|hit(random).
  PRIMARY  = exact McNemar on the questions where BOTH modes hit.  That is the
             only stratum where the same item is observed under both modes, so
             it is the only properly paired form of the contrast.  The printed
             verdict for prediction (b) is read off THIS test and says so.
  SUPPORT  = a within-question permutation test (`p_acchit_perm`, >=10,000
             seeded draws): the referent/random labels are permuted WITHIN each
             question that contributes to both hit strata, the full
             h_r - h_c statistic is recomputed on every draw, and the observed
             value is referred to the permutation distribution of that same
             statistic.  Questions that reach only one stratum are held fixed
             (they have no partner to swap with); because the reference
             distribution is centred on its own permutation mean, their
             contribution is ancillary and cancels out of the two-sided
             p-value.  This is the reason p_acchit_perm tracks the paired
             McNemar closely -- it is the Monte-Carlo randomisation analogue of
             it, and it is reported as corroboration, not as a second opinion.
  FOOTNOTE = a two-sided Fisher exact test on the two hit strata
             (`p_acchit_diff`, name kept for the downstream consumer).  It is
             SECONDARY AND UNCALIBRATED: the two strata are NOT disjoint
             question sets -- the measured overlap (reported inline and in the
             JSON as `acchit_overlap`) is roughly half of the referent stratum
             and roughly two thirds of the random stratum -- so Fisher's
             independence assumption is violated.  No verdict is read off it.
* Every null is reported as "no difference larger than <MDE> points", MDE =
  minimum detectable effect at 80% power, alpha=0.05 two-sided, for that n and
  that observed discordance/base rate.  The string "no difference" never
  appears unqualified.
* All arms here are 8 frames.  Budget is stated in every printed table row.
* Wilson 95% CIs.  CIs on pooled dose-response are NOMINAL: the pool stacks the
  same question under several modes/backbones, so those rows are not
  independent and the interval is optimistic.  Stated inline.

Usage:  python solutions/shared/analysis/analyze_lantern.py
        python solutions/shared/analysis/analyze_lantern.py --selftest      # synthetic fixture
Writes: solutions/shared/analysis/lantern_stats.json
CPU only, seconds.
"""

import argparse
import json
import math
import os
import random
import sys
import tempfile
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, fisher_exact, norm

# ----------------------------------------------------------------------------
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEF_SEL = os.path.join(REPO, "analysis3", "selanal", "selections.jsonl")
DEF_RES = os.path.join(REPO, "results_baseline")
DEF_OUT = os.path.join(REPO, "analysis3", "selanal", "lantern_stats.json")

# pipeline dir -> selection mode
PIPE2MODE = {
    "kf_referent": "referent",
    "kf_chunk": "chunk",
    "kf_random": "random",
    "kf_uniform8": "uniform",
}
MODES = ["referent", "chunk", "random", "uniform"]
BACKBONES = ["internvl3-14b", "qwen2.5-vl-7b", "ovis2.5-9b", "videochat-flash-7b"]
FRAME_BUDGET = 8        # every arm in Path 1 is 8 frames
N_VIDEOS_TOTAL = 449    # corpus size, for the partial-dump coverage guard
N_QUESTIONS_TOTAL = 3233

Z = norm.ppf(0.975)      # 1.959964
ZB = norm.ppf(0.80)      # 0.841621

N_BOOT = 10000           # cluster-bootstrap draws (house minimum)
N_PERM = 10000           # within-question permutation draws (house minimum)
SEED = 20260819          # every resampling procedure in this file is seeded

# Sentinel for the (rare, and currently unused in this file) case of genuinely
# independent rows.  `mcnemar_exact` has NO default for cluster_ids precisely so
# that a pooled contrast cannot silently take the unclustered path: the caller
# must either hand over the cluster ids or write INDEPENDENT_ROWS on purpose.
INDEPENDENT_ROWS = "__independent_rows__"


# ------------------------------- stats utils --------------------------------
def wilson(k, n, z=Z):
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return [None, None]
    p = k / n
    d = 1.0 + z * z / n
    ctr = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [max(0.0, ctr - half), min(1.0, ctr + half)]


def _cluster_boot_sums(agg, n_boot=N_BOOT, seed=SEED, max_cells=4_000_000):
    """Seeded nonparametric CLUSTER bootstrap of column sums.

    `agg` is (G, k): row g holds the per-cluster SUMS of k quantities over all
    rows belonging to cluster g (so a cluster is resampled whole, which is the
    point).  Returns (n_boot, k) of resampled column sums.

    Vectorised in chunks: draw a (chunk, G) matrix of cluster indices, turn it
    into multiplicities with one flat bincount, then a single matmul against
    `agg`.  Chunk size is capped so the working set stays ~<100 MB.
    """
    agg = np.asarray(agg, dtype=np.float64)
    G, k = agg.shape
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, k), dtype=np.float64)
    if G == 0:
        return out
    chunk = max(1, min(n_boot, int(max_cells // max(G, 1))))
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, G, size=(b, G))
        flat = (np.arange(b, dtype=np.int64)[:, None] * G + idx).ravel()
        cnt = np.bincount(flat, minlength=b * G).reshape(b, G).astype(np.float64)
        out[done:done + b] = cnt @ agg
        done += b
    return out


def _boot_two_sided_p(boot, null=0.0):
    """Two-sided bootstrap p for H0: theta == null, by percentile inversion.
    Floored at 1/(B+1) -- a bootstrap can never certify a p below its
    resolution, and reporting 0.0 would be a lie."""
    boot = np.asarray(boot, dtype=np.float64)
    boot = boot[np.isfinite(boot)]
    B = boot.size
    if B == 0:
        return 1.0
    lo = float(np.mean(boot <= null))
    hi = float(np.mean(boot >= null))
    p = 2.0 * min(lo, hi)
    return float(min(1.0, max(p, 1.0 / (B + 1))))


def mcnemar_exact(pairs, cluster_ids, n_boot=N_BOOT, seed=SEED):
    """Paired McNemar on `pairs` = list of (a_correct, b_correct) in {0,1}.

    `cluster_ids` is REQUIRED and has NO default (defect P2: a pooled contrast
    that stacks the same question across backbones inflates n by the number of
    backbones and understates the MDE by 20-75%).  Pass the question key of
    every pair, in the same order as `pairs`; pass INDEPENDENT_ROWS only when
    you have positively established that every row is a distinct unit.

    Behaviour
      * all ids distinct (or INDEPENDENT_ROWS)  -> deff = 1.0 exactly, reported
        p is the EXACT McNemar p (binomtest on b vs c, two-sided), and the CI on
        delta comes from the cluster bootstrap (which then degenerates to the
        ordinary bootstrap).  This is the house-rule case.
      * ids repeat (genuinely pooled data)      -> the reported p and CI come
        from a seeded cluster bootstrap that resamples whole CLUSTERS; the
        exact p is kept as `p_exact_unclustered` and flagged invalid; the design
        effect deff is estimated as var_cluster_boot(delta)/var_iid(delta) and
        the caller's MDE must be evaluated at n/deff (see `mde_mcnemar_clustered`).

    Returns dict with n, b, c, delta (=a-b accuracy difference), p, plus
    p_method, n_clusters, deff, ci95, p_exact_unclustered.
    """
    n = len(pairs)
    b = sum(1 for a, x in pairs if a == 1 and x == 0)   # a right, b wrong
    c = sum(1 for a, x in pairs if a == 0 and x == 1)   # a wrong, b right
    p_exact = 1.0 if b + c == 0 else float(
        binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)
    delta = (b - c) / n if n else float("nan")
    res = {"n": n, "b": b, "c": c, "delta": delta, "p": p_exact,
           "p_method": "exact_mcnemar", "n_clusters": n, "deff": 1.0,
           "ci95": [None, None], "p_exact_unclustered": p_exact}
    if n == 0:
        res["n_clusters"] = 0
        return res

    if cluster_ids is INDEPENDENT_ROWS or cluster_ids == INDEPENDENT_ROWS:
        ids = list(range(n))
    else:
        ids = list(cluster_ids)
        if len(ids) != n:
            raise ValueError("cluster_ids has length %d but there are %d pairs"
                             % (len(ids), n))

    # per-row signed discordance: delta = mean(d)
    d = np.array([(1 if (a == 1 and x == 0) else (-1 if (a == 0 and x == 1) else 0))
                  for a, x in pairs], dtype=np.float64)

    order = {}
    gidx = np.empty(n, dtype=np.int64)
    for i, cid in enumerate(ids):
        if cid not in order:
            order[cid] = len(order)
        gidx[i] = order[cid]
    G = len(order)
    res["n_clusters"] = G

    agg = np.zeros((G, 2), dtype=np.float64)
    np.add.at(agg, (gidx, 0), d)
    np.add.at(agg, (gidx, 1), 1.0)

    bs = _cluster_boot_sums(agg, n_boot=n_boot, seed=seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot_delta = np.where(bs[:, 1] > 0, bs[:, 0] / np.maximum(bs[:, 1], 1e-12), np.nan)
    finite = boot_delta[np.isfinite(boot_delta)]
    if finite.size:
        res["ci95"] = [float(np.percentile(finite, 2.5)),
                       float(np.percentile(finite, 97.5))]

    if G == n:
        # one row per cluster: nothing is pooled, exact McNemar stands.
        res["deff"] = 1.0
        return res

    # genuinely clustered: exact p is invalid, bootstrap takes over.
    var_iid = float(d.var(ddof=1)) / n if n > 1 else 0.0
    var_boot = float(finite.var(ddof=1)) if finite.size > 1 else 0.0
    deff = (var_boot / var_iid) if var_iid > 0 else 1.0
    res["deff"] = float(max(1.0, deff))
    res["p"] = _boot_two_sided_p(boot_delta, 0.0)
    res["p_method"] = "cluster_bootstrap(%d draws, seed=%d)" % (n_boot, seed)
    return res


def mde_mcnemar_clustered(mc, power=0.80, alpha=0.05):
    """MDE in PROPORTION units for a (possibly clustered) McNemar result dict,
    evaluated at the EFFECTIVE sample size n/deff and n_disc/deff.  For deff=1
    this is identical to mde_mcnemar(n, b+c)."""
    deff = max(1.0, float(mc.get("deff", 1.0)))
    n_eff = mc["n"] / deff
    disc_eff = (mc["b"] + mc["c"]) / deff
    return mde_mcnemar(n_eff, disc_eff, power=power, alpha=alpha)


def mde_mcnemar(n, n_disc, power=0.80, alpha=0.05):
    """Minimum detectable effect (in PROPORTION units, multiply by 100 for
    points) for an exact/normal-approx McNemar with n pairs of which n_disc are
    discordant, at the given power.

    Under H1 the discordant split is psi = b/(b+c). Normal approximation:
        (psi-.5)*sqrt(n_disc) = z_{a/2}*0.5 + z_power*sqrt(psi(1-psi))
    Solved by fixed point (2 sweeps is ample). Effect on accuracy difference is
        delta = (b-c)/n = n_disc*(2*psi-1)/n.
    Assumes the discordance rate stays at its observed value under H1, which is
    the standard planning assumption.
    """
    if n == 0 or n_disc == 0:
        return None
    za = norm.ppf(1 - alpha / 2)
    zb = norm.ppf(power)
    psi = 0.5
    for _ in range(60):
        new = 0.5 + (za * 0.5 + zb * math.sqrt(max(psi * (1 - psi), 1e-12))) / math.sqrt(n_disc)
        new = min(new, 0.999999)
        if abs(new - psi) < 1e-12:
            psi = new
            break
        psi = new
    return n_disc * (2 * psi - 1) / n


def mde_two_prop(n1, n2, p_pool, power=0.80, alpha=0.05):
    """MDE (proportion units) for an UNPAIRED two-proportion contrast at the
    pooled base rate p_pool. Normal approximation, equal-variance planning."""
    if not n1 or not n2:
        return None
    za = norm.ppf(1 - alpha / 2)
    zb = norm.ppf(power)
    v = p_pool * (1 - p_pool) * (1.0 / n1 + 1.0 / n2)
    return (za + zb) * math.sqrt(max(v, 1e-12))


def within_question_perm_acchit(v_hit_r, v_hit_c, d_both, n_perm=N_PERM, seed=SEED):
    """WITHIN-QUESTION permutation test for prediction (b) (defect P3).

    The statistic is the full unpaired-looking contrast
        T = h_r - h_c = (sum over referent-hit questions)/n1
                      - (sum over random-hit  questions)/n2
    but the only exchange the null licenses is *within a question that reaches
    BOTH hit strata*: for such a question the same item was observed under both
    modes, so under H0 ("a frame inside the window is worth the same however it
    was chosen") its two outcomes may be swapped between the modes.

    v_hit_r / v_hit_c : sums of `correct` over the referent-hit / random-hit
                        strata (the observed n1*h_r and n2*h_c), as (sum, n).
    d_both            : array of (pr[k] - pc[k]) over the questions in BOTH
                        strata.  Swapping question k moves d_both[k] out of the
                        referent sum and into the random sum, which is exactly
                        what the swap does -- no approximation.

    Questions reaching only one stratum have no partner and are held fixed.
    They shift the observed T and the permutation mean by the SAME amount, so
    the two-sided p (which refers |T - mean(T_perm)| to the permutation
    distribution of |T - mean(T_perm)|) is free of them: their contribution is
    ancillary.  This is why the test agrees closely with the both-hit McNemar
    -- it is that test's randomisation analogue, and is reported as
    corroboration of it.
    """
    (R0, n1), (C0, n2) = v_hit_r, v_hit_c
    d_both = np.asarray(d_both, dtype=np.float64)
    if n1 == 0 or n2 == 0 or d_both.size == 0 or not np.any(d_both != 0):
        return {"p": None, "n_swappable": int(d_both.size),
                "n_informative": int(np.sum(d_both != 0)) if d_both.size else 0,
                "t_obs": None, "n_perm": 0}
    rng = np.random.default_rng(seed)
    t_obs = R0 / n1 - C0 / n2
    ts = np.empty(n_perm, dtype=np.float64)
    m = d_both.size
    chunk = max(1, min(n_perm, int(4_000_000 // max(m, 1))))
    done = 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        swap = (rng.random((b, m)) < 0.5).astype(np.float64)
        S = swap @ d_both                      # net mass moved r -> c
        ts[done:done + b] = (R0 - S) / n1 - (C0 + S) / n2
        done += b
    ctr = float(ts.mean())
    obs_dev = abs(t_obs - ctr)
    p = (1.0 + float(np.sum(np.abs(ts - ctr) >= obs_dev - 1e-12))) / (n_perm + 1.0)
    return {"p": float(min(1.0, p)), "n_swappable": int(m),
            "n_informative": int(np.sum(d_both != 0)),
            "t_obs": float(t_obs), "n_perm": int(n_perm)}


def mediation_cluster_bootstrap(hit_r, hit_c, pr, pc, cluster_ids,
                                n_boot=N_BOOT, seed=SEED):
    """Seeded CLUSTER bootstrap (clusters = question keys) of the whole
    mediation decomposition.  Recomputes the SAME exact identity on every draw
    -- P_r, P_c, h_r, m_r, h_c, m_c are all re-estimated inside the draw, so the
    resulting intervals carry the sampling uncertainty of the conditionals too,
    not just of the marginals.  Degenerate strata inside a draw take the same
    0.0 convention as the point estimate.

    Returns arrays: total, expl_hit, expl_cond, pct (= 100*expl_hit/total, NaN
    where the drawn total is numerically zero).
    """
    hit_r = np.asarray(hit_r, dtype=np.float64)
    hit_c = np.asarray(hit_c, dtype=np.float64)
    pr = np.asarray(pr, dtype=np.float64)
    pc = np.asarray(pc, dtype=np.float64)
    n = hit_r.size
    order, gidx = {}, np.empty(n, dtype=np.int64)
    for i, cid in enumerate(cluster_ids):
        if cid not in order:
            order[cid] = len(order)
        gidx[i] = order[cid]
    G = len(order)
    cols = np.stack([np.ones(n), hit_r, hit_c,
                     pr * hit_r, pr * (1 - hit_r),
                     pc * hit_c, pc * (1 - hit_c), pr, pc], axis=1)
    agg = np.zeros((G, cols.shape[1]), dtype=np.float64)
    np.add.at(agg, gidx, cols)

    bs = _cluster_boot_sums(agg, n_boot=n_boot, seed=seed)
    N = bs[:, 0]
    ok = N > 0
    N = np.where(ok, N, np.nan)
    nhr, nhc = bs[:, 1], bs[:, 2]
    nmr, nmc = N - nhr, N - nhc
    safe = lambda num, den: np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    P_r, P_c = bs[:, 1] / N, bs[:, 2] / N
    h_r, m_r = safe(bs[:, 3], nhr), safe(bs[:, 4], nmr)
    h_c, m_c = safe(bs[:, 5], nhc), safe(bs[:, 6], nmc)
    total = bs[:, 7] / N - bs[:, 8] / N
    dP, dh, dm = P_r - P_c, h_r - h_c, m_r - m_c
    Pbar, hbar, mbar = (P_r + P_c) / 2, (h_r + h_c) / 2, (m_r + m_c) / 2
    expl_hit = dP * (hbar - mbar)
    expl_cond = Pbar * dh + (1 - Pbar) * dm
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(np.abs(total) > 1e-12, 100.0 * expl_hit / total, np.nan)
    return {"total": total, "expl_hit": expl_hit, "expl_cond": expl_cond,
            "pct": pct, "n_clusters": G}


def _pct_ci(arr, total_boot, total_obs):
    """Percentile CI for a RATIO whose denominator can change sign under
    resampling.  Returns (ci, frac_unstable).  If more than 5% of the draws put
    the denominator at/through zero the ratio has no interpretable interval and
    the CI is returned as None rather than as a pair of large numbers."""
    arr = np.asarray(arr, dtype=np.float64)
    tb = np.asarray(total_boot, dtype=np.float64)
    bad = ~np.isfinite(arr) | ~np.isfinite(tb) | (np.sign(tb) != np.sign(total_obs)) \
        | (np.abs(tb) < 1e-9)
    frac_bad = float(np.mean(bad)) if bad.size else 1.0
    good = arr[~bad]
    if frac_bad > 0.05 or good.size < 100:
        return None, frac_bad
    return [float(np.percentile(good, 2.5)), float(np.percentile(good, 97.5))], frac_bad


def _fmt_ci(ci, fmt="%+.4f"):
    if not ci or ci[0] is None or ci[1] is None:
        return "n/a"
    return (fmt + ", " + fmt) % (ci[0], ci[1])


def quartiles(xs):
    """median / p25 / p75, linear interpolation, no numpy dependency needed."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"median": None, "p25": None, "p75": None}

    def q(f):
        if len(xs) == 1:
            return float(xs[0])
        i = f * (len(xs) - 1)
        lo = int(math.floor(i))
        hi = min(lo + 1, len(xs) - 1)
        return float(xs[lo] + (xs[hi] - xs[lo]) * (i - lo))

    return {"median": q(0.50), "p25": q(0.25), "p75": q(0.75)}


def bucket(n_in_window):
    if n_in_window >= 3:
        return "3+"
    return str(int(n_in_window))


BUCKETS = ["0", "1", "2", "3+"]


# --------------------------------- loading ----------------------------------
def load_selections(path):
    """-> sel[key][mode] = row.  Last write wins (file may be appended to while
    the dumper runs); truncated final line tolerated."""
    sel = defaultdict(dict)
    bad = 0
    if not os.path.exists(path):
        sys.exit(f"[FATAL] selections file not found: {path}")
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if "key" not in r or "mode" not in r:
                bad += 1
                continue
            sel[r["key"]][r["mode"]] = r
    return dict(sel), bad


def load_preds(res_dir):
    """-> preds[backbone][mode][key] = 0/1"""
    preds = defaultdict(lambda: defaultdict(dict))
    for pipe, mode in PIPE2MODE.items():
        pdir = os.path.join(res_dir, pipe)
        if not os.path.isdir(pdir):
            continue
        for bb in sorted(os.listdir(pdir)):
            f = os.path.join(pdir, bb, "predictions.jsonl")
            if not os.path.exists(f):
                continue
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    preds[bb][mode][r["key"]] = int(bool(r.get("is_correct")))
    return {b: dict(m) for b, m in preds.items()}


# --------------------------------- analysis ---------------------------------
def analyse(sel, preds):
    out = {}
    lines = []      # printed report

    def say(s=""):
        lines.append(s)

    modes_present = [m for m in MODES if any(m in v for v in sel.values())]
    # keys where EVERY present mode has a selection row -> the paired substrate
    all_keys = sorted(k for k, v in sel.items() if all(m in v for m in modes_present))
    win_keys = [k for k in all_keys if sel[k][modes_present[0]].get("ev_f0") is not None]
    win_set = set(win_keys)

    out["n_keys"] = len(all_keys)
    out["n_with_window"] = len(win_keys)

    # ---- coverage guard: dump_selections.py walks videos in id order, so a
    # partial file is a PREFIX of the video list, not a random sample of it.
    # Anything computed on a prefix is PRELIMINARY and must be labelled so.
    n_vid = len({k.split("|")[0] for k in all_keys})
    cov = n_vid / N_VIDEOS_TOTAL
    out["coverage"] = {"n_videos": n_vid, "n_videos_total": N_VIDEOS_TOTAL,
                       "frac": cov, "complete": cov >= 0.98,
                       "questions_total": N_QUESTIONS_TOTAL}

    say("=" * 78)
    say("PATH 1 MECHANISM ANALYSIS -- keyframe selection vs its random control")
    say("=" * 78)
    if not out["coverage"]["complete"]:
        say("*** PRELIMINARY -- PARTIAL DUMP ***")
        say("selections.jsonl covers %d/%d videos (%.1f%%). dump_selections.py walks"
            % (n_vid, N_VIDEOS_TOTAL, 100 * cov))
        say("videos in id order, so this is a PREFIX of the corpus, NOT a random")
        say("sample: the covered questions are not exchangeable with the full set and")
        say("their unconditional accuracy can differ materially from the headline")
        say("numbers. Do not quote any figure below until coverage reaches 100%.")
        say("")
    say("DIAGNOSTIC, NOT A METHOD: every hit/miss and dose quantity below is")
    say("conditioned on ORACLE, answer-informed evidence windows. It explains an")
    say("already-measured gain; it can never be used to select frames at test time.")
    say("Frame budget: ALL arms = %d frames (budget-matched by construction;" % FRAME_BUDGET)
    say("uniform-8 is an exact subset of the 64-frame candidate pool).")
    say("")
    say("keys with all %d modes dumped : %d" % (len(modes_present), len(all_keys)))
    say("of which have an oracle window : %d" % len(win_keys))
    say("modes present                  : %s" % ", ".join(modes_present))
    say("")

    # ---------------- (a) hit rate ----------------
    hit_rate = {}
    say("-" * 78)
    say("(a) EVIDENCE-WINDOW HIT RATE  [8 frames/arm, n=%d questions with a window]" % len(win_keys))
    say("-" * 78)
    say("%-10s %8s %8s   %s" % ("mode", "P_hit", "n", "Wilson 95% CI"))
    for m in modes_present:
        k = sum(1 for key in win_keys if sel[key][m]["hit"])
        n = len(win_keys)
        ci = wilson(k, n)
        hit_rate[m] = {"rate": (k / n if n else None), "n": n,
                       "ci95": [None if c is None else float(c) for c in ci]}
        say("%-10s %8s %8d   [%s, %s]" % (
            m,
            "%.4f" % (k / n) if n else "n/a", n,
            "%.4f" % ci[0] if ci[0] is not None else "n/a",
            "%.4f" % ci[1] if ci[1] is not None else "n/a"))
    out["hit_rate"] = hit_rate

    # paired McNemar on the hit indicator itself
    hr_delta = {}
    say("")
    say("paired exact McNemar on the HIT INDICATOR (same questions, same pool):")
    for other, label in (("random", "referent_minus_random"),
                         ("uniform", "referent_minus_uniform")):
        if "referent" not in modes_present or other not in modes_present:
            continue
        pairs = [(int(bool(sel[k]["referent"]["hit"])), int(bool(sel[k][other]["hit"])))
                 for k in win_keys]
        # clusters = question keys (P2). One row per question here, so deff=1
        # and the exact McNemar p stands; the argument is mandatory so that a
        # later pooled variant cannot silently skip the correction.
        r = mcnemar_exact(pairs, cluster_ids=win_keys)
        mde = mde_mcnemar_clustered(r)
        hr_delta[label] = {"delta": r["delta"], "p": r["p"], "n": r["n"],
                           "b": r["b"], "c": r["c"],
                           "mde80_points": None if mde is None else 100 * mde,
                           "p_method": r["p_method"], "deff": r["deff"],
                           "n_clusters": r["n_clusters"], "ci95_delta": r["ci95"]}
        verdict = ("SIGNIFICANT" if r["p"] < 0.05 else
                   "null: no difference larger than %.2f points at 80%% power" % (100 * mde)
                   if mde is not None else "null (no discordant pairs)")
        say("  referent - %-8s = %+.4f  (b=%d c=%d, n=%d)  p=%.3g  %s"
            % (other, r["delta"], r["b"], r["c"], r["n"], r["p"], verdict))
    out["hit_rate_delta"] = hr_delta

    # ---------------- (b) accuracy conditioned on hit ----------------
    backbones = sorted(preds.keys())
    acc_by_hit = {}
    say("")
    say("-" * 78)
    say("(b) ACCURACY | HIT vs ACCURACY | MISS   [DIAGNOSTIC, oracle-conditioned]")
    say("    8 frames/arm. If hitting is the whole mechanism, the `hit` column is")
    say("    flat across modes.")
    say("-" * 78)
    say("%-18s %-10s %14s %14s" % ("backbone", "mode", "acc|hit (n)", "acc|miss (n)"))
    for bb in backbones:
        acc_by_hit[bb] = {}
        for m in modes_present:
            pm = preds[bb].get(m, {})
            hk = [k for k in win_keys if k in pm and sel[k][m]["hit"]]
            mk = [k for k in win_keys if k in pm and not sel[k][m]["hit"]]
            ah = sum(pm[k] for k in hk) / len(hk) if hk else None
            am = sum(pm[k] for k in mk) / len(mk) if mk else None
            acc_by_hit[bb][m] = {"hit": {"acc": ah, "n": len(hk)},
                                 "miss": {"acc": am, "n": len(mk)}}
            say("%-18s %-10s %14s %14s" % (
                bb, m,
                "%.4f (%d)" % (ah, len(hk)) if ah is not None else "n/a (0)",
                "%.4f (%d)" % (am, len(mk)) if am is not None else "n/a (0)"))
    out["acc_by_hit"] = acc_by_hit

    # ---------------- dose-response ----------------
    # pooled over (backbone, mode, key). NOMINAL CIs: rows are not independent.
    dose_cnt = {b: [0, 0] for b in BUCKETS}          # [correct, n]
    dose_bb = {bb: {b: [0, 0] for b in BUCKETS} for bb in backbones}
    for bb in backbones:
        for m in modes_present:
            pm = preds[bb].get(m, {})
            for k in win_keys:
                if k not in pm:
                    continue
                b = bucket(sel[k][m]["n_in_window"])
                dose_cnt[b][0] += pm[k]
                dose_cnt[b][1] += 1
                dose_bb[bb][b][0] += pm[k]
                dose_bb[bb][b][1] += 1

    dose_pooled = {}
    for b in BUCKETS:
        c, n = dose_cnt[b]
        ci = wilson(c, n)
        dose_pooled[b] = {"acc": (c / n if n else None), "n": n,
                          "ci95": [None if x is None else float(x) for x in ci]}
    out["dose_pooled"] = dose_pooled
    out["dose_by_backbone"] = {bb: {b: {"acc": (dose_bb[bb][b][0] / dose_bb[bb][b][1]
                                                if dose_bb[bb][b][1] else None),
                                        "n": dose_bb[bb][b][1]}
                                    for b in BUCKETS} for bb in backbones}

    say("")
    say("-" * 78)
    say("DOSE-RESPONSE: accuracy vs #selected frames inside the oracle window")
    say("[DIAGNOSTIC. 8 frames/arm. Pooled over backbones x modes -> rows share")
    say(" questions, so the pooled Wilson CIs are NOMINAL / optimistic.]")
    say("-" * 78)
    say("%-6s %10s %8s   %s" % ("n_in_w", "acc", "n", "Wilson 95% CI (nominal)"))
    for b in BUCKETS:
        d = dose_pooled[b]
        say("%-6s %10s %8d   [%s, %s]" % (
            b, "%.4f" % d["acc"] if d["acc"] is not None else "n/a", d["n"],
            "%.4f" % d["ci95"][0] if d["ci95"][0] is not None else "n/a",
            "%.4f" % d["ci95"][1] if d["ci95"][1] is not None else "n/a"))
    say("")
    say("per backbone (acc / n):")
    say("%-18s %s" % ("backbone", "  ".join("%10s" % b for b in BUCKETS)))
    for bb in backbones:
        cells = []
        for b in BUCKETS:
            d = out["dose_by_backbone"][bb][b]
            cells.append("%10s" % ("%.3f/%d" % (d["acc"], d["n"]) if d["acc"] is not None else "n/a"))
        say("%-18s %s" % (bb, "  ".join(cells)))

    # ---------------- mediation ----------------
    mediation = {}
    say("")
    say("-" * 78)
    say("MEDIATION: is acc(referent) - acc(random) explained by hit rate?")
    say("  total = dP*(hbar-mbar)  +  [Pbar*dh + (1-Pbar)*dm]      (exact identity)")
    say("  8 frames/arm. Paired on keys BOTH arms answered AND that have a window.")
    say("-" * 78)
    for bb in backbones:
        pr = preds[bb].get("referent", {})
        pc = preds[bb].get("random", {})
        keys = [k for k in win_keys if k in pr and k in pc]
        if not keys:
            continue
        n = len(keys)
        hit_r = {k: bool(sel[k]["referent"]["hit"]) for k in keys}
        hit_c = {k: bool(sel[k]["random"]["hit"]) for k in keys}

        P_r = sum(hit_r.values()) / n
        P_c = sum(hit_c.values()) / n
        hk_r = [k for k in keys if hit_r[k]]
        mk_r = [k for k in keys if not hit_r[k]]
        hk_c = [k for k in keys if hit_c[k]]
        mk_c = [k for k in keys if not hit_c[k]]
        h_r = sum(pr[k] for k in hk_r) / len(hk_r) if hk_r else 0.0
        m_r = sum(pr[k] for k in mk_r) / len(mk_r) if mk_r else 0.0
        h_c = sum(pc[k] for k in hk_c) / len(hk_c) if hk_c else 0.0
        m_c = sum(pc[k] for k in mk_c) / len(mk_c) if mk_c else 0.0

        acc_r = sum(pr[k] for k in keys) / n
        acc_c = sum(pc[k] for k in keys) / n
        total = acc_r - acc_c

        dP, dh, dm = P_r - P_c, h_r - h_c, m_r - m_c
        Pbar, hbar, mbar = (P_r + P_c) / 2, (h_r + h_c) / 2, (m_r + m_c) / 2
        expl_hit = dP * (hbar - mbar)
        expl_cond = Pbar * dh + (1 - Pbar) * dm
        residual = total - (expl_hit + expl_cond)

        # ---- the (b) test ---------------------------------------------------
        # The two hit strata are NOT different question sets: they are two
        # heavily OVERLAPPING subsets of the same paired key set (a question is
        # in both whenever both modes happened to hit it).  The overlap is
        # measured here and reported; it is why Fisher is a footnote and the
        # both-hit McNemar is the primary test.
        a1, n1 = sum(pr[k] for k in hk_r), len(hk_r)
        a2, n2 = sum(pc[k] for k in hk_c), len(hk_c)
        both = [k for k in keys if hit_r[k] and hit_c[k]]
        n_ov = len(both)
        overlap = {"n_referent_hit": n1, "n_random_hit": n2, "n_overlap": n_ov,
                   "frac_of_referent_stratum": (n_ov / n1) if n1 else None,
                   "frac_of_random_stratum": (n_ov / n2) if n2 else None}

        # SECONDARY / FOOTNOTE: unpaired Fisher exact.  Its independence
        # assumption is violated by the overlap above, so its p is uncalibrated
        # and NO verdict is read off it.  Field name kept for the consumer.
        if n1 and n2:
            p_fisher = float(fisher_exact([[a1, n1 - a1], [a2, n2 - a2]],
                                          alternative="two-sided")[1])
            p_pool = (a1 + a2) / (n1 + n2)
            mde_h = mde_two_prop(n1, n2, p_pool)
        else:
            p_fisher, mde_h = 1.0, None

        # PRIMARY: exact McNemar on the questions where BOTH modes hit -- the
        # only stratum in which the same item is seen under both modes.
        mc_both = (mcnemar_exact([(pr[k], pc[k]) for k in both], cluster_ids=both)
                   if both else None)
        mde_both = mde_mcnemar_clustered(mc_both) if mc_both else None

        # SUPPORT: within-question permutation on the full h_r - h_c statistic.
        perm = within_question_perm_acchit(
            (a1, n1), (a2, n2), [pr[k] - pc[k] for k in both])

        # paired McNemar on the headline (unconditional) gain
        mc_total = mcnemar_exact([(pr[k], pc[k]) for k in keys], cluster_ids=keys)
        mde_total = mde_mcnemar_clustered(mc_total)

        # ---- P7: pct_mediated only when its denominator is a real quantity --
        total_is_sig = (mc_total["p"] < 0.05) and (abs(total) >= 1e-4)
        boot = mediation_cluster_bootstrap(
            [1.0 if hit_r[k] else 0.0 for k in keys],
            [1.0 if hit_c[k] else 0.0 for k in keys],
            [pr[k] for k in keys], [pc[k] for k in keys], keys)
        pct_val, pct_ci, pct_bad, pct_note = None, None, None, None
        if total_is_sig:
            pct_val = 100.0 * expl_hit / total
            pct_ci, pct_bad = _pct_ci(boot["pct"], boot["total"], total)
            if pct_ci is None:
                pct_note = ("ratio unstable: %.1f%% of cluster-bootstrap draws put the "
                            "total gain at or through zero" % (100 * pct_bad))
        else:
            pct_note = ("suppressed: total gain is not significant "
                        "(paired McNemar p=%.3g, |gain|=%.4f) -- a percentage of a "
                        "denominator indistinguishable from zero is not a quantity"
                        % (mc_total["p"], abs(total)))

        def _ci(a):
            a = a[np.isfinite(a)]
            return ([float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
                    if a.size else [None, None])

        mediation[bb] = {
            "total_gain": total,
            "explained_by_hitrate": expl_hit,
            "explained_by_cond_acc": expl_cond,
            "pct_mediated": pct_val,
            "acc_hit_referent": h_r,
            "acc_hit_random": h_c,
            "p_acchit_diff": p_fisher,
            # --- (b)-test, repaired (P3) -------------------------------------
            # PRIMARY paired test; the printed verdict reads off this one.
            "p_acchit_paired": (None if mc_both is None else mc_both["p"]),
            "p_acchit_perm": perm["p"],
            "acchit_overlap": overlap,
            "acchit_primary_test": "paired_exact_mcnemar_on_both_hit",
            "p_acchit_diff_note": (
                "SECONDARY/FOOTNOTE ONLY -- Fisher exact, unpaired. The two hit "
                "strata overlap by %d questions = %s of the referent stratum and "
                "%s of the random stratum, so the independence assumption is "
                "violated and this p is uncalibrated. Read the verdict off "
                "p_acchit_paired." % (
                    n_ov,
                    "n/a" if not n1 else "%.1f%%" % (100.0 * n_ov / n1),
                    "n/a" if not n2 else "%.1f%%" % (100.0 * n_ov / n2))),
            "p_acchit_perm_note": (
                "within-question permutation, %d seeded draws (seed=%d); labels "
                "swapped only within the %d questions reaching BOTH hit strata "
                "(%d of them informative). Corroborates p_acchit_paired by "
                "construction." % (perm["n_perm"], SEED, perm["n_swappable"],
                                   perm["n_informative"])),
            "pct_mediated_ci95": pct_ci,
            "pct_mediated_note": pct_note,
            "total_gain_ci95": _ci(boot["total"]),
            "explained_by_hitrate_ci95": _ci(boot["expl_hit"]),
            "explained_by_cond_acc_ci95": _ci(boot["expl_cond"]),
            "bootstrap": {"n_boot": N_BOOT, "seed": SEED,
                          "clusters": "question key", "n_clusters": boot["n_clusters"]},
            # --- extras (diagnostic bookkeeping; required keys above are intact)
            "residual": residual,
            "n": n,
            "p_hit_referent": P_r,
            "p_hit_random": P_c,
            "acc_miss_referent": m_r,
            "acc_miss_random": m_c,
            "acc_referent": acc_r,
            "acc_random": acc_c,
            "n_hit_referent": n1,
            "n_hit_random": n2,
            "mde80_acchit_points": None if mde_h is None else 100 * mde_h,
            "mde80_acchit_paired_points": None if mde_both is None else 100 * mde_both,
            "p_total_gain_mcnemar": mc_total["p"],
            "mde80_total_points": None if mde_total is None else 100 * mde_total,
            "paired_bothhit": (None if mc_both is None else
                               {"n": mc_both["n"], "delta": mc_both["delta"],
                                "p": mc_both["p"],
                                "mde80_points": None if mde_both is None else 100 * mde_both,
                                "p_method": mc_both["p_method"],
                                "deff": mc_both["deff"],
                                "ci95_delta": mc_both["ci95"]}),
        }

        say("")
        say("%s   [n=%d paired questions with a window, 8 frames/arm]" % (bb, n))
        say("  acc(referent)=%.4f  acc(random)=%.4f   total_gain=%+.4f  "
            "(exact McNemar p=%.3g)" % (acc_r, acc_c, total, mc_total["p"]))
        if mc_total["p"] >= 0.05:
            say("    -> null: %s" % (
                "no gain larger than %.2f points at 80%% power" % (100 * mde_total)
                if mde_total is not None else "no discordant pairs; n too small for an MDE"))
        say("  P_hit  ref=%.4f  rand=%.4f   dP=%+.4f" % (P_r, P_c, dP))
        say("  acc|hit  ref=%.4f (n=%d)  rand=%.4f (n=%d)   dh=%+.4f" % (h_r, n1, h_c, n2, dh))
        say("  acc|miss ref=%.4f (n=%d)  rand=%.4f (n=%d)   dm=%+.4f"
            % (m_r, len(mk_r), m_c, len(mk_c), dm))
        say("  explained_by_hitrate  = %+.4f  95%% CI [%s]"
            % (expl_hit, _fmt_ci(mediation[bb]["explained_by_hitrate_ci95"])))
        say("  explained_by_cond_acc = %+.4f  95%% CI [%s]   <- NOT explained by the mediator"
            % (expl_cond, _fmt_ci(mediation[bb]["explained_by_cond_acc_ci95"])))
        say("  residual (must be ~0)  = %+.2e" % residual)
        if pct_val is None:
            say("  pct_mediated          = n/a  (%s)" % pct_note)
        else:
            say("  pct_mediated          = %.1f%%  95%% CI [%s]  (key-clustered "
                "bootstrap, %d draws, seed=%d)"
                % (pct_val,
                   "n/a" if pct_ci is None else "%.1f%%, %.1f%%" % (pct_ci[0], pct_ci[1]),
                   N_BOOT, SEED))
            if pct_ci is None:
                say("        ^ CI withheld: %s" % pct_note)

        # ---- prediction (b): verdict is read off the PAIRED test -------------
        say("  (b)-test acc|hit(referent) vs acc|hit(random):")
        if mc_both is None or mc_both["n"] == 0:
            say("    PRIMARY [paired exact McNemar, BOTH-hit questions]: no both-hit "
                "questions; prediction (b) is UNTESTED at this n.")
        else:
            if mc_both["p"] < 0.05:
                verd = ("SIGNIFICANT -> acc|hit is NOT flat across modes, so hitting is "
                        "NOT the whole mechanism.")
            elif mde_both is not None:
                verd = ("null: no acc|hit difference larger than %.2f points at 80%% "
                        "power -> prediction (b) is NOT refuted at this n."
                        % (100 * mde_both))
            else:
                verd = "null (no discordant pairs); prediction (b) is UNTESTED at this n."
            say("    PRIMARY [paired exact McNemar on the %d questions where BOTH modes"
                % mc_both["n"])
            say("            hit -- the only stratum where the same item is seen under")
            say("            both modes]: delta=%+.4f  b=%d c=%d  p=%.3g"
                % (mc_both["delta"], mc_both["b"], mc_both["c"], mc_both["p"]))
            say("            VERDICT (read off the PAIRED both-hit McNemar): %s" % verd)
        if perm["p"] is not None:
            say("    SUPPORT [within-question permutation of the referent/random labels,")
            say("            %d seeded draws (seed=%d), %d swappable questions of which"
                % (perm["n_perm"], SEED, perm["n_swappable"]))
            say("            %d informative; statistic = h_r - h_c = %+.4f]: p=%.3g"
                % (perm["n_informative"], perm["t_obs"], perm["p"]))
        else:
            say("    SUPPORT [within-question permutation]: n/a (no swappable/informative "
                "question in both strata)")
        say("    FOOTNOTE, NOT A VERDICT [Fisher exact on the two hit strata]: p=%.3g."
            % p_fisher)
        say("            These strata are NOT different question sets: they OVERLAP by")
        say("            %d questions = %s of the referent stratum and %s of the random"
            % (n_ov,
               "n/a" if not n1 else "%.1f%%" % (100.0 * n_ov / n1),
               "n/a" if not n2 else "%.1f%%" % (100.0 * n_ov / n2)))
        say("            stratum, so Fisher's independence assumption is violated and")
        say("            this p is uncalibrated. It is reported for continuity only.")
    out["mediation"] = mediation

    # ---------------- nearest_s ----------------
    nearest = {}
    say("")
    say("-" * 78)
    say("HOW CLOSE DOES A SELECTION GET? nearest_s = seconds from window centre to")
    say("the nearest selected frame (all questions with a window, 8 frames/arm).")
    say("-" * 78)
    say("%-10s %10s %10s %10s %8s" % ("mode", "p25", "median", "p75", "n"))
    for m in modes_present:
        vals = [sel[k][m]["nearest_s"] for k in win_keys
                if sel[k][m].get("nearest_s") is not None]
        q = quartiles(vals)
        nearest[m] = q
        say("%-10s %10s %10s %10s %8d" % (
            m,
            "%.3f" % q["p25"] if q["p25"] is not None else "n/a",
            "%.3f" % q["median"] if q["median"] is not None else "n/a",
            "%.3f" % q["p75"] if q["p75"] is not None else "n/a",
            len(vals)))
    # same, restricted to MISSES only -- "how close does a miss get?"
    say("")
    say("restricted to MISSES only (hit == False):")
    say("%-10s %10s %10s %10s %8s" % ("mode", "p25", "median", "p75", "n"))
    for m in modes_present:
        vals = [sel[k][m]["nearest_s"] for k in win_keys
                if sel[k][m].get("nearest_s") is not None and not sel[k][m]["hit"]]
        q = quartiles(vals)
        nearest[m] = dict(nearest[m])
        nearest[m]["miss_only"] = q
        say("%-10s %10s %10s %10s %8d" % (
            m,
            "%.3f" % q["p25"] if q["p25"] is not None else "n/a",
            "%.3f" % q["median"] if q["median"] is not None else "n/a",
            "%.3f" % q["p75"] if q["p75"] is not None else "n/a",
            len(vals)))
    out["nearest_s"] = nearest

    say("")
    say("=" * 78)
    say("REMINDER: hit/miss/dose/mediation all use ORACLE evidence windows.")
    say("They are DIAGNOSTICS that explain the measured gain. None is a method,")
    say("and none may be quoted as an achievable accuracy.")
    say("=" * 78)
    return out, "\n".join(lines)


# --------------------------------- selftest ---------------------------------
def _selftest():
    """Synthetic fixture with a KNOWN planted mediation structure.

    Ground truth planted:
        P_hit(referent) = 0.60,  P_hit(random) = 0.20,  P_hit(uniform) = 0.20
        acc|hit  = 0.60 for BOTH referent and random   (pure mediation, dh = 0)
        acc|miss = 0.20 for BOTH referent and random               (dm = 0)
      => total_gain      = 0.60*0.60+0.40*0.20 - (0.20*0.60+0.80*0.20) = 0.44-0.28 = 0.16
         explained_by_hitrate = dP*(hbar-mbar) = 0.40*(0.60-0.20) = 0.16
         explained_by_cond_acc = 0
         pct_mediated = 100%
    A second backbone plants a DELIBERATE mediation FAILURE:
        same hit rates, but acc|hit(referent)=0.80 vs acc|hit(random)=0.60,
        acc|miss both 0.20  =>  dh=+0.20, dm=0
         total = 0.60*0.80+0.40*0.20 - 0.28 = 0.56-0.28 = 0.28
         expl_hit  = 0.40*((0.80+0.60)/2 - 0.20) = 0.40*0.50 = 0.20
         expl_cond = Pbar*dh = 0.40*0.20 = 0.08     (0.20+0.08 = 0.28 exactly)
         pct_mediated = 71.43%
    Frames are constructed so that n_in_window and hit are internally consistent.
    """
    rng = random.Random(20260819)
    tmp = tempfile.mkdtemp(prefix="path1_fixture_")
    seld = os.path.join(tmp, "selections.jsonl")
    resd = os.path.join(tmp, "results_baseline")

    NQ = 8000
    N_TOTAL, FPS = 1000, 25.0
    EV0, EV1 = 400, 460                      # oracle window, absolute frames
    plan = {"referent": 0.60, "random": 0.20, "uniform": 0.20, "chunk": 0.35}
    keys = ["vid_%04d|q%d" % (i % N_VIDEOS_TOTAL + 1, i // N_VIDEOS_TOTAL) for i in range(NQ)]
    hit_of = {m: {} for m in plan}

    with open(seld, "w") as fh:
        for qi, key in enumerate(keys):
            has_win = (qi % 40 != 0)          # 2.5% of questions have no window
            for m, p in plan.items():
                h = rng.random() < p
                hit_of[m][key] = h and has_win
                if h and has_win:
                    n_in = rng.choice([1, 1, 1, 2, 2, 3, 4])
                    frames = sorted(rng.sample(range(EV0, EV1 + 1), n_in) +
                                    rng.sample(range(0, EV0 - 30), 8 - n_in))
                    near = 0.0
                else:
                    n_in = 0
                    frames = sorted(rng.sample(range(0, EV0 - 30), 8))
                    near = abs((EV0 + EV1) / 2 - frames[-1]) / FPS
                vid, qid = key.split("|")
                fh.write(json.dumps({
                    "key": key, "real_key": "REAL%s|%s" % (vid, qid),
                    "video_id": vid, "question_id": qid, "mode": m,
                    "sel_pool": list(range(8)), "sel_frames": frames,
                    "n_total": N_TOTAL, "fps": FPS,
                    "ev_f0": EV0 if has_win else None,
                    "ev_f1": EV1 if has_win else None,
                    "ev_span_frames": (EV1 - EV0) if has_win else None,
                    "n_in_window": n_in if has_win else 0,
                    "hit": (n_in > 0) if has_win else None,
                    "nearest_s": near if has_win else None,
                    "spread": (frames[-1] - frames[0]) / (N_TOTAL - 1),
                    "chunk_ids": sorted({f * 8 // N_TOTAL for f in frames}),
                }) + "\n")

    # accuracy: P(correct | hit, mode, backbone) planted exactly as documented
    acc_plan = {
        "bbA": {"referent": (0.60, 0.20), "random": (0.60, 0.20),
                "uniform": (0.60, 0.20), "chunk": (0.60, 0.20)},
        "bbB": {"referent": (0.80, 0.20), "random": (0.60, 0.20),
                "uniform": (0.60, 0.20), "chunk": (0.70, 0.20)},
    }
    for pipe, m in PIPE2MODE.items():
        for bb, ap in acc_plan.items():
            d = os.path.join(resd, pipe, bb)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "predictions.jsonl"), "w") as fh:
                for key in keys:
                    h = hit_of[m][key]
                    p = ap[m][0] if h else ap[m][1]
                    vid, qid = key.split("|")
                    fh.write(json.dumps({
                        "key": key, "model": bb, "pipeline": pipe,
                        "video_id": vid, "question_id": qid,
                        "capability": "test", "reid": "single_shot",
                        "predicted": "A", "correct": "A",
                        "is_correct": bool(rng.random() < p)}) + "\n")

    sel, bad = load_selections(seld)
    preds = load_preds(resd)
    st, report = analyse(sel, preds)

    ok = True

    def chk(name, got, want, tol):
        nonlocal ok
        good = got is not None and abs(got - want) <= tol
        ok &= good
        print("  [%s] %-38s got=%s want=%s tol=%s"
              % ("PASS" if good else "FAIL", name,
                 "%.4f" % got if got is not None else "None", want, tol))

    print("\n" + "=" * 78)
    print("FIXTURE SELF-TEST (synthetic selections.jsonl + predictions.jsonl,")
    print("schemas identical to the real files; %d questions x 4 modes x 2 backbones)" % NQ)
    print("=" * 78)
    chk("hit_rate referent (planted .60)", st["hit_rate"]["referent"]["rate"], 0.60, 0.02)
    chk("hit_rate random (planted .20)", st["hit_rate"]["random"]["rate"], 0.20, 0.02)
    chk("hit_delta ref-rand (planted .40)", st["hit_rate_delta"]["referent_minus_random"]["delta"], 0.40, 0.03)

    mA, mB = st["mediation"]["bbA"], st["mediation"]["bbB"]
    chk("bbA total_gain (planted .16)", mA["total_gain"], 0.16, 0.035)
    chk("bbA expl_by_hitrate (.16)", mA["explained_by_hitrate"], 0.16, 0.035)
    chk("bbA expl_by_cond_acc (0)", mA["explained_by_cond_acc"], 0.0, 0.030)
    chk("bbA pct_mediated (100%)", mA["pct_mediated"], 100.0, 20.0)
    chk("bbA acc|hit ref (.60)", mA["acc_hit_referent"], 0.60, 0.035)
    chk("bbA acc|hit rand (.60)", mA["acc_hit_random"], 0.60, 0.055)
    chk("bbB total_gain (planted .28)", mB["total_gain"], 0.28, 0.035)
    chk("bbB expl_by_hitrate (.20)", mB["explained_by_hitrate"], 0.20, 0.035)
    chk("bbB expl_by_cond_acc (.08)", mB["explained_by_cond_acc"], 0.08, 0.030)
    chk("bbB pct_mediated (71.4%)", mB["pct_mediated"], 71.43, 12.0)
    chk("bbB acc|hit ref (.80)", mB["acc_hit_referent"], 0.80, 0.035)

    # decomposition must sum EXACTLY
    for bb, m in st["mediation"].items():
        r = m["total_gain"] - (m["explained_by_hitrate"] + m["explained_by_cond_acc"])
        good = abs(r) < 1e-9
        ok &= good
        print("  [%s] %-38s residual=%.3e" % ("PASS" if good else "FAIL",
                                              "decomposition sums (%s)" % bb, r))
    # ---- prediction (b): the PRIMARY test is the paired both-hit McNemar ----
    # The planted regimes must be recovered by the test the verdict is read off,
    # not merely by the demoted Fisher footnote.
    gA, gB = mA["p_acchit_paired"] < 0.05, mB["p_acchit_paired"] < 0.05
    ok &= (not gA) and gB
    print("  [%s] bbA PRIMARY paired both-hit McNemar: planted acc|hit NULL correctly "
          "NOT flagged (p=%.3g, n=%d)"
          % ("PASS" if not gA else "FAIL", mA["p_acchit_paired"], mA["paired_bothhit"]["n"]))
    print("  [%s] bbB PRIMARY paired both-hit McNemar: planted mediation FAILURE "
          "detected (p=%.3g, n=%d)"
          % ("PASS" if gB else "FAIL", mB["p_acchit_paired"], mB["paired_bothhit"]["n"]))
    pA, pB = mA["p_acchit_perm"] < 0.05, mB["p_acchit_perm"] < 0.05
    ok &= (not pA) and pB
    print("  [%s] bbA within-question permutation agrees with PRIMARY (p=%.3g)"
          % ("PASS" if not pA else "FAIL", mA["p_acchit_perm"]))
    print("  [%s] bbB within-question permutation agrees with PRIMARY (p=%.3g)"
          % ("PASS" if pB else "FAIL", mB["p_acchit_perm"]))
    # demoted Fisher must still be emitted (a consumer reads it) and must still
    # recover the planted regimes -- it is wrong about *calibration*, not sign.
    fA, fB = mA["p_acchit_diff"] < 0.05, mB["p_acchit_diff"] < 0.05
    ok &= (not fA) and fB
    print("  [%s] Fisher footnote still emitted under p_acchit_diff (bbA %.3g, bbB %.3g)"
          % ("PASS" if (not fA) and fB else "FAIL",
             mA["p_acchit_diff"], mB["p_acchit_diff"]))
    # the overlap that invalidates Fisher must be MEASURED, not asserted away
    ovA = mA["acchit_overlap"]
    ov_ok = (ovA["n_overlap"] > 0 and ovA["frac_of_referent_stratum"] is not None
             and 0.0 < ovA["frac_of_referent_stratum"] <= 1.0)
    ok &= ov_ok
    print("  [%s] acc|hit strata overlap measured: %d questions = %.1f%% of referent, "
          "%.1f%% of random stratum"
          % ("PASS" if ov_ok else "FAIL", ovA["n_overlap"],
             100 * ovA["frac_of_referent_stratum"], 100 * ovA["frac_of_random_stratum"]))
    # ---- P7: pct_mediated must be a number ONLY on a significant total ------
    p7_ok = True
    for bb, m in st["mediation"].items():
        sig = m["p_total_gain_mcnemar"] < 0.05 and abs(m["total_gain"]) >= 1e-4
        good = (m["pct_mediated"] is not None) == sig
        p7_ok &= good
        print("  [%s] %-38s total p=%.3g -> pct_mediated=%s"
              % ("PASS" if good else "FAIL", "pct_mediated gated on total (%s)" % bb,
                 m["p_total_gain_mcnemar"],
                 "None" if m["pct_mediated"] is None else "%.1f%%" % m["pct_mediated"]))
    ok &= p7_ok
    # planted totals here ARE significant, so a CI must be attached and must
    # bracket the point estimate
    for bb, m in st["mediation"].items():
        if m["pct_mediated"] is None:
            continue
        ci = m["pct_mediated_ci95"]
        good = ci is not None and ci[0] <= m["pct_mediated"] <= ci[1]
        ok &= good
        print("  [%s] %-38s %.1f%% in [%s]"
              % ("PASS" if good else "FAIL", "pct_mediated CI brackets point (%s)" % bb,
                 m["pct_mediated"], "n/a" if ci is None else "%.1f, %.1f" % (ci[0], ci[1])))
    # ---- P2: clustering machinery is wired and cannot be bypassed -----------
    try:
        mcnemar_exact([(1, 0), (0, 1)])
        clus_ok = False
    except TypeError:
        clus_ok = True
    ok &= clus_ok
    print("  [%s] mcnemar_exact refuses to run without cluster_ids"
          % ("PASS" if clus_ok else "FAIL"))
    # a deliberately POOLED contrast (same question stacked 3x) must be caught:
    # deff > 1, p from the cluster bootstrap, MDE inflated relative to naive.
    pooled_pairs, pooled_keys = [], []
    _rng = random.Random(7)
    for q in range(300):
        base = _rng.random() < 0.5
        for _ in range(3):                      # 3 "backbones" of the same item
            pooled_pairs.append((int(base or _rng.random() < 0.10),
                                 int((not base) and _rng.random() < 0.10)))
            pooled_keys.append("q%d" % q)
    mc_pool = mcnemar_exact(pooled_pairs, cluster_ids=pooled_keys)
    mde_naive = mde_mcnemar(mc_pool["n"], mc_pool["b"] + mc_pool["c"])
    mde_clus = mde_mcnemar_clustered(mc_pool)
    pool_ok = (mc_pool["n"] == 900 and mc_pool["n_clusters"] == 300
               and mc_pool["deff"] > 1.5
               and mc_pool["p_method"].startswith("cluster_bootstrap")
               and mde_clus > mde_naive)
    ok &= pool_ok
    print("  [%s] pooled McNemar clustered: n=%d over %d clusters, deff=%.2f, "
          "MDE %.2f -> %.2f pts, p via %s"
          % ("PASS" if pool_ok else "FAIL", mc_pool["n"], mc_pool["n_clusters"],
             mc_pool["deff"], 100 * mde_naive, 100 * mde_clus,
             mc_pool["p_method"].split("(")[0]))
    # unique clusters must reduce EXACTLY to the exact McNemar (deff==1)
    uniq = mcnemar_exact(pooled_pairs, cluster_ids=list(range(len(pooled_pairs))))
    uniq_ok = (uniq["deff"] == 1.0 and uniq["p_method"] == "exact_mcnemar"
               and abs(uniq["p"] - uniq["p_exact_unclustered"]) < 1e-15)
    ok &= uniq_ok
    print("  [%s] unique clusters -> deff=1.0 and the EXACT McNemar p is kept"
          % ("PASS" if uniq_ok else "FAIL"))
    # dose must be monotone 0 < 1 <= 3+
    dp = st["dose_pooled"]
    mono = dp["0"]["acc"] < dp["1"]["acc"] and dp["0"]["acc"] < dp["3+"]["acc"]
    ok &= mono
    print("  [%s] dose monotone: 0=%.3f 1=%.3f 2=%.3f 3+=%.3f"
          % ("PASS" if mono else "FAIL", dp["0"]["acc"], dp["1"]["acc"],
             dp["2"]["acc"], dp["3+"]["acc"]))
    # schema completeness
    need = {"n_keys", "n_with_window", "hit_rate", "hit_rate_delta", "acc_by_hit",
            "dose_pooled", "dose_by_backbone", "mediation", "nearest_s", "coverage"}
    miss = need - set(st)
    need_med = {"total_gain", "explained_by_hitrate", "explained_by_cond_acc",
                "pct_mediated", "acc_hit_referent", "acc_hit_random",
                "p_acchit_diff", "p_acchit_paired", "p_acchit_perm",
                "acchit_overlap", "pct_mediated_ci95", "paired_bothhit"}
    miss |= {"mediation.%s" % k for k in (need_med - set(mA))}
    ok &= not miss
    print("  [%s] output schema complete (missing: %s)"
          % ("PASS" if not miss else "FAIL", sorted(miss) or "none"))
    json.dumps(st)   # must be serialisable
    print("=" * 78)
    print("FIXTURE SELF-TEST: %s" % ("ALL PASS" if ok else "FAILURES ABOVE"))
    print("=" * 78)
    return 0 if ok else 1


# ----------------------------------- main -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selections", default=DEF_SEL)
    ap.add_argument("--results", default=DEF_RES)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(_selftest())

    sel, bad = load_selections(a.selections)
    if bad:
        print("[warn] skipped %d unparseable/incomplete lines (dumper may still be running)"
              % bad)
    preds = load_preds(a.results)
    st, report = analyse(sel, preds)
    print(report)
    with open(a.out, "w") as fh:
        json.dump(st, fh, indent=2)
    print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
