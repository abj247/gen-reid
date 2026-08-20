#!/usr/bin/env python3
"""Publication figures for the two-granularity retrieval analysis (ICLR).

Reads the PRIMARY tolerance-corrected stats JSONs and emits two vector PDFs
(+ 200 dpi PNG previews):

    fig_lantern_evidence_mediation.pdf
        "Query conditioning finds the evidence -- and that is about a third of
         the story."
        (a) evidence-window hit rate by mode (4 bars, 95% CIs, value labels),
            with referent - random annotated (paired exact McNemar).
        (b) the MEDIATION DECOMPOSITION as a waterfall: the total accuracy gain
            acc(keyframe) - acc(random) split into the part routed through the
            hit rate and the part routed through accuracy GIVEN a hit, each with
            its key-clustered bootstrap CI. The honest point is that the obvious
            mediator is real but PARTIAL (37.3% mediated), so the panel is drawn
            so the two routes are directly comparable in height.

    fig_cairn_concentration_dose.pdf
        "Hit rate is non-monotone in concentration; the retrieval unit must
         match the evidence scale."
        (a) the CONCENTRATION axis: x = distinct chunks touched, plotted with the
            axis REVERSED so concentration increases to the right; y = P(hit)
            with 95% CIs; marker area proportional to that mode's accuracy, which
            is printed in the label. The inverted U (uniform / random low, keyframe
            at the peak, chunk falling back) is the message.
        (b) the DOSE CURVE over the number of selected frames inside the window
            (0/1/2/3/4+) with Wilson CIs, n per point and the 12.5% chance line,
            annotated with the MODE-STRATIFIED saturation result.

WHAT CHANGED vs the previous version of this script (both old hypotheses died):
  * There is NO breadth-vs-depth trade-off plane any more. keyframe DOMINATES
    chunk on both axes (hit rate 57.73 vs 51.76, depth 2.002 vs 1.862), so a
    trade-off plane drawn through dominating points would be wrong. Fig 2a is now
    the one-dimensional CONCENTRATION axis, in which performance is non-monotone.
  * Fig 1b is no longer the dose curve (which lives in Fig 2b); it is the
    mediation decomposition, because "hitting the evidence" explains only 37.3%
    [21.4, 70.6] of the gain and acc|hit is itself significantly higher for the
    query-conditioned arm (paired both-hit McNemar, +2.62 pts, p = 0.049).

WINDOW TOLERANCE (default; --windows raw for the robustness version):
  Oracle windows were localised on a dense frame grid (median step ~146 video
  frames). Half the questions name a SINGLE dense frame, so the raw window is one
  frame wide and asking whether 8 frames land on it measures nothing. Every window
  is therefore widened by the grid half-step (per video, corpus median fallback);
  the tolerance is a property of the annotation and was never tuned against
  accuracy. See widen_windows.py. --windows raw reproduces both figures on the
  un-widened windows as a robustness check.

COLOUR MAPPING (Okabe-Ito, colourblind-safe; identical in BOTH figures):
    referent / keyframe (Path 1, frame-level CLIP top-k) -> blue         #0072B2
    chunk               (Path 2, chunk-level retrieval)  -> vermillion   #D55E00
    random              (control: same 64-frame pool)    -> grey         #999999
    uniform             (control: plain uniform-8)       -> bluish green #009E73
Two colours appear ONLY in Fig 1b, for the two mediation routes, and are never
used for a selection mode: hit-rate route -> sky blue #56B4E9, conditional-accuracy
route -> reddish purple #CC79A7 (both Okabe-Ito). Quantities pooled over all four
modes (the dose curve in Fig 2b) are drawn in neutral #333333, never in a mode
colour.

CAVEAT CARRIED INTO THE FIGURES: every quantity conditioned on the oracle
evidence window (hit/miss splits, mediation, dose curve) is a DIAGNOSTIC, not a
method -- the windows are answer-informed. Every such panel is labelled
"oracle-window diagnostic, not a method".

Usage:
    python make_figures.py [--windows tol|raw] [--stats-dir DIR] [--out-dir DIR]
                           [--prefix STR]

CPU only, no network, ~1 second.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, set before pyplot import
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------------------------------------------- style ----
PALETTE = {
    "referent": "#0072B2",  # Path 1  (frame-level keyframe selection)
    "chunk":    "#D55E00",  # Path 2  (chunk-level retrieval)
    "random":   "#999999",  # control (query conditioning)
    "uniform":  "#009E73",  # control (pipeline)
}
ROUTE_HIT = "#56B4E9"   # Fig 1b only: gain routed through P(hit)
ROUTE_ACC = "#CC79A7"   # Fig 1b only: gain routed through accuracy | hit
NEUTRAL = "#333333"     # pooled-over-modes quantities (Fig 2b)

PRETTY = {
    "referent": "keyframe\n(Path 1)",
    "chunk":    "chunk\n(Path 2)",
    "random":   "random\n(control)",
    "uniform":  "uniform-8\n(control)",
}
SHORT = {"referent": "keyframe", "chunk": "chunk", "random": "random", "uniform": "uniform-8"}
MODE_ORDER = ["referent", "chunk", "random", "uniform"]
MARKERS = {"referent": "o", "chunk": "s", "random": "^", "uniform": "D"}
CHANCE = 12.5  # 8-way MCQ
PRIMARY_BACKBONES = ["internvl3-14b", "qwen2.5-vl-7b", "ovis2.5-9b"]
MEDIATION_BACKBONE = "internvl3-14b"

RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.2,
    "figure.dpi": 200,
    "savefig.bbox": None,       # fixed 6.8in canvas: see save() / subplots_adjust
    "savefig.pad_inches": 0.02,
}


# ------------------------------------------------------------- utilities ----
def wilson(k: float, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval on a proportion, returned in PERCENT."""
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0))
    return (100.0 * max((c - h) / d, 0.0), 100.0 * min((c + h) / d, 1.0))


def block_scale(values) -> float:
    """Multiplier that puts a BLOCK of rates/accuracies on a percent scale.

    Decided once per block (all its rates <= 1 => fractions), so that a small
    DELTA inside the same block is not mistaken for a fraction. Returns 100.0
    (fractions) or 1.0 (already percent).
    """
    vals = [abs(float(v)) for v in values if v is not None]
    return 100.0 if vals and max(vals) <= 1.0 else 1.0


def conv(x, k: float):
    """Apply a block scale, with a sanity fallback for mixed-convention files."""
    if x is None:
        return None
    y = float(x) * k
    return y if abs(y) <= 100.0 + 1e-9 else float(x)


def ci_pct(entry, acc_key="acc", k=1.0):
    """Return (lo, hi) in percent for a {acc/rate, n, ci95} record."""
    if entry is None:
        return None
    val = entry.get(acc_key, entry.get("rate"))
    if val is None:
        return None
    val = conv(val, k)
    n = int(entry.get("n") or 0)
    ci = entry.get("ci95")
    if ci and len(ci) == 2 and all(c is not None for c in ci):
        lo, hi = conv(ci[0], k), conv(ci[1], k)
        if lo is not None and hi is not None and hi >= lo:
            return (lo, hi)
    if n:
        return wilson(val / 100.0 * n, n)
    return None


def yerr_from(val, ci):
    if ci is None:
        return (0.0, 0.0)
    return (max(val - ci[0], 0.0), max(ci[1] - val, 0.0))


def r2(x) -> str:
    """Signed 2-dp string with HALF-UP rounding (1.525 -> +1.53, not +1.52)."""
    v = float(x)
    q = math.floor(abs(v) * 100.0 + 0.5) / 100.0
    return f"{'-' if v < 0 else '+'}{q:.2f}"


def fmt_p(p) -> str:
    if p is None:
        return "p n/a"
    p = float(p)
    if p == 0:
        return "p < 1e-16"
    if p < 1e-4:
        return f"p = {p:.1e}"
    if p < 0.001:
        return f"p = {p:.2e}"
    if p < 0.01:
        return f"p = {p:.4f}"
    return f"p = {p:.3f}"


def load_json(path, label, required=True):
    if not os.path.exists(path):
        if not required:
            return {}
        sys.exit(f"[make_figures] missing {label}: {path}\n"
                 f"  run the upstream stats step first, or pass --stats-dir")
    with open(path) as fh:
        return json.load(fh)


def bucket_sort_key(b: str):
    return (1, 0) if b.endswith("+") else (0, int(b))


def strip_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, pad=2)


def save(fig, out_dir, stem):
    pdf = os.path.join(out_dir, stem + ".pdf")
    png = os.path.join(out_dir, stem + ".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    return pdf, png


def pooled_mode_accuracy(p1, backbones):
    """Overall accuracy (%) per mode, macro-averaged over `backbones`.

    Each backbone contributes its own hit/miss-weighted accuracy on the same
    n=2,962 questions, then backbones are averaged unweighted (they are the same
    questions, so a macro average is the honest pooling). Used for the marker AREA
    and the printed accuracy in Fig 2a.
    """
    acc_by_hit = p1.get("acc_by_hit") or {}
    have = [b for b in backbones if b in acc_by_hit] or list(acc_by_hit)
    cells = [(rec.get("acc")) for b in have for cell in (acc_by_hit[b] or {}).values()
             for rec in ((cell or {}).get("hit") or {}, (cell or {}).get("miss") or {})]
    k = block_scale(cells)
    per_mode = {}
    for b in have:
        for mode, cell in (acc_by_hit[b] or {}).items():
            num = den = 0.0
            for part in ("hit", "miss"):
                rec = (cell or {}).get(part) or {}
                n = int(rec.get("n") or 0)
                a = conv(rec.get("acc"), k)
                if n and a is not None:
                    num += a * n
                    den += n
            if den:
                per_mode.setdefault(mode, []).append(num / den)
    return ({m: sum(v) / len(v) for m, v in per_mode.items() if v}, have)


# ------------------------------------------------------------- figure 1 ----
def figure1(p1, out_dir, stem, tag):
    """(a) hit rate by mode.  (b) mediation waterfall."""
    hr = p1.get("hit_rate") or {}
    modes = [m for m in MODE_ORDER if m in hr] + [m for m in hr if m not in MODE_ORDER]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.05))

    # ---- (a) evidence-window hit rate by mode ---------------------------
    k_hr = block_scale([hr[m].get("rate") for m in modes])
    xs = list(range(len(modes)))
    vals, errs = [], [[], []]
    for m in modes:
        v = conv(hr[m].get("rate"), k_hr)
        vals.append(v)
        lo, hi = yerr_from(v, ci_pct(hr[m], "rate", k_hr))
        errs[0].append(lo)
        errs[1].append(hi)
    axa.bar(xs, vals, width=0.62,
            color=[PALETTE.get(m, "#444444") for m in modes],
            edgecolor="none", zorder=2)
    axa.errorbar(xs, vals, yerr=errs, fmt="none", ecolor="#222222",
                 elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=3)
    for x, v, hi in zip(xs, vals, errs[1]):
        axa.text(x, v + hi + 1.0, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=7.5, color="#222222")

    top = max(v + e for v, e in zip(vals, errs[1]))
    axa.set_ylim(0, min(105, top * 1.52 + 6))
    axa.set_xticks(xs)
    axa.set_xticklabels([PRETTY.get(m, m) for m in modes])
    axa.set_ylabel("evidence-window hit rate (%)")
    axa.set_xlabel("selection mode")
    strip_ax(axa)

    d = (p1.get("hit_rate_delta") or {}).get("referent_minus_random")
    if d and "referent" in modes and "random" in modes:
        i, j = modes.index("referent"), modes.index("random")
        ytop = max(vals[i] + errs[1][i], vals[j] + errs[1][j])
        bar_y = ytop + (axa.get_ylim()[1] - ytop) * 0.30
        axa.plot([i, i, j, j], [ytop + 3.0, bar_y, bar_y, ytop + 3.0],
                 lw=0.8, color="#222222", clip_on=False)
        dv = conv(d.get("delta"), k_hr)
        axa.text((i + j) / 2.0 + 0.35, bar_y + 1.0,
                 f"keyframe $-$ random = {dv:+.2f} pts\n{fmt_p(d.get('p'))}"
                 f"  (paired exact McNemar)\nn = {d.get('n', 0):,}",
                 ha="center", va="bottom", fontsize=6.8, color="#222222")

    nwin = p1.get("n_with_window")
    nkeys = p1.get("n_keys")
    axa.text(0.0, -0.27,
             f"n = {nwin:,} questions with an oracle window"
             + (f" (of {nkeys:,})" if nkeys else "") + tag
             + "\n8 frames/arm from one 64-frame pool"
               "\noracle-window diagnostic, not a method",
             transform=axa.transAxes, fontsize=7, color="#555555", va="top")

    # ---- (b) mediation waterfall ----------------------------------------
    med_all = p1.get("mediation") or {}
    bb = MEDIATION_BACKBONE if MEDIATION_BACKBONE in med_all else (
        next(iter(med_all)) if med_all else None)
    med = med_all.get(bb) or {}
    if not med:
        axb.text(0.5, 0.5, "mediation stats unavailable", ha="center", va="center",
                 transform=axb.transAxes, fontsize=8, color="#777777")
    else:
        h = 100.0 * float(med["explained_by_hitrate"])        # pts
        a = 100.0 * float(med["explained_by_cond_acc"])       # pts
        tot = 100.0 * float(med["total_gain"])                # pts
        ci_h = [100.0 * c for c in med.get("explained_by_hitrate_ci95", [h, h])]
        ci_a = [100.0 * c for c in med.get("explained_by_cond_acc_ci95", [a, a])]
        ci_t = [100.0 * c for c in med.get("total_gain_ci95", [tot, tot])]

        # waterfall: route 1 from 0, route 2 stacked on it, total as an open bar
        bx = [0, 1, 2]
        bottoms = [0.0, h, 0.0]
        heights = [h, a, tot]
        tops = [h, h + a, tot]
        faces = [ROUTE_HIT, ROUTE_ACC, "none"]
        edges = ["none", "none", NEUTRAL]
        for x, b0, hh, fc, ec in zip(bx, bottoms, heights, faces, edges):
            axb.bar([x], [hh], bottom=[b0], width=0.58, color=fc, edgecolor=ec,
                    linewidth=1.1 if ec != "none" else 0.0, zorder=2)
        # connectors
        axb.plot([0.29, 0.71], [h, h], lw=0.8, ls=(0, (3, 2)), color="#888888", zorder=1)
        axb.plot([1.29, 1.71], [h + a, tot], lw=0.8, ls=(0, (3, 2)), color="#888888", zorder=1)

        # CIs on each component (drawn at the segment top)
        cis = [ci_h, ci_a, ci_t]
        eb_lo = [max(heights[i] - cis[i][0], 0.0) for i in range(3)]
        eb_hi = [max(cis[i][1] - heights[i], 0.0) for i in range(3)]
        axb.errorbar(bx, tops, yerr=[eb_lo, eb_hi], fmt="none", ecolor="#222222",
                     elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=4)

        labels = [f"+{h:.2f}\n[{ci_h[0]:.2f}, {ci_h[1]:.2f}]",
                  f"+{a:.2f}\n[{ci_a[0]:.2f}, {ci_a[1]:.2f}]",
                  f"+{tot:.2f}\n[{ci_t[0]:.2f}, {ci_t[1]:.2f}]"]
        cols = [ROUTE_HIT, ROUTE_ACC, NEUTRAL]
        for x, t, hi, lab, c in zip(bx, tops, eb_hi, labels, cols):
            axb.text(x, t + hi + 0.09, lab, ha="center", va="bottom", fontsize=7,
                     color=c if c != ROUTE_HIT else "#2b7fae")

        axb.set_xticks(bx)
        # D1: the plotted term is `explained_by_cond_acc`, which INCLUDES the
        # acc|miss route (negative) and is computed on all n questions -- NOT on
        # the both-hit subset. Do not label it "acc | hit" or "same-hit".
        axb.set_xticklabels(["via hit rate\n$\\Delta$P(hit)",
                             "via conditional\naccuracy (hit\nand miss strata)",
                             "total gain"])
        axb.tick_params(axis="x", labelsize=7.5)
        # two lines: the single-line label is longer than the axes and was
        # being clipped at the bottom of the fixed canvas
        axb.set_ylabel("contribution to keyframe\ngain (pts)", fontsize=8)
        axb.set_ylim(0, max(ci_t[1], h + a) * 2.05 + 0.4)
        axb.axhline(0, lw=0.8, color="#222222")
        strip_ax(axb)

        pm = med.get("pct_mediated")
        pm_ci = med.get("pct_mediated_ci95") or [None, None]
        pb = med.get("paired_bothhit") or {}
        note_lines = []
        if pm is not None:
            note_lines.append(f"{pm:.1f}% mediated by hit rate"
                              + (f",\n95% CI [{pm_ci[0]:.1f}, {pm_ci[1]:.1f}]"
                                 if pm_ci[0] is not None else ""))
        if pb:
            # D2: never assert a positive verdict over a null. The raw-window
            # panel has p = 0.085; there the box must report the bound, not a
            # finding.
            d_pb = 100.0 * float(pb.get("delta", 0.0))
            p_pb = pb.get("p")
            mde_pb = pb.get("mde80_points")
            bc = (f", b={pb['b']} c={pb['c']}" if pb.get("b") is not None
                  and pb.get("c") is not None else "")
            sig = p_pb is not None and float(p_pb) < 0.05
            if sig:
                note_lines.append(
                    f"acc | hit is NOT flat: {d_pb:+.2f} pts"
                    f"\non the {pb.get('n', 'NA')} both-hit questions,")
                note_lines.append(f"{fmt_p(p_pb)} (paired exact McNemar{bc})")
            else:
                note_lines.append(
                    f"acc | hit: {d_pb:+.2f} pts, {pb.get('n', 'NA')} both-hit"
                    f" questions,\n{fmt_p(p_pb)} (paired exact McNemar{bc})"
                    " -- NULL:")
                note_lines.append(
                    "no difference larger than "
                    + (f"{float(mde_pb):.2f} pts" if mde_pb is not None else "the MDE")
                    + " at 80% power")
        axb.text(0.02, 0.97, "\n".join(note_lines), transform=axb.transAxes,
                 ha="left", va="top",
                 fontsize=6.8, color="#222222",
                 bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#bbbbbb", lw=0.6))

        boot = med.get("bootstrap") or {}
        axb.text(0.0, -0.27,
                 f"n = {med.get('n', 0):,} questions, {bb}{tag}\n"
                 f"CIs: key-clustered bootstrap, {boot.get('n_boot', 0):,} draws"
                 "\noracle-window diagnostic, not a method",
                 transform=axb.transAxes, fontsize=7, color="#555555", va="top")

    fig.subplots_adjust(left=0.085, right=0.985, top=0.96, bottom=0.36, wspace=0.30)
    return save(fig, out_dir, stem)


# ------------------------------------------------------------- figure 2 ----
def figure2(p2, p2x, p1, out_dir, stem, tag):
    """(a) concentration axis (non-monotone).  (b) dose curve + saturation."""
    bd = p2.get("breadth_depth") or {}
    modes = [m for m in MODE_ORDER if m in bd] + [m for m in bd if m not in MODE_ORDER]
    backbones = (p2x.get("backbones_pooled") or PRIMARY_BACKBONES)
    acc_by_mode, used_bb = pooled_mode_accuracy(p1, backbones) if p1 else ({}, [])
    hr = p1.get("hit_rate") or {} if p1 else {}

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.45))

    # ---- (a) the CONCENTRATION axis --------------------------------------
    k_bd = block_scale([bd[m].get("hit_rate") for m in modes])
    k_hr = block_scale([(hr.get(m) or {}).get("rate") for m in modes]) if hr else 1.0
    conc = {m: float(bd[m].get("mean_distinct_chunks") or 0.0) for m in modes}
    phit = {m: conv(bd[m].get("hit_rate"), k_bd) for m in modes}
    depth = {m: float(bd[m].get("mean_depth_given_hit") or 0.0) for m in modes}
    err = {}
    for m in modes:
        ci = ci_pct(hr.get(m), "rate", k_hr) if hr.get(m) else None
        if ci is None:
            n = int((hr.get(m) or {}).get("n") or p2.get("n_keys") or 0)
            ci = wilson(phit[m] / 100.0 * n, n) if n else None
        err[m] = yerr_from(phit[m], ci)

    have_acc = all(m in acc_by_mode for m in modes) and bool(acc_by_mode)

    def area(m):  # marker AREA strictly proportional to accuracy
        return 6.0 * acc_by_mode[m] if have_acc else 90.0

    order = sorted(modes, key=lambda m: -conc[m])   # 8.00 -> 2.00 = left -> right
    axa.plot([conc[m] for m in order], [phit[m] for m in order],
             lw=1.0, color="#bbbbbb", zorder=1)
    for m in modes:
        axa.errorbar([conc[m]], [phit[m]],
                     yerr=[[err[m][0]], [err[m][1]]], fmt="none",
                     ecolor=PALETTE.get(m, "#444444"), elinewidth=0.9,
                     capsize=2.5, capthick=0.9, zorder=2)
    for m in sorted(modes, key=lambda mm: -area(mm)):   # big markers first
        axa.scatter([conc[m]], [phit[m]], s=area(m), marker=MARKERS.get(m, "o"),
                    facecolor=PALETTE.get(m, "#444444"), edgecolor="white",
                    linewidth=0.6, zorder=3)
    # label placement: peak/valley aware, chosen so nothing overprints
    lab_off = {"uniform": (7, 8, "left", "bottom"),     # leftmost: fan right
               "random": (0, -13, "center", "top"),
               "referent": (0, 13, "center", "bottom"),
               "chunk": (-5, -9, "right", "top")}
    for m in modes:
        lab = SHORT.get(m, m) + (f"  ({acc_by_mode[m]:.1f}%)" if have_acc else "")
        ox, oy, ha, va = lab_off.get(m, (0, -12, "center", "top"))
        anchor = phit[m] + (err[m][1] if va == "bottom" else -err[m][0])
        axa.annotate(lab, (conc[m], anchor), textcoords="offset points",
                     xytext=(ox, oy), ha=ha, va=va, fontsize=7,
                     color=PALETTE.get(m, "#444444"))

    axa.set_xlim(max(conc.values()) + 0.9, min(conc.values()) - 0.9)  # REVERSED
    ys_hi = [phit[m] + err[m][1] for m in modes]
    ys_lo = [phit[m] - err[m][0] for m in modes]
    span = max(ys_hi) - min(ys_lo)
    axa.set_ylim(min(ys_lo) - span * 0.72, max(ys_hi) + span * 0.42)
    axa.set_xlabel("distinct chunks touched  (of 8)\nconcentration increases $\\rightarrow$")
    axa.set_ylabel("evidence-window hit rate (%)")
    strip_ax(axa)
    if "referent" in depth and "chunk" in depth:
        axa.text(0.98, 0.02,
                 "E[in-window frames | hit]\n"
                 f"keyframe  {depth['referent']:.2f} @ {conc['referent']:.2f} chunks\n"
                 f"chunk      {depth['chunk']:.2f} @ {conc['chunk']:.2f} chunks",
                 transform=axa.transAxes, ha="right", va="bottom", fontsize=6.6,
                 color="#222222", multialignment="left",
                 bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#bbbbbb", lw=0.6))
    sub = ("marker area $\\propto$ accuracy (printed), "
           f"{len(used_bb)} backbones" if have_acc else "marker area fixed (accuracy unavailable)")
    axa.text(0.0, -0.36,
             f"n = {p2.get('n_keys', 0):,} questions, 8 frames/arm{tag}\n{sub}"
             "\noracle-window diagnostic, not a method",
             transform=axa.transAxes, fontsize=7, color="#555555", va="top")

    # ---- (b) the DOSE CURVE + saturation --------------------------------
    dc = dict(p2.get("depth_curve") or {})
    # the "0" (miss) point lives in the *_extra file as the unpaired miss stratum
    if "0" not in dc and p2x.get("acc_given_miss") is not None:
        dc["0"] = {"acc": p2x["acc_given_miss"], "n": p2x.get("n_miss_obs")}
    buckets = sorted(dc.keys(), key=bucket_sort_key)
    bx = list(range(len(buckets)))
    k_dc = block_scale([dc[b].get("acc") for b in buckets])
    bv = [conv(dc[b].get("acc"), k_dc) for b in buckets]
    bn = [int(dc[b].get("n") or 0) for b in buckets]
    be = [[], []]
    for b, v in zip(buckets, bv):
        lo, hi = yerr_from(v, ci_pct(dc[b], "acc", k_dc))
        be[0].append(lo)
        be[1].append(hi)

    axb.axhline(CHANCE, ls=(0, (4, 3)), lw=0.9, color="#666666", zorder=1)
    axb.text(-0.62, CHANCE + 0.5, "chance (12.5%)",
             ha="left", va="bottom", fontsize=7, color="#666666")
    axb.errorbar(bx, bv, yerr=be, marker="o", ms=4.2, color=NEUTRAL, mfc=NEUTRAL,
                 mec=NEUTRAL, ecolor=NEUTRAL, elinewidth=0.9, capsize=2.5,
                 capthick=0.9, zorder=3)
    for x, v, n, hi in zip(bx, bv, bn, be[1]):
        axb.annotate(f"{v:.2f}\nn={n:,}", (x, v + hi), textcoords="offset points",
                     xytext=(0, 4), ha="center", va="bottom", fontsize=6.8,
                     color="#333333")
    # do not silently smooth the non-monotone tail: name it
    if len(bv) >= 2 and bv[-1] < bv[-2]:
        axb.annotate(f"{buckets[-1]} sits BELOW {buckets[-2]}\n(n={bn[-1]:,}; not smoothed)",
                     (bx[-1], bv[-1] - be[0][-1]), textcoords="offset points",
                     xytext=(-2, -7), ha="right", va="top", fontsize=6.8,
                     color="#8a5a00")

    # B1: the mode-balanced quantity is a BINARY contrast (depth>=2 vs depth==1),
    # NOT a per-frame slope -- "+X pts each" is wrong by 2-3x and is self-refuting
    # on a curve whose last point falls. The genuine per-extra-frame estimates
    # (question-FE coefficient, mode-stratified WLS slope) are printed alongside,
    # including whether the slope CI excludes zero.
    mv = p2.get("marginal_value") or {}
    satx = (p2x.get("saturation_stratified") or {})
    sat = (satx.get("mode_balanced") or {})
    extra_pts = mv.get("additional_beyond_first_pts", sat.get("diff_pts"))
    sat_ci = sat.get("ci95_boot_clustered_on_key")
    sat_p = sat.get("p_bootstrap", mv.get("p_additional_beyond_first"))
    mde = mv.get("additional_beyond_first_mde_pts", sat.get("mde_pts"))
    n_pairs, n_skeys = satx.get("n_pairs"), satx.get("n_keys")
    fe = (((p2x.get("conditional_model_question_fe") or {}).get("coef") or {})
          .get("extra_frames_(depth-1)") or {})
    wls = p2x.get("dose_slope_mode_stratified")
    wls_ci = p2x.get("dose_slope_mode_stratified_ci95")
    wls_excl = p2x.get("dose_slope_mode_stratified_excludes_zero")
    lines = []
    if extra_pts is not None:
        lines.append(f"$\\geq$2 vs exactly 1 in-window frame: "
                     f"{float(extra_pts):+.2f} pts")
        if sat_ci:
            lines.append(f"95% CI [{sat_ci[0]:.2f}, {sat_ci[1]:.2f}]"
                         + (f", MDE {float(mde):.2f} pts" if mde is not None else ""))
    if sat_p is not None:
        n_boot = satx.get("n_boot")
        floor = n_boot and abs(float(sat_p) - 1.0 / float(n_boot)) < 1e-12
        pstr = (f"p $\\leq$ {float(sat_p):.0e}" if floor else fmt_p(sat_p))
        lines.append(f"mode-stratified key-clustered bootstrap, {pstr}")
    if n_pairs and n_skeys:
        lines.append(f"n = {int(n_pairs):,} pairs / {int(n_skeys):,} questions"
                     + (f", {int(satx['n_boot']):,} draws" if satx.get("n_boot") else ""))
    if fe.get("coef_pts") is not None:
        lines.append(f"per extra frame: {float(fe['coef_pts']):+.2f} pts "
                     f"(question FE, {fmt_p(fe.get('p'))})")
    if wls is not None and wls_ci:
        lines.append(f"mode-strat. slope {r2(wls)}, CI "
                     f"[{wls_ci[0]:.2f}, {wls_ci[1]:.2f}] "
                     + ("excludes 0" if wls_excl else "includes 0"))
    if extra_pts is not None:
        lines.append("a STEP from 1 to $\\geq$2, not a per-frame slope")
    if mv.get("saturates") is not None:
        # B3: never print an unconditional verdict over an effect smaller than
        # its own minimum detectable effect (raw windows: +2.98 vs MDE 3.70).
        underpowered = (extra_pts is not None and mde is not None
                        and abs(float(extra_pts)) < float(mde))
        if underpowered:
            lines.append(f"UNDERPOWERED: |{float(extra_pts):+.2f}| < its own "
                         f"MDE {float(mde):.2f} pts, no verdict")
        else:
            lines.append("depth does NOT saturate" if not mv.get("saturates")
                         else "depth saturates")
    if lines:
        axb.text(0.03, 0.985, "\n".join(lines), transform=axb.transAxes,
                 ha="left", va="top", fontsize=6.0, color="#222222",
                 bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#bbbbbb", lw=0.6))

    axb.set_xticks(bx)
    axb.set_xticklabels([b + "\n(miss)" if b == "0" else b for b in buckets])
    axb.set_xlim(-0.72, len(buckets) - 0.35)
    lo_y = min(min(v - e for v, e in zip(bv, be[0])), CHANCE)
    hi_y = max(v + e for v, e in zip(bv, be[1]))
    axb.set_ylim(max(0.0, lo_y - 2.5), hi_y + (hi_y - lo_y) * 1.72 + 2.0)
    axb.set_xlabel("selected frames inside the evidence window")
    axb.set_ylabel("accuracy (%)")
    strip_ax(axb)
    axb.text(0.0, -0.36,
             f"n = {sum(bn):,} (question x backbone), 8 frames/arm{tag}\n"
             f"pooled over all four modes and {len(backbones)} backbones"
             "\noracle-window diagnostic, not a method",
             transform=axb.transAxes, fontsize=7, color="#555555", va="top")

    fig.subplots_adjust(left=0.085, right=0.99, top=0.96, bottom=0.35, wspace=0.30)
    return save(fig, out_dir, stem)


# ------------------------------------------------------------------ main ----
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--windows", choices=("tol", "raw"), default="tol",
                    help="tol (PRIMARY, tolerance-corrected windows; default) or "
                         "raw (un-widened windows, robustness check)")
    ap.add_argument("--stats-dir", default=here,
                    help="directory holding the lantern/cairn stats JSONs")
    ap.add_argument("--out-dir", default=None, help="where to write the figures")
    ap.add_argument("--prefix", default="fig_", help="output filename prefix")
    args = ap.parse_args(argv)

    suf = "_tol" if args.windows == "tol" else ""
    p1 = load_json(os.path.join(args.stats_dir, f"lantern_stats{suf}.json"),
                   f"lantern_stats{suf}.json")
    p2 = load_json(os.path.join(args.stats_dir, f"cairn_stats{suf}.json"),
                   f"cairn_stats{suf}.json")
    p2x = load_json(os.path.join(args.stats_dir, f"cairn_stats{suf}_extra.json"),
                    f"cairn_stats{suf}_extra.json", required=False)

    out_dir = args.out_dir or args.stats_dir
    os.makedirs(out_dir, exist_ok=True)
    stem_suf = "" if args.windows == "tol" else "_rawwindows"
    tag = "" if args.windows == "tol" else "\n[RAW windows: robustness check]"

    with plt.rc_context(RC):
        f1 = figure1(p1, out_dir, f"{args.prefix}lantern_evidence_mediation{stem_suf}", tag)
        f2 = figure2(p2, p2x, p1, out_dir,
                     f"{args.prefix}cairn_concentration_dose{stem_suf}", tag)

    for pdf, png in (f1, f2):
        print(f"wrote {pdf} ({os.path.getsize(pdf)} B) and {png} ({os.path.getsize(png)} B)")


if __name__ == "__main__":
    main()
