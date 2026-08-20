#!/usr/bin/env python
"""The single method-section figure for the ICLR submission: both paths' mechanism in three panels.

Authored at the FINAL printed size (5.5in = ICLR \\linewidth) rather than at an arbitrary size and
scaled by graphicx, so the tick/label point sizes in the PDF are the point sizes on the page.

(a) evidence-window hit rate by selection mode  -> Path 1 finds the evidence
(b) the exact mediation split                   -> ...but that is only a third of the gain
(c) hit rate against concentration              -> Path 2's granularity is the wrong one

Colours (Okabe-Ito, colourblind-safe), identical to the appendix figures:
  keyframe / Path 1  #0072B2   chunk / Path 2  #D55E00   random  #999999   uniform-8  #009E73

All three panels condition on ORACLE evidence windows: they are DIAGNOSTICS that explain the
measured gain, never methods and never an achievable accuracy.
"""
import json, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
})
D = str(Path(__file__).resolve().parent)
C = {"referent": "#0072B2", "chunk": "#D55E00", "random": "#999999", "uniform": "#009E73"}
# short tick labels: at 5.5in across three panels the full names collide, and the path/control
# identity is carried by colour (stated in the caption) plus panel (c)'s in-place annotations.
LBL = {"referent": "keyfr.", "chunk": "chunk", "random": "rand.", "uniform": "unif."}

p1 = json.load(open(f"{D}/lantern_stats_tol.json"))
p2 = json.load(open(f"{D}/cairn_stats_tol.json"))
med = p1["mediation"]["internvl3-14b"]
order = ["referent", "chunk", "random", "uniform"]

fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.85))

# ---- (a) hit rate ---------------------------------------------------------
a = ax[0]
hr = p1["hit_rate"]
v = [hr[m]["rate"] * 100 for m in order]
lo = [(hr[m]["rate"] - hr[m]["ci95"][0]) * 100 for m in order]
hi = [(hr[m]["ci95"][1] - hr[m]["rate"]) * 100 for m in order]
a.bar(range(4), v, yerr=[lo, hi], color=[C[m] for m in order], width=0.68,
      error_kw=dict(lw=0.7, capsize=2, capthick=0.7, ecolor="#333333"))
for i, x in enumerate(v):
    a.text(i, x + hi[i] + 1.6, f"{x:.1f}", ha="center", fontsize=6.5)
d = p1["hit_rate_delta"]["referent_minus_random"]
a.plot([0, 0, 2, 2], [76, 79, 79, 58], lw=0.6, c="#333333")
a.text(1.0, 80.5, f"+{d['delta']*100:.1f} pts", ha="center", fontsize=6.2)
a.set_xticks(range(4)); a.set_xticklabels([LBL[m] for m in order], fontsize=6.2)
a.set_ylabel("evidence-window hit rate (%)"); a.set_ylim(0, 95)
a.set_title("(a) it finds the evidence", fontsize=7, pad=3)

# ---- (b) mediation --------------------------------------------------------
b = ax[1]
terms = [("via hit rate", med["explained_by_hitrate"], med.get("explained_by_hitrate_ci95"), "#56B4E9"),
         ("via conditional\naccuracy", med["explained_by_cond_acc"], med.get("explained_by_cond_acc_ci95"), "#CC79A7"),
         ("total gain", med["total_gain"], med.get("total_gain_ci95"), "none")]
for i, (nm, val, ci, col) in enumerate(terms):
    y0 = 0 if i != 1 else terms[0][1] * 100
    h = val * 100
    b.bar(i, h, bottom=y0 if i == 1 else 0, width=0.6,
          color=col if col != "none" else "white", edgecolor="#333333", lw=0.7)
    if ci:
        b.errorbar(i, (y0 + h) if i == 1 else h, yerr=[[(h - (ci[0] * 100)) if i != 1 else (h - ci[0] * 100)],
                                                      [((ci[1] * 100) - h)]],
                   fmt="none", ecolor="#333333", lw=0.7, capsize=2, capthick=0.7)
    b.text(i, (y0 + h if i == 1 else h) + 0.75, f"{h:+.2f}", ha="center", fontsize=6.2)
b.plot([0.3, 0.7], [terms[0][1] * 100] * 2, ls=(0, (2, 2)), lw=0.6, c="#888888")
b.plot([1.3, 1.7], [(terms[0][1] + terms[1][1]) * 100] * 2, ls=(0, (2, 2)), lw=0.6, c="#888888")
b.set_xticks(range(3)); b.set_xticklabels(["hit rate", "cond.\naccuracy", "total"], fontsize=6.2)
b.set_ylabel("contribution to gain (pts)")
b.set_ylim(0, max(t[2][1] for t in terms if t[2]) * 100 * 1.42)
b.text(0.02, 0.97, f"{med['pct_mediated']:.1f}% mediated\n95% CI [{med['pct_mediated_ci95'][0]:.0f}, "
                   f"{med['pct_mediated_ci95'][1]:.0f}]", transform=b.transAxes, va="top", fontsize=6.2,
       bbox=dict(fc="white", ec="#bbbbbb", lw=0.5, pad=1.8))
b.set_title("(b) worth only a third", fontsize=7, pad=3)

# ---- (c) concentration ----------------------------------------------------
c = ax[2]
bd = p2["breadth_depth"]
xs = [bd[m]["mean_distinct_chunks"] for m in order]
ys = [bd[m]["hit_rate"] for m in order]
srt = np.argsort([-x for x in xs])
c.plot([xs[i] for i in srt], [ys[i] for i in srt], lw=0.8, c="#cccccc", zorder=1)
mk = {"referent": "o", "chunk": "s", "random": "^", "uniform": "D"}
for m in order:
    e = [(hr[m]["rate"] - hr[m]["ci95"][0]) * 100, (hr[m]["ci95"][1] - hr[m]["rate"]) * 100]
    c.errorbar(bd[m]["mean_distinct_chunks"], bd[m]["hit_rate"], yerr=[[e[0]], [e[1]]],
               fmt=mk[m], ms=4.5, c=C[m], ecolor=C[m], lw=0.7, capsize=1.6, capthick=0.6, zorder=3)
c.annotate("keyframe", (bd["referent"]["mean_distinct_chunks"], bd["referent"]["hit_rate"]),
           textcoords="offset points", xytext=(2, 5), fontsize=6, color=C["referent"])
c.annotate("chunk", (bd["chunk"]["mean_distinct_chunks"], bd["chunk"]["hit_rate"]),
           textcoords="offset points", xytext=(-20, -10), fontsize=6, color=C["chunk"])
c.annotate("random", (bd["random"]["mean_distinct_chunks"], bd["random"]["hit_rate"]),
           textcoords="offset points", xytext=(-6, -12), fontsize=6, color="#777777")
c.annotate("uniform-8", (bd["uniform"]["mean_distinct_chunks"], bd["uniform"]["hit_rate"]),
           textcoords="offset points", xytext=(-4, 6), fontsize=6, color=C["uniform"])
c.set_xlim(8.8, 1.2)   # inverted, with margin so the rightmost marker+label stay on canvas
c.set_xlabel("segments touched (of 8)\nconcentration $\\rightarrow$", fontsize=6.5)
c.set_ylabel("evidence-window hit rate (%)")
c.set_ylim(min(ys) - 7, max(ys) + 8)
c.set_title("(c) an optimal granularity", fontsize=7, pad=3)

for a_ in ax:
    a_.spines["top"].set_visible(False); a_.spines["right"].set_visible(False)
    a_.tick_params(width=0.6, length=2.5)

fig.subplots_adjust(left=0.078, right=0.985, top=0.875, bottom=0.365, wspace=0.46)
fig.text(0.5, 0.055, "$n$ = 2,962 questions with an oracle evidence window; every arm reads 8 frames "
                     "from one 64-frame pool.", ha="center", fontsize=5.6, color="#555555")
fig.text(0.5, 0.008, "Oracle windows are a DIAGNOSTIC that explains the gain, never a method.",
         ha="center", fontsize=5.6, color="#555555")
for ext in ("pdf", "png"):
    fig.savefig(f"{D}/fig_method_two_paths.{ext}", dpi=200)
print("wrote fig_method_two_paths.pdf/.png")
