#!/usr/bin/env python3
"""
CVPR-grade analysis plots for text-only VLM bias on MovieChat-1k.

Reads per-checkpoint predictions.jsonl from results_text_only_v2/ AND the
filtered benchmark (combined_all_hard_v3.json) and produces 8 publication
plots covering:
  1. overall_accuracy           - headline bar chart with random baseline
  2. per_capability_heatmap     - accuracy by (model, capability), full set
  3. scaling_curve              - accuracy vs log(params) per family
  4. option_letter_preference   - position/letter bias heatmap
  5. model_agreement            - pairwise Cohen's-kappa-like agreement
  6. debias_calibration_sweep   - tau vs (kept, mean_acc) trade-off
  7. before_after_debias        - per-model accuracy pre/post filter
  8. residual_bias_post_filter  - per-capability delta-from-baseline, debiased

All plots: serif typography, 300 DPI, PDF + PNG, colorblind-safe palette.

Usage:
    python plots_text_only_analysis.py --output_dir plots_text_only_analysis
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# -----------------------------------------------------------------------------
# Publication-grade style
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # editable text in PDF
    "ps.fonttype": 42,
})

# Colorblind-safe palette (Wong 2011)
FAMILY_COLORS = {
    "Qwen2.5-VL":  "#0072B2",   # blue
    "Qwen3-VL":    "#009E73",   # green
    "InternVL3":   "#D55E00",   # vermillion
    "Ovis2.5":     "#CC79A7",   # pink-magenta
    "Gemma3":      "#E69F00",   # orange
    "Video-LLaVA": "#56B4E9",   # sky blue
}


# -----------------------------------------------------------------------------
# Model metadata: parameter counts (B) and family
# -----------------------------------------------------------------------------
MODEL_META = {
    # key (display name)         : (params_B, family,        ckpt_dir)
    "Qwen2.5-VL-3B":   (3.0,  "Qwen2.5-VL",  "qwen2.5-vl-3b"),
    "Qwen2.5-VL-7B":   (7.0,  "Qwen2.5-VL",  "qwen2.5-vl-7b"),
    "Qwen3-VL-2B":     (2.0,  "Qwen3-VL",    "qwen3-vl-real-2b"),
    "Qwen3-VL-4B":     (4.0,  "Qwen3-VL",    "qwen3-vl-real-4b"),
    "Qwen3-VL-8B":     (8.0,  "Qwen3-VL",    "qwen3-vl-real-8b"),
    "InternVL3-2B":    (2.0,  "InternVL3",   "internvl3-2b"),
    "InternVL3-8B":    (8.0,  "InternVL3",   "internvl3-8b"),
    "InternVL3-14B":   (14.0, "InternVL3",   "internvl3-14b"),
    "Ovis2.5-2B":      (2.0,  "Ovis2.5",     "ovis2.5-2b"),
    "Ovis2.5-9B":      (9.0,  "Ovis2.5",     "ovis2.5-9b"),
    "Gemma3-4B":       (4.0,  "Gemma3",      "gemma3-4b"),
    "Gemma3-12B":      (12.0, "Gemma3",      "gemma3-12b"),
    "Video-LLaVA-7B":  (7.0,  "Video-LLaVA", "video-llava"),
}


N_OPTIONS = 8
RANDOM_BASELINE = 100.0 / N_OPTIONS  # 12.5


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_jsonl(path: Path):
    """Load JSONL, deduplicate on (video_id, question_id) keeping first."""
    if not path.exists():
        return []
    rows, seen = [], set()
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            k = (r.get("video_id"), r.get("question_id"))
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    return rows


def load_filtered_keys(bench_path: Path):
    """Return set of (video_id, question_id) kept by the committee filter,
       and per-q metadata dict."""
    with bench_path.open() as f:
        bench = json.load(f)
    kept_keys = set()
    metadata = {}
    for v in bench.get("videos", []):
        vid = v.get("video_id")
        if not vid:
            continue
        for q in v.get("questions", []):
            qid = q.get("question_id")
            if not qid:
                continue
            kept_keys.add((vid, qid))
            metadata[(vid, qid)] = q.get("metadata", {})
    return kept_keys, metadata


def compute_metrics(preds_dir: Path, filtered_keys, metadata):
    """For each known model, compute overall + per-capability accuracy on:
       (a) the FULL 7390 set, (b) the FILTERED set."""
    out = {}
    for display, (params_B, family, ckpt) in MODEL_META.items():
        jsonl = preds_dir / ckpt / "predictions.jsonl"
        rows = load_jsonl(jsonl)
        if not rows:
            print(f"WARN: no predictions for {display} at {jsonl}")
            continue

        full_correct = full_total = 0
        filt_correct = filt_total = 0
        cap_full = defaultdict(lambda: [0, 0])
        cap_filt = defaultdict(lambda: [0, 0])
        letter_counts = Counter()
        per_q = {}  # (vid,qid) -> predicted letter on filtered set
        for r in rows:
            k = (r.get("video_id"), r.get("question_id"))
            cap_full_label = r.get("capability", "unknown")
            pred = r.get("predicted", "")
            is_correct = r.get("is_correct", False)
            letter_counts[pred] += 1

            full_total += 1
            cap_full[cap_full_label][1] += 1
            if is_correct:
                full_correct += 1
                cap_full[cap_full_label][0] += 1

            if k in filtered_keys:
                cap_filt_label = metadata.get(k, {}).get("capability", cap_full_label)
                filt_total += 1
                cap_filt[cap_filt_label][1] += 1
                if is_correct:
                    filt_correct += 1
                    cap_filt[cap_filt_label][0] += 1
                per_q[k] = pred

        out[display] = {
            "params_B": params_B,
            "family": family,
            "ckpt": ckpt,
            "full_acc": 100.0 * full_correct / max(full_total, 1),
            "full_n": full_total,
            "full_correct": full_correct,
            "filt_acc": 100.0 * filt_correct / max(filt_total, 1),
            "filt_n": filt_total,
            "filt_correct": filt_correct,
            "cap_full": {c: 100.0*v[0]/v[1] for c, v in cap_full.items() if v[1]},
            "cap_filt": {c: 100.0*v[0]/v[1] for c, v in cap_filt.items() if v[1]},
            "cap_filt_n": {c: v[1] for c, v in cap_filt.items() if v[1]},
            "letter_counts": dict(letter_counts),
            "per_q": per_q,
        }
    return out


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def binom_ci(p_pct, n, conf=0.95):
    """Wilson 95% CI for a proportion. Returns half-width in pct points."""
    if n <= 0:
        return 0.0
    from math import sqrt
    p = p_pct / 100.0
    z = 1.96 if conf == 0.95 else 2.576
    denom = 1.0 + z*z/n
    half = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return half * 100.0


def plot_overall_accuracy(metrics, out_dir, kind="full", wide=False):
    """Horizontal bar chart with random-baseline line and Wilson CIs.
       wide=True: force x-limit to the same 0-40% scale as the full-set plot, so
       readers don't read tight auto-scaled bars as a bigger effect than it is."""
    rows = sorted(metrics.items(),
                  key=lambda x: x[1][f"{kind}_acc"], reverse=True)
    names = [r[0] for r in rows]
    accs = [r[1][f"{kind}_acc"] for r in rows]
    ns = [r[1][f"{kind}_n"] for r in rows]
    cis = [binom_ci(a, n) for a, n in zip(accs, ns)]
    colors = [FAMILY_COLORS[r[1]["family"]] for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ypos = np.arange(len(names))
    bars = ax.barh(ypos, accs, xerr=cis, color=colors,
                   edgecolor="black", linewidth=0.6,
                   error_kw={"ecolor": "black", "elinewidth": 0.9, "capsize": 2.5})
    ax.axvline(RANDOM_BASELINE, color="#555555", linestyle="--", linewidth=1.0,
               label=f"Random ({RANDOM_BASELINE:.1f}%)")
    for i, (a, n) in enumerate(zip(accs, ns)):
        ax.text(a + 0.6, i, f"{a:.2f}%", va="center", fontsize=9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Text-only score (%)")
    base_title = ("Text-only score on the full 7390-question benchmark"
                  if kind == "full"
                  else f"Text-only score on committee-filtered set ({rows[0][1]['filt_n']} questions)")
    if wide:
        base_title += "\n(scaled to 0-40% for visual comparability with the pre-debias plot)"
    ax.set_title(base_title, pad=8)

    # Family legend (only families present)
    fams = []
    for n in names:
        f = metrics[n]["family"]
        if f not in fams:
            fams.append(f)
    handles = [mpl.patches.Patch(facecolor=FAMILY_COLORS[f], edgecolor="black",
                                 linewidth=0.6, label=f) for f in fams]
    handles.append(mpl.lines.Line2D([0], [0], color="#555555", linestyle="--",
                                    label=f"Random ({RANDOM_BASELINE:.1f}%)"))
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False)
    if wide:
        ax.set_xlim(0, 40)
    else:
        ax.set_xlim(0, max(accs) * 1.18)
    suffix = "_wide" if wide else ""
    save(fig, out_dir, f"01_overall_accuracy_{kind}{suffix}")


def plot_per_capability_heatmap(metrics, out_dir, kind="full"):
    """Heatmap (rows = models, cols = capabilities), diverging cmap at baseline.
       Drops capabilities with fewer than MIN_N samples (too noisy to plot)."""
    MIN_N = 20
    rows = sorted(metrics.items(),
                  key=lambda x: x[1][f"{kind}_acc"], reverse=True)
    names = [r[0] for r in rows]
    # Use filtered-set sample counts when available; for full we compute from any model
    cap_n = defaultdict(int)
    for n, m in rows:
        # take counts from cap_filt_n if filtered, else recompute approx from cap_full
        if kind == "filt":
            for c, nn in m.get("cap_filt_n", {}).items():
                cap_n[c] = max(cap_n[c], nn)
        else:
            cap_n[c] if False else None  # placeholder
    if kind == "full":
        # Recover full sample counts: use any one model's prediction count for each cap
        any_model = rows[0][0]
        ckpt = MODEL_META[any_model][2]
        cnt = Counter()
        with (Path("results_text_only_v2") / ckpt / "predictions.jsonl").open() as f:
            seen = set()
            for line in f:
                r = json.loads(line)
                k = (r.get("video_id"), r.get("question_id"))
                if k in seen: continue
                seen.add(k)
                cnt[r.get("capability", "unknown")] += 1
        cap_n = cnt
    caps = sorted([c for c, n in cap_n.items() if n >= MIN_N])

    M = np.zeros((len(names), len(caps)))
    for i, (n, m) in enumerate(rows):
        for j, c in enumerate(caps):
            M[i, j] = m[f"cap_{kind}"].get(c, np.nan)

    vmax = max(np.nanmax(M), RANDOM_BASELINE + 1.0)
    vmin = min(np.nanmin(M), RANDOM_BASELINE - 1.0)
    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=RANDOM_BASELINE, vmax=vmax)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    fig, ax = plt.subplots(figsize=(1.3 * len(caps) + 3.0, 0.42 * len(names) + 2.2))
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(names)):
        for j in range(len(caps)):
            v = M[i, j]
            color = "black" if abs(v - RANDOM_BASELINE) < 5 else "white"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color=color, fontsize=9)
    cap_labels = [f"{c}\n(n={cap_n[c]})" for c in caps]
    ax.set_xticks(range(len(caps)))
    ax.set_xticklabels(cap_labels, rotation=0, ha="center")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_title(("Per-capability text-only accuracy "
                  + ("(full set, n=7390)" if kind == "full"
                     else f"(committee-filtered set, n={rows[0][1]['filt_n']})")),
                 pad=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Accuracy (%)")
    cbar.ax.axhline(RANDOM_BASELINE, color="black", linewidth=0.8)
    ax.grid(False)
    save(fig, out_dir, f"02_per_capability_heatmap_{kind}")


def plot_scaling_curve(metrics, out_dir, kind="full", wide=False):
    """Accuracy vs log(params), one line per family.
       wide=True forces y-axis to [0, 40] so post-debias differences are read
       on the same scale as the pre-debias plot - prevents visual exaggeration."""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    families = defaultdict(list)
    for n, m in metrics.items():
        families[m["family"]].append((m["params_B"], m[f"{kind}_acc"], n))
    for fam, pts in families.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        labels = [p[2] for p in pts]
        marker = "o"
        ax.plot(xs, ys, marker=marker, markersize=8,
                color=FAMILY_COLORS[fam], linewidth=1.6,
                label=fam, markeredgecolor="black", markeredgewidth=0.8)
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab.replace(fam + "-", ""), (x, y),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=8, color=FAMILY_COLORS[fam])
    ax.axhline(RANDOM_BASELINE, color="#555555", linestyle="--", linewidth=1.0,
               label=f"Random ({RANDOM_BASELINE:.1f}%)")
    ax.set_xscale("log")
    ax.set_xlabel("Model size (parameters, B)  - log scale")
    ax.set_ylabel("Text-only score (%)")
    base_title = ("Scaling of text-only leakage  -  larger models leak more"
                  if kind == "full"
                  else "Residual text-only leakage vs scale (filtered set)")
    if wide:
        base_title += "\n(y-axis fixed to 0-40% for comparability with the pre-debias plot)"
    ax.set_title(base_title, pad=8)
    ax.legend(loc="upper left", ncol=2)
    ax.set_xticks([2, 3, 4, 7, 8, 12, 14])
    ax.set_xticklabels(["2B", "3B", "4B", "7B", "8B", "12B", "14B"])
    ax.get_xaxis().set_minor_locator(mpl.ticker.NullLocator())
    if wide:
        ax.set_ylim(0, 40)
    suffix = "_wide" if wide else ""
    save(fig, out_dir, f"03_scaling_curve_{kind}{suffix}")


def plot_letter_preference(metrics, out_dir):
    """Heatmap: models x letter A-H, % of predictions on each letter (full set)."""
    rows = sorted(metrics.items(),
                  key=lambda x: x[1]["full_acc"], reverse=True)
    names = [r[0] for r in rows]
    letters = list("ABCDEFGH")
    M = np.zeros((len(names), len(letters)))
    for i, (n, m) in enumerate(rows):
        total = sum(m["letter_counts"].values()) or 1
        for j, L in enumerate(letters):
            M[i, j] = 100.0 * m["letter_counts"].get(L, 0) / total
    uniform = 100.0 / len(letters)
    norm = mpl.colors.TwoSlopeNorm(vmin=min(M.min(), uniform-1),
                                   vcenter=uniform,
                                   vmax=max(M.max(), uniform+1))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(names) + 2.2))
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(names)):
        for j in range(len(letters)):
            v = M[i, j]
            color = "black" if abs(v - uniform) < 4 else "white"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color=color, fontsize=9)
    ax.set_xticks(range(len(letters)))
    ax.set_xticklabels(letters)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Letter chosen")
    ax.set_title(f"Option-letter preference (uniform = {uniform:.1f}%)", pad=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Prediction share (%)")
    ax.grid(False)
    save(fig, out_dir, "04_option_letter_preference")


def plot_model_agreement(metrics, out_dir):
    """Pairwise agreement matrix on the FULL set (fraction of questions where
       both models picked the same letter)."""
    rows = sorted(metrics.items(),
                  key=lambda x: x[1]["full_acc"], reverse=True)
    names = [r[0] for r in rows]
    # Build aligned letter arrays
    preds_map = {}
    for n, m in rows:
        ckpt = m["ckpt"]
        jl = Path("results_text_only_v2") / ckpt / "predictions.jsonl"
        seen = set()
        d = {}
        with jl.open() as f:
            for line in f:
                r = json.loads(line)
                k = (r.get("video_id"), r.get("question_id"))
                if k in seen:
                    continue
                seen.add(k)
                d[k] = r.get("predicted", "")
        preds_map[n] = d
    common = set.intersection(*[set(d.keys()) for d in preds_map.values()])
    common = sorted(common)
    arr = np.array([[preds_map[n][k] for k in common] for n in names])
    N = len(names)
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M[i, j] = 100.0 * np.mean(arr[i] == arr[j])
    fig, ax = plt.subplots(figsize=(1.0 * N + 1.4, 0.95 * N + 1.4))
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    im = ax.imshow(M, cmap=cmap, vmin=12.5, vmax=100, aspect="equal")
    for i in range(N):
        for j in range(N):
            color = "white" if M[i, j] > 55 else "black"
            ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center",
                    fontsize=8, color=color)
    ax.set_xticks(range(N)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(N)); ax.set_yticklabels(names)
    ax.set_title("Pairwise agreement of predicted letters (full set)", pad=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Agreement (%)")
    ax.grid(False)
    save(fig, out_dir, "05_model_agreement")


def plot_calibration_sweep(out_dir):
    """Show how tau_correct trades off kept-count vs mean committee accuracy."""
    report_path = Path("combined_all_hard_v3_debias_report.json")
    if not report_path.exists():
        print(f"skip calibration plot - {report_path} not found")
        return
    rep = json.load(open(report_path))
    sweep = rep["sweep"]
    taus = [r["tau_correct"] for r in sweep]
    kept = [r["kept"] for r in sweep]
    mean_acc = [r["mean_acc"] for r in sweep]
    chosen = rep["tau_correct"]
    target = rep["target_acc"]

    fig, ax1 = plt.subplots(figsize=(6.6, 4.2))
    ax1.set_xlabel(r"$\tau_{\mathrm{correct}}$  (drop a question if $\geq\tau$ models guess it correctly)")
    color1 = "#0072B2"
    color2 = "#D55E00"
    ax1.plot(taus, kept, marker="o", color=color1, linewidth=1.8,
             markeredgecolor="black", markeredgewidth=0.8, label="Questions kept")
    ax1.set_ylabel("Questions kept", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.invert_xaxis()

    ax2 = ax1.twinx()
    ax2.spines.right.set_visible(True)
    ax2.plot(taus, mean_acc, marker="s", color=color2, linewidth=1.8,
             markeredgecolor="black", markeredgewidth=0.8, label="Mean committee acc.")
    ax2.set_ylabel("Mean committee accuracy (%)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.axhline(target, color=color2, linestyle=":", linewidth=1.0, alpha=0.7)
    ax2.axhline(RANDOM_BASELINE, color="#555555", linestyle="--", linewidth=1.0,
                alpha=0.7, label=f"Random ({RANDOM_BASELINE:.1f}%)")

    ax1.axvline(chosen, color="black", linestyle="-.", linewidth=1.0, alpha=0.7)
    ax1.text(chosen, max(kept) * 0.92, f" chosen $\\tau$ = {chosen}",
             fontsize=10, color="black")

    ax1.set_title("Committee-filter calibration sweep", pad=8)
    ax1.grid(True, alpha=0.25)
    save(fig, out_dir, "06_calibration_sweep")


def plot_before_after(metrics, out_dir):
    """Grouped bar: each model gets two bars (full vs filtered)."""
    rows = sorted(metrics.items(), key=lambda x: x[1]["full_acc"], reverse=True)
    names = [r[0] for r in rows]
    full = [r[1]["full_acc"] for r in rows]
    filt = [r[1]["filt_acc"] for r in rows]
    N = len(names)
    x = np.arange(N)
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(8.0, 0.6 * N), 4.6))
    b1 = ax.bar(x - w/2, full, w, color="#D55E00", edgecolor="black", linewidth=0.6,
                label=f"Before debias (n=7390)")
    b2 = ax.bar(x + w/2, filt, w, color="#0072B2", edgecolor="black", linewidth=0.6,
                label=f"After committee filter (n={rows[0][1]['filt_n']})")
    ax.axhline(RANDOM_BASELINE, color="#555555", linestyle="--", linewidth=1.0,
               label=f"Random ({RANDOM_BASELINE:.1f}%)")
    for xi, fa, fi in zip(x, full, filt):
        ax.text(xi - w/2, fa + 0.5, f"{fa:.1f}", ha="center", fontsize=7.5)
        ax.text(xi + w/2, fi + 0.5, f"{fi:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Text-only accuracy (%)")
    ax.set_title("Effect of adversarial committee filtering on text-only leakage", pad=8)
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(full) * 1.18)
    save(fig, out_dir, "07_before_after_debias")


def plot_scaling_overlay(metrics, out_dir, wide=False):
    """Side-by-side full vs filtered scaling curves on the SAME y axis,
       so the reader can see (a) absolute drop after debias and
       (b) residual slope. wide=True uses y=[0,40]."""
    fams = defaultdict(list)
    for n, m in metrics.items():
        fams[m["family"]].append((m["params_B"], m["full_acc"], m["filt_acc"], n))
    for k in fams:
        fams[k].sort()

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for fam, pts in fams.items():
        xs = [p[0] for p in pts]
        full_ys = [p[1] for p in pts]
        filt_ys = [p[2] for p in pts]
        color = FAMILY_COLORS[fam]
        ax.plot(xs, full_ys, marker="o", markersize=8, linestyle="-",
                color=color, linewidth=1.8, markeredgecolor="black",
                markeredgewidth=0.8, label=f"{fam} (full)")
        ax.plot(xs, filt_ys, marker="s", markersize=8, linestyle=":",
                color=color, linewidth=1.4, markeredgecolor="black",
                markeredgewidth=0.6, alpha=0.95, label=f"{fam} (filt)")
        # Shaded gap = leakage closed
        ax.fill_between(xs, full_ys, filt_ys, color=color, alpha=0.07,
                        linewidth=0)
    ax.axhline(RANDOM_BASELINE, color="#555555", linestyle="--", linewidth=1.0,
               label=f"Random ({RANDOM_BASELINE:.1f}%)")
    ax.set_xscale("log")
    ax.set_xlabel("Model size (parameters, B)  - log scale")
    ax.set_ylabel("Text-only score (%)")
    title = ("Scaling effect on text-only leakage: full benchmark (solid) "
             "vs committee-filtered (dotted)")
    if wide:
        title += "\n(y-axis 0-40% to match pre-debias scale)"
    ax.set_title(title, pad=8)
    ax.set_xticks([2, 3, 4, 7, 8, 12, 14])
    ax.set_xticklabels(["2B", "3B", "4B", "7B", "8B", "12B", "14B"])
    ax.get_xaxis().set_minor_locator(mpl.ticker.NullLocator())
    if wide:
        ax.set_ylim(0, 40)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1,
              frameon=False, fontsize=9)
    suffix = "_wide" if wide else ""
    save(fig, out_dir, f"09_scaling_overlay{suffix}")


def plot_scaling_slopes(metrics, out_dir):
    """For each family with >=2 checkpoints, fit acc = a + b*log10(params).
       Plot the slope b (pp per 10x params). This summarises 'how much
       does scaling worsen the leakage?' in one number per family."""
    fams = defaultdict(list)
    for n, m in metrics.items():
        fams[m["family"]].append((m["params_B"], m["full_acc"], m["filt_acc"]))
    rows = []
    for fam, pts in fams.items():
        if len(pts) < 2:
            continue
        xs = np.log10(np.array([p[0] for p in pts]))
        full = np.array([p[1] for p in pts])
        filt = np.array([p[2] for p in pts])
        b_full = np.polyfit(xs, full, 1)[0]
        b_filt = np.polyfit(xs, filt, 1)[0]
        rows.append((fam, b_full, b_filt, len(pts)))
    rows.sort(key=lambda r: -r[1])

    names = [r[0] for r in rows]
    full_slopes = [r[1] for r in rows]
    filt_slopes = [r[2] for r in rows]
    colors = [FAMILY_COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(names))
    w = 0.4
    ax.bar(x - w/2, full_slopes, w, color=colors, edgecolor="black", linewidth=0.6,
           label="Full benchmark (pre-debias)")
    ax.bar(x + w/2, filt_slopes, w, color=colors, edgecolor="black", linewidth=0.6,
           alpha=0.45, label="Committee-filtered (post-debias)", hatch="//")
    ax.axhline(0, color="black", linewidth=0.7)
    for xi, b in zip(x - w/2, full_slopes):
        ax.text(xi, b + (0.2 if b >= 0 else -0.6), f"{b:+.1f}",
                ha="center", fontsize=8, fontweight="bold")
    for xi, b in zip(x + w/2, filt_slopes):
        ax.text(xi, b + (0.2 if b >= 0 else -0.6), f"{b:+.1f}",
                ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Scaling slope  (pp per 10x params)")
    ax.set_title("Per-family scaling slope of text-only leakage\n"
                 r"$\Delta$ score per decade of model size", pad=8)
    ax.legend(loc="upper right")
    save(fig, out_dir, "10_scaling_slopes")


def plot_scaling_per_capability(metrics, out_dir, kind="full", wide=False):
    """5 small-multiples (one per capability) showing scaling within each."""
    MIN_N = 20
    cap_n = defaultdict(int)
    for _, m in metrics.items():
        for c, nn in m.get("cap_filt_n", {}).items():
            cap_n[c] = max(cap_n[c], nn)
    caps = sorted([c for c, n in cap_n.items() if n >= MIN_N])

    fams = defaultdict(list)
    for n, m in metrics.items():
        fams[m["family"]].append((m["params_B"], m, n))
    for f in fams:
        fams[f].sort()

    n_caps = len(caps)
    fig, axes = plt.subplots(1, n_caps, figsize=(2.8 * n_caps, 3.6),
                             sharey=True)
    if n_caps == 1:
        axes = [axes]
    for ax_idx, c in enumerate(caps):
        ax = axes[ax_idx]
        for fam, pts in fams.items():
            xs, ys = [], []
            for p_B, m, name in pts:
                v = m[f"cap_{kind}"].get(c)
                if v is None:
                    continue
                xs.append(p_B); ys.append(v)
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", markersize=6, color=FAMILY_COLORS[fam],
                    linewidth=1.5, markeredgecolor="black", markeredgewidth=0.6,
                    label=fam)
        ax.axhline(RANDOM_BASELINE, color="#555555", linestyle="--", linewidth=0.9)
        ax.set_xscale("log")
        ax.set_xticks([2, 4, 8, 14])
        ax.set_xticklabels(["2B", "4B", "8B", "14B"])
        ax.get_xaxis().set_minor_locator(mpl.ticker.NullLocator())
        ax.set_title(f"{c}\n(n={cap_n[c]})", fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Text-only score (%)")
        ax.set_xlabel("params")
        if wide:
            ax.set_ylim(0, 40)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
    suptitle = ("Per-capability scaling " +
                ("(full set)" if kind == "full" else "(filtered set)"))
    if wide:
        suptitle += "    y=[0,40]"
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    suffix = "_wide" if wide else ""
    save(fig, out_dir, f"11_scaling_per_capability_{kind}{suffix}")


def plot_scaling_buckets(metrics, out_dir, kind="full", wide=False):
    """Group models into size buckets (~2B / 3-4B / 7-9B / 12-14B) and show
       mean +/- range per family within each bucket. Makes 'does scale matter
       for THIS family' very legible."""
    def bucket(p):
        if p <= 2.5:   return "~2B"
        if p <= 4.5:   return "~3-4B"
        if p <= 9.5:   return "~7-9B"
        return "~12-15B"
    buckets = ["~2B", "~3-4B", "~7-9B", "~12-15B"]
    families = sorted({m["family"] for m in metrics.values()})

    grid = {f: {b: [] for b in buckets} for f in families}
    for n, m in metrics.items():
        grid[m["family"]][bucket(m["params_B"])].append(m[f"{kind}_acc"])

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(buckets))
    n_fam = len(families)
    w = 0.78 / n_fam
    for i, fam in enumerate(families):
        means = []
        for b in buckets:
            vals = grid[fam][b]
            means.append(np.mean(vals) if vals else np.nan)
        ax.bar(x + (i - (n_fam - 1) / 2) * w, means, w, color=FAMILY_COLORS[fam],
               edgecolor="black", linewidth=0.6, label=fam)
        for xi, mv in zip(x + (i - (n_fam - 1) / 2) * w, means):
            if not np.isnan(mv):
                ax.text(xi, mv + 0.5, f"{mv:.1f}", ha="center", fontsize=7.5)
    ax.axhline(RANDOM_BASELINE, color="#555555", linestyle="--", linewidth=1.0,
               label=f"Random ({RANDOM_BASELINE:.1f}%)")
    ax.set_xticks(x); ax.set_xticklabels(buckets)
    ax.set_ylabel("Text-only score (%)")
    title = ("Scaling effect by family and size bucket "
             + ("(full set)" if kind == "full" else "(filtered set)"))
    if wide:
        title += "    y=[0,40]"
        ax.set_ylim(0, 40)
    ax.set_title(title, pad=8)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    suffix = "_wide" if wide else ""
    save(fig, out_dir, f"12_scaling_buckets_{kind}{suffix}")


def plot_residual_bias(metrics, out_dir):
    """Heatmap of (acc - baseline) per (model, capability) on filtered set,
       diverging at 0. Skips tiny categories."""
    MIN_N = 20
    rows = sorted(metrics.items(),
                  key=lambda x: x[1]["filt_acc"], reverse=True)
    names = [r[0] for r in rows]
    cap_n = defaultdict(int)
    for _, m in rows:
        for c, nn in m.get("cap_filt_n", {}).items():
            cap_n[c] = max(cap_n[c], nn)
    caps = sorted([c for c, n in cap_n.items() if n >= MIN_N])
    M = np.zeros((len(names), len(caps)))
    for i, (n, m) in enumerate(rows):
        for j, c in enumerate(caps):
            M[i, j] = m["cap_filt"].get(c, np.nan) - RANDOM_BASELINE
    vmax = max(abs(np.nanmin(M)), abs(np.nanmax(M)), 2)
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    fig, ax = plt.subplots(figsize=(1.3 * len(caps) + 3.0, 0.42 * len(names) + 2.2))
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(names)):
        for j in range(len(caps)):
            v = M[i, j]
            sign = "+" if v >= 0 else ""
            ax.text(j, i, f"{sign}{v:.1f}", ha="center", va="center",
                    color="black" if abs(v) < vmax * 0.55 else "white", fontsize=9)
    cap_labels = [f"{c}\n(n={cap_n[c]})" for c in caps]
    ax.set_xticks(range(len(caps)))
    ax.set_xticklabels(cap_labels, rotation=0, ha="center")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_title("Residual text-only leakage after committee filter\n"
                 r"$\Delta$ vs. 12.5% baseline (filtered set)", pad=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(r"$\Delta$ accuracy (pp)")
    ax.grid(False)
    save(fig, out_dir, "08_residual_bias_post_filter")


# -----------------------------------------------------------------------------
# Save helper
# -----------------------------------------------------------------------------
def save(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    print(f"  saved {name}.pdf + .png")


# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds_dir", default="results_text_only_v2")
    p.add_argument("--bench_filtered", default="combined_all_hard_v3.json")
    p.add_argument("--output_dir", default="plots_text_only_analysis")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filtered_keys, metadata = load_filtered_keys(Path(args.bench_filtered))
    print(f"Filtered set: {len(filtered_keys)} questions")

    metrics = compute_metrics(Path(args.preds_dir), filtered_keys, metadata)
    print(f"Loaded {len(metrics)} model checkpoints\n")

    plot_overall_accuracy(metrics, out_dir, kind="full")
    plot_overall_accuracy(metrics, out_dir, kind="filt")
    plot_overall_accuracy(metrics, out_dir, kind="filt", wide=True)
    plot_per_capability_heatmap(metrics, out_dir, kind="full")
    plot_per_capability_heatmap(metrics, out_dir, kind="filt")
    plot_scaling_curve(metrics, out_dir, kind="full")
    plot_scaling_curve(metrics, out_dir, kind="filt")
    plot_scaling_curve(metrics, out_dir, kind="filt", wide=True)
    plot_letter_preference(metrics, out_dir)
    plot_model_agreement(metrics, out_dir)
    plot_calibration_sweep(out_dir)
    plot_before_after(metrics, out_dir)
    plot_residual_bias(metrics, out_dir)
    # New scaling-focused analyses
    plot_scaling_overlay(metrics, out_dir, wide=False)
    plot_scaling_overlay(metrics, out_dir, wide=True)
    plot_scaling_slopes(metrics, out_dir)
    plot_scaling_per_capability(metrics, out_dir, kind="full")
    plot_scaling_per_capability(metrics, out_dir, kind="filt")
    plot_scaling_per_capability(metrics, out_dir, kind="filt", wide=True)
    plot_scaling_buckets(metrics, out_dir, kind="full")
    plot_scaling_buckets(metrics, out_dir, kind="filt")
    plot_scaling_buckets(metrics, out_dir, kind="filt", wide=True)

    # Dump CSV of all numbers for the paper appendix
    rows = []
    for name, m in metrics.items():
        rows.append({
            "model": name, "family": m["family"], "params_B": m["params_B"],
            "full_acc": m["full_acc"], "full_n": m["full_n"],
            "filt_acc": m["filt_acc"], "filt_n": m["filt_n"],
            **{f"full_cap_{c}": v for c, v in m["cap_full"].items()},
            **{f"filt_cap_{c}": v for c, v in m["cap_filt"].items()},
        })
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    print(f"\nsummary.csv saved")
    print(f"\nAll plots written to: {out_dir}/")


if __name__ == "__main__":
    main()
