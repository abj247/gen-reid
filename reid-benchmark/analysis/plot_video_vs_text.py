#!/usr/bin/env python3
"""
CVPR-grade plots comparing VIDEO+TEXT vs TEXT-ONLY on the committee-debiased
MovieChat-1k benchmark (3667 questions, random baseline 12.5%).

Video predictions live in results_video_v2/ keyed by REAL YouTube IDs;
text-only predictions live in results_text_only_v2/ keyed by ANONYMIZED ids.
We map real->anon via video_id_mapping.json and match per (anon_id, qid).

Plots:
  v01_overall_video             - video+text accuracy bar chart
  v02_per_capability_video      - accuracy by (model, capability), video+text
  v03_scaling_video             - video+text accuracy vs log(params) by family
  v04_video_gain                - HEADLINE diverging bars: (video - text) per model
  v05_paired_slope              - slope graph text-only -> video+text per model
  v06_video_gain_by_capability  - heatmap of video gain per (model, capability)
  v07_scatter_text_vs_video     - scatter x=text-only y=video+text, diag=no-gain

Usage:
    python plots_video_vs_text.py --output_dir plots_video_vs_text
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.linewidth": 0.9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

FAMILY_COLORS = {
    "Qwen2.5-VL": "#0072B2", "Qwen3-VL": "#009E73", "InternVL3": "#D55E00",
    "Ovis2.5": "#CC79A7", "Gemma3": "#E69F00", "Video-LLaVA": "#56B4E9",
}

# ckpt dir -> (display, params_B, family)
MODELS = {
    "ovis2.5-2b":      ("Ovis2.5-2B", 2.0, "Ovis2.5"),
    "internvl3-2b":    ("InternVL3-2B", 2.0, "InternVL3"),
    "qwen3-vl-real-2b":("Qwen3-VL-2B", 2.0, "Qwen3-VL"),
    "qwen2.5-vl-3b":   ("Qwen2.5-VL-3B", 3.0, "Qwen2.5-VL"),
    "qwen3-vl-real-4b":("Qwen3-VL-4B", 4.0, "Qwen3-VL"),
    "gemma3-4b":       ("Gemma3-4B", 4.0, "Gemma3"),
    "qwen2.5-vl-7b":   ("Qwen2.5-VL-7B", 7.0, "Qwen2.5-VL"),
    "qwen3-vl-real-8b":("Qwen3-VL-8B", 8.0, "Qwen3-VL"),
    "ovis2.5-9b":      ("Ovis2.5-9B", 9.0, "Ovis2.5"),
    "internvl3-8b":    ("InternVL3-8B", 8.0, "InternVL3"),
    "video-llava":     ("Video-LLaVA-7B", 7.0, "Video-LLaVA"),
    "gemma3-12b":      ("Gemma3-12B", 12.0, "Gemma3"),
    "internvl3-14b":   ("InternVL3-14B", 14.0, "InternVL3"),
}

N_OPT = 8
BASE = 100.0 / N_OPT


def save(fig, out_dir, name):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf"); fig.savefig(out_dir / f"{name}.png")
    plt.close(fig); print(f"  saved {name}")


def load_jsonl(path, key_map=None):
    """Return {(anon_id, qid): is_correct}. If key_map given, map video_id via it."""
    d = {}; seen = set()
    if not Path(path).exists():
        return d
    for line in open(path):
        r = json.loads(line)
        vid = r.get("video_id")
        if key_map is not None:
            vid = key_map.get(vid, vid)
        k = (vid, r.get("question_id"))
        if k in seen:
            continue
        seen.add(k)
        d[k] = bool(r.get("is_correct", False))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", default="results_video_v2")
    ap.add_argument("--text_dir", default="results_text_only_v2")
    ap.add_argument("--bench", default="combined_all_hard_v3.json")
    ap.add_argument("--mapping", default="video_id_mapping.json")
    ap.add_argument("--output_dir", default="plots_video_vs_text")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    real_to_anon = json.load(open(args.mapping))["real_to_anon"]

    # capability metadata per (anon_id, qid)
    bench = json.load(open(args.bench))
    cap_of = {}
    for v in bench["videos"]:
        vid = v.get("video_id")
        for q in v.get("questions", []):
            cap_of[(vid, q.get("question_id"))] = q.get("metadata", {}).get("capability", "unknown")

    # Build matched metrics
    M = {}  # display -> dict
    for ckpt, (disp, pB, fam) in MODELS.items():
        vid = load_jsonl(f"{args.video_dir}/{ckpt}/predictions.jsonl", key_map=real_to_anon)
        txt = load_jsonl(f"{args.text_dir}/{ckpt}/predictions.jsonl")
        common = sorted(set(vid) & set(txt))
        if not common:
            print(f"WARN no common keys for {disp}")
            continue
        v_acc = 100.0 * sum(vid[k] for k in common) / len(common)
        t_acc = 100.0 * sum(txt[k] for k in common) / len(common)
        # per-capability
        capv = defaultdict(lambda: [0, 0]); capt = defaultdict(lambda: [0, 0])
        for k in common:
            c = cap_of.get(k, "unknown")
            capv[c][1] += 1; capt[c][1] += 1
            if vid[k]: capv[c][0] += 1
            if txt[k]: capt[c][0] += 1
        M[disp] = {
            "params_B": pB, "family": fam, "n": len(common),
            "v_acc": v_acc, "t_acc": t_acc, "gain": v_acc - t_acc,
            "cap_v": {c: 100.0*a[0]/a[1] for c, a in capv.items() if a[1] >= 20},
            "cap_t": {c: 100.0*a[0]/a[1] for c, a in capt.items() if a[1] >= 20},
            "cap_n": {c: a[1] for c, a in capv.items() if a[1] >= 20},
        }
    print(f"Loaded {len(M)} models")

    # ---- v01 overall video+text ----
    rows = sorted(M.items(), key=lambda x: -x[1]["v_acc"])
    names = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    y = np.arange(len(names))
    ax.barh(y, [r[1]["v_acc"] for r in rows],
            color=[FAMILY_COLORS[r[1]["family"]] for r in rows],
            edgecolor="black", linewidth=0.6)
    ax.axvline(BASE, color="#555555", ls="--", lw=1.0)
    for i, r in enumerate(rows):
        ax.text(r[1]["v_acc"] + 0.3, i, f"{r[1]['v_acc']:.1f}%", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlabel("Video+Text score (%)")
    ax.set_title("Video+Text accuracy on committee-debiased set "
                 f"(n={rows[0][1]['n']}, baseline {BASE:.1f}%)", pad=8)
    fams = []
    for r in rows:
        if r[1]["family"] not in fams: fams.append(r[1]["family"])
    handles = [mpl.patches.Patch(facecolor=FAMILY_COLORS[f], edgecolor="black",
               linewidth=0.6, label=f) for f in fams]
    handles.append(mpl.lines.Line2D([0],[0],color="#555555",ls="--",label=f"Random ({BASE:.1f}%)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5,-0.13), ncol=4)
    ax.set_xlim(0, max(r[1]["v_acc"] for r in rows)*1.18)
    save(fig, out, "v01_overall_video")

    # ---- v04 HEADLINE video gain diverging bars ----
    rows_g = sorted(M.items(), key=lambda x: x[1]["gain"])
    names_g = [r[0] for r in rows_g]
    gains = [r[1]["gain"] for r in rows_g]
    colors = [FAMILY_COLORS[r[1]["family"]] for r in rows_g]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y = np.arange(len(names_g))
    ax.barh(y, gains, color=colors, edgecolor="black", linewidth=0.6)
    ax.axvline(0, color="black", lw=0.9)
    for i, g in enumerate(gains):
        ax.text(g + (0.08 if g >= 0 else -0.08), i, f"{g:+.1f}", va="center",
                ha="left" if g >= 0 else "right", fontsize=9, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(names_g)
    ax.set_xlabel("Video gain  =  (Video+Text)  -  (Text-only)   [pp]")
    ax.set_title("How much each model actually USES the video\n"
                 "(gain over its own text-only score, same questions)", pad=8)
    ax.set_xlim(min(gains)-1.2, max(gains)+1.2)
    save(fig, out, "v04_video_gain")

    # ---- v05 paired slope text->video ----
    rows_s = sorted(M.items(), key=lambda x: -x[1]["v_acc"])
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for disp, m in rows_s:
        c = FAMILY_COLORS[m["family"]]
        ax.plot([0, 1], [m["t_acc"], m["v_acc"]], "-o", color=c, lw=1.6,
                markersize=6, markeredgecolor="black", markeredgewidth=0.6)
        ax.text(1.02, m["v_acc"], f" {disp} ({m['gain']:+.1f})", va="center",
                fontsize=8, color=c)
    ax.axhline(BASE, color="#555555", ls="--", lw=1.0)
    ax.text(0.5, BASE+0.2, f"Random {BASE:.1f}%", ha="center", fontsize=8, color="#555555")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Text-only", "Video+Text"])
    ax.set_xlim(-0.15, 1.7)
    ax.set_ylabel("Score (%)")
    ax.set_title("Text-only -> Video+Text on the same questions\n"
                 "(slope = video contribution)", pad=8)
    save(fig, out, "v05_paired_slope")

    # ---- v07 scatter text vs video ----
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    lo, hi = 10, 26
    ax.plot([lo, hi], [lo, hi], ls=":", color="#888888", lw=1.2, label="No video gain (y=x)")
    ax.axhline(BASE, color="#bbbbbb", ls="--", lw=0.8)
    ax.axvline(BASE, color="#bbbbbb", ls="--", lw=0.8)
    for disp, m in M.items():
        c = FAMILY_COLORS[m["family"]]
        ax.scatter(m["t_acc"], m["v_acc"], s=70, color=c, edgecolor="black",
                   linewidth=0.7, zorder=3)
        ax.annotate(disp, (m["t_acc"], m["v_acc"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=7.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Text-only score (%)"); ax.set_ylabel("Video+Text score (%)")
    ax.set_title("Vision dependence: points above the diagonal use the video", pad=8)
    fams = sorted({m["family"] for m in M.values()})
    handles = [mpl.lines.Line2D([0],[0],marker="o",color="w",markerfacecolor=FAMILY_COLORS[f],
               markeredgecolor="black",markersize=8,label=f) for f in fams]
    handles.append(mpl.lines.Line2D([0],[0],ls=":",color="#888888",label="y = x (no gain)"))
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    save(fig, out, "v07_scatter_text_vs_video")

    # ---- v02 per-capability video heatmap ----
    caps = sorted({c for m in M.values() for c in m["cap_v"] if c != "unknown"})
    cap_n = {c: max(m["cap_n"].get(c, 0) for m in M.values()) for c in caps}
    rows_c = sorted(M.items(), key=lambda x: -x[1]["v_acc"])
    names_c = [r[0] for r in rows_c]
    Mat = np.array([[m["cap_v"].get(c, np.nan) for c in caps] for _, m in rows_c])
    vmax = max(np.nanmax(Mat), BASE+1); vmin = min(np.nanmin(Mat), BASE-1)
    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=BASE, vmax=vmax)
    fig, ax = plt.subplots(figsize=(1.3*len(caps)+3, 0.42*len(names_c)+2.2))
    im = ax.imshow(Mat, cmap=sns.diverging_palette(220,20,as_cmap=True), norm=norm, aspect="auto")
    for i in range(len(names_c)):
        for j in range(len(caps)):
            v = Mat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                    color="black" if abs(v-BASE) < 5 else "white")
    ax.set_xticks(range(len(caps))); ax.set_xticklabels([f"{c}\n(n={cap_n[c]})" for c in caps])
    ax.set_yticks(range(len(names_c))); ax.set_yticklabels(names_c)
    ax.set_title("Video+Text accuracy by capability", pad=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("Accuracy (%)")
    ax.grid(False)
    save(fig, out, "v02_per_capability_video")

    # ---- v06 video gain by capability heatmap ----
    Gmat = np.array([[ (m["cap_v"].get(c, np.nan) - m["cap_t"].get(c, np.nan))
                       for c in caps] for _, m in rows_c])
    g = np.nanmax(np.abs(Gmat)); g = max(g, 2)
    norm = mpl.colors.TwoSlopeNorm(vmin=-g, vcenter=0, vmax=g)
    fig, ax = plt.subplots(figsize=(1.3*len(caps)+3, 0.42*len(names_c)+2.2))
    im = ax.imshow(Gmat, cmap=sns.diverging_palette(20,150,as_cmap=True), norm=norm, aspect="auto")
    for i in range(len(names_c)):
        for j in range(len(caps)):
            v = Gmat[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=9,
                    color="black" if abs(v) < g*0.55 else "white")
    ax.set_xticks(range(len(caps))); ax.set_xticklabels([f"{c}\n(n={cap_n[c]})" for c in caps])
    ax.set_yticks(range(len(names_c))); ax.set_yticklabels(names_c)
    ax.set_title("Video gain by capability  (Video+Text  -  Text-only, pp)", pad=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label(r"$\Delta$ accuracy (pp)")
    ax.grid(False)
    save(fig, out, "v06_video_gain_by_capability")

    # ---- v03 scaling video+text ----
    fams = defaultdict(list)
    for disp, m in M.items():
        fams[m["family"]].append((m["params_B"], m["v_acc"], disp))
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for fam, pts in fams.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=FAMILY_COLORS[fam],
                lw=1.6, markersize=8, markeredgecolor="black", markeredgewidth=0.8, label=fam)
        for p in pts:
            ax.annotate(p[2].replace(fam+"-",""), (p[0],p[1]), xytext=(4,4),
                        textcoords="offset points", fontsize=8, color=FAMILY_COLORS[fam])
    ax.axhline(BASE, color="#555555", ls="--", lw=1.0, label=f"Random ({BASE:.1f}%)")
    ax.set_xscale("log"); ax.set_xticks([2,3,4,7,8,12,14])
    ax.set_xticklabels(["2B","3B","4B","7B","8B","12B","14B"])
    ax.get_xaxis().set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel("Model size (params, B) - log scale"); ax.set_ylabel("Video+Text score (%)")
    ax.set_title("Scaling of video+text performance by family", pad=8)
    ax.legend(loc="upper left", ncol=2)
    save(fig, out, "v03_scaling_video")

    # CSV
    df = pd.DataFrame([{
        "model": k, "family": m["family"], "params_B": m["params_B"], "n": m["n"],
        "video_text_acc": m["v_acc"], "text_only_acc": m["t_acc"], "video_gain_pp": m["gain"],
    } for k, m in sorted(M.items(), key=lambda x:-x[1]["v_acc"])])
    df.to_csv(out / "video_vs_text_summary.csv", index=False)
    print(f"\nAll plots + CSV written to: {out}/")


if __name__ == "__main__":
    main()
