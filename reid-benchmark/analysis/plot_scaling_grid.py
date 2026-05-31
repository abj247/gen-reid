#!/usr/bin/env python3
"""
2-row x 5-column per-capability scaling grid.
  Row 1: TEXT-ONLY  per-capability scaling (5 capabilities)
  Row 2: VIDEO+TEXT per-capability scaling (5 capabilities)
Each cell: accuracy vs log(params), one line per model family, baseline line.
Shared wide y-axis (0-40%) so the two regimes are visually comparable.

Per-model accuracies are computed on the SAME matched question set (questions
present in both that model's text-only and video predictions), so the two rows
are apples-to-apples.

Usage:
    python plot_scaling_grid_2row.py --output_dir plots_video_vs_text
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.linewidth": 0.9, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})

FAMILY_COLORS = {
    "Qwen2.5-VL": "#0072B2", "Qwen3-VL": "#009E73", "InternVL3": "#D55E00",
    "Ovis2.5": "#CC79A7", "Gemma3": "#E69F00", "Video-LLaVA": "#56B4E9",
    # Held-out LVU families:
    "VideoChat-Flash": "#F0E442", "LongVU": "#882255", "MA-LMM": "#117733",
}
MODELS = {
    "ovis2.5-2b": ("Ovis2.5-2B", 2.0, "Ovis2.5"),
    "internvl3-2b": ("InternVL3-2B", 2.0, "InternVL3"),
    "qwen3-vl-real-2b": ("Qwen3-VL-2B", 2.0, "Qwen3-VL"),
    "qwen2.5-vl-3b": ("Qwen2.5-VL-3B", 3.0, "Qwen2.5-VL"),
    "qwen3-vl-real-4b": ("Qwen3-VL-4B", 4.0, "Qwen3-VL"),
    "gemma3-4b": ("Gemma3-4B", 4.0, "Gemma3"),
    "qwen2.5-vl-7b": ("Qwen2.5-VL-7B", 7.0, "Qwen2.5-VL"),
    "qwen3-vl-real-8b": ("Qwen3-VL-8B", 8.0, "Qwen3-VL"),
    "ovis2.5-9b": ("Ovis2.5-9B", 9.0, "Ovis2.5"),
    "internvl3-8b": ("InternVL3-8B", 8.0, "InternVL3"),
    "video-llava": ("Video-LLaVA-7B", 7.0, "Video-LLaVA"),
    "gemma3-12b": ("Gemma3-12B", 12.0, "Gemma3"),
    "internvl3-14b": ("InternVL3-14B", 14.0, "InternVL3"),
    # Held-out LVU models (added after the original committee was set):
    "videochat-flash-2b": ("VideoChat-Flash-2B", 2.0, "VideoChat-Flash"),
    "videochat-flash-7b": ("VideoChat-Flash-7B", 7.0, "VideoChat-Flash"),
    "longvu-qwen2-7b":    ("LongVU-Qwen2-7B", 7.0, "LongVU"),
    "ma-lmm-vicuna7b":    ("MA-LMM-Vicuna-7B", 7.0, "MA-LMM"),
}
N_OPT = 8
BASE = 100.0 / N_OPT


def load_preds(path, key_map=None):
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
    ap.add_argument("--ymax", type=float, default=40.0)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    real_to_anon = json.load(open(args.mapping))["real_to_anon"]
    bench = json.load(open(args.bench))
    cap_of = {}
    for v in bench["videos"]:
        vid = v.get("video_id")
        for q in v.get("questions", []):
            cap_of[(vid, q.get("question_id"))] = q.get("metadata", {}).get("capability", "unknown")

    caps = ["activity", "appearance", "interaction", "location", "object_state"]
    cap_n = defaultdict(int)

    # metrics[modality][family] = list of (params, {cap: acc})
    txt_pts = defaultdict(list)
    vid_pts = defaultdict(list)
    for ckpt, (disp, pB, fam) in MODELS.items():
        vp = load_preds(f"{args.video_dir}/{ckpt}/predictions.jsonl", real_to_anon)
        tp = load_preds(f"{args.text_dir}/{ckpt}/predictions.jsonl")
        common = set(vp) & set(tp) & set(cap_of)
        if not common:
            continue
        capv = defaultdict(lambda: [0, 0]); capt = defaultdict(lambda: [0, 0])
        for k in common:
            c = cap_of[k]
            capv[c][1] += 1; capt[c][1] += 1
            if vp[k]: capv[c][0] += 1
            if tp[k]: capt[c][0] += 1
        for c in caps:
            cap_n[c] = max(cap_n[c], capt[c][1])
        txt_pts[fam].append((pB, disp, {c: 100.0*capt[c][0]/capt[c][1] if capt[c][1] else np.nan for c in caps}))
        vid_pts[fam].append((pB, disp, {c: 100.0*capv[c][0]/capv[c][1] if capv[c][1] else np.nan for c in caps}))

    fig, axes = plt.subplots(2, len(caps), figsize=(3.0*len(caps), 6.6), sharey=True)
    row_data = [("Text-only", txt_pts), ("Video+Text", vid_pts)]
    for row, (rowlabel, pts) in enumerate(row_data):
        for col, c in enumerate(caps):
            ax = axes[row, col]
            for fam, plist in pts.items():
                plist_sorted = sorted(plist)
                xs = [p[0] for p in plist_sorted]
                ys = [p[2][c] for p in plist_sorted]
                ax.plot(xs, ys, "-o", color=FAMILY_COLORS[fam], lw=1.4, markersize=5,
                        markeredgecolor="black", markeredgewidth=0.5, label=fam)
            ax.axhline(BASE, color="#555555", ls="--", lw=0.9)
            ax.set_xscale("log")
            ax.set_xticks([2, 4, 8, 14]); ax.set_xticklabels(["2B", "4B", "8B", "14B"])
            ax.get_xaxis().set_minor_locator(mpl.ticker.NullLocator())
            ax.set_ylim(0, args.ymax)
            if row == 0:
                ax.set_title(f"{c}\n(n={cap_n[c]})", fontsize=11)
            if row == 1:
                ax.set_xlabel("params")
            if col == 0:
                ax.set_ylabel(f"{rowlabel}\nscore (%)", fontsize=11)
    # one shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(mpl.lines.Line2D([0], [0], color="#555555", ls="--", label=f"Random ({BASE:.1f}%)"))
    labels.append(f"Random ({BASE:.1f}%)")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=9)
    fig.suptitle("Per-capability scaling: text-only (top) vs video+text (bottom)  "
                 f"[committee-debiased set, y=0-{int(args.ymax)}%]",
                 fontsize=13, y=1.06)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "v08_scaling_grid_text_vs_video.pdf")
    fig.savefig(out / "v08_scaling_grid_text_vs_video.png")
    plt.close(fig)
    print(f"saved v08_scaling_grid_text_vs_video (.pdf/.png) -> {out}/")


if __name__ == "__main__":
    main()
