#!/usr/bin/env python
"""Build the Figure 1 teaser for the PersistQA paper.

Left panel: five uniformly sampled frames from one benchmark video with a real
question (vid_0001, first question) underneath. Right panel: frame-count sweep
comparing a full-fidelity model against a token-compression model, with the
chance level and the fidelity gap annotated.

Run with /home/ab260989/.conda/envs/reid/bin/python.
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

VIDEO = "/home/c3-0/datasets/moviechat1k-test/--hendERqm0.mp4"
BENCH = "/home/ab260989/gen-reid/combined_all_hard_v3_retagged.json"
VIDEO_ID = "vid_0002"
QID = "q5"
# fixed timestamps that tell the re-identification story of the same dog across appearance changes:
# red hat (referent introduced) -> bucket hat -> brown cap -> ghost costume -> straw hat + pumpkins (queried)
FRAME_TS = [0.28, 0.47, 0.53, 0.59, 0.78]
FRAME_LABELS = ["referent introduced\n(red hat)", "", "", "", "queried moment\n(pumpkins)"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "figures", "fig1_teaser.pdf")

# Palette (validated defaults)
BLUE = "#2a78d6"      # full-fidelity series
RED = "#e34948"       # token-compression series
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
})


def load_frames(path):
    from decord import VideoReader
    vr = VideoReader(path)
    idx = [int(t * len(vr)) for t in FRAME_TS]
    return [vr[i].asnumpy() for i in idx], idx, len(vr)


def load_question():
    with open(BENCH) as f:
        data = json.load(f)
    video = next(v for v in data["videos"] if v["video_id"] == VIDEO_ID)
    return next(q for q in video["questions"] if q["question_id"] == QID)


def abbreviate(text, maxlen=34):
    return text if len(text) <= maxlen else text[: maxlen - 1].rstrip() + "…"


def main():
    frames, idx, total = load_frames(VIDEO)
    q = load_question()

    fig = plt.figure(figsize=(13, 3.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.05, 1.0], wspace=0.10,
                           left=0.005, right=0.985, top=0.96, bottom=0.13)

    # ---------------- left panel: frame strip + question ----------------
    gsl = gridspec.GridSpecFromSubplotSpec(
        2, 5, subplot_spec=gs[0], height_ratios=[2.5, 1.15],
        hspace=0.06, wspace=0.03)
    for k, fr in enumerate(frames):
        ax = fig.add_subplot(gsl[0, k])
        ax.imshow(fr)
        ax.set_xticks([])
        ax.set_yticks([])
        hl = FRAME_LABELS[k] != ""
        for s in ax.spines.values():
            s.set_edgecolor(RED if hl else BASE)
            s.set_linewidth(1.6 if hl else 0.6)
        ax.set_title(f"t = {idx[k] / total:0.0%}", fontsize=9.5,
                     color=MUTED, pad=2)
        if hl:
            ax.set_xlabel(FRAME_LABELS[k], fontsize=9.5, color=RED,
                          labelpad=2, linespacing=1.05)

    axq = fig.add_subplot(gsl[1, :])
    axq.axis("off")
    question = q["question_text"]
    correct = q["correct_answer"]
    opts = q["options"]
    shown = [correct, "A", "B"]
    shown = list(dict.fromkeys(shown))[:3]
    parts = []
    for letter in shown:
        text = abbreviate(opts[letter])
        mark = " ✓" if letter == correct else ""
        parts.append(f"({letter}) {text}{mark}")
    import textwrap
    qwrapped = "\n".join(textwrap.wrap("Q: " + question, width=118))
    axq.text(0.0, 0.98, qwrapped, fontsize=10.5, color=INK,
             ha="left", va="top", linespacing=1.35,
             transform=axq.transAxes)
    axq.text(0.0, 0.22, "    ".join(parts) + "    … 8 options total",
             fontsize=10, color=INK2, ha="left", va="top",
             transform=axq.transAxes)

    # ---------------- right panel: fidelity-vs-compression sweep --------
    axr = fig.add_subplot(gs[1])
    x = [8, 16, 32]
    ff = [23.4, 25.3, 26.6]           # InternVL3-14B, full fidelity
    tc = [18.1, 18.8, 19.2]           # Video-XL-Pro at 32/64/128 frames

    axr.axhline(12.5, color=MUTED, lw=1.0, ls=":", zorder=1)
    axr.text(7.3, 12.9, "chance (12.5)", fontsize=7.8, color=MUTED,
             va="bottom", ha="left")

    axr.plot(x, ff, color=BLUE, lw=2.0, marker="o", ms=5, zorder=4,
             label="InternVL3-14B (full fidelity)")
    axr.plot(x, tc, color=RED, lw=2.0, ls="--", marker="s", ms=4.5,
             zorder=3, label="Video-XL-Pro (token compression,\n32/64/128 frames)")

    # fidelity gap annotation at the largest budget
    axr.fill_between([28.5, 32], tc[-1], ff[-1], color=BLUE, alpha=0.12,
                     zorder=2, lw=0)
    axr.annotate("", xy=(30.2, ff[-1]), xytext=(30.2, tc[-1]),
                 arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.0))
    axr.text(29.6, (ff[-1] + tc[-1]) / 2, "fidelity gap",
             fontsize=10, color=INK2, ha="right", va="center")

    axr.set_xscale("log", base=2)
    axr.set_xticks(x)
    axr.set_xticklabels([str(v) for v in x])
    axr.minorticks_off()
    axr.set_xlim(7, 40)
    axr.set_ylim(10, 29)
    axr.set_xlabel("frames given to the model", fontsize=10.5)
    axr.set_ylabel("accuracy (%)", fontsize=10.5)
    axr.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    axr.set_axisbelow(True)
    for side in ("top", "right"):
        axr.spines[side].set_visible(False)
    axr.tick_params(labelsize=8)
    axr.legend(loc="upper left", fontsize=9.5, frameon=False,
               handlelength=2.2, borderaxespad=0.2)

    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {os.path.abspath(OUT)}")
    print(f"question used: {question}")
    print(f"options shown: {shown} (correct {correct})")


if __name__ == "__main__":
    main()
