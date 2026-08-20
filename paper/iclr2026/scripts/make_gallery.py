"""Qualitative failure gallery for the ICLR paper (fig:gallery).

Picks 2 multi_hop_tracking questions and 1 role_position_swap question whose
videos exist locally, extracts 4 uniform frames each, and shows the question,
the gold option, and what InternVL3-14B and GPT-5.4-mini predicted.

Run with /home/ab260989/.conda/envs/reid/bin/python from anywhere.
Writes paper/iclr2026/figures/fig_gallery.pdf.
"""
import json
import os
import textwrap

import decord
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/ab260989/gen-reid"
VIDEO_DIR = "/home/c3-0/datasets/moviechat1k-test"
OUT_PDF = os.path.join(ROOT, "paper/iclr2026/figures/fig_gallery.pdf")

MODELS = {
    "internvl3-14b": "InternVL3-14B",
    "gpt-5.4-mini": "GPT-5.4-mini",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "text.color": "#1a1a1a",
})


def load_predictions():
    preds = {}
    for m in MODELS:
        preds[m] = {}
        path = os.path.join(ROOT, "results_video_v2", m, "predictions.jsonl")
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                preds[m][(r["video_id"], r["question_id"])] = r
    return preds


def real_base(anon, a2r):
    real = a2r.get(anon)
    if real is None:
        return None
    return real[:-4] if real.endswith(".mp4") else real


def collect_candidates():
    bench = json.load(open(os.path.join(ROOT, "combined_all_hard_v3_retagged.json")))
    a2r = json.load(open(os.path.join(ROOT, "video_id_mapping.json")))["anon_to_real"]
    preds = load_predictions()
    cands = {"multi_hop_tracking": [], "role_position_swap": []}
    for v in bench["videos"]:
        rb = real_base(v["video_id"], a2r)
        if rb is None:
            continue
        vpath = os.path.join(VIDEO_DIR, rb + ".mp4")
        if not os.path.exists(vpath):
            continue
        for q in v["questions"]:
            cat = q["metadata"].get("reid_canonical")
            if cat not in cands:
                continue
            key = (rb, q["question_id"])
            if any(key not in preds[m] for m in MODELS):
                continue
            picks = {m: preds[m][key]["predicted"] for m in MODELS}
            both_wrong = all(picks[m] != q["correct_answer"] for m in MODELS)
            cands[cat].append({
                "video_path": vpath,
                "real_id": rb,
                "qid": q["question_id"],
                "question": q["question_text"],
                "options": q["options"],
                "gold": q["correct_answer"],
                "picks": picks,
                "both_wrong": both_wrong,
                "cat": cat,
            })
    return cands


# Curated picks for the gallery (all are both-model failures). Falls back to
# automatic selection if any of these is unavailable.
PREFERRED = [("--uyzf7X_0c", "q8"), ("-0SHIbuEO3w", "q17"), ("-4Qk4eACpXI", "q15")]


def pick_examples(cands):
    """Prefer curated failures (both models wrong) from distinct videos."""
    chosen, used_videos = [], set()
    by_key = {(c["real_id"], c["qid"]): c for lst in cands.values() for c in lst}
    if all(k in by_key for k in PREFERRED):
        for k in PREFERRED:
            chosen.append(by_key[k])
        # keep ordering: multi_hop_tracking examples first
        chosen.sort(key=lambda c: c["cat"] != "multi_hop_tracking")
        return chosen

    def take(cat, n):
        pool = [c for c in cands[cat] if c["both_wrong"]] + \
               [c for c in cands[cat] if not c["both_wrong"]]
        got = 0
        for c in pool:
            if got == n:
                break
            if c["real_id"] in used_videos:
                continue
            chosen.append(c)
            used_videos.add(c["real_id"])
            got += 1
        assert got == n, f"could not find {n} examples for {cat}"

    take("multi_hop_tracking", 2)
    take("role_position_swap", 1)
    return chosen


def extract_frames(video_path, n=4):
    """Uniform frames over the interior of the video (skips studio intro and
    outro cards that bookend the source clips)."""
    vr = decord.VideoReader(video_path)
    lo, hi = int(0.05 * (len(vr) - 1)), int(0.75 * (len(vr) - 1))
    idx = np.linspace(lo, hi, n, dtype=int)
    frames = vr.get_batch(idx).asnumpy()
    ts = [i / max(vr.get_avg_fps(), 1e-6) for i in idx]
    del vr
    return frames, ts


CAT_LABEL = {
    "multi_hop_tracking": "multi hop tracking",
    "role_position_swap": "role and position swap",
}


def main():
    cands = collect_candidates()
    examples = pick_examples(cands)
    for ex in examples:
        print(f"chosen: {ex['cat']} {ex['real_id']} {ex['qid']} gold={ex['gold']} "
              f"picks={ex['picks']} both_wrong={ex['both_wrong']}")

    n_rows, n_cols = len(examples), 4
    # height tuned so each image row matches the 16:9 frame aspect (no vertical whitespace),
    # with a compact text row under it.
    fig = plt.figure(figsize=(8.6, 5.5))
    gs = fig.add_gridspec(2 * n_rows, n_cols,
                          height_ratios=[1.0, 0.58] * n_rows,
                          hspace=0.04, wspace=0.02,
                          left=0.01, right=0.99, top=0.99, bottom=0.01)

    for r, ex in enumerate(examples):
        frames, ts = extract_frames(ex["video_path"], n_cols)
        for c in range(n_cols):
            ax = fig.add_subplot(gs[2 * r, c])
            ax.imshow(frames[c])
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.text(0.02, 0.965, f"t={ts[c]:.0f}s", transform=ax.transAxes,
                    fontsize=7, color="white", va="top",
                    bbox=dict(facecolor="black", alpha=0.55, pad=1.5,
                              edgecolor="none"))
            if c == 0:
                ax.text(0.02, 0.03, CAT_LABEL[ex["cat"]], transform=ax.transAxes,
                        fontsize=7.5, color="white", va="bottom", style="italic",
                        bbox=dict(facecolor="#8a3033", alpha=0.85, pad=2.0,
                                  edgecolor="none"))

        tax = fig.add_subplot(gs[2 * r + 1, :])
        tax.axis("off")
        qtxt = textwrap.fill("Q: " + ex["question"], width=142)
        gold_txt = textwrap.fill(
            f"Gold ({ex['gold']}): {ex['options'][ex['gold']]}", width=142)
        pick_parts = []
        for m, disp in MODELS.items():
            p = ex["picks"][m]
            mark = "correct" if p == ex["gold"] else "wrong"
            opt = ex["options"].get(p, "?")
            if len(opt) > 42:
                opt = opt[:39] + "..."
            pick_parts.append(f"{disp} picked {p} ({opt}) [{mark}]")
        picks_txt = "   ".join(pick_parts)
        tax.text(0.0, 0.92, qtxt, fontsize=8.2, va="top", color="#1a1a1a")
        n_qlines = qtxt.count("\n") + 1
        y = 0.92 - 0.30 * n_qlines
        tax.text(0.0, y, gold_txt, fontsize=8.2, va="top", color="#1f6f43")
        tax.text(0.0, y - 0.30, picks_txt, fontsize=8.0, va="top",
                 color="#8a3033")

    fig.savefig(OUT_PDF)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
