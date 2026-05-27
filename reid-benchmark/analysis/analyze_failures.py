#!/usr/bin/env python3
"""
Failure-mode analysis on the committee-debiased MovieChat-1k set.

Two questions:
  (A) Modality overlap - of the questions a model gets WRONG in text-only,
      how many stay wrong with video? (and the full 2x2 contingency)
  (B) Where do models fail - by capability / difficulty / referral strategy /
      and especially by Re-ID challenge type. The Re-ID challenge field is
      near-free-text, so we bucket it with a keyword taxonomy. The buckets
      that demand long-range temporal memory (cross-scene, multi-hop,
      long-term tracking, occlusion recovery) are exactly the ones a
      streaming / hierarchical-memory method would target.

Outputs (in --output_dir):
  TEXT  failure_report.txt
  PLOTS f01_modality_overlap            stacked bars per model (4 outcomes)
        f02_text_wrong_stays_wrong      bar: % of text-wrong that stay wrong w/ video
        f03_failure_by_capability       video+text error rate by capability
        f04_failure_by_difficulty       video+text error rate by difficulty
        f05_failure_by_reid_bucket      error rate by Re-ID challenge bucket  *** memory motivation
        f06_videogain_by_reid_bucket    video gain by Re-ID bucket
        f07_universally_hard            count of all-model-wrong questions by bucket x capability
  CSV   failure_summary.csv
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.linewidth": 0.9, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})

MODELS = {
    "ovis2.5-2b": "Ovis2.5-2B", "internvl3-2b": "InternVL3-2B",
    "qwen3-vl-real-2b": "Qwen3-VL-2B", "qwen2.5-vl-3b": "Qwen2.5-VL-3B",
    "qwen3-vl-real-4b": "Qwen3-VL-4B", "gemma3-4b": "Gemma3-4B",
    "qwen2.5-vl-7b": "Qwen2.5-VL-7B", "qwen3-vl-real-8b": "Qwen3-VL-8B",
    "ovis2.5-9b": "Ovis2.5-9B", "internvl3-8b": "InternVL3-8B",
    "video-llava": "Video-LLaVA-7B", "gemma3-12b": "Gemma3-12B",
    "internvl3-14b": "InternVL3-14B",
}

# Re-ID challenge taxonomy: ordered; first matching pattern wins.
# The first four are the "memory-demanding" buckets.
REID_BUCKETS = [
    ("Cross-scene / camera-cut", r"cross[\s-]?scene|camera\s?(cut|angle|change)|across camera|scene change|different scene"),
    ("Multi-hop tracking",       r"multi[\s-]?hop|intermediate event|multiple event|chain"),
    ("Long-term tracking",       r"long[\s-]?term|through multiple cuts|significant movement|over time|entire video|throughout"),
    ("Occlusion recovery",       r"occlusion|occluded|reappear|re-appear|disappear|out of frame|off[\s-]?screen"),
    ("Disambiguation (similar)", r"similar[\s-]?looking|disambiguat|look[\s-]?alike|same outfit|similar appearance|distinguish"),
    ("Role/position swap",       r"role|position swap|swap|positions? change|exchange"),
    ("Action-sequence tracking", r"action sequence|sequence of action|action[\s-]?based|after.*action|appearance[\s-]?after"),
    ("Spatial relationship",     r"spatial|location tracking|relative position|to final location|movement to"),
]


def bucket_reid(text):
    t = (text or "").lower()
    for name, pat in REID_BUCKETS:
        if re.search(pat, t):
            return name
    return "Other / single-shot"


MEMORY_BUCKETS = {"Cross-scene / camera-cut", "Multi-hop tracking",
                  "Long-term tracking", "Occlusion recovery"}

# Canonical labels produced by the LLM re-tagger (llm_retag_reid.py).
# If metadata["reid_canonical"] is present we use it instead of the keyword bucket.
CANON_ORDER = [
    "cross_scene_reid", "long_term_tracking", "multi_hop_tracking", "occlusion_recovery",
    "appearance_change", "action_sequence", "spatial_relationship",
    "disambiguation_similar", "role_position_swap", "single_shot",
]
CANON_MEMORY = {"cross_scene_reid", "long_term_tracking", "multi_hop_tracking", "occlusion_recovery"}


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


def save(fig, out, name):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png")
    plt.close(fig); print(f"  saved {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", default="results_video_v2")
    ap.add_argument("--text_dir", default="results_text_only_v2")
    ap.add_argument("--bench", default="combined_all_hard_v3.json")
    ap.add_argument("--mapping", default="video_id_mapping.json")
    ap.add_argument("--output_dir", default="analysis_failures_v3")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    real_to_anon = json.load(open(args.mapping))["real_to_anon"]

    # metadata per (anon, qid). Prefer LLM canonical tag if present.
    bench = json.load(open(args.bench))
    meta = {}
    use_canon = False
    for v in bench["videos"]:
        vid = v.get("video_id")
        for q in v.get("questions", []):
            m = q.get("metadata", {})
            canon = m.get("reid_canonical")
            if canon:
                use_canon = True
                reid = canon
            else:
                reid = bucket_reid(m.get("re-id_challenge", ""))
            meta[(vid, q.get("question_id"))] = {
                "capability": m.get("capability", "unknown"),
                "difficulty": m.get("difficulty", "Unknown"),
                "referral": m.get("referral_strategy", "unknown"),
                "reid": reid,
            }
    # Pick the right ordering + memory-set for plots based on which tags we have
    global REID_BUCKETS
    if use_canon:
        reid_order_global = CANON_ORDER
        memory_set = CANON_MEMORY
        print("Using LLM canonical Re-ID tags (metadata.reid_canonical)")
    else:
        reid_order_global = [b[0] for b in REID_BUCKETS] + ["Other / single-shot"]
        memory_set = MEMORY_BUCKETS
        print("Using keyword Re-ID buckets")

    # load all model preds (video + text), matched
    vid_preds = {}; txt_preds = {}
    for ckpt in MODELS:
        vid_preds[ckpt] = load_preds(f"{args.video_dir}/{ckpt}/predictions.jsonl", real_to_anon)
        txt_preds[ckpt] = load_preds(f"{args.text_dir}/{ckpt}/predictions.jsonl")

    # common key set across everything (so all models score the same questions)
    common = None
    for ckpt in MODELS:
        ks = set(vid_preds[ckpt]) & set(txt_preds[ckpt]) & set(meta)
        common = ks if common is None else (common & ks)
    common = sorted(common)
    print(f"Common matched questions across all models: {len(common)}")

    lines = []
    def P(s=""):
        print(s); lines.append(s)

    P("=" * 70)
    P("FAILURE-MODE ANALYSIS  (committee-debiased set)")
    P("=" * 70)
    P(f"Matched questions (all 13 models, video & text, with metadata): {len(common)}")

    # ---------- (A) Modality overlap ----------
    P("\n" + "-" * 70)
    P("(A) MODALITY OVERLAP  -  text-only vs video+text, per model")
    P("-" * 70)
    P(f"{'Model':16s} {'both_ok':>8s} {'vid_only':>8s} {'txt_only':>8s} {'both_wrong':>10s} {'TWstayW':>8s}")
    overlap_rows = []
    agg = Counter()
    for ckpt, disp in MODELS.items():
        tt = tw_vc = tc_vw = ff = 0
        for k in common:
            t = txt_preds[ckpt][k]; v = vid_preds[ckpt][k]
            if t and v: tt += 1
            elif (not t) and v: tw_vc += 1
            elif t and (not v): tc_vw += 1
            else: ff += 1
        n = len(common)
        text_wrong = tw_vc + ff
        stay_wrong = 100.0 * ff / text_wrong if text_wrong else 0.0
        P(f"{disp:16s} {tt:8d} {tw_vc:8d} {tc_vw:8d} {ff:10d} {stay_wrong:7.1f}%")
        overlap_rows.append((disp, tt, tw_vc, tc_vw, ff, stay_wrong))
        agg["tt"] += tt; agg["tw_vc"] += tw_vc; agg["tc_vw"] += tc_vw; agg["ff"] += ff
    tot = sum(agg.values())
    P(f"\nAGGREGATE over all model-questions (n={tot}):")
    P(f"  both correct          : {agg['tt']:6d} ({100*agg['tt']/tot:.1f}%)")
    P(f"  video fixed (T-wrong->V-right): {agg['tw_vc']:6d} ({100*agg['tw_vc']/tot:.1f}%)")
    P(f"  video broke (T-right->V-wrong): {agg['tc_vw']:6d} ({100*agg['tc_vw']/tot:.1f}%)")
    P(f"  both wrong            : {agg['ff']:6d} ({100*agg['ff']/tot:.1f}%)")
    tw_all = agg['tw_vc'] + agg['ff']
    P(f"\n  Of all text-only WRONG answers, {100*agg['ff']/tw_all:.1f}% remain wrong with video.")
    P(f"  => video rescues only {100*agg['tw_vc']/tw_all:.1f}% of text-only failures.")

    # ---------- (B) Failure by metadata ----------
    def err_rate_by(field, preds):
        """aggregate error rate over all models, grouped by metadata field value"""
        grp = defaultdict(lambda: [0, 0])  # value -> [errors, total]
        for ckpt in MODELS:
            for k in common:
                val = meta[k][field]
                grp[val][1] += 1
                if not preds[ckpt][k]:
                    grp[val][0] += 1
        return {v: 100.0 * a[0] / a[1] for v, a in grp.items()}, {v: a[1]//len(MODELS) for v, a in grp.items()}

    P("\n" + "-" * 70)
    P("(B) WHERE MODELS FAIL  (video+text error rate, aggregated over 13 models)")
    P("-" * 70)
    for field, label in [("capability", "CAPABILITY"), ("difficulty", "DIFFICULTY"),
                         ("referral", "REFERRAL STRATEGY"), ("reid", "RE-ID CHALLENGE BUCKET")]:
        errv, nper = err_rate_by(field, vid_preds)
        errt, _ = err_rate_by(field, txt_preds)
        P(f"\n  {label}:")
        P(f"    {'value':28s} {'n/model':>8s} {'V+T err':>8s} {'Txt err':>8s} {'video_gain':>10s}")
        for val in sorted(errv, key=lambda x: -errv[x]):
            gain = errt[val] - errv[val]  # error reduction from video = positive good
            P(f"    {val:28s} {nper[val]:8d} {errv[val]:7.1f}% {errt[val]:7.1f}% {gain:+9.1f}pp")

    # ---------- (C) Universally hard questions ----------
    P("\n" + "-" * 70)
    P("(C) UNIVERSALLY HARD QUESTIONS  (wrong in video+text by >=11 of 13 models)")
    P("-" * 70)
    nwrong = {k: sum(1 for ckpt in MODELS if not vid_preds[ckpt][k]) for k in common}
    THRESH = 11
    hard = [k for k in common if nwrong[k] >= THRESH]
    P(f"  {len(hard)} of {len(common)} questions ({100*len(hard)/len(common):.1f}%) are missed by >={THRESH}/13 models even WITH video.")
    by_cap = Counter(meta[k]["capability"] for k in hard)
    by_reid = Counter(meta[k]["reid"] for k in hard)
    by_diff = Counter(meta[k]["difficulty"] for k in hard)
    # base rates for lift
    base_cap = Counter(meta[k]["capability"] for k in common)
    base_reid = Counter(meta[k]["reid"] for k in common)
    P("\n  Hard-question concentration by Re-ID bucket (lift = share among hard / share overall):")
    for b in sorted(by_reid, key=lambda x: -by_reid[x]):
        share_hard = by_reid[b] / len(hard)
        share_all = base_reid[b] / len(common)
        lift = share_hard / share_all if share_all else 0
        mem = "  <-- memory-demanding" if b in memory_set else ""
        P(f"    {b:28s} {by_reid[b]:4d}  ({100*share_hard:4.1f}% of hard, lift {lift:.2f}x){mem}")
    mem_share_hard = sum(by_reid[b] for b in memory_set) / len(hard) if hard else 0
    mem_share_all = sum(base_reid[b] for b in memory_set) / len(common)
    P(f"\n  Memory-demanding buckets = {100*mem_share_all:.1f}% of all questions "
      f"but {100*mem_share_hard:.1f}% of universally-hard questions "
      f"(lift {mem_share_hard/mem_share_all:.2f}x).")
    P("  => A streaming / hierarchical-memory method should target these buckets.")

    (out / "failure_report.txt").write_text("\n".join(lines))
    print(f"\nReport -> {out/'failure_report.txt'}")

    # ================= PLOTS =================
    # f01 modality overlap stacked bars
    rows = sorted(overlap_rows, key=lambda r: -(r[1]+r[2]))  # by accuracy-ish
    names = [r[0] for r in rows]
    tt = np.array([r[1] for r in rows]); twvc = np.array([r[2] for r in rows])
    tcvw = np.array([r[3] for r in rows]); ff = np.array([r[4] for r in rows])
    n = len(common)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(names))
    ax.bar(x, 100*tt/n, label="Correct in both", color="#2c7fb8")
    ax.bar(x, 100*twvc/n, bottom=100*tt/n, label="Video fixed (txt-wrong-> right)", color="#7fcdbb")
    ax.bar(x, 100*tcvw/n, bottom=100*(tt+twvc)/n, label="Video broke (txt-right-> wrong)", color="#fec44f")
    ax.bar(x, 100*ff/n, bottom=100*(tt+twvc+tcvw)/n, label="Wrong in both", color="#de2d26")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("% of questions"); ax.set_ylim(0, 100)
    ax.set_title("Text-only vs Video+Text outcome decomposition", pad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.grid(False)
    save(fig, out, "f01_modality_overlap")

    # f02 text-wrong stays wrong
    rows2 = sorted(overlap_rows, key=lambda r: -r[5])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    yv = [r[5] for r in rows2]; nm = [r[0] for r in rows2]
    ax.barh(np.arange(len(nm)), yv, color="#de2d26", edgecolor="black", linewidth=0.6)
    for i, vv in enumerate(yv):
        ax.text(vv+0.4, i, f"{vv:.1f}%", va="center", fontsize=9)
    ax.set_yticks(np.arange(len(nm))); ax.set_yticklabels(nm); ax.invert_yaxis()
    ax.set_xlabel("% of text-only errors that REMAIN errors with video")
    ax.set_title("How often the video fails to rescue a text-only mistake", pad=8)
    ax.set_xlim(0, 100)
    save(fig, out, "f02_text_wrong_stays_wrong")

    # f03/f04/f05 error-rate bar charts
    def bar_errrate(field, fname, title, order=None):
        errv, nper = err_rate_by(field, vid_preds)
        errt, _ = err_rate_by(field, txt_preds)
        vals = order or sorted(errv, key=lambda x: -errv[x])
        vals = [v for v in vals if v in errv]
        fig, ax = plt.subplots(figsize=(max(7, 0.9*len(vals)+3), 4.4))
        x = np.arange(len(vals)); w = 0.38
        ax.bar(x - w/2, [errv[v] for v in vals], w, label="Video+Text error", color="#de2d26", edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, [errt[v] for v in vals], w, label="Text-only error", color="#fdae6b", edgecolor="black", linewidth=0.5)
        for xi, v in zip(x, vals):
            ax.text(xi - w/2, errv[v]+0.5, f"{errv[v]:.0f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([f"{v}\n(n={nper[v]})" for v in vals], rotation=0)
        ax.set_ylabel("Error rate (%)"); ax.set_ylim(0, 100)
        ax.axhline(87.5, color="#555", ls="--", lw=1.0, label="Random (87.5% err)")
        ax.set_title(title, pad=8); ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        save(fig, out, fname)

    bar_errrate("capability", "f03_failure_by_capability",
                "Failure rate by capability (lower = model handles it better)")
    bar_errrate("difficulty", "f04_failure_by_difficulty",
                "Failure rate by difficulty", order=["Easy", "Medium", "Hard"])
    reid_order = reid_order_global
    bar_errrate("reid", "f05_failure_by_reid_bucket",
                "Failure rate by Re-ID challenge type  (memory-demanding buckets first)",
                order=reid_order)

    # f06 video gain by reid bucket
    errv, nper = err_rate_by("reid", vid_preds)
    errt, _ = err_rate_by("reid", txt_preds)
    vals = [v for v in reid_order if v in errv]
    gains = [errt[v] - errv[v] for v in vals]  # error reduction
    colors = ["#1a9850" if v in memory_set else "#91cf60" for v in vals]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(vals))
    ax.bar(x, gains, color=colors, edgecolor="black", linewidth=0.6)
    for xi, g in zip(x, gains):
        ax.text(xi, g+0.1, f"{g:+.1f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{v}\n(n={nper[v]})" for v in vals], rotation=30, ha="right")
    ax.set_ylabel("Error reduction from video (pp)")
    ax.set_title("How much the video helps, by Re-ID challenge type\n"
                 "(dark green = memory-demanding buckets)", pad=8)
    ax.axhline(0, color="black", lw=0.8); ax.grid(axis="y", alpha=0.25)
    save(fig, out, "f06_videogain_by_reid_bucket")

    # f07 universally-hard heatmap: reid bucket x capability
    caps = sorted(set(meta[k]["capability"] for k in common))
    buckets = reid_order
    Mh = np.zeros((len(buckets), len(caps)))
    for k in hard:
        bi = buckets.index(meta[k]["reid"]); ci = caps.index(meta[k]["capability"])
        Mh[bi, ci] += 1
    fig, ax = plt.subplots(figsize=(1.1*len(caps)+3, 0.5*len(buckets)+2))
    im = ax.imshow(Mh, cmap="Reds", aspect="auto")
    for i in range(len(buckets)):
        for j in range(len(caps)):
            if Mh[i, j] > 0:
                ax.text(j, i, int(Mh[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if Mh[i, j] > Mh.max()*0.6 else "black")
    ax.set_xticks(range(len(caps))); ax.set_xticklabels(caps, rotation=30, ha="right")
    ax.set_yticks(range(len(buckets))); ax.set_yticklabels(buckets)
    ax.set_title(f"Universally-hard questions (missed by >={THRESH}/13 models WITH video)\n"
                 f"by Re-ID challenge x capability  (total {len(hard)})", pad=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("# hard questions")
    ax.grid(False)
    save(fig, out, "f07_universally_hard")

    # CSV
    pd.DataFrame([{"model": r[0], "both_correct": r[1], "video_fixed": r[2],
                   "video_broke": r[3], "both_wrong": r[4], "text_wrong_stays_wrong_pct": r[5]}
                  for r in overlap_rows]).to_csv(out / "failure_summary.csv", index=False)
    print(f"\nAll outputs in {out}/")


if __name__ == "__main__":
    main()
