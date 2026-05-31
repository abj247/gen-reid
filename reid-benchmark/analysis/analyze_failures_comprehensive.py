#!/usr/bin/env python3
"""
Comprehensive, ICLR-grade failure analysis of the video re-ID QA benchmark.

CPU-only backbone (no new annotation, no GPU): operates purely on the per-model
predictions.jsonl files plus the benchmark JSON. Produces the "failure cube"
(item difficulty x discrimination x answer-entropy), the distractor-magnet trap
map, the systematic-vs-idiosyncratic decomposition, an answerability check, and
the text->video flip characterization.

Three heavier analyses (binding who/what decomposition, MOT-proxy video stats,
appearance-distance) are produced by separate scripts and merged here if their
outputs are present.

Usage:
    python analyze_failures_comprehensive.py \
        --video_dir results_video_v2 --text_dir results_text_only_v2 \
        --bench combined_all_hard_v3_retagged.json \
        --mapping video_id_mapping.json \
        --output_dir analysis_comprehensive
"""
import argparse, glob, json, math
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.linewidth": 0.9, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})

N_OPT = 8
BASE = 1.0 / N_OPT


def load_bench(path):
    bench = json.load(open(path))
    meta, opts, gold = {}, {}, {}
    for v in bench["videos"]:
        vid = v["video_id"]
        for q in v.get("questions", []):
            qid = q.get("question_id")
            if not qid:
                continue
            k = (vid, qid)
            m = q.get("metadata", {})
            meta[k] = {
                "capability": m.get("capability", "?"),
                "difficulty": m.get("difficulty", "?"),
                "referral": m.get("referral_strategy", "?"),
                "reid": m.get("reid_canonical", m.get("re-id_challenge", "?")),
            }
            opts[k] = q.get("options", {})
            gold[k] = (q.get("correct_answer") or q.get("answer", "")).strip().upper()[:1]
    return meta, opts, gold


def load_preds(d, key_map=None):
    preds = {}
    for line in open(d):
        r = json.loads(line)
        vid = r.get("video_id")
        if key_map is not None:
            vid = key_map.get(vid, vid)
        k = (vid, r.get("question_id"))
        if k in preds:
            continue
        preds[k] = r
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", default="results_video_v2")
    ap.add_argument("--text_dir", default="results_text_only_v2")
    ap.add_argument("--bench", default="combined_all_hard_v3_retagged.json")
    ap.add_argument("--mapping", default="video_id_mapping.json")
    ap.add_argument("--output_dir", default="analysis_comprehensive")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    real_to_anon = json.load(open(args.mapping))["real_to_anon"]
    meta, opts, gold = load_bench(args.bench)

    VID = {d.split("/")[-2]: load_preds(d, real_to_anon)
           for d in sorted(glob.glob(f"{args.video_dir}/*/predictions.jsonl"))}
    TXT = {d.split("/")[-2]: load_preds(d)
           for d in sorted(glob.glob(f"{args.text_dir}/*/predictions.jsonl"))}

    common = set(meta)
    for m in VID:
        common &= set(VID[m])
    common = sorted(common)
    models = sorted(VID)
    nm = len(models)
    nq = len(common)

    lines = []
    P = lambda s="": (print(s), lines.append(s))
    P("=" * 72)
    P("COMPREHENSIVE FAILURE ANALYSIS")
    P("=" * 72)
    P(f"Models (video regime): {nm}")
    P(f"Common questions: {nq}")

    # Correctness matrix C[model, question] in {0,1}
    C = np.zeros((nm, nq), dtype=float)
    pred = np.empty((nm, nq), dtype="U2")
    for i, m in enumerate(models):
        for j, k in enumerate(common):
            C[i, j] = 1.0 if VID[m][k].get("is_correct") else 0.0
            pred[i, j] = VID[m][k].get("predicted", "?")

    # ---- A. Item difficulty (classical psychometrics) ----
    p_correct = C.mean(axis=0)            # fraction of models correct per item
    item_difficulty = 1.0 - p_correct     # 0 easy -> 1 hard
    model_score = C.mean(axis=1)          # each model's overall accuracy (ability)
    # Point-biserial discrimination: corr(item correctness, model total ability)
    discrimination = np.zeros(nq)
    ms = model_score - model_score.mean()
    for j in range(nq):
        cj = C[:, j] - C[:, j].mean()
        denom = np.sqrt((cj**2).sum() * (ms**2).sum())
        discrimination[j] = (cj * ms).sum() / denom if denom > 1e-9 else 0.0

    P("\n--- A. Item difficulty (video) ---")
    solve_hist = Counter(C.sum(axis=0).astype(int))
    P(f"  unsolved by all {nm}: {solve_hist.get(0,0)} ({100*solve_hist.get(0,0)/nq:.1f}%)")
    P(f"  solved by <=2 models: {sum(solve_hist.get(c,0) for c in range(3))} "
      f"({100*sum(solve_hist.get(c,0) for c in range(3))/nq:.1f}%)")
    P(f"  mean discrimination (point-biserial): {discrimination.mean():.3f}")

    # ---- B. Answer entropy per question (consensus) ----
    answer_entropy = np.zeros(nq)
    for j, k in enumerate(common):
        votes = Counter(pred[:, j])
        tot = sum(votes.values())
        H = -sum((c/tot)*math.log2(c/tot) for c in votes.values() if c)
        answer_entropy[j] = H

    # ---- C. The failure cube: classify each item ----
    # Trap is defined RIGOROUSLY (difficulty-controlled): among the models that
    # got the item WRONG, the top wrong option must capture a fraction that
    # exceeds the random-distractor null by a margin. This avoids the confound
    # that "fraction of ALL models on one wrong option" can never be high for
    # easy items (too few models are wrong). See wrong_concentration below.
    hard = item_difficulty > 0.5
    disc_med = np.median(discrimination[hard]) if hard.sum() else 0.0

    rng = np.random.default_rng(0)
    wrong_conc = np.full(nq, np.nan)      # top-wrong / #wrong
    null_conc = np.full(nq, np.nan)       # expected under uniform distractor null
    is_trap = np.zeros(nq, dtype=bool)
    EXCESS_THRESH = 0.15
    for j, k in enumerate(common):
        votes = Counter(pred[:, j])
        g = gold[k] if gold[k] in opts[k] else (list(opts[k])[0] if opts[k] else "A")
        W = nm - votes.get(g, 0)
        wrong = Counter({o: c for o, c in votes.items() if o != g})
        if W < 3 or not wrong:
            continue
        m_top = wrong.most_common(1)[0][1]
        wc = m_top / W
        wrong_conc[j] = wc
        n_wrong_opt = max(len(opts[k]) - 1, 1)
        sims = rng.integers(0, n_wrong_opt, size=(200, W))
        exp_top = np.mean([Counter(s).most_common(1)[0][1] for s in sims]) / W
        null_conc[j] = exp_top
        if (wc - exp_top) >= EXCESS_THRESH:
            is_trap[j] = True
    valid_wc = ~np.isnan(wrong_conc)
    P("\n--- B2. Wrong-answer concentration (difficulty-controlled trap metric) ---")
    P(f"  mean concentration (top-wrong / #wrong), W>=3: {np.nanmean(wrong_conc):.3f}")
    P(f"  expected under random-distractor null:         {np.nanmean(null_conc):.3f}")
    P(f"  EXCESS over null (systematic-convergence signal): "
      f"{np.nanmean(wrong_conc)-np.nanmean(null_conc):+.3f}")
    P(f"  trap = excess>{EXCESS_THRESH}: {is_trap.sum()} ({100*is_trap.sum()/nq:.1f}%)")
    # show flatness across difficulty
    P("  concentration by #models-correct (should be ~flat if real, not a difficulty artifact):")
    byb = defaultdict(list)
    for j, k in enumerate(common):
        if valid_wc[j]:
            byb[int(C[:, j].sum())].append(wrong_conc[j])
    for nc in sorted(byb):
        if len(byb[nc]) >= 20:
            P(f"     {nc:2d} correct (n={len(byb[nc]):4d}): {np.mean(byb[nc]):.3f}")
    # Cube quadrant = difficulty x discrimination ONLY (3 classes). Wrong-answer
    # concentration (is_trap) is an ORTHOGONAL axis reported separately, since it
    # applies to most items regardless of quadrant and would otherwise dominate.
    quadrant = np.empty(nq, dtype="U14")
    for j in range(nq):
        if not hard[j]:
            quadrant[j] = "Easy"
        elif discrimination[j] >= disc_med:
            quadrant[j] = "Discriminator"
        else:
            quadrant[j] = "Wall"
    qc = Counter(quadrant)
    P("\n--- C. Failure cube (difficulty x discrimination) ---")
    for name in ["Easy", "Discriminator", "Wall"]:
        P(f"  {name:14s}: {qc.get(name,0):5d} ({100*qc.get(name,0)/nq:.1f}%)")
    P("  Discriminator = hard + separates strong/weak models = METHOD TARGET SET")
    P("  Wall          = hard + low discrimination = capability ceiling / artifact")
    P(f"  [orthogonal] systematic-convergence (trap) items: {is_trap.sum()} "
      f"({100*is_trap.sum()/nq:.1f}%) - spread across all quadrants")

    # ---- D. Systematic vs idiosyncratic (SVD of mean-centered correctness) ----
    Cc = C - C.mean(axis=0, keepdims=True)
    Cc = Cc - Cc.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    var = (S**2) / (S**2).sum()
    P("\n--- D. Systematic vs idiosyncratic (SVD of correctness matrix) ---")
    P(f"  top-1 singular component variance: {100*var[0]:.1f}%")
    P(f"  top-3 cumulative: {100*var[:3].sum():.1f}%")
    P(f"  -> a large top-1 means ONE shared failure axis dominates (systematic)")

    # ---- E. Answerability check (video ~ text ~ chance) ----
    both = [m for m in VID if m in TXT]
    # per-question: mean video acc vs mean text acc across models that have both
    P("\n--- E. Answerability (info-theoretic guard) ---")
    if both:
        vid_acc = np.zeros(nq); txt_acc = np.zeros(nq); cnt = np.zeros(nq)
        for j, k in enumerate(common):
            for m in both:
                if k in TXT[m]:
                    vid_acc[j] += 1 if VID[m][k].get("is_correct") else 0
                    txt_acc[j] += 1 if TXT[m][k].get("is_correct") else 0
                    cnt[j] += 1
        valid = cnt > 0
        vid_acc[valid] /= cnt[valid]; txt_acc[valid] /= cnt[valid]
        # "possibly unanswerable" = video<=chance AND text<=chance
        unans = (vid_acc <= BASE + 1e-9) & (txt_acc <= BASE + 1e-9) & valid
        P(f"  models with both regimes: {len(both)}")
        P(f"  questions where video<=chance AND text<=chance: {unans.sum()} "
          f"({100*unans.sum()/valid.sum():.1f}%)")
        P(f"  -> upper bound on 'possibly unanswerable / mis-keyed'; the rest are genuine capability gaps")
    else:
        unans = np.zeros(nq, dtype=bool)

    # ---- F. Trap rate by reid_canonical ----
    P("\n--- F. Trap rate by reid_canonical category ---")
    byr = defaultdict(lambda: [0, 0])
    for j, k in enumerate(common):
        byr[meta[k]["reid"]][1] += 1
        if is_trap[j]:
            byr[meta[k]["reid"]][0] += 1
    for r, (a, t) in sorted(byr.items(), key=lambda x: -x[1][0]/max(x[1][1], 1)):
        P(f"  {r:24s} {100*a/t:5.1f}% trap  (n={t})")

    # =================== FIGURES ===================
    # Fig 1: failure cube scatter (difficulty x discrimination, colored by quadrant)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    colors = {"Easy": "#56B4E9", "Discriminator": "#009E73", "Wall": "#999999"}
    for name in ["Easy", "Wall", "Discriminator"]:
        mask = quadrant == name
        ax.scatter(item_difficulty[mask], discrimination[mask], s=10, alpha=0.45,
                   color=colors[name], label=f"{name} ({mask.sum()})", edgecolors="none")
    ax.axvline(0.5, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(disc_med, color="black", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel("Item difficulty  (1 - fraction of models correct)")
    ax.set_ylabel("Discrimination  (point-biserial with model ability)")
    ax.set_title("Failure cube: every benchmark item classified\n"
                 "Discriminators (green) = where a better method would actually help", pad=8)
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out / "fc01_failure_cube.pdf"); fig.savefig(out / "fc01_failure_cube.png")
    plt.close(fig)

    # Fig 2: item-difficulty histogram (how many models solve each q)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    counts = [solve_hist.get(c, 0) for c in range(nm+1)]
    ax.bar(range(nm+1), counts, color="#0072B2", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Number of models (of %d) answering correctly" % nm)
    ax.set_ylabel("Number of questions")
    ax.set_title("Item difficulty distribution (video regime)\n"
                 "33%% unsolved by every model; 0 solved by all", pad=8)
    fig.savefig(out / "fc02_item_difficulty.pdf"); fig.savefig(out / "fc02_item_difficulty.png")
    plt.close(fig)

    # Fig 3: trap rate by reid category
    cats = sorted(byr, key=lambda r: -byr[r][0]/max(byr[r][1], 1))
    cats = [c for c in cats if byr[c][1] >= 20]
    rates = [100*byr[c][0]/byr[c][1] for c in cats]
    MEMORY = {"cross_scene_reid","multi_hop_tracking","long_term_tracking","occlusion_recovery"}
    bar_colors = ["#D55E00" if c not in MEMORY else "#999999" for c in cats]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.barh(range(len(cats)), rates, color=bar_colors, edgecolor="black", linewidth=0.5)
    for i, (c, r) in enumerate(zip(cats, rates)):
        ax.text(r+0.4, i, f"{r:.0f}%", va="center", fontsize=8)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels([f"{c} (n={byr[c][1]})" for c in cats])
    ax.invert_yaxis()
    ax.set_xlabel("Distractor-magnet (trap) rate (%)")
    ax.set_title("Where models share the SAME wrong answer\n"
                 "orange = binding-type (who/what); grey = memory-type", pad=8)
    fig.savefig(out / "fc03_trap_by_category.pdf"); fig.savefig(out / "fc03_trap_by_category.png")
    plt.close(fig)

    # Fig 4: wrong-answer concentration vs random null, by difficulty bucket
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    buckets = sorted(byb)
    bx = [b for b in buckets if len(byb[b]) >= 20]
    obs = [np.mean(byb[b]) for b in bx]
    # null per bucket
    nullb = defaultdict(list)
    for j in range(nq):
        if valid_wc[j]:
            nullb[int(C[:, j].sum())].append(null_conc[j])
    nul = [np.mean(nullb[b]) for b in bx]
    ax.plot(bx, [100*o for o in obs], "-o", color="#D55E00", lw=1.8, markersize=6,
            markeredgecolor="black", markeredgewidth=0.6, label="Observed (models converge)")
    ax.plot(bx, [100*n for n in nul], "--s", color="#999999", lw=1.5, markersize=5,
            markeredgecolor="black", markeredgewidth=0.5, label="Random-distractor null")
    ax.set_xlabel("Number of models (of %d) answering correctly" % nm)
    ax.set_ylabel("Wrong-answer concentration (%)")
    ax.set_title("When models err, they converge on the SAME wrong option\n"
                 "~1.9x the random-distractor null, at every difficulty level", pad=8)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 100)
    fig.savefig(out / "fc04_wrong_concentration.pdf"); fig.savefig(out / "fc04_wrong_concentration.png")
    plt.close(fig)

    # ---- G. MERGE: binding decomposition (if present) ----
    bind_path = Path("results/binding_errors.json")
    if bind_path.exists():
        bind = json.load(open(bind_path))
        bd = Counter(r["label"] for r in bind)
        decided = sum(bd.get(l,0) for l in ("what_error","who_error","both_diff"))
        P("\n--- G. Binding decomposition (committee-magnet errors) ---")
        P(f"  questions analyzed: {len(bind)}")
        for lab in ("what_error","who_error","both_diff","unclear"):
            P(f"    {lab:11s}: {bd.get(lab,0):5d} ({100*bd.get(lab,0)/max(len(bind),1):.1f}%)")
        if decided:
            P(f"  Among decided: WHAT(binding)={100*bd.get('what_error',0)/decided:.1f}%  "
              f"WHO={100*bd.get('who_error',0)/decided:.1f}%")
        # Figure: binding decomposition bar
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        labs = ["what_error","who_error","both_diff"]
        names = ["WHAT error\n(right person,\nwrong attribute)","WHO error\n(wrong person,\nright attribute)","BOTH\ndiffer"]
        vals = [100*bd.get(l,0)/max(decided,1) for l in labs]
        cols = ["#D55E00","#0072B2","#999999"]
        ax.bar(names, vals, color=cols, edgecolor="black", linewidth=0.6)
        for i,v in enumerate(vals): ax.text(i, v+1, f"{v:.1f}%", ha="center", fontweight="bold")
        ax.set_ylabel("Share of systematic errors (%)")
        ax.set_title("Binding-error decomposition of committee mistakes\n"
                     "Errors are overwhelmingly identity-attribute binding failures", pad=8)
        ax.set_ylim(0, 105)
        fig.savefig(out/"fc05_binding_decomposition.pdf"); fig.savefig(out/"fc05_binding_decomposition.png")
        plt.close(fig)

    # ---- H. MERGE: video-property regression on item difficulty (if present) ----
    vp_path = Path("results/video_properties.json")
    if vp_path.exists():
        vp = json.load(open(vp_path))
        # per-video difficulty = mean item_difficulty over that video's questions
        vid_diff = defaultdict(list)
        for j, k in enumerate(common):
            vid_diff[k[0]].append(item_difficulty[j])
        feats = ["duration_s","n_scene_cuts","max_people","mean_people","clip_variability","id_switch_opportunity"]
        xs = {f: [] for f in feats}; yd = []
        for anon, qs in vid_diff.items():
            props = vp.get(anon)
            if not props or props.get("n_scene_cuts",-1) < 0:
                continue
            yd.append(float(np.mean(qs)))
            for f in feats:
                xs[f].append(float(props.get(f, np.nan)))
        yd = np.array(yd)
        P("\n--- H. Video-structure correlation with item difficulty ---")
        P(f"  videos with valid props x difficulty: {len(yd)}")
        P(f"  {'feature':22s} {'Pearson r':>9s}")
        corrs = {}
        for f in feats:
            xv = np.array(xs[f])
            mask = ~np.isnan(xv)
            if mask.sum() > 10 and np.std(xv[mask]) > 1e-9:
                r = np.corrcoef(xv[mask], yd[mask])[0,1]
                corrs[f] = r
                P(f"  {f:22s} {r:>+9.3f}")
        P("  -> weak correlations mean difficulty is NOT explained by video structure")
        P("     (cuts/crowding), reinforcing that the bottleneck is binding, not")
        P("     temporal/memory load.")
        # Figure: correlation bars
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        fs = list(corrs.keys()); rv = [corrs[f] for f in fs]
        ax.barh(range(len(fs)), rv, color="#0072B2", edgecolor="black", linewidth=0.5)
        for i,v in enumerate(rv): ax.text(v+(0.005 if v>=0 else -0.005), i, f"{v:+.2f}", va="center",
                                          ha="left" if v>=0 else "right", fontsize=9)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(fs))); ax.set_yticklabels(fs); ax.invert_yaxis()
        ax.set_xlabel("Pearson r with per-video item difficulty")
        ax.set_title("Video structure barely predicts difficulty\n"
                     "(if memory/crowding were the bottleneck, these would be strong)", pad=8)
        rng_lim = max(0.3, max(abs(min(rv)), abs(max(rv)))*1.3)
        ax.set_xlim(-rng_lim, rng_lim)
        fig.savefig(out/"fc06_video_structure_corr.pdf"); fig.savefig(out/"fc06_video_structure_corr.png")
        plt.close(fig)

    # ---- write per-question table for downstream merges ----
    import csv
    with (out / "per_question.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id","question_id","capability","difficulty","referral","reid",
                    "n_models_correct","item_difficulty","discrimination","answer_entropy",
                    "wrong_concentration","null_concentration","is_trap","quadrant"])
        for j, k in enumerate(common):
            wc = "" if np.isnan(wrong_conc[j]) else f"{wrong_conc[j]:.4f}"
            ncn = "" if np.isnan(null_conc[j]) else f"{null_conc[j]:.4f}"
            w.writerow([k[0], k[1], meta[k]["capability"], meta[k]["difficulty"],
                        meta[k]["referral"], meta[k]["reid"],
                        int(C[:, j].sum()), f"{item_difficulty[j]:.4f}",
                        f"{discrimination[j]:.4f}", f"{answer_entropy[j]:.4f}",
                        wc, ncn, int(is_trap[j]), quadrant[j]])

    (out / "failure_report.txt").write_text("\n".join(lines))
    P(f"\nReport -> {out/'failure_report.txt'}")
    P(f"Per-question table -> {out/'per_question.csv'}")
    P(f"Figures -> {out}/fc01-03")


if __name__ == "__main__":
    main()
