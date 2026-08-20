#!/usr/bin/env python3
"""
Adversarial committee filtering for the MovieChat-1k benchmark.

Reads `raw_results.json` from a multi-model text-only evaluation, computes
per-question leakage statistics, and emits a filtered benchmark JSON that
the committee can no longer solve above chance.

Leakage signals computed per question:
  - leak_score   : # committee members that picked the correct letter (0..N)
  - mode_count   : max # members converging on any single letter
  - mode_letter  : that letter
  - entropy      : Shannon entropy of the committee's letter distribution
                   (low = consensus = textual leak; high = disagreement = visual)

Filtering rule (a question is DROPPED if any of these fire):
  - leak_score >= tau_correct
        => committee converges on the gold via broad-spectrum textual cues.
  - mode_count >= tau_mode AND mode_letter != correct
        => committee converges on a wrong attractor (distractor is too
           plausible OR gold is mis-specified).

Calibration:
  - We sweep tau_correct from 12 down to 1; for each value, we report
    (kept, mean committee accuracy on kept, per-model accuracy on kept).
  - We auto-pick the largest tau_correct for which mean acc on the kept set
    is <= --target_acc (default 13.0%).

Usage:
    python debias_via_committee.py \\
        --raw_results <results>/text_only/raw_results.json \\
        --benchmark   <work>/combined_all_text_only.json \\
        --output      <work>/persistqa_filtered.json \\
        --target_acc  13.0

    # Force a specific tau and skip the sweep:
    python debias_via_committee.py ... --tau_correct 5 --tau_mode 9

Output:
    - combined_all_hard_v3.json    : filtered benchmark, schema-identical to input
    - debias_report_v3.json        : per-question leak stats + sweep table
    - debias_report_v3.txt         : human-readable summary
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_results(raw_path: Path) -> Tuple[Dict[Tuple[str, str], Dict[str, str]],
                                          Dict[Tuple[str, str], str],
                                          List[str]]:
    """Returns ({(video_id, question_id): {model_name: predicted_letter}},
                {(vid,qid): correct_letter},
                [model_name, ...]).
    Deduplicates: if the same (model, vid, qid) appears multiple times
    (a known checkpointing-restart artifact), keeps the first prediction."""
    with raw_path.open() as f:
        data = json.load(f)
    preds_by_q: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
    correct_by_q: Dict[Tuple[str, str], str] = {}
    dup_count_per_model: Dict[str, int] = {}
    for model_name, mres in data.items():
        seen = set()
        dups = 0
        for p in mres.get("predictions", []):
            key = (p["video_id"], p["question_id"])
            if (model_name, key) in seen:
                dups += 1
                continue
            seen.add((model_name, key))
            preds_by_q[key][model_name] = p.get("predicted", "")
            correct_by_q[key] = p.get("correct", "")
        dup_count_per_model[model_name] = dups
    total_dups = sum(dup_count_per_model.values())
    if total_dups:
        print(f"[load_results] deduplicated {total_dups} duplicate predictions "
              f"across {sum(1 for v in dup_count_per_model.values() if v)} models")
    return preds_by_q, correct_by_q, list(data.keys())


def compute_leak_stats(preds_by_q, correct_by_q):
    """Returns dict[(vid,qid)] -> {leak_score, mode_count, mode_letter, entropy, n_models}."""
    stats = {}
    for key, model_preds in preds_by_q.items():
        correct = correct_by_q[key]
        letters = list(model_preds.values())
        n = len(letters)
        leak_score = sum(1 for l in letters if l == correct)
        cnt = Counter(letters)
        mode_letter, mode_count = cnt.most_common(1)[0]
        total = sum(cnt.values()) or 1
        entropy = -sum((c / total) * math.log2(c / total) for c in cnt.values() if c)
        stats[key] = {
            "leak_score": leak_score,
            "mode_count": mode_count,
            "mode_letter": mode_letter,
            "entropy": entropy,
            "n_models": n,
            "correct": correct,
        }
    return stats


def question_is_kept(s: Dict[str, Any], tau_correct: int, tau_mode: int) -> bool:
    if s["leak_score"] >= tau_correct:
        return False
    if s["mode_count"] >= tau_mode and s["mode_letter"] != s["correct"]:
        return False
    return True


def evaluate_filter(stats, tau_correct, tau_mode, preds_by_q, model_names):
    """Per-model accuracy on the kept set."""
    kept_keys = [k for k, s in stats.items() if question_is_kept(s, tau_correct, tau_mode)]
    n_kept = len(kept_keys)
    if n_kept == 0:
        return n_kept, {}, 0.0
    per_model_acc = {}
    for m in model_names:
        correct = 0
        total = 0
        for k in kept_keys:
            if m in preds_by_q[k]:
                total += 1
                if preds_by_q[k][m] == stats[k]["correct"]:
                    correct += 1
        per_model_acc[m] = (100.0 * correct / total) if total else 0.0
    mean_acc = sum(per_model_acc.values()) / max(len(per_model_acc), 1)
    return n_kept, per_model_acc, mean_acc


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_results", required=True)
    p.add_argument("--benchmark", required=True,
                   help="Source benchmark JSON to filter (e.g. combined_all_text_only.json)")
    p.add_argument("--output", required=True,
                   help="Output filtered benchmark JSON path")
    p.add_argument("--report_dir", default=None,
                   help="Directory to write debias_report.* (default: alongside --output)")
    p.add_argument("--target_acc", type=float, default=13.0,
                   help="Calibration target: pick the largest tau_correct giving mean acc <= target")
    p.add_argument("--tau_correct", type=int, default=None,
                   help="Override calibration: drop if leak_score >= this")
    p.add_argument("--tau_mode", type=int, default=None,
                   help="Override calibration: drop if mode_count >= this AND mode != correct. "
                        "Default: ceil(0.66 * n_models) at chosen tau_correct.")
    return p.parse_args()


def main():
    args = parse_args()
    raw_path = Path(args.raw_results)
    bench_path = Path(args.benchmark)
    out_path = Path(args.output)
    report_dir = Path(args.report_dir) if args.report_dir else out_path.parent
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw results: {raw_path}")
    preds_by_q, correct_by_q, model_names = load_results(raw_path)
    n_models = len(model_names)
    n_q = len(preds_by_q)
    print(f"Committee: {n_models} models, {n_q} questions")
    print(f"Models: {', '.join(model_names)}")

    stats = compute_leak_stats(preds_by_q, correct_by_q)

    # Distribution of leak_score
    leak_hist = Counter(s["leak_score"] for s in stats.values())
    mode_hist = Counter(s["mode_count"] for s in stats.values())
    print("\nleak_score histogram (# correct out of committee):")
    for k in sorted(leak_hist):
        bar = "#" * (leak_hist[k] * 60 // max(leak_hist.values()))
        print(f"  {k:2d}: {leak_hist[k]:5d}  {bar}")
    print("\nmode_count histogram (consensus on any one letter):")
    for k in sorted(mode_hist):
        bar = "#" * (mode_hist[k] * 60 // max(mode_hist.values()))
        print(f"  {k:2d}: {mode_hist[k]:5d}  {bar}")

    # Calibration sweep
    sweep_rows = []
    print(f"\nCalibration sweep (target mean committee accuracy <= {args.target_acc:.1f}%):")
    print(f"{'tau_corr':>9} {'tau_mode':>9} {'kept':>6} {'%kept':>7} {'mean_acc':>10}")
    for tau_correct in range(n_models, 0, -1):
        tau_mode = max(int(math.ceil(0.66 * n_models)), tau_correct)
        n_kept, per_model, mean_acc = evaluate_filter(
            stats, tau_correct, tau_mode, preds_by_q, model_names
        )
        sweep_rows.append({
            "tau_correct": tau_correct, "tau_mode": tau_mode,
            "kept": n_kept, "pct_kept": 100.0 * n_kept / n_q,
            "mean_acc": mean_acc, "per_model_acc": per_model,
        })
        print(f"{tau_correct:>9d} {tau_mode:>9d} {n_kept:>6d} "
              f"{100.0*n_kept/n_q:>6.1f}% {mean_acc:>9.2f}%")

    # Choose tau
    if args.tau_correct is not None:
        chosen_correct = args.tau_correct
        chosen_mode = args.tau_mode if args.tau_mode is not None \
                      else max(int(math.ceil(0.66 * n_models)), chosen_correct)
        print(f"\nUsing user-specified tau_correct={chosen_correct}, tau_mode={chosen_mode}")
    else:
        candidates = [r for r in sweep_rows if r["mean_acc"] <= args.target_acc]
        if not candidates:
            chosen = sweep_rows[-1]  # tightest filter
            print(f"\nWARN: no tau achieved mean_acc <= {args.target_acc:.1f}%. "
                  f"Using tightest (tau_correct={chosen['tau_correct']}).")
        else:
            # Largest tau (least aggressive filter) still meeting target
            chosen = max(candidates, key=lambda r: r["tau_correct"])
        chosen_correct = chosen["tau_correct"]
        chosen_mode = chosen["tau_mode"]
        print(f"\nChosen tau_correct={chosen_correct}, tau_mode={chosen_mode} "
              f"(kept {chosen['kept']}, mean_acc={chosen['mean_acc']:.2f}%)")

    # Final per-model accuracy at chosen tau
    n_kept, per_model_acc, mean_acc = evaluate_filter(
        stats, chosen_correct, chosen_mode, preds_by_q, model_names
    )
    print(f"\nPer-model accuracy on filtered set (n={n_kept}):")
    for m in sorted(per_model_acc, key=lambda x: -per_model_acc[x]):
        delta = per_model_acc[m] - 100.0 / 8
        sign = "+" if delta >= 0 else ""
        print(f"  {m:30s}  {per_model_acc[m]:6.2f}%  ({sign}{delta:.2f}pp vs 12.5)")

    # Apply filter to the source benchmark
    print(f"\nReading source benchmark: {bench_path}")
    with bench_path.open() as f:
        bench = json.load(f)

    kept_keys_set = {k for k, s in stats.items()
                     if question_is_kept(s, chosen_correct, chosen_mode)}
    new_videos = []
    kept_count = 0
    for v in bench.get("videos", []):
        vid = v.get("video_id", "")
        kept_qs = [q for q in v.get("questions", [])
                   if (vid, q.get("question_id", "")) in kept_keys_set]
        if kept_qs:
            v_copy = dict(v)
            v_copy["questions"] = kept_qs
            new_videos.append(v_copy)
            kept_count += len(kept_qs)

    new_bench = dict(bench)
    new_bench["videos"] = new_videos
    new_bench["debiasing_version"] = "v3_committee_filter"
    new_bench["debiasing_applied"] = bench.get("debiasing_applied", []) + [
        f"committee_filter(tau_correct={chosen_correct},tau_mode={chosen_mode},"
        f"committee_size={n_models})"
    ]
    new_bench["total_questions"] = kept_count
    new_bench["committee_filter_target_acc"] = args.target_acc
    new_bench["committee_filter_models"] = model_names

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(new_bench, f, indent=2)
    print(f"\nWrote filtered benchmark: {out_path}  ({kept_count} questions)")

    # Reports
    report_json = report_dir / (out_path.stem + "_debias_report.json")
    with report_json.open("w") as f:
        json.dump({
            "committee": model_names,
            "n_questions_in": n_q,
            "n_questions_out": kept_count,
            "tau_correct": chosen_correct,
            "tau_mode": chosen_mode,
            "target_acc": args.target_acc,
            "achieved_mean_acc": mean_acc,
            "per_model_acc": per_model_acc,
            "leak_score_hist": dict(leak_hist),
            "mode_count_hist": dict(mode_hist),
            "sweep": sweep_rows,
        }, f, indent=2)
    print(f"Wrote report JSON: {report_json}")

    report_txt = report_dir / (out_path.stem + "_debias_report.txt")
    with report_txt.open("w") as f:
        f.write(f"Committee filter v3 - {n_q} -> {kept_count} questions "
                f"({100*kept_count/n_q:.1f}% kept)\n")
        f.write(f"tau_correct={chosen_correct}, tau_mode={chosen_mode}\n")
        f.write(f"Mean committee acc on kept: {mean_acc:.2f}%  "
                f"(target {args.target_acc:.1f}%, baseline 12.5%)\n\n")
        f.write("Per-model accuracy on filtered set:\n")
        for m in sorted(per_model_acc, key=lambda x: -per_model_acc[x]):
            f.write(f"  {m:30s}  {per_model_acc[m]:6.2f}%\n")
    print(f"Wrote report TXT:  {report_txt}")


if __name__ == "__main__":
    main()
