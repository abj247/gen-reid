#!/usr/bin/env python
"""Authoritative corrected leaderboard: exclude the undecodable videos EVERYWHERE.

The defect (verified 2026-08-03). 296 questions from 30 videos cannot be decoded. Local-model
runs scored them WRONG (n=3,595); the Gemini and Video-XL runs EXCLUDED them (n=3,299, which is
exactly 3,595 - 296). Open models were therefore penalised 1.1-2.2 points against the
proprietary ones purely by infrastructure, not capability. The bad-prediction key sets are
byte-identical across every local model, confirming one shared cause.

Protocol adopted (user decision): exclude those 30 videos from EVERY model and from the
benchmark counts. A model's own unparseable output still counts WRONG -- that is model
behaviour, not infrastructure. GPT-5.5's content-filter declines are reported under all skip
protocols because a refusal is likewise the model's own behaviour.

Emits: corrected accuracies, the frozen key manifest, and the consensus statistics
("no question solved by all", "% solved by none") recomputed over the corrected model set.

Run:  python analysis3/corrected_leaderboard.py
"""
import json
import os

import os as _os
from pathlib import Path as _Path
ROOT = _os.environ.get("PERSISTQA_ROOT") or str(_Path(__file__).resolve().parents[2])
RES = f"{ROOT}/results_video_v2"
BAD = {"ERROR", "INVALID", "None", "", "nan", "null"}

GROUPS = [
    ("Proprietary", ["gpt-5.5", "gemini-3.1-flash-lite", "gpt-5.4-mini", "gpt-5.4-nano"]),
    ("Open-source VLMs", ["videochat-flash-7b", "internvl3-14b", "videochat-flash-2b",
                          "internvl3-8b", "ovis2.5-9b", "ovis2.5-2b", "gemma3-12b",
                          "internvl3-2b", "gemma3-4b", "qwen2.5-vl-7b", "qwen3-vl-real-8b",
                          "qwen3-vl-real-4b", "qwen3-vl-real-2b", "qwen2.5-vl-3b",
                          "video-llava"]),
    ("Long-video models", ["video-xl-pro", "video-xl", "longvu-qwen2-7b", "ma-lmm-vicuna7b"]),
]
ALL = [m for _, ms in GROUPS for m in ms]


def load(m):
    p = f"{RES}/{m}/predictions.jsonl"
    if not os.path.exists(p):
        return None
    by = {}
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        by.setdefault((r["video_id"], r["question_id"]), r)   # first occurrence wins
    return by


def main():
    preds = {m: load(m) for m in ALL}
    preds = {m: v for m, v in preds.items() if v}

    # the undecodable set: keys any local model marked unparseable (identical across them)
    ref = preds["internvl3-14b"]
    undec_keys = {k for k, r in ref.items() if str(r.get("predicted")) in BAD}
    undec_vids = {v for v, _ in undec_keys}
    universe = {k for k in ref if k not in undec_keys}

    print(f"undecodable videos   : {len(undec_vids)}")
    print(f"undecodable questions: {len(undec_keys)}")
    print(f"CORRECTED benchmark  : {len(universe)} questions "
          f"over {len({v for v, _ in universe})} videos\n")

    print(f"{'model':<24}{'published':>10}{'corrected':>11}{'delta':>8}{'n':>7}{'own-bad':>9}")
    table = {}
    for g, ms in GROUPS:
        print(f"-- {g}")
        rows = []
        for m in ms:
            d = preds.get(m)
            if not d:
                continue
            pub = 100.0 * sum(r["is_correct"] for r in d.values()) / len(d)
            use = {k: r for k, r in d.items() if k in universe}
            own_bad = sum(1 for r in use.values() if str(r.get("predicted")) in BAD)
            # own unparseable counts WRONG (model behaviour), so keep it in the denominator
            cor = 100.0 * sum(r["is_correct"] for r in use.values()) / max(len(use), 1)
            rows.append((cor, m, pub, len(use), own_bad))
        for cor, m, pub, n, ob in sorted(rows, reverse=True):
            table[m] = (cor, n)
            print(f"{m:<24}{pub:>10.2f}{cor:>11.2f}{cor - pub:>+8.2f}{n:>7}{ob:>9}")

    # ---- consensus statistics over the corrected set ----
    full = [m for m in ALL if m in preds and len(set(preds[m]) & universe) > 0.95 * len(universe)]
    common = set(universe)
    for m in full:
        common &= set(preds[m])
    n_all = sum(1 for k in common if all(preds[m][k]["is_correct"] for m in full))
    n_none = sum(1 for k in common if not any(preds[m][k]["is_correct"] for m in full))
    print(f"\n=== CONSENSUS over {len(full)} models with near-full coverage "
          f"(common n={len(common)}) ===")
    print(f"  solved by ALL  : {n_all}  ({100.0*n_all/max(len(common),1):.2f}%)")
    print(f"  solved by NONE : {n_none}  ({100.0*n_none/max(len(common),1):.2f}%)")
    print(f"  models counted : {len(full)}")

    out = {"undecodable_videos": sorted(undec_vids),
           "undecodable_questions": len(undec_keys),
           "corrected_n_questions": len(universe),
           "corrected_n_videos": len({v for v, _ in universe}),
           "accuracies": {m: round(v[0], 2) for m, v in table.items()},
           "n_per_model": {m: v[1] for m, v in table.items()},
           "consensus": {"models": len(full), "common_n": len(common),
                         "solved_by_all": n_all, "solved_by_none": n_none,
                         "pct_none": round(100.0 * n_none / max(len(common), 1), 2)}}
    p = f"{ROOT}/analysis3/corrected_leaderboard.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n-> {p}")
    print("   (figure scripts must exclude these videos too, or table and figures disagree)")


if __name__ == "__main__":
    main()
