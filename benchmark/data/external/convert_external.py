#!/usr/bin/env python
"""Convert MLVU and LongVideoBench into the benchmark schema every runner in this repo expects.

Emits `{videos: [{video_id, video_path, questions: [...]}]}` so `eval_bimba_video_text.py` and
`eval_timeviper_video_text.py` can run these benchmarks with no changes to their model, decode,
prompt or parsing paths -- only the five guards listed in the plan.

Traps this file exists to avoid (each was verified against the real data, not assumed)
--------------------------------------------------------------------------------------
* MLVU's `answer` is the option TEXT, not a letter. It must be resolved through
  `candidates.index(answer)`, and a miss must RAISE -- defaulting to A would produce a plausible
  accuracy number that is silently wrong.
* MLVU videos live in nine subdirectories and 363 basenames COLLIDE across them. The subdirectory
  is the JSON filename stem (`1_plotQA.json` -> `1_plotQA/`); a flat basename index picks the
  wrong video for those 363.
* MLVU's `8_sub_scene` and `9_summary` are generation tasks with no `candidates`. Skipping them is
  what yields the standard 2,174-question M-avg dev set.
* LongVideoBench is NOT uniformly 4-option: 983 questions have 5 candidates, 353 have 4, 1 has 3.
  Chance is ~21%, not 25%. Never hardcode the option count.
* The vendored `external/BIMBA/.../DATAS/eval/MLVU/formatted_dataset.json` DISAGREES with the
  cluster's native JSON on 11 items (at least `ego_73.mp4` has an entirely different candidate
  set). The native cluster JSONs are the source of truth here.

Usage
-----
  python analysis3/bench/convert_external.py --bench mlvu
  python analysis3/bench/convert_external.py --bench lvb --long_only
  python analysis3/bench/convert_external.py --bench lvb --long_only --stratified_subset 800
"""
import argparse
import collections
import json
import os
import zlib
from pathlib import Path

LETTERS = "ABCDEFGH"

MLVU_ROOT = "/home/c3-0/datasets/MLVU"
LVB_ROOT = "/home/c3-0/datasets/LongVideoBench"
OUT_DIR = "/home/ab260989/gen-reid/analysis3/bench"

# MLVU's generation tasks carry no `candidates`; only these seven are MCQ.
MLVU_MCQ_FILES = ["1_plotQA", "2_needle", "3_ego", "4_count",
                  "5_order", "6_anomaly_reco", "7_topic_reasoning"]


def _options(candidates):
    """{A: text, ...} for however many candidates this question has (3, 4 or 5 in practice)."""
    if not candidates:
        raise ValueError("question has no candidates")
    if len(candidates) > len(LETTERS):
        raise ValueError(f"{len(candidates)} candidates exceeds the {len(LETTERS)} letters available")
    return {LETTERS[i]: str(c) for i, c in enumerate(candidates)}


def _strip_letter_prefix(c):
    """BIMBA's vendored files prefix candidates with 'A. '; the native cluster files do not.

    Applied defensively so the same converter works if someone points it at a prefixed file.
    """
    s = str(c)
    if len(s) > 3 and s[0] in LETTERS and s[1:3] in (". ", ") "):
        return s[3:]
    return s


def convert_mlvu(long_only=False):
    """2,174 MCQ across 7 task types. `answer` is option text; subdir is the JSON stem."""
    by_video = collections.OrderedDict()
    n_q = 0
    for stem in MLVU_MCQ_FILES:
        path = f"{MLVU_ROOT}/json/{stem}.json"
        items = json.load(open(path))
        for i, it in enumerate(items):
            cands = it.get("candidates")
            if not cands:
                continue                                  # generation task rows, if any leak in
            cands = [_strip_letter_prefix(c) for c in cands]
            answer = _strip_letter_prefix(it["answer"])
            if answer not in cands:
                raise ValueError(
                    f"{stem}[{i}]: answer text {answer!r} is not among the candidates "
                    f"{cands!r}; refusing to guess a letter")
            # 5 of the 2,174 MLVU items list the correct text TWICE (e.g. 1_plotQA[335],
            # candidates ['Six','Three','Three','Eight'], answer 'Three'), so a model choosing the
            # duplicate letter is scored wrong despite giving identical text. The standard MLVU
            # eval resolves this with candidates.index(), i.e. first occurrence -- matched here so
            # the numbers stay comparable to published protocol, but flagged so the 5 can be
            # dropped in a sensitivity check rather than silently absorbed.
            ambiguous = cands.count(answer) > 1

            # The subdirectory IS the json stem. 363 basenames collide across subdirs, so a
            # basename-only id would silently point at the wrong video for those.
            vid = f"{stem}/{Path(it['video']).stem}"
            vpath = f"{MLVU_ROOT}/video/{stem}/{it['video']}"
            dur = float(it.get("duration") or 0.0)
            if long_only and dur and dur < 180.0:
                continue

            by_video.setdefault(vid, {"video_id": vid, "video_path": vpath, "questions": []})
            by_video[vid]["questions"].append({
                "question_id": f"{stem}_{i:04d}",
                "question_text": it["question"],
                "options": _options(cands),
                "correct_answer": LETTERS[cands.index(answer)],
                "metadata": {"capability": it.get("question_type", stem),
                             "duration": dur, "source": "mlvu", "task_file": stem,
                             "ambiguous_gold": ambiguous},
            })
            n_q += 1
    return list(by_video.values()), n_q


def convert_lvb(long_only=False, split="val"):
    """1,337 val questions. `correct_choice` is a 0-based int; option count varies 3/4/5."""
    path = f"{LVB_ROOT}/lvb_{split}.json"
    items = json.load(open(path))
    by_video = collections.OrderedDict()
    n_q = 0
    for it in items:
        if "correct_choice" not in it:
            raise ValueError(f"{path} has no ground truth (this is the test split); use val")
        cands = [_strip_letter_prefix(c) for c in it["candidates"]]
        ci = int(it["correct_choice"])
        if not 0 <= ci < len(cands):
            raise ValueError(f"{it['id']}: correct_choice {ci} out of range for "
                             f"{len(cands)} candidates")
        dg = int(it.get("duration_group") or 0)
        if long_only and dg not in (600, 3600):
            continue                                      # grounding a 15 s clip is meaningless

        vid = it["video_id"]
        vpath = f"{LVB_ROOT}/videos/{it['video_path']}"
        by_video.setdefault(vid, {"video_id": vid, "video_path": vpath, "questions": []})
        by_video[vid]["questions"].append({
            "question_id": str(it["id"]),
            "question_text": it["question"],
            "options": _options(cands),
            "correct_answer": LETTERS[ci],
            "metadata": {"capability": it.get("question_category", ""),
                         "duration": float(it.get("duration") or 0.0),
                         "duration_group": dg,
                         "level": it.get("level", ""),
                         "topic_category": it.get("topic_category", ""),
                         "source": "lvb"},
        })
        n_q += 1
    return list(by_video.values()), n_q


def stratified_subset(videos, n_target, seed=42):
    """Seeded subset stratified by capability x duration tercile, returned as a key list.

    Mirrors `analysis3/mem/mem_subset_core.json`: a flat list of "<video_id>|<question_id>" keys
    that a runner consumes via --subset_file. Deterministic without importing numpy: the ordering
    inside each stratum is a crc32 of the key, so the same subset comes back on any machine.
    """
    keys = [(f"{v['video_id']}|{q['question_id']}",
             q["metadata"].get("capability", ""),
             float(q["metadata"].get("duration") or 0.0))
            for v in videos for q in v["questions"]]
    if n_target >= len(keys):
        return [k for k, _, _ in keys]

    durs = sorted(d for _, _, d in keys)
    t1, t2 = durs[len(durs) // 3], durs[2 * len(durs) // 3]
    strata = collections.defaultdict(list)
    for k, cap, d in keys:
        strata[(cap, 0 if d <= t1 else 1 if d <= t2 else 2)].append(k)
    for v in strata.values():
        v.sort(key=lambda k: zlib.crc32(f"{seed}|{k}".encode()))

    out, names = [], sorted(strata)
    i = 0
    while len(out) < n_target and any(strata[s] for s in names):
        s = names[i % len(names)]
        if strata[s]:
            out.append(strata[s].pop())
        i += 1
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["mlvu", "lvb"])
    ap.add_argument("--long_only", action="store_true",
                    help="MLVU: drop <180 s. LVB: keep duration_group in {600, 3600}.")
    ap.add_argument("--stratified_subset", type=int, default=0,
                    help="also emit a seeded key-list subset of this size")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.bench == "mlvu":
        videos, n_q = convert_mlvu(args.long_only)
    else:
        videos, n_q = convert_lvb(args.long_only)

    # Path resolution is verified here, not at run time: a missing video 12 hours into a job is
    # far more expensive than a failed conversion.
    missing = [v["video_id"] for v in videos if not os.path.isfile(v["video_path"])]
    if missing:
        raise SystemExit(f"{len(missing)} videos do not exist on disk, e.g. {missing[:5]}")

    opt_counts = collections.Counter(len(q["options"]) for v in videos for q in v["questions"])
    caps = collections.Counter(q["metadata"].get("capability", "")
                               for v in videos for q in v["questions"])
    chance = 100.0 * sum(n / k for k, n in opt_counts.items()) / max(n_q, 1)

    tag = f"{args.bench}{'_long' if args.long_only else ''}"
    out = args.out or f"{OUT_DIR}/{tag}_mcq.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"benchmark_name": tag, "total_questions": n_q,
                   "num_options": dict(sorted(opt_counts.items())),
                   "random_baseline_pct": round(chance, 2),
                   "source": {"mlvu": MLVU_ROOT, "lvb": LVB_ROOT}[args.bench],
                   "videos": videos}, f)

    print(f"[{tag}] {len(videos)} videos, {n_q} questions, 0 missing videos -> {out}")
    print(f"  options per question: {dict(sorted(opt_counts.items()))}  "
          f"-> random baseline {chance:.2f}%")
    print(f"  capabilities: {dict(caps.most_common())}")
    n_amb = sum(1 for v in videos for q in v["questions"]
                if q["metadata"].get("ambiguous_gold"))
    if n_amb:
        print(f"  NOTE {n_amb} questions have the gold text duplicated across two options; "
              f"resolved to the first (standard MLVU protocol) and flagged ambiguous_gold=true")

    if args.stratified_subset:
        keys = stratified_subset(videos, args.stratified_subset)
        sub = f"{OUT_DIR}/subsets/{tag}_sub{args.stratified_subset}.json"
        Path(sub).parent.mkdir(parents=True, exist_ok=True)
        json.dump(keys, open(sub, "w"))
        sc = collections.Counter(
            q["metadata"].get("capability", "")
            for v in videos for q in v["questions"]
            if f"{v['video_id']}|{q['question_id']}" in set(keys))
        print(f"  subset: {len(keys)} keys -> {sub}")
        print(f"  subset capabilities: {dict(sc.most_common())}")


if __name__ == "__main__":
    main()
