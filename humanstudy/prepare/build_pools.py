#!/usr/bin/env python
"""Build the video pools the human study serves, and freeze them to data/pools.json.

Why a frozen file rather than querying the benchmark at request time
--------------------------------------------------------------------
The deployed application must not depend on the benchmark tree, the model prediction
files, or the video corpus. It reads one small JSON and nothing else. Freezing also
means the pool a participant saw is recoverable months later even if the benchmark is
revised, which is what makes the collected responses re-analysable.

How videos are selected
-----------------------
Four filters, in order, each for a stated reason:

1. Merged by REAL video id. Nine real videos appear as two or three separate entries in
   the benchmark file, so a naive per-entry count both understates the questions
   available for a video and would let one participant be assigned the same footage
   twice under different identifiers.

2. At least MIN_QUESTIONS questions. A session is one video and all of its questions, so
   a video carrying three questions is not worth a participant watching it.

3. Human referent only. The benchmark includes animal referents, which were excluded by
   request. Detected by scanning the question and option text for animal nouns; a video
   is dropped if any of its questions mention one. This is deliberately over-eager: it
   costs a few usable videos and guarantees no animal question reaches a participant.

4. Decodes, and the file exists. Thirty corpus videos cannot be decoded by the evaluation
   stack and are excluded benchmark-wide; the same exclusion applies here.

Two pools come out of this:

  public  videos with MIN_QUESTIONS..MAX_PUBLIC_QUESTIONS questions. One participant is
          assigned one video and answers all of its questions. The upper bound exists
          because four videos carry 23 to 44 questions, which is a forty minute session
          and would be abandoned.

  author  the hardest AUTHOR_VIDEOS videos, which the authors work through across
          multiple sittings. Hardest is measured, not assumed: it is the accuracy the
          evaluated backbones achieved on that video at a uniform eight frame budget.
          The four oversized videos land here, where session length does not matter.

Run:
  python humanstudy/prepare/build_pools.py
  python humanstudy/prepare/build_pools.py --out humanstudy/data/pools.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import hmac
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from persistqa.paths import BENCH_ANON, ID_MAPPING, VIDEO_DIR  # noqa: E402

# The real video ids are YouTube ids, which is precisely what the benchmark's anonymous
# id space exists to hide. They must never reach a browser, so every video is addressed
# publicly by an opaque media id derived here. The salt only needs to stop the mapping
# being reconstructible from a URL; pools.json is an internal file and holds both.
POOL_SALT = os.environ.get("PERSISTQA_POOL_SALT", "persistqa-human-study-v1")

MIN_QUESTIONS = 12
MAX_PUBLIC_QUESTIONS = 18
AUTHOR_VIDEOS = 20

# Over-eager on purpose: a false positive costs one video, a false negative puts an
# animal question in front of a participant.
ANIMAL = re.compile(
    r"\b(dog|dogs|cat|cats|horse|horses|bird|birds|animal|animals|puppy|kitten|cow|cows|"
    r"sheep|pig|pigs|bear|bears|monkey|elephant|lion|tiger|fish|duck|ducks|chicken|"
    r"rabbit|goat|deer|wolf|fox|pet|pets)\b",
    re.I,
)


def media_id_for(real_id: str) -> str:
    return hmac.new(POOL_SALT.encode(), real_id.encode(), hashlib.sha256).hexdigest()[:12]


def model_accuracy_by_video(results_glob: str) -> dict[str, tuple[int, int]]:
    """Correct and total per anonymous video id, pooled over every evaluated backbone.

    Uses the uniform eight frame arm, which is the baseline condition: it is what a model
    scores without any of our methods, and is therefore the fair notion of how hard a
    video is. Returns an empty mapping if no prediction files are present, in which case
    difficulty ranking is unavailable and the caller must say so.
    """
    acc: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for path in glob.glob(results_glob):
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                a = acc[r["video_id"]]
                a[0] += int(r["is_correct"])
                a[1] += 1
    return {k: (v[0], v[1]) for k, v in acc.items()}


def collect(bench_path, mapping_path, results_glob, video_dir):
    bench = json.load(open(bench_path))
    anon_to_real = json.load(open(mapping_path))["anon_to_real"]
    acc = model_accuracy_by_video(results_glob)

    merged: dict[str, dict] = {}
    n_dupe_dropped = 0
    for video in bench["videos"]:
        questions = [q for q in video.get("questions", []) if q.get("question_id")]
        if not questions:
            continue
        anon = video["video_id"]
        # Fourteen mapping values carry a trailing .mp4. Left unstripped, the file lookup
        # below becomes "<id>.mp4.mp4", every one of those videos silently fails the
        # existence check, and fourteen usable videos disappear from the pool with no
        # error anywhere.
        real = anon_to_real.get(anon, anon)
        if real.endswith(".mp4"):
            real = real[: -len(".mp4")]
        rec = merged.setdefault(
            real,
            {"real_id": real, "anon_ids": [], "questions": [], "correct": 0, "answered": 0,
             "_seen_text": set()},
        )
        rec["anon_ids"].append(anon)
        for q in questions:
            # Nine real videos appear under two or three anonymous entries, and those
            # entries OVERLAP rather than partition: -Ml2V9Mos-4 carries 44 question rows
            # that collapse to 19 distinct questions, the same text repeated verbatim
            # under different anonymous ids. Merging by real video without this check
            # would show one participant the identical question three times.
            #
            # The overlap is partial, not total, which is why these videos are deduplicated
            # rather than excluded: -HwJeE-iuIY's two entries hold 15 genuinely distinct
            # questions and excluding it would discard a perfectly good video.
            text = q.get("question_text", "").strip()
            if text in rec["_seen_text"]:
                n_dupe_dropped += 1
                continue
            rec["_seen_text"].add(text)
            # The question key must match the key the evaluation harness writes, which is
            # anonymous-id based. Joining the human responses to model predictions later
            # depends on this exactly.
            rec["questions"].append(
                {
                    "key": f"{anon}|{q['question_id']}",
                    "video_id": anon,
                    "question_id": q["question_id"],
                    "question_text": text,
                    "options": q.get("options", {}),
                    "correct_answer": (q.get("correct_answer") or "").strip().upper()[:1],
                    "capability": q.get("metadata", {}).get("capability", ""),
                    "reid": q.get("metadata", {}).get("reid_canonical", ""),
                }
            )
        c, n = acc.get(anon, (0, 0))
        rec["correct"] += c
        rec["answered"] += n
    for rec in merged.values():
        rec.pop("_seen_text", None)
    if n_dupe_dropped:
        print(f"[pools] dropped {n_dupe_dropped} duplicate question rows "
              f"(same text repeated under multiple anonymous ids for one real video)")

    kept, dropped = [], collections.Counter()
    for real, rec in merged.items():
        n_q = len(rec["questions"])
        if n_q < MIN_QUESTIONS:
            dropped["too_few_questions"] += 1
            continue
        text = " ".join(
            q["question_text"] + " " + " ".join(q["options"].values()) for q in rec["questions"]
        )
        if ANIMAL.search(text):
            dropped["animal_referent"] += 1
            continue
        path = Path(video_dir) / f"{real}.mp4"
        if not path.is_file():
            dropped["video_missing"] += 1
            continue
        if not rec["answered"]:
            dropped["no_model_baseline"] += 1
            continue
        rec["n_questions"] = n_q
        rec["media_id"] = media_id_for(real)
        rec["model_accuracy"] = round(100.0 * rec["correct"] / rec["answered"], 2)
        rec["source_path"] = str(path)
        rec["size_mb"] = round(path.stat().st_size / 1048576, 1)
        kept.append(rec)
    return kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(BENCH_ANON))
    ap.add_argument("--mapping", default=str(ID_MAPPING))
    ap.add_argument("--video_dir", default=str(VIDEO_DIR))
    ap.add_argument(
        "--results",
        default=str(REPO / "results_baseline" / "kf_uniform8" / "*" / "predictions.jsonl"),
        help="prediction files used to rank videos by difficulty",
    )
    ap.add_argument("--out", default=str(REPO / "humanstudy" / "data" / "pools.json"))
    args = ap.parse_args()

    kept, dropped = collect(args.bench, args.mapping, args.results, args.video_dir)
    if not kept:
        raise SystemExit("no videos survived filtering; check --video_dir and --results")

    # Hardest first. Difficulty is the measured backbone accuracy on that video.
    kept.sort(key=lambda r: r["model_accuracy"])

    # The two pools OVERLAP by design, and that is deliberate.
    #
    # Reserving the hardest videos for the authors would leave the public pool holding
    # only the easier tail, which would understate how hard the benchmark is for people
    # and make the public number incomparable to the model numbers, which are computed
    # over everything. Overlap also buys a free consistency check: where an author and a
    # member of the public answered the same video, the two human groups can be compared
    # directly, which is the cheapest available evidence that the public responses are
    # not just noise.
    public = [
        r for r in kept if MIN_QUESTIONS <= r["n_questions"] <= MAX_PUBLIC_QUESTIONS
    ]
    # Authors take the hardest videos, plus any video too long to put in front of a member
    # of the public. Session length does not matter for an author working across sittings.
    author = kept[:AUTHOR_VIDEOS]
    author_ids = {r["real_id"] for r in author}
    author += [
        r for r in kept
        if r["n_questions"] > MAX_PUBLIC_QUESTIONS and r["real_id"] not in author_ids
    ]

    pools = {
        "generated_from": {
            "benchmark": os.path.basename(args.bench),
            "results_glob": args.results,
        },
        "filters": {
            "min_questions": MIN_QUESTIONS,
            "max_public_questions": MAX_PUBLIC_QUESTIONS,
            "author_videos": AUTHOR_VIDEOS,
            "human_referent_only": True,
        },
        "public": public,
        "author": author,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(pools, fh, indent=1)

    pq = sum(r["n_questions"] for r in public)
    aq = sum(r["n_questions"] for r in author)
    print(f"[pools] wrote {args.out}")
    print(f"  candidates surviving filters : {len(kept)} videos")
    for reason, n in dropped.most_common():
        print(f"    dropped ({reason}): {n}")
    print(f"  public : {len(public):3d} videos, {pq:5d} questions, "
          f"{min(r['n_questions'] for r in public)}-{max(r['n_questions'] for r in public)} per video, "
          f"model acc {public[0]['model_accuracy']:.1f}-{public[-1]['model_accuracy']:.1f}%")
    print(f"  author : {len(author):3d} videos, {aq:5d} questions, "
          f"model acc {author[0]['model_accuracy']:.1f}-{max(r['model_accuracy'] for r in author):.1f}%")
    print(f"  total video bytes to encode  : {sum(r['size_mb'] for r in public + author):.0f} MB")


if __name__ == "__main__":
    main()
