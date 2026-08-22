"""Pool loading and video assignment.

The assignment rule is the part of this file that matters. Everything else is lookup.

Why not uniform random
----------------------
Drawing a video uniformly at random for each participant sounds fair and is not. With 76
videos and, say, 120 participants, uniform sampling leaves a long tail of videos with one
response or none, while others collect five. Per-video human accuracy over one respondent
is uninformative, so the tail is wasted effort, and the aggregate is skewed toward
whichever videos happened to be drawn often.

Instead: draw uniformly at random from among the videos that currently have the FEWEST
completed sessions. Early on that is every video, so the first pass is effectively a
random permutation; afterwards it fills the thinnest coverage first. The result converges
to even coverage without ever being deterministic, which matters because a deterministic
order would let a returning participant predict, and would correlate assignment with time
of day.

In-progress sessions count toward the load. Without that, several people starting at once
are all handed the same video, which is exactly what happens when a study link is shared.
"""
from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session as OrmSession

from .db import Session as StudySession

POOLS_PATH = Path(__file__).resolve().parents[1] / "data" / "pools.json"


@lru_cache(maxsize=1)
def load_pools() -> dict:
    """Read the frozen pool file once per process.

    The deployed app depends on this file and nothing else from the research tree: no
    benchmark JSON, no prediction files, no video corpus. Regenerate it with
    prepare/build_pools.py when the benchmark changes.
    """
    with open(POOLS_PATH) as fh:
        pools = json.load(fh)
    for name in ("public", "author"):
        if not pools.get(name):
            raise RuntimeError(f"pool '{name}' is empty in {POOLS_PATH}")
    return pools


@lru_cache(maxsize=1)
def video_index() -> dict[str, dict]:
    """real_video_id -> pool record, merged across both pools."""
    pools = load_pools()
    index: dict[str, dict] = {}
    for record in pools["public"] + pools["author"]:
        index.setdefault(record["real_id"], record)
    return index


@lru_cache(maxsize=1)
def question_index() -> dict[str, dict]:
    """question_key -> question record, across every pooled video."""
    index: dict[str, dict] = {}
    for record in video_index().values():
        for question in record["questions"]:
            index[question["key"]] = question
    return index


def pool_video_ids(mode: str) -> list[str]:
    pools = load_pools()
    key = "author" if mode == "author" else "public"
    return [r["real_id"] for r in pools[key]]


def load_counts(db: OrmSession, video_ids: list[str]) -> dict[str, int]:
    """Sessions per video that are completed or still open.

    Abandoned sessions are excluded: a video someone walked away from still needs
    answers, so it should not be treated as covered.
    """
    rows = db.execute(
        select(StudySession.video_id, func.count(StudySession.id))
        .where(StudySession.video_id.in_(video_ids))
        .where(StudySession.status.in_(("completed", "in_progress")))
        .group_by(StudySession.video_id)
    ).all()
    counts = {video_id: 0 for video_id in video_ids}
    for video_id, n in rows:
        counts[video_id] = n
    return counts


def choose_video(db: OrmSession, mode: str, exclude: set[str] | None = None) -> str | None:
    """Pick the next video for a participant, balancing coverage.

    `exclude` carries the videos this participant has already been assigned, so nobody is
    shown the same footage twice. In author mode that is how a session walks through the
    set without repeats; in public mode it only matters for someone who returns.
    """
    candidates = [v for v in pool_video_ids(mode) if not exclude or v not in exclude]
    if not candidates:
        return None
    counts = load_counts(db, candidates)
    fewest = min(counts[v] for v in candidates)
    thinnest = [v for v in candidates if counts[v] == fewest]
    return random.choice(thinnest)


def question_order(video_id: str, seed: str) -> list[str]:
    """The frozen question order for one assignment.

    Presentation order is shuffled per session, deterministically from the session id, so
    that any ordering effect (fatigue on later questions, for instance) is spread across
    questions rather than always landing on whichever ones the benchmark file happens to
    list last. Seeding from the session id keeps it reproducible for audit.
    """
    keys = [q["key"] for q in video_index()[video_id]["questions"]]
    rng = random.Random(seed)
    rng.shuffle(keys)
    return keys
