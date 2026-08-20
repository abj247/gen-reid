"""Filesystem layout resolution.

Every path used by the repository is derived here rather than hardcoded at the
call site. The repository root is taken from the PERSISTQA_ROOT environment
variable when it is set, and otherwise inferred from the location of this file,
so a clone works without configuration and a relocated checkout works by
exporting one variable.

Video corpora are not part of the repository. Set PERSISTQA_VIDEO_DIR to the
directory holding the source .mp4 files; see benchmark/data/README.md for how
that directory is expected to be laid out.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("PERSISTQA_ROOT", Path(__file__).resolve().parent.parent))

BENCHMARK_DIR = ROOT / "benchmark"
DATA_DIR = BENCHMARK_DIR / "data"
SOLUTIONS_DIR = ROOT / "solutions"

# Benchmark data files. See benchmark/data/README.md for the schema of each.
BENCH_ANON = DATA_DIR / "persistqa.json"
BENCH_REAL = DATA_DIR / "persistqa_real_ids.json"
ID_MAPPING = DATA_DIR / "video_id_mapping.json"
EVIDENCE_WINDOWS = DATA_DIR / "evidence_windows.json"

# Outputs. Kept out of version control; see .gitignore.
RESULTS_DIR = Path(os.environ.get("PERSISTQA_RESULTS", ROOT / "results_baseline"))

VIDEO_DIR = Path(os.environ.get("PERSISTQA_VIDEO_DIR", "/home/c3-0/datasets/moviechat1k-test"))


def video_path(video_id: str, id_map: dict | None = None) -> Path | None:
    """Resolve a video id to a file, trying the real id before the anonymous one.

    The benchmark ships two id spaces (see benchmark/data/README.md). Callers
    holding either one can pass the mapping and get the same answer.
    """
    candidates = [video_id]
    if id_map:
        real = id_map.get(video_id)
        if real:
            candidates.insert(0, real)
    for cand in candidates:
        p = VIDEO_DIR / f"{cand}.mp4"
        if p.is_file():
            return p
    return None
