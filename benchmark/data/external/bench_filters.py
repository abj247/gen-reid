"""Benchmark-scoped filters shared by eval_bimba_video_text.py and eval_timeviper_video_text.py.

Both runners are acknowledged copy-paste twins. These two filters decide WHICH QUESTIONS GET
SCORED, so if the two copies ever drift the two models are silently evaluated on different key
sets and every cross-model comparison becomes meaningless. Hence one definition, imported by both.

Pure stdlib on purpose: the runners live in different conda envs (`bimba` has torch 2.1.2 /
transformers 4.40.0.dev0, `timeviper` has torch 2.7.0 / transformers 4.56.2), so anything imported
by both must not depend on either stack.
"""
import json
import os

import os as _os
from pathlib import Path as _Path

ROOT = _os.environ.get("PERSISTQA_ROOT") or str(_Path(__file__).resolve().parents[3])
UNDECODABLE = f"{ROOT}/benchmark/data/external/undecodable_videos.json"
ID_MAPPING = f"{ROOT}/benchmark/data/video_id_mapping.json"
CORE_SUBSET = f"{ROOT}/benchmark/data/external/subset_core.json"


def project_undecodable():
    """The 30 undecodable MovieChat videos, in BOTH id spaces. Empty set if the files are absent.

    Returned rather than applied: the caller only filters when the loaded benchmark actually
    contains one of these ids, so running MLVU / LongVideoBench through the same code path is a
    no-op instead of a silent filter on an unrelated key space.
    """
    if not os.path.isfile(UNDECODABLE):
        return set()
    undec = set(json.load(open(UNDECODABLE)))
    if os.path.isfile(ID_MAPPING):
        r2a = json.load(open(ID_MAPPING)).get("real_to_anon", {})
        undec |= {r2a.get(v, v) for v in list(undec)}
    return undec


def subset_keys(subset="all", subset_file=""):
    """Set of "<video_id>|<question_id>" keys to keep, or None for "keep everything".

    `--subset core` is the MovieChat method subset (780 keys). `--subset_file` points at any key
    list in the same format, e.g. the stratified external-benchmark subsets emitted by
    analysis3/bench/convert_external.py --stratified_subset.
    """
    path = subset_file or (CORE_SUBSET if subset == "core" else "")
    if not path:
        return None
    if not os.path.isfile(path):
        raise SystemExit(f"subset file not found: {path}")
    keys = json.load(open(path))
    if not isinstance(keys, list) or not keys:
        raise SystemExit(f"subset file {path} is not a non-empty JSON list of keys")
    return set(keys)
