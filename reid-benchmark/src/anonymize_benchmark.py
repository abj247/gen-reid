#!/usr/bin/env python3
"""
Anonymize video_id fields in the MovieChat-1k merged debiased benchmark.

Reads the source JSON, replaces every real video_id (e.g. YouTube IDs like
"-_HlyIgHUa0") with a deterministic anonymized ID (vid_0001, vid_0002, ...),
and writes a parallel mapping file so we can join back to the original IDs
when comparing text-only and video-mode results.

The anonymized IDs are assigned by sorting the original IDs lexicographically.
This keeps ordering stable across runs and matches the positional zip used by
compare_text_vs_video.py.

Usage:
    python strip_video_ids.py \\
        --input  /home/c3-0/datasets/moviechat1k-test/combined_all.json \\
        --output combined_all_text_only.json \\
        --mapping video_id_mapping.json
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Path to source combined_all.json")
    p.add_argument("--output", required=True, help="Path to write anonymized JSON")
    p.add_argument("--mapping", required=True, help="Path to write {anon_id: real_id} mapping JSON")
    args = p.parse_args()

    src = Path(args.input)
    out = Path(args.output)
    mapping_path = Path(args.mapping)

    print(f"Loading: {src}")
    with src.open() as f:
        data = json.load(f)

    videos = data.get("videos", [])
    print(f"Videos: {len(videos)}")

    sorted_videos = sorted(videos, key=lambda v: v.get("video_id", ""))

    anon_to_real = {}
    real_to_anon = {}
    new_videos = []
    for i, v in enumerate(sorted_videos, start=1):
        anon_id = f"vid_{i:04d}"
        real_id = v.get("video_id", "")
        anon_to_real[anon_id] = real_id
        real_to_anon[real_id] = anon_id
        v_copy = {k: val for k, val in v.items() if k != "video_id"}
        v_copy["video_id"] = anon_id
        new_videos.append(v_copy)

    new_data = {k: val for k, val in data.items() if k != "videos"}
    new_data["videos"] = new_videos
    new_data["video_id_anonymized"] = True
    new_data["video_id_mapping_file"] = str(mapping_path.name)

    total_q = sum(len(v.get("questions", [])) for v in new_videos)
    print(f"Total questions: {total_q}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(new_data, f, indent=2)
    print(f"Wrote anonymized benchmark: {out}")

    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w") as f:
        json.dump({"anon_to_real": anon_to_real, "real_to_anon": real_to_anon}, f, indent=2)
    print(f"Wrote mapping: {mapping_path}")

    sample_anon = next(iter(anon_to_real))
    print(f"\nSpot check: {sample_anon} -> {anon_to_real[sample_anon]}")


if __name__ == "__main__":
    main()
