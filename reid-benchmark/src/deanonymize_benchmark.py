#!/usr/bin/env python3
"""
De-anonymize the committee-filtered benchmark so video_id becomes the real
YouTube ID again. The video+text evaluator expects video_id to match the
.mp4 filename in /home/c3-0/datasets/moviechat1k-test/.

Usage:
    python deanonymize_filtered.py \\
        --input   combined_all_hard_v3.json \\
        --mapping video_id_mapping.json \\
        --output  combined_all_hard_v3_real_ids.json
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--mapping", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.input) as f:
        bench = json.load(f)
    with open(args.mapping) as f:
        mp = json.load(f)
    anon_to_real = mp["anon_to_real"]

    new_videos = []
    missing = 0
    for v in bench.get("videos", []):
        anon = v.get("video_id", "")
        real = anon_to_real.get(anon)
        if real is None:
            print(f"WARN: no mapping for {anon} - skipping")
            missing += 1
            continue
        v_copy = dict(v)
        v_copy["video_id"] = real
        v_copy["anon_id"] = anon
        new_videos.append(v_copy)

    out = dict(bench)
    out["videos"] = new_videos
    out["video_id_anonymized"] = False
    out["video_id_source"] = "real_youtube_ids_via_mapping"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    total_q = sum(len(v.get("questions", [])) for v in new_videos)
    print(f"Wrote {args.output}: {len(new_videos)} videos, {total_q} questions, "
          f"missing={missing}")


if __name__ == "__main__":
    main()
