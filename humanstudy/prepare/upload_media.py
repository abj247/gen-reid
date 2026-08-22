#!/usr/bin/env python
"""Upload the prepared videos to S3-compatible object storage.

Why object storage rather than the application
-----------------------------------------------
Every participant streams one video, averaging about seven megabytes. Free application
hosting meters egress in single-digit gigabytes, so serving video from the app exhausts
the allowance within a few hundred sessions and then either bills or breaks. Object
storage with free or cheap egress moves those bytes off the critical path entirely, and
the application's own egress drops to HTML and JSON, a few tens of kilobytes per session.

Cloudflare R2 is the intended target because its egress is free and its free storage tier
comfortably holds this pool. Any S3-compatible endpoint works: pass --endpoint-url.

Credentials come from the environment, never from a file in the repository:
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

Run:
  export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
  python humanstudy/prepare/upload_media.py \\
      --endpoint-url https://<account-id>.r2.cloudflarestorage.com \\
      --bucket persistqa-study --prefix media/

Then set VIDEO_BASE_URL to the public base URL for that prefix.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", default=str(REPO / "humanstudy" / "data" / "video"))
    ap.add_argument("--pools", default=str(REPO / "humanstudy" / "data" / "pools.json"))
    ap.add_argument("--endpoint-url", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        import boto3
    except ImportError:
        raise SystemExit("boto3 is required: pip install boto3")

    if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
        raise SystemExit("set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in the environment")

    pools = json.load(open(args.pools))
    expected = {r["media_id"] for r in pools["public"] + pools["author"]}
    video_dir = Path(args.video_dir)
    present = {p.stem for p in video_dir.glob("*.mp4") if not p.name.endswith(".tmp.mp4")}

    # A video in the pool with no uploaded file becomes a participant staring at a dead
    # player, so the gap is reported before anything is uploaded rather than discovered
    # by whoever is assigned that video.
    missing = sorted(expected - present)
    if missing:
        print(f"[warn] {len(missing)} pooled videos have no prepared file and will 404 "
              f"if assigned: {missing[:5]}{' ...' if len(missing) > 5 else ''}", file=sys.stderr)
        print("       run prepare/encode_videos.py first", file=sys.stderr)

    to_upload = sorted(expected & present)
    total_mb = sum((video_dir / f"{m}.mp4").stat().st_size for m in to_upload) / 1048576
    print(f"[upload] {len(to_upload)} files, {total_mb:.0f} MB -> "
          f"{args.bucket}/{args.prefix}  ({args.endpoint_url})")
    if args.dry_run:
        return

    client = boto3.client("s3", endpoint_url=args.endpoint_url)
    for i, media_id in enumerate(to_upload, 1):
        key = f"{args.prefix}{media_id}.mp4"
        client.upload_file(
            str(video_dir / f"{media_id}.mp4"), args.bucket, key,
            # Content-Type must be set explicitly or the browser will not treat the
            # response as playable media. Long immutable caching is safe because the
            # filename is a content-derived id that never changes meaning.
            ExtraArgs={"ContentType": "video/mp4",
                       "CacheControl": "public, max-age=31536000, immutable"},
        )
        if i % 10 == 0 or i == len(to_upload):
            print(f"  [{i}/{len(to_upload)}]", flush=True)
    print("[upload] done. Set VIDEO_BASE_URL to the public base URL for this prefix.")


if __name__ == "__main__":
    main()
