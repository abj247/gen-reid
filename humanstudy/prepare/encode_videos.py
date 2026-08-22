#!/usr/bin/env python
"""Prepare the pooled videos so a browser can stream and seek them cheaply.

Why this step exists
--------------------
Seeking is the real problem. Source files carry the moov atom at the end of the file, so
a browser must pull most of the file before it can seek at all. Participants are expected
to scrub back to re-check who was wearing what, and without `-movflags +faststart` that
scrub stalls. Relocating the atom is the single change that matters here.

What this deliberately does NOT do
----------------------------------
It does not re-encode by default. The instinct is to transcode everything to a uniform
480p, and on this corpus that is actively wrong: the sources are already 640x360 H.264 at
a modest bitrate, so a 480p CRF 26 pass upscales them and produces MORE bytes while
throwing away a generation of quality. Measured on two pool videos, that pass produced
128% of the original size. The default is therefore a stream copy, which is lossless,
near-instant, and still fixes seeking.

Re-encoding runs only for files above REMUX_UNDER_MB or taller than MAX_HEIGHT, where the
file is genuinely too heavy for a first page load and trading quality for bytes is worth
it. The scale filter uses min(MAX_HEIGHT, ih) so a short source is never upscaled.

Audio is preserved. The questions are visual, but muting the video changes the viewing
experience and would be an unnecessary confound.

Run:
  python humanstudy/prepare/encode_videos.py                 # prepare everything in the pools
  python humanstudy/prepare/encode_videos.py --limit 3       # smoke test first
  python humanstudy/prepare/encode_videos.py --force         # redo existing outputs
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

MAX_HEIGHT = 480      # ceiling, never a target: upscaling adds bytes and no information
CRF = 28              # only applied to the oversized files that are actually re-encoded
PRESET = "slow"       # encode time is a one-off, bytes are paid on every participant
AUDIO_KBPS = 96
REMUX_UNDER_MB = 12   # below this, copying the streams is strictly better than re-encoding


def probe(src: Path, ffprobe: str) -> tuple[int, int]:
    """Return (height, size_mb). Height 0 means the probe failed."""
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=height", "-of", "csv=p=0", str(src)],
            check=True, capture_output=True, text=True, timeout=60).stdout.strip()
        height = int(out.splitlines()[0])
    except Exception:
        height = 0
    return height, src.stat().st_size / 1048576


def encode(src: Path, dst: Path, ffmpeg: str, ffprobe: str) -> tuple[bool, str, str]:
    """Prepare one video for the browser. Returns (ok, mode, error).

    The corpus is already 640x360 H.264 at a modest bitrate, so a blanket re-encode makes
    files LARGER while discarding a generation of quality. Measured on two samples: a
    480p CRF 26 pass produced 128% of the original bytes. So the default is a stream copy
    that only relocates the moov atom, which is lossless, near-instant, and fixes the
    seeking problem that actually affects participants.

    Re-encoding is reserved for files that are genuinely too heavy to put on a first page
    load, or that are taller than the ceiling. Those are the only cases where spending
    quality to save bytes is the right trade.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    height, size_mb = probe(src, ffprobe)

    if size_mb <= REMUX_UNDER_MB and 0 < height <= MAX_HEIGHT:
        mode = "remux"
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
               "-c", "copy", "-movflags", "+faststart", str(tmp)]
    else:
        mode = "reencode"
        # min() in the filter means a 360p source stays 360p; only taller sources shrink.
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
               "-vf", f"scale=-2:'min({MAX_HEIGHT},ih)'",
               "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
               "-crf", str(CRF), "-preset", PRESET,
               "-c:a", "aac", "-b:a", f"{AUDIO_KBPS}k", "-ac", "2",
               "-movflags", "+faststart", str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        tail = (e.stderr or "").strip().splitlines()
        return False, mode, (tail[-1][:160] if tail else "ffmpeg failed")
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return False, mode, "timeout"

    # A remux that somehow grew is not worth keeping; fall back rather than ship bytes.
    if mode == "remux" and tmp.stat().st_size > src.stat().st_size * 1.05:
        tmp.unlink(missing_ok=True)
        return encode_reencode_only(src, dst, ffmpeg)
    tmp.replace(dst)
    return True, mode, ""


def encode_reencode_only(src: Path, dst: Path, ffmpeg: str) -> tuple[bool, str, str]:
    tmp = dst.with_suffix(".tmp.mp4")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
           "-vf", f"scale=-2:'min({MAX_HEIGHT},ih)'",
           "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
           "-crf", str(CRF), "-preset", PRESET,
           "-c:a", "aac", "-b:a", f"{AUDIO_KBPS}k", "-ac", "2",
           "-movflags", "+faststart", str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return False, "reencode", str(e)[:160]
    tmp.replace(dst)
    return True, "reencode", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=str(REPO / "humanstudy" / "data" / "pools.json"))
    ap.add_argument("--out", default=str(REPO / "humanstudy" / "data" / "video"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe must both be on PATH")

    pools = json.load(open(args.pools))
    # A video in both pools must be encoded once, not twice.
    videos = {r["real_id"]: r for r in pools["public"] + pools["author"]}
    items = sorted(videos.values(), key=lambda r: r["media_id"])
    if args.limit:
        items = items[: args.limit]

    out_dir = Path(args.out)
    print(f"[encode] {len(items)} videos -> {out_dir}  "
          f"(remux under {REMUX_UNDER_MB} MB, else h264 <={MAX_HEIGHT}p crf{CRF}; faststart always)", flush=True)

    done = skipped = failed = 0
    modes = {"remux": 0, "reencode": 0}
    src_mb = dst_mb = 0.0
    for i, rec in enumerate(items, 1):
        src = Path(rec["source_path"])
        dst = out_dir / f"{rec['media_id']}.mp4"
        if dst.is_file() and not args.force:
            skipped += 1
            dst_mb += dst.stat().st_size / 1048576
            src_mb += src.stat().st_size / 1048576
            continue
        if not src.is_file():
            print(f"  [missing] {rec['media_id']}", flush=True)
            failed += 1
            continue
        ok, mode, err = encode(src, dst, ffmpeg, ffprobe)
        if not ok:
            print(f"  [fail] {rec['media_id']} ({mode}): {err}", flush=True)
            failed += 1
            continue
        done += 1
        modes[mode] = modes.get(mode, 0) + 1
        src_mb += src.stat().st_size / 1048576
        dst_mb += dst.stat().st_size / 1048576
        if i % 10 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] encoded={done} skipped={skipped} failed={failed}", flush=True)

    print(f"[encode] done: {done} prepared ({modes['remux']} remuxed, {modes['reencode']} re-encoded), "
          f"{skipped} already present, {failed} failed", flush=True)
    if dst_mb:
        print(f"[encode] {src_mb:.0f} MB source -> {dst_mb:.0f} MB encoded "
              f"({100 * dst_mb / max(src_mb, 1e-9):.0f}% of original)", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
