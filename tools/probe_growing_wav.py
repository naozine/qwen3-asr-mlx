"""Observe a WAV file that Audio Hijack (or any recorder) is still writing to.

Runs for ~30 seconds, sampling the file every 2s:
- File size
- ffprobe-reported duration (format=duration)
- Duration as parsed via `ffmpeg -ignore_length 1` (tolerant to header lies)
- Whether ffmpeg can decode the whole current content

Usage:
    uv run python tools/probe_growing_wav.py path/to/live_recording.wav
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], timeout: int = 15) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"


def ffprobe_duration(path: Path) -> str:
    rc, out, err = run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return out.strip() if rc == 0 and out.strip() else f"ERR({err[:60]})"


def ffmpeg_scan_duration(path: Path, ignore_length: bool) -> str:
    cmd = ["ffmpeg", "-v", "error", "-stats"]
    if ignore_length:
        cmd += ["-ignore_length", "1"]
    cmd += ["-i", str(path), "-f", "null", "-"]
    rc, _, err = run(cmd)
    # ffmpeg prints progress lines to stderr with time=HH:MM:SS.xx
    matches = re.findall(r"time=(\d+:\d+:\d+\.\d+)", err)
    if matches:
        return matches[-1]
    return f"no-time (rc={rc})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args()

    if not args.path.exists():
        print(f"waiting for {args.path} to appear...")
        while not args.path.exists():
            time.sleep(0.3)

    print(f"probing {args.path}")
    print(f"{'i':>3}  {'size':>12}  {'ffprobe_dur':>14}  "
          f"{'ffmpeg_scan':>12}  {'with_ignore_len':>14}  growth")
    prev_size = 0
    for i in range(args.samples):
        size = args.path.stat().st_size if args.path.exists() else 0
        prob = ffprobe_duration(args.path)
        scan_raw = ffmpeg_scan_duration(args.path, ignore_length=False)
        scan_ign = ffmpeg_scan_duration(args.path, ignore_length=True)
        growth = size - prev_size
        prev_size = size
        print(
            f"{i:>3}  {size:>12}  {prob:>14}  "
            f"{scan_raw:>12}  {scan_ign:>14}  +{growth}"
        )
        sys.stdout.flush()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
