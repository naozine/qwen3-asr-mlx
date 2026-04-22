"""Consume /transcribe-stream, print progress, save merged JSON.

Useful for long audio where the synchronous /transcribe endpoint would
risk an HTTP timeout. The saved JSON uses the same schema as
``transcribe.py``, so it can be dropped into ``player.html`` for
timeline playback.

Usage:
    uv run python stream_client.py /path/to/audio.wav \
        [--url http://localhost:8000] [--language Japanese] \
        [--chunk-duration 60] [--output result_stream.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="audio path as seen by the server (absolute recommended)")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--language", default="Japanese")
    parser.add_argument("--context", default="")
    parser.add_argument("--chunk-duration", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=Path("result_stream.json"))
    args = parser.parse_args()

    data = {
        "path": str(Path(args.audio).expanduser().resolve()),
        "language": args.language,
        "chunk_duration": str(args.chunk_duration),
    }
    if args.context:
        data["context"] = args.context

    endpoint = args.url.rstrip("/") + "/transcribe-stream"
    print(f"POST {endpoint}  path={data['path']}")
    resp = requests.post(endpoint, data=data, stream=True, timeout=None)
    resp.raise_for_status()

    words: list[dict] = []
    text_parts: list[str] = []
    total_duration = 0.0
    chunk_count = 0
    backend = ""
    wall_start = time.time()

    buffer = ""
    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
        buffer += chunk
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            event_name = "message"
            data_lines: list[str] = []
            for line in frame.split("\n"):
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            if not data_lines:
                continue
            payload = json.loads("\n".join(data_lines))

            if event_name == "start":
                total_duration = payload["total_duration"]
                chunk_count = payload["chunk_count"]
                backend = payload.get("backend", "")
                chunker = payload.get("chunker", "")
                speech = payload.get("speech_coverage", total_duration)
                print(
                    f"start  backend={backend}  chunker={chunker}  "
                    f"duration={total_duration:.1f}s  chunks={chunk_count}  "
                    f"speech={speech:.1f}s"
                )
            elif event_name == "chunk":
                words.extend(payload["words"])
                text_parts.append(payload["text"])
                progress = (
                    (payload["chunk_start"] + payload["chunk_duration"]) / total_duration
                    if total_duration
                    else 0
                )
                wall = time.time() - wall_start
                print(
                    f"  chunk {payload['chunk_index'] + 1}/{chunk_count} "
                    f"@ {payload['chunk_start']:.1f}s  "
                    f"asr {payload['asr_seconds']:.1f}s  "
                    f"({progress * 100:.1f}%, wall {wall:.0f}s)"
                )
            elif event_name == "done":
                wall = time.time() - wall_start
                print(
                    f"done   words={payload['total_words']}  "
                    f"wall={wall:.1f}s  total_asr={payload['total_asr_seconds']:.1f}s"
                )
            elif event_name == "error":
                print(f"error: {payload.get('message')}", file=sys.stderr)
                sys.exit(1)

    result = {
        "text": "".join(text_parts).strip(),
        "language": args.language,
        "words": words,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved  {args.output}")


if __name__ == "__main__":
    main()
