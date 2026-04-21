"""Whisper CLI: a thin wrapper around transcriber_whisper.

Output schema matches transcribe.py so both results can be dropped into
``player.html`` for visual diffing.

Usage:
    uv run python transcribe_whisper.py path/to/audio.wav [--language ja]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import transcriber_whisper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="input audio file (wav/mp3/...)")
    parser.add_argument("--model", default=transcriber_whisper.ASR_MODEL_ID)
    parser.add_argument("--language", default="ja", help="ISO code e.g. ja, en")
    parser.add_argument("--output", type=Path, default=Path("result_whisper.json"))
    args = parser.parse_args()

    transcriber_whisper.ASR_MODEL_ID = args.model

    print(f"Loading model: {args.model}")
    t0 = time.time()
    transcriber_whisper.load_models()
    print(f"  -> {time.time() - t0:.1f}s")

    print("Transcribing with word_timestamps...")
    result = transcriber_whisper.transcribe(args.audio, language=args.language)
    print(f"  -> {result.asr_seconds:.1f}s")
    print(f"  Detected language: {result.language}")
    print(f"  Transcript: {result.text}")
    print(f"  Words with timestamps: {len(result.words)}")

    args.output.write_text(
        json.dumps(
            {
                "text": result.text,
                "language": result.language,
                "words": [asdict(w) for w in result.words],
                "asr_seconds": result.asr_seconds,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
