"""Qwen3-ASR + ForcedAligner CLI.

Usage:
    uv run python transcribe.py path/to/audio.wav [--language Japanese]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from transcriber import ASR_MODEL_ID, ALIGNER_MODEL_ID, load_models, transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="input audio file (wav/mp3/...)")
    parser.add_argument("--language", default="Japanese")
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    parser.add_argument("--context", default="", help="hotwords / domain terms to inject")
    args = parser.parse_args()

    print(f"Loading models: {ASR_MODEL_ID} / {ALIGNER_MODEL_ID}")
    t0 = time.time()
    load_models()
    print(f"  -> {time.time() - t0:.1f}s")

    print("Transcribing and aligning...")
    result = transcribe(args.audio, language=args.language, context=args.context or None)
    print(f"  ASR: {result.asr_seconds:.1f}s  /  Align: {result.align_seconds:.1f}s")
    print(f"  Detected language: {result.language}")
    print(f"  Transcript: {result.text}")
    for w in result.words:
        print(f"  [{w.start:7.2f} - {w.end:7.2f}] {w.text}")

    args.output.write_text(
        json.dumps(
            {
                "text": result.text,
                "language": result.language,
                "words": [asdict(w) for w in result.words],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
