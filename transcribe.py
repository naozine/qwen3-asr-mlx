"""Qwen3-ASR-1.7B (bf16) で転写し、Qwen3-ForcedAligner-0.6B (8bit) で単語タイムスタンプを付与。

Usage:
    uv run python transcribe.py path/to/audio.wav [--language Japanese]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mlx_audio.stt.utils import load_model

ASR_MODEL = "mlx-community/Qwen3-ASR-1.7B-bf16"
ALIGNER_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="入力音声ファイル (wav/mp3/...)")
    parser.add_argument(
        "--language",
        default="Japanese",
        help="ForcedAligner 用の言語名 (Japanese/English/Chinese/Korean など)",
    )
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    parser.add_argument("--context", default="", help="ホットワード等のコンテキスト")
    args = parser.parse_args()

    print(f"[1/2] ASR ロード中: {ASR_MODEL}")
    t0 = time.time()
    asr = load_model(ASR_MODEL)
    print(f"  → {time.time() - t0:.1f}s")

    print("[1/2] 転写中...")
    t0 = time.time()
    asr_kwargs = {"verbose": False}
    if args.context:
        asr_kwargs["context"] = args.context
    result = asr.generate(str(args.audio), **asr_kwargs)
    transcript = result.text
    print(f"  → {time.time() - t0:.1f}s")
    print(f"  検出言語: {result.language}")
    print(f"  転写: {transcript}")

    print(f"\n[2/2] ForcedAligner ロード中: {ALIGNER_MODEL}")
    t0 = time.time()
    aligner = load_model(ALIGNER_MODEL)
    print(f"  → {time.time() - t0:.1f}s")

    print("[2/2] タイムスタンプ付与中...")
    t0 = time.time()
    align = aligner.generate(str(args.audio), text=transcript, language=args.language)
    print(f"  → {time.time() - t0:.1f}s")

    words = [
        {"text": it.text, "start": it.start_time, "end": it.end_time}
        for it in align.items
    ]
    for w in words:
        print(f"  [{w['start']:7.2f} - {w['end']:7.2f}] {w['text']}")

    args.output.write_text(
        json.dumps(
            {"text": transcript, "language": result.language, "words": words},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n保存: {args.output}")


if __name__ == "__main__":
    main()
