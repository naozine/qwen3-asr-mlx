"""Shared transcription logic used by both the CLI and the REST API."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mlx_audio.stt.utils import load_model

ASR_MODEL_ID = "mlx-community/Qwen3-ASR-1.7B-bf16"
ALIGNER_MODEL_ID = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"

_lock = threading.Lock()
_asr = None
_aligner = None


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class TranscriptionResult:
    text: str
    language: str
    words: list[Word]
    asr_seconds: float
    align_seconds: float


def load_models() -> None:
    """Load ASR and aligner models. No-op if already loaded."""
    global _asr, _aligner
    with _lock:
        if _asr is None:
            _asr = load_model(ASR_MODEL_ID)
        if _aligner is None:
            _aligner = load_model(ALIGNER_MODEL_ID)


def transcribe(
    audio_path: str | Path,
    language: str = "Japanese",
    context: Optional[str] = None,
) -> TranscriptionResult:
    """Transcribe an audio file and attach word-level timestamps.

    MLX models are treated as non-thread-safe, so the whole pipeline is
    serialized under a single lock.
    """
    import time

    load_models()
    audio_str = str(audio_path)

    with _lock:
        t0 = time.time()
        asr_kwargs: dict = {"verbose": False}
        if context:
            asr_kwargs["context"] = context
        asr_out = _asr.generate(audio_str, **asr_kwargs)
        asr_seconds = time.time() - t0

        t0 = time.time()
        align = _aligner.generate(audio_str, text=asr_out.text, language=language)
        align_seconds = time.time() - t0

    words = [Word(text=it.text, start=it.start_time, end=it.end_time) for it in align.items]
    lang = asr_out.language
    if isinstance(lang, list):
        lang = lang[0] if lang else ""
    return TranscriptionResult(
        text=asr_out.text,
        language=str(lang),
        words=words,
        asr_seconds=asr_seconds,
        align_seconds=align_seconds,
    )
