"""Whisper transcription backend with the same interface as transcriber.py.

Produces `TranscriptionResult` (defined in transcriber) so api.py can use
either backend interchangeably.
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Optional

from mlx_audio.stt.utils import load_model

from transcriber import TranscriptionResult, Word

ASR_MODEL_ID = "mlx-community/whisper-large-v3-turbo"
# Whisper has built-in word timestamps; no separate aligner is needed.
ALIGNER_MODEL_ID = "(built-in)"

_lock = threading.Lock()
_model = None

# mlx-community repos typically ship weights only; we pull the rest of the
# HF WhisperProcessor files from the matching openai/* repo on first run.
_PROCESSOR_FILES = [
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
]

# Qwen3-ASR uses language *names* ("Japanese"); Whisper uses ISO codes ("ja").
# We accept both for API compatibility.
_LANGUAGE_NAME_TO_CODE = {
    "japanese": "ja",
    "english": "en",
    "chinese": "zh",
    "korean": "ko",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "italian": "it",
    "portuguese": "pt",
    "russian": "ru",
    "arabic": "ar",
}


def _openai_equivalent(model_id: str) -> str | None:
    """Map an mlx-community Whisper repo to the matching openai repo."""
    if not model_id.startswith("mlx-community/"):
        return None
    name = model_id.split("/", 1)[1]
    for suffix in ("-q4", "-q8", "-fp16", "-bf16", "-mlx"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return f"openai/{name}"


def ensure_whisper_processor(model_id: str) -> None:
    """Top up missing HF processor files from openai/* before first load."""
    if "whisper" not in model_id.lower():
        return
    source_id = _openai_equivalent(model_id)
    if source_id is None:
        return

    from huggingface_hub import hf_hub_download, snapshot_download

    local_dir = Path(snapshot_download(repo_id=model_id))
    missing = [f for f in _PROCESSOR_FILES if not (local_dir / f).exists()]
    if not missing:
        return

    print(f"Fetching {len(missing)} processor file(s) from {source_id}...")
    for fn in missing:
        try:
            src = hf_hub_download(repo_id=source_id, filename=fn)
            shutil.copy(src, local_dir / fn)
        except Exception:
            pass  # optional files may not exist in the source repo


def extract_words(segments: list) -> list[Word]:
    """Flatten Whisper's segment/word structure into a flat Word list."""
    words: list[Word] = []
    for seg in segments:
        for w in seg.get("words", []) or []:
            if isinstance(w, dict):
                text, start, end = w.get("word", ""), w.get("start"), w.get("end")
            else:
                text = getattr(w, "word", "")
                start = getattr(w, "start", None)
                end = getattr(w, "end", None)
            if start is None or end is None:
                continue
            words.append(Word(text=text.strip(), start=float(start), end=float(end)))
    return words


def load_models() -> None:
    """Load the Whisper model. No-op if already loaded."""
    global _model
    with _lock:
        if _model is None:
            ensure_whisper_processor(ASR_MODEL_ID)
            _model = load_model(ASR_MODEL_ID)


def _to_language_code(language: str) -> Optional[str]:
    if not language:
        return None
    if len(language) == 2:
        return language.lower()
    return _LANGUAGE_NAME_TO_CODE.get(language.lower(), language.lower())


def transcribe(
    audio_path: str | Path,
    language: str = "Japanese",
    context: Optional[str] = None,  # unused; kept for API compatibility
) -> TranscriptionResult:
    """Transcribe an audio file with Whisper (word timestamps built-in)."""
    del context  # Whisper does not use hotword context prompts
    load_models()

    with _lock:
        t0 = time.time()
        result = _model.generate(
            str(audio_path),
            language=_to_language_code(language),
            word_timestamps=True,
            verbose=False,
        )
        asr_seconds = time.time() - t0

    words = extract_words(result.segments)
    return TranscriptionResult(
        text=result.text,
        language=str(result.language),
        words=words,
        asr_seconds=asr_seconds,
        align_seconds=0.0,  # alignment is fused into the ASR pass
    )
