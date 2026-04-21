"""Qwen3-ASR / Whisper REST API (synchronous).

Run:
    uv run uvicorn api:app --host 0.0.0.0 --port 8000
    # LAN clients can reach the OpenAPI UI at http://<host-ip>:8000/docs

Select the backend with the ``ASR_BACKEND`` environment variable:
    ASR_BACKEND=qwen3    (default) Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B
    ASR_BACKEND=whisper  Whisper-large-v3-turbo (word timestamps built in)

Endpoints:
    GET  /health
    POST /transcribe  (multipart: file, language, context)
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transcriber import TranscriptionResult

ASR_BACKEND = os.environ.get("ASR_BACKEND", "qwen3").lower()

if ASR_BACKEND == "whisper":
    from transcriber_whisper import (
        ALIGNER_MODEL_ID,
        ASR_MODEL_ID,
        load_models,
        transcribe,
    )
elif ASR_BACKEND == "qwen3":
    from transcriber import (
        ALIGNER_MODEL_ID,
        ASR_MODEL_ID,
        load_models,
        transcribe,
    )
else:
    raise ValueError(
        f"Unknown ASR_BACKEND={ASR_BACKEND!r} (expected 'qwen3' or 'whisper')"
    )


class WordOut(BaseModel):
    text: str
    start: float
    end: float


class TranscribeOut(BaseModel):
    text: str
    language: str
    words: list[WordOut]
    asr_seconds: float
    align_seconds: float


class HealthOut(BaseModel):
    status: str
    backend: str
    asr_model: str
    aligner_model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the selected backend at startup (first run downloads weights).
    await run_in_threadpool(load_models)
    yield


app = FastAPI(
    title="Qwen3-ASR / Whisper Transcription API",
    description=(
        "Multilingual transcription with word-level timestamps on MLX. "
        "Select the backend via the ASR_BACKEND env var (qwen3 / whisper)."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# Allow calls from the browser tester (file://) and other LAN origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    return HealthOut(
        status="ok",
        backend=ASR_BACKEND,
        asr_model=ASR_MODEL_ID,
        aligner_model=ALIGNER_MODEL_ID,
    )


@app.post("/transcribe", response_model=TranscribeOut)
async def transcribe_endpoint(
    file: UploadFile = File(..., description="Audio file (wav/mp3/m4a/flac)"),
    language: str = Form("Japanese", description="Language name or ISO code"),
    context: Optional[str] = Form(None, description="Hotwords (Qwen3 backend only)"),
) -> TranscribeOut:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await file.read())

    try:
        result: TranscriptionResult = await run_in_threadpool(
            transcribe, tmp_path, language, context
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"transcription failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return TranscribeOut(
        text=result.text,
        language=result.language,
        words=[WordOut(**asdict(w)) for w in result.words],
        asr_seconds=result.asr_seconds,
        align_seconds=result.align_seconds,
    )
