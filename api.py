"""Qwen3-ASR + ForcedAligner REST API (synchronous).

Run:
    uv run uvicorn api:app --host 0.0.0.0 --port 8000
    # LAN clients can reach the OpenAPI UI at http://<host-ip>:8000/docs

Endpoints:
    GET  /health
    POST /transcribe  (multipart: file, language, context)
"""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transcriber import (
    ALIGNER_MODEL_ID,
    ASR_MODEL_ID,
    TranscriptionResult,
    load_models,
    transcribe,
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
    asr_model: str
    aligner_model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models at startup (first run also downloads weights, which is slow).
    await run_in_threadpool(load_models)
    yield


app = FastAPI(
    title="Qwen3-ASR Transcription API",
    description=(
        "Multilingual transcription with word-level timestamps, powered by "
        "Qwen3-ASR-1.7B (bf16) + Qwen3-ForcedAligner-0.6B (8bit) on MLX."
    ),
    version="0.1.0",
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
    return HealthOut(status="ok", asr_model=ASR_MODEL_ID, aligner_model=ALIGNER_MODEL_ID)


@app.post("/transcribe", response_model=TranscribeOut)
async def transcribe_endpoint(
    file: UploadFile = File(..., description="Audio file (wav/mp3/m4a/flac)"),
    language: str = Form("Japanese", description="ForcedAligner language name"),
    context: Optional[str] = Form(None, description="Hotwords / domain terms for ASR"),
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
