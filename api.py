"""Qwen3-ASR / Whisper REST API (synchronous).

Run:
    uv run uvicorn api:app --host 0.0.0.0 --port 8000
    # LAN clients can reach the OpenAPI UI at http://<host-ip>:8000/docs

Select the backend with the ``ASR_BACKEND`` environment variable:
    ASR_BACKEND=qwen3    (default) Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B
    ASR_BACKEND=whisper  Whisper-large-v3-turbo (word timestamps built in)

Endpoints:
    GET  /health
    POST /transcribe         (multipart: file, language, context)
    POST /transcribe-stream  (multipart: file, language, context, chunk_duration)
                             Streams SSE events ("start", "chunk", "done", "error")
                             as each fixed-duration chunk finishes processing.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chunker import AudioChunks
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


def _sse(event: str, data: dict) -> str:
    """Format a single Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/transcribe-stream")
async def transcribe_stream_endpoint(
    file: UploadFile = File(..., description="Audio file (wav/mp3/m4a/flac)"),
    language: str = Form("Japanese"),
    context: Optional[str] = Form(None),
    chunk_duration: float = Form(60.0, description="Seconds per chunk"),
) -> StreamingResponse:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        upload_path = Path(tmp.name)
        tmp.write(await file.read())

    async def event_gen():
        ac: AudioChunks | None = None
        try:
            ac = await run_in_threadpool(
                lambda: AudioChunks(upload_path, chunk_duration=chunk_duration).__enter__()
            )
            yield _sse(
                "start",
                {
                    "total_duration": ac.total_duration,
                    "chunk_duration": chunk_duration,
                    "chunk_count": len(ac.chunks),
                    "backend": ASR_BACKEND,
                    "asr_model": ASR_MODEL_ID,
                },
            )

            total_asr = 0.0
            total_align = 0.0
            total_words = 0
            accumulated_text = []

            for idx, (offset, chunk_path) in enumerate(ac.chunks):
                try:
                    result: TranscriptionResult = await run_in_threadpool(
                        transcribe, chunk_path, language, context
                    )
                except Exception as exc:
                    yield _sse("error", {"chunk_index": idx, "message": str(exc)})
                    return

                words_out = [
                    {
                        "text": w.text,
                        "start": w.start + offset,
                        "end": w.end + offset,
                    }
                    for w in result.words
                ]
                accumulated_text.append(result.text)
                total_asr += result.asr_seconds
                total_align += result.align_seconds
                total_words += len(words_out)

                yield _sse(
                    "chunk",
                    {
                        "chunk_index": idx,
                        "chunk_start": offset,
                        "chunk_duration": min(chunk_duration, ac.total_duration - offset),
                        "text": result.text,
                        "words": words_out,
                        "asr_seconds": result.asr_seconds,
                        "align_seconds": result.align_seconds,
                    },
                )

            yield _sse(
                "done",
                {
                    "text": "".join(accumulated_text).strip(),
                    "total_words": total_words,
                    "total_asr_seconds": total_asr,
                    "total_align_seconds": total_align,
                },
            )
        except Exception as exc:
            yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            if ac is not None:
                await run_in_threadpool(lambda: ac.__exit__(None, None, None))
            upload_path.unlink(missing_ok=True)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
