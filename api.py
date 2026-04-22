"""Qwen3-ASR / Whisper REST API (synchronous + SSE streaming).

Run:
    uv run uvicorn api:app --host 0.0.0.0 --port 8000
    # LAN clients can reach the OpenAPI UI at http://<host-ip>:8000/docs

Select the backend with the ``ASR_BACKEND`` environment variable:
    ASR_BACKEND=qwen3    (default) Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B
    ASR_BACKEND=whisper  Whisper-large-v3-turbo (word timestamps built in)

Select the chunking strategy with the ``CHUNKER`` environment variable:
    CHUNKER=vad    (default) silero-vad speech-aware chunking (skips silence)
    CHUNKER=fixed  fixed-duration chunks (may hallucinate on silence)

Restrict where the ``path=`` parameter can point to with ``AUDIO_ROOT``:
    AUDIO_ROOT=/data/audio  only files inside this directory are accepted.

Endpoints:
    GET  /health
    POST /transcribe         (multipart: file OR path, language, context)
    POST /transcribe-stream  (multipart: file OR path, language, context,
                              chunk_duration)
                             Streams SSE events ("start", "chunk", "done",
                             "error") as each chunk finishes processing.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from audio_io import decode_to_numpy, resolve_audio_path
from chunker import AudioChunks
from transcriber import TranscriptionResult
from vad_chunker import VADChunks

ASR_BACKEND = os.environ.get("ASR_BACKEND", "qwen3").lower()
CHUNKER = os.environ.get("CHUNKER", "vad").lower()


def _make_chunker(audio, chunk_duration: float):
    """Pick the chunker based on the CHUNKER env var. ``audio`` may be a
    filesystem path or a pre-decoded 16 kHz mono float32 numpy array."""
    if CHUNKER == "vad":
        return VADChunks(audio, chunk_duration=chunk_duration)
    if CHUNKER == "fixed":
        return AudioChunks(audio, chunk_duration=chunk_duration)
    raise ValueError(f"Unknown CHUNKER={CHUNKER!r} (expected 'vad' or 'fixed')")


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
    chunker: str
    asr_model: str
    aligner_model: str
    audio_root: Optional[str] = None


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
    version="0.3.0",
    lifespan=lifespan,
)

# Allow calls from the browser tester (file://) and other LAN origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _resolve_audio_source(
    file: Optional[UploadFile],
    path: Optional[str],
) -> tuple[object, Callable[[], None]]:
    """Return (audio_for_chunker, cleanup_fn).

    - ``path`` mode: resolve against AUDIO_ROOT and decode to numpy here so
      the chunker does not have to re-do it. cleanup_fn is a no-op.
    - ``file`` mode: save upload to a temp file, return the path; the
      chunker will decode. cleanup_fn removes the temp file.
    """
    got_file = file is not None and (file.filename or "") != ""
    if path and got_file:
        raise HTTPException(400, "provide either `file` or `path`, not both")
    if path:
        try:
            resolved = resolve_audio_path(path)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except (PermissionError, IsADirectoryError) as e:
            raise HTTPException(400, str(e))
        audio_np = await run_in_threadpool(decode_to_numpy, resolved)
        return audio_np, lambda: None
    if got_file:
        suffix = Path(file.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(await file.read())
        return tmp_path, lambda: tmp_path.unlink(missing_ok=True)
    raise HTTPException(400, "must provide `file` (upload) or `path` (local path)")


_AUDIO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
}


@app.get("/audio")
async def get_audio(path: str):
    """Stream an audio file so the browser can play it alongside the
    transcript when the client uses ``path=`` instead of uploading. The
    path is validated against ``AUDIO_ROOT`` just like ``/transcribe``.
    ``FileResponse`` supports Range requests so ``<audio>`` seeking works.
    """
    try:
        resolved = resolve_audio_path(path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except (PermissionError, IsADirectoryError) as e:
        raise HTTPException(400, str(e))
    media_type = _AUDIO_MIME.get(resolved.suffix.lower(), "application/octet-stream")
    return FileResponse(resolved, media_type=media_type, filename=resolved.name)


@app.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    return HealthOut(
        status="ok",
        backend=ASR_BACKEND,
        chunker=CHUNKER,
        asr_model=ASR_MODEL_ID,
        aligner_model=ALIGNER_MODEL_ID,
        audio_root=os.environ.get("AUDIO_ROOT"),
    )


@app.post("/transcribe", response_model=TranscribeOut)
async def transcribe_endpoint(
    file: Optional[UploadFile] = File(None, description="Audio upload (wav/mp3/m4a/flac)"),
    path: Optional[str] = Form(None, description="Local file path on the server"),
    language: str = Form("Japanese", description="Language name or ISO code"),
    context: Optional[str] = Form(None, description="Hotwords (Qwen3 backend only)"),
) -> TranscribeOut:
    audio_src, cleanup = await _resolve_audio_source(file, path)
    try:
        result: TranscriptionResult = await run_in_threadpool(
            transcribe, audio_src, language, context
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"transcription failed: {exc}") from exc
    finally:
        cleanup()

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
    file: Optional[UploadFile] = File(None, description="Audio upload (wav/mp3/m4a/flac)"),
    path: Optional[str] = Form(None, description="Local file path on the server"),
    language: str = Form("Japanese"),
    context: Optional[str] = Form(None),
    chunk_duration: float = Form(60.0, description="Seconds per chunk"),
) -> StreamingResponse:
    audio_src, cleanup = await _resolve_audio_source(file, path)

    async def event_gen():
        ac = None
        try:
            ac = await run_in_threadpool(
                lambda: _make_chunker(audio_src, chunk_duration).__enter__()
            )
            yield _sse(
                "start",
                {
                    "total_duration": ac.total_duration,
                    "chunk_duration": chunk_duration,
                    "chunk_count": len(ac.chunks),
                    "chunker": CHUNKER,
                    "speech_coverage": getattr(ac, "speech_coverage", ac.total_duration),
                    "backend": ASR_BACKEND,
                    "asr_model": ASR_MODEL_ID,
                },
            )

            total_asr = 0.0
            total_align = 0.0
            total_words = 0
            accumulated_text = []

            for idx, (offset, chunk_array) in enumerate(ac.chunks):
                try:
                    result: TranscriptionResult = await run_in_threadpool(
                        transcribe, chunk_array, language, context
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

                chunk_end = min(offset + chunk_duration, ac.total_duration)
                yield _sse(
                    "chunk",
                    {
                        "chunk_index": idx,
                        "chunk_start": offset,
                        "chunk_duration": chunk_end - offset,
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
            cleanup()

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
