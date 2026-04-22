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
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

DEBUG_FOLLOW = os.environ.get("DEBUG_FOLLOW", "0") not in ("0", "", "false", "False")


def _dbg(msg: str) -> None:
    if DEBUG_FOLLOW:
        print(f"[follow] {msg}", file=sys.stderr, flush=True)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from audio_io import decode_from_offset, decode_range, decode_to_numpy, resolve_audio_path
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


async def _resolve_audio_path(
    file: Optional[UploadFile],
    path: Optional[str],
) -> tuple[Path, Callable[[], None]]:
    """Return (audio_path, cleanup_fn).

    - ``path`` mode: resolve against AUDIO_ROOT. ``cleanup`` is a no-op.
    - ``file`` mode: save the upload to a temp file. ``cleanup`` removes it.

    Both endpoints work off a concrete filesystem path — the streaming
    endpoint needs it for ``follow`` tail re-reads.
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
        return resolved, lambda: None
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
    audio_path, cleanup = await _resolve_audio_path(file, path)
    try:
        audio_np = await run_in_threadpool(decode_to_numpy, audio_path)
        result: TranscriptionResult = await run_in_threadpool(
            transcribe, audio_np, language, context
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
    start_offset: float = Form(0.0, description="Seconds into the audio to start from"),
    follow: bool = Form(
        False,
        description="If true, keep reading as the file grows until idle_timeout_seconds elapses without any size change.",
    ),
    idle_timeout_seconds: float = Form(
        15.0,
        description="In follow mode, finish after this many seconds without new decodable audio. Default is 15 because some recorders (e.g. Audio Hijack) commit header updates every ~10 seconds.",
    ),
    interim_threshold_seconds: float = Form(
        10.0,
        description="Chunks shorter than this are flagged interim=true (candidates for future revision). Only applied in follow mode.",
    ),
    consolidation_window_seconds: float = Form(
        60.0,
        description="In follow mode, once this many seconds of unconsolidated audio have been emitted, re-transcribe them as one window and emit final chunks whose `revises` field lists the chunk_ids they replace.",
    ),
) -> StreamingResponse:
    import asyncio

    audio_path, cleanup = await _resolve_audio_path(file, path)

    async def event_gen():
        import itertools

        chunk_id_counter = itertools.count(1)

        def next_chunk_id() -> str:
            return f"c_{next(chunk_id_counter):05d}"

        total_asr = 0.0
        total_align = 0.0
        total_words = 0
        accumulated_text: list[str] = []
        offset_sec = start_offset
        last_size = -1
        last_growth_ts = time.time()
        stopped_reason = "eof"

        # Consolidation state (follow mode only)
        recent_chunks: list[tuple[str, float, float]] = []  # (id, start, end)
        consolidation_start = start_offset

        async def process_audio(
            audio_np,
            base_offset: float,
            force_interim: Optional[bool],
            revises_for_first: list[str],
            chunker_chunk_duration: float,
        ):
            """Chunk + transcribe + emit. Yields (event_str, chunk_info, result)."""
            nonlocal total_asr, total_align, total_words
            ac_inner = await run_in_threadpool(
                lambda: _make_chunker(audio_np, chunker_chunk_duration).__enter__()
            )
            try:
                revises_pending = list(revises_for_first)
                for rel_offset, chunk_array in ac_inner.chunks:
                    abs_start = base_offset + rel_offset
                    try:
                        result: TranscriptionResult = await run_in_threadpool(
                            transcribe, chunk_array, language, context
                        )
                    except Exception as exc:
                        yield (
                            _sse(
                                "error",
                                {"chunk_start": abs_start, "message": str(exc)},
                            ),
                            None,
                            None,
                        )
                        return

                    chunk_id = next_chunk_id()
                    chunk_len = len(chunk_array) / 16000.0
                    if force_interim is not None:
                        is_interim = force_interim
                    else:
                        is_interim = follow and chunk_len < interim_threshold_seconds

                    words_out = [
                        {
                            "text": w.text,
                            "start": w.start + abs_start,
                            "end": w.end + abs_start,
                        }
                        for w in result.words
                    ]
                    accumulated_text.append(result.text)
                    total_asr += result.asr_seconds
                    total_align += result.align_seconds
                    total_words += len(words_out)

                    event = _sse(
                        "chunk",
                        {
                            "chunk_id": chunk_id,
                            "chunk_start": abs_start,
                            "chunk_duration": chunk_len,
                            "text": result.text,
                            "words": words_out,
                            "asr_seconds": result.asr_seconds,
                            "align_seconds": result.align_seconds,
                            "interim": is_interim,
                            "revises": revises_pending,
                        },
                    )
                    revises_pending = []
                    yield (
                        event,
                        (chunk_id, abs_start, abs_start + chunk_len, is_interim),
                        result,
                    )
            finally:
                await run_in_threadpool(
                    lambda: ac_inner.__exit__(None, None, None)
                )

        try:
            yield _sse(
                "start",
                {
                    "chunk_duration": chunk_duration,
                    "start_offset": start_offset,
                    "follow": follow,
                    "idle_timeout_seconds": idle_timeout_seconds,
                    "interim_threshold_seconds": interim_threshold_seconds,
                    "consolidation_window_seconds": consolidation_window_seconds,
                    "chunker": CHUNKER,
                    "backend": ASR_BACKEND,
                    "asr_model": ASR_MODEL_ID,
                },
            )

            loop_iter = 0
            while True:
                loop_iter += 1
                try:
                    current_size = audio_path.stat().st_size
                except FileNotFoundError:
                    yield _sse("error", {"message": f"file disappeared: {audio_path}"})
                    stopped_reason = "missing"
                    return

                size_changed = current_size != last_size
                _dbg(
                    f"iter={loop_iter} size={current_size} changed={size_changed} "
                    f"offset={offset_sec:.2f} cons_start={consolidation_start:.2f} "
                    f"recent={len(recent_chunks)} idle={time.time()-last_growth_ts:.1f}s"
                )
                last_size = current_size
                if size_changed:
                    audio_tail = await run_in_threadpool(
                        decode_from_offset, audio_path, offset_sec
                    )
                    _dbg(
                        f"  decode_from_offset({offset_sec:.2f}) -> "
                        f"{len(audio_tail)} samples ({len(audio_tail)/16000.0:.2f}s)"
                    )
                    if len(audio_tail) > 0:
                        last_growth_ts = time.time()
                        tail_duration = len(audio_tail) / 16000.0
                        errored = False
                        async for event, info, _result in process_audio(
                            audio_tail,
                            offset_sec,
                            None,  # auto-decide interim via threshold
                            [],
                            chunk_duration,
                        ):
                            yield event
                            if info is None:
                                errored = True
                                break
                            if follow:
                                recent_chunks.append(info[:3])
                        if errored:
                            stopped_reason = "chunk_error"
                            return
                        offset_sec += tail_duration
                        _dbg(f"  tail processed, offset -> {offset_sec:.2f}")

                        # Consolidation trigger: prefer a natural VAD silence
                        # boundary close to the target window length instead
                        # of slicing mid-phrase at exactly window_seconds.
                        if follow and recent_chunks:
                            accumulated = offset_sec - consolidation_start
                            target = consolidation_start + consolidation_window_seconds
                            lower = consolidation_start + consolidation_window_seconds * 0.5
                            upper = consolidation_start + consolidation_window_seconds * 1.5

                            best_end: Optional[float] = None
                            best_diff = float("inf")
                            if accumulated >= consolidation_window_seconds * 0.5:
                                for _, _c_start, c_end in recent_chunks:
                                    if lower <= c_end <= upper:
                                        diff = abs(c_end - target)
                                        if diff < best_diff:
                                            best_end = c_end
                                            best_diff = diff

                            # Force-cut safety net: if we overshot the window
                            # by 50% without finding a natural boundary, fall
                            # back to the last chunk end we have.
                            if best_end is None and accumulated >= consolidation_window_seconds * 1.5:
                                best_end = recent_chunks[-1][2]

                            _dbg(
                                f"  consolidation: accumulated={accumulated:.2f}s, "
                                f"target={consolidation_window_seconds:.0f}s, "
                                f"best_end={best_end}, will_fire={best_end is not None}"
                            )
                            if best_end is not None:
                                cons_start = consolidation_start
                                cons_duration = best_end - cons_start
                                cons_audio = await run_in_threadpool(
                                    decode_range, audio_path, cons_start, cons_duration
                                )
                                _dbg(
                                    f"  consolidate [{cons_start:.2f}, {best_end:.2f}] "
                                    f"({cons_duration:.2f}s) -> {len(cons_audio)} samples"
                                )
                                to_revise = [c for c in recent_chunks if c[2] <= best_end]
                                to_keep = [c for c in recent_chunks if c[2] > best_end]
                                if len(cons_audio) > 0:
                                    revises_ids = [c[0] for c in to_revise]
                                    cons_errored = False
                                    async for event, info, _result in process_audio(
                                        cons_audio,
                                        cons_start,
                                        False,  # consolidated = final
                                        revises_ids,
                                        consolidation_window_seconds,
                                    ):
                                        yield event
                                        if info is None:
                                            cons_errored = True
                                            break
                                    if cons_errored:
                                        stopped_reason = "chunk_error"
                                        return
                                recent_chunks = to_keep
                                consolidation_start = best_end
                                # Consolidation itself is real progress; refresh the
                                # idle-timeout clock so that a slow re-transcription
                                # (e.g. Qwen3 on a 200s catch-up window) doesn't
                                # trip idle_timeout the instant it finishes.
                                last_growth_ts = time.time()
                                _dbg(
                                    f"  consolidation done, cons_start -> {consolidation_start:.2f}, "
                                    f"recent kept={len(to_keep)}"
                                )

                if not follow:
                    break
                if time.time() - last_growth_ts > idle_timeout_seconds:
                    stopped_reason = "idle_timeout"
                    _dbg(
                        f"idle_timeout: {time.time()-last_growth_ts:.1f}s since "
                        f"last growth > {idle_timeout_seconds}s, exiting"
                    )
                    break
                try:
                    await asyncio.sleep(0.5)
                except asyncio.CancelledError:
                    stopped_reason = "cancelled"
                    raise

            yield _sse(
                "done",
                {
                    "reason": stopped_reason,
                    "end_offset": offset_sec,
                    "text": "".join(accumulated_text).strip(),
                    "total_words": total_words,
                    "total_asr_seconds": total_asr,
                    "total_align_seconds": total_align,
                },
            )
        except Exception as exc:
            yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
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
