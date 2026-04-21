"""Qwen3-ASR + ForcedAligner の REST API (同期版)。

起動:
    uv run uvicorn api:app --host 0.0.0.0 --port 8000
    # LAN内の他PCから http://<このMacのIP>:8000/docs で OpenAPI を確認可能

エンドポイント:
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
    # 起動時にモデルロード (初回はDL込みで時間がかかる)
    await run_in_threadpool(load_models)
    yield


app = FastAPI(
    title="Qwen3-ASR Transcription API",
    description="Qwen3-ASR-1.7B (bf16) + Qwen3-ForcedAligner-0.6B (8bit) で日本語含む多言語の転写+単語タイムスタンプを返す。",
    version="0.1.0",
    lifespan=lifespan,
)

# ブラウザテスター (file://) や LAN内ページからの呼び出しを許可
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
    file: UploadFile = File(..., description="音声ファイル (wav/mp3/m4a/flac)"),
    language: str = Form("Japanese", description="ForcedAligner 言語名"),
    context: Optional[str] = Form(None, description="ホットワード等のコンテキスト"),
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
        raise HTTPException(status_code=500, detail=f"転写失敗: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return TranscribeOut(
        text=result.text,
        language=result.language,
        words=[WordOut(**asdict(w)) for w in result.words],
        asr_seconds=result.asr_seconds,
        align_seconds=result.align_seconds,
    )
