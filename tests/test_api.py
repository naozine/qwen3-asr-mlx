"""FastAPI エンドポイントのスモークテスト。

モデルのロード・実際の推論はモック化し、APIの形状だけ検証する。
本物のモデルを使う統合テストは時間とディスクを大量に消費するため対象外。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    import api
    import transcriber

    # モデルロードを no-op に (テストで HuggingFace からDLしない)
    monkeypatch.setattr(transcriber, "load_models", lambda: None)
    monkeypatch.setattr(api, "load_models", lambda: None)

    with TestClient(api.app) as c:
        yield c


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "Qwen3-ASR" in body["asr_model"]
    assert "ForcedAligner" in body["aligner_model"]


def test_transcribe_missing_file_returns_422(client):
    r = client.post("/transcribe", data={"language": "Japanese"})
    assert r.status_code == 422


def test_transcribe_returns_expected_shape(client, monkeypatch):
    import api
    from transcriber import TranscriptionResult, Word

    fake = TranscriptionResult(
        text="こんにちは世界",
        language="Japanese",
        words=[
            Word(text="こんにちは", start=0.0, end=0.5),
            Word(text="世界", start=0.5, end=0.9),
        ],
        asr_seconds=0.12,
        align_seconds=0.08,
    )
    monkeypatch.setattr(api, "transcribe", lambda path, lang, ctx: fake)

    files = {"file": ("sample.wav", b"RIFF....FAKE", "audio/wav")}
    r = client.post(
        "/transcribe",
        files=files,
        data={"language": "Japanese", "context": "世界"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "こんにちは世界"
    assert body["language"] == "Japanese"
    assert len(body["words"]) == 2
    assert body["words"][0] == {"text": "こんにちは", "start": 0.0, "end": 0.5}
    assert body["asr_seconds"] == pytest.approx(0.12)


def test_transcribe_propagates_server_error(client, monkeypatch):
    import api

    def raises(path, lang, ctx):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(api, "transcribe", raises)
    files = {"file": ("x.wav", b"\0", "audio/wav")}
    r = client.post("/transcribe", files=files, data={"language": "Japanese"})
    assert r.status_code == 500
    assert "synthetic failure" in r.json()["detail"]
