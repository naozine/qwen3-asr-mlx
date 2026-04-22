"""Smoke tests for the FastAPI endpoints.

Model loading and inference are mocked — we only validate the API shape.
Real-model integration tests are intentionally out of scope: they consume
significant time and disk (weights download).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    import api
    import transcriber

    # No-op model loading so tests never hit the Hugging Face Hub.
    monkeypatch.setattr(transcriber, "load_models", lambda: None)
    monkeypatch.setattr(api, "load_models", lambda: None)

    with TestClient(api.app) as c:
        yield c


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"] in {"qwen3", "whisper"}
    assert body["asr_model"]  # non-empty


def test_transcribe_missing_audio_returns_400(client):
    r = client.post("/transcribe", data={"language": "Japanese"})
    assert r.status_code == 400
    assert "file" in r.json()["detail"] or "path" in r.json()["detail"]


def test_transcribe_rejects_both_file_and_path(client):
    files = {"file": ("x.wav", b"\0", "audio/wav")}
    r = client.post(
        "/transcribe",
        files=files,
        data={"language": "Japanese", "path": "/tmp/other.wav"},
    )
    assert r.status_code == 400


def test_transcribe_path_not_found_returns_404(client):
    r = client.post(
        "/transcribe",
        data={"language": "Japanese", "path": "/nonexistent/path/__does_not_exist__.wav"},
    )
    assert r.status_code == 404


def test_health_reports_chunker_and_backend(client):
    body = client.get("/health").json()
    assert body["chunker"] in {"vad", "fixed"}
    assert body["backend"] in {"qwen3", "whisper"}


def test_transcribe_returns_expected_shape(client, monkeypatch):
    import api
    import numpy as np
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
    monkeypatch.setattr(api, "decode_to_numpy", lambda p: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(api, "transcribe", lambda audio, lang, ctx: fake)

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
    import numpy as np

    def raises(audio, lang, ctx):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(api, "decode_to_numpy", lambda p: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(api, "transcribe", raises)
    files = {"file": ("x.wav", b"\0", "audio/wav")}
    r = client.post("/transcribe", files=files, data={"language": "Japanese"})
    assert r.status_code == 500
    assert "synthetic failure" in r.json()["detail"]
