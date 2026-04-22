# Qwen3-ASR / Whisper × MLX Transcript Player

**English**: this page | **日本語**: [README.ja.md](README.ja.md)

A local speech-to-text server on Apple Silicon, built around MLX with two interchangeable model backends and a VAD-based streaming pipeline. Exposes a CLI, a FastAPI server with Server-Sent Events streaming, and a standalone HTML player for playback and visual inspection.

- **Qwen3-ASR-1.7B (bf16)** + **Qwen3-ForcedAligner-0.6B** for accuracy-first, hotword-aware transcription.
- **Whisper large-v3-turbo (fp16)** for fast single-pass transcription with built-in word timestamps.
- Both backends share one JSON schema, one API contract, and the same browser UI.

<!-- ![demo](docs/demo.gif) -->

## Why

- **Two strong backends, one schema** — pick Qwen3 or Whisper at server startup (`ASR_BACKEND`); everything downstream stays unchanged.
- **VAD-aware long-audio streaming** — silero-vad trims silence, the audio is chunked in memory, and each chunk's transcript is pushed to the client over Server-Sent Events as it finishes. A 2-hour recording streams back in ~10–30 minutes on an M4 MacBook.
- **Apple Silicon native** — MLX throughout, no PyTorch. silero-vad runs via onnxruntime.
- **Designed to be called** — an upload-less `path=` mode plus an `AUDIO_ROOT` jail and `GET /audio` endpoint make this server easy to pair with a separate audio-management web app.
- **Fully local** — no cloud round-trips. Suitable for private meetings and sensitive recordings.

## Architecture

```
 audio file ──▶ ffmpeg pipe ──▶ numpy (in memory)
                                    │
                                    ▼
                           ┌──────────────────┐
                           │  CHUNKER=vad     │ (default, silero-vad / ONNX)
                           │  CHUNKER=fixed   │ (uniform chunks)
                           └────────┬─────────┘
                                    │  speech-only numpy slices
                ┌───────────────────┼───────────────────┐
                ▼                                       ▼
    ┌─────────────────────────────┐        ┌─────────────────────────────┐
    │ ASR_BACKEND=qwen3           │        │ ASR_BACKEND=whisper         │
    │   Qwen3-ASR-1.7B-bf16       │        │   whisper-large-v3-turbo    │
    │   Qwen3-ForcedAligner-0.6B  │        │   (word timestamps built in)│
    └──────────────┬──────────────┘        └──────────────┬──────────────┘
                   │                                      │
                   └──────────────────┬───────────────────┘
                                      ▼
                       { text, language, words:[{text,start,end}], ... }
                                      │
         ┌────────────────────────────┼───────────────────────────┐
         ▼                            ▼                           ▼
  ┌─────────────┐            ┌─────────────────┐        ┌──────────────────┐
  │  CLI        │            │  FastAPI        │        │  stream_client   │
  │  transcribe │            │  + SSE streaming│◀──────▶│  (CLI, SSE sink) │
  └─────────────┘            └────────┬────────┘        └──────────────────┘
                                      ▼
                             ┌──────────────────┐
                             │  player.html     │
                             │  (timeline UI,   │
                             │   streaming in)  │
                             └──────────────────┘
```

## Models

| Backend | Model | License | Notes |
| --- | --- | --- | --- |
| `qwen3` | [mlx-community/Qwen3-ASR-1.7B-bf16](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16) | Apache-2.0 | Accuracy-first, multilingual, hotword context |
| `qwen3` | [mlx-community/Qwen3-ForcedAligner-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit) | Apache-2.0 | Word-level forced alignment (≤5 min per chunk) |
| `whisper` | [mlx-community/whisper-large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo) | MIT | fp16, 809M params, native word timestamps |
| VAD | [silero-vad](https://github.com/snakers4/silero-vad) (ONNX, ~2 MB) | MIT | Auto-downloaded on first use |

MLX ports courtesy of [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio).

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4 series)
- Python ≥3.12 (`uv` recommended)
- `ffmpeg` (installed and on `PATH`)
- First run downloads roughly **4.8 GB** for Qwen3 or **1.6 GB** for Whisper into `~/.cache/huggingface/`.

## Quick Start

```bash
git clone <this-repo> && cd qwen3-asr-mlx
uv sync
```

### 1. CLI

```bash
# Qwen3-ASR + ForcedAligner (accuracy-first, hotword prompts)
uv run python transcribe.py path/to/audio.wav --language Japanese
# -> result.json

# Whisper large-v3-turbo (fast, native word timestamps)
uv run python transcribe_whisper.py path/to/audio.wav --language ja
# -> result_whisper.json
```

### 2. REST API

```bash
# Default: Qwen3-ASR backend with VAD chunking
uv run uvicorn api:app --host 0.0.0.0 --port 8000

# Whisper backend, uniform (non-VAD) chunks, with an audio jail
ASR_BACKEND=whisper CHUNKER=fixed AUDIO_ROOT=$PWD \
    uv run uvicorn api:app --host 0.0.0.0 --port 8000

# Check active configuration
curl http://localhost:8000/health
# {"status":"ok","backend":"qwen3","chunker":"vad","asr_model":"...","audio_root":"..."}
```

Open http://localhost:8000/docs for the interactive OpenAPI UI.

#### Synchronous transcription

```bash
# Via upload
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" -F "language=Japanese"

# Via local path (no upload)
curl -X POST http://localhost:8000/transcribe \
  -F "path=subdir/audio.wav" -F "language=Japanese"
```

#### Streaming (SSE) for long audio

```bash
curl -N -X POST http://localhost:8000/transcribe-stream \
  -F "path=20260326.wav" -F "chunk_duration=60"
```

Events emitted: `start` (duration, chunk count), `chunk` (per-chunk text + absolute-time words), `done` (totals), `error`.

A helper CLI consumes this stream and writes a merged JSON compatible with `player.html`:

```bash
uv run python stream_client.py ./20260326.wav --chunk-duration 60
# -> result_stream.json (+ live progress on stdout)
```

#### Audio retrieval

```bash
curl "http://localhost:8000/audio?path=subdir/audio.wav" -o /tmp/a.wav
```

Supports HTTP Range so `<audio>` seeking works. Same `AUDIO_ROOT` rules apply.

### 3. Browser UI (`player.html`)

Open `player.html` directly. Three ways to feed it data:

1. **Pre-generated JSON**: drag-and-drop both the audio file and a `result*.json` produced by a CLI.
2. **Upload + transcribe**: drop the audio alone, keep "Upload file" selected, and click **Send transcription request**.
3. **Path + transcribe**: switch to "Server path", type the path (relative to `AUDIO_ROOT` if set), and submit. The browser pulls the audio from `/audio?path=…` so playback works without uploading.

Stream mode (on by default) appends text chunk-by-chunk as the server processes each VAD segment.

#### UI features
- Two views: **Flow** (silences visualized, paragraph-style) / **Timeline** (absolute time-based layout, red playhead).
- Font size varies with speaking rate (faster speech = smaller type).
- Click any word to seek; the active word highlights during playback.
- Keyboard: `Space` play/pause, `←`/`→` skip ±5s.
- `/health` is polled when entering path mode so the placeholder reflects the server's `AUDIO_ROOT`.

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `ASR_BACKEND` | `qwen3` | Transcription model: `qwen3` or `whisper` |
| `CHUNKER` | `vad` | Chunking strategy: `vad` (silero-vad) or `fixed` (uniform) |
| `AUDIO_ROOT` | *(unset)* | When set, `path=` must resolve inside this directory; relative paths are resolved against it. |

Switching any of these requires a server restart.

## Choosing a Backend

| Criterion | `qwen3` | `whisper` |
| --- | --- | --- |
| Model size / first download | ~4.8 GB | ~1.6 GB |
| Memory footprint | ~5 GB | ~1.6 GB |
| Typical speed on M4 | ~3–4× realtime | ~6–10× realtime |
| Hotword / context prompts | ✅ | ❌ |
| Long audio (with VAD chunking) | ✅ (chunks stay ≤ aligner limit) | ✅ |
| Japanese hallucinations on silence | rare | common (mitigated by VAD) |
| Non-English accuracy | very strong | strong |

Rule of thumb for Japanese: **qwen3 + VAD** for quality, **whisper + VAD** when speed matters. VAD alone already eliminates most of Whisper's idle-silence artifacts ("ご視聴ありがとうございました" etc.).

## API Response Example

```json
{
  "text": "今日はいい天気ですね。",
  "language": "Japanese",
  "words": [
    {"text": "今日", "start": 0.12, "end": 0.45},
    {"text": "は",   "start": 0.45, "end": 0.58}
  ],
  "asr_seconds": 3.2,
  "align_seconds": 1.1
}
```

## Tests

```bash
uv run pytest
```

Smoke tests in `tests/test_api.py` cover `/health` and the validation paths of `/transcribe` with the model layer mocked, so CI never touches Hugging Face. Tests exercise whichever backend `ASR_BACKEND` selects.

## Known Limitations

- **Qwen3-ForcedAligner is limited to 5-minute audio per call.** The VAD chunker keeps chunks comfortably below this, but disabling VAD (`CHUNKER=fixed`) with long chunk durations on the Qwen3 backend will fail.
- **Synchronous `/transcribe` endpoint only.** For long audio use `/transcribe-stream`; the synchronous endpoint can hit HTTP timeouts on audio over ~15 minutes.
- **CORS is `*`**. Intended for local/LAN use; restrict origins before exposing publicly.
- Concurrent transcription requests are serialized internally (MLX models are assumed non-thread-safe).
- Switching `ASR_BACKEND`, `CHUNKER`, or `AUDIO_ROOT` requires a server restart — there is no hot-swap.

## Roadmap

- [ ] Async job API (`/jobs/{id}/status`) so long transcriptions survive client disconnects
- [ ] Authentication for multi-tenant deployment (`X-API-Key` or similar)
- [ ] Accuracy benchmark (Qwen3 vs. Whisper vs. kotoba-whisper) with reference data
- [ ] GitHub Actions (lint / test)

## Thanks

- Alibaba Qwen team — Qwen3-ASR / Qwen3-ForcedAligner
- OpenAI — Whisper
- Silero Team — silero-vad
- Apple — MLX
- Prince Canuma ([@Blaizzy](https://github.com/Blaizzy)) — mlx-audio
- mlx-community — MLX-converted model weights

## License

MIT — see [LICENSE](LICENSE). Note the third-party models carry their own licenses (Qwen3-ASR / Qwen3-ForcedAligner: Apache-2.0; Whisper: MIT; silero-vad: MIT); comply with each when using them.
