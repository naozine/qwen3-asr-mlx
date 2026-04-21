# Qwen3-ASR / Whisper × MLX Transcript Player

**English**: this page | **日本語**: [README.ja.md](README.ja.md)

A minimal local setup for running state-of-the-art ASR on Apple Silicon via MLX, with word-level timestamps and a browser UI for playback — exposed through CLI, REST API, and a standalone HTML player. Two interchangeable backends:

- **Qwen3-ASR-1.7B (bf16)** + **Qwen3-ForcedAligner-0.6B** for accuracy-first, hotword-aware transcription
- **Whisper large-v3-turbo (fp16)** for fast single-pass transcription with built-in word timestamps

Switch at server startup via the `ASR_BACKEND` environment variable.

<!-- ![demo](docs/demo.gif) -->

## Why

- **Two strong backends, one schema** — swap Qwen3-ASR ↔ Whisper without touching the UI or CLI contract.
- **Apple Silicon native** — powered by MLX, roughly 3–4× faster than PyTorch on the same hardware.
- **Word-level timing** — Qwen3 via `Qwen3-ForcedAligner`, Whisper via its own cross-attention. Both produce the same `{text, start, end}` schema so you can click any word to jump to that moment in playback.
- **Fully local** — no cloud, no audio leaves your machine. Suitable for private meetings or sensitive recordings.

## Architecture

```
                 ┌─────────────────────────────┐
                 │ ASR_BACKEND=qwen3           │
  ┌──────────┐   │   Qwen3-ASR-1.7B-bf16 ──▶   │ transcript
  │  audio   │──▶│   Qwen3-ForcedAligner ──▶   │ word timestamps
  │ (wav/..) │   │                             │
  └──────────┘   │ ASR_BACKEND=whisper         │
                 │   whisper-large-v3-turbo ─▶ │ transcript + word timestamps
                 └──────────┬──────────────────┘
                            │
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
       ┌──────────────┐ ┌──────────┐ ┌────────────────┐
       │ CLI (Qwen3)  │ │ CLI      │ │ FastAPI server │
       │ transcribe.py│ │ (Whisper)│ └──────┬─────────┘
       └──────────────┘ └──────────┘        ▼
                                    ┌────────────────┐
                                    │  player.html   │
                                    │  (timeline UI) │
                                    └────────────────┘
```

## Models

| Backend | Model | License | Notes |
| --- | --- | --- | --- |
| `qwen3` | [mlx-community/Qwen3-ASR-1.7B-bf16](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16) | Apache-2.0 | Accuracy-first, multilingual, hotword context |
| `qwen3` | [mlx-community/Qwen3-ForcedAligner-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit) | Apache-2.0 | Word-level forced alignment (≤5 min) |
| `whisper` | [mlx-community/whisper-large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo) | MIT | fp16, 809M params, native word timestamps |

MLX ports courtesy of [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio).

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4 series)
- Python ≥3.12 (`uv` recommended)
- `ffmpeg` (only needed for non-WAV inputs)
- First run downloads roughly **4.8 GB** for Qwen3 or **1.6 GB** for Whisper to `~/.cache/huggingface/`.

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
# Default: Qwen3-ASR backend
uv run uvicorn api:app --host 0.0.0.0 --port 8000

# Whisper backend
ASR_BACKEND=whisper uv run uvicorn api:app --host 0.0.0.0 --port 8000

# Check which backend is active
curl http://localhost:8000/health
# {"status":"ok","backend":"qwen3","asr_model":"...","aligner_model":"..."}
```

Open http://localhost:8000/docs for the interactive OpenAPI UI.

`curl` example:
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "language=Japanese" \
  -F "context=proper nouns, jargon"
```

### 3. Browser UI (`player.html`)

Open `player.html` directly. Two ways to feed it data:

- **Pre-generated JSON**: drag-and-drop both the audio file and a `result*.json` produced by a CLI.
- **Transcribe on the fly**: drop the audio alone, then use the “API transcribe” panel to call the running server (whichever backend it was started with).

#### UI features
- Two views: **Flow** (silences visualized, paragraph-style) / **Timeline** (absolute time-based layout).
- Font size varies with speaking rate (faster speech = smaller type).
- Click any word to seek, active word highlights during playback, a red playhead sweeps across the timeline.
- Keyboard: `Space` play/pause, `←`/`→` skip ±5s.

## Choosing a Backend

| Criterion | `qwen3` | `whisper` |
| --- | --- | --- |
| Model size / first download | ~4.8 GB | ~1.6 GB |
| Memory footprint | ~5 GB | ~1.6 GB |
| Typical speed on M4 | baseline | ~2–3× faster |
| Hotword / context prompts | ✅ | ❌ |
| Long-audio (>5 min) word timestamps | requires chunking (TODO) | handled natively |
| Non-English accuracy (incl. Japanese) | very strong | strong |

Rule of thumb: start with **whisper** for speed and long audio; switch to **qwen3** when you need hotword prompts or the highest accuracy on short domain-specific clips.

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

Ships with API smoke tests in `tests/test_api.py`. Model loading and inference are mocked — real-model integration tests are intentionally out of scope (too slow, too much disk). Tests run against whichever backend `ASR_BACKEND` selects.

## Known Limitations

- **Qwen3-ForcedAligner is limited to 5-minute audio.** Anything longer needs manual chunking and timestamp stitching (TODO). Whisper backend is not affected.
- **Synchronous API only.** Requests longer than ~15 minutes may hit HTTP timeouts.
- **CORS is `*`**. Intended for local/LAN use; restrict origins before exposing publicly.
- Concurrent requests are serialized internally (MLX models are assumed non-thread-safe).
- Switching `ASR_BACKEND` requires a server restart — there is no hot-swap.

## Roadmap

- [ ] Long-audio pipeline that works around the 5-minute Qwen3 aligner limit
- [ ] Async job API (`/jobs/{id}/status`)
- [ ] Accuracy benchmark (Qwen3 vs. Whisper vs. kotoba-whisper) with reference data
- [ ] GitHub Actions (lint / test)

## Thanks

- Alibaba Qwen team — Qwen3-ASR / Qwen3-ForcedAligner
- OpenAI — Whisper
- Apple — MLX
- Prince Canuma ([@Blaizzy](https://github.com/Blaizzy)) — mlx-audio
- mlx-community — MLX quantized / bf16 / fp16 model conversions

## License

MIT — see [LICENSE](LICENSE). Note that the models themselves (Qwen3-ASR / Qwen3-ForcedAligner are Apache-2.0; Whisper is MIT); you must comply with each model's terms when using it.
