# Qwen3-ASR × MLX Transcript Player

**English**: this page | **日本語**: [README.ja.md](README.ja.md)

A minimal local setup for running **Qwen3-ASR-1.7B (bf16)** on Apple Silicon via MLX, with word-level timestamps and a browser UI for playback — exposed through CLI, REST API, and a standalone HTML player.

<!-- ![demo](docs/demo.gif) -->

## Why

- **Accuracy first** — runs Qwen3-ASR (which beats Whisper large-v3 on many benchmarks) at full bf16, no quantization.
- **Apple Silicon native** — powered by MLX, roughly 3–4× faster than PyTorch on the same hardware.
- **Timing information** — word-level timestamps via `Qwen3-ForcedAligner-0.6B`, so you can build anything that needs to jump to specific parts of the audio (a Descript-style editor, study tools, subtitle workflows).
- **Fully local** — no cloud, no audio leaves your machine. Suitable for private meetings or sensitive recordings.

## Architecture

```
  ┌──────────┐        ┌────────────────────────┐
  │  audio   │──┐     │ mlx-audio (MLX)        │
  │ (wav/..) │  ├───▶ │  Qwen3-ASR-1.7B-bf16   │──▶ transcript
  └──────────┘  │     └────────────────────────┘
                │     ┌────────────────────────┐
                └───▶ │ Qwen3-ForcedAligner    │──▶ word timestamps
                      │  0.6B-8bit             │
                      └────────────────────────┘
                                 │
                     ┌───────────┴────────────┐
                     ▼                        ▼
            ┌─────────────────┐     ┌──────────────────┐
            │  CLI (JSON out) │     │  FastAPI server  │
            └─────────────────┘     └────────┬─────────┘
                                             ▼
                                    ┌──────────────────┐
                                    │  player.html     │
                                    │  (timeline UI)   │
                                    └──────────────────┘
```

## Models

| Model | License | Purpose |
| --- | --- | --- |
| [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) → [mlx-community/Qwen3-ASR-1.7B-bf16](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16) | Apache-2.0 | Multilingual speech recognition |
| [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) → [mlx-community/Qwen3-ForcedAligner-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit) | Apache-2.0 | Word-level forced alignment |

MLX ports courtesy of [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio).

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4 series)
- Python ≥3.12 (`uv` recommended)
- `ffmpeg` (only needed for non-WAV inputs)
- First run downloads roughly **4.8 GB** of weights to `~/.cache/huggingface/`.

## Quick Start

```bash
git clone <this-repo> && cd qwen3-asr-mlx
uv sync
```

### 1. CLI

```bash
uv run python transcribe.py path/to/audio.wav --language Japanese
# Writes result.json with the transcript + word timestamps.
```

### 2. REST API

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs for the interactive OpenAPI UI.
```

`curl` example:
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "language=Japanese" \
  -F "context=proper nouns, jargon"
```

### 3. Browser UI (`player.html`)

Open `player.html` directly. Two ways to feed it data:

- **Pre-generated JSON**: drag-and-drop both the audio file and a `result.json` produced by the CLI.
- **Transcribe on the fly**: drop the audio alone, then use the “API transcribe” panel to call the running server.

#### UI features
- Two views: **Flow** (silences visualized, paragraph-style) / **Timeline** (absolute time-based layout).
- Font size varies with speaking rate (faster speech = smaller type).
- Click any word to seek, active word highlights during playback, a red playhead sweeps across the timeline.
- Keyboard: `Space` play/pause, `←`/`→` skip ±5s.

## CLI Options

| Option | Description |
| --- | --- |
| `--language` | ForcedAligner language (`Japanese`, `English`, `Chinese`, `Korean`, etc.) |
| `--context` | Hotwords / domain terms injected into the ASR prompt for better accuracy |
| `--output` | Output JSON path (default: `result.json`) |

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

Ships with API smoke tests in `tests/test_api.py`. Model loading and inference are mocked — real-model integration tests are intentionally out of scope (too slow, too much disk).

## Known Limitations

- **ForcedAligner is limited to 5-minute audio.** Anything longer needs manual chunking and timestamp stitching (TODO).
- **Synchronous API only.** Requests longer than ~15 minutes may hit HTTP timeouts.
- **CORS is `*`**. Intended for local/LAN use; restrict origins before exposing publicly.
- Concurrent requests are serialized internally (MLX models are assumed non-thread-safe).

## Roadmap

- [ ] Long-audio pipeline that works around the 5-minute aligner limit
- [ ] Async job API (`/jobs/{id}/status`)
- [ ] Accuracy comparison scripts vs. Whisper
- [ ] GitHub Actions (lint / test)

## Thanks

- Alibaba Qwen team — Qwen3-ASR / Qwen3-ForcedAligner
- Apple — MLX
- Prince Canuma ([@Blaizzy](https://github.com/Blaizzy)) — mlx-audio
- mlx-community — MLX quantized / bf16 model conversions

## License

MIT — see [LICENSE](LICENSE). Note that the models themselves (Qwen3-ASR / Qwen3-ForcedAligner) are distributed under Apache-2.0; you must comply with their terms when using them.
