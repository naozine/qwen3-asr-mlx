# Qwen3-ASR / Whisper × MLX Transcript Player

**English**: [README.md](README.md) | **日本語**: このページ

Apple Silicon 上の MLX で動く音声文字起こしサーバ。2つのバックエンドを切替可能、VADベースのストリーミング処理で長尺に対応。CLI / SSEストリーミング対応のFastAPIサーバ / ブラウザUIから使える。

- **Qwen3-ASR-1.7B (bf16)** + **Qwen3-ForcedAligner-0.6B** — 精度優先、ホットワード対応
- **Whisper large-v3-turbo (fp16)** — 高速なワンパス転写、word timestamps 内蔵
- どちらのバックエンドでも出力スキーマ・APIコントラクト・ブラウザUIは共通

<!-- ![demo](docs/demo.gif) -->

## Why

- **2つの強いバックエンド、共通スキーマ** — `ASR_BACKEND` で起動時に選ぶだけ、下流は一切変えずに切替可
- **VAD連動の長尺ストリーミング** — silero-vad で無音を除外、メモリ上でチャンク化、確定した chunk から順次 Server-Sent Events でクライアントへ返す。M4 MacBook で 2時間音声を ~10〜30分で処理
- **Apple Silicon ネイティブ** — 全面 MLX、PyTorch 依存なし。silero-vad は onnxruntime で実行
- **他サービスから呼ばれる設計** — アップロード不要の `path=` モード、`AUDIO_ROOT` ジェイル、`GET /audio` エンドポイントを備え、将来の音声管理ウェブアプリと組合せやすい
- **ローカル完結** — クラウド往復なし。プライベートな会議録や機密音声でも安心

## アーキテクチャ

```
 音声ファイル ──▶ ffmpeg pipe ──▶ numpy (in memory)
                                    │
                                    ▼
                           ┌──────────────────┐
                           │  CHUNKER=vad     │ (デフォルト, silero-vad / ONNX)
                           │  CHUNKER=fixed   │ (固定秒数チャンク)
                           └────────┬─────────┘
                                    │  話速部分だけのnumpyスライス
                ┌───────────────────┼───────────────────┐
                ▼                                       ▼
    ┌─────────────────────────────┐        ┌─────────────────────────────┐
    │ ASR_BACKEND=qwen3           │        │ ASR_BACKEND=whisper         │
    │   Qwen3-ASR-1.7B-bf16       │        │   whisper-large-v3-turbo    │
    │   Qwen3-ForcedAligner-0.6B  │        │   (word timestamps 内蔵)     │
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
  │  transcribe │            │  + SSE streaming│◀──────▶│  (CLIのSSE購読)  │
  └─────────────┘            └────────┬────────┘        └──────────────────┘
                                      ▼
                             ┌──────────────────┐
                             │  player.html     │
                             │  (タイムラインUI, │
                             │   ストリーム対応)  │
                             └──────────────────┘
```

## 使用モデル

| バックエンド | モデル | ライセンス | 備考 |
| --- | --- | --- | --- |
| `qwen3` | [mlx-community/Qwen3-ASR-1.7B-bf16](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16) | Apache-2.0 | 精度優先、多言語、ホットワード対応 |
| `qwen3` | [mlx-community/Qwen3-ForcedAligner-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit) | Apache-2.0 | 単語アライメント (1チャンク5分以内) |
| `whisper` | [mlx-community/whisper-large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo) | MIT | fp16、809Mパラメータ、word timestamps 内蔵 |
| VAD | [silero-vad](https://github.com/snakers4/silero-vad) (ONNX、~2MB) | MIT | 初回実行時に自動DL |

MLX移植版は [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) を利用。

## 動作要件

- Apple Silicon Mac (M1/M2/M3/M4 系)
- Python ≥3.12 (`uv` 推奨)
- `ffmpeg` (インストール済み+ PATH 通し)
- 初回DL: Qwen3で約 **4.8 GB**、Whisperで約 **1.6 GB** (`~/.cache/huggingface/`)

## Quick Start

```bash
git clone <this-repo> && cd qwen3-asr-mlx
uv sync
```

### 1. CLI

```bash
# Qwen3-ASR + ForcedAligner (精度優先、ホットワード注入可)
uv run python transcribe.py path/to/audio.wav --language Japanese
# → result.json

# Whisper large-v3-turbo (高速、word timestamps 内蔵)
uv run python transcribe_whisper.py path/to/audio.wav --language ja
# → result_whisper.json
```

### 2. REST API

```bash
# デフォルト: Qwen3-ASR + VAD
uv run uvicorn api:app --host 0.0.0.0 --port 8000

# Whisper + 固定チャンカー + AUDIO_ROOT ジェイル
ASR_BACKEND=whisper CHUNKER=fixed AUDIO_ROOT=$PWD \
    uv run uvicorn api:app --host 0.0.0.0 --port 8000

# 稼働状態の確認
curl http://localhost:8000/health
# {"status":"ok","backend":"qwen3","chunker":"vad","asr_model":"...","audio_root":"..."}
```

`http://localhost:8000/docs` で OpenAPI 画面。

#### 同期エンドポイント `/transcribe`

```bash
# アップロード方式
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" -F "language=Japanese"

# ローカルパス方式 (アップロードなし)
curl -X POST http://localhost:8000/transcribe \
  -F "path=subdir/audio.wav" -F "language=Japanese"
```

#### ストリーミング `/transcribe-stream` (長尺向け)

```bash
curl -N -X POST http://localhost:8000/transcribe-stream \
  -F "path=20260326.wav" -F "chunk_duration=60"
```

送信イベント: `start` (音声長・チャンク数) / `chunk` (chunk単位の転写+絶対時刻単語) / `done` (合計) / `error`

SSEを購読してマージ済みJSONを保存するヘルパCLI `stream_client.py` も同梱:

```bash
uv run python stream_client.py ./20260326.wav --chunk-duration 60
# → result_stream.json (stdoutには進捗)
```

#### 音声取得 `/audio`

```bash
curl "http://localhost:8000/audio?path=subdir/audio.wav" -o /tmp/a.wav
```

HTTP Range 対応なので `<audio>` のシーク付き再生に使える。`AUDIO_ROOT` の制約は `/transcribe` と同じ。

### 3. ブラウザUI (`player.html`)

`player.html` をダブルクリックで開く。3通りの使い方:

1. **JSON読み込み**: CLIで生成した `result*.json` と音声をドラッグ&ドロップ
2. **アップロード + 転写**: 音声をドロップ → 「Upload file」のまま「Send transcription request」
3. **パス + 転写**: 「Server path」に切替、パスを入力 (`AUDIO_ROOT` があれば相対パス可) → 送信。音声は `/audio?path=…` から自動取得され再生可

ストリームモード(デフォルトON)で VAD chunk 単位に文字が追記されていく。

#### UI機能
- 2ビュー: **流れ** (無音可視化+段落化) / **タイムライン** (絶対時間軸、赤い再生ヘッド)
- 話速に応じたフォントサイズ (早口は小さく)
- 単語クリックでシーク、再生中の単語ハイライト
- `Space` 再生/停止、`←`/`→` 5秒送り
- Server path 選択時に `/health` を参照、placeholderに `AUDIO_ROOT` ルールを反映

## 環境変数

| 変数 | デフォルト | 目的 |
| --- | --- | --- |
| `ASR_BACKEND` | `qwen3` | 転写モデル: `qwen3` or `whisper` |
| `CHUNKER` | `vad` | チャンク戦略: `vad` (silero-vad) or `fixed` (固定秒) |
| `AUDIO_ROOT` | *(未設定)* | 設定時、`path=` はこのディレクトリ配下に限定。相対パスはこの配下を起点に解決。 |

変更時はサーバ再起動が必要。

## バックエンドの選び方

| 指標 | `qwen3` | `whisper` |
| --- | --- | --- |
| モデルサイズ / 初回DL | ~4.8 GB | ~1.6 GB |
| メモリ消費 | ~5 GB | ~1.6 GB |
| M4 での速度感 | ~3〜4倍速 | ~6〜10倍速 |
| ホットワード / コンテキスト注入 | ✅ | ❌ |
| 長尺対応 (VAD併用) | ✅ (チャンクは aligner 制約内) | ✅ |
| 日本語の無音ハルシネーション | 少ない | 出やすい (VADで抑制) |
| 日本語など非英語の精度 | 非常に強い | 強い |

日本語用途の目安: 品質なら **qwen3 + VAD**、速度なら **whisper + VAD**。VAD だけでも Whisper の「ご視聴ありがとうございました」系の無音区間ハルシネーションは大幅に減る。

## API レスポンス例

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

## テスト

```bash
uv run pytest
```

`tests/test_api.py` に `/health` と `/transcribe` のバリデーション系スモークテスト。モデル層はモック化しておりCIでHugging Faceに触りません。`ASR_BACKEND` で選ばれたバックエンドに対して実行されます。

## 既知の制約

- **Qwen3-ForcedAligner は 5分/呼び出しまで**。VADチャンカーは制約内に収まるが、`CHUNKER=fixed` + Qwen3 + 長い `chunk_duration` の組合せは失敗する
- **`/transcribe` は同期のみ**。15分超の音声は `/transcribe-stream` を使う (同期だとHTTPタイムアウト)
- **CORS は `*`**。ローカル/LAN用途前提。外部公開時はオリジン制限を
- 並列転写リクエストは内部でシリアル化 (MLXモデルはスレッドセーフでない前提)
- `ASR_BACKEND` / `CHUNKER` / `AUDIO_ROOT` の切替はサーバ再起動が必要

## ロードマップ

- [ ] 非同期ジョブAPI (`/jobs/{id}/status`) で長尺処理のクライアント切断耐性
- [ ] 認証 (multi-tenant 対応、`X-API-Key` 等)
- [ ] 精度ベンチマーク (Qwen3 vs Whisper vs kotoba-whisper)
- [ ] GitHub Actions (lint / test)

## Thanks

- Alibaba Qwen team — Qwen3-ASR / Qwen3-ForcedAligner
- OpenAI — Whisper
- Silero Team — silero-vad
- Apple — MLX
- Prince Canuma ([@Blaizzy](https://github.com/Blaizzy)) — mlx-audio
- mlx-community — MLX量子化/bf16/fp16変換モデル

## License

MIT — [LICENSE](LICENSE) 参照。使用モデルは Qwen3系 Apache-2.0、Whisper MIT、silero-vad MIT で配布。利用時はそれぞれの条項にも従ってください。
