# Qwen3-ASR / Whisper × MLX Transcript Player

**English**: [README.md](README.md) | **日本語**: このページ

Apple Silicon で 2 つの ASR バックエンドを使い分け、単語タイムスタンプ付きの転写をブラウザ上で可視化する最小構成。CLI / REST API / ブラウザUI の3経路から叩ける。

- **Qwen3-ASR-1.7B (bf16)** + **Qwen3-ForcedAligner-0.6B** — 精度優先、ホットワードで単語を効かせられる
- **Whisper large-v3-turbo (fp16)** — 高速なワンパス転写、word timestamps が本体に内蔵

サーバ起動時に `ASR_BACKEND` 環境変数で切替。

<!-- ![demo](docs/demo.gif) -->

## Why
- **2つの強いバックエンド、1つのスキーマ** — UI や CLI の出力形式を維持したまま Qwen3-ASR ↔ Whisper を差し替え可能
- **Apple Silicon ネイティブ**: MLX 経由で PyTorch 比 3〜4倍
- **単語タイムスタンプ**: Qwen3 は ForcedAligner、Whisper は自前の cross-attention。両方とも `{text, start, end}` の同じスキーマで返すので、UI は意識せず扱える
- **ローカル完結**: クラウド不要、音声は外に出ない

## アーキテクチャ

```
                 ┌─────────────────────────────┐
                 │ ASR_BACKEND=qwen3           │
  ┌──────────┐   │   Qwen3-ASR-1.7B-bf16 ──▶   │ 転写
  │  audio   │──▶│   Qwen3-ForcedAligner ──▶   │ 単語タイムスタンプ
  │ (wav/..) │   │                             │
  └──────────┘   │ ASR_BACKEND=whisper         │
                 │   whisper-large-v3-turbo ─▶ │ 転写 + 単語タイムスタンプ
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

## 使用モデル

| バックエンド | モデル | ライセンス | 備考 |
| --- | --- | --- | --- |
| `qwen3` | [mlx-community/Qwen3-ASR-1.7B-bf16](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16) | Apache-2.0 | 精度優先、多言語、ホットワード対応 |
| `qwen3` | [mlx-community/Qwen3-ForcedAligner-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit) | Apache-2.0 | 単語アライメント (5分以内) |
| `whisper` | [mlx-community/whisper-large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo) | MIT | fp16、809Mパラメータ、word timestamps 内蔵 |

MLX移植版は [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) を利用。

## 動作要件

- Apple Silicon Mac (M1/M2/M3/M4 系)
- Python ≥3.12 (`uv` 推奨)
- `ffmpeg` (WAV以外を扱う場合)
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
# デフォルト: Qwen3-ASR
uv run uvicorn api:app --host 0.0.0.0 --port 8000

# Whisper
ASR_BACKEND=whisper uv run uvicorn api:app --host 0.0.0.0 --port 8000

# 稼働中のバックエンド確認
curl http://localhost:8000/health
# {"status":"ok","backend":"qwen3","asr_model":"...","aligner_model":"..."}
```

`http://localhost:8000/docs` で OpenAPI 画面。

curl 例:
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "language=Japanese" \
  -F "context=固有名詞,業界用語"
```

### 3. ブラウザUI (`player.html`)

`player.html` をダブルクリックで開く。2通りの使い方:

- **JSON読み込み**: CLIで生成した `result*.json` と音声ファイルをドラッグ&ドロップ
- **APIで転写**: 音声ファイルをドロップ → 「APIで転写」ボタン (サーバ起動時に選んだバックエンドで処理される)

#### UI機能
- 2ビュー切替: **流れ** (無音可視化+段落化) / **タイムライン** (時間軸絶対配置)
- 話速に応じたフォントサイズ (早口は小さく)
- 単語クリックでシーク、再生中の単語ハイライト、再生ヘッド表示
- `Space` 再生/停止、`←`/`→` 5秒送り

## バックエンドの選び方

| 指標 | `qwen3` | `whisper` |
| --- | --- | --- |
| モデルサイズ / 初回DL | ~4.8 GB | ~1.6 GB |
| メモリ消費 | ~5 GB | ~1.6 GB |
| M4 での速度感 | 基準 | ~2〜3倍速 |
| ホットワード / コンテキスト注入 | ✅ | ❌ |
| 長尺 (5分超) の単語タイムスタンプ | チャンク分割が必要 (TODO) | 本体で対応 |
| 日本語など非英語の精度 | 非常に強い | 強い |

目安: **whisper** を速度・長尺用のデフォルトに、ドメイン固有の短い音声や固有名詞注入が必要なら **qwen3** へ切替。

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

モデルロード・推論はモック化した API スモークテスト (`tests/test_api.py`) を同梱。実モデルによる統合テストは対象外。`ASR_BACKEND` で選択したバックエンドに対して走ります。

## 既知の制約

- **Qwen3-ForcedAligner は 5分以内の音声**対象。長尺対応は自前でチャンク分割+連結が必要 (TODO)。Whisper バックエンドでは該当せず
- **同期API のみ**。15分を超える音声は HTTP タイムアウトに注意
- **CORS は `*` で開放**。ローカル/LAN用途前提。外部公開時はオリジン制限を
- 並列リクエストは内部でシリアル化される (MLXモデルはスレッドセーフでない前提)
- `ASR_BACKEND` の切替はサーバ再起動が必要 (hot-swap不可)

## ロードマップ

- [ ] Qwen3-ForcedAligner の 5分制約を吸収する長尺チャンクパイプライン
- [ ] 非同期ジョブAPI (`/jobs/{id}/status`)
- [ ] 精度ベンチマーク (Qwen3 vs Whisper vs kotoba-whisper)
- [ ] GitHub Actions (lint / test)

## Thanks

- Alibaba Qwen team — Qwen3-ASR / Qwen3-ForcedAligner
- OpenAI — Whisper
- Apple — MLX
- Prince Canuma ([@Blaizzy](https://github.com/Blaizzy)) — mlx-audio
- mlx-community — MLX量子化・bf16・fp16 変換モデルの提供

## License

MIT — [LICENSE](LICENSE) 参照。使用モデルは Qwen3 系 が Apache-2.0、Whisper が MIT で配布されています。利用時はそれぞれの条項にも従ってください。
