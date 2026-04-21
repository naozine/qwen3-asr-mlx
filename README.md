# qwen3-asr-test

Apple Silicon (M4) 上で **Qwen3-ASR-1.7B (bf16)** + **Qwen3-ForcedAligner-0.6B (8bit)** を MLX で動かし、転写と単語タイムスタンプを取得する最小構成。

## 構成

| 用途 | モデル | サイズ |
| --- | --- | --- |
| 転写 (精度優先) | `mlx-community/Qwen3-ASR-1.7B-bf16` | 約 4.08 GB |
| タイムスタンプ | `mlx-community/Qwen3-ForcedAligner-0.6B-8bit` | 約 0.7 GB |

## セットアップ

```bash
uv sync
# ffmpeg は既にインストール済み想定 (WAV以外を扱う場合に必要)
```

## 使い方

```bash
uv run python transcribe.py sample.wav --language Japanese
```

オプション:

- `--language`: `Japanese` / `English` / `Chinese` / `Korean` など (ForcedAligner 用)
- `--context`: 固有名詞・専門用語などのホットワードを渡すと ASR の精度が上がる
- `--output`: JSON 出力先 (デフォルト `result.json`)

## 出力例

```json
{
  "text": "今日はいい天気ですね。",
  "language": "ja",
  "words": [
    {"text": "今日", "start": 0.12, "end": 0.45},
    {"text": "は",   "start": 0.45, "end": 0.58}
  ]
}
```

## 初回実行時の注意

- モデル初回ロード時に Hugging Face から約 4.8 GB をダウンロード (`~/.cache/huggingface/`)
- ForcedAligner は最大 **5 分** の音声まで。長尺は事前にチャンク分割推奨
- ストリーミング転写はタイムスタンプ非対応

## 音声フォーマット

- WAV (16kHz モノラル推奨) が最速
- MP3 等は ffmpeg でデコードされる

## REST API (同期版)

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000
# http://localhost:8000/docs で OpenAPI ドキュメントを確認
```

curl例:

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@test.wav" \
  -F "language=Japanese" \
  -F "context=Claude,MLX,Qwen"
```

- 起動時にモデルをロード (初回はDLで数分)
- MLXモデルは内部でシリアル化 (複数リクエストは順次処理)
- 長尺 (15分超) はタイムアウトに注意。本格的な長尺処理は将来の非同期ジョブ版で対応予定
