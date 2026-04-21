# Qwen3-ASR × MLX Transcript Player

**English**: [README.md](README.md) | **日本語**: このページ

Apple Silicon で **Qwen3-ASR-1.7B (bf16)** を動かし、単語タイムスタンプ付きの転写をブラウザ上で可視化する最小構成。CLI / REST API / ブラウザUI の3つのインターフェースから叩ける。

<!-- ![demo](docs/demo.gif) -->

## Why
- **精度優先**: Whisper large-v3 を多くのベンチで上回る Qwen3-ASR を bf16 で動かす (量子化なし)
- **Apple Silicon で完結**: MLX 経由で M系Mac のGPU/Neural Engineを活用。PyTorch比 3〜4倍の速度
- **時間情報**: 転写だけでは用途が限られるので、`Qwen3-ForcedAligner-0.6B` で単語単位のタイムスタンプを付与
- **ローカル完結**: クラウドAPIに投げない。プライベートな会話・会議録音をそのまま扱える

## アーキテクチャ

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
            │  CLI (JSON出力)  │     │  FastAPI server  │
            └─────────────────┘     └────────┬─────────┘
                                             ▼
                                    ┌──────────────────┐
                                    │  player.html     │
                                    │  (timeline UI)   │
                                    └──────────────────┘
```

## 使用モデル

| モデル | ライセンス | 用途 |
| --- | --- | --- |
| [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) → [mlx-community/Qwen3-ASR-1.7B-bf16](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16) | Apache-2.0 | 音声認識 (多言語) |
| [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) → [mlx-community/Qwen3-ForcedAligner-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit) | Apache-2.0 | 単語タイムスタンプ付与 |

MLX移植版は [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) を利用。

## 動作要件

- Apple Silicon Mac (M1/M2/M3/M4 系)
- Python ≥3.12 (`uv` 推奨)
- `ffmpeg` (WAV以外を扱う場合)
- 初回は約 **4.8 GB** のモデルダウンロードあり (`~/.cache/huggingface/`)

## Quick Start

```bash
git clone <this-repo> && cd <this-repo>
uv sync
```

### 1. CLI でワンショット

```bash
uv run python transcribe.py path/to/audio.wav --language Japanese
# → result.json に単語タイムスタンプ付きJSONを出力
```

### 2. REST API サーバ

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000
# http://localhost:8000/docs で OpenAPI を確認
```

curl 例:
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "language=Japanese" \
  -F "context=固有名詞,業界用語"
```

### 3. ブラウザUI (`player.html`)

`player.html` をダブルクリックで開く。選択肢:

- **JSON読み込み**: CLIで生成した `result.json` と音声ファイルをドラッグ&ドロップ
- **API経由で転写**: 音声ファイルをドロップ → 「APIで転写」ボタン

#### UI機能
- 2ビュー切替: **流れ** (無音可視化+段落化) / **タイムライン** (時間軸絶対配置)
- 話速に応じたフォントサイズ (早口は小さく)
- 単語クリックでシーク、再生中の単語ハイライト、再生ヘッド表示
- `Space` 再生/停止、`←`/`→` 5秒送り

## CLI オプション

| オプション | 説明 |
| --- | --- |
| `--language` | ForcedAligner の言語名 (`Japanese` / `English` / `Chinese` / `Korean` ほか) |
| `--context` | 固有名詞・専門用語を事前注入して精度向上 (ASR hotword) |
| `--output` | 出力JSONパス (default: `result.json`) |

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

## 既知の制約

- **ForcedAligner は 5分以内の音声**対象。長尺対応は自前でチャンク分割+連結が必要 (TODO)
- **同期API のみ**。15分を超える音声は HTTP タイムアウトに注意
- **CORS は `*` で開放**。ローカル/LAN用途前提。外部公開時はオリジン制限を
- 並列リクエストは内部でシリアル化される (MLXモデルはスレッドセーフでない前提)

## テスト

```bash
uv run pytest
```

モデルロード・推論はモック化した API スモークテスト (`tests/test_api.py`) を同梱。実モデルによる統合テストは対象外 (手動)。

## ロードマップ

- [ ] ForcedAligner 5分制約を吸収する長尺チャンクパイプライン
- [ ] 非同期ジョブAPI (`/jobs/{id}/status`)
- [ ] Whisper との精度比較スクリプト
- [ ] GitHub Actions (lint / test)

## Thanks

- Alibaba Qwen team — Qwen3-ASR / Qwen3-ForcedAligner
- Apple — MLX
- Prince Canuma ([@Blaizzy](https://github.com/Blaizzy)) — mlx-audio
- mlx-community — MLX量子化・bf16変換モデルの提供

## License

MIT — [LICENSE](LICENSE) 参照。なお使用するモデル (Qwen3-ASR / Qwen3-ForcedAligner) は Apache-2.0 ライセンスで配布されており、利用時はそちらの条項にも従ってください。
