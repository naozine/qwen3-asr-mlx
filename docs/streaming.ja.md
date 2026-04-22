# ストリーミング転写の仕組み (`/transcribe-stream`)

**English**: [streaming.md](streaming.md)

`/transcribe-stream` は **Server-Sent Events (SSE)** で、チャンクごとの転写結果をリアルタイムに返します。長尺音声をクライアント切断に強く、メモリ定常で処理できるようにするための仕組みです。この文書では実装の各レイヤーを順に見ていきます。

## 全体フロー

```
Client                      Server                          Worker
  │                            │                                │
  ├─POST /transcribe-stream ──▶│                                │
  │   (path=...)               ├─ audio decode (ffmpeg pipe)    │
  │                            ├─ VAD 推論 (onnxruntime)        │
  │                            ├─ chunks = [..., ...]           │
  │◀── event: start ───────────┤                                │
  │                            ├─ run_in_threadpool ────────────▶ transcribe(ch0)
  │                            │   (async loop freed)            │   ...
  │                            │◀───────────────────────────────── ~5s後
  │◀── event: chunk ───────────┤                                │
  │                            ├─ run_in_threadpool ────────────▶ transcribe(ch1)
  │◀── event: chunk ───────────┤                                │
  │   ...                      │                                │
  │◀── event: done ────────────┤                                │
  │   (接続 close)              │                                │
```

## 1. 通信プロトコル: SSE

HTTPの**片方向ストリーム**。ごく普通のPOSTレスポンスが**閉じずに伸び続けるだけ**。ワイヤ上はテキスト、1フレーム = `event:` 行 + `data:` 行 + 空行。

```
event: start
data: {"total_duration": 180.0, "chunk_count": 6}

event: chunk
data: {"chunk_index": 0, "words": [...]}

event: chunk
data: {"chunk_index": 1, "words": [...]}

event: done
data: {"total_words": 231}

```

- Content-Type: `text/event-stream`
- ポーリング不要、サーバから push
- WebSocketより軽量、ただし単方向

## 2. サーバ側: FastAPI の async generator + StreamingResponse

```python
@app.post("/transcribe-stream")
async def transcribe_stream_endpoint(...):
    async def event_gen():
        yield _sse("start", {...})
        for idx, (offset, chunk) in enumerate(ac.chunks):
            result = await run_in_threadpool(transcribe, chunk, ...)
            yield _sse("chunk", {...})   # chunk 完了ごとに即 flush
        yield _sse("done", {...})

    return StreamingResponse(event_gen(), media_type="text/event-stream")
```

**async generator の挙動が肝**:

- `yield` で値を返すたび、関数は**中断**されて呼び出し元(Starlette)に制御が戻る
- Starlette はその値を HTTP ボディとして即 TCP ソケットに書き出す → クライアントへ届く
- 次の `next()` で関数は中断地点から再開

この性質だけで自然にストリームが作れます。

## 3. 重い処理を async ループから逃がす: `run_in_threadpool`

`transcribe()` は MLX で数秒〜10秒回る CPU/GPU 処理。これを普通に `await` で呼ぶと他のリクエストも詰まる:

```python
result = await run_in_threadpool(transcribe, chunk, language, context)
```

- 内部で `asyncio.to_thread` 相当、別スレッドで `transcribe()` を実行
- メインの async イベントループは**解放**されるので `/health` 等は詰まらない
- MLXモデルはスレッドセーフでない前提なので、`transcriber.py` 内で `threading.Lock` により実質シリアル化(並列リクエストが来ても安全)

## 4. `_sse()` ヘルパ

```python
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
```

- `event:` 行 + `data:` 行 + **空行** (`\n\n`) で 1 フレーム
- `ensure_ascii=False` で日本語をそのままテキストに

## 5. バッファリング回避のヘッダ

```python
headers={
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
```

これが無いとリバースプロキシ (nginx等) が数十KB溜めてから一気に流す、という**リアルタイム感が死ぬ**挙動になる。

## 6. 異常終了・クリーンアップ

```python
async def event_gen():
    ac = None
    try:
        ac = ... __enter__()
        yield ...
        for ...:
            try:
                result = await run_in_threadpool(transcribe, ...)
            except Exception as exc:
                yield _sse("error", {...})
                return
            yield _sse("chunk", {...})
        yield _sse("done", {...})
    except Exception as exc:
        yield _sse("error", {...})
    finally:
        if ac: ac.__exit__(...)
        cleanup()
```

クライアント途中切断 → `yield` が送信失敗 → `GeneratorExit` 発生 → `finally` で後片付け。アップロードtmpやVADチャンカーの一時領域も解放される。

## 7. クライアント側: `fetch` + ReadableStream で SSE を自前パース

`EventSource` API は便利だが **POST + multipart が送れない** (GETのみ) ので `fetch` を使う:

```javascript
const res = await fetch(url, { method: 'POST', body: fd });
const reader = res.body.getReader();
const decoder = new TextDecoder('utf-8');
let buffer = '';
while (true) {
  const { value, done } = await reader.read();  // 新バイト列が来たら戻る
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  let sep;
  while ((sep = buffer.indexOf('\n\n')) !== -1) {
    const frame = buffer.slice(0, sep);
    buffer = buffer.slice(sep + 2);
    // event: / data: を分解してイベント処理
  }
}
```

**ポイント**:
- `value` は `Uint8Array`、**フレーム途中のバイト列が届く可能性がある**
- なので **buffer に溜め、`\n\n` が見えたら切り出す**
- `TextDecoder({ stream: true })` は UTF-8 マルチバイト文字が途中で切れても適切に扱う
- `stream_client.py` の Python 側も同じロジックを使っている

## 8. この方式の性質

- **メモリ消費定常**: chunk 結果を即送ってから次 chunk へ進むので、サーバ側のバッファは必要最小限
- **クライアント切断に強い**: `GeneratorExit` 経由で自動でリソース解放
- **HTTP/1.1 で十分**: WebSocket や HTTP/2 push は不要
- **プロキシ/CDN 通過可**: `X-Accel-Buffering: no` さえ付けばOK
- **ポーリングより効率的**: 新しい chunk が出来るまでコネクションは idle、実処理のない HTTP 往復ゼロ

## 9. 関連ファイル

- サーバ: [`api.py`](../api.py) の `/transcribe-stream` と `_sse`
- チャンカー: [`vad_chunker.py`](../vad_chunker.py) / [`chunker.py`](../chunker.py)
- ブラウザクライアント: [`player.html`](../player.html) の `submitStreaming()` と `parseSSE()`
- CLI クライアント: [`stream_client.py`](../stream_client.py)
