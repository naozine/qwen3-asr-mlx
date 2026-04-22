# How `/transcribe-stream` works

**日本語**: [streaming.ja.md](streaming.ja.md)

`/transcribe-stream` returns per-chunk transcription results in real time via **Server-Sent Events (SSE)**. The design keeps memory usage flat, survives client disconnects cleanly, and lets long-audio workloads feel interactive.

## Overall flow

```
Client                      Server                          Worker
  │                            │                                │
  ├─POST /transcribe-stream ──▶│                                │
  │   (path=...)               ├─ audio decode (ffmpeg pipe)    │
  │                            ├─ VAD inference (onnxruntime)   │
  │                            ├─ chunks = [..., ...]           │
  │◀── event: start ───────────┤                                │
  │                            ├─ run_in_threadpool ────────────▶ transcribe(ch0)
  │                            │   (async loop freed)            │   ...
  │                            │◀───────────────────────────────── ~5s later
  │◀── event: chunk ───────────┤                                │
  │                            ├─ run_in_threadpool ────────────▶ transcribe(ch1)
  │◀── event: chunk ───────────┤                                │
  │   ...                      │                                │
  │◀── event: done ────────────┤                                │
  │   (connection closes)       │                                │
```

## 1. Wire protocol: SSE

SSE is a **one-way HTTP stream**: an ordinary POST response that simply never closes. On the wire it is plain text — each frame is an `event:` line, a `data:` line, then a blank line.

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
- Push-based, no polling
- Lighter than WebSocket, at the cost of being unidirectional

## 2. Server: FastAPI async generator + `StreamingResponse`

```python
@app.post("/transcribe-stream")
async def transcribe_stream_endpoint(...):
    async def event_gen():
        yield _sse("start", {...})
        for idx, (offset, chunk) in enumerate(ac.chunks):
            result = await run_in_threadpool(transcribe, chunk, ...)
            yield _sse("chunk", {...})   # flushed immediately per chunk
        yield _sse("done", {...})

    return StreamingResponse(event_gen(), media_type="text/event-stream")
```

The async-generator semantics do the heavy lifting:

- Every `yield` suspends the function and hands control back to Starlette.
- Starlette writes the yielded value straight to the TCP socket, so the client receives it right away.
- The next `next()` resumes the generator where it left off.

That is the whole streaming trick.

## 3. Keeping the event loop free: `run_in_threadpool`

`transcribe()` is a several-second CPU/GPU call through MLX. Calling it directly under `await` would stall every other request:

```python
result = await run_in_threadpool(transcribe, chunk, language, context)
```

- Internally equivalent to `asyncio.to_thread`: runs the blocking call on a worker thread.
- The main event loop is free, so `/health` (and other sessions) stay responsive.
- MLX models are not thread-safe, so `transcriber.py` guards them with `threading.Lock`. Even if parallel requests arrive, inference is serialized safely.

## 4. The `_sse()` helper

```python
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
```

- One frame = `event:` line + `data:` line + **blank line** (`\n\n`).
- `ensure_ascii=False` keeps Japanese (and any other non-ASCII text) human-readable on the wire.

## 5. Headers that prevent buffering

```python
headers={
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
```

Without these, reverse proxies (nginx and friends) may accumulate tens of kilobytes before flushing, killing the interactive feel.

## 6. Shutdown and cleanup

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

If the client disconnects mid-stream, the next `yield` raises `GeneratorExit`, which runs `finally`. Upload temp files and VAD scratch memory are released even in the error path.

## 7. Client: parse SSE manually with `fetch` + `ReadableStream`

The browser's `EventSource` API is convenient but **GET-only** — it can't send a multipart POST body. Using `fetch` instead:

```javascript
const res = await fetch(url, { method: 'POST', body: fd });
const reader = res.body.getReader();
const decoder = new TextDecoder('utf-8');
let buffer = '';
while (true) {
  const { value, done } = await reader.read();  // resolves when new bytes arrive
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  let sep;
  while ((sep = buffer.indexOf('\n\n')) !== -1) {
    const frame = buffer.slice(0, sep);
    buffer = buffer.slice(sep + 2);
    // split on event:/data: and dispatch
  }
}
```

Things to watch for:

- `value` is a `Uint8Array`; a frame can arrive split across multiple reads, so **buffer and slice on `\n\n`**.
- `TextDecoder({ stream: true })` handles UTF-8 multibyte characters that straddle chunk boundaries.
- `stream_client.py` uses the same logic on the Python side via `requests.iter_content`.

## 8. Properties of this design

- **Flat memory**: chunk results are flushed then discarded, so server-side buffers stay small.
- **Disconnect-safe**: `GeneratorExit` fires in `finally`, releasing resources automatically.
- **HTTP/1.1 is enough**: no WebSocket, no HTTP/2 push required.
- **Proxy-friendly**: works through any HTTP-aware middlebox given `X-Accel-Buffering: no`.
- **More efficient than polling**: no round-trips while the server is busy, and the connection stays idle between chunks.

## 9. Source locations

- Server endpoint and `_sse` helper: [`api.py`](../api.py)
- Chunkers: [`vad_chunker.py`](../vad_chunker.py), [`chunker.py`](../chunker.py)
- Browser client: `submitStreaming()` / `parseSSE()` in [`player.html`](../player.html)
- CLI client: [`stream_client.py`](../stream_client.py)
