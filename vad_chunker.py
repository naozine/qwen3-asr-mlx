"""VAD-aware audio chunking using silero-vad (ONNX runtime).

Detects speech regions in the decoded audio array, then packs / splits them
into numpy chunks up to ``chunk_duration``. Silence regions are skipped
entirely, which keeps Whisper from hallucinating filler text on empty
audio and reduces wasted ASR time on long recordings.

Exposes the same interface as ``chunker.AudioChunks`` so ``api.py`` can
select between the two transparently.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from audio_io import SAMPLE_RATE, decode_to_numpy

_VAD_FRAME_SAMPLES = 512  # silero-vad v5 expects 512 samples per step at 16 kHz
_VAD_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
_VAD_MODEL_PATH = Path.home() / ".cache" / "silero-vad" / "silero_vad.onnx"


def _ensure_vad_model() -> Path:
    """Download the silero-vad ONNX model on first use."""
    if _VAD_MODEL_PATH.exists():
        return _VAD_MODEL_PATH
    _VAD_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    import requests  # transitively available via mlx-audio; uses certifi

    print(f"Downloading silero-vad to {_VAD_MODEL_PATH}...")
    with requests.get(_VAD_MODEL_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(_VAD_MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    return _VAD_MODEL_PATH


_session = None


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        _session = ort.InferenceSession(
            str(_ensure_vad_model()),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
    return _session


def _vad_probs(audio: np.ndarray) -> np.ndarray:
    """Return per-frame speech probability (one value per 512-sample frame).

    silero-vad v5 expects 512 samples of new audio prepended with a 64-sample
    context window from the previous frame; the state is carried forward.
    """
    session = _get_session()
    sr = np.array(SAMPLE_RATE, dtype=np.int64)
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, 64), dtype=np.float32)

    num_frames = len(audio) // _VAD_FRAME_SAMPLES
    probs = np.empty(num_frames, dtype=np.float32)
    for i in range(num_frames):
        frame = audio[i * _VAD_FRAME_SAMPLES : (i + 1) * _VAD_FRAME_SAMPLES]
        x = np.concatenate([context, frame[np.newaxis, :]], axis=1)
        prob, state = session.run(None, {"input": x, "sr": sr, "state": state})
        probs[i] = prob.item()
        context = x[:, -64:]
    return probs


def _detect_speech_segments(
    audio: np.ndarray,
    threshold: float = 0.5,
    neg_threshold: float = 0.35,
    min_speech_ms: int = 250,
    min_silence_ms: int = 300,
    speech_pad_ms: int = 150,
) -> List[Tuple[float, float]]:
    """Return list of (start_sec, end_sec) speech regions using hysteresis."""
    probs = _vad_probs(audio)
    frame_sec = _VAD_FRAME_SAMPLES / SAMPLE_RATE
    min_speech_frames = max(1, int(min_speech_ms / 1000 / frame_sec))
    min_silence_frames = max(1, int(min_silence_ms / 1000 / frame_sec))
    pad_sec = speech_pad_ms / 1000
    total_sec = len(audio) / SAMPLE_RATE

    segments_frames: List[Tuple[int, int]] = []
    in_speech = False
    seg_start = 0
    silence_run = 0

    for i, p in enumerate(probs):
        if not in_speech:
            if p >= threshold:
                in_speech = True
                seg_start = i
                silence_run = 0
        else:
            if p < neg_threshold:
                silence_run += 1
                if silence_run >= min_silence_frames:
                    seg_end = i - silence_run + 1
                    if seg_end - seg_start >= min_speech_frames:
                        segments_frames.append((seg_start, seg_end))
                    in_speech = False
                    silence_run = 0
            else:
                silence_run = 0

    if in_speech:
        seg_end = len(probs)
        if seg_end - seg_start >= min_speech_frames:
            segments_frames.append((seg_start, seg_end))

    result: List[Tuple[float, float]] = []
    for s, e in segments_frames:
        start_sec = max(0.0, s * frame_sec - pad_sec)
        end_sec = min(total_sec, e * frame_sec + pad_sec)
        result.append((start_sec, end_sec))
    return result


def _pack_chunks(
    segments: List[Tuple[float, float]],
    max_chunk_sec: float,
) -> List[Tuple[float, float]]:
    """Combine adjacent speech segments into chunks up to ``max_chunk_sec``.

    Splits any single segment longer than the cap into cap-sized pieces,
    then greedily merges consecutive pieces whenever the combined span
    still fits. Silence between merged pieces stays in the chunk, which
    is fine for both ASR backends and keeps timestamps continuous.
    """
    if not segments:
        return []

    split: List[Tuple[float, float]] = []
    for s, e in segments:
        while e - s > max_chunk_sec:
            split.append((s, s + max_chunk_sec))
            s += max_chunk_sec
        split.append((s, e))

    packed: List[Tuple[float, float]] = [split[0]]
    for s, e in split[1:]:
        ps, _ = packed[-1]
        if e - ps <= max_chunk_sec:
            packed[-1] = (ps, e)
        else:
            packed.append((s, e))
    return packed


class VADChunks:
    """VAD-aware chunker. Context manager for API symmetry with AudioChunks.

    ``chunks`` becomes a list of (absolute_start_sec, numpy_view) covering
    only speech regions of the source. ``total_duration`` is the full audio
    length so client-side progress bars still advance through silent gaps.
    ``src`` can be a path (decoded with ffmpeg on enter) or a pre-decoded
    ``np.ndarray`` at 16 kHz mono float32.
    """

    def __init__(
        self,
        src: str | Path | np.ndarray,
        chunk_duration: float = 60.0,
        min_silence_duration: float = 0.3,
    ) -> None:
        self._src = src
        self.chunk_duration = float(chunk_duration)
        self.min_silence_duration = float(min_silence_duration)
        self._audio: np.ndarray | None = None
        self.chunks: List[Tuple[float, np.ndarray]] = []
        self.total_duration: float = 0.0
        self.speech_coverage: float = 0.0

    def __enter__(self) -> "VADChunks":
        if isinstance(self._src, np.ndarray):
            self._audio = self._src.astype(np.float32, copy=False)
        else:
            self._audio = decode_to_numpy(Path(self._src))
        self.total_duration = len(self._audio) / SAMPLE_RATE

        segments = _detect_speech_segments(
            self._audio, min_silence_ms=int(self.min_silence_duration * 1000)
        )
        self.speech_coverage = sum(e - s for s, e in segments)
        chunk_ranges = _pack_chunks(segments, max_chunk_sec=self.chunk_duration)

        for start, end in chunk_ranges:
            s_idx = int(start * SAMPLE_RATE)
            e_idx = int(end * SAMPLE_RATE)
            self.chunks.append((start, self._audio[s_idx:e_idx]))
        return self

    def __exit__(self, *exc) -> None:
        self.chunks = []
        self._audio = None
