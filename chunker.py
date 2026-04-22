"""Fixed-duration audio chunking (in-memory).

Splits a decoded audio array into equal-length float32 numpy views.
The context-manager interface mirrors ``vad_chunker.VADChunks`` so the
two can be swapped in ``api.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from audio_io import SAMPLE_RATE, decode_to_numpy


class AudioChunks:
    """Uniform-duration chunker. Context manager for API symmetry.

    Usage::

        with AudioChunks(src, chunk_duration=60.0) as ac:
            for offset_sec, chunk in ac.chunks:
                transcribe(chunk)

    ``src`` can be a path (decoded with ffmpeg on enter) or a pre-decoded
    ``np.ndarray`` at 16 kHz mono float32.
    """

    def __init__(
        self,
        src: str | Path | np.ndarray,
        chunk_duration: float = 60.0,
    ) -> None:
        self._src = src
        self.chunk_duration = float(chunk_duration)
        self._audio: np.ndarray | None = None
        self.chunks: List[Tuple[float, np.ndarray]] = []
        self.total_duration: float = 0.0

    def __enter__(self) -> "AudioChunks":
        if isinstance(self._src, np.ndarray):
            self._audio = self._src.astype(np.float32, copy=False)
        else:
            self._audio = decode_to_numpy(Path(self._src))
        self.total_duration = len(self._audio) / SAMPLE_RATE

        samples_per_chunk = int(self.chunk_duration * SAMPLE_RATE)
        min_tail_samples = int(0.1 * SAMPLE_RATE)
        offset = 0
        while offset + min_tail_samples < len(self._audio):
            end = min(offset + samples_per_chunk, len(self._audio))
            self.chunks.append((offset / SAMPLE_RATE, self._audio[offset:end]))
            offset += samples_per_chunk
        return self

    def __exit__(self, *exc) -> None:
        self.chunks = []
        self._audio = None
