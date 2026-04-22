"""Fixed-duration audio chunking via ffmpeg.

Splits any ffmpeg-decodable input into equal-length mono 16 kHz WAV chunks
so long audio can be processed incrementally and results streamed back as
each chunk completes. Suitable for the MVP streaming pipeline; a VAD-based
chunker can later be dropped in with the same interface.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


SAMPLE_RATE = 16000


def probe_duration(path: str | Path) -> float:
    """Return audio duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _decode_to_wav(src: Path, dst: Path) -> None:
    """Decode (and resample) the input to a mono 16 kHz WAV."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(dst),
        ],
        capture_output=True,
        check=True,
    )


def _slice(src: Path, offset: float, duration: float, dst: Path) -> None:
    """Cut a chunk from a WAV using stream copy (no re-encode)."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{offset:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(src),
            "-c", "copy",
            str(dst),
        ],
        capture_output=True,
        check=True,
    )


class AudioChunks:
    """Context manager yielding fixed-duration WAV chunks for an audio file.

    Usage::

        with AudioChunks(path, chunk_duration=60.0) as ac:
            print(ac.total_duration)
            for offset_sec, chunk_path in ac.chunks:
                transcribe(chunk_path)

    Temporary files are cleaned up on exit.
    """

    def __init__(
        self,
        src: str | Path,
        chunk_duration: float = 60.0,
    ) -> None:
        self.src = Path(src)
        self.chunk_duration = float(chunk_duration)
        self._tmpdir: Path | None = None
        self.chunks: list[tuple[float, Path]] = []
        self.total_duration: float = 0.0

    def __enter__(self) -> "AudioChunks":
        self._tmpdir = Path(tempfile.mkdtemp(prefix="qwen_asr_chunks_"))
        wav = self._tmpdir / "source.wav"
        _decode_to_wav(self.src, wav)
        self.total_duration = probe_duration(wav)

        min_tail = 0.1  # skip chunks shorter than this (ffmpeg floating-point artifacts)
        offset = 0.0
        idx = 0
        while offset + min_tail < self.total_duration:
            chunk = self._tmpdir / f"chunk_{idx:04d}.wav"
            remaining = self.total_duration - offset
            duration = min(self.chunk_duration, remaining)
            _slice(wav, offset, duration, chunk)
            self.chunks.append((offset, chunk))
            offset += self.chunk_duration
            idx += 1

        wav.unlink(missing_ok=True)
        return self

    def __exit__(self, *exc) -> None:
        if self._tmpdir and self._tmpdir.exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = None
