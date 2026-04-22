"""Audio decoding and path validation helpers.

Decoding goes through ffmpeg over stdout pipe, so we never materialize a
full decoded WAV on disk. Path validation optionally enforces an
``AUDIO_ROOT`` jail for the ``path=`` API mode.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def resolve_audio_path(raw_path: str) -> Path:
    """Resolve and sanity-check an externally supplied audio path.

    When ``AUDIO_ROOT`` is set, a **relative** path is resolved against it
    and **absolute** paths still must end up inside ``AUDIO_ROOT``. When
    ``AUDIO_ROOT`` is unset, the raw path is used as-is. Symlinks are
    followed before the containment check so they cannot be used to escape.
    """
    root_env = os.environ.get("AUDIO_ROOT")
    raw = Path(raw_path).expanduser()
    if root_env:
        root_p = Path(root_env).expanduser().resolve()
        p = (raw if raw.is_absolute() else (root_p / raw)).resolve(strict=False)
        try:
            p.relative_to(root_p)
        except ValueError as e:
            raise PermissionError(
                f"path outside AUDIO_ROOT={root_env}: {raw_path}"
            ) from e
    else:
        p = raw.resolve(strict=False)

    if not p.exists():
        raise FileNotFoundError(f"audio not found: {raw_path}")
    if not p.is_file():
        raise IsADirectoryError(f"not a file: {raw_path}")
    return p


def decode_to_numpy(src: str | Path) -> np.ndarray:
    """Decode any ffmpeg-supported format to 16 kHz mono float32 in memory."""
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error",
            "-i", str(src),
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-f", "s16le",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def decode_from_offset(src: str | Path, offset_sec: float) -> np.ndarray:
    """Decode from ``offset_sec`` to the current EOF.

    Uses ``-ignore_length 1`` so the decoder reads past a stale WAV header
    size if the recorder (e.g. Audio Hijack) has not updated it yet; this
    is essential for the ``follow=true`` streaming mode where the file is
    still being written.
    """
    if offset_sec <= 0:
        args_ss: list[str] = []
    else:
        args_ss = ["-ss", f"{offset_sec:.3f}"]
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error",
            "-ignore_length", "1",
            *args_ss,
            "-i", str(src),
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-f", "s16le",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0
