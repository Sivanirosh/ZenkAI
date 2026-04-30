"""Piper-based TTS for Phase 2.

`synthesise(text, speed)` spawns the Piper binary, feeds the text on stdin,
reads raw PCM from stdout, then wraps it in a WAV header and yields the bytes
in 4 KB chunks. A 10s subprocess timeout guards against hanging renders
(AGENT.md rule). Results are memoised in a small LRU so replaying the same
sentence at the same speed is instant.

When Piper is not installed locally (no binary or no model file) we raise
`PiperUnavailable` — routers turn this into a 503 so the frontend can fall
back to the browser's `speechSynthesis`.
"""

from __future__ import annotations

import asyncio
import io
import logging
import struct
from collections import OrderedDict
from typing import AsyncIterator, Tuple

from backend.config import get_settings

logger = logging.getLogger(__name__)

_WAV_CHUNK_SIZE = 4096
_DEFAULT_SAMPLE_RATE = 22050
_DEFAULT_CHANNELS = 1
_DEFAULT_SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM
_PIPER_TIMEOUT_SECONDS = 10.0
_LRU_CAPACITY = 10


class PiperUnavailable(RuntimeError):
    """Raised when the Piper binary or voice model is missing."""

    def __init__(self, detail: str, install_hint: str) -> None:
        super().__init__(detail)
        self.detail = detail
        self.install_hint = install_hint


def _wav_header(pcm_len: int, sample_rate: int) -> bytes:
    """Build a 44-byte RIFF/WAVE header for mono 16-bit PCM."""
    byte_rate = sample_rate * _DEFAULT_CHANNELS * _DEFAULT_SAMPLE_WIDTH_BYTES
    block_align = _DEFAULT_CHANNELS * _DEFAULT_SAMPLE_WIDTH_BYTES
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", 36 + pcm_len),
            b"WAVE",
            b"fmt ",
            struct.pack("<I", 16),                # fmt chunk size
            struct.pack("<H", 1),                 # PCM
            struct.pack("<H", _DEFAULT_CHANNELS),
            struct.pack("<I", sample_rate),
            struct.pack("<I", byte_rate),
            struct.pack("<H", block_align),
            struct.pack("<H", 8 * _DEFAULT_SAMPLE_WIDTH_BYTES),
            b"data",
            struct.pack("<I", pcm_len),
        ]
    )


_cache: "OrderedDict[Tuple[str, float], bytes]" = OrderedDict()


def _cache_get(key: Tuple[str, float]) -> bytes | None:
    val = _cache.get(key)
    if val is not None:
        _cache.move_to_end(key)
    return val


def _cache_put(key: Tuple[str, float], value: bytes) -> None:
    _cache[key] = value
    _cache.move_to_end(key)
    while len(_cache) > _LRU_CAPACITY:
        _cache.popitem(last=False)


def clear_cache() -> None:
    """Exposed for tests."""
    _cache.clear()


def is_available() -> bool:
    """Best-effort check used at startup to log availability."""
    settings = get_settings()
    return settings.piper_binary.exists() and settings.piper_model.exists()


async def _run_piper(text: str, speed: float) -> bytes:
    """Run Piper once and return the full WAV bytes."""
    settings = get_settings()

    if not settings.piper_binary.exists():
        raise PiperUnavailable(
            detail=f"Piper binary not found at {settings.piper_binary}",
            install_hint=(
                "Install Piper and set PIPER_BINARY in .env. "
                "See scripts/download_piper_model.sh."
            ),
        )
    if not settings.piper_model.exists():
        raise PiperUnavailable(
            detail=f"Piper voice model not found at {settings.piper_model}",
            install_hint=(
                "Download a German voice (e.g. de_DE-thorsten-high.onnx) and "
                "set PIPER_MODEL in .env."
            ),
        )

    length_scale = max(0.5, min(2.0, 1.0 / max(0.1, speed)))

    args = [
        str(settings.piper_binary),
        "--model",
        str(settings.piper_model),
        "--length_scale",
        f"{length_scale:.3f}",
        "--output_raw",
    ]

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=(text + "\n").encode("utf-8")),
            timeout=_PIPER_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise PiperUnavailable(
            detail="Piper subprocess timed out after 10s",
            install_hint="Try shorter text, or verify the Piper voice model.",
        ) from exc

    if proc.returncode != 0:
        logger.warning(
            "piper returned %s: %s",
            proc.returncode,
            (stderr or b"").decode("utf-8", errors="replace")[:500],
        )
        raise PiperUnavailable(
            detail=f"Piper failed (exit {proc.returncode})",
            install_hint=(
                "Check Piper logs; most issues are a bad voice model path or "
                "a missing espeak-ng dependency on Linux."
            ),
        )

    header = _wav_header(len(stdout), _DEFAULT_SAMPLE_RATE)
    return header + stdout


async def synthesise(text: str, speed: float = 1.0) -> AsyncIterator[bytes]:
    """Yield WAV bytes in ~4 KB chunks. Memoised per (text, speed)."""
    text = (text or "").strip()
    if not text:
        return
    speed = float(speed)
    key = (text, round(speed, 2))

    cached = _cache_get(key)
    if cached is not None:
        logger.debug("tts cache hit (%d bytes)", len(cached))
        buf = io.BytesIO(cached)
        while True:
            chunk = buf.read(_WAV_CHUNK_SIZE)
            if not chunk:
                return
            yield chunk
        return  # pragma: no cover

    wav = await _run_piper(text, speed)
    _cache_put(key, wav)

    buf = io.BytesIO(wav)
    while True:
        chunk = buf.read(_WAV_CHUNK_SIZE)
        if not chunk:
            return
        yield chunk
