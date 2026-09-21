"""Piper-based TTS for Phase 2.

`synthesise(text, speed)` spawns the Piper binary, feeds the text on stdin,
reads the complete WAV from stdout and yields the bytes in 4 KB chunks.
A 10s subprocess timeout guards against hanging renders. Results are
memoised so replaying the same sentence at the same speed is instant.

When Piper is not installed locally (no binary or no model file) we raise
`PiperUnavailable` — routers turn this into a 503 so the frontend can fall
back to the browser's `speechSynthesis`.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import AsyncIterator

from backend.config import get_settings

logger = logging.getLogger(__name__)

_WAV_CHUNK_SIZE = 4096
_PIPER_TIMEOUT_SECONDS = 10.0
_CACHE_CAPACITY = 10


class PiperUnavailable(RuntimeError):
    """Raised when the Piper binary or voice model is missing."""

    def __init__(self, detail: str, install_hint: str) -> None:
        super().__init__(detail)
        self.detail = detail
        self.install_hint = install_hint


# ponytail: plain dict with FIFO trim; LRU ordering is irrelevant at 10 entries
_cache: dict[tuple[str, float], bytes] = {}


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

    return stdout


async def synthesise(text: str, speed: float = 1.0) -> AsyncIterator[bytes]:
    """Yield WAV bytes in ~4 KB chunks. Memoised per (text, speed)."""
    text = (text or "").strip()
    if not text:
        return
    key = (text, round(float(speed), 2))
    if key not in _cache:
        _cache[key] = await _run_piper(text, speed)
        while len(_cache) > _CACHE_CAPACITY:
            _cache.pop(next(iter(_cache)))
    buf = io.BytesIO(_cache[key])
    while chunk := buf.read(_WAV_CHUNK_SIZE):
        yield chunk
