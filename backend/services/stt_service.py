"""faster-whisper STT for Phase 2.

The Whisper model is loaded lazily on first use and cached for the lifetime
of the process (AGENT.md: never reload per request). Calling `transcribe`
before faster-whisper is installed raises `WhisperUnavailable`; the router
turns that into a 503 with an install hint so the frontend can disable the
mic gracefully.

Accepted inputs: WAV, WebM, OGG — up to 10 MB. Transcription runs on the
thread pool because faster-whisper's API is synchronous.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

from backend.config import get_settings

logger = logging.getLogger(__name__)

_MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB
_ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/webm",
    "audio/ogg",
    "audio/mpeg",
    "audio/mp4",
}


class WhisperUnavailable(RuntimeError):
    def __init__(self, detail: str, install_hint: str) -> None:
        super().__init__(detail)
        self.detail = detail
        self.install_hint = install_hint


@dataclass(frozen=True)
class STTResult:
    transcript: str
    confidence: float
    language: str
    duration_s: float


_model_cache: object | None = None


def _ext_for_content_type(content_type: str) -> str:
    ct = content_type.lower().split(";")[0].strip()
    mapping = {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/wave": ".wav",
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
    }
    return mapping.get(ct, ".bin")


def _load_model() -> object:
    """Lazy singleton loader for faster-whisper."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise WhisperUnavailable(
            detail="faster-whisper is not installed",
            install_hint=(
                "Install with: `pip install faster-whisper` "
                "(or `pip install -e .[voice]`)."
            ),
        ) from exc

    settings = get_settings()
    logger.info(
        "Loading Whisper model size=%s device=%s",
        settings.whisper_model_size,
        settings.whisper_device,
    )
    _model_cache = WhisperModel(
        settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type="int8",
    )
    return _model_cache


def is_available() -> bool:
    """Best-effort check used at startup (does NOT eager-load the model)."""
    try:
        import importlib

        importlib.import_module("faster_whisper")
        return True
    except ImportError:
        return False


def _validate(audio_bytes: bytes, content_type: str) -> None:
    if len(audio_bytes) == 0:
        raise ValueError("audio payload is empty")
    if len(audio_bytes) > _MAX_AUDIO_BYTES:
        raise ValueError(
            f"audio payload is {len(audio_bytes)} bytes (max {_MAX_AUDIO_BYTES})"
        )
    ct = content_type.lower().split(";")[0].strip()
    if ct not in _ALLOWED_CONTENT_TYPES:
        raise ValueError(
            f"unsupported content-type '{content_type}'; "
            f"accepted: {sorted(_ALLOWED_CONTENT_TYPES)}"
        )


def _run_sync(path: str, language: Optional[str]) -> STTResult:
    model = _load_model()
    segments, info = model.transcribe(  # type: ignore[attr-defined]
        path,
        language=language,
        beam_size=1,
        vad_filter=True,
    )
    transcript_parts: list[str] = []
    conf_sum = 0.0
    conf_count = 0
    for seg in segments:
        transcript_parts.append(seg.text.strip())
        if getattr(seg, "avg_logprob", None) is not None:
            conf_sum += float(seg.avg_logprob)
            conf_count += 1
    transcript = " ".join(p for p in transcript_parts if p).strip()
    avg_logprob = conf_sum / conf_count if conf_count else -1.0
    # Map avg_logprob (usually in [-1, 0]) to a pseudo-confidence in [0, 1].
    confidence = max(0.0, min(1.0, 1.0 + avg_logprob))
    return STTResult(
        transcript=transcript,
        confidence=round(confidence, 3),
        language=getattr(info, "language", "de") or "de",
        duration_s=float(getattr(info, "duration", 0.0) or 0.0),
    )


async def transcribe(
    audio_bytes: bytes,
    content_type: str,
    *,
    language: Optional[str] = "de",
) -> STTResult:
    """Validate and transcribe audio via faster-whisper."""
    _validate(audio_bytes, content_type)

    suffix = _ext_for_content_type(content_type)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.close()
        return await asyncio.to_thread(_run_sync, tmp.name, language)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
