"""faster-whisper STT for Phase 2.

The Whisper model is loaded lazily on first use and cached for the lifetime
of the process (AGENT.md: never reload per request). Calling `transcribe`
before faster-whisper is installed raises `WhisperUnavailable`; the router
turns that into a 503 with an install hint so the frontend can disable the
mic gracefully.

Accepted inputs: WAV, WebM, OGG — up to 10 MB. Transcription runs on the
thread pool because faster-whisper's API is synchronous.

STT chain order (PIVOT_ROADMAP §B.5, revisited 2026-04-30):

The original B.5 design tried Gemma 4 audio multimodal first and used
faster-whisper as the fallback. In practice the Gemma 4 builds shipped
through Ollama today (e.g. ``gemma4:e4b``) have no audio encoder at all
— Ollama silently drops the ``audio`` field, the model answers the bare
text prompt, and politely refuses in German. That refusal then becomes
the "user transcript" because nothing in the chain can tell a real STT
output from a capability denial.

We therefore reverse the priority by default: faster-whisper first
(reliable, offline, cheap), Gemma audio only when explicitly opted in
via ``LINGUAMATE_GEMMA_AUDIO=1`` — re-enable when Google ships an
audio-capable Gemma 4 build through Ollama.
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
    engine: str = "whisper"


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


def _detect_device(configured: str) -> str:
    """Return the best available compute device.

    If the config says ``cpu`` we respect it.  Otherwise we try CUDA
    (via ``torch`` or ``ctranslate2``'s own CUDA check) and fall back to
    ``cpu`` silently so a missing CUDA driver never crashes startup.
    """
    if configured.lower() == "cpu":
        return "cpu"
    # Try torch first (available in many ML envs); ctranslate2 can also tell us.
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            logger.info("CUDA available — using GPU for Whisper inference.")
            return "cuda"
    except ImportError:
        pass
    try:
        import ctranslate2  # type: ignore
        if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
            logger.info("CUDA available (ctranslate2) — using GPU for Whisper.")
            return "cuda"
    except Exception:  # noqa: BLE001 — any failure means no CUDA
        pass
    logger.info(
        "CUDA not available (configured device=%s); falling back to CPU.",
        configured,
    )
    return "cpu"


def _load_model() -> object:
    """Lazy singleton loader for faster-whisper.

    Device is resolved at first load: ``WHISPER_DEVICE=auto`` (or any
    non-``cpu`` value) triggers CUDA auto-detection; ``cpu`` is used
    unconditionally when explicitly set.
    """
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
    device = _detect_device(settings.whisper_device)
    compute_type = "float16" if device == "cuda" else "int8"
    logger.info(
        "Loading Whisper model size=%s device=%s compute_type=%s",
        settings.whisper_model_size,
        device,
        compute_type,
    )
    _model_cache = WhisperModel(
        settings.whisper_model_size,
        device=device,
        compute_type=compute_type,
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
        engine="whisper",
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


# ─── Gemma-4-audio-first STT (B.5) ──────────────────────────────────────


async def _gemma_audio_transcribe(
    audio_bytes: bytes,
    content_type: str,
    *,
    language: str,
) -> Optional[STTResult]:
    """Try the active LLM provider's native audio-multimodal STT.

    Returns ``None`` when the provider does not support audio input or
    the call failed. We never raise — the caller falls back cleanly.
    """
    try:
        from backend.llm import get_provider  # noqa: WPS433
    except Exception:  # pragma: no cover — defensive
        return None

    provider = get_provider()
    audio_fn = getattr(provider, "transcribe_audio", None)
    if audio_fn is None:
        return None

    try:
        text = await audio_fn(
            audio_bytes,
            content_type=content_type,
            language=language,
            timeout=20.0,
        )
    except Exception as exc:
        logger.info("Gemma audio STT failed, falling back: %s", exc)
        return None
    if not text:
        return None
    cleaned = text.strip()
    return STTResult(
        transcript=cleaned,
        confidence=0.85,
        language=language,
        duration_s=0.0,
        engine="gemma_audio",
    )


def _gemma_audio_enabled() -> bool:
    """Opt-in switch for the Gemma 4 audio-multimodal STT path.

    Defaults to OFF because no public Gemma 4 build available through
    Ollama today actually has an audio encoder — calling it just makes the
    model politely refuse and that refusal then poisons the transcript
    pipeline. Set ``LINGUAMATE_GEMMA_AUDIO=1`` once a real audio-capable
    Gemma 4 variant is in your Ollama library.
    """
    raw = os.environ.get("LINGUAMATE_GEMMA_AUDIO", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


async def transcribe_with_fallback(
    audio_bytes: bytes,
    content_type: str,
    *,
    language: str = "de",
) -> STTResult:
    """STT chain (PIVOT_ROADMAP §B.5, revised default order):

    1. ``faster-whisper`` (default — works offline, deterministic, cheap).
    2. Gemma 4 audio-multimodal — only attempted when
       ``LINGUAMATE_GEMMA_AUDIO=1`` AND Whisper is unavailable.
    3. Empty-string stub with engine=``"unavailable"`` so the caller
       can present a graceful "I didn't catch that" UI without raising.
    """
    _validate(audio_bytes, content_type)

    try:
        return await transcribe(audio_bytes, content_type, language=language)
    except WhisperUnavailable as exc:
        logger.info("Whisper unavailable: %s", exc.detail)
    except ValueError as exc:
        logger.info("Whisper rejected input: %s", exc)
    except Exception as exc:  # pragma: no cover — last-resort safety net
        logger.warning("Whisper crashed: %s", exc)

    if _gemma_audio_enabled():
        gemma = await _gemma_audio_transcribe(
            audio_bytes, content_type, language=language
        )
        if gemma is not None:
            return gemma

    return STTResult(
        transcript="",
        confidence=0.0,
        language=language,
        duration_s=0.0,
        engine="unavailable",
    )
