"""Pronunciation diff (PIVOT_ROADMAP §B.8).

``score_pronunciation(reference_text, learner_audio, ...)`` returns:

    PronunciationScore(
        overall: float,            # 0..1
        segments: list[Segment],   # per-word scores + hints
        reference_audio_path: str, # cached Piper WAV (sha256-named)
        critique: Optional[str],   # short German hint from Gemma 4 (best-effort)
        engine: str,               # "librosa" | "skeleton"
    )

Pipeline
--------

1. **Synthesise reference WAV** with Piper (existing
   :mod:`backend.services.tts_service`). Cached in
   ``data/memory/captures/audio_ref/<sha>.wav`` so the second attempt
   on the same sentence is instant.

2. **MFCC extraction** for both clips via
   ``librosa.feature.mfcc(n_mfcc=13)``. We resample to 22.05 kHz and
   normalise so volume doesn't dominate the cost.

3. **DTW alignment** with ``librosa.sequence.dtw``. We compute the
   per-window cosine distance along the warp path, then bucket distances
   to per-word scores by splitting both sequences into equal-length
   windows (a coarse but robust forced-alignment that doesn't require a
   phoneme model).

4. **Optional Gemma 4 critique** — ``provider.transcribe_audio`` with a
   prompt asking for a one-line German hint. Falls back to silence if
   the model lacks audio support.

5. **Skeleton fallback** — librosa missing OR audio shorter than 100 ms
   → return ``engine="skeleton"`` with an empty segments list. The route
   stays alive; the UI just shows "offline".

Reference WAV cache
-------------------

Path: ``data/memory/captures/audio_ref/<sha>.wav``
Hash: ``sha256(reference_text + voice_model_sha)`` so swapping the
Piper voice invalidates the cache. We never bypass tmp+rename — the
cache must never expose a half-written WAV.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_MIN_AUDIO_SECONDS = 0.1
_DEFAULT_SAMPLE_RATE = 22_050
_TARGET_HOP = 512
_N_MFCC = 13
_REFERENCE_DIR = "captures/audio_ref"
_AUDIO_CRITIQUE_TIMEOUT_S = 5.0


# ─── Public types ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class PronunciationSegment:
    word: str
    score: float           # 0..1
    hint: Optional[str] = None


@dataclass
class PronunciationScore:
    overall: Optional[float]
    segments: list[PronunciationSegment]
    reference_text: str
    reference_audio_path: Optional[str]
    critique: Optional[str]
    engine: str
    duration_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "segments": [
                {"word": s.word, "score": s.score, "hint": s.hint}
                for s in self.segments
            ],
            "reference_text": self.reference_text,
            "reference_audio_path": self.reference_audio_path,
            "critique": self.critique,
            "engine": self.engine,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


# ─── Public entry point ─────────────────────────────────────────────────


async def score_pronunciation(
    reference_text: str,
    learner_audio: bytes,
    *,
    content_type: str = "audio/wav",
    target_cefr: Optional[str] = None,
    provider: Any = None,
) -> PronunciationScore:
    """Score one learner utterance against a Piper reference."""
    import time

    started = time.monotonic()
    reference_text = (reference_text or "").strip()
    if not reference_text:
        raise ValueError("reference_text is required")
    if not learner_audio:
        raise ValueError("learner_audio payload is empty")

    # 1) Reference WAV
    try:
        ref_wav, ref_path = await _ensure_reference_wav(reference_text)
    except Exception as exc:
        logger.info("audio_capture: piper unavailable (%s)", exc)
        ref_wav, ref_path = b"", None

    # 2) Skeleton if librosa is missing
    librosa = _try_import_librosa()
    if librosa is None or not ref_wav:
        return _skeleton(reference_text, ref_path, started, reason="librosa-or-piper")

    # 3) Decode + MFCC + DTW
    try:
        learner_pcm, learner_sr = _decode_to_mono(learner_audio, librosa=librosa)
        ref_pcm, ref_sr = _decode_to_mono(ref_wav, librosa=librosa)
    except Exception as exc:
        logger.info("audio_capture: decode failed (%s)", exc)
        return _skeleton(reference_text, ref_path, started, reason=f"decode:{exc}")

    if (
        len(learner_pcm) < _MIN_AUDIO_SECONDS * learner_sr
        or len(ref_pcm) < _MIN_AUDIO_SECONDS * ref_sr
    ):
        return _skeleton(reference_text, ref_path, started, reason="too-short")

    overall, segments = _score_via_dtw(
        librosa,
        learner_pcm=learner_pcm,
        learner_sr=learner_sr,
        ref_pcm=ref_pcm,
        ref_sr=ref_sr,
        words=_words_for(reference_text),
    )

    # 4) Optional Gemma critique (best-effort)
    critique = await _maybe_audio_critique(
        learner_audio,
        content_type=content_type,
        reference_text=reference_text,
        target_cefr=target_cefr,
        provider=provider,
    )

    duration_ms = int((time.monotonic() - started) * 1000)
    return PronunciationScore(
        overall=overall,
        segments=segments,
        reference_text=reference_text,
        reference_audio_path=str(ref_path) if ref_path else None,
        critique=critique,
        engine="librosa",
        duration_ms=duration_ms,
        metadata={"sample_rate": learner_sr},
    )


# ─── Reference WAV cache ────────────────────────────────────────────────


async def _ensure_reference_wav(text: str) -> tuple[bytes, Path]:
    cache_dir = _audio_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    sha = hashlib.sha256(_voice_signature(text).encode("utf-8")).hexdigest()
    path = cache_dir / f"{sha}.wav"
    if path.exists():
        try:
            return path.read_bytes(), path
        except OSError:  # pragma: no cover
            pass

    wav = await _run_piper(text)
    if not wav:
        raise RuntimeError("piper returned empty audio")

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as f:
            f.write(wav)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:  # pragma: no cover
                pass
        os.replace(tmp, path)
    except OSError:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:  # pragma: no cover
            pass
        raise
    return wav, path


def _voice_signature(text: str) -> str:
    voice = ""
    try:
        from backend.config import get_settings  # noqa: WPS433

        settings = get_settings()
        voice = str(getattr(settings, "piper_model", "") or "")
    except Exception:  # pragma: no cover
        voice = ""
    return f"{voice}::{text}"


def _audio_cache_dir() -> Path:
    try:
        from backend.memory.manager import get_memory  # noqa: WPS433

        return Path(get_memory()._root) / _REFERENCE_DIR  # type: ignore[union-attr]
    except Exception:  # pragma: no cover
        return Path("/tmp") / _REFERENCE_DIR  # noqa: S108


async def _run_piper(text: str) -> bytes:
    try:
        from backend.services import tts_service  # noqa: WPS433

        chunks: list[bytes] = []
        async for chunk in tts_service.synthesise(text, speed=1.0):
            chunks.append(chunk)
        return b"".join(chunks)
    except Exception as exc:  # pragma: no cover — bubble up so caller fallback fires
        raise RuntimeError(f"piper synthesise failed: {exc}") from exc


# ─── librosa pipeline ───────────────────────────────────────────────────


def _try_import_librosa() -> Optional[Any]:
    try:
        import librosa  # type: ignore  # noqa: WPS433

        return librosa
    except ImportError:  # pragma: no cover — optional
        return None


def _decode_to_mono(payload: bytes, *, librosa: Any) -> tuple[Any, int]:
    """Return (mono float32 PCM, sample rate)."""
    import numpy as np  # noqa: WPS433

    # Fast path for plain WAV — avoids audioread shenanigans on CI boxes.
    if payload[:4] == b"RIFF":
        try:
            with wave.open(io.BytesIO(payload), "rb") as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                width = wf.getsampwidth()
                channels = wf.getnchannels()
            if width == 2:
                pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif width == 1:
                pcm = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                pcm = pcm.reshape(-1, channels).mean(axis=1)
            if sr != _DEFAULT_SAMPLE_RATE:
                pcm = librosa.resample(pcm, orig_sr=sr, target_sr=_DEFAULT_SAMPLE_RATE)
                sr = _DEFAULT_SAMPLE_RATE
            return pcm, sr
        except Exception as exc:  # pragma: no cover
            logger.info("audio_capture: wave decode failed (%s); using librosa.load", exc)

    # General path — let librosa handle it.
    pcm, sr = librosa.load(
        io.BytesIO(payload), sr=_DEFAULT_SAMPLE_RATE, mono=True, res_type="kaiser_fast"
    )
    return pcm, sr


def _score_via_dtw(
    librosa: Any,
    *,
    learner_pcm: Any,
    learner_sr: int,
    ref_pcm: Any,
    ref_sr: int,
    words: list[str],
) -> tuple[float, list[PronunciationSegment]]:
    import numpy as np  # noqa: WPS433

    learner_mfcc = librosa.feature.mfcc(
        y=learner_pcm, sr=learner_sr, n_mfcc=_N_MFCC, hop_length=_TARGET_HOP
    )
    ref_mfcc = librosa.feature.mfcc(
        y=ref_pcm, sr=ref_sr, n_mfcc=_N_MFCC, hop_length=_TARGET_HOP
    )

    # DTW expects features as columns
    D, wp = librosa.sequence.dtw(
        X=ref_mfcc, Y=learner_mfcc, metric="cosine", subseq=False
    )

    # Mean cosine distance along warp path
    distances: list[float] = []
    for ref_idx, learner_idx in wp:
        try:
            a = ref_mfcc[:, int(ref_idx)]
            b = learner_mfcc[:, int(learner_idx)]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0.0:
                continue
            distances.append(1.0 - float(np.dot(a, b) / denom))
        except (IndexError, ValueError):  # pragma: no cover
            continue
    if not distances:
        return 0.0, []

    avg = float(np.mean(distances))
    overall = max(0.0, min(1.0, 1.0 - avg))

    # Per-word bucket
    segments: list[PronunciationSegment] = []
    if words:
        n_words = len(words)
        per_word = max(1, len(distances) // n_words)
        for i, word in enumerate(words):
            start = i * per_word
            end = (i + 1) * per_word if i < n_words - 1 else len(distances)
            slice_ = distances[start:end]
            if not slice_:
                slice_ = [avg]
            seg_score = max(0.0, min(1.0, 1.0 - float(np.mean(slice_))))
            hint = _hint_for(seg_score)
            segments.append(
                PronunciationSegment(word=word, score=seg_score, hint=hint)
            )
    return overall, segments


def _words_for(text: str) -> list[str]:
    return re.findall(r"[\wäöüÄÖÜß]+", text)


def _hint_for(score: float) -> Optional[str]:
    if score >= 0.85:
        return None
    if score >= 0.65:
        return "fast da — etwas klarer artikulieren"
    if score >= 0.4:
        return "Silben langsamer aussprechen"
    return "diesen Teil noch einmal üben"


# ─── Optional Gemma critique ────────────────────────────────────────────


async def _maybe_audio_critique(
    learner_audio: bytes,
    *,
    content_type: str,
    reference_text: str,
    target_cefr: Optional[str],
    provider: Any,
) -> Optional[str]:
    if provider is None:
        try:
            from backend.llm import get_provider as _get_provider  # noqa: WPS433

            provider = _get_provider()
        except Exception:  # pragma: no cover
            return None
    transcribe = getattr(provider, "transcribe_audio", None)
    if transcribe is None:
        return None
    prompt = (
        "Du bist Mira, eine deutsche Aussprachetrainerin. "
        f"Der Lernende sollte sagen: „{reference_text}\u201c. "
        f"Hörniveau: {target_cefr or 'B1'}. "
        "Schreibe EINEN kurzen, freundlichen Satz mit konkretem Hinweis. "
        "Keine Wiederholung des Referenzsatzes."
    )
    try:
        critique = await asyncio.wait_for(
            transcribe(
                audio_bytes=learner_audio,
                content_type=content_type,
                prompt=prompt,
                language="de",
            ),
            timeout=_AUDIO_CRITIQUE_TIMEOUT_S,
        )
    except (asyncio.TimeoutError, TypeError):
        return None
    except Exception as exc:  # pragma: no cover
        logger.debug("audio_capture critique failed: %s", exc)
        return None
    if not critique:
        return None
    return str(critique).strip().splitlines()[0][:200]


# ─── Skeleton fallback ──────────────────────────────────────────────────


def _skeleton(
    reference_text: str,
    ref_path: Optional[Path],
    started: float,
    *,
    reason: str,
) -> PronunciationScore:
    import time

    duration_ms = int((time.monotonic() - started) * 1000)
    return PronunciationScore(
        overall=None,
        segments=[],
        reference_text=reference_text,
        reference_audio_path=str(ref_path) if ref_path else None,
        critique="Aussprache-Bewertung offline.",
        engine="skeleton",
        duration_ms=duration_ms,
        metadata={"reason": reason},
    )


__all__ = ["PronunciationScore", "PronunciationSegment", "score_pronunciation"]
