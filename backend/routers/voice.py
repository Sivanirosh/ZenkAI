"""Voice endpoints: /tts (Piper) and /stt (faster-whisper).

Both endpoints degrade gracefully: if the underlying binary/package is not
installed we return a 503 with an install hint so the frontend can fall back
(browser speechSynthesis for TTS; disable mic with hint for STT).
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from backend.models.pydantic_models import SttResponse, TtsRequest
from backend.services import stt_service, tts_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/tts")
async def tts(req: TtsRequest) -> StreamingResponse:
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    try:
        # Run once eagerly so we can surface PiperUnavailable as 503 before
        # headers are flushed. tts_service caches the result so the real
        # StreamingResponse below reads from the cache (no double synthesis).
        async for _ in tts_service.synthesise(text, speed=req.speed):
            break
    except tts_service.PiperUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"detail": exc.detail, "install_hint": exc.install_hint},
        ) from exc

    async def gen() -> AsyncIterator[bytes]:
        async for chunk in tts_service.synthesise(text, speed=req.speed):
            yield chunk

    return StreamingResponse(
        gen(),
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )


@router.post("/stt", response_model=SttResponse)
async def stt(audio: UploadFile = File(...)) -> SttResponse:
    content_type = audio.content_type or "application/octet-stream"
    audio_bytes = await audio.read()

    try:
        result = await stt_service.transcribe(audio_bytes, content_type)
    except stt_service.WhisperUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"detail": exc.detail, "install_hint": exc.install_hint},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SttResponse(
        transcript=result.transcript,
        confidence=result.confidence,
        language=result.language,
        duration_s=result.duration_s,
    )
