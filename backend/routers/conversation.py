"""Konversation HTTP surface (PIVOT_ROADMAP §B.5/§B.17).

Two routes:

- ``POST /api/v1/conversation/turn`` — multipart audio + form fields.
  Runs the speech-loop turn (STT → write event → LLM → write event)
  and returns both transcripts plus metadata. The frontend then
  fires ``/voice/tts`` against the assistant text to play it.

- ``GET /api/v1/conversation/recent`` — used by the demo + the
  Konversation screen to repaint history after a reload. Reads
  straight from the rotated ``transcripts.<NNNN>.jsonl`` files so we
  honour the §7.6 "filesystem is the source of truth" pattern.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from backend.conversation import (
    ConversationTurnRequest,
    list_recent_transcripts,
    run_turn,
    stream_turn,
)
from backend.multimodal import score_pronunciation

logger = logging.getLogger(__name__)
router = APIRouter()


_MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB — same as /voice/stt


@router.post("/turn")
async def conversation_turn(
    audio: UploadFile = File(...),
    session_id: str = Form(...),
    competency_id: Optional[str] = Form(default=None),
    scenario: Optional[str] = Form(default=None),
    target_cefr: Optional[str] = Form(default=None),
    language: str = Form(default="de"),
) -> dict[str, Any]:
    """Run one full Konversation turn end-to-end."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="audio payload is empty")
    if len(audio_bytes) > _MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"audio payload is {len(audio_bytes)} bytes "
                f"(max {_MAX_AUDIO_BYTES})"
            ),
        )
    sid = (session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required")

    req = ConversationTurnRequest(
        audio_bytes=audio_bytes,
        content_type=audio.content_type or "application/octet-stream",
        session_id=sid,
        competency_id=competency_id or None,
        scenario=scenario or None,
        target_cefr=target_cefr or None,
        language=language or "de",
    )
    result = await run_turn(req)
    return result.to_dict()


@router.post("/stream")
async def conversation_stream(
    audio: UploadFile = File(...),
    session_id: str = Form(...),
    competency_id: Optional[str] = Form(default=None),
    scenario: Optional[str] = Form(default=None),
    target_cefr: Optional[str] = Form(default=None),
    language: str = Form(default="de"),
) -> StreamingResponse:
    """Streaming SSE variant of /conversation/turn.

    Emits three event types so the frontend can show the user bubble
    immediately after STT and stream the assistant reply token-by-token:

        event: transcribed   — user transcript (show 'Du' bubble NOW)
        event: token         — {"delta": "..."} one LLM chunk at a time
        event: done          — full metadata + final assistant text (play TTS)
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="audio payload is empty")
    if len(audio_bytes) > _MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"audio payload is {len(audio_bytes)} bytes (max {_MAX_AUDIO_BYTES})",
        )
    sid = (session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required")

    req = ConversationTurnRequest(
        audio_bytes=audio_bytes,
        content_type=audio.content_type or "application/octet-stream",
        session_id=sid,
        competency_id=competency_id or None,
        scenario=scenario or None,
        target_cefr=target_cefr or None,
        language=language or "de",
    )
    return StreamingResponse(
        stream_turn(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/recent")
async def conversation_recent(limit: int = 20) -> dict[str, Any]:
    """Return the most recent transcripts (newest first)."""
    if limit < 1 or limit > 200:
        raise HTTPException(
            status_code=400, detail="limit must be in [1, 200]"
        )
    entries = list_recent_transcripts(limit=limit)
    return {
        "count": len(entries),
        "transcripts": [e.__dict__ for e in entries],
    }


@router.post("/pronounce")
async def conversation_pronounce(
    audio: UploadFile = File(...),
    reference_text: str = Form(...),
    target_cefr: Optional[str] = Form(default=None),
) -> dict[str, Any]:
    """Score the learner's pronunciation against a Piper reference (B.8)."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="audio payload is empty")
    if len(audio_bytes) > _MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"audio payload is {len(audio_bytes)} bytes "
                f"(max {_MAX_AUDIO_BYTES})"
            ),
        )
    text = (reference_text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="reference_text is required")

    try:
        score = await score_pronunciation(
            text,
            audio_bytes,
            content_type=audio.content_type or "audio/wav",
            target_cefr=target_cefr or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return score.to_dict()
