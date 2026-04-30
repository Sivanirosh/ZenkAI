"""Capture HTTP surface (PIVOT_ROADMAP §B.6/§B.18).

- ``POST /api/v1/capture/image`` — multipart image + form fields.
  Persists the frame, runs the multimodal vision pipeline, returns
  the transcript + words + advice.

- ``GET /api/v1/capture/recent`` — used by the Capture screen and
  the demo to repaint history after a reload.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.capture import (
    CaptureRequest,
    list_recent_captures,
    process_capture,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/image")
async def capture_image(
    image: UploadFile = File(...),
    session_id: str = Form(...),
    surface_kind: str = Form(default="text"),
    target_competency_id: Optional[str] = Form(default=None),
    target_cefr: str = Form(default="B1"),
    note: str = Form(default=""),
) -> dict[str, Any]:
    image_bytes = await image.read()
    sid = (session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required")

    req = CaptureRequest(
        image_bytes=image_bytes,
        content_type=image.content_type or "application/octet-stream",
        session_id=sid,
        surface_kind=surface_kind or "text",
        target_competency_id=target_competency_id or None,
        target_cefr=target_cefr or "B1",
        note=note or "",
    )
    try:
        result = await process_capture(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.to_dict()


@router.get("/recent")
async def capture_recent(limit: int = 20) -> dict[str, Any]:
    if limit < 1 or limit > 200:
        raise HTTPException(
            status_code=400, detail="limit must be in [1, 200]"
        )
    rows = list_recent_captures(limit=limit)
    return {"count": len(rows), "captures": rows}
