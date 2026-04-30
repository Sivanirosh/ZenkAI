"""Agent SSE endpoint — POST /api/v1/agent/turn.

Receives a small JSON description of the learner's situation and streams
back the agent's reasoning + tool dispatches as Server-Sent Events. The
event protocol is documented in PIVOT_ROADMAP.md §6.3 (Outcome example):

    event: thought          data: "<one chunk of free-text>"
    event: tool_intent      data: {"name":..., "args":...}
    event: tool_result      data: {<tool-specific JSON>}
    event: tool_error       data: {<{tool, code, message, detail}>}
    event: error            data: {"message": "...", ...}
    event: done             data: {"turn_ms": 890, "session_id": "..."}

The endpoint never raises after the first byte hits the wire — failures
become ``error`` events followed by exactly one ``done`` event so the
frontend can always finalise its UI state.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.agent.atlas import build_atlas_payload
from backend.agent.atrium import build_atrium_payload
from backend.agent.controller import AgentEvent, AgentEventKind, TurnRequest, turn

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/atrium")
async def get_atrium() -> dict[str, Any]:
    """Return the data the Atrium home screen needs in one round-trip.

    No LLM call, no SSE — this must be cheap enough to render on first
    paint. PIVOT_ROADMAP §8.2.
    """
    return build_atrium_payload().to_dict()


@router.get("/atlas")
async def get_atlas(goal_id: Optional[str] = None) -> dict[str, Any]:
    """Return the Atlas literal-map payload for the active goal.

    404 when no goal/plan exists yet (the Onboarding flow hasn't run).
    PIVOT_ROADMAP §B.4 + §8.3.
    """
    payload = build_atlas_payload(goal_id=goal_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="no_active_plan")
    return payload.to_dict()


class TurnPayload(BaseModel):
    """Request body for ``POST /agent/turn``.

    All fields are optional; the controller picks sensible defaults
    (current_screen='reader', no time budget, no current paragraph).
    """

    current_screen: str = Field(default="reader", min_length=1, max_length=32)
    time_budget_min: Optional[int] = Field(default=None, ge=1, le=480)
    current_paragraph_id: Optional[str] = None
    work_id: Optional[str] = None
    user_message: Optional[str] = Field(default=None, max_length=4000)
    session_id: Optional[str] = Field(default=None, max_length=64)
    goal_text: Optional[str] = Field(default=None, max_length=2000)
    extras: dict[str, Any] = Field(default_factory=dict)


@router.post("/turn")
async def post_turn(payload: TurnPayload) -> StreamingResponse:
    """Stream one agent turn as SSE."""
    request = TurnRequest(
        current_screen=payload.current_screen,
        time_budget_min=payload.time_budget_min,
        current_paragraph_id=payload.current_paragraph_id,
        work_id=payload.work_id,
        user_message=payload.user_message,
        session_id=payload.session_id,
        goal_text=payload.goal_text,
        extras=dict(payload.extras),
    )

    async def event_stream() -> AsyncIterator[bytes]:
        """Wrap controller.turn so a top-level error still closes the SSE cleanly.

        We deliberately keep the rescue ``done`` emission OUTSIDE ``finally``:
        yielding from a finally block while a ``GeneratorExit`` is propagating
        (which is exactly what happens when uvicorn aclose()s the streaming
        response after the client disconnects) raises
        ``RuntimeError: async generator ignored GeneratorExit`` and pollutes
        the logs on every turn. We also ``aclose()`` the inner
        ``turn(...)`` async-generator explicitly (via try/finally rather than
        ``contextlib.aclosing``, which would require Python 3.10+).
        """
        emitted_done = False
        agen = turn(request)
        try:
            async for ev in agen:
                if ev.kind == AgentEventKind.DONE:
                    emitted_done = True
                yield ev.to_sse().encode("utf-8")
        except Exception as exc:  # pragma: no cover — defence in depth
            logger.exception("agent turn raised: %s", exc)
            err_ev = AgentEvent(
                kind=AgentEventKind.ERROR,
                data={"message": "internal_error", "detail": str(exc)},
            )
            yield err_ev.to_sse().encode("utf-8")
        finally:
            await agen.aclose()

        if not emitted_done:
            done_ev = AgentEvent(
                kind=AgentEventKind.DONE,
                data={"session_id": request.session_id},
            )
            yield done_ev.to_sse().encode("utf-8")

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",  # disable nginx proxy buffering if any
    }
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=headers,
    )
