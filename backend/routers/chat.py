"""Chat / RAG streaming endpoint.

Uses Server-Sent Events over a POST. Each `token` event carries one delta
chunk of the AI answer. A final `done` event signals completion; an `error`
event is emitted when the LLM fallback kicks in *and* we still want the
client to know the stream is over gracefully.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from backend.models.pydantic_models import ChatRequest
from backend.services import llm_service

logger = logging.getLogger(__name__)
router = APIRouter()


def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/message")
async def chat_message(req: ChatRequest) -> StreamingResponse:
    question = req.question.strip()
    if not question:
        async def empty() -> AsyncIterator[str]:
            yield _sse(
                "error",
                {"detail": "question must not be empty"},
            )
            yield _sse("done", {})

        return StreamingResponse(
            empty(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def gen() -> AsyncIterator[str]:
        try:
            async for chunk in llm_service.stream_answer(
                question=question,
                paragraph_id=req.paragraph_id,
                history=req.history,
            ):
                yield _sse("token", {"delta": chunk})
        except Exception as exc:  # pragma: no cover — defensive
            logger.exception("chat stream crashed: %s", exc)
            yield _sse(
                "error",
                {"detail": "stream failed", "message": str(exc)},
            )
        finally:
            yield _sse("done", {})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
