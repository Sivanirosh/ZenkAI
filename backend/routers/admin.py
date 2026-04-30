"""Runtime admin endpoints.

Exposes a minimal surface to verify the Ollama connection, list installed
models, and update runtime options (active model, thinking mode, temperature).
Intended for a small in-app settings panel, not a full admin UI.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.services import llm_service
from backend.services.runtime_config import LlmOptions, get_llm_options, set_llm_options

logger = logging.getLogger(__name__)
router = APIRouter()


class LlmOptionsPayload(BaseModel):
    model: Optional[str] = None
    think: Optional[bool] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)


def _serialise(options: LlmOptions) -> dict:
    return {
        "model": options.model,
        "think": options.think,
        "temperature": options.temperature,
    }


@router.get("/ollama")
async def get_ollama_status() -> dict:
    """Return the active options, Ollama reachability, and installed models."""
    options = get_llm_options()
    probe = await llm_service.probe_ollama()
    return {
        "options": _serialise(options),
        "ollama": {
            "reachable": probe.get("reachable", False),
            "base_url": get_settings().ollama_base_url,
            "models": probe.get("models", []),
            "configured_available": probe.get("configured_available", False),
            "error": probe.get("error"),
        },
    }


@router.put("/ollama")
async def update_ollama_options(payload: LlmOptionsPayload) -> dict:
    """Patch the runtime LLM options."""
    if payload.model is not None and not payload.model.strip():
        raise HTTPException(status_code=400, detail="model must not be empty")

    if payload.model is not None:
        try:
            models = await _installed_models()
        except httpx.HTTPError as exc:
            logger.warning("Ollama unreachable while validating model: %s", exc)
            models = None
        if models is not None and payload.model not in models:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"model '{payload.model}' is not installed. "
                    f"Available: {', '.join(models) or '(none)'}."
                ),
            )

    updated = set_llm_options(
        model=payload.model.strip() if payload.model else None,
        think=payload.think,
        temperature=payload.temperature,
    )
    logger.info("LLM options updated: %s", _serialise(updated))
    return {"options": _serialise(updated)}


async def _installed_models() -> list[str]:
    settings = get_settings()
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    async with httpx.AsyncClient(timeout=3.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
    return [m.get("name") for m in data.get("models", []) if m.get("name")]


@router.get("/agent/health")
async def get_agent_health() -> dict:
    """Quick check the frontend can render as a 'tier badge' next to Mira.

    Returns the active provider, the model that will run the next turn, and
    whether tool calling is supported. Tool calling is hard-coded ``true``
    because we only ship providers that support it; switching to a no-tools
    fallback would change this flag.
    """
    options = get_llm_options()
    return {
        "provider": "ollama",
        "model": options.model,
        "tool_calling_supported": True,
    }


@router.post("/sessions/{session_id}/close")
async def close_session(session_id: str) -> dict:
    """Force-close a Mira session.

    Flushes the JSONL buffer, snapshots ``mastery.parquet`` atomically,
    and appends to the session index. Used by the milestone demo and by
    the frontend's "end session" button.
    """
    from backend.memory import get_memory  # local import to keep router cold-start light

    memory = get_memory()
    memory.close_session(session_id)
    return {"session_id": session_id, "closed": True}


class EvidencePayload(BaseModel):
    competency_id: str = Field(min_length=1)
    quality: float = Field(ge=0.0, le=1.0)
    surface_form: Optional[str] = None
    notes: Optional[str] = None
    source: str = Field(default="agent_post")


@router.post("/evidence")
async def write_evidence(payload: EvidencePayload) -> dict:
    """Append one evidence event (PIVOT_ROADMAP §B.13/§B.14 demo surface).

    Exposed so the milestone demo and the future "I read this" button
    can drive ``mastery.update`` without owning the DuckDB connection.
    The same write path triggers the factsheet writer and vector index.
    """
    from backend.mastery import model as mastery_model  # noqa: WPS433

    try:
        row = mastery_model.update(
            payload.competency_id,
            payload.quality,
            source=payload.source,
            surface_form=payload.surface_form,
            notes=payload.notes,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "competency_id": row.competency_id,
        "confidence": row.confidence,
        "variance": row.variance,
        "evidence_count": row.evidence_count,
    }


class CompactPayload(BaseModel):
    deadline_ms: int = Field(default=30_000, ge=500, le=300_000)


@router.post("/memory/compact")
async def compact_memory(payload: Optional[CompactPayload] = None) -> dict:
    """Run one idempotent compaction pass against the live memory root.

    Returns the ``CompactionReport`` (see backend/memory/compactor.py).
    Returns 423 (locked) if another compactor is currently running so
    the caller knows to retry later instead of treating the response
    as success.
    """
    from backend.memory import get_memory  # local import keeps cold-start light
    from backend.memory.compactor import CompactionLockedError

    body = payload or CompactPayload()
    memory = get_memory()
    try:
        report = memory.compact(deadline_ms=body.deadline_ms)
    except CompactionLockedError as exc:
        raise HTTPException(status_code=423, detail=str(exc)) from exc
    return report.to_dict()
