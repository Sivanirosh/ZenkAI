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
