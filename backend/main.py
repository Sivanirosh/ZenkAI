"""FastAPI entry point for ZenkAI.

Wires routers under `/api/v1` and serves the static single-page frontend
from `static/` on the same origin.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.config import get_settings
from backend.models import db
from backend.routers import admin, annotations, chat, corpus, vocab, voice, words
from backend.services import llm_service, stt_service, tts_service

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    logger = logging.getLogger(__name__)
    logger.info("ZenkAI starting (duckdb=%s)", settings.duckdb_path)
    db.init_schema()

    probe = await llm_service.probe_ollama()
    if not probe["reachable"]:
        logger.warning(
            "Ollama unreachable at %s (%s). Chat + word annotations will use "
            "the offline fallback until the server is reachable.",
            settings.ollama_base_url,
            probe["error"],
        )
    elif not probe["configured_available"]:
        available = ", ".join(probe["models"]) or "(none)"
        logger.warning(
            "Ollama is reachable but the configured model '%s' is NOT installed. "
            "Available models: %s. Run `ollama pull %s` or set OLLAMA_MODEL in "
            ".env to one of the available models.",
            probe["configured"],
            available,
            probe["configured"],
        )
    else:
        logger.info(
            "Ollama ok \u2014 using model '%s' (%d installed).",
            probe["configured"],
            len(probe["models"]),
        )

    logger.info(
        "Voice tooling: piper=%s whisper=%s",
        "ok" if tts_service.is_available() else "missing (browser TTS fallback)",
        "ok" if stt_service.is_available() else "missing (STT disabled)",
    )
    yield
    db.close_connection()
    logger.info("ZenkAI stopped")


def create_app() -> FastAPI:
    get_settings()
    app = FastAPI(
        title="ZenkAI API",
        version="0.1.0",
        description="Backend for the ZenkAI AI-assisted German reader.",
        lifespan=lifespan,
    )

    prefix = "/api/v1"
    app.include_router(corpus.router, prefix=f"{prefix}/corpus", tags=["corpus"])
    app.include_router(vocab.router, prefix=f"{prefix}/vocab", tags=["vocab"])
    app.include_router(
        words.router, prefix=f"{prefix}/words", tags=["words (deprecated SRS)"]
    )
    app.include_router(
        annotations.router, prefix=f"{prefix}/annotations", tags=["annotations"]
    )
    app.include_router(voice.router, prefix=f"{prefix}/voice", tags=["voice"])
    app.include_router(chat.router, prefix=f"{prefix}/chat", tags=["chat"])
    app.include_router(admin.router, prefix=f"{prefix}/admin", tags=["admin"])

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


app = create_app()
