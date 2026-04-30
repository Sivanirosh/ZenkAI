"""Retrieval for literary Q&A.

Phase 2 ships *structural* retrieval only: the current paragraph plus up to
`window` neighbours in reading order, strictly within the same work (AGENT.md
RAG rule: never mix works). This is cheap, deterministic, and produces the
same "neighbouring paragraphs" context described in CLAUDE.md without requiring
Qdrant or an embedding model to be installed.

Semantic retrieval via Qdrant + `nomic-embed-text` remains the long-term plan
and will plug in through the same `retrieve(...)` facade. The shape returned
here (`list[Paragraph]`) is the contract consumers rely on.
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.models.pydantic_models import Paragraph
from backend.services import corpus_service

logger = logging.getLogger(__name__)


def retrieve_structural(paragraph_id: str, window: int = 2) -> list[Paragraph]:
    """Return the paragraph and up to `window` neighbours on each side."""
    return corpus_service.get_paragraph_neighbours(paragraph_id, window=window)


def retrieve(
    paragraph_id: str,
    work_id: Optional[str] = None,  # noqa: ARG001 — reserved for semantic path
    top_k: int = 5,
) -> list[Paragraph]:
    """Public facade. Delegates to structural retrieval for now.

    `top_k` is interpreted as "max paragraphs in the window", so we pick a
    half-window that yields at most that many rows (anchor + 2*half).
    """
    half = max(0, (top_k - 1) // 2)
    result = retrieve_structural(paragraph_id, window=half)
    if not result:
        logger.debug("rag.retrieve: empty window for %s", paragraph_id)
    return result


def format_context(paragraphs: list[Paragraph], anchor_id: str) -> str:
    """Render a window as a prompt-ready context block.

    The anchor paragraph is marked with ">>>" so the LLM can keep its focus on
    the passage the learner is actually reading.
    """
    if not paragraphs:
        return "(kein zusätzlicher Kontext verfügbar)"
    lines: list[str] = []
    for p in paragraphs:
        marker = ">>>" if p.id == anchor_id else "   "
        lines.append(f"{marker} [{p.chapter}.{p.position}] {p.text.strip()}")
    return "\n\n".join(lines)


# ─── Qdrant / embeddings (deferred) ──────────────────────────────────────


async def embed_question(text: str) -> list[float]:  # pragma: no cover
    """Placeholder for Phase 2+ semantic path."""
    raise NotImplementedError(
        "Semantic RAG is pending — install `nomic-embed-text` via Ollama and "
        "wire Qdrant before enabling."
    )
