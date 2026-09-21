"""Prompt-context formatting for literary Q&A.

Retrieval is structural: the current paragraph plus up to `window` neighbours
in reading order, strictly within the same work (never mix works). The
paragraphs themselves come from `corpus_service.get_paragraph_neighbours`;
this module only renders a window as a prompt-ready context block.
"""

from __future__ import annotations

from backend.models.pydantic_models import Paragraph


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
