"""Word state + SRS endpoints.

DEPRECATED — the SRS engine lives in an external app. See
``.agent/decisions/0001-external-srs-pivot.md``. The review and scheduling
endpoints here are preserved for one migration cycle so that legacy tests
and the ingestion smoke-checks keep working, but the frontend no longer
consumes them. New code should use ``backend/routers/vocab.py``.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models import db
from backend.models.pydantic_models import (
    ProgressSummary,
    ReviewRequest,
    Word,
    WordState,
    WordWithState,
)
from backend.services import word_service

router = APIRouter()


def _fetch_word(word_id: str) -> Word | None:
    row = db.cursor().execute(
        """
        SELECT id, lemma, pos, gender, plural_form, etymology,
               definition_de, definition_en
        FROM words WHERE id = ?
        """,
        [word_id],
    ).fetchone()
    if row is None:
        return None
    return Word(
        id=row[0],
        lemma=row[1],
        pos=row[2],
        gender=row[3],
        plural_form=row[4],
        etymology=row[5],
        definition_de=row[6],
        definition_en=row[7],
    )


@router.get("/progress", response_model=ProgressSummary)
def progress() -> ProgressSummary:
    summary = word_service.get_progress_summary()
    return ProgressSummary(**summary)


@router.get("/queue/review", response_model=list[WordState], deprecated=True)
def review_queue(limit: int = 20) -> list[WordState]:
    return word_service.get_review_queue(limit=limit)


@router.get("/{word_id}", response_model=WordWithState)
def get_word(word_id: str) -> WordWithState:
    word = _fetch_word(word_id)
    if word is None:
        raise HTTPException(status_code=404, detail=f"Word '{word_id}' not found")
    state = word_service.get_state(word_id) or WordState(word_id=word_id)
    return WordWithState(**word.model_dump(), state=state)


@router.post("/{word_id}/seen", response_model=WordState, deprecated=True)
def mark_seen(word_id: str) -> WordState:
    if _fetch_word(word_id) is None:
        raise HTTPException(status_code=404, detail=f"Word '{word_id}' not found")
    return word_service.mark_seen(word_id)


@router.post("/{word_id}/opened", response_model=WordState, deprecated=True)
def mark_opened(word_id: str) -> WordState:
    if _fetch_word(word_id) is None:
        raise HTTPException(status_code=404, detail=f"Word '{word_id}' not found")
    return word_service.mark_opened(word_id)


@router.post("/{word_id}/reviewed", response_model=WordState, deprecated=True)
def review(word_id: str, body: ReviewRequest) -> WordState:
    if _fetch_word(word_id) is None:
        raise HTTPException(status_code=404, detail=f"Word '{word_id}' not found")
    return word_service.record_review(word_id, body.quality)


@router.post("/{word_id}/known", response_model=WordState, deprecated=True)
def mark_known(word_id: str) -> WordState:
    if _fetch_word(word_id) is None:
        raise HTTPException(status_code=404, detail=f"Word '{word_id}' not found")
    return word_service.mark_known(word_id)
