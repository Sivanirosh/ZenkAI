"""Corpus endpoints: works, chapters, paragraphs with annotated tokens."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models.pydantic_models import (
    Chapter,
    Paragraph,
    ParagraphBatchRequest,
    ParagraphWithTokens,
    Work,
    WorkProgress,
    WorkWithProgress,
)
from backend.services import corpus_service

router = APIRouter()


@router.get("/works", response_model=list[WorkWithProgress])
def list_works() -> list[WorkWithProgress]:
    return corpus_service.list_works()


@router.get("/works/{work_id}", response_model=Work)
def get_work(work_id: str) -> Work:
    work = corpus_service.get_work(work_id)
    if work is None:
        raise HTTPException(status_code=404, detail=f"Work '{work_id}' not found")
    return work


@router.get("/works/{work_id}/chapters", response_model=list[Chapter])
def list_chapters(work_id: str) -> list[Chapter]:
    if corpus_service.get_work(work_id) is None:
        raise HTTPException(status_code=404, detail=f"Work '{work_id}' not found")
    return corpus_service.list_chapters(work_id)


@router.get("/works/{work_id}/progress", response_model=WorkProgress)
def work_progress(work_id: str) -> WorkProgress:
    if corpus_service.get_work(work_id) is None:
        raise HTTPException(status_code=404, detail=f"Work '{work_id}' not found")
    return corpus_service.get_work_progress(work_id)


@router.get(
    "/works/{work_id}/chapters/{chapter}/paragraphs",
    response_model=list[Paragraph],
)
def list_chapter_paragraphs(work_id: str, chapter: int) -> list[Paragraph]:
    """Return all paragraphs in a chapter in reading order.

    Lets clients paginate the reader without probing for paragraph IDs one
    request at a time.
    """
    if corpus_service.get_work(work_id) is None:
        raise HTTPException(status_code=404, detail=f"Work '{work_id}' not found")
    paragraphs = corpus_service.list_paragraphs_for_chapter(work_id, chapter)
    if not paragraphs:
        raise HTTPException(
            status_code=404,
            detail=f"No paragraphs for chapter {chapter} of '{work_id}'",
        )
    return paragraphs


@router.get("/paragraphs/{paragraph_id}", response_model=ParagraphWithTokens)
def get_paragraph(paragraph_id: str) -> ParagraphWithTokens:
    paragraph = corpus_service.get_paragraph_with_tokens(paragraph_id)
    if paragraph is None:
        raise HTTPException(
            status_code=404, detail=f"Paragraph '{paragraph_id}' not found"
        )
    return paragraph


@router.post("/paragraphs/batch", response_model=list[ParagraphWithTokens])
def get_paragraphs_batch(body: ParagraphBatchRequest) -> list[ParagraphWithTokens]:
    """Return several annotated paragraphs in one round-trip.

    Used by the reader on every page turn to fetch all paragraphs of the
    current page (and prefetch the next page) with a single request.
    Preserves input order. If any id is unknown the whole batch fails with
    404 so the client doesn't have to reconcile partial results.
    """
    results = corpus_service.get_paragraphs_with_tokens(body.ids)
    if len(results) != len(body.ids):
        found = {p.id for p in results}
        missing = [pid for pid in body.ids if pid not in found]
        raise HTTPException(
            status_code=404,
            detail=f"Paragraphs not found: {', '.join(missing)}",
        )
    return results
