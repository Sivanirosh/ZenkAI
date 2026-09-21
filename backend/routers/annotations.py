"""Annotation endpoint: word annotation (LLM or mock)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models.pydantic_models import WordAnnotation, WordAnnotationRequest
from backend.services import llm_service

router = APIRouter()


@router.post("/word", response_model=WordAnnotation)
async def annotate_word(req: WordAnnotationRequest) -> WordAnnotation:
    if not req.word.strip():
        raise HTTPException(status_code=400, detail="word must not be empty")
    return await llm_service.annotate_word(
        word=req.word,
        sentence=req.sentence,
        grammatical_role=req.grammatical_role,
        case_label=req.case_label,
    )
