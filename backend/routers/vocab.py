"""Vocabulary harvester endpoints.

See decision ``.agent/decisions/0001-external-srs-pivot.md``. This router is
the reader's only state-mutating surface for word triage; the legacy
``/api/v1/words/*`` SRS endpoints are deprecated.
"""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from backend.models import db
from backend.models.pydantic_models import (
    VocabEnqueueRequest,
    VocabEntry,
    VocabKnownRequest,
    VocabOverrideRequest,
    VocabStatusMap,
)
from backend.services import vocab_service

logger = logging.getLogger(__name__)

router = APIRouter()


def _assert_word_exists(word_id: str) -> None:
    row = db.cursor().execute(
        "SELECT 1 FROM words WHERE id = ?", [word_id]
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Word '{word_id}' not found")


@router.post("/queue", response_model=VocabEntry)
def enqueue(body: VocabEnqueueRequest) -> VocabEntry:
    _assert_word_exists(body.word_id)
    return vocab_service.enqueue(
        body.word_id,
        paragraph_id=body.paragraph_id,
        sentence=body.sentence,
    )


@router.post("/known", response_model=VocabEntry)
def mark_known(body: VocabKnownRequest) -> VocabEntry:
    _assert_word_exists(body.word_id)
    return vocab_service.mark_known(body.word_id)


@router.delete("/queue/{word_id}", status_code=204)
def remove(word_id: str) -> Response:
    vocab_service.remove(word_id)
    return Response(status_code=204)


@router.get("/queue", response_model=list[VocabEntry])
def list_queue(
    status: Optional[str] = Query(default="queued"),
    limit: Optional[int] = Query(default=None, ge=1, le=10000),
) -> list[VocabEntry]:
    if status not in {"queued", "known", "exported", "all"}:
        raise HTTPException(
            status_code=400,
            detail="status must be one of queued|known|exported|all",
        )
    return vocab_service.list_queue(
        status=None if status == "all" else status,  # type: ignore[arg-type]
        limit=limit,
    )


@router.post("/statuses", response_model=VocabStatusMap)
def get_statuses(body: dict) -> VocabStatusMap:
    """Batch status lookup used to hydrate the reader on page load."""
    ids = body.get("word_ids") or []
    if not isinstance(ids, list) or any(not isinstance(w, str) for w in ids):
        raise HTTPException(
            status_code=400, detail="word_ids must be a list of strings"
        )
    if len(ids) > 2000:
        raise HTTPException(status_code=400, detail="word_ids limit is 2000")
    return VocabStatusMap(statuses=vocab_service.get_status_map(ids))


@router.get("/counts")
def counts() -> dict[str, int]:
    return vocab_service.counts_by_status()


@router.patch("/queue/{word_id}", response_model=VocabEntry)
def update_overrides(word_id: str, body: VocabOverrideRequest) -> VocabEntry:
    entry = vocab_service.update_overrides(
        word_id,
        question=body.question,
        answer=body.answer,
        extra_tags=body.extra_tags,
    )
    if entry is None:
        raise HTTPException(
            status_code=404, detail=f"Vocab entry '{word_id}' not found"
        )
    return entry


@router.post("/queue/{word_id}/annotate", response_model=VocabEntry)
async def annotate(word_id: str) -> VocabEntry:
    entry = await vocab_service.ensure_annotated(word_id)
    if entry is None:
        raise HTTPException(
            status_code=404, detail=f"Vocab entry '{word_id}' not found"
        )
    return entry


@router.post("/export")
def export_csv() -> FileResponse:
    """Stream the queued rows as CSV and flip them to ``exported`` atomically.

    Returns 204 when the queue is empty so the frontend can disable the
    Export button without surfacing a confusing download.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="vocab-export-"))
    out_path = tmp_dir / (
        f"zenkai-vocab-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv"
    )
    written = vocab_service.export_to_csv(out_path)
    if written == 0:
        logger.info("Export requested but queue is empty")
        return Response(status_code=204)  # type: ignore[return-value]

    def _cleanup() -> None:
        try:
            out_path.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except OSError:
            logger.debug("Failed to clean up %s", tmp_dir, exc_info=True)

    return FileResponse(
        path=out_path,
        media_type="text/csv",
        filename=out_path.name,
        background=BackgroundTask(_cleanup),
    )
