"""Two-stage retriever (PIVOT_ROADMAP §B.15).

Stage 1 (cheap):
- For ``transcripts``: BM25 over the rotated JSONL transcript bodies.
  Uses ``rank_bm25.BM25Okapi`` with a tiny tokeniser; pure Python, ~50 ms
  on the demo corpus (5k lines).
- For ``evidence``: DuckDB ``WHERE`` clause on competency / time
  (we don't BM25-index evidence, the rows are short and structured).
- Returns up to N=200 candidate ``VectorRow`` shaped objects.

Stage 2 (expensive):
- ``store_vector.get_index().search(query, top_k, filter)`` re-rerank.
- Intersection by ``id`` only (we trust the cheap stage's recall).

Wall-clock guard:
- If Stage 1 alone uses ≥ ½ the deadline, we skip Stage 2 and return
  the Stage 1 top-k. The agent always gets *something* on time.

This module sits one level above ``store_vector``: the vector backend
doesn't know about BM25; the retriever does. Recall policies that want
Tier-3 simply call ``recall_two_stage(...)`` with the right source +
filter.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[\wäöüÄÖÜß]+", re.UNICODE)
_DEFAULT_STAGE1_LIMIT = 200
_DEFAULT_TOP_K = 5
_DEFAULT_DEADLINE_MS = 250
_TRANSCRIPT_GLOB = "transcripts.*.jsonl"


@dataclass(frozen=True)
class RetrievedRow:
    """Lightweight result row from the two-stage retriever."""

    id: str
    text: str
    source: str
    score: float
    metadata: dict[str, Any]


# ─── Public entry point ─────────────────────────────────────────────────


def recall_two_stage(
    query: str,
    *,
    source: str = "transcripts",
    top_k: int = _DEFAULT_TOP_K,
    filter: Optional[dict[str, Any]] = None,
    deadline_ms: int = _DEFAULT_DEADLINE_MS,
    memory_root: Optional[Path] = None,
) -> list[RetrievedRow]:
    """Run the two-stage pipeline and return up to ``top_k`` rows.

    Never raises — every failure mode (missing dep, no corpus, blown
    deadline) returns ``[]`` so RecallPolicies can treat Tier-3 as
    purely additive.
    """
    if not query or not query.strip():
        return []

    deadline = time.monotonic() + max(0.05, deadline_ms / 1000.0)
    half = deadline_ms / 2000.0  # half budget in seconds

    root = memory_root
    if root is None:
        try:
            from backend.memory.manager import get_memory  # noqa: WPS433

            root = Path(get_memory()._root)  # type: ignore[union-attr]
        except Exception:  # pragma: no cover
            return []

    stage1_started = time.monotonic()
    candidates: list[RetrievedRow] = _stage1(
        query, source=source, filter=filter, root=root, limit=_DEFAULT_STAGE1_LIMIT
    )
    stage1_elapsed = time.monotonic() - stage1_started

    if not candidates:
        return []

    if stage1_elapsed >= half or time.monotonic() >= deadline:
        # Out of budget — skip stage 2.
        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates[: max(1, top_k)]

    # Stage 2 — vector rerank, intersection-by-id.
    rerank_remaining = max(0.05, deadline - time.monotonic())
    return _stage2(
        query,
        candidates,
        source=source,
        top_k=top_k,
        filter=filter,
        deadline_seconds=rerank_remaining,
    )


# ─── Stage 1 ────────────────────────────────────────────────────────────


def _stage1(
    query: str,
    *,
    source: str,
    filter: Optional[dict[str, Any]],
    root: Path,
    limit: int,
) -> list[RetrievedRow]:
    if source == "transcripts":
        return _bm25_transcripts(query, root=root, filter=filter, limit=limit)
    if source == "evidence":
        return _evidence_where(query, filter=filter, limit=limit)
    return []


def _bm25_transcripts(
    query: str,
    *,
    root: Path,
    filter: Optional[dict[str, Any]],
    limit: int,
) -> list[RetrievedRow]:
    konv_dir = root / "konversation"
    if not konv_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(konv_dir.glob(_TRANSCRIPT_GLOB)):
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(obj.get("text") or "").strip()
            if not text:
                continue
            meta = {
                "role": str(obj.get("role") or ""),
                "session_id": str(obj.get("session_id") or ""),
                "scenario": str(obj.get("scenario") or ""),
                "competency_id": str(obj.get("competency_id") or ""),
            }
            if filter and not _matches_filter(meta, filter):
                continue
            rows.append({"obj": obj, "text": text, "meta": meta})
    if not rows:
        return []

    # BM25 needs >= 2 docs to give meaningful scores; tiny corpora fall
    # back to overlap-of-tokens so a single matching doc still surfaces.
    if len(rows) < 2:
        return _substring_score_rows(query, rows, source="transcripts", limit=limit)

    try:
        from rank_bm25 import BM25Okapi  # noqa: WPS433
    except ImportError:  # pragma: no cover — pinned in pyproject
        return _substring_score_rows(query, rows, source="transcripts", limit=limit)

    corpus = [_tokenise(r["text"]) for r in rows]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenise(query))
    ranked = sorted(
        zip(rows, scores), key=lambda pair: pair[1], reverse=True
    )[:limit]
    out: list[RetrievedRow] = []
    for r, score in ranked:
        if score <= 0:
            continue
        obj = r["obj"]
        out.append(
            RetrievedRow(
                id=str(obj.get("id") or ""),
                text=r["text"],
                source="transcripts",
                score=float(score),
                metadata=r["meta"],
            )
        )
    if not out:
        return _substring_score_rows(query, rows, source="transcripts", limit=limit)
    return out


def _evidence_where(
    query: str,
    *,
    filter: Optional[dict[str, Any]],
    limit: int,
) -> list[RetrievedRow]:
    try:
        from backend.models import db  # noqa: WPS433
    except Exception:  # pragma: no cover
        return []

    where: list[str] = []
    params: list[Any] = []
    if filter and "competency_id" in filter:
        where.append("competency_id = ?")
        params.append(str(filter["competency_id"]))
    if filter and "since" in filter:
        where.append("occurred_at >= ?")
        params.append(str(filter["since"]))
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    sql = (
        "SELECT id, competency_id, surface_form, quality, source, occurred_at "
        f"FROM evidence{where_sql} ORDER BY occurred_at DESC LIMIT ?"
    )
    params.append(int(limit))
    try:
        rows = db.cursor().execute(sql, params).fetchall()
    except Exception as exc:  # pragma: no cover
        logger.debug("evidence stage1 failed: %s", exc)
        return []

    out: list[RetrievedRow] = []
    q_tokens = set(_tokenise(query))
    for row in rows:
        eid, comp, surface, quality, src, ts = row
        text = str(surface or "")
        if not text.strip():
            continue
        # Cheap textual overlap so the BM25-less branch still ranks.
        overlap = len(q_tokens.intersection(_tokenise(text)))
        out.append(
            RetrievedRow(
                id=str(eid),
                text=text,
                source="evidence",
                score=float(overlap if overlap else 0.001),
                metadata={
                    "competency_id": str(comp or ""),
                    "quality": float(quality or 0.0),
                    "source_kind": str(src or ""),
                    "occurred_at": str(ts or ""),
                },
            )
        )
    return out


def _substring_score_rows(
    query: str,
    rows: list[dict[str, Any]],
    *,
    source: str,
    limit: int,
) -> list[RetrievedRow]:
    """Fallback when rank-bm25 is missing — overlap-of-tokens."""
    q_tokens = set(_tokenise(query))
    scored: list[tuple[float, dict[str, Any]]] = []
    for r in rows:
        overlap = len(q_tokens.intersection(_tokenise(r["text"])))
        if overlap == 0:
            continue
        scored.append((float(overlap), r))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    out: list[RetrievedRow] = []
    for score, r in scored[:limit]:
        obj = r["obj"]
        out.append(
            RetrievedRow(
                id=str(obj.get("id") or ""),
                text=r["text"],
                source=source,
                score=score,
                metadata=r["meta"],
            )
        )
    return out


# ─── Stage 2 ────────────────────────────────────────────────────────────


def _stage2(
    query: str,
    candidates: list[RetrievedRow],
    *,
    source: str,
    top_k: int,
    filter: Optional[dict[str, Any]],
    deadline_seconds: float,
) -> list[RetrievedRow]:
    try:
        from backend.memory import store_vector as _store_vector  # noqa: WPS433
    except Exception:  # pragma: no cover
        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates[: max(1, top_k)]

    try:
        index = _store_vector.get_index()
    except Exception:  # pragma: no cover
        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates[: max(1, top_k)]

    if isinstance(index, _store_vector.NullVectorIndex):
        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates[: max(1, top_k)]

    started = time.monotonic()
    try:
        hits = index.search(
            query,
            top_k=max(top_k * 4, 20),
            source=source,
            filter=filter,
        )
    except Exception as exc:  # pragma: no cover
        logger.debug("stage 2 search failed: %s", exc)
        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates[: max(1, top_k)]

    if (time.monotonic() - started) > deadline_seconds and not hits:
        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates[: max(1, top_k)]

    candidate_ids = {c.id: c for c in candidates}
    merged: list[RetrievedRow] = []
    seen: set[str] = set()
    for hit in hits:
        rid = hit.row.id
        if rid in candidate_ids and rid not in seen:
            seen.add(rid)
            merged.append(
                RetrievedRow(
                    id=rid,
                    text=hit.row.text,
                    source=source,
                    score=float(hit.score),
                    metadata=hit.row.metadata,
                )
            )

    # Backfill remaining slots with stage-1 leftovers (preserves recall
    # when the vector index is sparse).
    if len(merged) < top_k:
        leftovers = [c for c in candidates if c.id not in seen]
        leftovers.sort(key=lambda r: r.score, reverse=True)
        merged.extend(leftovers[: top_k - len(merged)])

    return merged[: max(1, top_k)]


# ─── Helpers ────────────────────────────────────────────────────────────


def _tokenise(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _matches_filter(meta: dict[str, Any], filt: dict[str, Any]) -> bool:
    for k, v in filt.items():
        if k == "since":  # special-case; handled in caller
            continue
        if str(meta.get(k, "")).strip() != str(v).strip():
            return False
    return True


__all__ = ["RetrievedRow", "recall_two_stage"]
