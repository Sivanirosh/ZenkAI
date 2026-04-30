"""Vector index for Mira's long-term memory (PIVOT_ROADMAP §B.14).

Three implementations behind one Protocol:

- ``LocalEmbeddedVectorIndex`` (default) — embeddings via ``fastembed``,
  vectors persisted to ``data/memory/embeddings/<source>.parquet`` with
  cosine search done in numpy. Self-contained, no network, ~30 MB ONNX
  one-time download to ``~/.cache/fastembed`` on first use.
- ``QdrantVectorIndex`` — only constructed when
  ``LINGUAMATE_VECTOR_BACKEND=qdrant``. Same Protocol, useful when we
  want to ship a real vector DB on a workstation.
- ``NullVectorIndex`` — fallback when neither backend is importable.
  ``search`` returns ``[]`` so every caller already handles cold-start
  gracefully (B.15's two-stage retriever, etc.).

Sources we know how to rebuild from JSONL today:

- ``transcripts``  — ``konversation/transcripts.NNNN.jsonl`` (B.5)
- ``evidence``     — ``evidence/ledger.jsonl``               (B.13 / mastery)

Persistence:
  data/memory/embeddings/<source>.parquet
    columns: id (str), text (str), vector (list<float32>),
             metadata (json str), ts (str)

Idempotence: ``add`` upserts by ``id`` (last write wins); the parquet is
re-written atomically (tmp+rename). ``rebuild_from_jsonl`` truncates +
re-fills, so re-running is safe.

The whole module is async-safe enough for the agent loop: embedding is
done lazily (model loaded on first call), and add/search lock the
in-memory cache.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


_EMBEDDINGS_DIR_NAME = "embeddings"
_DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_EMBED_DIM = 384
_VECTOR_BACKEND_ENV = "LINGUAMATE_VECTOR_BACKEND"
_OFFLINE_ENV = "LINGUAMATE_OFFLINE"


# ─── Public types ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class VectorRow:
    """One thing we want to be able to recall by semantic similarity."""

    id: str
    text: str
    source: str                              # 'transcripts' | 'evidence' | ...
    metadata: dict[str, Any] = field(default_factory=dict)
    ts: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


@dataclass(frozen=True)
class VectorHit:
    row: VectorRow
    score: float


@runtime_checkable
class VectorIndex(Protocol):
    """Async-naive vector index. All ops are in-process / cheap I/O."""

    def add(self, rows: Iterable[VectorRow]) -> int: ...

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        source: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[VectorHit]: ...

    def count(self, *, source: Optional[str] = None) -> int: ...

    def rebuild_from_jsonl(self, source: str) -> int: ...


# ─── Embedder protocol + lazy fastembed loader ──────────────────────────


class _Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class _FastembedAdapter:
    """Wraps ``fastembed.TextEmbedding`` in our minimal Protocol."""

    def __init__(self, model_name: str = _DEFAULT_EMBED_MODEL) -> None:
        from fastembed import TextEmbedding  # type: ignore

        self._model = TextEmbedding(model_name=model_name)
        self.name = model_name
        self.dim = _DEFAULT_EMBED_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for vec in self._model.embed(texts):
            out.append([float(x) for x in vec])
        return out


def _load_default_embedder() -> Optional[_Embedder]:
    if os.environ.get(_OFFLINE_ENV):
        logger.info("vector_store: offline mode — embedder disabled")
        return None
    try:
        return _FastembedAdapter()
    except Exception as exc:  # pragma: no cover — fastembed not installed
        logger.info("vector_store: fastembed unavailable (%s) — using Null index", exc)
        return None


# ─── LocalEmbeddedVectorIndex ───────────────────────────────────────────


class LocalEmbeddedVectorIndex:
    """Default backend — fastembed + numpy + parquet on disk.

    Per ``source`` we keep one ``.parquet`` and one in-memory
    ``np.ndarray`` of vectors so search is pure numpy. Both are
    rebuilt cold from disk on construction and written atomically on
    every ``add`` / ``rebuild_from_jsonl`` call.
    """

    def __init__(
        self,
        *,
        root: Path,
        embedder: Optional[_Embedder] = None,
    ) -> None:
        self._root = Path(root)
        self._embed_dir = self._root / _EMBEDDINGS_DIR_NAME
        self._embed_dir.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._lock = threading.RLock()
        self._cache: dict[str, list[VectorRow]] = {}     # source -> rows
        self._vectors: dict[str, list[list[float]]] = {}  # source -> raw vecs
        self._loaded: set[str] = set()

    # ── Embedder lifecycle ─────────────────────────────────────────────

    def _ensure_embedder(self) -> Optional[_Embedder]:
        if self._embedder is not None:
            return self._embedder
        self._embedder = _load_default_embedder()
        return self._embedder

    # ── Disk IO ────────────────────────────────────────────────────────

    def _parquet_path(self, source: str) -> Path:
        return self._embed_dir / f"{source}.parquet"

    def _load_source(self, source: str) -> None:
        if source in self._loaded:
            return
        path = self._parquet_path(source)
        if not path.exists():
            self._cache[source] = []
            self._vectors[source] = []
            self._loaded.add(source)
            return
        try:
            import pyarrow.parquet as pq  # noqa: WPS433

            table = pq.read_table(path)
            rows_dicts = table.to_pylist()
        except Exception as exc:  # pragma: no cover
            logger.warning("vector_store: could not read %s (%s)", path, exc)
            rows_dicts = []

        rows: list[VectorRow] = []
        vectors: list[list[float]] = []
        for d in rows_dicts:
            try:
                meta_raw = d.get("metadata") or "{}"
                meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)
            except (json.JSONDecodeError, TypeError):
                meta = {}
            rows.append(
                VectorRow(
                    id=str(d.get("id")),
                    text=str(d.get("text") or ""),
                    source=source,
                    metadata=meta,
                    ts=str(d.get("ts") or ""),
                )
            )
            raw = d.get("vector") or []
            vectors.append([float(x) for x in raw])
        self._cache[source] = rows
        self._vectors[source] = vectors
        self._loaded.add(source)

    def _flush_source(self, source: str) -> None:
        rows = self._cache.get(source) or []
        vectors = self._vectors.get(source) or []
        if not rows:
            # Nothing to flush — but still ensure file is gone for cleanliness.
            path = self._parquet_path(source)
            if path.exists():
                try:
                    path.unlink()
                except OSError:  # pragma: no cover
                    pass
            return
        try:
            import pyarrow as pa  # noqa: WPS433
            import pyarrow.parquet as pq  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover — pyproject pins it
            logger.warning("vector_store: pyarrow missing (%s) — skipping flush", exc)
            return

        table = pa.table(
            {
                "id": [r.id for r in rows],
                "text": [r.text for r in rows],
                "vector": vectors,
                "metadata": [json.dumps(r.metadata, ensure_ascii=False) for r in rows],
                "ts": [r.ts for r in rows],
            }
        )
        path = self._parquet_path(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            pq.write_table(table, tmp)
            os.replace(tmp, path)
        except Exception as exc:  # pragma: no cover
            logger.warning("vector_store: flush failed for %s (%s)", path, exc)
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    # ── Public API ─────────────────────────────────────────────────────

    def add(self, rows: Iterable[VectorRow]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        embedder = self._ensure_embedder()
        if embedder is None:
            return 0

        # Group by source so we batch the embed call.
        per_source: dict[str, list[VectorRow]] = {}
        for r in rows:
            per_source.setdefault(r.source, []).append(r)

        added = 0
        with self._lock:
            for source, batch in per_source.items():
                self._load_source(source)
                texts = [r.text for r in batch]
                try:
                    vecs = embedder.embed(texts)
                except Exception as exc:  # pragma: no cover
                    logger.warning("vector_store: embed failed (%s)", exc)
                    continue
                if len(vecs) != len(batch):
                    logger.warning(
                        "vector_store: embed length mismatch (%s vs %s)",
                        len(vecs), len(batch),
                    )
                    continue
                self._upsert(source, batch, vecs)
                added += len(batch)
                self._flush_source(source)
        return added

    def _upsert(
        self, source: str, new_rows: list[VectorRow], new_vecs: list[list[float]]
    ) -> None:
        rows = self._cache[source]
        vecs = self._vectors[source]
        existing_ids = {r.id: i for i, r in enumerate(rows)}
        for new_row, new_vec in zip(new_rows, new_vecs):
            idx = existing_ids.get(new_row.id)
            if idx is None:
                rows.append(new_row)
                vecs.append(new_vec)
                existing_ids[new_row.id] = len(rows) - 1
            else:
                rows[idx] = new_row
                vecs[idx] = new_vec

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        source: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[VectorHit]:
        embedder = self._ensure_embedder()
        if embedder is None:
            return []
        try:
            qvec = embedder.embed([query])
        except Exception as exc:  # pragma: no cover
            logger.warning("vector_store: query embed failed (%s)", exc)
            return []
        if not qvec:
            return []
        q = qvec[0]

        with self._lock:
            sources = [source] if source else self._all_sources()
            for s in sources:
                self._load_source(s)
            candidates: list[VectorHit] = []
            for s in sources:
                rows = self._cache.get(s, [])
                vecs = self._vectors.get(s, [])
                for row, vec in zip(rows, vecs):
                    if filter and not _matches_filter(row.metadata, filter):
                        continue
                    score = _cosine(q, vec)
                    candidates.append(VectorHit(row=row, score=score))
        candidates.sort(key=lambda h: h.score, reverse=True)
        return candidates[: max(1, top_k)]

    def count(self, *, source: Optional[str] = None) -> int:
        with self._lock:
            sources = [source] if source else self._all_sources()
            for s in sources:
                self._load_source(s)
            return sum(len(self._cache.get(s, [])) for s in sources)

    def rebuild_from_jsonl(self, source: str) -> int:
        """Read the canonical JSONL for ``source`` and re-embed everything.

        Truncates the existing parquet (semantically: full rebuild). Use
        from the compactor or from a startup self-heal job. Cheap when
        the corpus is small (the demo box stays well under 5 k rows).
        """
        rows = list(_jsonl_rows_for(source, self._root))
        with self._lock:
            self._cache[source] = []
            self._vectors[source] = []
            self._loaded.add(source)
        if not rows:
            self._flush_source(source)
            return 0
        return self.add(rows)

    def _all_sources(self) -> list[str]:
        sources = set(self._cache.keys())
        for path in self._embed_dir.glob("*.parquet"):
            sources.add(path.stem)
        return sorted(sources)


# ─── QdrantVectorIndex ──────────────────────────────────────────────────


class QdrantVectorIndex:
    """Optional Qdrant-backed adapter (same Protocol)."""

    COLLECTION = "linguamate"

    def __init__(self, *, root: Path, embedder: Optional[_Embedder] = None) -> None:
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.http.models import Distance, VectorParams  # type: ignore

        path = root / "qdrant"
        path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(path))
        self._embedder = embedder or _load_default_embedder()
        if self._embedder is None:
            raise RuntimeError("Qdrant adapter requires a working embedder")
        try:
            self._client.get_collection(self.COLLECTION)
        except Exception:
            self._client.recreate_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(size=self._embedder.dim, distance=Distance.COSINE),
            )

    def add(self, rows: Iterable[VectorRow]) -> int:  # pragma: no cover — optional
        from qdrant_client.http.models import PointStruct  # type: ignore

        rows = list(rows)
        if not rows or self._embedder is None:
            return 0
        vecs = self._embedder.embed([r.text for r in rows])
        points = [
            PointStruct(
                id=r.id,
                vector=vec,
                payload={"text": r.text, "source": r.source, "ts": r.ts, **r.metadata},
            )
            for r, vec in zip(rows, vecs)
        ]
        self._client.upsert(collection_name=self.COLLECTION, points=points)
        return len(points)

    def search(  # pragma: no cover — optional
        self,
        query: str,
        *,
        top_k: int = 5,
        source: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[VectorHit]:
        if self._embedder is None:
            return []
        qvec = self._embedder.embed([query])[0]
        hits = self._client.search(
            collection_name=self.COLLECTION,
            query_vector=qvec,
            limit=top_k,
        )
        out: list[VectorHit] = []
        for h in hits:
            payload = h.payload or {}
            row = VectorRow(
                id=str(h.id),
                text=str(payload.get("text") or ""),
                source=str(payload.get("source") or source or ""),
                metadata={k: v for k, v in payload.items() if k not in {"text", "source", "ts"}},
                ts=str(payload.get("ts") or ""),
            )
            if source and row.source != source:
                continue
            if filter and not _matches_filter(row.metadata, filter):
                continue
            out.append(VectorHit(row=row, score=float(h.score)))
        return out

    def count(self, *, source: Optional[str] = None) -> int:  # pragma: no cover — optional
        info = self._client.get_collection(self.COLLECTION)
        return int(getattr(info, "points_count", 0) or 0)

    def rebuild_from_jsonl(self, source: str) -> int:  # pragma: no cover — optional
        rows = list(_jsonl_rows_for(source, Path(self._client._location).parent))
        if not rows:
            return 0
        return self.add(rows)


# ─── NullVectorIndex ────────────────────────────────────────────────────


class NullVectorIndex:
    """No-op fallback. Always safe to call; ``search`` returns ``[]``."""

    def add(self, rows: Iterable[VectorRow]) -> int:
        return 0

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        source: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[VectorHit]:
        return []

    def count(self, *, source: Optional[str] = None) -> int:
        return 0

    def rebuild_from_jsonl(self, source: str) -> int:
        return 0


# ─── JSONL → VectorRow adapters ─────────────────────────────────────────


def _jsonl_rows_for(source: str, memory_root: Path) -> Iterable[VectorRow]:
    if source == "transcripts":
        yield from _transcripts_rows(memory_root)
    elif source == "evidence":
        yield from _evidence_rows(memory_root)
    else:
        return


def _transcripts_rows(memory_root: Path) -> Iterable[VectorRow]:
    konv_dir = memory_root / "konversation"
    if not konv_dir.exists():
        return
    for path in sorted(konv_dir.glob("transcripts.*.jsonl")):
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
            yield VectorRow(
                id=str(obj.get("id") or f"{path.name}:{hash(line) & 0xFFFFFFFF:x}"),
                text=text,
                source="transcripts",
                metadata={
                    "role": str(obj.get("role") or ""),
                    "session_id": str(obj.get("session_id") or ""),
                    "scenario": str(obj.get("scenario") or ""),
                },
                ts=str(obj.get("ts") or obj.get("occurred_at") or ""),
            )


def _evidence_rows(memory_root: Path) -> Iterable[VectorRow]:
    ledger = memory_root / "evidence" / "ledger.jsonl"
    if not ledger.exists():
        return
    try:
        raw = ledger.read_text(encoding="utf-8")
    except OSError:
        return
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = str(obj.get("surface_form") or obj.get("text") or "").strip()
        if not text:
            continue
        yield VectorRow(
            id=str(obj.get("id") or f"ev:{hash(line) & 0xFFFFFFFF:x}"),
            text=text,
            source="evidence",
            metadata={
                "competency_id": str(obj.get("competency_id") or ""),
                "quality": float(obj.get("quality") or 0.0),
                "source_kind": str(obj.get("source") or ""),
            },
            ts=str(obj.get("occurred_at") or ""),
        )


# ─── Helpers ────────────────────────────────────────────────────────────


def _cosine(a: list[float], b: list[float]) -> float:
    try:
        import numpy as np  # noqa: WPS433

        va = np.asarray(a, dtype="float32")
        vb = np.asarray(b, dtype="float32")
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)
    except Exception:  # pragma: no cover
        return 0.0


def _matches_filter(meta: dict[str, Any], filt: dict[str, Any]) -> bool:
    for k, v in filt.items():
        if meta.get(k) != v:
            return False
    return True


# ─── Singleton ──────────────────────────────────────────────────────────


_INDEX: Optional[VectorIndex] = None
_INDEX_LOCK = threading.Lock()


def get_index() -> VectorIndex:
    """Return the process-wide vector index, building it on first use."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    with _INDEX_LOCK:
        if _INDEX is not None:
            return _INDEX
        _INDEX = _build_index()
        return _INDEX


def reset_for_tests(*, embedder: Optional[_Embedder] = None) -> VectorIndex:
    """Replace the singleton with a fresh local index pinned to MemoryManager root.

    Tests must call this between cases so the parquet cache from a prior
    case can't leak into the new one.
    """
    global _INDEX
    with _INDEX_LOCK:
        from backend.memory.manager import get_memory  # noqa: WPS433

        try:
            root = Path(get_memory()._root)  # type: ignore[union-attr]
        except Exception:
            root = Path("/tmp")  # noqa: S108 — defensive fallback
        _INDEX = LocalEmbeddedVectorIndex(root=root, embedder=embedder)
        return _INDEX


def _build_index() -> VectorIndex:
    backend = (os.environ.get(_VECTOR_BACKEND_ENV) or "local").lower()
    try:
        from backend.memory.manager import get_memory  # noqa: WPS433

        root = Path(get_memory()._root)  # type: ignore[union-attr]
    except Exception as exc:  # pragma: no cover — only seen in unit tests
        logger.debug("vector_store: MemoryManager unavailable (%s)", exc)
        return NullVectorIndex()

    if backend == "qdrant":
        try:
            return QdrantVectorIndex(root=root)
        except Exception as exc:  # pragma: no cover — qdrant optional
            logger.warning("vector_store: qdrant unavailable (%s); using local", exc)

    if backend == "null":
        return NullVectorIndex()

    embedder = _load_default_embedder()
    if embedder is None:
        return NullVectorIndex()
    return LocalEmbeddedVectorIndex(root=root, embedder=embedder)


__all__ = [
    "LocalEmbeddedVectorIndex",
    "NullVectorIndex",
    "QdrantVectorIndex",
    "VectorHit",
    "VectorIndex",
    "VectorRow",
    "get_index",
    "reset_for_tests",
]
