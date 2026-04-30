"""Capture service — frame on disk + memory ledger + multimodal call.

PIVOT_ROADMAP §B.6 (rooms), §B.7 (vision pipeline), §B.18 (image
on-disk under sha256 + ``captures/index.parquet``).

One ``process_capture`` call:

1. Validates the image bytes (≤ 4 MB, recognisable image header).
2. Computes a sha256 and writes the bytes to
   ``data/memory/captures/img/<sha>.<ext>``.  ``captures/img/`` is in
   the audit-trail ``.gitignore`` so big binaries never enter git.
3. Calls ``backend.multimodal.vision_capture.describe_capture`` —
   Gemma 4 vision with deterministic skeleton fallback.
4. Appends one row to ``captures/index.parquet`` (sha256, surface
   kind, file path, transcript head, captured_at, model).
5. Writes a ``capture_done`` event to the per-session JSONL via the
   MemoryManager so a turn can later cite the capture.

Returns a ``CaptureResult`` the router serialises straight to JSON.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.memory.manager import Event, get_memory
from backend.multimodal import describe_capture

logger = logging.getLogger(__name__)


_MAX_IMAGE_BYTES = 4 * 1024 * 1024  # 4 MB — generous but bounded
_SUPPORTED_EXTS: dict[str, bytes] = {
    "png": b"\x89PNG\r\n\x1a\n",
    "jpg": b"\xff\xd8\xff",
    "webp": b"RIFF",
}
_INDEX_LOCK = threading.Lock()


# ─── Public dataclasses ─────────────────────────────────────────────────


@dataclass(frozen=True)
class CaptureRequest:
    image_bytes: bytes
    content_type: str
    session_id: str
    surface_kind: str = "text"
    target_competency_id: Optional[str] = None
    target_cefr: str = "B1"
    note: str = ""


@dataclass
class CaptureResult:
    id: str
    sha256: str
    image_path: str
    surface_kind: str
    transcript: str
    words: list[dict[str, Any]] = field(default_factory=list)
    advice: str = ""
    engine: str = ""
    model: Optional[str] = None
    captured_at: str = ""
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─── Public entry point ─────────────────────────────────────────────────


async def process_capture(req: CaptureRequest) -> CaptureResult:
    """Run one Capture turn end-to-end. Never raises after validation."""
    started = datetime.now(timezone.utc)
    _validate(req.image_bytes, req.content_type)

    ext = _ext_for(req.image_bytes, req.content_type)
    sha = hashlib.sha256(req.image_bytes).hexdigest()
    image_path = _persist_image(req.image_bytes, sha=sha, ext=ext)

    vision = await describe_capture(
        req.image_bytes,
        target_cefr=req.target_cefr,
        surface_kind=req.surface_kind,
        note=req.note,
        timeout=30.0,
    )

    cap_id = f"cap-{uuid.uuid4().hex[:12]}"
    captured_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    row = {
        "id": cap_id,
        "sha256": sha,
        "session_id": req.session_id,
        "surface_kind": req.surface_kind,
        "target_competency_id": req.target_competency_id,
        "image_path": str(image_path),
        "transcript_head": (vision.transcript or "")[:160],
        "engine": vision.engine,
        "model": vision.model,
        "captured_at": captured_at,
    }
    _append_index_row(row)
    _append_meta_jsonl(row)

    mm = get_memory()
    mm.write_event(
        Event(
            session_id=req.session_id,
            kind="capture_done",
            payload={
                "capture_id": cap_id,
                "sha256": sha,
                "surface_kind": req.surface_kind,
                "engine": vision.engine,
                "transcript_head": (vision.transcript or "")[:160],
            },
        )
    )

    # B.14 — index the transcript head so explain_grammar / capture_text
    # tier-3 retrievals can semantically pull this capture later.
    if (vision.transcript or "").strip():
        try:
            from backend.memory import store_vector as _store_vector  # noqa: WPS433

            _store_vector.get_index().add(
                [
                    _store_vector.VectorRow(
                        id=cap_id,
                        text=vision.transcript,
                        source="transcripts",
                        metadata={
                            "role": "capture",
                            "session_id": req.session_id,
                            "scenario": req.surface_kind,
                            "competency_id": req.target_competency_id or "",
                            "sha256": sha,
                        },
                        ts=captured_at,
                    )
                ]
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("vector index add (capture) failed: %s", exc)

    finished = datetime.now(timezone.utc)
    duration_ms = int((finished - started).total_seconds() * 1000)
    return CaptureResult(
        id=cap_id,
        sha256=sha,
        image_path=str(image_path),
        surface_kind=req.surface_kind,
        transcript=vision.transcript,
        words=[asdict_word(w) for w in vision.words],
        advice=vision.advice,
        engine=vision.engine,
        model=vision.model,
        captured_at=captured_at,
        duration_ms=duration_ms,
    )


def list_recent_captures(*, limit: int = 20) -> list[dict[str, Any]]:
    """Tail the captures/index.parquet (newest first). Empty if no captures."""
    captures_dir = _captures_dir()
    if not captures_dir:
        return []
    parquet_path = captures_dir / "index.parquet"
    if parquet_path.exists():
        try:
            import pyarrow.parquet as pq

            rows = pq.read_table(parquet_path).to_pylist()
        except Exception as exc:
            logger.info("captures/index.parquet read failed: %s", exc)
            rows = []
    else:
        rows = []

    if not rows:
        meta_path = captures_dir / "meta.jsonl"
        if meta_path.exists():
            try:
                rows = [
                    json.loads(line)
                    for line in meta_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except OSError:
                rows = []

    rows.sort(key=lambda r: r.get("captured_at") or "", reverse=True)
    return rows[: max(0, int(limit))]


# ─── Internals ──────────────────────────────────────────────────────────


def asdict_word(w: Any) -> dict[str, Any]:
    return asdict(w) if hasattr(w, "__dataclass_fields__") else dict(w)


def _captures_dir() -> Optional[Path]:
    try:
        mm = get_memory()
        return Path(mm._root) / "captures"  # type: ignore[union-attr]
    except Exception:  # pragma: no cover
        return None


def _validate(image_bytes: bytes, content_type: str) -> None:
    if not image_bytes:
        raise ValueError("image payload is empty")
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        raise ValueError(
            f"image payload is {len(image_bytes)} bytes (max {_MAX_IMAGE_BYTES})"
        )
    ct = (content_type or "").lower().split(";")[0].strip()
    if ct and not ct.startswith("image/"):
        raise ValueError(f"unsupported content-type '{content_type}'")


def _ext_for(image_bytes: bytes, content_type: str) -> str:
    for ext, magic in _SUPPORTED_EXTS.items():
        if image_bytes.startswith(magic):
            return ext
    ct = (content_type or "").lower().split(";")[0].strip()
    if "png" in ct:
        return "png"
    if "webp" in ct:
        return "webp"
    if "jpeg" in ct or "jpg" in ct:
        return "jpg"
    return "bin"


def _persist_image(image_bytes: bytes, *, sha: str, ext: str) -> Path:
    captures_dir = _captures_dir()
    if captures_dir is None:
        raise RuntimeError("MemoryManager not available; cannot persist image")
    img_dir = captures_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    path = img_dir / f"{sha}.{ext}"
    if path.exists():
        return path
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as f:
            f.write(image_bytes)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:  # pragma: no cover
                pass
        os.replace(tmp, path)
    except OSError:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:  # pragma: no cover
            pass
        raise
    return path


def _append_index_row(row: dict[str, Any]) -> None:
    captures_dir = _captures_dir()
    if captures_dir is None:
        return
    captures_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = captures_dir / "index.parquet"
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover
        logger.info("pyarrow missing — skipping captures/index.parquet append")
        return

    with _INDEX_LOCK:
        existing: list[dict[str, Any]]
        if parquet_path.exists():
            try:
                existing = pq.read_table(parquet_path).to_pylist()
            except Exception as exc:
                logger.warning(
                    "captures/index.parquet read failed (%s); rewriting fresh",
                    exc,
                )
                existing = []
        else:
            existing = []
        existing.append(_normalise_row(row))
        try:
            table = pa.Table.from_pylist(existing)
            tmp = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
            pq.write_table(table, tmp)
            os.replace(tmp, parquet_path)
        except Exception as exc:  # pragma: no cover
            logger.warning("captures/index.parquet write failed: %s", exc)


def _append_meta_jsonl(row: dict[str, Any]) -> None:
    """Side-channel for tools that don't read parquet (Compactor, jq).

    The §7.6 Compactor reads JSONL only, not Parquet. We mirror each
    capture row to ``captures/meta.jsonl`` so the gist + archive
    phases see new rows without learning Parquet.
    """
    captures_dir = _captures_dir()
    if captures_dir is None:
        return
    captures_dir.mkdir(parents=True, exist_ok=True)
    path = captures_dir / "meta.jsonl"
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_normalise_row(row), ensure_ascii=False) + "\n")
    except OSError as exc:  # pragma: no cover
        logger.warning("captures/meta.jsonl append failed: %s", exc)


def _normalise_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "sha256": str(row.get("sha256") or ""),
        "session_id": str(row.get("session_id") or ""),
        "surface_kind": str(row.get("surface_kind") or ""),
        "target_competency_id": (
            str(row.get("target_competency_id"))
            if row.get("target_competency_id")
            else None
        ),
        "image_path": str(row.get("image_path") or ""),
        "transcript_head": str(row.get("transcript_head") or ""),
        "engine": str(row.get("engine") or ""),
        "model": (str(row.get("model")) if row.get("model") else None),
        "captured_at": str(row.get("captured_at") or ""),
    }
