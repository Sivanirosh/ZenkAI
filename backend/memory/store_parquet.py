"""Atomic Parquet writers (PIVOT_ROADMAP §A.12).

Two surfaces:

- ``snapshot_mastery(dest)`` — read the entire ``mastery`` table, write
  a Parquet snapshot. Called from ``MemoryManager.snapshot_mastery``
  and ``MemoryManager.close_session``.

- ``append_session_index_row(dest, row)`` — append one row to the
  rolling ``sessions/index.parquet`` file. Called from
  ``MemoryManager.close_session``.

Both writers are atomic in the dest-file sense: we write to a sibling
``*.tmp`` then ``os.replace`` it onto the destination. If the writer
crashes mid-write the destination file is left untouched (the
``*.tmp`` becomes orphan garbage; tests assert there's no leftover).

We keep ``pyarrow`` import at function scope so the module imports even
on a freshly cloned environment that hasn't run ``pip install`` yet.
DuckDB pulls pyarrow in as a transitive dependency, so production hits
never hit the ImportError branch.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from backend.models import db

logger = logging.getLogger(__name__)


def _atomic_write_table(table: Any, dest: Path) -> Path:
    """Write ``table`` to ``dest`` atomically via a sibling .tmp file.

    Re-raises whatever the writer raised (so test injection of a fault
    surfaces cleanly), but always cleans up the tmp file before
    re-raising so the dest stays untouched and we leave no garbage.
    """
    import pyarrow.parquet as pq

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        pq.write_table(table, tmp)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:  # pragma: no cover
            pass
        raise
    os.replace(tmp, dest)
    return dest


def snapshot_mastery(dest: Path) -> Optional[Path]:
    """Write a Parquet snapshot of the ``mastery`` table to ``dest``.

    Returns the dest path on success, ``None`` when the table has no
    rows (we deliberately don't write a zero-row file — that confuses
    downstream readers; the absence of the file means "nothing to snapshot").
    """
    try:
        import pyarrow as pa
    except ImportError:  # pragma: no cover
        logger.info("pyarrow missing — skipping mastery snapshot")
        return None

    rows = db.cursor().execute(
        """
        SELECT competency_id, mu, sigma, last_evidence,
               evidence_count, next_review_at
        FROM mastery
        """
    ).fetchall()
    if not rows:
        return None

    table = pa.table(
        {
            "competency_id":   [r[0] for r in rows],
            "mu":              [float(r[1]) for r in rows],
            "sigma":           [float(r[2]) for r in rows],
            "last_evidence":   [_to_iso(r[3]) for r in rows],
            "evidence_count":  [int(r[4] or 0) for r in rows],
            "next_review_at":  [_to_iso(r[5]) for r in rows],
        }
    )
    return _atomic_write_table(table, Path(dest))


def append_session_index_row(dest: Path, row: dict[str, Any]) -> Path:
    """Read existing parquet (if any), append the row, atomic rewrite.

    Cost is fine for a handful of sessions per day; if this ever
    becomes hot we'd switch to a row-group append.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    dest = Path(dest)
    if dest.exists():
        existing = pq.read_table(dest).to_pylist()
    else:
        existing = []
    existing.append(_normalise_index_row(row))
    table = pa.Table.from_pylist(existing)
    return _atomic_write_table(table, dest)


def _normalise_index_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": str(row.get("session_id", "")),
        "closed_at": _to_iso(row.get("closed_at")) or "",
        "files": list(row.get("files") or []),
    }


def _to_iso(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v)
