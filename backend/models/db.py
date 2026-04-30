"""DuckDB connection management for Lesekamerad.

FastAPI dispatches sync endpoints on a threadpool, so multiple requests may
hit the database concurrently. A bare `DuckDBPyConnection` is NOT safe to
share across threads — doing so produces sporadic `TypeError: 'NoneType' is
not subscriptable` and phantom 404s because the cursor state of one query
gets clobbered by another.

DuckDB's officially supported multi-thread pattern is to open one connection
per process and obtain a fresh `cursor()` per logical operation. Every
service in this codebase uses `db.cursor()` for its queries.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import duckdb

from backend.config import get_settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_connection: Optional[duckdb.DuckDBPyConnection] = None

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return the process-wide DuckDB connection, creating it lazily.

    Prefer :func:`cursor` for all query execution in request handlers.
    """
    global _connection
    if _connection is not None:
        return _connection

    with _lock:
        if _connection is None:
            settings = get_settings()
            db_path = Path(settings.duckdb_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Opening DuckDB at %s", db_path)
            _connection = duckdb.connect(str(db_path))
    return _connection


def cursor() -> duckdb.DuckDBPyConnection:
    """Return a fresh per-call cursor safe to use from any thread.

    DuckDB cursors share the underlying database but carry their own
    result/iteration state, which is what makes concurrent access safe.
    """
    return get_connection().cursor()


def init_schema(conn: Optional[duckdb.DuckDBPyConnection] = None) -> None:
    """Apply `schema.sql` idempotently to the given (or default) connection."""
    target = conn if conn is not None else get_connection()
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    target.execute(sql)
    logger.info("Schema applied from %s", _SCHEMA_PATH)


def reset_for_tests() -> duckdb.DuckDBPyConnection:
    """Swap the singleton for an in-memory connection. Tests only."""
    global _connection
    with _lock:
        _connection = duckdb.connect(":memory:")
        init_schema(_connection)
    return _connection


def close_connection() -> None:
    """Close and clear the singleton connection (used at shutdown / tests)."""
    global _connection
    with _lock:
        if _connection is not None:
            _connection.close()
            _connection = None
