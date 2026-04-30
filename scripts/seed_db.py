"""Create the DuckDB file and apply the schema. Idempotent."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_settings  # noqa: E402
from backend.models import db  # noqa: E402


def main() -> int:
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("seed_db")
    settings = get_settings()
    logger.info("Seeding DuckDB at %s", settings.duckdb_path)
    conn = db.get_connection()
    db.init_schema(conn)
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchall()
    logger.info("Tables present: %s", ", ".join(t[0] for t in tables))
    db.close_connection()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
