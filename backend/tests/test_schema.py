"""Schema-shape sanity tests.

Cheap assertions that ``init_schema()`` actually creates the tables /
columns the rest of the codebase relies on. Catches regressions where
someone deletes a `CREATE TABLE` block in `schema.sql` without noticing.
"""

from __future__ import annotations

from backend.models import db


def _existing_tables() -> set[str]:
    rows = db.cursor().execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main'"
    ).fetchall()
    return {r[0] for r in rows}


def _columns_of(table: str) -> set[str]:
    rows = db.cursor().execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'main' AND table_name = ?",
        [table],
    ).fetchall()
    return {r[0] for r in rows}


def test_phase_a_tables_all_present() -> None:
    tables = _existing_tables()
    expected = {
        "works",
        "paragraphs",
        "words",
        "word_occurrences",
        "word_states",
        "sessions",
        "vocab_queue",
        "goals",
        "competencies",
        "mastery",
        "evidence",
        "atlas_plans",
    }
    missing = expected - tables
    assert not missing, f"missing tables: {sorted(missing)}"


def test_atlas_plans_has_planner_columns() -> None:
    cols = _columns_of("atlas_plans")
    assert {"id", "goal_id", "horizon", "plan", "created_at", "superseded_by"} <= cols


def test_mastery_has_posterior_columns() -> None:
    cols = _columns_of("mastery")
    assert {
        "competency_id",
        "mu",
        "sigma",
        "last_evidence",
        "evidence_count",
    } <= cols


def test_atlas_plans_accepts_a_planner_row() -> None:
    cur = db.cursor()
    cur.execute(
        "INSERT INTO goals (id, raw_text, parsed) VALUES (?, ?, ?)",
        ["goal-1", "Become B2 medical", '{"target_cefr":"B2"}'],
    )
    cur.execute(
        "INSERT INTO atlas_plans (id, goal_id, horizon, plan) "
        "VALUES (?, ?, ?, ?)",
        ["plan-1", "goal-1", "30d", '{"weeks":4,"districts":[]}'],
    )
    row = cur.execute(
        "SELECT id, goal_id, horizon FROM atlas_plans WHERE id = 'plan-1'"
    ).fetchone()
    assert row == ("plan-1", "goal-1", "30d")
