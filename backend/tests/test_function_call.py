"""tools.dispatch contract: validation, unknown-tool rejection, error shape."""

from __future__ import annotations

import pytest

from backend.agent import tools
from backend.models import db


def test_dispatch_rejects_unknown_tool() -> None:
    with pytest.raises(tools.ToolError) as excinfo:
        tools.dispatch("does_not_exist", {})
    err = excinfo.value
    assert err.code == "unknown_tool"
    assert "available" in err.detail


def test_dispatch_validates_args_via_pydantic() -> None:
    # update_mastery requires a delta in [-1, 1].
    with pytest.raises(tools.ToolError) as excinfo:
        tools.dispatch("update_mastery", {"competency_id": "verbs.modal", "delta": 2.0})
    err = excinfo.value
    assert err.code == "invalid_args"
    assert any("delta" in str(e.get("loc", "")) for e in err.detail["errors"])


def test_dispatch_update_mastery_persists_evidence_and_returns_summary() -> None:
    out = tools.dispatch(
        "update_mastery",
        {
            "competency_id": "verbs.modal",
            "delta": 1.0,
            "source": "agent_post",
            "surface_form": "ich kann gehen",
        },
    )
    assert out["competency_id"] == "verbs.modal"
    assert 0 <= out["confidence"] <= 100
    assert out["evidence_count"] == 1

    row = db.cursor().execute(
        "SELECT competency_id, source, surface_form FROM evidence"
    ).fetchone()
    assert row[0] == "verbs.modal"
    assert row[1] == "agent_post"
    assert row[2] == "ich kann gehen"


def test_dispatch_update_mastery_rejects_unknown_source() -> None:
    with pytest.raises(tools.ToolError) as excinfo:
        tools.dispatch(
            "update_mastery",
            {"competency_id": "verbs.modal", "delta": 0.5, "source": "telepathy"},
        )
    # bad_source can be caught by the Pydantic enum (via TOOL_SCHEMAS) OR
    # by our explicit check; both are acceptable.
    assert excinfo.value.code in {"bad_source", "invalid_args"}


def test_dispatch_recommend_text_returns_no_paragraph_when_corpus_empty() -> None:
    with pytest.raises(tools.ToolError) as excinfo:
        tools.dispatch("recommend_text", {"reason": "warm-up"})
    assert excinfo.value.code == "no_paragraph"


def test_dispatch_recommend_text_returns_first_paragraph_for_seeded_work() -> None:
    cur = db.cursor()
    cur.execute(
        "INSERT INTO works (id, title, author) VALUES (?, ?, ?)",
        ["w1", "Verwandlung", "Kafka"],
    )
    cur.execute(
        "INSERT INTO paragraphs (id, work_id, chapter, position, text, word_count) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ["p1", "w1", 1, 1, "Als Gregor Samsa eines Morgens...", 5],
    )
    cur.execute(
        "INSERT INTO paragraphs (id, work_id, chapter, position, text, word_count) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ["p2", "w1", 1, 2, "Was ist mit mir geschehen?", 4],
    )

    out = tools.dispatch("recommend_text", {"reason": "start"})
    assert out["paragraph_id"] == "p1"
    assert out["work_id"] == "w1"
    assert out["why"] == "start"

    out2 = tools.dispatch(
        "recommend_text",
        {"current_paragraph_id": "p1", "reason": "continuation"},
    )
    assert out2["paragraph_id"] == "p2"


def test_dispatch_start_drill_returns_stub_plan() -> None:
    out = tools.dispatch(
        "start_drill",
        {"competency_id": "medical.symptoms", "count": 5},
    )
    assert out["competency_id"] == "medical.symptoms"
    assert out["count"] == 5
    assert "current_confidence" in out
    assert out["plan_kind"] == "stub"


def test_tool_schemas_have_all_handlers_registered() -> None:
    # Belt-and-suspenders: every TOOL_SCHEMAS entry has a matching _HANDLERS row.
    schema_names = {t.name for t in tools.TOOL_SCHEMAS}
    handler_names = set(tools._HANDLERS)
    assert schema_names == handler_names
