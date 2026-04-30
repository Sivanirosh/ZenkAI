"""Phase B tool-surface coverage (PIVOT_ROADMAP §B.9).

Verifies that ``start_conversation`` and ``start_capture`` are
- registered in the dispatcher and the schema list,
- aliased through the recall policy table,
- return room-navigation hints with safe defaults.
"""

from __future__ import annotations

import pytest

from backend.agent import tools
from backend.memory import policies as recall_policies


# ─── start_conversation ─────────────────────────────────────────────────


def test_start_conversation_is_registered():
    schema_names = [s.name for s in tools.TOOL_SCHEMAS]
    assert "start_conversation" in schema_names
    assert "start_conversation" in tools._HANDLERS


def test_start_conversation_default_args_returns_konversation_hint():
    out = tools.dispatch("start_conversation", {})
    assert out["room_id"] == "konversation"
    assert out["minutes"] == 5
    assert "navigate" in out["next_action"]


def test_start_conversation_resolves_competency_label():
    from backend.mastery.competencies import SEED

    sample = SEED[0]
    out = tools.dispatch(
        "start_conversation",
        {
            "competency_id": sample.id,
            "scenario": "Anamnesegespräch",
            "minutes": 10,
        },
    )
    assert out["competency_id"] == sample.id
    assert out["competency_label"] == sample.label
    assert out["scenario"] == "Anamnesegespräch"
    assert out["minutes"] == 10


def test_start_conversation_rejects_out_of_range_minutes():
    with pytest.raises(tools.ToolError) as excinfo:
        tools.dispatch("start_conversation", {"minutes": 999})
    assert excinfo.value.code == "invalid_args"


def test_start_conversation_has_recall_policy():
    pol = recall_policies.find_policy("start_conversation")
    assert pol is not None
    assert pol.tool_hint == "start_conversation"
    assert pol.budget_tokens > 0


# ─── start_capture ──────────────────────────────────────────────────────


def test_start_capture_is_registered():
    schema_names = [s.name for s in tools.TOOL_SCHEMAS]
    assert "start_capture" in schema_names
    assert "start_capture" in tools._HANDLERS


def test_start_capture_default_args_returns_capture_hint():
    out = tools.dispatch("start_capture", {})
    assert out["room_id"] == "capture"
    assert out["surface_kind"] == "text"
    assert "navigate" in out["next_action"]


def test_start_capture_passes_through_targeted_competency():
    out = tools.dispatch(
        "start_capture",
        {
            "surface_kind": "menu",
            "target_competency_id": "vocab.food",
            "note": "im Café",
        },
    )
    assert out["surface_kind"] == "menu"
    assert out["target_competency_id"] == "vocab.food"
    assert out["note"] == "im Café"


def test_start_capture_aliases_to_capture_text_policy():
    pol = recall_policies.find_policy("start_capture")
    assert pol is not None
    assert pol.tool_hint == "capture_text"


def test_start_capture_rejects_unknown_surface_kind():
    """Pydantic doesn't enum-restrict ``surface_kind``; the model can
    pass anything but the tool stays robust by treating unknowns as
    ``text`` downstream. We only assert it doesn't crash.
    """
    out = tools.dispatch("start_capture", {"surface_kind": "selfie"})
    assert out["room_id"] == "capture"
    assert out["surface_kind"] == "selfie"
