"""Curriculum planner tests (PIVOT_ROADMAP §B.2)."""

from __future__ import annotations

import json

import pytest

from backend.curriculum import planner
from backend.mastery import competencies as competencies_seed
from backend.models import db


def _medical_goal() -> dict:
    return {
        "domain": "medical",
        "deadline_iso": None,
        "target_cefr": "B2",
        "scenarios": ["Anamnese mit Patientin", "Aufklärungsgespräch"],
        "motivations": ["FSP-Vorbereitung"],
        "language_profile": {"native": "fr", "current_cefr": "B1"},
    }


def _seed_goal_row(goal_id: str = "goal-test") -> str:
    db.cursor().execute(
        "INSERT INTO goals (id, raw_text, parsed) VALUES (?, ?, ?)",
        [goal_id, "raw", json.dumps(_medical_goal())],
    )
    return goal_id


# ─── Skeleton (deterministic path, LLM disabled) ────────────────────────


def test_skeleton_only_yields_a_valid_plan() -> None:
    plan = planner.generate_plan(_medical_goal(), enrich_via_llm=False)
    assert plan["horizon"] == "30d"
    assert len(plan["weeks"]) == 4
    assert len(plan["districts"]) >= 3

    valid_ids = {c.id for c in competencies_seed.SEED}
    for d in plan["districts"]:
        assert all(c in valid_ids for c in d["competencies"])
        assert d["status"] in {"mastered", "current", "queued", "locked"}
        assert isinstance(d["week_index"], int) and 1 <= d["week_index"] <= 4

    for w in plan["weeks"]:
        assert all(c in valid_ids for c in w["competencies"])
        assert isinstance(w["doors"], list) and w["doors"]
        assert isinstance(w["title"], str) and w["title"]


def test_skeleton_uses_default_horizon_when_invalid() -> None:
    plan = planner.generate_plan(_medical_goal(), horizon="lol", enrich_via_llm=False)
    assert plan["horizon"] == "30d"


def test_skeleton_picks_other_cluster_for_unknown_domain() -> None:
    plan = planner.generate_plan(
        {**_medical_goal(), "domain": "skydiving"},
        enrich_via_llm=False,
    )
    district_ids = {d["id"] for d in plan["districts"]}
    # The "other" cluster always includes first_steps + core_grammar.
    assert "first_steps" in district_ids
    assert "core_grammar" in district_ids


# ─── Validation: drops invalid competency IDs ───────────────────────────


def test_validation_drops_unknown_competency_ids() -> None:
    plan = planner.generate_plan(_medical_goal(), enrich_via_llm=False)
    plan["districts"].append(
        {
            "id": "phantom",
            "label": "Phantom",
            "competencies": ["nonexistent.id", "also.fake"],
            "prerequisites": [],
            "week_index": 1,
            "status": "queued",
        }
    )
    valid_ids = {c.id for c in competencies_seed.SEED}
    cleaned = planner._validate(plan, valid_ids, weeks_total=4, horizon="30d")
    assert all(d["id"] != "phantom" for d in cleaned["districts"])


def test_validation_keeps_only_known_edges() -> None:
    plan = planner.generate_plan(_medical_goal(), enrich_via_llm=False)
    valid_ids = {c.id for c in competencies_seed.SEED}
    plan["edges"] = [
        ["does_not_exist", "anatomy"],
        ["anatomy", "diagnostics"],
    ]
    cleaned = planner._validate(plan, valid_ids, weeks_total=4, horizon="30d")
    # The "does_not_exist" edge gets dropped; the valid one survives if both
    # endpoints are real district ids.
    district_ids = {d["id"] for d in cleaned["districts"]}
    for a, b in cleaned["edges"]:
        assert a in district_ids and b in district_ids


# ─── Persistence ────────────────────────────────────────────────────────


def test_persist_plan_round_trips() -> None:
    goal_id = _seed_goal_row()
    plan = planner.generate_plan(_medical_goal(), enrich_via_llm=False)
    plan_id = planner.persist_plan(goal_id, plan, horizon="30d")
    assert plan_id.startswith("plan-")

    fetched = planner.latest_plan_for_goal(goal_id)
    assert fetched is not None
    fetched_id, fetched_plan, horizon = fetched
    assert fetched_id == plan_id
    assert horizon == "30d"
    assert fetched_plan["horizon"] == "30d"
    assert len(fetched_plan["districts"]) == len(plan["districts"])


def test_persist_plan_supersedes_previous_plan() -> None:
    goal_id = _seed_goal_row()
    plan1 = planner.generate_plan(_medical_goal(), enrich_via_llm=False)
    id1 = planner.persist_plan(goal_id, plan1, horizon="30d")

    plan2 = planner.generate_plan(
        {**_medical_goal(), "target_cefr": "C1"}, enrich_via_llm=False
    )
    id2 = planner.persist_plan(goal_id, plan2, horizon="60d")

    assert id1 != id2
    fetched = planner.latest_plan_for_goal(goal_id)
    assert fetched is not None
    assert fetched[0] == id2

    sup = db.cursor().execute(
        "SELECT superseded_by FROM atlas_plans WHERE id = ?", [id1]
    ).fetchone()
    assert sup[0] == id2


def test_latest_plan_for_goal_returns_none_when_missing() -> None:
    assert planner.latest_plan_for_goal("does-not-exist") is None


# ─── Tool wiring ─────────────────────────────────────────────────────────


def test_plan_curriculum_tool_persists_a_row() -> None:
    from backend.agent import tools

    goal_id = _seed_goal_row()
    out = tools.dispatch(
        "plan_curriculum",
        {"goal_id": goal_id, "horizon": "30d"},
    )
    assert out["plan_id"].startswith("plan-")
    assert out["week_count"] >= 1
    assert out["district_count"] >= 1


def test_plan_curriculum_tool_rejects_unknown_goal() -> None:
    from backend.agent import tools

    with pytest.raises(tools.ToolError) as excinfo:
        tools.dispatch("plan_curriculum", {"goal_id": "ghost"})
    assert excinfo.value.code == "goal_not_found"


def test_replan_atlas_tool_supersedes_previous() -> None:
    from backend.agent import tools

    goal_id = _seed_goal_row()
    first = tools.dispatch(
        "plan_curriculum", {"goal_id": goal_id, "horizon": "30d"}
    )
    second = tools.dispatch(
        "replan_atlas", {"goal_id": goal_id, "horizon": "60d"}
    )
    assert first["plan_id"] != second["plan_id"]
    sup = db.cursor().execute(
        "SELECT superseded_by FROM atlas_plans WHERE id = ?", [first["plan_id"]]
    ).fetchone()
    assert sup[0] == second["plan_id"]
