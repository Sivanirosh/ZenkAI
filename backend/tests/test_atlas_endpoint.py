"""Atlas endpoint tests (PIVOT_ROADMAP §B.4)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.agent.atlas import build_atlas_payload
from backend.curriculum import planner
from backend.main import app
from backend.mastery import model as mastery_model
from backend.models import db


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _seed_goal_with_plan(
    *,
    target_cefr: str = "B2",
    domain: str = "medical",
    horizon: str = "30d",
) -> tuple[str, str]:
    parsed = {
        "domain": domain,
        "target_cefr": target_cefr,
        "deadline_iso": None,
        "scenarios": [],
        "motivations": ["test"],
        "language_profile": {"native": "en", "current_cefr": "B1"},
    }
    goal_id = "goal-atlas-1"
    db.cursor().execute(
        "INSERT INTO goals (id, raw_text, parsed) VALUES (?, ?, ?)",
        [goal_id, "raw", json.dumps(parsed)],
    )
    plan = planner.generate_plan(parsed, horizon=horizon, enrich_via_llm=False)
    plan_id = planner.persist_plan(goal_id, plan, horizon=horizon)
    return goal_id, plan_id


# ─── No plan → None / 404 ───────────────────────────────────────────────


def test_build_atlas_returns_none_without_plan() -> None:
    assert build_atlas_payload() is None


def test_get_atlas_returns_404_without_plan(client: TestClient) -> None:
    res = client.get("/api/v1/agent/atlas")
    assert res.status_code == 404
    assert res.json()["detail"] == "no_active_plan"


# ─── Payload shape ──────────────────────────────────────────────────────


def test_build_atlas_payload_has_required_shape() -> None:
    goal_id, plan_id = _seed_goal_with_plan()
    payload = build_atlas_payload()
    assert payload is not None
    body = payload.to_dict()
    assert body["goal"]["id"] == goal_id
    assert body["plan_id"] == plan_id
    assert body["horizon"] == "30d"
    assert isinstance(body["districts"], list) and body["districts"]
    assert isinstance(body["edges"], list)
    assert isinstance(body["current_route"], list) and body["current_route"]
    for d in body["districts"]:
        assert {"id", "label", "week_index", "status", "competencies"} <= set(d)
        assert d["status"] in {"mastered", "current", "queued", "locked"}


def test_get_atlas_returns_payload(client: TestClient) -> None:
    _seed_goal_with_plan()
    res = client.get("/api/v1/agent/atlas")
    assert res.status_code == 200
    body = res.json()
    assert "districts" in body and body["districts"]
    assert "current_route" in body


# ─── Status recomputation against live mastery ──────────────────────────


def test_status_marks_district_mastered_when_all_competencies_high() -> None:
    _seed_goal_with_plan(domain="daily")
    # Bring every competency in the first_steps district to mastery.
    payload = build_atlas_payload()
    assert payload is not None
    first_steps = next(d for d in payload.districts if d.id == "first_steps")
    for c in first_steps.competencies:
        for _ in range(40):
            mastery_model.update(c.competency_id, 1.0, source="drill")

    payload2 = build_atlas_payload()
    assert payload2 is not None
    fs2 = next(d for d in payload2.districts if d.id == "first_steps")
    assert fs2.status == "mastered"
    assert "first_steps" not in payload2.current_route


def test_current_route_contains_first_unfinished_week() -> None:
    _seed_goal_with_plan(domain="daily")
    payload = build_atlas_payload()
    assert payload is not None
    current_districts = [d for d in payload.districts if d.status == "current"]
    assert current_districts
    min_week = min(d.week_index for d in current_districts)
    assert all(d.week_index == min_week for d in current_districts)


def test_atlas_payload_excludes_superseded_plan(client: TestClient) -> None:
    goal_id, plan_id_a = _seed_goal_with_plan(horizon="30d")
    parsed = json.loads(
        db.cursor().execute("SELECT parsed FROM goals WHERE id = ?", [goal_id])
        .fetchone()[0]
    )
    plan2 = planner.generate_plan(parsed, horizon="60d", enrich_via_llm=False)
    plan_id_b = planner.persist_plan(goal_id, plan2, horizon="60d")
    res = client.get("/api/v1/agent/atlas")
    body = res.json()
    assert body["plan_id"] == plan_id_b
    assert body["horizon"] == "60d"
    assert body["plan_id"] != plan_id_a
