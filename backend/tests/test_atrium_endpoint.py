"""Atrium home-screen endpoint tests (PIVOT_ROADMAP §8.2)."""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from backend.agent.atrium import (
    Door,
    _greeting_for,
    _today_plan,
    build_atrium_payload,
)
from backend.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ─── Pure helpers ────────────────────────────────────────────────────────


def test_greeting_buckets() -> None:
    assert _greeting_for(datetime(2026, 4, 30, 8, 0)) == "guten Morgen"
    assert _greeting_for(datetime(2026, 4, 30, 12, 30)) == "Mahlzeit"
    assert _greeting_for(datetime(2026, 4, 30, 16, 0)) == "guten Nachmittag"
    assert _greeting_for(datetime(2026, 4, 30, 20, 0)) == "guten Abend"
    assert _greeting_for(datetime(2026, 4, 30, 2, 0)) == "noch wach?"


def test_today_plan_falls_back_when_no_mastery() -> None:
    plan = _today_plan(due=[], top=[])
    assert len(plan) == 2
    assert all(isinstance(s, str) and s for s in plan)


# ─── Builder ────────────────────────────────────────────────────────────


def test_build_atrium_payload_has_required_shape() -> None:
    payload = build_atrium_payload(now=datetime(2026, 4, 30, 9, 0))
    body = payload.to_dict()

    assert body["greeting"] == "guten Morgen"
    assert isinstance(body["today_plan"], list) and len(body["today_plan"]) == 2
    assert len(body["doors"]) == 4
    assert {d["id"] for d in body["doors"]} == {"reader", "voice", "capture", "atlas"}
    assert isinstance(body["mastery_top"], list)
    assert "generated_at" in body


def test_build_atrium_payload_surfaces_mastery_after_evidence() -> None:
    from backend.mastery import model as mastery_model

    mastery_model.update("verbs.modal", 0.7, source="drill")
    payload = build_atrium_payload(now=datetime(2026, 4, 30, 9, 0))
    ids = [m["competency_id"] for m in payload.to_dict()["mastery_top"]]
    assert "verbs.modal" in ids


# ─── HTTP layer ──────────────────────────────────────────────────────────


def test_get_atrium_returns_json(client: TestClient) -> None:
    res = client.get("/api/v1/agent/atrium")
    assert res.status_code == 200
    body = res.json()
    assert "greeting" in body
    assert isinstance(body["doors"], list) and len(body["doors"]) == 4
    assert "mastery_top" in body
    assert "today_plan" in body and len(body["today_plan"]) == 2


def test_get_atrium_doors_have_label_and_subtitle(client: TestClient) -> None:
    res = client.get("/api/v1/agent/atrium")
    body = res.json()
    for d in body["doors"]:
        assert {"id", "label", "subtitle"} <= set(d)
        assert isinstance(d["label"], str) and d["label"]
