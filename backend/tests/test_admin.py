"""Admin endpoints: verify + patch runtime LLM options."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.services.runtime_config import get_llm_options, set_llm_options


def _patch_ollama(monkeypatch, *, reachable=True, models=None, error=None):
    async def _fake_probe(**_kw):
        return {
            "reachable": reachable,
            "models": list(models or []),
            "configured": get_llm_options().model,
            "configured_available": get_llm_options().model in (models or []),
            "error": error,
        }

    monkeypatch.setattr("backend.routers.admin.llm_service.probe_ollama", _fake_probe)


def test_get_ollama_status_returns_options_and_probe(monkeypatch):
    set_llm_options(model="qwen3.5:latest", think=False, temperature=0.4)
    _patch_ollama(monkeypatch, reachable=True, models=["qwen3.5:latest", "llama3:8b"])

    client = TestClient(create_app())
    res = client.get("/api/v1/admin/ollama")

    assert res.status_code == 200
    data = res.json()
    assert data["options"] == {
        "model": "qwen3.5:latest",
        "think": False,
        "temperature": 0.4,
    }
    assert data["ollama"]["reachable"] is True
    assert "llama3:8b" in data["ollama"]["models"]


def test_put_ollama_updates_runtime_options(monkeypatch):
    set_llm_options(model="qwen3.5:latest", think=False, temperature=0.3)
    _patch_ollama(monkeypatch, reachable=True, models=["qwen3.5:latest", "llama3:8b"])

    client = TestClient(create_app())
    res = client.put(
        "/api/v1/admin/ollama",
        json={"model": "llama3:8b", "think": True, "temperature": 0.9},
    )

    assert res.status_code == 200, res.text
    assert res.json()["options"] == {
        "model": "llama3:8b",
        "think": True,
        "temperature": 0.9,
    }
    current = get_llm_options()
    assert current.model == "llama3:8b"
    assert current.think is True
    assert current.temperature == pytest.approx(0.9)


def test_put_ollama_rejects_unknown_model(monkeypatch):
    set_llm_options(model="qwen3.5:latest")
    _patch_ollama(monkeypatch, reachable=True, models=["qwen3.5:latest"])

    client = TestClient(create_app())
    res = client.put(
        "/api/v1/admin/ollama",
        json={"model": "ghost:1b"},
    )

    assert res.status_code == 400
    assert "ghost:1b" in res.json()["detail"]
    assert get_llm_options().model == "qwen3.5:latest"


def test_put_ollama_clamps_temperature(monkeypatch):
    set_llm_options(model="qwen3.5:latest")
    _patch_ollama(monkeypatch, reachable=True, models=["qwen3.5:latest"])

    client = TestClient(create_app())
    res = client.put("/api/v1/admin/ollama", json={"temperature": 5.0})

    assert res.status_code == 422  # pydantic ge/le validation
