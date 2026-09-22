"""Static frontend serving.

The single-page app lives in `static/` and is mounted at `/` by FastAPI
(same origin — no CORS, no separate frontend server).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import create_app


def test_index_served() -> None:
    client = TestClient(create_app())
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    assert "ZenkAI" in res.text


def test_assets_served() -> None:
    client = TestClient(create_app())
    for path, marker in (
        ("/app.js", "Leseraum"),
        ("/styles.css", "--amber-50"),
    ):
        res = client.get(path)
        assert res.status_code == 200, path
        assert marker in res.text, path


def test_api_routes_not_shadowed() -> None:
    client = TestClient(create_app())
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/api/v1/corpus/works").status_code == 200
