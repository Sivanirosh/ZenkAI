"""Chat SSE framing tests."""

from __future__ import annotations

from typing import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.services import llm_service


async def _fake_stream(*_args, **_kwargs) -> AsyncIterator[str]:
    for chunk in ["Hallo ", "Welt."]:
        yield chunk


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setattr(llm_service, "stream_answer", _fake_stream)
    return TestClient(create_app())


def test_chat_message_emits_token_then_done(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/api/v1/chat/message",
        json={"question": "Warum Ungeziefer?", "paragraph_id": None},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = b"".join(response.iter_raw()).decode("utf-8")

    assert "event: token" in body
    assert '"delta": "Hallo "' in body
    assert '"delta": "Welt."' in body
    assert body.rstrip().endswith("{}")
    assert "event: done" in body


def test_chat_message_rejects_empty_question(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/api/v1/chat/message",
        json={"question": "   ", "paragraph_id": None},
    ) as response:
        assert response.status_code == 200
        body = b"".join(response.iter_raw()).decode("utf-8")

    assert "event: error" in body
    assert "must not be empty" in body
    assert "event: done" in body
