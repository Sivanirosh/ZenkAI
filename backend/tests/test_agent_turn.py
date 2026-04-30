"""Agent turn integration test against a deterministic FakeProvider.

Proves end-to-end:
  - controller.turn dispatches each ``ProviderEvent.tool_call`` through
    ``tools.dispatch`` and yields the right ``AgentEvent`` sequence
  - the MemoryManager sees a turn_open / tool_intent / tool_result /
    turn_close ledger written to JSONL
  - max_tool_calls_per_turn is enforced
  - SSE serialisation matches the Outcome example in the plan
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Iterable, Optional

import pytest

from backend.agent.controller import (
    AgentEvent,
    AgentEventKind,
    TurnRequest,
    turn,
)
from backend.llm.provider import ProbeReport, ProviderEvent
from backend.memory import MemoryManager
from backend.models import db
from backend.services.runtime_config import LlmOptions


# ─── A scriptable LLMProvider stand-in ───────────────────────────────────


class FakeProvider:
    """Yields a fixed list of provider event lists (one per outer-loop call).

    First call = ``script[0]``, second call = ``script[1]``, ... When the
    script runs out we yield an empty stream + done so the controller exits.
    """

    name = "fake"

    def __init__(self, script: list[list[ProviderEvent]]) -> None:
        self._script = list(script)
        self.calls: list[list[dict]] = []

    async def generate(self, *_a, **_kw) -> Optional[str]:  # pragma: no cover
        return None

    async def stream_chat(self, *_a, **_kw) -> AsyncIterator[str]:  # pragma: no cover
        if False:
            yield ""

    async def chat_with_tools(
        self, messages, tools, *, options, timeout=60.0
    ) -> AsyncIterator[ProviderEvent]:
        self.calls.append(list(messages))
        events: Iterable[ProviderEvent] = (
            self._script.pop(0) if self._script else []
        )
        for ev in events:
            yield ev
        yield ProviderEvent.done()

    async def list_models(self, **_kw) -> list[str]:
        return ["fake:1b"]

    async def probe(self, **_kw) -> ProbeReport:
        return ProbeReport(
            reachable=True,
            models=["fake:1b"],
            configured="fake:1b",
            configured_available=True,
        )


def _seed_corpus() -> None:
    cur = db.cursor()
    cur.execute(
        "INSERT INTO works (id, title, author) VALUES (?, ?, ?)",
        ["w1", "Verwandlung", "Kafka"],
    )
    cur.execute(
        "INSERT INTO paragraphs (id, work_id, chapter, position, text, word_count) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ["p1", "w1", 1, 1, "Als Gregor Samsa erwachte ...", 5],
    )
    cur.execute(
        "INSERT INTO paragraphs (id, work_id, chapter, position, text, word_count) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ["p2", "w1", 1, 2, "Was ist mit mir geschehen?", 4],
    )


async def _collect(gen) -> list[AgentEvent]:
    out: list[AgentEvent] = []
    async for ev in gen:
        out.append(ev)
    return out


# ─── Tests ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_turn_dispatches_recommend_then_update_then_text(tmp_path):
    _seed_corpus()
    script = [
        [
            ProviderEvent.text_chunk(
                "Ich schlage vor, wir lesen 5 Minuten weiter."
            ),
            ProviderEvent.tool_call(
                "recommend_text",
                {"current_paragraph_id": "p1", "reason": "continuation"},
                call_id="c1",
            ),
        ],
        [
            ProviderEvent.tool_call(
                "update_mastery",
                {"competency_id": "verbs.modal", "delta": 0.0, "source": "agent_pre"},
                call_id="c2",
            ),
        ],
        [
            ProviderEvent.text_chunk("Bereit, wenn du bist."),
        ],
    ]
    provider = FakeProvider(script)
    memory = MemoryManager(root=tmp_path / "mem")

    request = TurnRequest(
        current_screen="reader",
        time_budget_min=12,
        current_paragraph_id="p1",
        session_id="sess-test",
    )

    events = await _collect(turn(request, provider=provider, memory=memory))
    kinds = [e.kind for e in events]

    assert AgentEventKind.THOUGHT in kinds
    intents = [e for e in events if e.kind == AgentEventKind.TOOL_INTENT]
    results = [e for e in events if e.kind == AgentEventKind.TOOL_RESULT]
    assert [i.data["name"] for i in intents] == ["recommend_text", "update_mastery"]
    assert results[0].data["paragraph_id"] == "p2"
    assert results[1].data["competency_id"] == "verbs.modal"
    assert kinds[-1] == AgentEventKind.DONE

    # Ledger is written and durable. Files are rotated; use the helper.
    files = memory.list_session_files("sess-test")
    assert files, "expected at least one rotated JSONL file"
    rows = memory.read_session_events("sess-test")
    kinds_logged = [r["kind"] for r in rows]
    assert "turn_open" in kinds_logged
    assert kinds_logged.count("tool_intent") == 0  # we don't log intents (only results)
    assert kinds_logged.count("tool_result") == 2
    assert kinds_logged[-1] == "turn_close"

    # Recall trace is enriched (no stub:true once the policy fires).
    trace_path = tmp_path / "mem" / "recall_trace.jsonl"
    assert trace_path.exists()
    trace_rows = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]
    assert trace_rows, "expected at least one recall_trace row"
    last = trace_rows[-1]
    assert last["policy"] == "open_turn"
    assert "stub" not in last
    assert "latency_ms" in last
    assert "tier1_sections" in last and "tier2_sections" in last and "tier3_sections" in last


@pytest.mark.asyncio
async def test_turn_caps_at_max_tool_calls_per_turn(tmp_path, monkeypatch):
    _seed_corpus()
    monkeypatch.setattr("backend.agent.policies.MAX_TOOL_CALLS_PER_TURN", 2)
    monkeypatch.setattr(
        "backend.agent.controller.policies.MAX_TOOL_CALLS_PER_TURN", 2
    )

    same_call = ProviderEvent.tool_call(
        "recommend_text", {"reason": "loop"}, call_id="loop"
    )
    script = [
        [same_call, same_call, same_call],  # three in one round
    ]
    provider = FakeProvider(script)
    memory = MemoryManager(root=tmp_path / "mem")

    events = await _collect(
        turn(TurnRequest(session_id="sess-cap"), provider=provider, memory=memory)
    )
    intents = [e for e in events if e.kind == AgentEventKind.TOOL_INTENT]
    errors = [e for e in events if e.kind == AgentEventKind.ERROR]

    # We should have dispatched exactly 2 (the cap), then errored before #3.
    assert len(intents) == 2
    assert any(err.data.get("message") == "max_tool_calls_per_turn" for err in errors)
    # And exactly one done at the very end.
    assert events[-1].kind == AgentEventKind.DONE


@pytest.mark.asyncio
async def test_turn_passes_tool_results_back_into_messages(tmp_path):
    _seed_corpus()
    script = [
        [ProviderEvent.tool_call("recommend_text", {"reason": "first"})],
        [ProviderEvent.text_chunk("Alles klar.")],
    ]
    provider = FakeProvider(script)
    memory = MemoryManager(root=tmp_path / "mem")

    await _collect(
        turn(TurnRequest(session_id="sess-msg"), provider=provider, memory=memory)
    )

    # The second model call must have seen the tool result appended.
    assert len(provider.calls) == 2
    second_call = provider.calls[1]
    tool_messages = [m for m in second_call if m["role"] == "tool"]
    assert len(tool_messages) == 1
    parsed = json.loads(tool_messages[0]["content"])
    assert "paragraph_id" in parsed


@pytest.mark.asyncio
async def test_turn_handles_invalid_tool_args_gracefully(tmp_path):
    _seed_corpus()
    script = [
        [
            ProviderEvent.tool_call(
                "update_mastery",
                {"competency_id": "verbs.modal", "delta": 99.0},  # invalid range
            )
        ],
        [ProviderEvent.text_chunk("Ich versuche es anders.")],
    ]
    provider = FakeProvider(script)
    memory = MemoryManager(root=tmp_path / "mem")

    events = await _collect(
        turn(TurnRequest(session_id="sess-bad"), provider=provider, memory=memory)
    )
    errors = [e for e in events if e.kind == AgentEventKind.TOOL_ERROR]
    assert errors and errors[0].data["code"] == "invalid_args"
    # Turn still completes cleanly.
    assert events[-1].kind == AgentEventKind.DONE


@pytest.mark.asyncio
async def test_turn_aclose_after_done_does_not_raise(tmp_path):
    """Regression: dev logs were spammed with
    ``RuntimeError: async generator ignored GeneratorExit`` after every
    ``POST /api/v1/agent/turn``. Root cause: the inner ``chat_with_tools``
    suspended after its terminal yield, then GC-time ``aclose()`` re-entered
    a yielding ``finally``. We now emit DONE outside ``finally`` and
    cooperatively close the inner generator. Prove both paths are clean by
    iterating until DONE then aclose()-ing the controller's generator.
    """
    _seed_corpus()
    script = [[ProviderEvent.text_chunk("Bereit.")]]
    provider = FakeProvider(script)
    memory = MemoryManager(root=tmp_path / "mem")

    gen = turn(
        TurnRequest(session_id="sess-aclose"),
        provider=provider,
        memory=memory,
    )
    seen: list[AgentEventKind] = []
    async for ev in gen:
        seen.append(ev.kind)
        if ev.kind == AgentEventKind.DONE:
            break

    assert seen[-1] == AgentEventKind.DONE
    # Must NOT raise RuntimeError("async generator ignored GeneratorExit").
    await gen.aclose()
    await gen.aclose()  # idempotent


@pytest.mark.asyncio
async def test_turn_aclose_midstream_does_not_raise(tmp_path):
    """Same regression but for the harder client-disconnect case: drop the
    iteration BEFORE the generator finishes (mimics undici / browser fetch
    aborting halfway). The controller must unwind cleanly."""
    _seed_corpus()
    script = [
        [
            ProviderEvent.text_chunk("Erste Zeile."),
            ProviderEvent.tool_call(
                "recommend_text",
                {"current_paragraph_id": "p1", "reason": "continuation"},
            ),
        ],
        [ProviderEvent.text_chunk("Zweite Zeile.")],
    ]
    provider = FakeProvider(script)
    memory = MemoryManager(root=tmp_path / "mem")
    gen = turn(
        TurnRequest(session_id="sess-disco", current_paragraph_id="p1"),
        provider=provider,
        memory=memory,
    )

    # Pull only the very first event then cancel — equivalent to the client
    # closing the SSE socket mid-stream.
    first = await gen.__anext__()
    assert first.kind in {
        AgentEventKind.THOUGHT,
        AgentEventKind.TOOL_INTENT,
        AgentEventKind.TOOL_RESULT,
        AgentEventKind.DONE,
    }
    await gen.aclose()
    await gen.aclose()


def test_agent_event_to_sse_format() -> None:
    ev = AgentEvent(
        kind=AgentEventKind.TOOL_INTENT,
        data={"name": "recommend_text", "args": {"reason": "x"}},
    )
    sse = ev.to_sse()
    # Exactly two lines + trailing blank line.
    assert sse.startswith("event: tool_intent\n")
    assert "\ndata: " in sse
    assert sse.endswith("\n\n")
    payload = sse.split("data: ", 1)[1].split("\n\n", 1)[0]
    assert json.loads(payload) == {
        "name": "recommend_text",
        "args": {"reason": "x"},
    }


def test_thought_event_serialises_string_payload() -> None:
    ev = AgentEvent(kind=AgentEventKind.THOUGHT, data="Hallo Welt")
    sse = ev.to_sse()
    payload = sse.split("data: ", 1)[1].split("\n\n", 1)[0]
    assert json.loads(payload) == "Hallo Welt"
