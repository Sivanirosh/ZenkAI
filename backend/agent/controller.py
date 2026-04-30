"""Agent turn controller — Observe → Decide → Act loop.

This is the loop sketched in PIVOT_ROADMAP.md §6.3. ``turn()`` is an async
generator that yields one ``AgentEvent`` per visible step (thought,
tool_intent, tool_result, tool_error, done). The router serialises each
yielded event as one SSE block.

Three pieces are deliberately stubbed in Phase A:
  - ``MemoryManager.recall(...)`` returns an empty Recall (no Tier-1/2/3).
  - The system prompt's ``goal_section`` is a static placeholder; the
    Atlas planner (Phase B) will replace it with the parsed goal row.
  - There is no Konversation/Capture-specific branching yet; the same
    turn function serves every screen until Phase B.

Hard caps enforced here so a misbehaving model can't hang the user:
  - ``policies.MAX_TOOL_CALLS_PER_TURN`` consecutive tool calls
  - ``policies.TURN_TIMEOUT_S`` wall-clock per turn
  - ``policies.TOOL_TIMEOUT_S`` per individual tool dispatch
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional

from backend.agent import policies, prompts, tools
from backend.llm import LLMProvider, ProviderEventKind, get_provider
from backend.llm.provider import ProviderEvent
from backend.mastery import model as mastery_model
from backend.memory import Event, MemoryManager, get_memory
from backend.services.runtime_config import LlmOptions, get_llm_options

logger = logging.getLogger(__name__)


# ─── Event types yielded by turn() ───────────────────────────────────────


class AgentEventKind(str, Enum):
    THOUGHT = "thought"
    TOOL_INTENT = "tool_intent"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    ERROR = "error"
    DONE = "done"


@dataclass(frozen=True)
class AgentEvent:
    """One event in the SSE stream the router emits."""

    kind: AgentEventKind
    data: Any = None

    def to_sse(self) -> str:
        """Serialise to a single ``event: ...\\ndata: ...\\n\\n`` SSE block."""
        # Strings get raw-encoded for human readability; dicts JSON-encoded.
        if isinstance(self.data, str):
            payload = json.dumps(self.data, ensure_ascii=False)
        else:
            payload = json.dumps(self.data or {}, ensure_ascii=False)
        return f"event: {self.kind.value}\ndata: {payload}\n\n"


# ─── Request shape (kept here, not Pydantic, to avoid a circular import) ─


@dataclass
class TurnRequest:
    current_screen: str = "reader"
    time_budget_min: Optional[int] = None
    current_paragraph_id: Optional[str] = None
    work_id: Optional[str] = None
    user_message: Optional[str] = None
    session_id: Optional[str] = None
    goal_text: Optional[str] = None
    options: Optional[LlmOptions] = None  # provider override (rarely needed)
    extras: dict[str, Any] = field(default_factory=dict)


# ─── Helpers ─────────────────────────────────────────────────────────────


def _build_observation_message(req: TurnRequest) -> str:
    """Render the learner-context block the model reads as the first user turn."""
    lines = [
        f"current_screen: {req.current_screen}",
    ]
    if req.time_budget_min is not None:
        lines.append(f"time_budget_min: {req.time_budget_min}")
    if req.current_paragraph_id:
        lines.append(f"current_paragraph_id: {req.current_paragraph_id}")
    if req.work_id:
        lines.append(f"work_id: {req.work_id}")
    if req.user_message:
        lines.append(f"learner_message: {req.user_message}")
    return "\n".join(lines)


def _summarise_recent_mastery(limit: int = 5) -> str:
    """Render the Top-N mastery rows as a fast supplemental context block.

    The ``recall_section`` (built from the RecallPolicy table) is the
    canonical source for context; this helper covers the case where the
    policy returned nothing (cold-start, empty DB) so the prompt always
    has *something* concrete to point at.
    """
    rows = mastery_model.list_top(limit=limit)
    if not rows:
        return "(no recent practice yet)"
    lines = []
    for r in rows:
        lines.append(
            f"- {r.competency_id}: confidence={r.confidence}/100 "
            f"variance={r.variance}/100 (n={r.evidence_count})"
        )
    return "\n".join(lines)


def _render_recall_section(recall) -> str:
    """Render a ``Recall`` bundle as labelled sections for the system prompt.

    Three tiers, each bracketed with its label so the model knows what
    it is reading. Empty tiers are skipped (the v2 prompt's placeholder
    handles the all-empty case at the call site).
    """
    if recall is None:
        return "(no recall slices loaded)"
    blocks: list[str] = []
    for tier_name, sections in (
        ("Tier 1 (always)", getattr(recall, "tier1", []) or []),
        ("Tier 2 (recent)", getattr(recall, "tier2", []) or []),
        ("Tier 3 (vector)", getattr(recall, "tier3", []) or []),
    ):
        if not sections:
            continue
        blocks.append(f"### {tier_name}")
        for s in sections:
            label = getattr(s, "label", "section")
            text = getattr(s, "text", "").strip()
            if not text:
                continue
            blocks.append(f"#### {label}\n{text}")
    if not blocks:
        return "(no recall slices loaded)"
    return "\n\n".join(blocks)


# ─── The turn loop ───────────────────────────────────────────────────────


async def turn(
    request: TurnRequest,
    *,
    provider: Optional[LLMProvider] = None,
    memory: Optional[MemoryManager] = None,
) -> AsyncIterator[AgentEvent]:
    """Run one Mira turn. Yields AgentEvents in stream order.

    ``provider`` and ``memory`` are injectable for tests (FakeProvider /
    in-tmp memory). In production both default to the process-wide singletons.

    Contract per PIVOT_ROADMAP §6.3:
        recall -> observe -> decide (model) -> act (tool) -> persist -> repeat
        ... ending with exactly one ``done`` event regardless of error path.
    """
    started = time.perf_counter()
    provider = provider or get_provider()
    memory = memory or get_memory()
    session_id = request.session_id or _new_session_id()

    options = request.options or get_llm_options()
    # 1) Recall (empty in Phase A) — but we still log a recall_trace row.
    recall = memory.recall(
        tool_hint=f"open_turn:{request.current_screen}",
        observation=request,
        args=request.extras,
        budget_tokens=policies.RECALL_TOKEN_BUDGET,
    )
    memory.write_event(
        Event(
            session_id=session_id,
            kind="turn_open",
            payload={
                "screen": request.current_screen,
                "time_budget_min": request.time_budget_min,
                "current_paragraph_id": request.current_paragraph_id,
                "work_id": request.work_id,
                "recall_trace_id": recall.trace_id,
            },
        )
    )

    # 2) Build the prompt.
    goal_section = (request.goal_text or "(no explicit goal yet)").strip()
    context_section = _summarise_recent_mastery()
    competencies_section = prompts.render_competencies_section()
    recall_section = _render_recall_section(recall)
    system_prompt = prompts.render_system_prompt(
        goal_section=goal_section,
        context_section=context_section,
        competencies_section=competencies_section,
        recall_section=recall_section,
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_observation_message(request)},
    ]

    tool_calls_made = 0
    text_buffer: list[str] = []

    # 3) Outer loop: stream from the model, dispatch tools as they arrive,
    #    feed each tool result back into messages and stream again. The
    #    loop ends when the model produces a turn with no tool calls
    #    (and yields its final text), or when we hit a cap.
    #
    # NOTE on the no-finally-yield discipline: the terminal ``done`` event
    # is emitted from each explicit completion path (see the trailing
    # ``yield AgentEvent(DONE, ...)`` after the try/except), NOT from the
    # ``finally`` block. Yielding inside ``finally`` while a
    # ``GeneratorExit`` is propagating raises
    # ``RuntimeError: async generator ignored GeneratorExit`` and leaks
    # task warnings on every aclose() (see backend/llm/ollama.py for the
    # matching producer-side discipline). Side-effects only in finally.
    fatal_exc: Optional[BaseException] = None
    try:
        while True:
            saw_tool_call = False
            # Per-turn deadline: how long we still have for this whole turn.
            elapsed = time.perf_counter() - started
            remaining = policies.TURN_TIMEOUT_S - elapsed
            if remaining <= 0.5:
                yield AgentEvent(
                    AgentEventKind.ERROR,
                    {"message": "turn_timeout", "elapsed_ms": round(elapsed * 1000)},
                )
                memory.write_event(
                    Event(
                        session_id=session_id,
                        kind="error",
                        payload={"reason": "turn_timeout"},
                    )
                )
                break

            stream_text: list[str] = []
            stream_tool_calls: list[ProviderEvent] = []
            stream_error: Optional[str] = None

            # Cooperatively close the inner provider generator *now* (not at
            # GC time) — this is what kills the rogue
            # ``RuntimeError: async generator ignored GeneratorExit`` warnings
            # under Ollama's streaming endpoint. ``contextlib.aclosing`` would
            # be the idiomatic equivalent but it requires Python 3.10+ and we
            # currently run on 3.9 (medvlm-base conda env).
            ev_iter = provider.chat_with_tools(
                messages,
                tools=tools.TOOL_SCHEMAS,
                options=options,
                timeout=min(remaining, policies.TURN_TIMEOUT_S),
            )
            try:
                async for pev in ev_iter:
                    if pev.kind == ProviderEventKind.TEXT and pev.text:
                        stream_text.append(pev.text)
                    elif pev.kind == ProviderEventKind.TOOL_CALL and pev.tool_name:
                        stream_tool_calls.append(pev)
                    elif pev.kind == ProviderEventKind.ERROR and pev.error:
                        stream_error = pev.error
                    elif pev.kind == ProviderEventKind.DONE:
                        break
            finally:
                await ev_iter.aclose()

            if stream_error is not None:
                yield AgentEvent(
                    AgentEventKind.ERROR,
                    {"message": stream_error},
                )
                memory.write_event(
                    Event(
                        session_id=session_id,
                        kind="error",
                        payload={"reason": "provider_error", "detail": stream_error},
                    )
                )
                break

            # Emit thought (free-text) once per outer loop iteration if any.
            joined_text = "".join(stream_text).strip()
            if joined_text:
                yield AgentEvent(AgentEventKind.THOUGHT, joined_text)
                text_buffer.append(joined_text)
                memory.write_event(
                    Event(
                        session_id=session_id,
                        kind="thought",
                        payload={"text": joined_text},
                    )
                )

            # Dispatch tool calls one at a time (sequential — Phase B can
            # opt parallel for read-only tools).
            for pev in stream_tool_calls:
                if tool_calls_made >= policies.MAX_TOOL_CALLS_PER_TURN:
                    yield AgentEvent(
                        AgentEventKind.ERROR,
                        {
                            "message": "max_tool_calls_per_turn",
                            "limit": policies.MAX_TOOL_CALLS_PER_TURN,
                        },
                    )
                    memory.write_event(
                        Event(
                            session_id=session_id,
                            kind="error",
                            payload={"reason": "max_tool_calls"},
                        )
                    )
                    saw_tool_call = False
                    break  # fall through to the post-loop done emission

                tool_calls_made += 1
                saw_tool_call = True
                args = pev.tool_args or {}
                yield AgentEvent(
                    AgentEventKind.TOOL_INTENT,
                    {"name": pev.tool_name, "args": args},
                )

                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(tools.dispatch, pev.tool_name, args),
                        timeout=policies.TOOL_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    err = {
                        "tool": pev.tool_name,
                        "code": "tool_timeout",
                        "message": (
                            f"tool exceeded {policies.TOOL_TIMEOUT_S}s budget"
                        ),
                    }
                    memory.write_event(
                        Event(
                            session_id=session_id,
                            kind="tool_error",
                            payload=err,
                        )
                    )
                    yield AgentEvent(AgentEventKind.TOOL_ERROR, err)
                    messages.append(
                        _tool_result_message(pev.tool_name, {"error": err})
                    )
                    continue
                except tools.ToolError as exc:
                    memory.write_event(
                        Event(
                            session_id=session_id,
                            kind="tool_error",
                            payload=exc.to_dict(),
                        )
                    )
                    yield AgentEvent(AgentEventKind.TOOL_ERROR, exc.to_dict())
                    messages.append(
                        _tool_result_message(pev.tool_name, {"error": exc.to_dict()})
                    )
                    continue

                # Persist BEFORE yielding so a crash mid-yield doesn't lose evidence.
                memory.write_event(
                    Event(
                        session_id=session_id,
                        kind="tool_result",
                        payload={
                            "name": pev.tool_name,
                            "args": args,
                            "result": result,
                        },
                    )
                )
                yield AgentEvent(AgentEventKind.TOOL_RESULT, result)
                messages.append(_tool_result_message(pev.tool_name, result))

            if not saw_tool_call:
                # Model produced text only -> turn ends.
                break

    except Exception as exc:  # pragma: no cover — defensive, controller guard
        fatal_exc = exc
        logger.exception("agent turn loop crashed: %s", exc)
        yield AgentEvent(
            AgentEventKind.ERROR,
            {"message": "internal_error", "detail": str(exc)},
        )
        memory.write_event(
            Event(
                session_id=session_id,
                kind="error",
                payload={"reason": "controller_crash", "detail": str(exc)},
            )
        )
    finally:
        # Side-effects only — yielding here would conflict with GeneratorExit
        # during aclose(). The terminal ``done`` event lives below.
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        memory.write_event(
            Event(
                session_id=session_id,
                kind="turn_close",
                payload={
                    "tool_calls": tool_calls_made,
                    "turn_ms": elapsed_ms,
                    "text_chunks": len(text_buffer),
                    "ok": fatal_exc is None,
                },
            )
        )
        memory.close_turn(session_id)

    # Single sentinel — even if we yielded an error above, the SSE client
    # expects exactly one done event to close the stream. Lives OUTSIDE
    # ``finally`` so a GeneratorExit during aclose() unwinds cleanly.
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    yield AgentEvent(
        AgentEventKind.DONE,
        {"turn_ms": elapsed_ms, "session_id": session_id},
    )


def _tool_result_message(name: str, result: Any) -> dict[str, str]:
    """Render a tool result as a follow-up user message for the next round.

    Ollama's ``role: tool`` is supported in 0.4+, but to keep the protocol
    portable we emit a synthetic user message with a clearly tagged body.
    The model has been trained (and the system prompt nudged) to interpret
    these as tool responses, not learner input.
    """
    return {
        "role": "tool",
        "name": name,
        "content": json.dumps(result, ensure_ascii=False),
    }


def _new_session_id() -> str:
    """Generate a session id that is filesystem-safe and roughly time-sortable."""
    # ISO-ish timestamp + 8 random hex chars; avoids collisions across workers.
    import datetime as _dt

    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"
