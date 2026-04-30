"""Tests for backend/memory/summariser.py (PIVOT_ROADMAP §B.12)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from backend.memory import compactor as compactor_module
from backend.memory import summariser
from backend.memory.manager import Event, MemoryManager


# ─── helpers ────────────────────────────────────────────────────────────


class _FakeProvider:
    """Stand-in LLMProvider whose ``generate`` returns a canned payload."""

    def __init__(self, payloads: list[Optional[str]]):
        self._payloads = list(payloads)
        self.calls: list[str] = []
        self.name = "fake"

    async def generate(
        self,
        prompt: str,
        *,
        options: Any = None,
        timeout: float = 5.0,
    ) -> Optional[str]:
        self.calls.append(prompt)
        if not self._payloads:
            return None
        return self._payloads.pop(0)


def _make_manager(tmp_path: Path) -> MemoryManager:
    return MemoryManager(root=tmp_path / "mem", init_git=False)


def _populate(mm: MemoryManager, sid: str) -> list[Path]:
    for kind, payload in (
        ("turn_open", {}),
        ("user_message", {"role": "user", "text": "Wo ist die nächste Bäckerei?"}),
        ("tool_intent", {"name": "explain_grammar", "args": {"competency_id": "grammar.cases"}}),
        ("tool_result", {"name": "explain_grammar", "result": {"ok": True}}),
        ("assistant_message", {"role": "assistant", "text": "Geh zwei Straßen geradeaus."}),
        ("turn_close", {"turn_ms": 230}),
    ):
        mm.write_event(Event(session_id=sid, kind=kind, payload=payload))
    mm.close_turn(sid)
    return mm.list_session_files(sid)


# ─── extraction ─────────────────────────────────────────────────────────


def test_facts_capture_event_count_and_tools(tmp_path: Path) -> None:
    mm = _make_manager(tmp_path)
    files = _populate(mm, "s-facts")
    facts = summariser._extract_facts("s-facts", files)
    assert facts.event_count == 6
    assert facts.tools_used == {"explain_grammar": 1}
    assert any("Bäckerei" in line for line in facts.text_events)


# ─── mechanical fallback ────────────────────────────────────────────────


def test_mechanical_fallback_when_no_provider(tmp_path: Path) -> None:
    mm = _make_manager(tmp_path)
    files = _populate(mm, "s-mech")
    fake = _FakeProvider([None])
    summary = summariser.summarise_session("s-mech", files, provider=fake)
    assert summary.source == "mechanical"
    assert summary.gist
    assert summary.summ
    # The provider was called once and got nothing back, then we fell back.
    assert len(fake.calls) <= 2  # one attempt + optional re-prompt


def test_mechanical_fallback_when_no_files(tmp_path: Path) -> None:
    summary = summariser.summarise_session("ghost", files=[])
    assert summary.source == "mechanical"
    assert "ghost" in summary.gist


# ─── LLM happy path ─────────────────────────────────────────────────────


def test_llm_happy_path(tmp_path: Path) -> None:
    mm = _make_manager(tmp_path)
    files = _populate(mm, "s-llm")
    payload = json.dumps({"gist": "Wegbeschreibung geübt.", "summ": "Erster Absatz.\n\nZweiter Absatz."})
    fake = _FakeProvider([payload])

    summary = summariser.summarise_session("s-llm", files, provider=fake)
    assert summary.source == "llm"
    assert summary.gist == "Wegbeschreibung geübt."
    assert "Erster Absatz." in summary.summ
    assert len(fake.calls) == 1


def test_llm_retries_once_on_malformed_json(tmp_path: Path) -> None:
    mm = _make_manager(tmp_path)
    files = _populate(mm, "s-retry")
    bad = "Hier ist die Antwort: { gist: kein json ;-)"
    good = json.dumps({"gist": "Kurz und gut.", "summ": "Eins.\n\nZwei."})
    fake = _FakeProvider([bad, good])

    summary = summariser.summarise_session("s-retry", files, provider=fake)
    assert summary.source == "llm"
    assert summary.gist == "Kurz und gut."
    assert len(fake.calls) == 2


def test_llm_falls_back_when_word_caps_unmet(tmp_path: Path) -> None:
    mm = _make_manager(tmp_path)
    files = _populate(mm, "s-empty")
    fake = _FakeProvider([json.dumps({"gist": "", "summ": ""})])

    summary = summariser.summarise_session("s-empty", files, provider=fake)
    assert summary.source == "mechanical"


def test_llm_extracts_embedded_json_blob(tmp_path: Path) -> None:
    mm = _make_manager(tmp_path)
    files = _populate(mm, "s-embed")
    raw = "Some preamble\n```\n" + json.dumps({"gist": "Embedded ok.", "summ": "Eins.\n\nZwei."}) + "\n```"
    fake = _FakeProvider([raw])

    summary = summariser.summarise_session("s-embed", files, provider=fake)
    assert summary.source == "llm"
    assert summary.gist == "Embedded ok."


# ─── write_summary_files ────────────────────────────────────────────────


def test_write_summary_files_round_trip(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    summary = summariser.SessionSummary(
        session_id="s-rt",
        gist="A gist.",
        summ="A summary.",
        source="llm",
    )
    gist_path, summ_path = summariser.write_summary_files(sessions_dir, summary)
    assert gist_path.read_text(encoding="utf-8").strip().endswith("_source: llm_")
    assert "A summary." in summ_path.read_text(encoding="utf-8")


def test_write_summary_files_is_idempotent(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    summary = summariser.SessionSummary(
        session_id="s-id",
        gist="Same gist.",
        summ="Same summary.",
        source="mechanical",
    )
    summariser.write_summary_files(sessions_dir, summary)
    mtime1 = (sessions_dir / "s-id.gist.md").stat().st_mtime_ns

    # Write again with identical body — file must not be touched.
    summariser.write_summary_files(sessions_dir, summary)
    mtime2 = (sessions_dir / "s-id.gist.md").stat().st_mtime_ns
    assert mtime1 == mtime2


# ─── close_session integration ──────────────────────────────────────────


def test_close_session_writes_gist_and_summ(tmp_path: Path) -> None:
    mm = _make_manager(tmp_path)
    sid = "s-close"
    mm.write_event(Event(session_id=sid, kind="turn_open"))
    mm.write_event(
        Event(
            session_id=sid,
            kind="user_message",
            payload={"role": "user", "text": "Hallo Mira."},
        )
    )
    mm.close_session(sid)

    sessions_dir = mm._root / "sessions"
    assert (sessions_dir / f"{sid}.gist.md").exists()
    assert (sessions_dir / f"{sid}.summ.md").exists()


# ─── compactor integration ──────────────────────────────────────────────


def test_compactor_uses_summariser_for_orphans(tmp_path: Path) -> None:
    mm = _make_manager(tmp_path)
    sid = "s-orphan"
    _populate(mm, sid)
    # No close_session — leaves an orphan file the compactor must pick up.

    report = compactor_module.run_for_manager(mm, deadline_ms=5000)
    assert report.gists_written >= 1
    assert (mm._root / "sessions" / f"{sid}.gist.md").exists()
