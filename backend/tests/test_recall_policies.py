"""RecallPolicy table tests (PIVOT_ROADMAP §A.13).

Asserts that the policy table:

- Knows about every Phase-A tool (open_turn, recommend_text,
  update_mastery, start_drill).
- Returns an empty Recall on a cold-start DB (every tool yields
  Recall.empty() rather than crashing).
- Loads real DuckDB rows once mastery exists.
- Honours the per-policy budget by trimming Tier 3 then Tier 2.
- Exposes itself via ``MemoryManager.recall(tool_hint=...)`` and
  writes an enriched trace row (policy + tier counts + latency_ms,
  no ``stub: true``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.mastery import model as mastery_model
from backend.memory import policies
from backend.memory.manager import MemoryManager, RecallContext, Section
from backend.memory.policies import POLICIES, find_policy


def test_policy_table_covers_all_phase_a_tools() -> None:
    expected = {"open_turn", "recommend_text", "update_mastery", "start_drill"}
    assert expected <= set(POLICIES)


def test_policy_table_covers_phase_b_tools() -> None:
    expected = {
        "parse_goal",
        "plan_curriculum",
        "replan_atlas",
        "start_conversation",
        "capture_text",
        "explain_grammar",
    }
    assert expected <= set(POLICIES)


def test_find_policy_matches_colon_prefix() -> None:
    assert find_policy("open_turn:reader").tool_hint == "open_turn"
    assert find_policy("recommend_text:next") is POLICIES["recommend_text"]
    assert find_policy("unknown") is None
    assert find_policy("") is None


def test_find_policy_resolves_capture_aliases() -> None:
    """B.11's start_capture is the same tool as §7.5's capture_text."""
    assert find_policy("start_capture") is POLICIES["capture_text"]
    assert find_policy("start_capture:room") is POLICIES["capture_text"]


def test_cold_start_returns_empty_recall(tmp_path: Path, monkeypatch) -> None:
    """No mastery, no profile, no goal — every policy still returns cleanly.

    We monkeypatch PROJECT_ROOT so the helpers don't accidentally read
    real ``data/memory/`` files left by a previous session.
    """
    monkeypatch.setattr(policies, "PROJECT_ROOT", tmp_path)
    for hint in ("open_turn", "recommend_text", "start_drill"):
        ctx = RecallContext(tool_hint=hint, observation=None, args={}, budget_tokens=2000)
        recall = POLICIES[hint].build(ctx)
        assert recall.tier1 == []
        assert recall.tier2 == []


def test_cold_start_phase_b_policies_return_empty(
    tmp_path: Path, monkeypatch
) -> None:
    """Every B.11 policy must be cold-start safe (no mastery, no goal)."""
    monkeypatch.setattr(policies, "PROJECT_ROOT", tmp_path)
    for hint in (
        "parse_goal",
        "plan_curriculum",
        "replan_atlas",
        "start_conversation",
        "capture_text",
        "explain_grammar",
    ):
        ctx = RecallContext(
            tool_hint=hint, observation=None, args={}, budget_tokens=POLICIES[hint].budget_tokens
        )
        recall = POLICIES[hint].build(ctx)
        assert recall.tier1 == []
        assert recall.tier2 == []
        assert recall.used_tokens == 0


def test_phase_b_budgets_are_within_section_7_5_envelope() -> None:
    """Sanity-check the §7.5 budget envelopes haven't drifted."""
    expected = {
        "parse_goal": 800,
        "plan_curriculum": 3500,
        "replan_atlas": 4000,
        "start_conversation": 3500,
        "capture_text": 1000,
        "explain_grammar": 2000,
    }
    for tool, budget in expected.items():
        assert POLICIES[tool].budget_tokens == budget


def test_plan_curriculum_loads_top_mastery_after_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(policies, "PROJECT_ROOT", tmp_path)
    mastery_model.update("verbs.modal", 0.7, source="drill")

    ctx = RecallContext(
        tool_hint="plan_curriculum",
        observation=None,
        args={},
        budget_tokens=3500,
    )
    recall = POLICIES["plan_curriculum"].build(ctx)
    labels = [s.label for s in recall.tier1]
    assert "top_mastery" in labels


def test_replan_atlas_loads_full_mastery_and_evidence_histogram(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(policies, "PROJECT_ROOT", tmp_path)
    mastery_model.update("verbs.modal", 1.0, source="drill")
    mastery_model.update("daily.greetings", 0.5, source="drill")

    ctx = RecallContext(
        tool_hint="replan_atlas",
        observation=None,
        args={},
        budget_tokens=4000,
    )
    recall = POLICIES["replan_atlas"].build(ctx)
    t1 = [s.label for s in recall.tier1]
    t2 = [s.label for s in recall.tier2]
    assert "full_mastery" in t1
    assert any(label.startswith("evidence_histogram") for label in t2)


def test_explain_grammar_targets_competency(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(policies, "PROJECT_ROOT", tmp_path)
    mastery_model.update("grammar.cases", 0.4, source="drill")
    mastery_model.update("grammar.cases", 0.8, source="drill")

    ctx = RecallContext(
        tool_hint="explain_grammar",
        observation=None,
        args={"competency_id": "grammar.cases"},
        budget_tokens=2000,
    )
    recall = POLICIES["explain_grammar"].build(ctx)
    labels = [s.label for s in recall.tier1] + [s.label for s in recall.tier2]
    assert any(label == "mastery:grammar.cases" for label in labels)
    assert any(label == "recent_evidence:grammar.cases" for label in labels)


def test_start_conversation_reads_recent_transcripts(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(policies, "PROJECT_ROOT", tmp_path)
    konv = tmp_path / "data" / "memory" / "konversation"
    konv.mkdir(parents=True)
    (konv / "session-1.jsonl").write_text("{}\n", encoding="utf-8")

    ctx = RecallContext(
        tool_hint="start_conversation",
        observation=None,
        args={},
        budget_tokens=3500,
    )
    recall = POLICIES["start_conversation"].build(ctx)
    labels = [s.label for s in recall.tier2]
    assert any(label.startswith("recent_transcripts") for label in labels)


def test_capture_text_reads_capture_meta(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(policies, "PROJECT_ROOT", tmp_path)
    captures = tmp_path / "data" / "memory" / "captures"
    captures.mkdir(parents=True)
    (captures / "meta.jsonl").write_text(
        '{"id":"c-1","kind":"menu","created_at":"2026-04-30T09:00:00"}\n',
        encoding="utf-8",
    )

    ctx = RecallContext(
        tool_hint="capture_text",
        observation=None,
        args={},
        budget_tokens=1000,
    )
    recall = POLICIES["capture_text"].build(ctx)
    labels = [s.label for s in recall.tier2]
    assert "recent_captures_meta" in labels


def test_open_turn_loads_top_mastery_after_evidence() -> None:
    mastery_model.update("verbs.modal", 0.7, source="drill")
    mastery_model.update("daily.greetings", 0.4, source="drill")

    ctx = RecallContext(tool_hint="open_turn", observation=None, args={}, budget_tokens=2000)
    recall = POLICIES["open_turn"].build(ctx)
    labels = [s.label for s in recall.tier1]
    assert "top_mastery" in labels
    body = next(s.text for s in recall.tier1 if s.label == "top_mastery")
    assert "verbs.modal" in body


def test_update_mastery_targets_a_specific_competency() -> None:
    mastery_model.update("verbs.modal", 1.0, source="drill")
    mastery_model.update("verbs.modal", 0.5, source="drill")

    ctx = RecallContext(
        tool_hint="update_mastery",
        observation=None,
        args={"competency_id": "verbs.modal"},
        budget_tokens=1500,
    )
    recall = POLICIES["update_mastery"].build(ctx)
    labels = [s.label for s in recall.tier1] + [s.label for s in recall.tier2]
    assert any(label == "mastery:verbs.modal" for label in labels)
    assert any(label == "recent_evidence:verbs.modal" for label in labels)


def test_start_drill_handles_missing_competency_gracefully() -> None:
    ctx = RecallContext(
        tool_hint="start_drill",
        observation=None,
        args={},
        budget_tokens=1500,
    )
    recall = POLICIES["start_drill"].build(ctx)
    assert recall.tier1 == []  # no competency_id, no top_mastery yet


def test_recommend_text_loads_goal_when_present(tmp_path: Path, monkeypatch) -> None:
    # _goal_json_section reads <PROJECT_ROOT>/data/memory/learner/goal.json,
    # so place the seed there inside the patched root.
    learner = tmp_path / "data" / "memory" / "learner"
    learner.mkdir(parents=True)
    (learner / "goal.json").write_text(
        json.dumps({"target_cefr": "B2", "domain": "medical"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(policies, "PROJECT_ROOT", tmp_path)

    ctx = RecallContext(
        tool_hint="recommend_text",
        observation=None,
        args={},
        budget_tokens=2500,
    )
    recall = POLICIES["recommend_text"].build(ctx)
    labels = [s.label for s in recall.tier1]
    assert "active_goal" in labels


# ─── End-to-end: MemoryManager dispatches via the policy ─────────────────


def test_memory_manager_dispatches_to_policy(tmp_path: Path) -> None:
    mastery_model.update("verbs.modal", 0.6, source="drill")
    mm = MemoryManager(root=tmp_path / "mem", init_git=False)

    recall = mm.recall(tool_hint="open_turn", args={}, budget_tokens=2000)
    assert any(s.label == "top_mastery" for s in recall.tier1)

    trace_path = tmp_path / "mem" / "recall_trace.jsonl"
    rows = [json.loads(l) for l in trace_path.read_text().splitlines()]
    assert rows[-1]["policy"] == "open_turn"
    assert rows[-1]["tier1_sections"] >= 1
    assert "stub" not in rows[-1]
    assert "latency_ms" in rows[-1]


def test_memory_manager_falls_back_to_stub_for_unknown_tool(tmp_path: Path) -> None:
    mm = MemoryManager(root=tmp_path / "mem", init_git=False)
    recall = mm.recall(tool_hint="brand_new_tool", args={}, budget_tokens=500)
    assert recall.tier1 == []
    trace_path = tmp_path / "mem" / "recall_trace.jsonl"
    rows = [json.loads(l) for l in trace_path.read_text().splitlines()]
    assert rows[-1].get("stub") is True


# ─── Budget enforcement ─────────────────────────────────────────────────


def test_enforce_budget_drops_tier3_first(tmp_path: Path) -> None:
    """The internal budget enforcer must protect Tier 1, drop Tier 3 first."""
    from backend.memory.manager import Recall, _enforce_budget

    big = Section(label="big", text="x" * 4000, tokens=1000)
    small = Section(label="small", text="x" * 400, tokens=100)
    recall = Recall(
        tier1=[small],
        tier2=[big],
        tier3=[big],
        used_tokens=2100,
        trace_id="t",
    )
    trimmed, dropped = _enforce_budget(recall, budget_tokens=300)
    assert dropped is True
    assert trimmed.tier1  # never dropped
    assert trimmed.tier3 == []
