"""Versioned prompt tests (PIVOT_ROADMAP §A.6 v2).

These tests pin the *intent* of the v2 changes (so a careless edit
that drops the no-loop rule fails CI), without freezing the exact
wording (which we want to keep iterating on).
"""

from __future__ import annotations

from backend.agent import prompts


def test_v1_and_v2_are_distinct() -> None:
    assert prompts.MIRA_SYSTEM_PROMPT_V1 != prompts.MIRA_SYSTEM_PROMPT_V2
    assert prompts.MIRA_SYSTEM_PROMPT is prompts.MIRA_SYSTEM_PROMPT_V2


def test_v2_drops_always_pick_a_tool_rule() -> None:
    assert "ALWAYS pick a tool" in prompts.MIRA_SYSTEM_PROMPT_V1
    assert "ALWAYS pick a tool" not in prompts.MIRA_SYSTEM_PROMPT_V2


def test_v2_carries_no_loop_and_stop_when_satisfied_rules() -> None:
    body = prompts.MIRA_SYSTEM_PROMPT_V2
    assert "AT MOST ONE tool" in body
    assert "Never call the same tool with the same arguments twice" in body
    # The "calm acknowledgement / silence is fine" intent.
    assert "Silence is fine" in body or "silence is fine" in body.lower()


def test_v2_has_competencies_and_recall_placeholders() -> None:
    body = prompts.MIRA_SYSTEM_PROMPT_V2
    assert "{competencies_section}" in body
    assert "{recall_section}" in body


def test_render_competencies_section_uses_seed_ids() -> None:
    rendered = prompts.render_competencies_section()
    # SEED includes verbs.modal as a known anchor; if someone deletes it
    # this test will (loudly) tell us so.
    assert "verbs.modal" in rendered
    # One line per competency.
    assert rendered.count("\n") >= 5


def test_render_system_prompt_falls_back_to_placeholders() -> None:
    out = prompts.render_system_prompt()
    assert "(competency catalog not loaded)" in out
    assert "(no recall slices loaded)" in out


def test_render_system_prompt_substitutes_all_sections() -> None:
    out = prompts.render_system_prompt(
        goal_section="goal: become B2 medical",
        context_section="ctx: practiced modals last night",
        competencies_section="- a.b  (A2 · grammar)  Some label",
        recall_section="### Tier 1\n#### top_mastery\nfoo",
    )
    assert "goal: become B2 medical" in out
    assert "Some label" in out
    assert "top_mastery" in out