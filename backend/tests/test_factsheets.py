"""Tests for backend/memory/factsheets.py (PIVOT_ROADMAP §B.13)."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.mastery import model as mastery_model
from backend.memory import factsheets, manager as memory_manager
from backend.memory import policies
from backend.memory.manager import RecallContext


def _write_evidence(competency_id: str, *, quality: float, surface_form: str = "") -> None:
    mastery_model.update(
        competency_id,
        quality,
        source="conversation",
        surface_form=surface_form,
    )


def _facts_path(competency_id: str) -> Path:
    root = Path(memory_manager.get_memory()._root)  # type: ignore[union-attr]
    return root / "facts" / f"{factsheets._safe_filename(competency_id)}.md"


# ─── Atomic IO + skeleton ───────────────────────────────────────────────


def test_first_update_creates_sheet_with_all_sections():
    factsheets.update_factsheet(
        factsheets.FactsheetEvidence(
            competency_id="grammar.cases",
            quality=0.9,
            surface_form="Ich gehe in die Schule.",
        ),
        factsheets.FactsheetMastery(confidence=72, variance=18),
    )
    body = _facts_path("grammar.cases").read_text(encoding="utf-8")
    assert "# grammar.cases" in body
    assert "## Confidence: 72/100  (σ = 18)" in body
    assert "Productive examples" in body
    assert "Recurring mistakes" in body
    assert "Next teaching priority" in body
    assert '"Ich gehe in die Schule."' in body


def test_load_factsheet_returns_none_when_missing():
    assert factsheets.load_factsheet("grammar.never_seen") is None


def test_failure_evidence_lands_in_mistakes_bucket():
    factsheets.update_factsheet(
        factsheets.FactsheetEvidence(
            competency_id="grammar.cases",
            quality=0.1,
            surface_form="Ich gehe in den Schule.",
        ),
        factsheets.FactsheetMastery(confidence=33, variance=22),
    )
    body = _facts_path("grammar.cases").read_text(encoding="utf-8")
    assert "## Recurring mistakes" in body
    assert '"Ich gehe in den Schule."' in body


def test_repeated_mistake_increments_count():
    ev = factsheets.FactsheetEvidence(
        competency_id="grammar.cases",
        quality=0.0,
        surface_form="den Schule",
    )
    mastery = factsheets.FactsheetMastery(confidence=10, variance=20)
    factsheets.update_factsheet(ev, mastery)
    factsheets.update_factsheet(ev, mastery)
    factsheets.update_factsheet(ev, mastery)
    body = _facts_path("grammar.cases").read_text(encoding="utf-8")
    assert '"den Schule" ×3' in body


def test_examples_bucket_keeps_at_most_five():
    for i in range(8):
        factsheets.update_factsheet(
            factsheets.FactsheetEvidence(
                competency_id="daily.greetings",
                quality=0.9,
                surface_form=f"Beispiel Nummer {i}",
            ),
            factsheets.FactsheetMastery(confidence=60 + i, variance=10),
        )
    body = _facts_path("daily.greetings").read_text(encoding="utf-8")
    examples = [line for line in body.splitlines() if line.startswith('- "Beispiel')]
    assert len(examples) <= 5
    assert '"Beispiel Nummer 7"' in body
    assert '"Beispiel Nummer 0"' not in body


def test_word_budget_evicts_examples_first():
    # Pump in long sentences until we exceed the cap, then verify pruning
    # kicked in (sheet still renders, examples bucket shrunk).
    long_sentence = " ".join(["wort"] * 60)
    for i in range(6):
        factsheets.update_factsheet(
            factsheets.FactsheetEvidence(
                competency_id="grammar.cases",
                quality=0.9,
                surface_form=f"{long_sentence} {i}",
            ),
            factsheets.FactsheetMastery(confidence=80, variance=8),
        )
    body = _facts_path("grammar.cases").read_text(encoding="utf-8")
    assert factsheets._word_count(body) <= factsheets._WORD_BUDGET + 25  # render header overhead allowance


def test_atomic_write_no_tmp_artefact_left_behind():
    factsheets.update_factsheet(
        factsheets.FactsheetEvidence(
            competency_id="grammar.cases",
            quality=0.7,
            surface_form="probe",
        ),
        factsheets.FactsheetMastery(confidence=50, variance=12),
    )
    facts_dir = Path(memory_manager.get_memory()._root) / "facts"  # type: ignore[union-attr]
    assert not list(facts_dir.glob("*.tmp"))


# ─── Mastery integration ────────────────────────────────────────────────


def test_mastery_update_writes_factsheet():
    _write_evidence("daily.greetings", quality=0.9, surface_form="Guten Tag")
    body = _facts_path("daily.greetings").read_text(encoding="utf-8")
    assert "daily.greetings" in body
    assert '"Guten Tag"' in body


def test_mastery_update_without_surface_form_still_updates_confidence():
    _write_evidence("daily.greetings", quality=0.9, surface_form="erstes")
    _write_evidence("daily.greetings", quality=0.85)  # no surface form
    body = _facts_path("daily.greetings").read_text(encoding="utf-8")
    assert "## Confidence:" in body
    # The earlier example must still be there even though we didn't
    # add a new one.
    assert '"erstes"' in body


# ─── RecallPolicy integration ───────────────────────────────────────────


def test_explain_grammar_recall_picks_up_factsheet():
    _write_evidence("grammar.cases", quality=0.9, surface_form="Wir lernen.")
    ctx = RecallContext(
        tool_hint="explain_grammar",
        observation=None,
        args={"competency_id": "grammar.cases"},
        budget_tokens=2000,
    )
    recall = policies._build_explain_grammar(ctx)
    labels = [s.label for s in recall.tier1]
    assert any(label.startswith("factsheet:") for label in labels)


def test_update_mastery_recall_picks_up_factsheet():
    _write_evidence("grammar.cases", quality=0.9, surface_form="Probe.")
    ctx = RecallContext(
        tool_hint="update_mastery",
        observation=None,
        args={"competency_id": "grammar.cases"},
        budget_tokens=1000,
    )
    recall = policies._build_update_mastery(ctx)
    labels = [s.label for s in recall.tier1]
    assert any(label.startswith("factsheet:") for label in labels)


def test_start_drill_recall_picks_up_factsheet():
    _write_evidence("grammar.cases", quality=0.9, surface_form="Drill probe.")
    ctx = RecallContext(
        tool_hint="start_drill",
        observation=None,
        args={"competency_id": "grammar.cases"},
        budget_tokens=1500,
    )
    recall = policies._build_start_drill(ctx)
    labels = [s.label for s in recall.tier1]
    assert any(label.startswith("factsheet:") for label in labels)
