"""Atrium home-screen helper (no LLM).

The Atrium is the new default screen the learner sees on opening
LinguaMate (PIVOT_ROADMAP §8.2). It must answer in one round-trip and
must never require an LLM call — that's what makes it feel instant.

This helper assembles the data the screen needs from the same sources
the agent loop uses (mastery, scheduler, corpus_service), so what the
learner sees on the home screen always matches what Mira will tell them
inside a turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from backend.mastery import model as mastery_model
from backend.mastery import scheduler


# ─── Public payload shape ────────────────────────────────────────────────


@dataclass
class Door:
    """One of the four doors at the bottom of the Atrium."""

    id: str
    label: str
    subtitle: str = ""


@dataclass
class MasterySummary:
    """Compact mastery row for the MasteryDial component."""

    competency_id: str
    confidence: int       # 0..100
    variance: int         # 0..100
    label: str
    cefr: Optional[str] = None


@dataclass
class AtriumPayload:
    """Everything the Atrium screen needs in one shot."""

    greeting: str
    today_plan: list[str] = field(default_factory=list)
    doors: list[Door] = field(default_factory=list)
    mastery_top: list[MasterySummary] = field(default_factory=list)
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "greeting": self.greeting,
            "today_plan": list(self.today_plan),
            "doors": [d.__dict__ for d in self.doors],
            "mastery_top": [m.__dict__ for m in self.mastery_top],
            "generated_at": self.generated_at,
        }


# ─── Builders ────────────────────────────────────────────────────────────


def _greeting_for(now: datetime) -> str:
    """German time-of-day greeting. No LLM, no surprises."""
    hour = now.hour
    if hour < 5:
        return "noch wach?"
    if hour < 11:
        return "guten Morgen"
    if hour < 14:
        return "Mahlzeit"
    if hour < 18:
        return "guten Nachmittag"
    if hour < 22:
        return "guten Abend"
    return "noch wach?"


def _today_plan(*, due: list, top: list) -> list[str]:
    """Two short capability sentences derived from real mastery state.

    We reach for *due* reviews first (the scheduler said "now"), then
    fall back to top-of-mind competencies. Sentences are intentionally
    plain — Phase B replaces them with the LLM-generated capability
    sentences from the planner.
    """
    plan: list[str] = []

    label_map = _label_lookup()

    for row in due[:1]:
        label = label_map.get(row.competency_id, row.competency_id)
        plan.append(f'die Wiederholung „{label}" abschliessen')

    for row in top[:2]:
        if row.competency_id in {p_row.competency_id for p_row in due[:1]}:
            continue
        label = label_map.get(row.competency_id, row.competency_id)
        plan.append(f"{label} eine Stufe sicherer machen")
        if len(plan) >= 2:
            break

    if not plan:
        plan = [
            "einen Patientenfall auf B1 zusammenfassen",
            "die Wörter aus gestern wiederholen",
        ]
    return plan[:2]


def _doors() -> list[Door]:
    """The four-door grid. Subtitle text is best-effort and may be empty."""
    return [
        Door(id="reader", label="Lesen", subtitle="zur nächsten Passage"),
        Door(id="voice", label="Sprechen", subtitle="5 Minuten"),
        Door(id="capture", label="Sehen", subtitle="kommt bald"),
        Door(id="atlas", label="Karte", subtitle="dein Lernplan"),
    ]


def _mastery_top(*, limit: int = 3) -> list[MasterySummary]:
    rows = mastery_model.list_top(limit=limit)
    label_map = _label_lookup()
    cefr_map = _cefr_lookup()
    summaries: list[MasterySummary] = []
    for r in rows:
        summaries.append(
            MasterySummary(
                competency_id=r.competency_id,
                confidence=r.confidence,
                variance=r.variance,
                label=label_map.get(r.competency_id, r.competency_id),
                cefr=cefr_map.get(r.competency_id),
            )
        )
    return summaries


def _label_lookup() -> dict[str, str]:
    from backend.mastery.competencies import SEED

    return {c.id: c.label for c in SEED}


def _cefr_lookup() -> dict[str, str]:
    from backend.mastery.competencies import SEED

    return {c.id: c.cefr for c in SEED if c.cefr}


# ─── Public entry point ─────────────────────────────────────────────────


def build_atrium_payload(*, now: Optional[datetime] = None) -> AtriumPayload:
    """Assemble the full Atrium payload. Pure: the only side effect is
    a few read-only DuckDB queries."""
    now = now or datetime.now()
    due = scheduler.due_competencies(limit=5, at=now)
    top = mastery_model.list_top(limit=3)

    return AtriumPayload(
        greeting=_greeting_for(now),
        today_plan=_today_plan(due=due, top=top),
        doors=_doors(),
        mastery_top=_mastery_top(limit=3),
        generated_at=now.isoformat(),
    )
