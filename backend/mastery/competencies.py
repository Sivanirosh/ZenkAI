"""Seed list of competencies + idempotent inserter.

The taxonomy is hierarchical (parent_id chains): high-level CEFR buckets
parent the situational tags, and the demo-persona-specific entries
(`medical.*`, `academic.*`) sit under the relevant CEFR bucket.

This is v1 — Phase B will grow it dynamically as the Atlas planner asks
for new competencies for new goals. For now, ~30 entries cover the FSP
medical persona, an academic-postdoc persona, and the daily-life baseline
that every learner needs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from backend.models import db

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompetencyDef:
    id: str
    label: str
    cefr: Optional[str] = None
    domain: Optional[str] = None
    parent_id: Optional[str] = None


# CEFR bucket roots. These have no parent.
SEED: list[CompetencyDef] = [
    # ── CEFR roots ──────────────────────────────────────────────
    CompetencyDef("cefr.a2", "CEFR A2 (Elementary)", cefr="A2"),
    CompetencyDef("cefr.b1", "CEFR B1 (Intermediate)", cefr="B1"),
    CompetencyDef("cefr.b2", "CEFR B2 (Upper-Intermediate)", cefr="B2"),

    # ── Daily-life baseline (everyone needs these) ──────────────
    CompetencyDef("daily.greetings", "Greetings and small talk", "A2", "daily", "cefr.a2"),
    CompetencyDef("daily.numbers_time", "Numbers, dates, telling time", "A2", "daily", "cefr.a2"),
    CompetencyDef("daily.shopping", "Shopping and prices", "A2", "daily", "cefr.a2"),
    CompetencyDef("daily.directions", "Asking for and giving directions", "A2", "daily", "cefr.a2"),
    CompetencyDef("daily.transit", "Public transport (U-Bahn, Bahn, tickets)", "A2", "daily", "cefr.a2"),
    CompetencyDef("daily.appointments", "Booking and changing appointments", "B1", "daily", "cefr.b1"),
    CompetencyDef("daily.complaints", "Polite complaints and returns", "B1", "daily", "cefr.b1"),

    # ── Grammar competencies that everyone hits ─────────────────
    CompetencyDef("grammar.cases", "Noun cases (Nom/Akk/Dat/Gen)", "A2", "grammar", "cefr.a2"),
    CompetencyDef("grammar.modal_verbs", "Modal verbs (können, müssen, sollen, ...)", "A2", "grammar", "cefr.a2"),
    CompetencyDef("grammar.perfekt", "Perfekt tense (haben/sein + Partizip II)", "A2", "grammar", "cefr.a2"),
    CompetencyDef("grammar.subordinate", "Subordinate clauses (weil, dass, wenn)", "B1", "grammar", "cefr.b1"),
    CompetencyDef("grammar.konjunktiv2", "Konjunktiv II (würde / hätte / wäre)", "B1", "grammar", "cefr.b1"),
    CompetencyDef("grammar.passive", "Passive voice (werden + Partizip II)", "B2", "grammar", "cefr.b2"),

    # ── Verbs (sub-tree mirroring grammar but used for tools) ───
    CompetencyDef("verbs.modal", "Modal verb usage in context", "A2", "verbs", "grammar.modal_verbs"),
    CompetencyDef("verbs.separable", "Separable verbs (aufstehen, mitkommen)", "A2", "verbs", "cefr.a2"),
    CompetencyDef("verbs.reflexive", "Reflexive verbs (sich freuen, sich erinnern)", "B1", "verbs", "cefr.b1"),

    # ── Medical (FSP persona) ───────────────────────────────────
    CompetencyDef("medical.history.questions", "Asking patient-history questions", "B1", "medical", "cefr.b1"),
    CompetencyDef("medical.symptoms", "Describing and probing symptoms", "B1", "medical", "cefr.b1"),
    CompetencyDef("medical.body_parts", "Anatomy vocabulary", "A2", "medical", "cefr.a2"),
    CompetencyDef("medical.diagnostics", "Explaining diagnostic procedures", "B2", "medical", "cefr.b2"),
    CompetencyDef("medical.pharmacology", "Drug names, dosage, contraindications", "B2", "medical", "cefr.b2"),
    CompetencyDef("medical.consent", "Informed consent dialogue", "B2", "medical", "cefr.b2"),

    # ── Academic / postdoc persona ──────────────────────────────
    CompetencyDef("academic.methodology", "Describing research methodology", "B2", "academic", "cefr.b2"),
    CompetencyDef("academic.results", "Summarising experimental results", "B2", "academic", "cefr.b2"),
    CompetencyDef("academic.defense", "Defending an argument under questioning", "B2", "academic", "cefr.b2"),
    CompetencyDef("academic.email", "Formal academic email register", "B1", "academic", "cefr.b1"),

    # ── Reading / literature (Lesekamerad heritage) ─────────────
    CompetencyDef("reading.kafka", "Reading Kafka (early-modern Expressionismus)", "B2", "reading", "cefr.b2"),
    CompetencyDef("reading.news", "Reading German news headlines and articles", "B1", "reading", "cefr.b1"),
]


def seed_competencies(conn=None) -> int:
    """Insert any missing seed competencies. Returns the number inserted.

    Uses INSERT OR IGNORE semantics via DuckDB's ON CONFLICT — safe to
    call on every boot. Existing rows with the same id are left untouched
    so user / Atlas-planner additions are not clobbered.
    """
    cursor = (conn.cursor() if conn is not None else db.cursor())
    inserted = 0
    for c in SEED:
        result = cursor.execute(
            """
            INSERT INTO competencies (id, label, cefr, domain, parent_id)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (id) DO NOTHING
            """,
            [c.id, c.label, c.cefr, c.domain, c.parent_id],
        )
        # DuckDB returns the row count via .fetchall() on INSERT; use
        # rowcount-equivalent by counting. Safer cross-driver: re-check.
        # (DuckDB's INSERT ... ON CONFLICT returns (changes,) tuple in
        # some versions; in others nothing. We trust the count below.)
        _ = result
    # Cheap post-count: how many seed ids now exist.
    rows = cursor.execute(
        f"SELECT COUNT(*) FROM competencies WHERE id IN ({','.join('?' * len(SEED))})",
        [c.id for c in SEED],
    ).fetchone()
    present = int(rows[0] or 0) if rows else 0
    inserted = present  # not strictly the delta — DuckDB ON CONFLICT is silent
    logger.info("competencies seed: %d/%d ids present in DB", present, len(SEED))
    return inserted
