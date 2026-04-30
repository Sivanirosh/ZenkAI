"""Mastery posterior + evidence ledger tests.

Covers the properties that matter for the tutor:
1. Default state is uncertain (mu = 0.5, sigma = 0.25).
2. A run of successes pushes mu up and sigma down (monotonic learning).
3. A run of failures pushes mu down.
4. Mixed evidence converges roughly to the empirical proportion.
5. Each update appends exactly one evidence row.
6. list_top returns rows ordered by most recent activity.
"""

from __future__ import annotations

import pytest

from backend.mastery import competencies, model
from backend.models import db


COMP = "verbs.modal"


def _evidence_count() -> int:
    return int(db.cursor().execute("SELECT COUNT(*) FROM evidence").fetchone()[0])


def test_default_state_is_uncertain() -> None:
    row = model.get(COMP)
    assert row.evidence_count == 0
    assert row.mu == pytest.approx(0.5, abs=0.01)
    assert row.sigma == pytest.approx(0.25, abs=0.01)


def test_repeated_success_raises_mu_and_lowers_sigma() -> None:
    initial = model.get(COMP)
    last = initial
    for _ in range(10):
        last = model.update(COMP, 1.0, source="drill")
    assert last.evidence_count == 10
    assert last.mu > initial.mu
    assert last.mu > 0.7  # 10 wins from a 0.5 baseline => clearly competent
    assert last.sigma < initial.sigma
    assert _evidence_count() == 10


def test_repeated_failure_lowers_mu() -> None:
    initial = model.get(COMP)
    last = initial
    for _ in range(10):
        last = model.update(COMP, 0.0, source="drill")
    assert last.evidence_count == 10
    assert last.mu < initial.mu
    assert last.mu < 0.3


def test_mixed_evidence_converges_to_empirical_proportion() -> None:
    # Equal mix of success/failure should keep mu near 0.5.
    for i in range(20):
        model.update(COMP, 1.0 if i % 2 == 0 else 0.0, source="drill")
    final = model.get(COMP)
    assert final.evidence_count == 20
    assert final.mu == pytest.approx(0.5, abs=0.05)


def test_partial_quality_is_clamped_and_recorded() -> None:
    out = model.update(COMP, 1.5, source="agent_post", surface_form="probe")
    assert out.evidence_count == 1
    # Recorded quality must be the clamped value (1.0).
    row = db.cursor().execute(
        "SELECT quality, source, surface_form FROM evidence WHERE competency_id = ?",
        [COMP],
    ).fetchone()
    assert row is not None
    assert float(row[0]) == pytest.approx(1.0)
    assert row[1] == "agent_post"
    assert row[2] == "probe"


def test_confidence_property_is_0_to_100_int() -> None:
    out = model.update(COMP, 1.0, source="drill")
    assert isinstance(out.confidence, int)
    assert 0 <= out.confidence <= 100


def test_list_top_orders_by_recency_and_filters_by_domain() -> None:
    # Touch a verb competency, then a medical one.
    model.update("verbs.modal", 0.8, source="drill")
    model.update("medical.symptoms", 0.6, source="conversation")

    top_all = model.list_top(limit=5)
    assert top_all[0].competency_id == "medical.symptoms"
    assert top_all[1].competency_id == "verbs.modal"

    top_medical = model.list_top(domain="medical", limit=5)
    assert [r.competency_id for r in top_medical] == ["medical.symptoms"]


def test_seed_competencies_inserts_v1_taxonomy() -> None:
    # The autouse in_memory_db fixture already seeded; just verify count.
    n = db.cursor().execute(
        "SELECT COUNT(*) FROM competencies WHERE id = ANY(?)",
        [[c.id for c in competencies.SEED]],
    ).fetchone()[0]
    assert n == len(competencies.SEED)
