"""Spacing function + scheduler persistence tests.

We don't pin specific day counts (the constants might evolve); we pin
the *properties* a sane scheduler must satisfy:

- Higher mu, all else equal, means a longer interval.
- Higher sigma (less certainty), all else equal, means a shorter interval.
- The interval is bounded between the floor and the ceiling.
- ``persist_next_review`` writes a value DuckDB can read back.
- ``due_competencies`` returns rows whose next_review_at has passed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from backend.mastery import model, scheduler
from backend.mastery.model import MasteryRow


def _row(*, mu: float, sigma: float) -> MasteryRow:
    return MasteryRow(competency_id="test", mu=mu, sigma=sigma, evidence_count=1)


# ─── Pure spacing function ───────────────────────────────────────────────


def test_interval_is_bounded() -> None:
    very_low = scheduler.interval_days(_row(mu=0.0, sigma=0.5))
    very_high = scheduler.interval_days(_row(mu=1.0, sigma=0.01))
    assert 0.5 <= very_low <= 90.0
    assert 0.5 <= very_high <= 90.0


def test_higher_mu_means_longer_interval() -> None:
    low = scheduler.interval_days(_row(mu=0.3, sigma=0.2))
    high = scheduler.interval_days(_row(mu=0.8, sigma=0.2))
    assert high > low


def test_higher_sigma_means_shorter_interval() -> None:
    confident = scheduler.interval_days(_row(mu=0.7, sigma=0.05))
    uncertain = scheduler.interval_days(_row(mu=0.7, sigma=0.30))
    assert confident > uncertain


def test_uncertain_low_confidence_floors_at_half_day() -> None:
    days = scheduler.interval_days(_row(mu=0.05, sigma=0.45))
    assert days >= 0.5


def test_high_confidence_low_variance_caps_at_ninety_days() -> None:
    days = scheduler.interval_days(_row(mu=1.0, sigma=0.01))
    assert days <= 90.0


def test_next_review_at_is_in_the_future() -> None:
    now = datetime(2026, 4, 30, 8, 0, tzinfo=timezone.utc)
    nxt = scheduler.next_review_at(_row(mu=0.5, sigma=0.25), now=now)
    assert nxt > now


# ─── Persistence + due_competencies ──────────────────────────────────────


def test_update_persists_a_next_review_at() -> None:
    model.update("verbs.modal", 1.0, source="drill")
    nxt = scheduler.persist_next_review("verbs.modal")
    assert nxt is not None
    assert nxt > datetime.now(timezone.utc)


def test_due_competencies_returns_rows_in_the_past() -> None:
    model.update("verbs.modal", 0.4, source="drill")
    # Force-set a past timestamp so the row is "due".
    from backend.models import db

    past = datetime.now(timezone.utc) - timedelta(hours=12)
    db.cursor().execute(
        "UPDATE mastery SET next_review_at = ? WHERE competency_id = ?",
        [past, "verbs.modal"],
    )
    due = scheduler.due_competencies(limit=5)
    ids = [r.competency_id for r in due]
    assert "verbs.modal" in ids


def test_due_competencies_excludes_unscheduled_rows() -> None:
    # No update yet -> no mastery row at all -> not in due list.
    due = scheduler.due_competencies(limit=5)
    assert due == []


def test_persist_skips_competencies_without_evidence() -> None:
    nxt = scheduler.persist_next_review("medical.symptoms")
    assert nxt is None
