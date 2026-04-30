"""Bayesian mastery model — one posterior per competency.

We treat each evidence event as a Bernoulli trial with quality in [0, 1]
(quality 1.0 = clean success, 0.0 = clean fail, fractional = scaffolded /
partial). The conjugate Beta prior gives a posterior in closed form:

    alpha_n+1 = alpha_n + quality
    beta_n+1  = beta_n  + (1 - quality)

with mean = alpha / (alpha + beta) and variance = alpha*beta /
((alpha+beta)^2 * (alpha+beta+1)).

Storage shortcut: we don't persist alpha/beta directly. We persist
``mu = mean`` and ``sigma = sqrt(variance)`` (both in [0, 1]) and
recover (alpha, beta) at update time via the standard Beta moment
inversion. This keeps the SQL schema human-readable (mu reads as a
confidence percentage) while the math stays principled.

The forgetting curve (PIVOT_ROADMAP.md §7.6 pattern 6) lives here too:
old, high-confidence rows get a tiny variance-inflation each time we
update (so confidence drifts back toward uncertainty if the learner
stops practicing). It's intentionally weak — Phase B replaces it with
a proper time-since-last-evidence half-life.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from typing import Optional

from backend.models import db

logger = logging.getLogger(__name__)


# Defaults match the schema (mu=0.5, sigma=0.25) — i.e. uniform-ish prior
# on a scale where 0=zero-skill and 1=full mastery.
_DEFAULT_MU = 0.5
_DEFAULT_SIGMA = 0.25
_MIN_SIGMA = 0.04   # never claim near-perfect certainty; some noise always remains
_MAX_SIGMA = 0.50   # the Beta family caps variance here when mu is around 0.5


@dataclass(frozen=True)
class MasteryRow:
    """Public representation of a competency's mastery state.

    ``confidence`` and ``variance`` are the 0..100 ints the API surfaces
    (rounded mu * 100 and sigma * 100). The float fields are kept for
    the agent's recall path which needs fine-grained ranking.
    """

    competency_id: str
    mu: float
    sigma: float
    evidence_count: int

    @property
    def confidence(self) -> int:
        return round(max(0.0, min(1.0, self.mu)) * 100)

    @property
    def variance(self) -> int:
        return round(max(0.0, min(1.0, self.sigma)) * 100)


# ─── (mu, sigma) <-> (alpha, beta) inversion ─────────────────────────────


def _ms_to_ab(mu: float, sigma: float) -> tuple[float, float]:
    """Invert (mean, std) of a Beta into (alpha, beta).

    From Var = mu*(1-mu)/(alpha+beta+1):
        alpha + beta = mu*(1-mu) / sigma^2  -  1
    Then alpha = mu * (alpha + beta), beta = (1-mu) * (alpha + beta).
    """
    mu_c = min(max(mu, 1e-3), 1 - 1e-3)
    var = max(sigma, _MIN_SIGMA) ** 2
    # Lower bound on alpha+beta to keep the Beta well-defined.
    nu = max(mu_c * (1 - mu_c) / var - 1.0, 1.0)
    alpha = mu_c * nu
    beta = (1 - mu_c) * nu
    return alpha, beta


def _ab_to_ms(alpha: float, beta: float) -> tuple[float, float]:
    total = alpha + beta
    mu = alpha / total
    var = alpha * beta / (total * total * (total + 1.0))
    sigma = math.sqrt(max(var, 0.0))
    sigma = max(_MIN_SIGMA, min(_MAX_SIGMA, sigma))
    return mu, sigma


# ─── Public API ──────────────────────────────────────────────────────────


def get(competency_id: str) -> MasteryRow:
    """Return the current mastery row, materialising defaults if absent.

    We do NOT insert a default row for read-only access — the table stays
    sparse. Callers that need persistence should call ``update``.
    """
    row = db.cursor().execute(
        """
        SELECT mu, sigma, evidence_count
        FROM mastery
        WHERE competency_id = ?
        """,
        [competency_id],
    ).fetchone()
    if row is None:
        return MasteryRow(
            competency_id=competency_id,
            mu=_DEFAULT_MU,
            sigma=_DEFAULT_SIGMA,
            evidence_count=0,
        )
    return MasteryRow(
        competency_id=competency_id,
        mu=float(row[0]),
        sigma=float(row[1]),
        evidence_count=int(row[2] or 0),
    )


def update(
    competency_id: str,
    quality: float,
    *,
    source: str,
    surface_form: Optional[str] = None,
    notes: Optional[str] = None,
) -> MasteryRow:
    """Append one evidence event and recompute the posterior.

    Returns the new MasteryRow. Both writes (evidence INSERT + mastery
    UPSERT) execute on the same cursor so they share a transaction
    on DuckDB's default autocommit mode.

    ``quality`` is clamped to [0, 1]. ``source`` is required and must be
    one of the convention strings ('conversation' | 'drill' | 'reader' |
    'capture' | 'agent_pre' | 'agent_post') — we don't enforce in code so
    Phase B can add new sources without a migration.
    """
    q = min(max(float(quality), 0.0), 1.0)
    cursor = db.cursor()

    # 1. Append to the immutable evidence ledger.
    evidence_id = str(uuid.uuid4())
    cursor.execute(
        """
        INSERT INTO evidence
            (id, competency_id, surface_form, quality, source, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [evidence_id, competency_id, surface_form, q, source, notes],
    )

    # 2. Read the current posterior (or default), update, write back.
    current = get(competency_id)
    alpha, beta = _ms_to_ab(current.mu, current.sigma)
    alpha += q
    beta += 1.0 - q
    mu, sigma = _ab_to_ms(alpha, beta)

    cursor.execute(
        """
        INSERT INTO mastery
            (competency_id, mu, sigma, last_evidence, evidence_count)
        VALUES (?, ?, ?, now(), ?)
        ON CONFLICT (competency_id) DO UPDATE SET
            mu = EXCLUDED.mu,
            sigma = EXCLUDED.sigma,
            last_evidence = EXCLUDED.last_evidence,
            evidence_count = EXCLUDED.evidence_count
        """,
        [competency_id, mu, sigma, current.evidence_count + 1],
    )

    # 3. Refresh the scheduler's next_review_at so the Atrium can sort
    # by what's due without recomputing on every read. Imported lazily
    # to avoid a top-level cycle (scheduler imports model.get).
    try:
        from backend.mastery import scheduler as _scheduler  # noqa: WPS433

        _scheduler.persist_next_review(competency_id)
    except Exception as exc:  # pragma: no cover — never block the write
        logger.warning("scheduler.persist_next_review failed: %s", exc)

    # 4. Refresh the per-competency fact sheet (PIVOT_ROADMAP §B.13).
    # Best-effort: an MD-write failure must never abort an evidence write.
    try:
        from backend.memory import factsheets as _factsheets  # noqa: WPS433

        _factsheets.update_factsheet(
            _factsheets.FactsheetEvidence(
                competency_id=competency_id,
                quality=q,
                surface_form=surface_form,
                notes=notes,
            ),
            _factsheets.FactsheetMastery(
                confidence=int(round(mu * 100)),
                variance=int(round(sigma * 100)),
            ),
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("factsheets.update_factsheet failed: %s", exc)

    # 5. Index the evidence text for semantic recall (PIVOT_ROADMAP §B.14).
    if surface_form and surface_form.strip():
        try:
            from backend.memory import store_vector as _store_vector  # noqa: WPS433

            _store_vector.get_index().add(
                [
                    _store_vector.VectorRow(
                        id=evidence_id,
                        text=surface_form,
                        source="evidence",
                        metadata={
                            "competency_id": competency_id,
                            "quality": q,
                            "source_kind": source,
                        },
                    )
                ]
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("vector index add (evidence) failed: %s", exc)

    return MasteryRow(
        competency_id=competency_id,
        mu=mu,
        sigma=sigma,
        evidence_count=current.evidence_count + 1,
    )


def list_top(
    *,
    domain: Optional[str] = None,
    limit: int = 10,
) -> list[MasteryRow]:
    """Return rows ordered by recency, optionally filtered by domain.

    Used by the agent's RecallPolicy (PIVOT_ROADMAP.md §7.5) when
    building Tier-1 context for ``open_turn``.
    """
    if domain is None:
        rows = db.cursor().execute(
            """
            SELECT m.competency_id, m.mu, m.sigma, m.evidence_count
            FROM mastery m
            ORDER BY m.last_evidence DESC NULLS LAST
            LIMIT ?
            """,
            [limit],
        ).fetchall()
    else:
        rows = db.cursor().execute(
            """
            SELECT m.competency_id, m.mu, m.sigma, m.evidence_count
            FROM mastery m
            JOIN competencies c ON c.id = m.competency_id
            WHERE c.domain = ?
            ORDER BY m.last_evidence DESC NULLS LAST
            LIMIT ?
            """,
            [domain, limit],
        ).fetchall()
    return [
        MasteryRow(
            competency_id=r[0],
            mu=float(r[1]),
            sigma=float(r[2]),
            evidence_count=int(r[3] or 0),
        )
        for r in rows
    ]
