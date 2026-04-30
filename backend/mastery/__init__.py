"""Mastery model: Bayesian skill confidence per competency.

Replaces XP/streaks (PIVOT_ROADMAP.md §5.5). Confidence rises only when
the learner produces the skill in context. Public surface:

- ``competencies.SEED`` — taxonomy seed (~30 ids covering CEFR + situational tags)
- ``competencies.seed_competencies()`` — idempotent insert into the DB
- ``model.update(competency_id, quality, source, surface_form?, notes?)`` — append evidence + recompute posterior
- ``model.get(competency_id)`` — fetch the current MasteryRow
- ``model.list_top(domain?, limit)`` — top-K rows (used by the agent's RecallPolicy)
"""

from __future__ import annotations
