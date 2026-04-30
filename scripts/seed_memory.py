#!/usr/bin/env python
"""Seed ``data/memory/`` with starter learner files.

Idempotent: never overwrites a non-empty file. Safe to re-run.

Writes:

- ``data/memory/learner/profile.md`` — a short prompt for the learner
  to fill in by hand. Read by the open_turn RecallPolicy as a Tier 1
  section so Mira always knows who she's talking to.
- ``data/memory/learner/goal.json`` — a parsed goal scaffold. The
  Phase B onboarding flow will overwrite this; the seed exists so the
  RecallPolicy code paths have something to read on a fresh checkout.

Run::

    python scripts/seed_memory.py
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.config import PROJECT_ROOT


PROFILE_TEMPLATE = """\
# Wer bist du?

Ein paar Sätze über dich, die Mira nutzt, um den Unterricht
massgeschneidert zu halten. Bitte direkt hier editieren — Mira liest
diese Datei am Anfang jeder Sitzung.

- **Name:**
- **Sprachstand (CEFR):**
- **Aktuelles Ziel:** (z. B. „Approbationsprüfung in 6 Monaten")
- **Was du heute können möchtest:** (1–2 konkrete Sätze)
- **Was dich begeistert:** (Themen, an denen du gerne lernst)
- **Was dich anstrengt:** (Bereiche, die Mira sanft angehen sollte)
"""

GOAL_TEMPLATE: dict = {
    "raw_text": "",
    "target_cefr": "B2",
    "domain": None,
    "deadline_iso": None,
    "weekly_hours": 5,
    "must_haves": [],
    "nice_to_haves": [],
}


def _write_if_new(path: Path, body: str) -> bool:
    """Write ``body`` to ``path`` only when the file is missing or empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return False
    path.write_text(body, encoding="utf-8")
    return True


def main() -> int:
    learner = PROJECT_ROOT / "data" / "memory" / "learner"
    learner.mkdir(parents=True, exist_ok=True)

    wrote_profile = _write_if_new(learner / "profile.md", PROFILE_TEMPLATE)
    wrote_goal = _write_if_new(
        learner / "goal.json",
        json.dumps(GOAL_TEMPLATE, ensure_ascii=False, indent=2) + "\n",
    )

    print(
        "seed_memory: profile.md={} goal.json={}".format(
            "wrote" if wrote_profile else "kept",
            "wrote" if wrote_goal else "kept",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
