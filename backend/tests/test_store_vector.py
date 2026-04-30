"""Tests for backend/memory/store_vector.py (PIVOT_ROADMAP §B.14)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable

import pytest

from backend.memory import manager as memory_manager
from backend.memory import store_vector as vector_store
from backend.memory.store_vector import (
    LocalEmbeddedVectorIndex,
    NullVectorIndex,
    VectorRow,
)


# ─── Fake embedder ──────────────────────────────────────────────────────


class _FakeEmbedder:
    """Deterministic 4-d embedder: one axis per anchor word.

    `Hund` -> dog axis, `Auto` -> car axis, etc. Unknown text gets a
    small random component on a fifth axis. This is just enough to make
    the cosine ranking test meaningful without needing fastembed.
    """

    name = "fake"
    dim = 4

    _ANCHORS = {
        "hund": (1.0, 0.0, 0.0, 0.0),
        "katze": (0.95, 0.05, 0.0, 0.0),
        "auto": (0.0, 1.0, 0.0, 0.0),
        "fahrrad": (0.0, 0.95, 0.05, 0.0),
        "haus": (0.0, 0.0, 1.0, 0.0),
        "wohnung": (0.0, 0.0, 0.95, 0.05),
    }

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        v = [0.0, 0.0, 0.0, 0.05]
        words = text.lower().split()
        for word in words:
            stripped = word.strip(".,!?;:\"'")
            anchor = self._ANCHORS.get(stripped)
            if anchor:
                for i, comp in enumerate(anchor):
                    v[i] += comp
        norm = math.sqrt(sum(x * x for x in v))
        if norm > 0:
            v = [x / norm for x in v]
        return v


@pytest.fixture()
def fake_index(tmp_path: Path) -> LocalEmbeddedVectorIndex:
    """Always-fresh local index pinned to MemoryManager root with fake embedder."""
    memory_manager.reset_for_tests(root=tmp_path / "vec_root")
    return vector_store.reset_for_tests(embedder=_FakeEmbedder())  # type: ignore[return-value]


# ─── Protocol contract ──────────────────────────────────────────────────


def test_null_index_satisfies_protocol():
    null = NullVectorIndex()
    assert null.add([VectorRow(id="x", text="t", source="evidence")]) == 0
    assert null.search("anything") == []
    assert null.count() == 0
    assert null.rebuild_from_jsonl("transcripts") == 0


# ─── add + count ────────────────────────────────────────────────────────


def test_add_and_count(fake_index: LocalEmbeddedVectorIndex) -> None:
    rows = [
        VectorRow(id="e1", text="Mein Hund ist groß.", source="evidence"),
        VectorRow(id="e2", text="Das Auto fährt schnell.", source="evidence"),
    ]
    added = fake_index.add(rows)
    assert added == 2
    assert fake_index.count() == 2
    assert fake_index.count(source="evidence") == 2
    assert fake_index.count(source="transcripts") == 0


def test_add_is_idempotent_on_id(fake_index: LocalEmbeddedVectorIndex) -> None:
    base = VectorRow(id="dup", text="Mein Hund ist groß.", source="evidence")
    fake_index.add([base])
    fake_index.add([base])
    assert fake_index.count() == 1


# ─── search / cosine ranking ────────────────────────────────────────────


def test_search_ranks_by_semantic_similarity(
    fake_index: LocalEmbeddedVectorIndex,
) -> None:
    fake_index.add(
        [
            VectorRow(id="e1", text="Mein Hund ist groß.", source="evidence"),
            VectorRow(id="e2", text="Das Auto fährt schnell.", source="evidence"),
            VectorRow(id="e3", text="Mein neues Haus.", source="evidence"),
        ]
    )
    hits = fake_index.search("Katze schläft", top_k=3)
    assert hits, "search should return at least one hit"
    assert hits[0].row.id == "e1", f"expected e1 (Hund) first, got {hits[0].row.id}"


def test_search_filter_by_metadata(fake_index: LocalEmbeddedVectorIndex) -> None:
    fake_index.add(
        [
            VectorRow(
                id="e1",
                text="Mein Hund ist groß.",
                source="evidence",
                metadata={"competency_id": "vocab.tiere"},
            ),
            VectorRow(
                id="e2",
                text="Mein Auto ist neu.",
                source="evidence",
                metadata={"competency_id": "vocab.fahrzeuge"},
            ),
        ]
    )
    hits = fake_index.search(
        "Tier", top_k=5, filter={"competency_id": "vocab.tiere"}
    )
    assert [h.row.id for h in hits] == ["e1"]


def test_search_filter_by_source(fake_index: LocalEmbeddedVectorIndex) -> None:
    fake_index.add(
        [
            VectorRow(id="e1", text="Hund.", source="evidence"),
            VectorRow(id="t1", text="Hund.", source="transcripts"),
        ]
    )
    hits = fake_index.search("Hund", top_k=5, source="transcripts")
    assert all(h.row.source == "transcripts" for h in hits)


# ─── Persistence (parquet round-trip) ───────────────────────────────────


def test_parquet_round_trip(tmp_path: Path) -> None:
    embedder = _FakeEmbedder()
    memory_manager.reset_for_tests(root=tmp_path / "vec_root")
    idx = vector_store.reset_for_tests(embedder=embedder)
    idx.add([VectorRow(id="e1", text="Mein Hund.", source="evidence")])  # type: ignore[union-attr]

    # Build a fresh index pointed at the same root and confirm the row is loaded.
    fresh = LocalEmbeddedVectorIndex(
        root=Path(memory_manager.get_memory()._root),  # type: ignore[union-attr]
        embedder=embedder,
    )
    assert fresh.count() == 1
    hits = fresh.search("Hund", top_k=1)
    assert hits and hits[0].row.id == "e1"


# ─── rebuild_from_jsonl ─────────────────────────────────────────────────


def test_rebuild_from_jsonl_transcripts(
    fake_index: LocalEmbeddedVectorIndex, tmp_path: Path
) -> None:
    konv_dir = Path(memory_manager.get_memory()._root) / "konversation"  # type: ignore[union-attr]
    konv_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": "t1", "role": "user", "text": "Mein Hund spielt.", "session_id": "s1"},
        {"id": "t2", "role": "assistant", "text": "Das Auto ist schnell.", "session_id": "s1"},
    ]
    (konv_dir / "transcripts.0001.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    added = fake_index.rebuild_from_jsonl("transcripts")
    assert added == 2
    assert fake_index.count(source="transcripts") == 2


def test_rebuild_truncates_then_refills(
    fake_index: LocalEmbeddedVectorIndex,
) -> None:
    fake_index.add([VectorRow(id="e1", text="Hund.", source="evidence")])
    # Now point evidence/ledger.jsonl at a fresh row only
    ledger = Path(memory_manager.get_memory()._root) / "evidence" / "ledger.jsonl"  # type: ignore[union-attr]
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        json.dumps(
            {
                "id": "e2",
                "competency_id": "vocab.tiere",
                "surface_form": "Katze.",
                "occurred_at": "2026-01-01T00:00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fake_index.rebuild_from_jsonl("evidence")
    assert fake_index.count(source="evidence") == 1


def test_rebuild_unknown_source_is_noop(
    fake_index: LocalEmbeddedVectorIndex,
) -> None:
    assert fake_index.rebuild_from_jsonl("not_a_source") == 0


# ─── get_index() singleton + Null fallback ──────────────────────────────


def test_get_index_returns_null_when_offline(monkeypatch, tmp_path: Path) -> None:
    memory_manager.reset_for_tests(root=tmp_path / "vec_root")
    monkeypatch.setenv("LINGUAMATE_OFFLINE", "1")
    vector_store._INDEX = None  # noqa: SLF001
    idx = vector_store.get_index()
    assert isinstance(idx, NullVectorIndex)
    monkeypatch.delenv("LINGUAMATE_OFFLINE", raising=False)
    vector_store._INDEX = None  # noqa: SLF001


def test_get_index_returns_null_when_backend_explicit(
    monkeypatch, tmp_path: Path
) -> None:
    memory_manager.reset_for_tests(root=tmp_path / "vec_root")
    monkeypatch.setenv("LINGUAMATE_VECTOR_BACKEND", "null")
    vector_store._INDEX = None  # noqa: SLF001
    idx = vector_store.get_index()
    assert isinstance(idx, NullVectorIndex)
    monkeypatch.delenv("LINGUAMATE_VECTOR_BACKEND", raising=False)
    vector_store._INDEX = None  # noqa: SLF001
