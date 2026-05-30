"""Tes service layer pakai FakeEmbedder (tanpa torch)."""

from __future__ import annotations

from memir.models import MessageRecord
from memir.services import IngestService, RecallService
from memir.store import SqliteStore

from .fakes import FakeEmbedder


def _rec(id_: str, user_id: str, content: str) -> MessageRecord:
    return MessageRecord(id_, user_id, "u" + user_id, "c1", content, int(id_))


def _setup():
    embedder = FakeEmbedder()
    store = SqliteStore(":memory:", dim=embedder.dim)
    ingest = IngestService(embedder, store)
    recall = RecallService(embedder, store, top_k=5, threshold=0.1)
    return ingest, recall, store


def test_ingest_dedups():
    ingest, _, store = _setup()
    assert ingest.ingest(_rec("1", "A", "deadline proyek besok")) is True
    assert ingest.ingest(_rec("1", "A", "deadline proyek besok")) is False
    assert store.count() == 1


def test_ingest_many_skips_existing():
    ingest, _, store = _setup()
    ingest.ingest(_rec("1", "A", "deadline proyek"))
    added = ingest.ingest_many(
        [_rec("1", "A", "deadline proyek"), _rec("2", "B", "cuaca hujan")]
    )
    assert added == 1
    assert store.count() == 2


def test_recall_finds_semantic_overlap():
    ingest, recall, _ = _setup()
    ingest.ingest_many(
        [
            _rec("1", "A", "deadline proyek besok"),
            _rec("2", "B", "cuaca hujan deras"),
            _rec("3", "A", "rapat tim pagi"),
        ]
    )
    results = recall.recall("deadline proyek")
    assert results
    assert results[0].record.id == "1"


def test_recall_author_filter():
    ingest, recall, _ = _setup()
    ingest.ingest_many(
        [
            _rec("1", "A", "deadline proyek besok"),
            _rec("2", "B", "deadline proyek juga"),
        ]
    )
    results = recall.recall("deadline proyek", user_id="B")
    assert [r.record.id for r in results] == ["2"]
