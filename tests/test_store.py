"""Tes store: ranking, threshold, top-k, filter author, idempotensi, persistensi."""

from __future__ import annotations

import numpy as np

from memir.models import MessageRecord
from memir.store import SqliteStore


def _norm(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _rec(i: int, user_id: str) -> MessageRecord:
    return MessageRecord(str(i), user_id, "u" + user_id, "c1", f"msg{i}", i)


def test_ranking_and_threshold():
    store = SqliteStore(":memory:", dim=3)
    store.add(_rec(1, "A"), _norm([1.0, 0.05, 0]))
    store.add(_rec(2, "B"), _norm([0.9, 0.1, 0]))
    store.add(_rec(3, "A"), _norm([0, 1.0, 0]))  # topik beda

    q = _norm([1.0, 0, 0])
    results = store.search(q, top_k=5, threshold=0.8)
    assert [r.record.id for r in results] == ["1", "2"]
    assert results[0].score >= results[1].score


def test_idempotent_insert():
    store = SqliteStore(":memory:", dim=3)
    assert store.add(_rec(1, "A"), _norm([1, 0, 0])) is True
    assert store.add(_rec(1, "A"), _norm([0, 1, 0])) is False
    assert store.count() == 1


def test_filter_by_author():
    store = SqliteStore(":memory:", dim=3)
    store.add(_rec(1, "A"), _norm([1.0, 0.05, 0]))
    store.add(_rec(2, "B"), _norm([0.9, 0.1, 0]))

    q = _norm([1.0, 0, 0])
    results = store.search(q, top_k=5, threshold=0.8, user_id="A")
    assert [r.record.id for r in results] == ["1"]


def test_top_k_limit():
    store = SqliteStore(":memory:", dim=3)
    for i in range(10):
        store.add(_rec(i, "A"), _norm([1.0, i * 0.001, 0]))
    assert len(store.search(_norm([1, 0, 0]), top_k=3, threshold=0.5)) == 3


def test_empty_when_below_threshold():
    store = SqliteStore(":memory:", dim=3)
    store.add(_rec(1, "A"), _norm([1, 0, 0]))
    assert store.search(_norm([0, 1, 0]), top_k=5, threshold=0.5) == []


def test_persistence_roundtrip(tmp_path):
    db = str(tmp_path / "memir.db")
    s1 = SqliteStore(db, dim=3)
    s1.add(MessageRecord("10", "A", "ua", "c1", "halo dunia", 100), _norm([1.0, 0.1, 0]))
    s1._conn.close()

    s2 = SqliteStore(db, dim=3)  # reload dari disk (BLOB → NumPy)
    assert s2.count() == 1
    res = s2.search(_norm([1, 0, 0]), top_k=5, threshold=0.8)
    assert res and res[0].record.content == "halo dunia"
    s2._conn.close()
