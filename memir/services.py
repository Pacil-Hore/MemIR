"""Service layer: orkestrasi embedder + store, bebas dari Discord.

Dipisah dari bot.py (SRP) supaya logika ingest/recall bisa diuji tanpa koneksi
Discord, dan bergantung pada abstraksi Embedder/MemoryStore (DIP), bukan
implementasi konkret.
"""

from __future__ import annotations

from .embedder import Embedder
from .models import MessageRecord, SearchResult
from .store import MemoryStore


class IngestService:
    def __init__(self, embedder: Embedder, store: MemoryStore) -> None:
        self._embedder = embedder
        self._store = store

    def ingest(self, record: MessageRecord) -> bool:
        if self._store.has(record.id):
            return False
        vec = self._embedder.embed_passages([record.content])[0]
        return self._store.add(record, vec)

    def ingest_many(self, records: list[MessageRecord]) -> int:
        new = [r for r in records if not self._store.has(r.id)]
        if not new:
            return 0
        vecs = self._embedder.embed_passages([r.content for r in new])
        return self._store.add_many(new, vecs)


class RecallService:
    def __init__(
        self,
        embedder: Embedder,
        store: MemoryStore,
        top_k: int,
        threshold: float,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._threshold = threshold

    def recall(self, query: str, guild_id: str, user_id: str | None = None) -> list[SearchResult]:
        qvec = self._embedder.embed_query(query)
        return self._store.search(qvec, self._top_k, self._threshold, guild_id, user_id)
