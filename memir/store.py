"""Persistensi + similarity search.

`MemoryStore` adalah seam upgrade (§7): tukar SqliteStore dengan FAISS/Qdrant
tanpa menyentuh service/bot. SqliteStore menyimpan metadata + embedding sebagai
BLOB di satu row (atomic, anti-desync), dan menjaga matriks NumPy in-memory
untuk cosine = dot product atas vektor ternormalisasi.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod

import numpy as np

from .models import MessageRecord, SearchResult


class MemoryStore(ABC):
    @abstractmethod
    def has(self, message_id: str) -> bool: ...

    @abstractmethod
    def add(self, record: MessageRecord, embedding: np.ndarray) -> bool:
        """Simpan satu pesan. Return False kalau id sudah ada (idempotent)."""

    @abstractmethod
    def add_many(self, records: list[MessageRecord], embeddings: np.ndarray) -> int:
        """Simpan batch pesan baru. Return jumlah yang benar-benar masuk."""

    @abstractmethod
    def search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        threshold: float,
        user_id: str | None = None,
    ) -> list[SearchResult]: ...

    @abstractmethod
    def count(self) -> int: ...


class SqliteStore(MemoryStore):
    def __init__(self, db_path: str, dim: int) -> None:
        self._dim = dim
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

        # Cache in-memory, sejajar per indeks:
        #   _order[i]  -> id pesan ke-i
        #   _vectors[i]-> embedding ke-i
        # _records: id -> MessageRecord. _matrix/_user_arr cache turunan (lazy).
        self._order: list[str] = []
        self._vectors: list[np.ndarray] = []
        self._records: dict[str, MessageRecord] = {}
        self._matrix: np.ndarray | None = None
        self._user_arr: np.ndarray | None = None
        self._load()

    # --- setup ---------------------------------------------------------------
    def _create_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                username   TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                content    TEXT NOT NULL,
                timestamp  INTEGER NOT NULL,
                embedding  BLOB NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id)"
        )
        self._conn.commit()

    def _load(self) -> None:
        rows = self._conn.execute(
            "SELECT id, user_id, username, channel_id, content, timestamp, embedding "
            "FROM messages ORDER BY timestamp ASC"
        ).fetchall()
        for id_, user_id, username, channel_id, content, timestamp, blob in rows:
            record = MessageRecord(
                id=id_,
                user_id=user_id,
                username=username,
                channel_id=channel_id,
                content=content,
                timestamp=timestamp,
            )
            self._append_memory(record, self._blob_to_vec(blob))

    # --- helpers -------------------------------------------------------------
    @staticmethod
    def _vec_to_blob(vec: np.ndarray) -> bytes:
        return np.asarray(vec, dtype=np.float32).tobytes()

    def _blob_to_vec(self, blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32).reshape(self._dim)

    def _append_memory(self, record: MessageRecord, vec: np.ndarray) -> None:
        self._order.append(record.id)
        self._vectors.append(vec)
        self._records[record.id] = record
        self._matrix = None  # invalidate cache turunan
        self._user_arr = None

    def _ensure_caches(self) -> None:
        if self._matrix is None:
            self._matrix = (
                np.vstack(self._vectors)
                if self._vectors
                else np.empty((0, self._dim), dtype=np.float32)
            )
        if self._user_arr is None:
            self._user_arr = np.array(
                [self._records[i].user_id for i in self._order], dtype=object
            )

    # --- API ----------------------------------------------------------------
    def has(self, message_id: str) -> bool:
        return message_id in self._records

    def add(self, record: MessageRecord, embedding: np.ndarray) -> bool:
        if self.has(record.id):
            return False
        self._conn.execute(
            "INSERT OR IGNORE INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.user_id,
                record.username,
                record.channel_id,
                record.content,
                record.timestamp,
                self._vec_to_blob(embedding),
            ),
        )
        self._conn.commit()
        self._append_memory(record, np.asarray(embedding, dtype=np.float32))
        return True

    def add_many(self, records: list[MessageRecord], embeddings: np.ndarray) -> int:
        added = 0
        for record, vec in zip(records, embeddings):
            if self.has(record.id):
                continue
            self._conn.execute(
                "INSERT OR IGNORE INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.user_id,
                    record.username,
                    record.channel_id,
                    record.content,
                    record.timestamp,
                    self._vec_to_blob(vec),
                ),
            )
            self._append_memory(record, np.asarray(vec, dtype=np.float32))
            added += 1
        self._conn.commit()
        return added

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        threshold: float,
        user_id: str | None = None,
    ) -> list[SearchResult]:
        if not self._order:
            return []
        self._ensure_caches()

        # Vektor ternormalisasi → dot product == cosine similarity.
        scores = self._matrix @ np.asarray(query_vec, dtype=np.float32)
        idxs = np.arange(len(self._order))

        if user_id is not None:  # filter author dulu, baru ranking (§5)
            mask = self._user_arr == user_id
            idxs, scores = idxs[mask], scores[mask]

        keep = scores >= threshold  # buang yang ngawur (§6)
        idxs, scores = idxs[keep], scores[keep]
        if idxs.size == 0:
            return []

        top = np.argsort(-scores)[:top_k]  # batasi yang kebanyakan (§6)
        return [
            SearchResult(self._records[self._order[int(idxs[t])]], float(scores[t]))
            for t in top
        ]

    def count(self) -> int:
        return len(self._order)
