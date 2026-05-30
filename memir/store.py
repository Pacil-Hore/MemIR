"""Persistensi + similarity search.

`MemoryStore` adalah seam upgrade (§7): tukar SqliteStore dengan FAISS/Qdrant
tanpa menyentuh service/bot. SqliteStore menyimpan metadata + embedding sebagai
BLOB di satu row (atomic, anti-desync), dan menjaga matriks NumPy in-memory
untuk cosine = dot product atas vektor ternormalisasi.
"""

from __future__ import annotations

import os
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
        guild_id: str,
        user_id: str | None = None,
    ) -> list[SearchResult]: ...

    @abstractmethod
    def count(self) -> int: ...


class SqliteStore(MemoryStore):
    def __init__(self, db_path: str, dim: int) -> None:
        self._dim = dim
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

        # Cache in-memory, sejajar per indeks:
        #   _order[i]   -> id pesan ke-i
        #   _vectors[i] -> embedding ke-i
        # _records: id -> MessageRecord. cache turunan (_matrix, _guild_arr,
        # _user_arr) dibangun lazy dan di-invalidate tiap ada insert.
        self._order: list[str] = []
        self._vectors: list[np.ndarray] = []
        self._records: dict[str, MessageRecord] = {}
        self._matrix: np.ndarray | None = None
        self._guild_arr: np.ndarray | None = None
        self._user_arr: np.ndarray | None = None
        self._load()

    # --- setup ---------------------------------------------------------------
    def _create_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id         TEXT PRIMARY KEY,
                guild_id   TEXT NOT NULL,
                user_id    TEXT NOT NULL,
                username   TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                content    TEXT NOT NULL,
                timestamp  INTEGER NOT NULL,
                embedding  BLOB NOT NULL
            )
            """
        )
        # Migration: tambah guild_id kalau upgrade dari schema lama.
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        if "guild_id" not in cols:
            self._conn.execute(
                "ALTER TABLE messages ADD COLUMN guild_id TEXT NOT NULL DEFAULT ''"
            )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_guild ON messages(guild_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id)"
        )
        self._conn.commit()

    def _load(self) -> None:
        rows = self._conn.execute(
            "SELECT id, guild_id, user_id, username, channel_id, content, timestamp, embedding "
            "FROM messages ORDER BY timestamp ASC"
        ).fetchall()
        for id_, guild_id, user_id, username, channel_id, content, timestamp, blob in rows:
            record = MessageRecord(
                id=id_,
                guild_id=guild_id,
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
        self._matrix = None
        self._guild_arr = None
        self._user_arr = None

    def _ensure_caches(self) -> None:
        if self._matrix is None:
            self._matrix = (
                np.vstack(self._vectors)
                if self._vectors
                else np.empty((0, self._dim), dtype=np.float32)
            )
        if self._guild_arr is None:
            self._guild_arr = np.array(
                [self._records[i].guild_id for i in self._order], dtype=object
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
            "INSERT OR IGNORE INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.guild_id,
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
                "INSERT OR IGNORE INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.guild_id,
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
        guild_id: str,
        user_id: str | None = None,
    ) -> list[SearchResult]:
        if not self._order:
            return []
        self._ensure_caches()

        scores = self._matrix @ np.asarray(query_vec, dtype=np.float32)
        idxs = np.arange(len(self._order))

        # Scope ke guild pemanggil (selalu).
        guild_mask = self._guild_arr == guild_id
        idxs, scores = idxs[guild_mask], scores[guild_mask]

        # Filter author (opsional).
        if user_id is not None:
            user_mask = self._user_arr[idxs] == user_id
            idxs, scores = idxs[user_mask], scores[user_mask]

        keep = scores >= threshold
        idxs, scores = idxs[keep], scores[keep]
        if idxs.size == 0:
            return []

        top = np.argsort(-scores)[:top_k]
        return [
            SearchResult(self._records[self._order[int(idxs[t])]], float(scores[t]))
            for t in top
        ]

    def count(self) -> int:
        return len(self._order)
