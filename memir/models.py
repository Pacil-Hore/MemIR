"""Tipe data inti. Tanpa dependensi ke Discord/SQLite/embedding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MessageRecord:
    """Satu pesan yang diindeks. Cermin §4 (Skema Data) di docs/MemIR-plan.md.

    `embedding` sengaja TIDAK disimpan di sini — vektor hidup di lapisan store
    (BLOB di SQLite + NumPy in-memory), supaya logika domain tetap bebas NumPy.
    """

    id: str          # message ID Discord (unik, idempotent)
    guild_id: str    # server asal — scope isolation antar server
    user_id: str     # author — sumber kebenaran untuk filter opsional
    username: str    # hanya untuk display; bisa berubah, jangan dipakai filter
    channel_id: str  # konteks; dipakai untuk membangun link ke pesan
    content: str     # teks mentah pesan
    timestamp: int   # epoch detik; untuk sorting & recency


@dataclass(frozen=True)
class SearchResult:
    """Satu kandidat hasil recall beserta skor similarity-nya."""

    record: MessageRecord
    score: float
