"""Konstanta kalibrasi & konfigurasi runtime, dikumpulkan di satu tempat (§7)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    discord_token: str

    db_path: str = "memir.db"
    model_name: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384
    command_prefix: str = "!"

    top_k: int = 5

    # Titik kalibrasi utama (§6). e5 + embedding ter-normalisasi cenderung kasih
    # cosine yang tinggi & rapat, jadi ambang efektifnya lebih tinggi dari angka
    # "umum" 0.3–0.5. Mulai di 0.80, lalu setel pakai query uji.
    threshold: float = 0.80

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        token = os.getenv("DISCORD_TOKEN")
        if not token:
            raise RuntimeError(
                "DISCORD_TOKEN belum diset. Salin .env.example jadi .env "
                "lalu isi token bot."
            )
        # Override opsional via env — dipakai saat deploy (volume Docker, dll).
        return cls(
            discord_token=token,
            db_path=os.getenv("MEMIR_DB_PATH", cls.db_path),
            model_name=os.getenv("MEMIR_MODEL", cls.model_name),
            threshold=float(os.getenv("MEMIR_THRESHOLD", cls.threshold)),
        )
