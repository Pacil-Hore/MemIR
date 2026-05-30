"""Entry point: `python -m memir`.

Composition root — satu-satunya tempat implementasi konkret dirakit. Semua
lapisan lain bergantung pada abstraksi (Embedder, MemoryStore).
"""

from __future__ import annotations

from .bot import build_bot
from .config import Config
from .embedder import E5Embedder
from .services import IngestService, RecallService
from .store import SqliteStore


def main() -> None:
    config = Config.from_env()

    print(f"Memuat model embedding: {config.model_name} …")
    embedder = E5Embedder(config.model_name, config.embedding_dim)
    store = SqliteStore(config.db_path, embedder.dim)

    ingest = IngestService(embedder, store)
    recall = RecallService(embedder, store, config.top_k, config.threshold)

    bot = build_bot(config, ingest, recall, store)
    bot.run(config.discord_token)


if __name__ == "__main__":
    main()
