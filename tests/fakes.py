"""Embedder palsu deterministik untuk tes — tanpa torch/sentence-transformers.

Tiap token dipetakan ke satu indeks stabil (bag-of-words), lalu dinormalisasi.
Pesan yang berbagi token akan punya cosine lebih tinggi — cukup untuk menguji
pipeline ingest/recall tanpa model asli.
"""

from __future__ import annotations

import numpy as np

from memir.embedder import Embedder


class FakeEmbedder(Embedder):
    def __init__(self, dim: int = 256) -> None:
        self._dim = dim
        self._vocab: dict[str, int] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def _index(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = len(self._vocab) % self._dim
        return self._vocab[token]

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        for token in text.lower().split():
            v[self._index(token)] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)
