"""Abstraksi embedding + implementasi e5.

`Embedder` adalah seam untuk Open/Closed & Dependency Inversion: service layer
hanya tahu interface ini, jadi naik tier model (e5-base/large) atau ganti
provider tidak menyentuh store/bot/recall.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensi vektor keluaran."""

    @abstractmethod
    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed dokumen yang diindeks. Bentuk: (len(texts), dim), float32, ternormalisasi."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed query pencarian. Bentuk: (dim,), float32, ternormalisasi."""


class E5Embedder(Embedder):
    """Model family `intfloat/multilingual-e5-*` via sentence-transformers.

    Dua hal yang gampang bikin bug diam-diam (§7) dan ditangani di sini:
      - Prefix wajib: "passage: " untuk dokumen, "query: " untuk query.
      - Normalisasi L2 sebelum dipakai → cosine == dot product, threshold konsisten.
    """

    _PASSAGE_PREFIX = "passage: "
    _QUERY_PREFIX = "query: "

    def __init__(self, model_name: str, expected_dim: int | None = None) -> None:
        # Import lokal: sentence-transformers/torch berat, jangan dibebankan ke
        # modul yang cuma butuh tipe.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        if expected_dim is not None and expected_dim != self._dim:
            raise ValueError(
                f"Dimensi model {model_name} = {self._dim}, "
                f"tapi config minta {expected_dim}. Samakan Config.embedding_dim."
            )

    @property
    def dim(self) -> int:
        return self._dim

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        prefixed = [prefix + t for t in texts]
        vecs = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        return self._encode(texts, self._PASSAGE_PREFIX)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text], self._QUERY_PREFIX)[0]
