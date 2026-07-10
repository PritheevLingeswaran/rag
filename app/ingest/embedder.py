"""Embedding interface + the Stage 2 deterministic embedder.

Embedder is a small protocol so the pipeline is agnostic to which model
produces vectors. embedder_id is part of every index version's identity:
indexes built by different embedders are never comparable or reusable, and
the id is recorded in Postgres so a model swap can't silently mix vector
spaces.

HashingEmbedder is NOT a semantic model. It hashes tokens into a
fixed-dimension bag-of-words vector (L2-normalized). It exists so Stage 2
can exercise the full ingest -> FAISS path with zero model download, full
determinism, and ~0 latency. Dense *semantic* retrieval (a real
sentence-transformer) is a later stage; swapping it in is implementing this
protocol and bumping embedder_id, nothing else. Until then, serving quality
still comes from BM25 -- eval metrics do not depend on this embedder.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from app.core.bm25 import tokenize
from app.errors import EmbeddingError


class Embedder(Protocol):
    @property
    def embedder_id(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dim), L2-normalized.

        Raises EmbeddingError on failure; must not return partial results.
        """
        ...


class HashingEmbedder:
    def __init__(self, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def embedder_id(self) -> str:
        return f"hashing-v1-d{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise EmbeddingError("embed_batch called with empty batch")
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "little") % self._dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                out[i, bucket] += sign
            norm = float(np.linalg.norm(out[i]))
            if norm > 0.0:
                out[i] /= norm
        return out
