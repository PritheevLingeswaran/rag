"""Dense retrieval over a FAISS inner-product index.

Isolated module: depends only on numpy + faiss, no other app code.
Vectors are expected L2-normalized so inner product == cosine similarity.
Deterministic: IndexFlatIP is exact (exhaustive) search; ties resolve by
FAISS's stable internal ordering (insertion order), same query -> same
ranking always.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class DenseIndex:
    """A FAISS IndexFlatIP paired with the chunk_id for each row."""

    def __init__(self, index: faiss.Index, chunk_ids: list[str]) -> None:
        if index.ntotal != len(chunk_ids):
            raise ValueError(
                f"index has {index.ntotal} vectors but {len(chunk_ids)} "
                f"chunk ids were provided"
            )
        self._index = index
        self._chunk_ids = chunk_ids

    @property
    def size(self) -> int:
        return self._index.ntotal

    @property
    def dim(self) -> int:
        return self._index.d

    @classmethod
    def from_vectors(cls, vectors: np.ndarray,
                     chunk_ids: list[str]) -> "DenseIndex":
        if vectors.ndim != 2:
            raise ValueError(f"expected 2D vector matrix, got ndim={vectors.ndim}")
        if vectors.dtype != np.float32:
            raise ValueError(f"expected float32 vectors, got {vectors.dtype}")
        if vectors.shape[0] != len(chunk_ids):
            raise ValueError(
                f"{vectors.shape[0]} vectors vs {len(chunk_ids)} chunk ids"
            )
        if vectors.shape[0] == 0:
            raise ValueError("cannot build dense index over zero vectors")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, list(chunk_ids))

    @classmethod
    def from_files(cls, index_path: Path, chunk_ids: list[str]) -> "DenseIndex":
        """Production boot path: load a prebuilt index file (Stage 2's
        FaissStore verifies its sha256 before handing us the path)."""
        return cls(faiss.read_index(str(index_path)), list(chunk_ids))

    def search(self, query_vector: np.ndarray,
               top_k: int = 10) -> list[tuple[str, float]]:
        """Return up to top_k (chunk_id, cosine_score) pairs, best first."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        q = np.asarray(query_vector, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if q.shape != (1, self.dim):
            raise ValueError(
                f"query vector shape {query_vector.shape} does not match "
                f"index dim {self.dim}"
            )
        scores, rows = self._index.search(q, min(top_k, self.size))
        return [
            (self._chunk_ids[row], float(score))
            for row, score in zip(rows[0], scores[0])
            if row != -1
        ]
