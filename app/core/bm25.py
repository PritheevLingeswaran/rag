"""Pure-Python/numpy BM25 (Okapi) index over inverted posting lists.

Isolated module: depends only on numpy, no other app code. Fully
deterministic: same corpus + same query always yields the same ranking,
with ties broken by document insertion order.

Implementation notes (sized for the 10k-50k chunk target on a 512MB box):
- Postings are stored per term as parallel numpy arrays (doc_idx int32,
  tf float32): ~8 bytes per posting vs ~60+ for python tuples. At 50k
  chunks x ~100 tokens that is ~40MB instead of ~300MB.
- During build, postings accumulate in array.array('i') buffers (doc_idx
  and tf interleaved) to avoid a transient python-object spike, then are
  frozen to numpy.
- Scoring is vectorized: one numpy gather/update per query term instead
  of a python loop over all documents.
"""

from __future__ import annotations

import math
import re
from array import array
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_ids: list[str] = []
        self._doc_lens: np.ndarray | None = None
        self._avgdl: float = 0.0
        # term -> (doc_idx int32 array, tf float32 array)
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._idf: dict[str, float] = {}

    @property
    def size(self) -> int:
        return len(self._doc_ids)

    def build(self, docs: list[tuple[str, str]]) -> None:
        """Index (doc_id, text) pairs. Must be called exactly once."""
        if self._doc_ids:
            raise RuntimeError("BM25Index.build() called twice")
        if not docs:
            raise ValueError("cannot build BM25 index over empty corpus")

        buffers: dict[str, array] = {}
        doc_lens = array("i")
        for doc_id, text in docs:
            idx = len(self._doc_ids)
            self._doc_ids.append(doc_id)
            counts = Counter(tokenize(text))
            doc_lens.append(sum(counts.values()))
            for term, tf in counts.items():
                buf = buffers.get(term)
                if buf is None:
                    buf = buffers[term] = array("i")
                buf.append(idx)
                buf.append(tf)

        n = len(self._doc_ids)
        self._doc_lens = np.asarray(doc_lens, dtype=np.float32)
        self._avgdl = float(self._doc_lens.mean())
        for term, buf in buffers.items():
            flat = np.frombuffer(buf, dtype=np.int32)
            doc_idx = flat[0::2].copy()
            tfs = flat[1::2].astype(np.float32)
            self._postings[term] = (doc_idx, tfs)
            df = len(doc_idx)
            # Standard BM25 IDF with 0.5 smoothing, floored at 0 to avoid
            # negative scores for terms in more than half the corpus.
            self._idf[term] = max(
                0.0, math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            )
        buffers.clear()

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return up to top_k (doc_id, score) pairs, best first.
        Ties break by insertion order; zero-score docs are never returned."""
        if not self._doc_ids:
            raise RuntimeError("BM25Index.search() called before build()")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        scores = np.zeros(len(self._doc_ids), dtype=np.float32)
        norm = self.k1 * (
            1.0 - self.b + self.b * self._doc_lens / self._avgdl
        )
        # Duplicate query terms count once: query-side tf weighting is a
        # BM25 variant; we keep the common simple form deliberately.
        for term in set(tokenize(query)):
            posting = self._postings.get(term)
            if posting is None:
                continue
            doc_idx, tfs = posting
            contrib = self._idf[term] * tfs * (self.k1 + 1.0) / (
                tfs + norm[doc_idx]
            )
            np.add.at(scores, doc_idx, contrib)

        k = min(top_k, len(scores))
        candidates = np.argpartition(-scores, k - 1)[:k]
        # Stable deterministic order: score desc, then insertion order asc.
        order = candidates[np.lexsort((candidates, -scores[candidates]))]
        return [
            (self._doc_ids[i], float(scores[i]))
            for i in order
            if scores[i] > 0.0
        ]
