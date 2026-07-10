"""Pure-Python BM25 (Okapi) index.

Dependency-free and fully deterministic: same corpus + same query always
yields the same ranking. Ties are broken by document insertion order, which
is itself deterministic given the corpus file. This is the Stage 0 first-stage
retriever; FAISS dense retrieval and fusion arrive in later stages.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_ids: list[str] = []
        self._doc_freqs: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}

    def build(self, docs: list[tuple[str, str]]) -> None:
        """Index (doc_id, text) pairs. Must be called exactly once."""
        if self._doc_ids:
            raise RuntimeError("BM25Index.build() called twice")
        if not docs:
            raise ValueError("cannot build BM25 index over empty corpus")
        for doc_id, text in docs:
            tokens = tokenize(text)
            self._doc_ids.append(doc_id)
            self._doc_freqs.append(Counter(tokens))
            self._doc_lens.append(len(tokens))
            self._df.update(set(tokens))
        n = len(self._doc_ids)
        self._avgdl = sum(self._doc_lens) / n
        # Standard BM25 IDF with 0.5 smoothing, floored at 0 to avoid
        # negative scores for terms in more than half the corpus.
        self._idf = {
            term: max(0.0, math.log((n - df + 0.5) / (df + 0.5) + 1.0))
            for term, df in self._df.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return up to top_k (doc_id, score) pairs, best first."""
        if not self._doc_ids:
            raise RuntimeError("BM25Index.search() called before build()")
        q_tokens = tokenize(query)
        scores = [0.0] * len(self._doc_ids)
        for term in q_tokens:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self._doc_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (
                    1.0 - self.b + self.b * self._doc_lens[i] / self._avgdl
                )
                scores[i] += idf * tf * (self.k1 + 1.0) / denom
        ranked = sorted(
            range(len(scores)), key=lambda i: (-scores[i], i)
        )[:top_k]
        return [(self._doc_ids[i], scores[i]) for i in ranked if scores[i] > 0.0]
