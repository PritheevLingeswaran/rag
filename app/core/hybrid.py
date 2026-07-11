"""Hybrid retrieval pipeline: BM25 + dense -> RRF -> cross-encoder rerank.

Composition only: every component (BM25Index, DenseIndex, embedder,
reranker) is injected, so this module has no model files, no I/O, and is
unit-testable with stubs. Wiring real components happens in the bootstrap
(eval harness / API startup), not here.

Candidate depths (defaults):
    bm25_top_n = dense_top_n = 30   first-stage candidates from each
    rerank_depth = 20               fused candidates scored by cross-encoder
    final_top_k = 10                returned results
Depths are the latency/quality dial: reranker cost is linear in
rerank_depth (see docs/loadtest_stage3.md for measured numbers).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.bm25 import BM25Index
from app.core.dense import DenseIndex
from app.core.rrf import rrf_fuse

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source: str  # 'rerank' | 'rrf'


@dataclass(frozen=True)
class HybridResult:
    query: str
    retrieved_chunk_ids: list[str]
    retrieved_texts: list[str]
    answer: str
    chunks: list[RetrievedChunk]


class HybridPipeline:
    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        embedder,
        reranker,
        chunk_texts: dict[str, str],
        bm25_top_n: int = 30,
        dense_top_n: int = 30,
        rerank_depth: int = 20,
        final_top_k: int = 10,
    ) -> None:
        if rerank_depth < final_top_k:
            raise ValueError("rerank_depth must be >= final_top_k")
        self.bm25 = bm25
        self.dense = dense
        self.embedder = embedder
        self.reranker = reranker
        self.chunk_texts = chunk_texts
        self.bm25_top_n = bm25_top_n
        self.dense_top_n = dense_top_n
        self.rerank_depth = rerank_depth
        self.final_top_k = final_top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        bm25_ids = [cid for cid, _ in self.bm25.search(query, self.bm25_top_n)]
        query_vec = self.embedder.embed_batch([query])[0]
        dense_ids = [cid for cid, _ in self.dense.search(query_vec, self.dense_top_n)]

        fused = rrf_fuse([bm25_ids, dense_ids], top_k=self.rerank_depth)
        if not fused:
            return []

        candidates = [(cid, self.chunk_texts[cid]) for cid, _ in fused]
        reranked = self.reranker.rerank(query, candidates, top_k=self.final_top_k)
        return [
            RetrievedChunk(
                chunk_id=cid, text=self.chunk_texts[cid],
                score=score, source="rerank",
            )
            for cid, score in reranked
        ]

    def run(self, query: str) -> HybridResult:
        chunks = self.retrieve(query)
        if chunks:
            sentences = _SENTENCE_RE.split(chunks[0].text)
            answer = " ".join(sentences[:2]).strip()
        else:
            answer = "No relevant documents found."
        return HybridResult(
            query=query,
            retrieved_chunk_ids=[c.chunk_id for c in chunks],
            retrieved_texts=[c.text for c in chunks],
            answer=answer,
            chunks=chunks,
        )
