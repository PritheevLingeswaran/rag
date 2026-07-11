"""Hybrid retrieval pipeline: BM25 + dense -> RRF -> cross-encoder rerank
under an adaptive per-request latency budget.

Composition only: every component (BM25Index, DenseIndex, embedder,
reranker) is injected, so this module has no model files, no I/O, and is
unit-testable with stubs. Wiring real components happens in bootstrap.

Rerank budget (rerank_budget_ms):
    Reranking proceeds micro-batch by micro-batch. Before each batch, the
    pipeline PREDICTS the batch's cost from an EWMA of measured
    per-passage rerank time (learned across requests, process-lifetime);
    a batch that would blow the remaining budget never starts. Scored
    candidates rank first (by cross-encoder score), unscored candidates
    follow in RRF order. Prediction matters on throttled CPU: one
    micro-batch of 5 costs ~3.5s at 0.1 CPU (measured), so a
    check-after-the-fact design would overshoot the budget by multiples.
    Until the first batch ever completes there is no estimate, so the
    very first request per process may overshoot once; server startup
    should seed the estimate with a warmup query.

    Every result carries an explicit rerank status -- degradation is
    always visible, never silent (same pattern as LLM-quota degradation):
        'full'            all candidates cross-encoder scored
        'partial'         budget hit mid-way; hybrid ordering as above
        'skipped_budget'  budget was <= 0; pure RRF order served
        'disabled'        rerank_depth == 0; pure RRF order served
        'no_candidates'   nothing retrieved

    Determinism: 'full' (budget None / generous) and forced-fallback
    (budget 0) are timing-independent and bit-reproducible -- those are
    the two modes the eval harness measures. 'partial' depends on wall
    time by design; its ordering is still deterministic GIVEN the stop
    point, and the stop point is logged.

Defaults for rerank_depth / rerank_budget_ms are set in app.config from
the CPU-throttled (0.1 CPU) load test, not laptop numbers -- see
docs/loadtest_stage4.md.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from app.core.bm25 import BM25Index
from app.core.dense import DenseIndex
from app.core.rrf import rrf_fuse
from app.logging_config import get_logger

logger = get_logger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

RERANK_FULL = "full"
RERANK_PARTIAL = "partial"
RERANK_SKIPPED_BUDGET = "skipped_budget"
RERANK_DISABLED = "disabled"
RERANK_NO_CANDIDATES = "no_candidates"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source: str  # 'rerank' | 'rrf'


@dataclass(frozen=True)
class RerankInfo:
    status: str
    scored: int          # candidates actually cross-encoder scored
    candidates: int      # candidates that entered the rerank stage
    elapsed_ms: float


@dataclass(frozen=True)
class HybridResult:
    query: str
    retrieved_chunk_ids: list[str]
    retrieved_texts: list[str]
    answer: str
    chunks: list[RetrievedChunk]
    rerank: RerankInfo


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
        rerank_budget_ms: float | None = None,
    ) -> None:
        if rerank_depth < 0:
            raise ValueError("rerank_depth must be >= 0 (0 disables rerank)")
        if rerank_depth and rerank_depth < final_top_k:
            raise ValueError("rerank_depth must be 0 or >= final_top_k")
        self.bm25 = bm25
        self.dense = dense
        self.embedder = embedder
        self.reranker = reranker
        self.chunk_texts = chunk_texts
        self.bm25_top_n = bm25_top_n
        self.dense_top_n = dense_top_n
        self.rerank_depth = rerank_depth
        self.final_top_k = final_top_k
        self.rerank_budget_ms = rerank_budget_ms
        # EWMA of measured per-passage rerank cost (ms); None until the
        # first batch ever completes. Shared across requests on purpose:
        # it is a property of the host, not of a query.
        self._rerank_ms_per_passage: float | None = None

    _EWMA_ALPHA = 0.3

    def _rerank_with_budget(
        self, query: str, fused: list[tuple[str, float]]
    ) -> tuple[list[RetrievedChunk], RerankInfo]:
        """Incrementally score fused candidates until done or until the
        NEXT batch is predicted to blow the remaining budget."""
        fused_pos = {cid: i for i, (cid, _) in enumerate(fused)}
        micro_batch = getattr(self.reranker, "micro_batch", 5)
        scored: list[tuple[str, float]] = []
        t0 = time.perf_counter()
        idx = 0
        while idx < len(fused):
            batch = fused[idx:idx + micro_batch]
            if self.rerank_budget_ms is not None:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                predicted_ms = (
                    self._rerank_ms_per_passage * len(batch)
                    if self._rerank_ms_per_passage is not None else 0.0
                )
                if elapsed_ms + predicted_ms >= self.rerank_budget_ms:
                    break
            batch_t0 = time.perf_counter()
            batch_scores = self.reranker.score(
                query, [self.chunk_texts[cid] for cid, _ in batch]
            )
            batch_ms = (time.perf_counter() - batch_t0) * 1000.0
            per_passage = batch_ms / len(batch)
            if self._rerank_ms_per_passage is None:
                self._rerank_ms_per_passage = per_passage
            else:
                self._rerank_ms_per_passage = (
                    self._EWMA_ALPHA * per_passage
                    + (1 - self._EWMA_ALPHA) * self._rerank_ms_per_passage
                )
            scored.extend(
                (cid, float(s)) for (cid, _), s in zip(batch, batch_scores)
            )
            idx += len(batch)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if idx >= len(fused):
            status = RERANK_FULL
        elif scored:
            status = RERANK_PARTIAL
        else:
            status = RERANK_SKIPPED_BUDGET
        info = RerankInfo(status, len(scored), len(fused), round(elapsed_ms, 1))

        # Scored candidates first (score desc, fused position asc for
        # ties), then unscored candidates in fused (RRF) order.
        scored_sorted = sorted(
            scored, key=lambda item: (-item[1], fused_pos[item[0]])
        )
        scored_ids = {cid for cid, _ in scored}
        rest = [(cid, rrf) for cid, rrf in fused if cid not in scored_ids]
        chunks = [
            RetrievedChunk(cid, self.chunk_texts[cid], s, "rerank")
            for cid, s in scored_sorted
        ] + [
            RetrievedChunk(cid, self.chunk_texts[cid], rrf, "rrf")
            for cid, rrf in rest
        ]
        return chunks[:self.final_top_k], info

    def retrieve(self, query: str) -> tuple[list[RetrievedChunk], RerankInfo]:
        from app.observability import RERANK_STATUS, RETRIEVAL_DURATION

        t_total = time.perf_counter()
        t0 = time.perf_counter()
        bm25_ids = [cid for cid, _ in self.bm25.search(query, self.bm25_top_n)]
        bm25_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        query_vec = self.embedder.embed_batch([query])[0]
        embed_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        dense_ids = [cid for cid, _ in self.dense.search(query_vec, self.dense_top_n)]
        dense_s = time.perf_counter() - t0

        candidate_depth = self.rerank_depth or self.final_top_k
        fused = rrf_fuse([bm25_ids, dense_ids], top_k=candidate_depth)

        if not fused:
            info = RerankInfo(RERANK_NO_CANDIDATES, 0, 0, 0.0)
            chunks: list[RetrievedChunk] = []
        elif self.rerank_depth == 0:
            chunks = [
                RetrievedChunk(cid, self.chunk_texts[cid], rrf, "rrf")
                for cid, rrf in fused[:self.final_top_k]
            ]
            info = RerankInfo(RERANK_DISABLED, 0, len(fused), 0.0)
        else:
            chunks, info = self._rerank_with_budget(query, fused)

        total_s = time.perf_counter() - t_total
        RETRIEVAL_DURATION.labels(stage="embed").observe(embed_s)
        RETRIEVAL_DURATION.labels(stage="bm25").observe(bm25_s)
        RETRIEVAL_DURATION.labels(stage="dense").observe(dense_s)
        RETRIEVAL_DURATION.labels(stage="rerank").observe(info.elapsed_ms / 1000.0)
        RETRIEVAL_DURATION.labels(stage="total").observe(total_s)
        RERANK_STATUS.labels(status=info.status).inc()

        # One line per request: the retrieval leg of the request trace.
        logger.info(
            "retrieval_completed",
            embed_ms=round(embed_s * 1000, 1),
            bm25_ms=round(bm25_s * 1000, 1),
            dense_ms=round(dense_s * 1000, 1),
            rerank_ms=info.elapsed_ms,
            rerank_status=info.status,
            rerank_scored=info.scored,
            candidates=info.candidates,
            total_ms=round(total_s * 1000, 1),
        )
        if info.status in (RERANK_PARTIAL, RERANK_SKIPPED_BUDGET):
            logger.info(
                "rerank_degraded", status=info.status, scored=info.scored,
                candidates=info.candidates, elapsed_ms=info.elapsed_ms,
                budget_ms=self.rerank_budget_ms,
            )
        return chunks, info

    def run(self, query: str) -> HybridResult:
        chunks, rerank_info = self.retrieve(query)
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
            rerank=rerank_info,
        )
