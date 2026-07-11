"""HybridPipeline unit tests with stub embedder/reranker -- no models,
no I/O; proves the pipeline (including the adaptive rerank budget and
its explicit degradation statuses) is testable in isolation."""

import time

import numpy as np
import pytest

from app.core.bm25 import BM25Index
from app.core.dense import DenseIndex
from app.core.hybrid import (
    RERANK_DISABLED,
    RERANK_FULL,
    RERANK_PARTIAL,
    RERANK_SKIPPED_BUDGET,
    HybridPipeline,
)


class StubEmbedder:
    """Maps known queries to fixed unit vectors."""

    def __init__(self, mapping: dict[str, list[float]], dim: int = 3) -> None:
        self.mapping = mapping
        self.dim = dim

    def embed_batch(self, texts):
        out = np.stack([
            np.asarray(self.mapping[t], dtype=np.float32) for t in texts
        ])
        return out / np.linalg.norm(out, axis=1, keepdims=True)


class StubReranker:
    """Scores passages by a fixed text->score mapping; optional per-batch
    sleep to simulate slow inference for budget tests."""

    micro_batch = 2

    def __init__(self, text_scores: dict[str, float],
                 sleep_per_batch_s: float = 0.0) -> None:
        self.text_scores = text_scores
        self.sleep_per_batch_s = sleep_per_batch_s
        self.batch_calls = 0

    def score(self, query, passages):
        self.batch_calls += 1
        if self.sleep_per_batch_s:
            time.sleep(self.sleep_per_batch_s)
        return np.asarray(
            [self.text_scores.get(p, -999.0) for p in passages],
            dtype=np.float32,
        )


TEXTS = {
    "doc1::c0": "cats are wonderful pets. they purr loudly.",
    "doc2::c0": "dogs are loyal companions. they bark at strangers.",
    "doc3::c0": "parrots can mimic human speech. they live long lives.",
}


@pytest.fixture()
def parts():
    bm25 = BM25Index()
    bm25.build(list(TEXTS.items()))
    dense = DenseIndex.from_vectors(
        np.eye(3, dtype=np.float32), list(TEXTS.keys())
    )
    return bm25, dense


def make_pipeline(parts, reranker, **kwargs):
    bm25, dense = parts
    embedder = StubEmbedder({
        "tell me about cats": [1.0, 0.0, 0.0],
        "dogs": [0.0, 0.0, 1.0],
        "pets": [1.0, 1.0, 1.0],
    })
    defaults = dict(rerank_depth=3, final_top_k=3)
    defaults.update(kwargs)
    return HybridPipeline(bm25, dense, embedder, reranker, TEXTS, **defaults)


# ---- full-rerank path ----

def test_full_rerank_orders_by_reranker_score(parts):
    reranker = StubReranker({
        TEXTS["doc3::c0"]: 9.0, TEXTS["doc1::c0"]: 5.0, TEXTS["doc2::c0"]: 1.0,
    })
    result = make_pipeline(parts, reranker).run("tell me about cats")
    assert result.rerank.status == RERANK_FULL
    assert result.rerank.scored == result.rerank.candidates == 3
    assert result.retrieved_chunk_ids[0] == "doc3::c0"
    assert result.answer.startswith("parrots can mimic human speech.")
    assert all(c.source == "rerank" for c in result.chunks)


def test_fused_candidates_include_both_retrievers(parts):
    reranker = StubReranker({t: 1.0 for t in TEXTS.values()})
    p = make_pipeline(parts, reranker)
    result = p.run("dogs")  # lexically doc2, vector points at doc3
    assert "doc2::c0" in result.retrieved_chunk_ids
    assert "doc3::c0" in result.retrieved_chunk_ids


# ---- budget fallback paths ----

def test_zero_budget_forces_pure_rrf_order(parts):
    reranker = StubReranker({TEXTS["doc3::c0"]: 9.0})
    p = make_pipeline(parts, reranker, rerank_budget_ms=0.0)
    result = p.run("tell me about cats")
    assert result.rerank.status == RERANK_SKIPPED_BUDGET
    assert result.rerank.scored == 0
    assert reranker.batch_calls == 0                # no compute spent
    assert all(c.source == "rrf" for c in result.chunks)
    # RRF order: doc1 is top of both retrievers for this query
    assert result.retrieved_chunk_ids[0] == "doc1::c0"


def test_budget_exceeded_mid_way_yields_partial(parts):
    # micro_batch=2, each batch sleeps 30ms, budget 20ms:
    # batch 1 runs (check happens before batch, elapsed 0 < 20), budget
    # is blown during it, batch 2 never starts -> 2 of 3 scored.
    reranker = StubReranker(
        {TEXTS["doc2::c0"]: 9.0, TEXTS["doc1::c0"]: 5.0, TEXTS["doc3::c0"]: 8.0},
        sleep_per_batch_s=0.030,
    )
    p = make_pipeline(parts, reranker, rerank_budget_ms=20.0)
    result = p.run("tell me about cats")
    assert result.rerank.status == RERANK_PARTIAL
    assert result.rerank.scored == 2
    assert result.rerank.candidates == 3
    assert reranker.batch_calls == 1
    # scored candidates first (by score), unscored appended in RRF order
    scored_part = [c for c in result.chunks if c.source == "rerank"]
    rrf_part = [c for c in result.chunks if c.source == "rrf"]
    assert len(scored_part) == 2 and len(rrf_part) == 1
    assert scored_part[0].score >= scored_part[1].score
    # unscored candidates keep their fused position at the tail
    assert result.retrieved_chunk_ids[-1] == rrf_part[0].chunk_id


def test_predictive_gate_skips_after_learning_cost(parts):
    """Once the EWMA knows a batch costs more than the budget, later
    requests skip reranking BEFORE spending any compute -- this is what
    prevents the one-batch overshoot (~3.5s at 0.1 CPU) from recurring."""
    reranker = StubReranker(
        {t: 1.0 for t in TEXTS.values()}, sleep_per_batch_s=0.030
    )
    p = make_pipeline(parts, reranker, rerank_budget_ms=20.0)

    first = p.run("tell me about cats")     # pays once, learns cost
    assert first.rerank.status == RERANK_PARTIAL
    assert p._rerank_ms_per_passage is not None
    calls_after_first = reranker.batch_calls

    second = p.run("tell me about cats")    # predicted over budget: skip
    assert second.rerank.status == RERANK_SKIPPED_BUDGET
    assert reranker.batch_calls == calls_after_first  # zero new compute
    assert all(c.source == "rrf" for c in second.chunks)


def test_generous_budget_reranks_fully(parts):
    reranker = StubReranker({t: 1.0 for t in TEXTS.values()},
                            sleep_per_batch_s=0.001)
    p = make_pipeline(parts, reranker, rerank_budget_ms=10_000.0)
    result = p.run("pets")
    assert result.rerank.status == RERANK_FULL
    assert result.rerank.scored == 3


def test_unlimited_budget_is_default(parts):
    reranker = StubReranker({t: 1.0 for t in TEXTS.values()})
    p = make_pipeline(parts, reranker)
    assert p.rerank_budget_ms is None
    assert p.run("pets").rerank.status == RERANK_FULL


# ---- disabled path ----

def test_depth_zero_disables_rerank_entirely(parts):
    reranker = StubReranker({TEXTS["doc3::c0"]: 9.0})
    p = make_pipeline(parts, reranker, rerank_depth=0, final_top_k=3)
    result = p.run("tell me about cats")
    assert result.rerank.status == RERANK_DISABLED
    assert reranker.batch_calls == 0
    assert all(c.source == "rrf" for c in result.chunks)
    assert len(result.chunks) == 3


def test_every_result_carries_rerank_info(parts):
    reranker = StubReranker({t: 1.0 for t in TEXTS.values()})
    for kwargs in (
        {}, {"rerank_budget_ms": 0.0}, {"rerank_depth": 0, "final_top_k": 3},
    ):
        result = make_pipeline(parts, reranker, **kwargs).run("pets")
        assert result.rerank.status
        assert result.rerank.candidates >= result.rerank.scored >= 0


# ---- validation ----

def test_negative_depth_rejected(parts):
    with pytest.raises(ValueError, match=">= 0"):
        make_pipeline(parts, StubReranker({}), rerank_depth=-1)


def test_depth_below_final_top_k_rejected(parts):
    with pytest.raises(ValueError, match="rerank_depth"):
        make_pipeline(parts, StubReranker({}), rerank_depth=2, final_top_k=3)
