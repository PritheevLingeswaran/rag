"""ONNX model integration tests (use the locally cached model files;
download on first run). Marked integration because they need ~50MB of
model assets, not because they need services."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.bootstrap import load_onnx_models
from app.errors import EmbeddingError


@pytest.fixture(scope="module")
def models():
    return load_onnx_models()


def test_embedder_output_shape_and_normalization(models):
    embedder, _ = models
    out = embedder.embed_batch(["hello world", "a longer sentence about databases"])
    assert out.shape == (2, 384)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_embedder_is_deterministic(models):
    embedder, _ = models
    a = embedder.embed_batch(["deterministic check"])
    b = embedder.embed_batch(["deterministic check"])
    assert np.array_equal(a, b)


def test_embedder_semantic_sanity(models):
    embedder, _ = models
    vecs = embedder.embed_batch([
        "how do database transactions work",
        "transactions in a relational database",
        "recipe for chocolate cake",
    ])
    sim_related = float(vecs[0] @ vecs[1])
    sim_unrelated = float(vecs[0] @ vecs[2])
    assert sim_related > sim_unrelated


def test_embedder_rejects_empty_batch(models):
    embedder, _ = models
    with pytest.raises(EmbeddingError, match="empty batch"):
        embedder.embed_batch([])


def test_reranker_prefers_relevant_passage(models):
    _, reranker = models
    ranked = reranker.rerank(
        "what does VACUUM do in PostgreSQL",
        [
            ("off", "Kafka topics are divided into partitions for parallelism."),
            ("on", "VACUUM reclaims storage occupied by dead tuples in PostgreSQL tables."),
        ],
    )
    assert ranked[0][0] == "on"
    assert ranked[0][1] > ranked[1][1]


def test_reranker_deterministic_for_fixed_micro_batch(models):
    """The model is dynamically int8-quantized: activation scales are
    computed per batch, so scores DO shift with batch composition
    (measured ~0.1-0.3 logits). What we require and pin down here is
    determinism under a fixed configuration -- same inputs, same
    micro_batch => bit-identical scores -- which is what makes the eval
    harness reproducible. Batch-composition sensitivity is documented in
    app/core/onnx_text.py."""
    _, reranker = models
    passages = [f"the database stores rows in pages number {i}" for i in range(7)]
    a = reranker.score("database storage", passages)
    b = reranker.score("database storage", passages)
    assert np.array_equal(a, b)


def test_reranker_ordering_robust_across_micro_batch_sizes(models):
    """Absolute logits shift with batch composition (quantization), but
    the ordering of clearly-different passages must not."""
    _, reranker = models
    candidates = [
        ("off1", "Kafka consumer groups assign partitions to consumers."),
        ("on", "VACUUM reclaims storage from dead tuples in PostgreSQL."),
        ("off2", "A Bloom filter can return false positives."),
        ("off3", "Round-robin load balancing rotates across backends."),
    ]
    query = "how does PostgreSQL reclaim dead tuple storage"
    old = reranker._micro_batch
    try:
        orders = []
        for mb in (1, 2, 4):
            reranker._micro_batch = mb
            ranked = reranker.rerank(query, candidates)
            orders.append(ranked[0][0])
    finally:
        reranker._micro_batch = old
    assert orders == ["on", "on", "on"]


def test_reranker_rejects_empty(models):
    _, reranker = models
    with pytest.raises(ValueError, match="no passages"):
        reranker.score("query", [])
