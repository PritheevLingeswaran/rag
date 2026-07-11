"""HybridPipeline unit tests with stub embedder/reranker -- no models,
no I/O; proves the pipeline is testable in isolation."""

import numpy as np
import pytest

from app.core.bm25 import BM25Index
from app.core.dense import DenseIndex
from app.core.hybrid import HybridPipeline


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
    """Scores passages by a fixed preference list (higher = earlier)."""

    def __init__(self, preference: list[str]) -> None:
        self.preference = preference
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query, candidates, top_k=None):
        self.calls.append((query, [cid for cid, _ in candidates]))
        def score(cid):
            return -self.preference.index(cid) if cid in self.preference else -999
        ranked = sorted(
            ((cid, float(score(cid))) for cid, _ in candidates),
            key=lambda x: -x[1],
        )
        return ranked[:top_k] if top_k else ranked


@pytest.fixture()
def pipeline_parts():
    texts = {
        "doc1::c0": "cats are wonderful pets. they purr loudly.",
        "doc2::c0": "dogs are loyal companions. they bark at strangers.",
        "doc3::c0": "parrots can mimic human speech. they live long lives.",
    }
    bm25 = BM25Index()
    bm25.build(list(texts.items()))
    vectors = np.eye(3, dtype=np.float32)
    dense = DenseIndex.from_vectors(vectors, list(texts.keys()))
    return texts, bm25, dense


def test_reranker_output_defines_final_order(pipeline_parts):
    texts, bm25, dense = pipeline_parts
    embedder = StubEmbedder({"tell me about cats": [1.0, 0.0, 0.0]})
    reranker = StubReranker(["doc3::c0", "doc1::c0", "doc2::c0"])
    p = HybridPipeline(bm25, dense, embedder, reranker, texts,
                       rerank_depth=3, final_top_k=3)
    result = p.run("tell me about cats")
    assert result.retrieved_chunk_ids[0] == "doc3::c0"
    assert result.answer.startswith("parrots can mimic human speech.")


def test_fused_candidates_include_both_retrievers(pipeline_parts):
    texts, bm25, dense = pipeline_parts
    # query lexically matches doc2 ("dogs"), vector points at doc3
    embedder = StubEmbedder({"dogs": [0.0, 0.0, 1.0]})
    reranker = StubReranker(["doc2::c0", "doc3::c0", "doc1::c0"])
    p = HybridPipeline(bm25, dense, embedder, reranker, texts,
                       rerank_depth=3, final_top_k=3)
    p.run("dogs")
    (_, candidate_ids), = reranker.calls
    assert "doc2::c0" in candidate_ids  # from BM25
    assert "doc3::c0" in candidate_ids  # from dense


def test_final_top_k_truncates(pipeline_parts):
    texts, bm25, dense = pipeline_parts
    embedder = StubEmbedder({"pets": [1.0, 1.0, 1.0]})
    reranker = StubReranker(["doc1::c0", "doc2::c0", "doc3::c0"])
    p = HybridPipeline(bm25, dense, embedder, reranker, texts,
                       rerank_depth=3, final_top_k=1)
    result = p.run("pets")
    assert len(result.chunks) == 1


def test_no_matches_yields_empty_answer_path(pipeline_parts):
    texts, bm25, dense = pipeline_parts
    embedder = StubEmbedder({"zzz qqq": [1.0, 0.0, 0.0]})
    reranker = StubReranker([])
    p = HybridPipeline(bm25, dense, embedder, reranker, texts)
    # dense always returns neighbors, so retrieval is non-empty; but a
    # query with no lexical match must still work end to end
    result = p.run("zzz qqq")
    assert result.answer
    assert len(result.retrieved_chunk_ids) > 0


def test_rerank_depth_must_cover_final_top_k(pipeline_parts):
    texts, bm25, dense = pipeline_parts
    with pytest.raises(ValueError, match="rerank_depth"):
        HybridPipeline(bm25, dense, StubEmbedder({}), StubReranker([]),
                       texts, rerank_depth=5, final_top_k=10)
