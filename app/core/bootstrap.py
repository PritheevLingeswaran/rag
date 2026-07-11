"""Builders that wire real components into pipelines.

This is the only place where model downloads, corpus loading, and pipeline
assembly meet. The eval harness and (later) the API startup both call
build_hybrid_from_corpus so there is exactly one construction path.
"""

from __future__ import annotations

from pathlib import Path

from app.core.bm25 import BM25Index
from app.core.corpus import load_chunks
from app.core.dense import DenseIndex
from app.core.hybrid import HybridPipeline
from app.core.onnx_text import (
    EMBED_MODEL_FILE,
    EMBED_MODEL_REPO,
    RERANK_MODEL_FILE,
    RERANK_MODEL_REPO,
    OnnxEmbedder,
    OnnxReranker,
    download_model,
)
from app.logging_config import get_logger

logger = get_logger(__name__)

EMBED_CHUNK_BATCH = 32


def load_onnx_models(cache_dir: str | None = None) -> tuple[OnnxEmbedder, OnnxReranker]:
    emb_model, emb_tok = download_model(
        EMBED_MODEL_REPO, EMBED_MODEL_FILE, cache_dir
    )
    rr_model, rr_tok = download_model(
        RERANK_MODEL_REPO, RERANK_MODEL_FILE, cache_dir
    )
    return OnnxEmbedder(emb_model, emb_tok), OnnxReranker(rr_model, rr_tok)


def build_hybrid_from_corpus(
    corpus_path: Path,
    cache_dir: str | None = None,
    **pipeline_kwargs,
) -> HybridPipeline:
    """Build the full hybrid pipeline from a corpus file: BM25 + freshly
    embedded dense index + cross-encoder. Embedding happens here (fine for
    the 60-chunk eval corpus; production serving loads a prebuilt FAISS
    version from Stage 2's FaissStore instead)."""
    embedder, reranker = load_onnx_models(cache_dir)
    chunks = load_chunks(corpus_path)
    texts = {c.chunk_id: c.text for c in chunks}

    bm25 = BM25Index()
    bm25.build([(c.chunk_id, c.text) for c in chunks])

    import numpy as np

    vecs = []
    ordered_ids = [c.chunk_id for c in chunks]
    for start in range(0, len(chunks), EMBED_CHUNK_BATCH):
        batch = chunks[start:start + EMBED_CHUNK_BATCH]
        vecs.append(embedder.embed_batch([c.text for c in batch]))
    dense = DenseIndex.from_vectors(np.vstack(vecs), ordered_ids)

    logger.info(
        "hybrid_pipeline_built", chunks=len(chunks),
        embedder_id=embedder.embedder_id,
    )
    return HybridPipeline(bm25, dense, embedder, reranker, texts,
                          **pipeline_kwargs)
