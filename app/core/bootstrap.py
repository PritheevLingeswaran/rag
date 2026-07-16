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
    embedded dense index + cross-encoder. Embedding happens here -- and
    this IS the production boot path today: the corpus ships in the image
    and is re-embedded at startup (~46s at 0.1 CPU for 60 chunks). The
    prebuilt-FaissStore load path (Stage 2 machinery) is the documented
    upgrade once the corpus outgrows boot-time embedding (10k+ chunks);
    it is NOT wired into serving yet, and needs object storage first --
    Render free disk is ephemeral (see Dockerfile note).

    rerank_depth / rerank_budget_ms default from Settings (which were set
    from the throttled load test); explicit kwargs override, which is how
    the eval harness pins its two deterministic modes (unlimited budget
    for 'hybrid', 0 for 'hybrid-fallback')."""
    from app.config import get_settings

    settings = get_settings()
    pipeline_kwargs.setdefault("rerank_depth", settings.rerank_depth)
    pipeline_kwargs.setdefault("rerank_budget_ms", settings.rerank_budget_ms)
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


class GenerationEvalAdapter:
    """Adapts GenerationService to the eval harness's pipeline interface
    (.run(query) -> result with retrieved_chunk_ids/retrieved_texts/answer,
    plus .chunk_texts for gold-id validation)."""

    def __init__(self, service, chunk_texts: dict[str, str]) -> None:
        self._service = service
        self.chunk_texts = chunk_texts

    def run(self, query: str):
        return self._service.answer(query)


def build_generation_pipeline(
    corpus_path: Path,
    cache_dir: str | None = None,
    alerts=None,
    redis_store=None,
    **pipeline_kwargs,
) -> GenerationEvalAdapter:
    """Full serving path for eval: hybrid retrieval + GenerationService.
    Uses the real Gemini client when GEMINI_API_KEY is set; otherwise the
    service serves its explicit 'degraded_no_llm' extractive path, which
    is exactly what a keyless deployment would serve -- the harness then
    measures that real degraded behavior, not a pretend LLM."""
    from app.config import get_settings
    from app.generation.llm_client import GeminiClient
    from app.generation.service import GenerationService

    settings = get_settings()
    pipeline = build_hybrid_from_corpus(corpus_path, cache_dir,
                                        **pipeline_kwargs)
    llm = None
    quota_guard = None
    if settings.gemini_api_key:
        llm = GeminiClient(
            api_key=settings.gemini_api_key,
            model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
            max_output_tokens=settings.llm_max_output_tokens,
        )
        from app.generation.quota import QuotaGuard, load_model_limits

        # Reuse the caller's RedisStore when given (the app lifespan passes
        # its command-budget-metered store, so quota accounting spends the
        # Upstash allowance visibly and shares one connection pool). A
        # fresh unmetered store is only for standalone use (eval harness).
        if redis_store is None and settings.redis_url:
            from app.storage.redis_store import RedisStore

            redis_store = RedisStore(settings.redis_url)
        quota_guard = QuotaGuard(
            load_model_limits(settings.llm_model),
            redis_store=redis_store,
            safety_margin=settings.quota_safety_margin,
            alerts=alerts,
        )
        logger.info("quota_guard_armed", **quota_guard.snapshot())

    fallback_llm = None
    fallback_guard = None
    if llm is not None and settings.groq_api_key:
        from app.generation.llm_client import GroqClient
        from app.generation.quota import QuotaGuard, load_model_limits

        fallback_llm = GroqClient(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            timeout_s=settings.llm_timeout_s,
            max_output_tokens=settings.llm_max_output_tokens,
        )
        fallback_guard = QuotaGuard(
            load_model_limits(settings.groq_model),
            redis_store=redis_store,
            safety_margin=settings.quota_safety_margin,
            alerts=alerts,
        )
        logger.info("fallback_llm_armed", **fallback_guard.snapshot())
    logger.info("generation_service_built",
                llm="gemini" if llm else "none (degraded_no_llm path)",
                fallback="groq" if fallback_llm else "none")
    return GenerationEvalAdapter(
        GenerationService(pipeline, llm, quota_guard=quota_guard,
                          fallback_llm=fallback_llm,
                          fallback_quota_guard=fallback_guard),
        pipeline.chunk_texts,
    )
