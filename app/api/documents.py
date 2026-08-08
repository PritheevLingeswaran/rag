"""Dev-mode document upload (Stage 9.8).

POST /v1/documents accepts one .txt/.md/.pdf file and adds its paragraph
chunks to the LIVE session pipeline (BM25 + dense are rebuilt in-process,
reusing the already-loaded ONNX models). Scope is deliberate and honest:

- development/staging only: production returns 403. Production ingestion
  is the versioned, audited CLI pipeline (app/ingest/cli.py) -- arbitrary
  user uploads would invalidate the eval contract (see README metric rule).
- session-scoped: uploads live in process memory and vanish on restart.
  The UI says so.

Chunking follows the corpus v1 contract (one paragraph = one chunk) so
uploaded chunks look and cite exactly like corpus chunks.
"""

from __future__ import annotations

import functools
import io
import re
import secrets

import anyio
from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi.responses import JSONResponse

from app.api.admission import QueueFullError
from app.api.deps import get_client_id
from app.config import get_settings
from app.core.corpus import Chunk
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()

MAX_UPLOAD_CHUNKS = 200  # keeps the in-process rebuild to a few seconds


def _extract_text(filename: str, data: bytes) -> str | None:
    """Return document text, or None for an unsupported extension."""
    lower = filename.lower()
    if lower.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="replace")
    if lower.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        # One page = at least one paragraph boundary: PDF text extraction
        # rarely preserves blank lines, so page joins guarantee chunking.
        return "\n\n".join(
            (page.extract_text() or "") for page in reader.pages
        )
    return None


def _extend_pipeline(old, new_chunks: list[Chunk]):
    """Rebuild BM25 + dense over old + new chunks, reusing the loaded
    models and tuned parameters. ponytail: full rebuild each upload;
    incremental index add is the upgrade if uploads outgrow ~1k chunks."""
    import numpy as np

    from app.core.bm25 import BM25Index
    from app.core.bootstrap import EMBED_CHUNK_BATCH
    from app.core.dense import DenseIndex
    from app.core.hybrid import HybridPipeline

    texts = dict(old.chunk_texts)
    for c in new_chunks:
        texts[c.chunk_id] = c.text

    items = list(texts.items())
    bm25 = BM25Index()
    bm25.build(items)

    ordered_ids = [cid for cid, _ in items]
    vecs = []
    for start in range(0, len(items), EMBED_CHUNK_BATCH):
        batch = items[start:start + EMBED_CHUNK_BATCH]
        vecs.append(old.embedder.embed_batch([t for _, t in batch]))
    dense = DenseIndex.from_vectors(np.vstack(vecs), ordered_ids)

    return HybridPipeline(
        bm25, dense, old.embedder, old.reranker, texts,
        bm25_top_n=old.bm25_top_n, dense_top_n=old.dense_top_n,
        rerank_depth=old.rerank_depth, final_top_k=old.final_top_k,
        rerank_budget_ms=old.rerank_budget_ms,
    )


@router.post("/v1/documents", tags=["documents"])
async def upload_document(request: Request, file: UploadFile,
                          client_id: str = Depends(get_client_id)):
    """Authenticated on the same terms as /v1/query. This endpoint MUTATES
    the served index and spends real CPU, so it must never be an easier
    target than the read path: staging carries API keys, and an unguarded
    write endpoint there would let anyone reshape what the system answers."""
    settings = get_settings()
    if settings.is_production:
        return JSONResponse(status_code=403, content={
            "error": "uploads are disabled in production; documents are "
                     "ingested through the versioned pipeline",
        })
    service = getattr(request.app.state, "service", None)
    if service is None:
        return JSONResponse(status_code=503, content={
            "error": "pipeline is not serving",
        })

    filename = file.filename or "upload"
    data = await file.read()
    try:
        text = _extract_text(filename, data)
    except Exception:
        logger.warning("upload_extract_failed", filename=filename)
        return JSONResponse(status_code=400, content={
            "error": "could not extract text from this file",
        })
    if text is None:
        return JSONResponse(status_code=415, content={
            "error": "unsupported file type; use .txt, .md or .pdf",
        })

    # Normalize CRLF/CR first: a Windows-authored file must chunk the
    # same as the LF corpus contract.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return JSONResponse(status_code=400, content={
            "error": "no extractable text in this file",
        })
    if len(paragraphs) > MAX_UPLOAD_CHUNKS:
        return JSONResponse(status_code=400, content={
            "error": f"document too large: {len(paragraphs)} chunks "
                     f"(max {MAX_UPLOAD_CHUNKS})",
        })

    slug = re.sub(r"[^a-z0-9]+", "-",
                  filename.rsplit(".", 1)[0].lower()).strip("-") or "doc"
    doc_id = f"upload-{slug}"
    title = filename
    chunks = [
        Chunk(chunk_id=f"{doc_id}::c{i}", doc_id=doc_id, title=title,
              text=para)
        for i, para in enumerate(paragraphs)
    ]

    # Re-embedding the corpus is the same CPU-bound work a query does, on
    # the same single core: it goes through admission control, or a burst
    # of uploads starves the query path the controller exists to protect.
    try:
        async with request.app.state.admission.admit():
            service.pipeline = await anyio.to_thread.run_sync(
                functools.partial(_extend_pipeline, service.pipeline, chunks)
            )
    except QueueFullError as exc:
        return JSONResponse(status_code=503, content={
            "error": "server at capacity; upload not queued",
            "retry_after_s": exc.retry_after_s,
        }, headers={"Retry-After": str(exc.retry_after_s)})
    # The corpus just changed: every cached answer was computed against
    # the OLD index and would hide the document just added. Bumping the
    # version re-namespaces the cache key, so stale entries can no longer
    # be addressed -- correct for the local cache AND a shared Redis,
    # which no single process could safely clear.
    request.app.state.corpus_version = secrets.token_hex(8)
    local_cache = getattr(request.app.state, "local_cache", None)
    if local_cache is not None:
        local_cache.clear()  # unreachable now, but don't leak the memory
    logger.info("document_uploaded", doc_id=doc_id,
                chunks_added=len(chunks),
                total_chunks=len(service.pipeline.chunk_texts))
    return {
        "doc_id": doc_id,
        "chunks_added": len(chunks),
        "total_chunks": len(service.pipeline.chunk_texts),
        "scope": "session (resets on restart)",
    }
