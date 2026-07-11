"""POST /query: the serving endpoint.

Request admission order (each layer rejects before the next spends work):
  1. API-key auth        -> 401 (constant-time compare; anonymous allowed
                                 only outside production, and logged)
  2. Per-client rate limit -> 429 + Retry-After (Redis fixed-window,
                                 fail-open on Redis outage per Stage 2)
  3. Admission control   -> 503 + Retry-After when the bounded queue is
                                 full (see app/api/admission.py)
  4. Pipeline execution in a worker thread (the pipeline is sync and
     CPU-bound; the event loop stays free to answer health checks and
     reject overload while requests run)

The response always carries the explicit degradation fields
(status / degraded / rerank_status) -- no silent downgrade anywhere.
"""

from __future__ import annotations

import hmac

import anyio
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api.admission import QueueFullError
from app.config import Settings, get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    query: str
    answer: str
    status: str
    degraded: bool
    rerank_status: str
    citations: list[str]
    retrieved_chunk_ids: list[str]
    retry_after_s: float | None = None


def _authenticate(request: Request, settings: Settings) -> str | None:
    """Returns client id, or None => the caller gets a 401."""
    keys = settings.api_key_list
    provided = request.headers.get("x-api-key", "")
    if keys:
        for key in keys:
            if hmac.compare_digest(provided, key):
                # client identity = stable non-reversible tag of the key
                return f"key:{key[:4]}...{len(key)}"
        return None
    if settings.is_production:
        # create_app refuses to boot like this; belt and suspenders.
        return None
    client_host = request.client.host if request.client else "unknown"
    logger.info("anonymous_request_dev_mode", client=client_host)
    return f"anon:{client_host}"


@router.post("/query")
async def query(
    body: QueryRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
):
    client_id = _authenticate(request, settings)
    if client_id is None:
        return JSONResponse(
            status_code=401,
            content={"error": "missing or invalid API key"},
        )

    redis_store = getattr(request.app.state, "redis_store", None)
    if redis_store is not None:
        decision = await anyio.to_thread.run_sync(
            redis_store.check_rate_limit, client_id,
            settings.rate_limit_per_minute, 60,
        )
        if not decision.allowed:
            return JSONResponse(
                status_code=429,
                content={"error": "rate limit exceeded",
                         "retry_after_s": decision.retry_after_s},
                headers={"Retry-After": str(decision.retry_after_s)},
            )

    admission = request.app.state.admission
    service = request.app.state.service
    try:
        async with admission.admit():
            result = await anyio.to_thread.run_sync(
                service.answer, body.query
            )
    except QueueFullError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "error": "server at capacity; request not queued",
                "retry_after_s": exc.retry_after_s,
            },
            headers={"Retry-After": str(exc.retry_after_s)},
        )

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        status=result.status,
        degraded=result.degraded,
        rerank_status=result.rerank_status,
        citations=result.citations,
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        retry_after_s=result.retry_after_s,
    )


@router.get("/admin/admission")
def admission_stats(request: Request) -> dict:
    """Introspection for operators and the load-test harness."""
    return request.app.state.admission.snapshot()
