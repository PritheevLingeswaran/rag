"""/v1/query: the serving endpoint (versioned, OpenAPI-documented).

Request admission order (each layer rejects before the next spends work):
  0. Size limit / request-id middleware        (app/api/middleware.py)
  1. API-key auth                              -> 401
  2. Per-key daily quota (cost guardrail)      -> 429 + Retry-After
  3. Per-key per-minute rate limit             -> 429 + Retry-After
  4. Admission control (bounded queue)         -> 503 + Retry-After
  5. Pipeline execution in a worker thread

Request/response schemas are strict: unknown fields are rejected
(extra='forbid'), sizes are bounded, and max_tokens is capped by the
server-side ceiling regardless of what the client asks for.

The response always carries the explicit degradation fields
(status / degraded / rerank_status) -- no silent downgrade anywhere.
Internal errors never reach the client: see the handlers in app/main.py.
"""

from __future__ import annotations

import anyio
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.api.admission import QueueFullError
from app.api.deps import get_client_id
from app.config import Settings, get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["query"])

SECONDS_PER_DAY = 86_400


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1, max_length=2000,
        description="The question to answer from the indexed corpus.",
    )
    max_tokens: int | None = Field(
        default=None, ge=16, le=1024,
        description="Optional cap on generated tokens; server clamps to "
                    "its own ceiling.",
    )


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    answer: str
    status: str = Field(description="ok | ok_partial_rejected | "
                                    "ok_no_answer | degraded_* | no_results")
    degraded: bool
    rerank_status: str
    citations: list[str]
    retrieved_chunk_ids: list[str]
    retry_after_s: float | None = None
    request_id: str | None = None
    cached: bool = False


# Only stable outcomes are cacheable: transient degradations (quota,
# capacity, provider errors) must not be replayed for an hour.
CACHEABLE_STATUSES = frozenset(
    {"ok", "ok_partial_rejected", "ok_no_answer", "degraded_no_llm",
     "no_results"}
)


def _cache_key(query_text: str, max_tokens: int | None) -> str:
    import hashlib

    normalized = " ".join(query_text.lower().split())
    return hashlib.sha256(
        f"v1|{normalized}|{max_tokens}".encode("utf-8")
    ).hexdigest()


def _rate_limited(retry_after_s: int, scope: str) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"error": f"{scope} exceeded",
                 "retry_after_s": retry_after_s},
        headers={"Retry-After": str(retry_after_s)},
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        401: {"description": "missing or invalid API key"},
        413: {"description": "request body too large"},
        422: {"description": "validation error"},
        429: {"description": "rate limit or daily quota exceeded"},
        503: {"description": "server at capacity; retry later"},
    },
)
async def query(
    body: QueryRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    client_id: str = Depends(get_client_id),
):
    from app.observability import ADMISSION_REJECTED, CACHE_REQUESTS, ERRORS

    request_id = getattr(request.state, "request_id", None)
    redis_store = getattr(request.app.state, "redis_store", None)
    if redis_store is not None:
        # daily per-key quota first (the scarcer resource), then RPM
        import time as _time

        day_used = await anyio.to_thread.run_sync(
            redis_store.bounded_incr,
            f"apiq:{client_id}:{_time.strftime('%Y%m%d', _time.gmtime())}",
            SECONDS_PER_DAY + 3600,
        )
        if day_used is not None and day_used > settings.daily_quota_per_key:
            seconds_to_utc_midnight = SECONDS_PER_DAY - int(
                _time.time() % SECONDS_PER_DAY
            )
            ERRORS.labels(type="daily_quota").inc()
            return _rate_limited(seconds_to_utc_midnight, "daily quota")

        decision = await anyio.to_thread.run_sync(
            redis_store.check_rate_limit, client_id,
            settings.rate_limit_per_minute, 60,
        )
        if not decision.allowed:
            ERRORS.labels(type="rate_limited").inc()
            return _rate_limited(decision.retry_after_s, "rate limit")

    # Response cache: a hit spends no admission slot, no pipeline
    # compute, and no LLM quota (Stage 2.5: the cache IS capacity).
    cache_key = _cache_key(body.query, body.max_tokens)
    if redis_store is not None:
        cached_raw = await anyio.to_thread.run_sync(
            redis_store.cache_get, cache_key
        )
        if cached_raw is not None:
            CACHE_REQUESTS.labels(result="hit").inc()
            import json as _json

            payload = _json.loads(cached_raw)
            payload["cached"] = True
            payload["request_id"] = request_id
            logger.info("request_completed", outcome="cache_hit",
                        status=payload.get("status"))
            return QueryResponse(**payload)
        CACHE_REQUESTS.labels(result="miss").inc()
    else:
        CACHE_REQUESTS.labels(result="bypass").inc()

    admission = request.app.state.admission
    service = request.app.state.service
    import functools

    call = functools.partial(service.answer, body.query)
    if body.max_tokens is not None:
        call = functools.partial(service.answer, body.query,
                                 max_output_tokens=body.max_tokens)
    try:
        async with admission.admit():
            result = await anyio.to_thread.run_sync(call)
    except QueueFullError as exc:
        ADMISSION_REJECTED.inc()
        ERRORS.labels(type="queue_full").inc()
        return JSONResponse(
            status_code=503,
            content={
                "error": "server at capacity; request not queued",
                "retry_after_s": exc.retry_after_s,
            },
            headers={"Retry-After": str(exc.retry_after_s)},
        )

    response = QueryResponse(
        query=result.query,
        answer=result.answer,
        status=result.status,
        degraded=result.degraded,
        rerank_status=result.rerank_status,
        citations=result.citations,
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        retry_after_s=result.retry_after_s,
        request_id=request_id,
    )

    if redis_store is not None and result.status in CACHEABLE_STATUSES:
        payload = response.model_dump()
        payload.pop("request_id", None)
        payload.pop("cached", None)
        import json as _json

        await anyio.to_thread.run_sync(
            functools.partial(
                redis_store.cache_set, cache_key,
                _json.dumps(payload).encode("utf-8"), settings.cache_ttl_s,
            )
        )

    logger.info(
        "request_completed", outcome="served",
        status=result.status, degraded=result.degraded,
        rerank_status=result.rerank_status,
        citations=len(result.citations),
    )
    return response


@router.get("/admin/admission", include_in_schema=False,
            dependencies=[Depends(get_client_id)])
def admission_stats(request: Request) -> dict:
    """Operator introspection; authenticated, hidden from public docs."""
    return request.app.state.admission.snapshot()
