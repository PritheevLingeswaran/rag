"""FastAPI application entrypoint.

Run with: uvicorn app.main:app
(single worker by design: the 512MB cap cannot hold two model copies,
and admission control state is per-process -- see app/api/admission.py)

Error policy: NO internal detail ever reaches a client. Unhandled
exceptions are logged with full traceback + request_id server-side; the
client receives {"error": "internal server error", "request_id": ...}
and nothing else. Validation errors (422) expose only field locations
and messages, which describe the CLIENT's input, not our internals.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.api.admission import AdmissionController
from app.api.health import router as health_router
from app.api.middleware import RequestIDMiddleware, RequestSizeLimitMiddleware
from app.api.query import router as query_router
from app.config import get_settings
from app.errors import ConfigurationError
from app.logging_config import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings)
    logger = get_logger(__name__)

    if settings.is_production and not settings.api_key_list:
        raise ConfigurationError(
            "API_KEYS must be set in production; refusing to serve an "
            "open endpoint"
        )

    app.state.admission = AdmissionController(
        max_concurrency=settings.admission_max_concurrency,
        max_queue_depth=settings.admission_max_queue_depth,
    )

    from app.reliability import AlertManager, ResourceBudget

    app.state.alerts = AlertManager(settings.alert_webhook_url)

    app.state.redis_store = None
    if settings.redis_url:
        from app.storage.redis_store import RedisStore

        app.state.redis_store = RedisStore(
            settings.redis_url,
            command_budget=ResourceBudget(
                "upstash_commands_daily",
                settings.redis_daily_command_budget,
                alerts=app.state.alerts,
            ),
        )

    if settings.serve_pipeline and not hasattr(app.state, "service"):
        from app.core.bootstrap import build_generation_pipeline

        adapter = build_generation_pipeline(Path(settings.corpus_path),
                                            alerts=app.state.alerts)
        app.state.service = adapter._service
        # Warmup: exercises the full path once so model sessions are hot
        # and the rerank-budget EWMA is seeded before the first user
        # request (otherwise the first request may overshoot the budget;
        # see app/core/hybrid.py).
        app.state.service.answer("warmup query to seed model sessions")
        logger.info("pipeline_warmed")

    logger.info(
        "app_startup",
        app_name=settings.app_name,
        environment=settings.environment,
        version=__version__,
        admission=app.state.admission.snapshot(),
        rate_limiting="redis" if app.state.redis_store else "disabled (no REDIS_URL)",
        cors_origins=settings.cors_origin_list or "none (cross-origin denied)",
    )
    yield
    logger.info("app_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    # Middleware (outermost first): request-id wraps everything so even
    # size-limit rejections carry an id; size limit runs before routing.
    app.add_middleware(RequestSizeLimitMiddleware,
                       max_bytes=settings.max_request_bytes)
    app.add_middleware(RequestIDMiddleware)

    # CORS: deny-by-default. Middleware is added only when origins are
    # explicitly configured; a wildcard in production is a config error.
    origins = settings.cors_origin_list
    if origins:
        if settings.is_production and "*" in origins:
            raise ConfigurationError(
                "CORS_ORIGINS='*' is not allowed in production"
            )
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["POST", "GET"],
            allow_headers=["content-type", "x-api-key"],
            allow_credentials=False,
            max_age=600,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        from app.observability import ERRORS

        ERRORS.labels(type="unhandled").inc()
        request_id = getattr(request.state, "request_id", None)
        get_logger(__name__).error(
            "unhandled_exception",
            request_id=request_id,
            path=request.url.path,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={"error": "internal server error",
                     "request_id": request_id},
            headers={"X-Request-ID": request_id or ""},
        )

    @app.middleware("http")
    async def http_metrics_middleware(request: Request, call_next):
        import time as _time

        from app.observability import HTTP_REQUEST_DURATION

        t0 = _time.perf_counter()
        response = await call_next(request)
        # Label with the ROUTE TEMPLATE, not the raw URL, to keep metric
        # cardinality bounded; unmatched paths group under 'unmatched'.
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        if path != "/metrics":  # do not meter the meter
            HTTP_REQUEST_DURATION.labels(
                method=request.method, path=path,
                status=str(response.status_code),
            ).observe(_time.perf_counter() - t0)
        return response

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        from starlette.responses import Response

        from app.observability import render_metrics

        payload, content_type = render_metrics()
        return Response(content=payload, media_type=content_type)

    app.include_router(health_router)
    app.include_router(query_router)
    return app


app = create_app()
