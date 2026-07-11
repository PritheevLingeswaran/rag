"""FastAPI application entrypoint.

Run with: uvicorn app.main:app
(single worker by design: the 512MB cap cannot hold two model copies,
and admission control state is per-process -- see app/api/admission.py)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app import __version__
from app.api.admission import AdmissionController
from app.api.health import router as health_router
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

    app.state.redis_store = None
    if settings.redis_url:
        from app.storage.redis_store import RedisStore

        app.state.redis_store = RedisStore(settings.redis_url)

    if settings.serve_pipeline and not hasattr(app.state, "service"):
        from app.core.bootstrap import build_generation_pipeline

        adapter = build_generation_pipeline(Path(settings.corpus_path))
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
    )
    yield
    logger.info("app_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(query_router)
    return app


app = create_app()
