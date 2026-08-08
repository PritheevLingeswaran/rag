"""Liveness and readiness endpoints -- deliberately separate.

/health (liveness) answers "is this process alive and configured?" It is
always 200 for a running process; a restart would not help whatever else
is wrong, so it must not fail on dependency trouble.

/health/ready (readiness) answers "can this instance serve a query right
now?" -- which is what a load balancer and Render's healthCheckPath
actually need to know. It returns 503 when the instance is configured to
serve but has no pipeline, because that instance can only produce 500s.
Before this split, such an instance reported "ok" forever while every
request 500'd AND paged the operator.

Readiness is intentionally cheap and local: the pipeline reference is
set once at boot and never unset, so this check costs nothing and cannot
flap. Redis/Postgres are deliberately NOT probed -- both degrade
gracefully by design (cache misses, fail-open rate limits, logged-out
sessions), so failing readiness on them would restart a healthy instance
that is serving correctly.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.config import Settings, get_settings

router = APIRouter()


def is_ready(request: Request) -> bool:
    """True when this instance can serve /v1/query."""
    settings = get_settings()
    if not settings.serve_pipeline:
        return True  # not meant to serve the pipeline (test/eval contexts)
    return getattr(request.app.state, "service", None) is not None


@router.get("/health")
def health(request: Request,
           settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "version": __version__,
        "environment": settings.environment,
        "ready": is_ready(request),
    }


@router.get("/health/ready")
def ready(request: Request,
          settings: Settings = Depends(get_settings)):
    if not is_ready(request):
        return JSONResponse(status_code=503, content={
            "status": "not_ready",
            "reason": "pipeline not loaded; this instance cannot serve",
        })
    return {"status": "ready", "version": __version__}
