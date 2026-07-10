"""Liveness/readiness endpoint.

Stage 1 scope: process is up and settings loaded successfully. It does not
yet check downstream dependencies (Postgres, Redis, index freshness) --
those checks are added in the stages that introduce those dependencies, so
this endpoint never claims health it hasn't verified.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app import __version__
from app.config import Settings, get_settings

router = APIRouter()


@router.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "version": __version__,
        "environment": settings.environment,
    }
