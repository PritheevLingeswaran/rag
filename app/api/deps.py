"""Shared API dependencies: authentication and client identity."""

from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, Request

from app.config import Settings, get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)


def get_client_id(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> str:
    """Authenticates the request and returns a stable client identity
    (used for rate limiting and quotas). 401 on failure; anonymous
    access is allowed only outside production, and logged."""
    keys = settings.api_key_list
    provided = request.headers.get("x-api-key", "")
    if keys:
        for key in keys:
            if hmac.compare_digest(provided, key):
                return f"key:{key[:4]}...{len(key)}"
        raise HTTPException(status_code=401,
                            detail="missing or invalid API key")
    if settings.is_production:
        # create_app refuses to boot keyless in production; this guards
        # against config reloads ever bypassing that.
        raise HTTPException(status_code=401,
                            detail="missing or invalid API key")
    client_host = request.client.host if request.client else "unknown"
    logger.info("anonymous_request_dev_mode", client=client_host)
    return f"anon:{client_host}"
