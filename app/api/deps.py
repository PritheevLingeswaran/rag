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
    from app.observability import ERRORS

    keys = settings.api_key_list
    provided = request.headers.get("x-api-key", "")
    if provided and keys:
        for key in keys:
            if hmac.compare_digest(provided, key):
                # Truncated digest, not a key prefix: two keys sharing
                # their first characters must not share rate-limit/quota
                # buckets. Still a non-reversible tag (privacy policy).
                import hashlib

                return f"key:{hashlib.sha256(key.encode()).hexdigest()[:12]}"
        ERRORS.labels(type="auth_failed").inc()
        raise HTTPException(status_code=401,
                            detail="missing or invalid API key")

    # Web session branch (Stage 9.6): opaque HttpOnly cookie -> Redis
    # session -> per-user identity. Checked only when no API key header
    # was presented; a bad/expired session falls through to the
    # key-required paths below (401 in production).
    from app.api.auth import load_session

    profile = load_session(request)
    if profile is not None and profile.get("user_id"):
        return f"user:{profile['user_id']}"

    if keys or settings.is_production:
        # create_app refuses to boot keyless in production; this guards
        # against config reloads ever bypassing that.
        raise HTTPException(status_code=401,
                            detail="missing or invalid API key")
    client_host = request.client.host if request.client else "unknown"
    logger.info("anonymous_request_dev_mode", client=client_host)
    return f"anon:{client_host}"
