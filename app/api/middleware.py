"""HTTP middleware: request IDs and request size limits.

Request ID: every request gets a UUID, bound into the structlog context
(so every log line of a request carries it), returned in X-Request-ID,
and included in error bodies -- clients report the id, operators grep it,
and no internal detail needs to leak to make errors diagnosable.

Size limit: requests with Content-Length above MAX_REQUEST_BYTES are
rejected 413 before the body is read. A missing Content-Length on a
body-carrying method is rejected 411 -- chunked uploads have no place on
this API and accepting them would bypass the size check.

Security headers: this app serves HTML and carries a session cookie, so
the browser-facing hardening belongs here rather than in a proxy we do
not control. The CSP is strict because the frontend earned it -- no
inline script, no inline style, no CDN (Stage 9.6's zero-dependency
choice pays off as a policy with no 'unsafe-inline' escape hatch).
"""

from __future__ import annotations

import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers["X-Request-ID"] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Browser-facing defenses. HSTS only in production: sending it from
    a local http:// dev server would pin the browser to https for
    localhost and break every other project on the same origin."""

    _CSP = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        # Google profile avatars are remote; data: covers inline favicons.
        "img-src 'self' https: data:; "
        "connect-src 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'none'; "
        "object-src 'none'"
    )

    def __init__(self, app, is_production: bool) -> None:
        super().__init__(app)
        self.is_production = is_production

    # FastAPI's Swagger UI is CDN-served with an inline bootstrap script,
    # so the strict policy cannot cover it. Scoped exception rather than
    # weakening the policy that protects the actual app: the docs page
    # renders a schema and holds no user data.
    _DOCS_CSP = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "img-src 'self' https: data:; "
        "frame-ancestors 'none'; "
        "object-src 'none'"
    )
    _DOCS_PATHS = frozenset({"/docs", "/redoc", "/docs/oauth2-redirect"})

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        headers = response.headers
        csp = (self._DOCS_CSP if request.url.path in self._DOCS_PATHS
               else self._CSP)
        headers.setdefault("Content-Security-Policy", csp)
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Referrer-Policy", "no-referrer")
        # This API has no need of camera/mic/geolocation; deny them so a
        # future XSS cannot ask either.
        headers.setdefault("Permissions-Policy",
                           "geolocation=(), microphone=(), camera=()")
        if self.is_production:
            headers.setdefault("Strict-Transport-Security",
                               "max-age=31536000; includeSubDomains")
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int,
                 upload_max_bytes: int | None = None) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes
        # The upload route carries whole documents; every other route
        # keeps the tight query-sized cap.
        self.upload_max_bytes = upload_max_bytes or max_bytes

    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            limit = (self.upload_max_bytes
                     if request.url.path == "/v1/documents"
                     else self.max_bytes)
            length = request.headers.get("content-length")
            if length is None:
                return JSONResponse(
                    status_code=411,
                    content={"error": "Content-Length required"},
                )
            try:
                n = int(length)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"error": "invalid Content-Length"},
                )
            if n > limit:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "request body too large",
                        "max_bytes": limit,
                    },
                )
        return await call_next(request)
