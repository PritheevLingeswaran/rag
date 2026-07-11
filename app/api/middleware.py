"""HTTP middleware: request IDs and request size limits.

Request ID: every request gets a UUID, bound into the structlog context
(so every log line of a request carries it), returned in X-Request-ID,
and included in error bodies -- clients report the id, operators grep it,
and no internal detail needs to leak to make errors diagnosable.

Size limit: requests with Content-Length above MAX_REQUEST_BYTES are
rejected 413 before the body is read. A missing Content-Length on a
body-carrying method is rejected 411 -- chunked uploads have no place on
this API and accepting them would bypass the size check.
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


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
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
            if n > self.max_bytes:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "request body too large",
                        "max_bytes": self.max_bytes,
                    },
                )
        return await call_next(request)
