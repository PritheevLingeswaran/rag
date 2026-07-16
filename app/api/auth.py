"""Google OAuth login (Stage 9.6, implementing the Stage 9.5 decisions).

Flow: /auth/google/login redirects to Google (authorization-code flow,
server-side confidential client) -> /auth/google/callback exchanges the
code at Google's token endpoint over TLS, validates the id_token CLAIMS
(aud/iss/exp), upserts the user row, stores an opaque session in Redis,
and sets an HttpOnly cookie. The browser never holds a key or token.

id_token signature is deliberately NOT verified: per OpenID Connect
Core 3.1.3.7, a client that received the token directly from the token
endpoint over TLS MAY rely on that channel instead of validating the
signature. This avoids a JWT/JWKS dependency; the claims checks below
are still mandatory and tested.

CSRF on the OAuth flow: a random `state` is set in a short-lived
HttpOnly cookie before redirecting and must match at the callback.
Google's tokens are discarded after verification -- we take identity,
not API access (privacy policy commitment).

Failure policy: every dependency gap is a clear, typed JSON error --
missing config => 503 auth_unavailable; a failed exchange => 502 with
no Google internals leaked; an invalid state => 400. The API-key
surface is untouched by any of this.
"""

from __future__ import annotations

import base64
import json
import secrets
import time

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
SESSION_COOKIE = "ragp_session"
STATE_COOKIE = "ragp_oauth_state"
STATE_TTL_S = 600


class AuthConfigError(Exception):
    pass


def _require_auth_deps(request: Request):
    """All three legs the login flow needs; absent => typed 503."""
    settings = get_settings()
    redis_store = getattr(request.app.state, "redis_store", None)
    if (not settings.google_client_id or not settings.google_client_secret
            or redis_store is None or not settings.database_url):
        raise AuthConfigError(
            "login unavailable: requires GOOGLE_CLIENT_ID, "
            "GOOGLE_CLIENT_SECRET, REDIS_URL and DATABASE_URL"
        )
    return settings, redis_store


def _auth_unavailable(exc: AuthConfigError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"error": str(exc)})


def _redirect_uri(request: Request) -> str:
    # Requires uvicorn --proxy-headers so base_url is https behind
    # Render's proxy (set in the Dockerfile CMD).
    return str(request.base_url).rstrip("/") + "/auth/google/callback"


def verify_id_token_claims(claims: dict, client_id: str,
                           now: float | None = None) -> dict:
    """Mandatory claim checks (OIDC 3.1.3.7). Returns the claims or
    raises ValueError with the failing check named."""
    if claims.get("aud") != client_id:
        raise ValueError("id_token aud mismatch")
    if claims.get("iss") not in ("accounts.google.com",
                                 "https://accounts.google.com"):
        raise ValueError("id_token iss mismatch")
    if float(claims.get("exp", 0)) <= (now if now is not None else time.time()):
        raise ValueError("id_token expired")
    if not claims.get("sub"):
        raise ValueError("id_token missing sub")
    return claims


def _decode_jwt_payload(id_token: str) -> dict:
    try:
        payload_b64 = id_token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        return json.loads(base64.urlsafe_b64decode(payload_b64))
    except (IndexError, ValueError) as exc:
        raise ValueError(f"undecodable id_token: {exc}") from exc


def _upsert_user(database_url: str, claims: dict) -> str:
    """Create-or-update the user row at login; returns internal user_id.
    The ONLY serving-path touch of Postgres (Stage 9.5 point 2)."""
    import psycopg

    with psycopg.connect(database_url) as conn:
        row = conn.execute(
            "SELECT user_id FROM users WHERE google_sub = %s",
            (claims["sub"],),
        ).fetchone()
        if row:
            user_id = row[0]
            conn.execute(
                "UPDATE users SET email = %s, display_name = %s, "
                "avatar_url = %s, last_login_at = now() WHERE user_id = %s",
                (claims.get("email", ""), claims.get("name"),
                 claims.get("picture"), user_id),
            )
        else:
            user_id = f"u_{secrets.token_urlsafe(12)}"
            conn.execute(
                "INSERT INTO users (user_id, google_sub, email, "
                "display_name, avatar_url) VALUES (%s, %s, %s, %s, %s)",
                (user_id, claims["sub"], claims.get("email", ""),
                 claims.get("name"), claims.get("picture")),
            )
        conn.commit()
    return user_id


@router.get("/google/login", include_in_schema=False)
def google_login(request: Request):
    try:
        settings, _ = _require_auth_deps(request)
    except AuthConfigError as exc:
        return _auth_unavailable(exc)
    state = secrets.token_urlsafe(24)
    params = httpx.QueryParams({
        "client_id": settings.google_client_id,
        "redirect_uri": _redirect_uri(request),
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
    })
    resp = RedirectResponse(f"{GOOGLE_AUTH_URL}?{params}", status_code=302)
    resp.set_cookie(STATE_COOKIE, state, max_age=STATE_TTL_S,
                    httponly=True, secure=settings.is_production,
                    samesite="lax")
    return resp


@router.get("/google/callback", include_in_schema=False)
def google_callback(request: Request, code: str = "", state: str = ""):
    try:
        settings, redis_store = _require_auth_deps(request)
    except AuthConfigError as exc:
        return _auth_unavailable(exc)

    expected_state = request.cookies.get(STATE_COOKIE)
    if not code or not state or not expected_state or state != expected_state:
        return JSONResponse(status_code=400,
                            content={"error": "invalid oauth state"})

    try:
        token_resp = httpx.post(GOOGLE_TOKEN_URL, timeout=10.0, data={
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": _redirect_uri(request),
        })
        token_resp.raise_for_status()
        id_token = token_resp.json()["id_token"]
        claims = verify_id_token_claims(
            _decode_jwt_payload(id_token), settings.google_client_id
        )
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        # No Google response detail reaches the client (error policy).
        logger.error("oauth_exchange_failed", error=str(exc))
        return JSONResponse(status_code=502,
                            content={"error": "google sign-in failed; try again"})

    try:
        user_id = _upsert_user(settings.database_url, claims)
    except Exception as exc:  # noqa: BLE001 - login must fail closed+clean
        logger.error("user_upsert_failed", error=str(exc))
        return JSONResponse(status_code=503,
                            content={"error": "sign-in temporarily unavailable"})

    session_id = secrets.token_urlsafe(32)
    profile = {
        "user_id": user_id,
        "email": claims.get("email", ""),
        "name": claims.get("name"),
        "avatar_url": claims.get("picture"),
    }
    # Profile snapshot lives IN the session so /auth/me and the identity
    # dependency never touch Postgres per request (Stage 9.5 point 2).
    if not redis_store.session_set(
        session_id, json.dumps(profile).encode(), settings.session_ttl_s
    ):
        return JSONResponse(status_code=503,
                            content={"error": "sign-in temporarily unavailable"})

    logger.info("user_logged_in", user_id=user_id)
    resp = RedirectResponse("/", status_code=302)
    resp.set_cookie(SESSION_COOKIE, session_id,
                    max_age=settings.session_ttl_s, httponly=True,
                    secure=settings.is_production, samesite="lax")
    resp.delete_cookie(STATE_COOKIE)
    return resp


def load_session(request: Request) -> dict | None:
    """Session profile for the current request, or None (= logged out;
    also the failure direction for any Redis problem)."""
    session_id = request.cookies.get(SESSION_COOKIE)
    if not session_id:
        return None
    redis_store = getattr(request.app.state, "redis_store", None)
    if redis_store is None:
        return None
    raw = redis_store.session_get(session_id)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


@router.get("/me", include_in_schema=False)
def me(request: Request):
    profile = load_session(request)
    if profile is None:
        return JSONResponse(status_code=401, content={"error": "not signed in"})
    return profile


@router.post("/logout", include_in_schema=False)
def logout(request: Request):
    session_id = request.cookies.get(SESSION_COOKIE)
    redis_store = getattr(request.app.state, "redis_store", None)
    if session_id and redis_store is not None:
        redis_store.session_delete(session_id)
    resp = Response(status_code=204)
    resp.delete_cookie(SESSION_COOKIE)
    return resp
