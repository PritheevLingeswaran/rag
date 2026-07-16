"""Stage 9.6: Google-auth session layer.

Covers the decisions from docs/stage9_5_frontend_auth.md the way the
Stage 5 suite covers API keys: identity branches, claim verification,
session lifecycle, per-user limits, and the fail-closed directions
(missing config => clear 503; bad session => logged out, never access).
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from app.api.auth import SESSION_COOKIE, verify_id_token_claims
from app.generation.service import GenerationResult
from app.main import create_app
from app.storage.redis_store import RateLimitDecision


class StubService:
    def __init__(self) -> None:
        self.calls = 0

    def answer(self, query, **kwargs):
        self.calls += 1
        return GenerationResult(
            query=query, answer="answer.", status="ok", degraded=False,
            citations=["doc::c0"], retrieved_chunk_ids=["doc::c0"],
            retrieved_texts=["text"], rerank_status="full",
        )


class FakeRedis:
    def __init__(self) -> None:
        self.kv: dict[str, bytes] = {}
        self.counters: dict[str, int] = {}
        self.rate_limit_calls: list[tuple[str, int]] = []

    def cache_get(self, key):
        return None

    def cache_set(self, key, value, ttl_s):
        pass

    def bounded_incr(self, key, ttl_s):
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    def check_rate_limit(self, client_id, limit, window_s):
        self.rate_limit_calls.append((client_id, limit))
        return RateLimitDecision(True, limit, 0)

    def session_set(self, sid, payload, ttl_s):
        self.kv[sid] = payload
        return True

    def session_get(self, sid):
        return self.kv.get(sid)

    def session_delete(self, sid):
        self.kv.pop(sid, None)


@pytest.fixture()
def web(monkeypatch):
    """App with a fake Redis carrying one live session."""
    from app.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    app.state.service = StubService()
    client = TestClient(app)
    with client:
        fake = FakeRedis()
        fake.session_set("sess-abc", json.dumps(
            {"user_id": "u_test1", "email": "t@example.com",
             "name": "Test", "avatar_url": None}
        ).encode(), 3600)
        client.app.state.redis_store = fake
        yield client, fake


# ---- id_token claim verification (the mandatory OIDC checks) ----

GOOD = {"aud": "client-1", "iss": "https://accounts.google.com",
        "exp": time.time() + 600, "sub": "g-123", "email": "a@b.c"}


def test_valid_claims_pass():
    assert verify_id_token_claims(dict(GOOD), "client-1")["sub"] == "g-123"


@pytest.mark.parametrize("mutation,match", [
    ({"aud": "someone-else"}, "aud"),
    ({"iss": "https://evil.example"}, "iss"),
    ({"exp": time.time() - 5}, "expired"),
    ({"sub": ""}, "sub"),
])
def test_bad_claims_rejected(mutation, match):
    claims = dict(GOOD)
    claims.update(mutation)
    with pytest.raises(ValueError, match=match):
        verify_id_token_claims(claims, "client-1")


# ---- session identity + per-user limits ----

def test_session_cookie_yields_user_identity_and_user_limits(web):
    client, fake = web
    client.cookies.set(SESSION_COOKIE, "sess-abc")
    resp = client.post("/v1/query", json={"query": "hello"})
    assert resp.status_code == 200
    # rate limit was checked with the USER identity and USER limit (10)
    assert fake.rate_limit_calls == [("user:u_test1", 10)]


def test_no_session_no_key_in_dev_falls_back_to_anon(web):
    client, fake = web
    resp = client.post("/v1/query", json={"query": "hello"})
    assert resp.status_code == 200
    client_id, limit = fake.rate_limit_calls[-1]
    assert client_id.startswith("anon:")
    assert limit == 30  # API-key/default limit, not the user limit


def test_garbage_session_cookie_is_logged_out_not_error(web):
    client, _ = web
    client.cookies.set(SESSION_COOKIE, "no-such-session")
    assert client.get("/auth/me").status_code == 401


def test_me_returns_profile_for_live_session(web):
    client, _ = web
    client.cookies.set(SESSION_COOKIE, "sess-abc")
    body = client.get("/auth/me").json()
    assert body["user_id"] == "u_test1"
    assert body["email"] == "t@example.com"


def test_logout_deletes_session_and_clears_cookie(web):
    client, fake = web
    client.cookies.set(SESSION_COOKIE, "sess-abc")
    resp = client.post("/auth/logout")
    assert resp.status_code == 204
    assert "sess-abc" not in fake.kv
    assert client.get("/auth/me").status_code == 401


# ---- fail-closed configuration ----

def test_login_route_503_with_clear_error_when_unconfigured(web):
    client, _ = web
    resp = client.get("/auth/google/login", follow_redirects=False)
    assert resp.status_code == 503
    assert "GOOGLE_CLIENT_ID" in resp.json()["error"]


def test_callback_rejects_missing_or_mismatched_state(monkeypatch, web):
    client, _ = web
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "csec")
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused/db")
    from app.config import get_settings

    get_settings.cache_clear()
    resp = client.get("/auth/google/callback?code=x&state=forged",
                      follow_redirects=False)
    assert resp.status_code == 400
    get_settings.cache_clear()


# ---- frontend serving ----

def test_root_serves_frontend_index(web):
    client, _ = web
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "ragp" in resp.text


def test_static_assets_served_under_app(web):
    client, _ = web
    assert client.get("/app/app.js").status_code == 200
    assert client.get("/app/style.css").status_code == 200
