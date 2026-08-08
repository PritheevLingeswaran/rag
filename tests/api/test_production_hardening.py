"""Stage 9.9 production hardening: readiness, security headers, and
authentication on the write endpoint.

Each test here corresponds to a defect found by probing a deployed
instance, not to a hypothetical: an instance serving without a pipeline
reported "ok" to Render forever while every request 500'd and paged;
the browser surface shipped with no CSP despite carrying a session
cookie; and the upload endpoint mutated the served index with no
authentication at all.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app


@pytest.fixture
def unready_client(monkeypatch):
    """An instance configured to serve but with no pipeline -- the exact
    misconfiguration that used to 500 and page on every request."""
    monkeypatch.setenv("SERVE_PIPELINE", "false")
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        # Force the 'should be serving but isn't' state: settings say
        # serve, app.state.service was never set.
        monkeypatch.setenv("SERVE_PIPELINE", "true")
        get_settings.cache_clear()
        yield client
    get_settings.cache_clear()


def test_readiness_reports_unready_instance(unready_client):
    resp = unready_client.get("/health/ready")
    assert resp.status_code == 503
    assert resp.json()["status"] == "not_ready"


def test_liveness_stays_200_but_declares_not_ready(unready_client):
    """Liveness must not fail: restarting fixes nothing here. It reports
    the truth in a field instead."""
    resp = unready_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["ready"] is False


def test_query_on_unready_instance_is_503_not_500(unready_client):
    resp = unready_client.post("/v1/query", json={"query": "raft leader"})
    assert resp.status_code == 503
    assert resp.json()["error"] == "service not ready"
    assert resp.headers["Retry-After"] == "30"


def test_ready_instance_reports_ready():
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:  # SERVE_PIPELINE unset => not serving
        assert client.get("/health/ready").status_code == 200
        assert client.get("/health").json()["ready"] is True


def test_security_headers_present_on_app_responses():
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        headers = client.get("/health").headers
        csp = headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp
        assert "'unsafe-inline'" not in csp        # the app earns a strict CSP
        assert "frame-ancestors 'none'" in csp
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["Referrer-Policy"] == "no-referrer"


def test_docs_get_a_scoped_csp_exception_not_a_weakened_policy():
    """Swagger UI needs its CDN + inline bootstrap; the app's own policy
    must stay strict."""
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        docs_csp = client.get("/docs").headers["Content-Security-Policy"]
        assert "cdn.jsdelivr.net" in docs_csp
        app_csp = client.get("/").headers["Content-Security-Policy"]
        assert "cdn.jsdelivr.net" not in app_csp


def test_frontend_assets_must_revalidate(monkeypatch):
    """Without Cache-Control a browser heuristically reuses cached assets
    without asking, so a deploy can render new HTML against old CSS."""
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        for path in ("/", "/app/style.css", "/app/app.js"):
            resp = client.get(path)
            assert resp.status_code == 200, path
            assert resp.headers["Cache-Control"] == "no-cache", path
            assert resp.headers.get("ETag"), path  # 304s stay cheap


def test_api_responses_are_not_given_frontend_cache_headers():
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        assert "Cache-Control" not in client.get("/health").headers


def test_hsts_only_in_production(monkeypatch):
    get_settings.cache_clear()
    with TestClient(create_app()) as dev:
        assert "Strict-Transport-Security" not in dev.get("/health").headers

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("API_KEYS", "test-key-for-prod-boot")
    get_settings.cache_clear()
    with TestClient(create_app()) as prod:
        assert "max-age=" in prod.get("/health").headers[
            "Strict-Transport-Security"
        ]
    get_settings.cache_clear()


def test_upload_requires_authentication_when_keys_are_configured(monkeypatch):
    """The write endpoint must be no easier a target than the read path."""
    monkeypatch.setenv("API_KEYS", "secret-key-1")
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        unauth = client.post(
            "/v1/documents",
            files={"file": ("notes.txt", "some text", "text/plain")},
        )
        assert unauth.status_code == 401
    get_settings.cache_clear()
