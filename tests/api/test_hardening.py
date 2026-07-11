"""Stage 5 hardening tests: exception leak-proofing (definition of
done), strict validation, size limits, CORS lockdown, daily quotas,
max_tokens clamping, and OpenAPI exposure."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.errors import ConfigurationError
from app.generation.service import GenerationResult
from app.main import create_app


def make_result(query: str) -> GenerationResult:
    return GenerationResult(
        query=query, answer="stub answer.", status="ok", degraded=False,
        citations=["doc::c0"], retrieved_chunk_ids=["doc::c0"],
        retrieved_texts=["text"], rerank_status="full",
    )


class StubService:
    def __init__(self) -> None:
        self.kwargs_seen: dict | None = None

    def answer(self, query: str, **kwargs) -> GenerationResult:
        self.kwargs_seen = kwargs
        return make_result(query)


class ExplodingService:
    """Simulates an internal bug with juicy secrets in the message."""

    def answer(self, query: str, **kwargs):
        raise RuntimeError(
            "psycopg2.OperationalError: FATAL password 'hunter2' for "
            "user ragp_admin at 10.0.3.17:5432 -- traceback follows"
        )


@pytest.fixture()
def client_factory(monkeypatch):
    def factory(service=None, env: dict | None = None,
                raise_server_exceptions: bool = True):
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)
        from app.config import get_settings

        get_settings.cache_clear()
        app = create_app()
        app.state.service = service or StubService()
        return TestClient(app,
                          raise_server_exceptions=raise_server_exceptions)
    return factory


# ---- DEFINITION OF DONE: no internal detail ever leaks ----

def test_internal_exception_leaks_nothing_to_client(client_factory):
    with client_factory(ExplodingService(),
                        raise_server_exceptions=False) as client:
        resp = client.post("/v1/query", json={"query": "trigger the bug"})

    assert resp.status_code == 500
    body = resp.json()
    # exactly two keys: a generic message and a support reference id
    assert set(body.keys()) == {"error", "request_id"}
    assert body["error"] == "internal server error"
    assert body["request_id"]
    assert resp.headers["x-request-id"] == body["request_id"]

    # nothing from the actual exception is present anywhere
    text = resp.text.lower()
    for secret in ("hunter2", "psycopg", "operationalerror", "10.0.3.17",
                   "ragp_admin", "traceback", "runtimeerror", ".py"):
        assert secret not in text


def test_malformed_json_body_is_clean_422(client_factory):
    with client_factory() as client:
        resp = client.post("/v1/query", content=b'{"query": !!!broken',
                           headers={"content-type": "application/json"})
    assert resp.status_code == 422
    assert "Traceback" not in resp.text
    assert "json" in resp.text.lower()  # describes the client's input


# ---- strict validation ----

def test_unknown_fields_rejected(client_factory):
    with client_factory() as client:
        resp = client.post("/v1/query", json={
            "query": "hello", "debug": True, "internal_flag": "x",
        })
    assert resp.status_code == 422


def test_query_length_bounds(client_factory):
    with client_factory() as client:
        assert client.post("/v1/query",
                           json={"query": ""}).status_code == 422
        assert client.post("/v1/query",
                           json={"query": "x" * 2001}).status_code == 422
        assert client.post("/v1/query",
                           json={"query": 12345}).status_code == 422


def test_max_tokens_bounds_and_passthrough(client_factory):
    service = StubService()
    with client_factory(service) as client:
        assert client.post("/v1/query", json={
            "query": "q", "max_tokens": 999999,
        }).status_code == 422
        assert client.post("/v1/query", json={
            "query": "q", "max_tokens": 4,
        }).status_code == 422
        resp = client.post("/v1/query", json={"query": "q", "max_tokens": 64})
        assert resp.status_code == 200
    assert service.kwargs_seen == {"max_output_tokens": 64}


# ---- request size limit ----

def test_oversized_body_rejected_413_before_parsing(client_factory):
    with client_factory() as client:
        resp = client.post("/v1/query",
                           json={"query": "x" * 20_000})
    assert resp.status_code == 413
    assert resp.json()["max_bytes"] == 16_384


# ---- CORS lockdown ----

def test_no_cors_headers_by_default(client_factory):
    with client_factory() as client:
        resp = client.post("/v1/query", json={"query": "q"},
                           headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in resp.headers


def test_cors_allows_only_configured_origin(client_factory):
    env = {"CORS_ORIGINS": "https://app.example.com"}
    with client_factory(env=env) as client:
        ok = client.post("/v1/query", json={"query": "q"},
                         headers={"Origin": "https://app.example.com"})
        assert ok.headers["access-control-allow-origin"] == "https://app.example.com"
        other = client.post("/v1/query", json={"query": "q"},
                            headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in other.headers


def test_wildcard_cors_refused_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("CORS_ORIGINS", "*")
    monkeypatch.setenv("API_KEYS", "k")
    from app.config import get_settings

    get_settings.cache_clear()
    with pytest.raises(ConfigurationError, match="CORS_ORIGINS"):
        create_app()


# ---- per-key daily quota (cost guardrail) ----

def test_daily_quota_429_with_retry_after(client_factory):
    class QuotaExhaustedStore:
        def bounded_incr(self, key, ttl_s):
            assert key.startswith("apiq:")
            return 501  # over the 500/day default

        def check_rate_limit(self, client_id, limit, window_s):
            raise AssertionError("must not reach per-minute check")

    with client_factory() as client:
        client.app.state.redis_store = QuotaExhaustedStore()
        resp = client.post("/v1/query", json={"query": "q"})
    assert resp.status_code == 429
    assert "daily quota" in resp.json()["error"]
    assert 0 < int(resp.headers["retry-after"]) <= 86_400


# ---- OpenAPI ----

def test_openapi_documents_v1(client_factory):
    with client_factory() as client:
        spec = client.get("/openapi.json").json()
        assert "/v1/query" in spec["paths"]
        props = spec["components"]["schemas"]["QueryRequest"]["properties"]
        assert props["query"]["maxLength"] == 2000
        assert client.get("/docs").status_code == 200
        # admin endpoint hidden from public docs
        assert "/v1/admin/admission" not in spec["paths"]
