"""Stage 6: /metrics endpoint, response cache behavior, and metric
movement under traffic -- with stub services and a fake Redis."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.generation.service import GenerationResult
from app.main import create_app
from app.storage.redis_store import RateLimitDecision


def make_result(query: str, status: str = "ok") -> GenerationResult:
    return GenerationResult(
        query=query, answer="answer.", status=status, degraded=False,
        citations=["doc::c0"], retrieved_chunk_ids=["doc::c0"],
        retrieved_texts=["text"], rerank_status="full",
    )


class StubService:
    def __init__(self, status: str = "ok") -> None:
        self.calls = 0
        self.status = status

    def answer(self, query: str, **kwargs) -> GenerationResult:
        self.calls += 1
        return make_result(query, self.status)


class FakeRedis:
    """In-memory stand-in implementing the RedisStore surface the
    endpoint uses."""

    def __init__(self) -> None:
        self.kv: dict[str, bytes] = {}
        self.counters: dict[str, int] = {}

    def cache_get(self, key):
        return self.kv.get(key)

    def cache_set(self, key, value, ttl_s):
        self.kv[key] = value

    def bounded_incr(self, key, ttl_s):
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    def check_rate_limit(self, client_id, limit, window_s):
        return RateLimitDecision(True, limit, 0)


@pytest.fixture()
def client(monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    app.state.service = StubService()
    test_client = TestClient(app)
    with test_client:
        test_client.app.state.redis_store = FakeRedis()
        yield test_client


def counter_value(counter, **labels) -> float:
    try:
        return counter.labels(**labels)._value.get()
    except Exception:  # noqa: BLE001 - metric introspection in tests only
        return 0.0


def test_metrics_token_auth_when_configured(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("METRICS_TOKEN", "scrape-secret")
    get_settings.cache_clear()
    app = create_app()
    app.state.service = StubService()
    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 401
        assert client.get(
            "/metrics", headers={"Authorization": "Bearer wrong"}
        ).status_code == 401
        ok = client.get(
            "/metrics", headers={"Authorization": "Bearer scrape-secret"}
        )
        assert ok.status_code == 200
        assert "ragp_" in ok.text
    monkeypatch.delenv("METRICS_TOKEN")
    get_settings.cache_clear()


def test_metrics_endpoint_exposes_prometheus_format(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "ragp_http_request_duration_seconds" in resp.text
    assert "ragp_cache_requests_total" in resp.text


def test_incompatible_cached_payload_is_a_miss_not_a_500(client):
    """Regression (schema drift): a cache entry written by a previous
    deploy whose fields no longer fit QueryResponse (extra='forbid')
    must be treated as a miss and served fresh, not 500 for the TTL."""
    from app.api.query import _cache_key

    stale = b'{"query": "q", "removed_field_from_old_schema": true}'
    client.app.state.redis_store.kv[_cache_key("What is RAFT?", None)] = stale

    resp = client.post("/v1/query", json={"query": "What is RAFT?"})
    assert resp.status_code == 200
    assert resp.json()["cached"] is False          # served fresh
    assert client.app.state.service.calls == 1     # pipeline actually ran


def test_identical_query_served_from_cache_second_time(client):
    from app.observability import CACHE_REQUESTS

    hits_before = counter_value(CACHE_REQUESTS, result="hit")
    service = client.app.state.service

    first = client.post("/v1/query", json={"query": "What is RAFT?"})
    assert first.status_code == 200
    assert first.json()["cached"] is False
    assert service.calls == 1

    # same query, different whitespace/case: normalized to the same key
    second = client.post("/v1/query", json={"query": "  what is raft? "})
    assert second.status_code == 200
    assert second.json()["cached"] is True
    assert service.calls == 1               # pipeline NOT touched
    # cached response keeps its own fresh request id
    assert second.json()["request_id"] != first.json()["request_id"]
    assert counter_value(CACHE_REQUESTS, result="hit") == hits_before + 1


def test_transient_degradations_are_not_cached(client):
    client.app.state.service = StubService(status="degraded_quota_throttled")
    service = client.app.state.service
    client.post("/v1/query", json={"query": "transient one"})
    client.post("/v1/query", json={"query": "transient one"})
    assert service.calls == 2  # both hit the pipeline; nothing was cached


def test_http_request_duration_labels_use_route_template(client):
    client.post("/v1/query", json={"query": "metrics check"})
    text = client.get("/metrics").text
    assert 'path="/v1/query"' in text
    assert "metrics check" not in text  # no user data in labels


def test_error_counter_moves_on_unhandled_exception(monkeypatch):
    from app.config import get_settings
    from app.observability import ERRORS

    class Exploding:
        def answer(self, query, **kwargs):
            raise RuntimeError("boom")

    get_settings.cache_clear()
    app = create_app()
    app.state.service = Exploding()
    before = counter_value(ERRORS, type="unhandled")
    with TestClient(app, raise_server_exceptions=False) as client:
        client.post("/v1/query", json={"query": "explode"})
    assert counter_value(ERRORS, type="unhandled") == before + 1


def test_citation_and_llm_metrics_move_with_real_service():
    """Drive the REAL GenerationService with a fabricating fake LLM and
    assert the citation/llm counters move."""
    from app.core.hybrid import RerankInfo, RetrievedChunk
    from app.generation.llm_client import LLMResponse
    from app.generation.service import GenerationService
    from app.observability import CITATION_SENTENCES, LLM_REQUESTS

    chunk = RetrievedChunk(
        "d::c0", "The token bucket refills tokens at a fixed rate.", 1.0,
        "rerank",
    )

    class P:
        def retrieve(self, q):
            return [chunk], RerankInfo("full", 1, 1, 1.0)

    class FabricatingLLM:
        def generate(self, prompt, **kwargs):
            return LLMResponse(
                "The token bucket refills tokens at a fixed rate [1]. "
                "It was invented on Mars in 1602 [1].",
                "fake", 10, 10,
            )

    ok_before = counter_value(LLM_REQUESTS, outcome="ok")
    unsupported_before = counter_value(CITATION_SENTENCES,
                                       verdict="unsupported")
    result = GenerationService(P(), FabricatingLLM()).answer("q")
    assert result.status == "ok_partial_rejected"
    assert counter_value(LLM_REQUESTS, outcome="ok") == ok_before + 1
    assert counter_value(
        CITATION_SENTENCES, verdict="unsupported"
    ) == unsupported_before + 1
