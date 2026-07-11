"""/query endpoint tests: auth, rate limiting, admission control 503,
and response shape -- with a stub service, no models."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.core.hybrid import RerankInfo
from app.generation.service import GenerationResult
from app.main import create_app
from app.storage.redis_store import RateLimitDecision


def make_result(query: str) -> GenerationResult:
    return GenerationResult(
        query=query, answer="stub answer.", status="degraded_no_llm",
        degraded=True, citations=["doc::c0"],
        retrieved_chunk_ids=["doc::c0"], retrieved_texts=["text"],
        rerank_status="full",
    )


class StubService:
    def __init__(self, block: threading.Event | None = None) -> None:
        self.block = block
        self.calls = 0

    def answer(self, query: str) -> GenerationResult:
        self.calls += 1
        if self.block is not None:
            assert self.block.wait(timeout=10), "test forgot to release"
        return make_result(query)


@pytest.fixture()
def client_factory(monkeypatch):
    def factory(service=None, env: dict | None = None):
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)
        from app.config import get_settings

        get_settings.cache_clear()
        app = create_app()
        app.state.service = service or StubService()
        return TestClient(app)
    return factory


def test_query_happy_path_carries_degradation_fields(client_factory):
    with client_factory() as client:
        resp = client.post("/query", json={"query": "what is a token bucket?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "stub answer."
    assert body["status"] == "degraded_no_llm"
    assert body["degraded"] is True
    assert body["rerank_status"] == "full"
    assert body["citations"] == ["doc::c0"]


def test_missing_api_key_is_401_when_keys_configured(client_factory):
    with client_factory(env={"API_KEYS": "s3cret-key-1"}) as client:
        resp = client.post("/query", json={"query": "q"})
        assert resp.status_code == 401
        resp = client.post("/query", json={"query": "q"},
                           headers={"x-api-key": "wrong"})
        assert resp.status_code == 401
        resp = client.post("/query", json={"query": "q"},
                           headers={"x-api-key": "s3cret-key-1"})
        assert resp.status_code == 200


def test_empty_query_rejected_422(client_factory):
    with client_factory() as client:
        assert client.post("/query", json={"query": ""}).status_code == 422


def test_rate_limit_429_with_retry_after(client_factory):
    class DenyingLimiter:
        def check_rate_limit(self, client_id, limit, window_s):
            return RateLimitDecision(allowed=False, remaining=0,
                                     retry_after_s=17)

    with client_factory() as client:
        client.app.state.redis_store = DenyingLimiter()
        resp = client.post("/query", json={"query": "q"})
    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "17"


def test_queue_full_returns_503_with_retry_after(client_factory):
    """Saturate max_concurrency=1 + max_queue_depth=1 deterministically:
    request A executes (blocked on an event), request B waits in the
    queue, request C must get an immediate 503 + Retry-After."""
    block = threading.Event()
    service = StubService(block=block)
    env = {"ADMISSION_MAX_CONCURRENCY": "1", "ADMISSION_MAX_QUEUE_DEPTH": "1"}
    with client_factory(service, env) as client:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(
                client.post, "/query", json={"query": "A"}
            )
            # wait until A is executing and B is queued
            deadline = time.time() + 5
            fut_b = None
            while time.time() < deadline:
                snap = client.get("/admin/admission").json()
                if snap["in_flight"] == 1 and fut_b is None:
                    fut_b = pool.submit(
                        client.post, "/query", json={"query": "B"}
                    )
                if snap["in_flight"] == 1 and snap["waiting"] == 1:
                    break
                time.sleep(0.02)
            else:
                pytest.fail("saturation never reached")

            resp_c = client.post("/query", json={"query": "C"})
            assert resp_c.status_code == 503
            assert "retry-after" in resp_c.headers
            assert int(resp_c.headers["retry-after"]) >= 1
            assert "not queued" in resp_c.json()["error"]

            block.set()
            assert fut_a.result().status_code == 200
            assert fut_b.result().status_code == 200

        snap = client.get("/admin/admission").json()
        assert snap["rejected_total"] == 1
        assert snap["admitted_total"] >= 2
    assert service.calls == 2   # C never reached the pipeline
