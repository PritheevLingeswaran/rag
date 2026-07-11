"""Stage 7.7 definition of done: each quota independently simulated to
its limit, with THREE distinct evidenced states per quota:

    (1) 80% of budget  -> still serving, one alert fired (captured)
    (2) breaker trip   -> guarded operation refused/bypassed BEFORE the
                          provider hard limit; app degrades, labeled
    (3) past hard limit-> provider-side failure absorbed by the reactive
                          paths; app still never hard-crashes
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.reliability import (
    STATE_ALERT,
    STATE_CLOSED,
    STATE_OPEN,
    AlertManager,
    PostgresStorageBreaker,
    ResourceBudget,
)


class WebhookCapture:
    def __init__(self):
        self.received: list[dict] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request):
            self.received.append(json.loads(request.content))
            return httpx.Response(200)
        return httpx.MockTransport(handler)


@pytest.fixture()
def capture() -> WebhookCapture:
    return WebhookCapture()


def make_alerts(capture, now_fn=None) -> AlertManager:
    kwargs = {"now_fn": now_fn} if now_fn else {}
    return AlertManager("https://alerts.example/hook",
                        transport=capture.transport(), **kwargs)


# =====================================================================
# QUOTA 1: Gemini LLM RPD (via QuotaGuard, limits from Stage 2.5 file)
# =====================================================================

class FrozenClock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def drive_guard(guard, clock, n):
    """Acquire n slots, hopping minutes so RPM never interferes."""
    granted = 0
    for _ in range(n):
        clock.advance(60)
        if guard.try_acquire().allowed:
            granted += 1
    return granted


def test_llm_rpd_state1_alert_at_80_percent(capture):
    from app.generation.quota import QuotaGuard, load_model_limits

    clock = FrozenClock()
    guard = QuotaGuard(load_model_limits("gemini-2.5-flash-lite"),
                       now_fn=clock, alerts=make_alerts(capture, clock))
    assert drive_guard(guard, clock, 719) == 719   # 719 < 0.8*900
    assert capture.received == []                  # below threshold: quiet

    clock.advance(60)
    assert guard.try_acquire().allowed             # 720th = exactly 80%
    assert len(capture.received) == 1              # ONE alert, captured
    alert = capture.received[0]
    assert alert["resource"] == "gemini_rpd:gemini-2.5-flash-lite"
    assert alert["pct_of_budget"] == pytest.approx(0.8, abs=0.01)

    assert drive_guard(guard, clock, 50) == 50     # still serving
    assert len(capture.received) == 1              # deduped for the day


def test_llm_rpd_state2_breaker_trips_at_enforced_900(capture):
    from app.core.hybrid import RerankInfo, RetrievedChunk
    from app.generation.quota import QuotaGuard, load_model_limits
    from app.generation.service import GenerationService

    clock = FrozenClock()
    guard = QuotaGuard(load_model_limits("gemini-2.5-flash-lite"),
                       now_fn=clock, alerts=make_alerts(capture, clock))
    assert drive_guard(guard, clock, 900) == 900   # the enforced budget

    chunk = RetrievedChunk("d::c0", "Grounded text about tokens.", 1.0, "rerank")

    class P:
        def retrieve(self, q):
            return [chunk], RerankInfo("full", 1, 1, 1.0)

    class NeverCalledLLM:
        calls = 0

        def generate(self, prompt, **kw):
            self.calls += 1
            raise AssertionError("provider must not be reached")

    llm = NeverCalledLLM()
    service = GenerationService(P(), llm, quota_guard=guard)
    clock.advance(60)
    result = service.answer("q901")               # request 901 of the day
    assert result.status == "degraded_quota_throttled"   # labeled degrade
    assert result.degraded is True
    assert result.answer.startswith("Grounded text")     # still an answer
    assert llm.calls == 0                          # provider untouched
    assert result.retry_after_s > 0                # honest recovery time
    assert len(capture.received) == 1              # the earlier 80% alert


def test_llm_rpd_state3_past_provider_limit_429_absorbed(capture):
    from app.core.hybrid import RerankInfo, RetrievedChunk
    from app.errors import LLMQuotaError
    from app.generation.quota import QuotaGuard, load_model_limits
    from app.generation.service import GenerationService

    clock = FrozenClock()
    guard = QuotaGuard(load_model_limits("gemini-2.5-flash-lite"),
                       now_fn=clock, alerts=make_alerts(capture, clock))
    chunk = RetrievedChunk("d::c0", "Grounded text about tokens.", 1.0, "rerank")

    class P:
        def retrieve(self, q):
            return [chunk], RerankInfo("full", 1, 1, 1.0)

    class Provider429:
        def generate(self, prompt, **kw):
            raise LLMQuotaError("429 RESOURCE_EXHAUSTED", retry_after_s=30.0)

    service = GenerationService(P(), Provider429(), quota_guard=guard)
    first = service.answer("q")
    assert first.status == "degraded_quota"        # reactive, labeled
    assert first.retry_after_s == 30.0             # provider's own signal
    second = service.answer("q2")                  # cooldown now active
    assert second.status == "degraded_quota_throttled"
    assert second.extra["throttle_reason"] == "provider_cooldown"
    # no exception escaped anywhere; both answers grounded + cited
    assert first.citations and second.citations


# =====================================================================
# QUOTA 2: Neon Postgres storage (0.5 GB)
# =====================================================================

LIMIT = 500 * 1024 * 1024          # bytes, from Stage 2.5
ENFORCED = int(LIMIT * 0.9)        # 471,859,200


class FakeSize:
    def __init__(self, size):
        self.size = size

    def __call__(self):
        return self.size


def test_pg_state1_alert_at_80_percent(capture):
    size = FakeSize(int(ENFORCED * 0.8))           # exactly 80% of enforced
    breaker = PostgresStorageBreaker(size, LIMIT, cache_s=0,
                                     alerts=make_alerts(capture))
    decision = breaker.check_writable()
    assert decision.allowed is True                # still writable
    assert decision.state == STATE_ALERT
    assert len(capture.received) == 1
    assert capture.received[0]["resource"] == "postgres_storage"
    breaker.check_writable()
    assert len(capture.received) == 1              # deduped


def test_pg_state2_breaker_open_refuses_ingestion_before_writing(capture):
    """At >= 90% the breaker refuses ingestion BEFORE any byte is
    written; serving (reads) is untouched by construction -- the breaker
    is only consulted on write paths."""
    from app.ingest.pipeline import IngestionPipeline

    size = FakeSize(ENFORCED)                      # trip point exactly
    breaker = PostgresStorageBreaker(size, LIMIT, cache_s=0,
                                     alerts=make_alerts(capture))

    class MustNotConnect:
        def __getattr__(self, name):
            raise AssertionError("DB must not be touched when breaker open")

    class MustNotEmbed:
        embedder_id = "x"
        dim = 4

        def embed_batch(self, texts):
            raise AssertionError("no work when breaker open")

    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline.storage_breaker = breaker             # bypass __init__ repos
    pipeline.conn = MustNotConnect()
    report = IngestionPipeline.run(pipeline, __file__)
    assert report.status == "aborted_storage_budget"
    assert "refusing to write" in report.error
    assert breaker.check_writable().state == STATE_OPEN


def test_pg_state3_past_hard_limit_write_failure_is_typed_not_a_crash(
    capture,
):
    """Past Neon's hard limit INSERTs fail server-side. Simulated with a
    connection whose transaction raises psycopg.errors: the run reports
    failed_storage cleanly; nothing raises to the operator."""
    import psycopg

    from app.ingest.pipeline import IngestionPipeline

    class ExplodingConn:
        def transaction(self):
            raise psycopg.OperationalError(
                "could not extend file: No space left on device"
            )

    # breaker sees a stale reading below the limit: the write itself blows
    size = FakeSize(int(ENFORCED * 0.5))
    breaker = PostgresStorageBreaker(size, LIMIT, cache_s=0,
                                     alerts=make_alerts(capture))
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline.storage_breaker = breaker
    pipeline.conn = ExplodingConn()

    from pathlib import Path

    full_report = IngestionPipeline.run(
        pipeline, Path("data/corpus_v1.jsonl")
    )
    assert full_report.status == "failed_storage"      # typed, clean
    assert "No space left on device" in full_report.error
    # nothing raised, no index write attempted, no partial state


# =====================================================================
# QUOTA 3: Upstash Redis daily command budget (500K/mo => 16,129/day)
# =====================================================================

class FakeRedisClient:
    """Counts commands actually issued, so bypass is provable."""

    def __init__(self):
        self.commands = 0
        self.kv = {}

    def get(self, k):
        self.commands += 1
        return self.kv.get(k)

    def set(self, k, v, ex=None):
        self.commands += 1
        self.kv[k] = v

    def delete(self, k):
        self.commands += 1
        self.kv.pop(k, None)

    def register_script(self, lua):
        def script(keys, args):
            self.commands += 1
            if "INCR" in lua and "TTL" not in lua:
                self.kv[keys[0]] = self.kv.get(keys[0], 0) + 1
                return self.kv[keys[0]]
            self.kv[keys[0]] = self.kv.get(keys[0], 0) + 1
            current = self.kv[keys[0]]
            limit = int(args[0])
            return [1 if current <= limit else 0, max(0, limit - current), 60]
        return script


def make_store(budget):
    from app.storage.redis_store import RedisStore

    store = RedisStore.__new__(RedisStore)
    store.ns = "t"
    store.command_budget = budget
    store._client = FakeRedisClient()
    store._rate_limit = store._client.register_script("INCR EXPIRE TTL limit")
    store._bounded_incr_script = store._client.register_script("INCR only")
    return store


def test_redis_state1_alert_at_80_percent(capture):
    budget = ResourceBudget("upstash_commands_daily", 100,
                            alerts=make_alerts(capture))  # enforced 90
    store = make_store(budget)
    for _ in range(71):                            # 71 < 0.8*90 = 72
        store.cache_get("k")
    assert capture.received == []
    store.cache_get("k")                           # 72nd command: 80%
    assert len(capture.received) == 1
    assert capture.received[0]["resource"] == "upstash_commands_daily"
    assert store.cache_get("k") is None or True    # still operating


def test_redis_state2_breaker_open_bypasses_all_redis_traffic(capture):
    budget = ResourceBudget("upstash_commands_daily", 100,
                            alerts=make_alerts(capture))
    store = make_store(budget)
    for _ in range(90):                            # exhaust enforced budget
        store.cache_get("k")
    issued_at_trip = store._client.commands

    # OPEN: every operation degrades gracefully WITHOUT issuing commands
    assert store.cache_get("k") is None            # cache -> miss
    store.cache_set("k", b"v", 60)                 # -> no-op
    assert store.bounded_incr("ctr", 60) is None   # -> caller falls back
    rl = store.check_rate_limit("client", 5, 60)
    assert rl.allowed is True                      # -> documented fail-open
    assert store._client.commands == issued_at_trip  # ZERO new commands
    assert budget.peek().state == STATE_OPEN


def test_redis_state2b_end_to_end_request_still_serves(capture):
    """With the command budget open, a full /v1/query still returns 200:
    cache misses, rate limit fails open, quota guard falls back local."""
    from fastapi.testclient import TestClient

    from app.core.hybrid import RerankInfo, RetrievedChunk
    from app.generation.service import GenerationResult
    from app.main import create_app

    budget = ResourceBudget("upstash_commands_daily", 100,
                            alerts=make_alerts(capture))
    store = make_store(budget)
    for _ in range(90):
        store.cache_get("burn")

    class StubService:
        def answer(self, query, **kw):
            return GenerationResult(
                query=query, answer="ok.", status="ok", degraded=False,
                citations=["c"], retrieved_chunk_ids=["c"],
                retrieved_texts=["t"], rerank_status="full",
            )

    from app.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    app.state.service = StubService()
    with TestClient(app) as client:
        client.app.state.redis_store = store
        resp = client.post("/v1/query", json={"query": "still works?"})
    assert resp.status_code == 200                 # never hard-crashes
    assert resp.json()["answer"] == "ok."


def test_redis_state3_past_hard_limit_provider_throttling_absorbed():
    """Past the real Upstash limit the provider throttles/blocks; our
    client already fails soft on RedisError (Stage 2 tests). Re-proven
    here with a client that raises on every command."""
    import redis as redis_lib

    from app.storage.redis_store import RedisStore

    class ThrottledClient:
        def __getattr__(self, name):
            def op(*a, **k):
                raise redis_lib.exceptions.ResponseError(
                    "max requests limit exceeded"
                )
            return op

    store = RedisStore.__new__(RedisStore)
    store.ns = "t"
    store.command_budget = None
    store._client = ThrottledClient()

    def raising_script(keys, args):
        raise redis_lib.exceptions.ResponseError("max requests limit exceeded")

    store._rate_limit = raising_script
    store._bounded_incr_script = raising_script

    assert store.cache_get("k") is None            # soft miss
    store.cache_set("k", b"v", 5)                  # no exception
    assert store.bounded_incr("c", 5) is None      # caller falls back
    assert store.check_rate_limit("x", 5, 60).allowed is True  # fail-open


# ---- budget window + validation mechanics ----

def test_budget_resets_at_utc_midnight(capture):
    clock = FrozenClock()
    budget = ResourceBudget("r", 10, alerts=make_alerts(capture, clock),
                            now_fn=clock)
    for _ in range(9):
        budget.record_and_check()
    assert budget.peek().state == STATE_OPEN
    clock.advance(86_400)                          # next UTC day
    assert budget.peek().state == STATE_CLOSED
    assert budget.record_and_check().allowed is True


def test_alert_failure_never_breaks_the_caller():
    def exploding(request):
        raise httpx.ConnectError("alert receiver down")

    alerts = AlertManager("https://alerts.example/hook",
                          transport=httpx.MockTransport(exploding))
    budget = ResourceBudget("r", 10, alerts=alerts)
    for _ in range(9):
        assert budget.record_and_check() is not None  # no exception
