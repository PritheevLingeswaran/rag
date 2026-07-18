"""Stage 4.5 definition-of-done: drive the system to the EXACT quota
boundaries sourced in Stage 2.5 (configs/free_tier_limits.json) and show
the degraded path activating instead of a hard failure.

Numbers under test come from the limits file itself, not re-typed here:
gemini-2.5-flash-lite = 15 RPM / 1,000 RPD (Stage 2.5, secondary-sourced),
enforced at the 0.9 safety margin => 13 RPM / 900 RPD.
"""

import pytest

from app.core.hybrid import RerankInfo, RetrievedChunk
from app.errors import ConfigurationError, LLMQuotaError
from app.generation.llm_client import LLMResponse
from app.generation.quota import (
    REASON_COOLDOWN,
    REASON_OK,
    REASON_RPD,
    REASON_RPM,
    QuotaGuard,
    load_model_limits,
    seconds_to_pacific_midnight,
)
from app.generation.service import GenerationService

MODEL = "gemini-2.5-flash-lite"
LIMITS = load_model_limits(MODEL)


class FakeClock:
    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def make_guard(clock=None, margin=0.9) -> QuotaGuard:
    return QuotaGuard(LIMITS, redis_store=None, safety_margin=margin,
                      now_fn=clock or FakeClock())


# ---- limits are read from the Stage 2.5 file, not invented ----

def test_limits_loaded_from_stage25_file():
    assert LIMITS.rpm == 15
    assert LIMITS.rpd == 1000
    primary = load_model_limits("gemini-3.1-flash-lite")
    assert primary.rpm == 15
    assert primary.rpd == 500


def test_groq_fallback_limits_loaded_from_stage25_file():
    groq = load_model_limits("llama-3.1-8b-instant")
    assert groq.rpm == 30
    assert groq.rpd == 14400


def test_unknown_model_refuses_to_guess():
    with pytest.raises(ConfigurationError, match="no free-tier limits"):
        load_model_limits("gemini-99-ultra")


def test_enforced_budget_is_90_percent_of_provider():
    guard = make_guard()
    assert guard.enforced_rpm == 13   # floor(15 * 0.9)
    assert guard.enforced_rpd == 900  # floor(1000 * 0.9)


# ---- RPM boundary ----

def test_rpm_boundary_allows_exactly_enforced_then_denies():
    clock = FakeClock()
    guard = make_guard(clock)
    decisions = [guard.try_acquire() for _ in range(guard.enforced_rpm)]
    assert all(d.allowed for d in decisions)
    assert decisions[-1].remaining_rpm == 0

    denied = guard.try_acquire()  # request #14 within the same minute
    assert denied.allowed is False
    assert denied.reason == REASON_RPM
    assert 0 < denied.retry_after_s <= 60


def test_rpm_window_resets_next_minute():
    clock = FakeClock()
    guard = make_guard(clock)
    for _ in range(guard.enforced_rpm + 1):
        guard.try_acquire()
    clock.advance(60)
    assert guard.try_acquire().allowed is True


def test_rpm_denied_requests_do_not_burn_daily_budget():
    """Regression (Bug 1): counting RPD before RPM let every RPM-denied
    request consume a daily slot -- sustained over-RPM traffic drained
    the whole day's budget in minutes. RPM-denied must leave RPD
    untouched."""
    clock = FakeClock()
    guard = make_guard(clock)
    # exhaust the minute, then hammer well past it
    for _ in range(guard.enforced_rpm + 50):
        guard.try_acquire()
    # next minute: only enforced_rpm requests actually spent RPD, so
    # nearly the whole day remains
    clock.advance(60)
    d = guard.try_acquire()
    assert d.allowed is True
    assert d.remaining_rpd == guard.enforced_rpd - guard.enforced_rpm - 1


def test_local_counter_gc_never_evicts_daily_key():
    """Regression (Bug 5): keep-last-N-sorted GC evicted the live :rpd:
    key (sorts before :rpm:), resetting the day count mid-day."""
    from app.generation.quota import _LocalCounters

    counters = _LocalCounters()
    day_key = "llmq:m:rpd:20260716"
    for i in range(500):
        counters.incr(day_key)
    # flood with >1000 distinct per-minute keys to trigger GC
    for minute in range(1100):
        counters.incr(f"llmq:m:rpm:{29000000 + minute}")
    assert counters.incr(day_key) == 501  # survived GC, not reset to 1


# ---- RPD boundary ----

def test_rpd_boundary_denies_with_retry_until_pacific_midnight():
    clock = FakeClock()
    guard = make_guard(clock)
    allowed = 0
    for i in range(guard.enforced_rpd + 5):
        # spread over minutes so RPM never interferes with the RPD test
        if i % guard.enforced_rpm == 0:
            clock.advance(60)
        d = guard.try_acquire()
        if d.allowed:
            allowed += 1
        else:
            assert d.reason == REASON_RPD
    assert allowed == guard.enforced_rpd  # exactly 900, never 901

    denied = guard.try_acquire()
    assert denied.reason == REASON_RPD
    assert denied.retry_after_s == pytest.approx(
        seconds_to_pacific_midnight(clock()), abs=2
    )


def test_rpd_resets_after_pacific_midnight():
    clock = FakeClock()
    guard = make_guard(clock)
    for i in range(guard.enforced_rpd + 1):
        if i % guard.enforced_rpm == 0:
            clock.advance(60)
        guard.try_acquire()
    assert guard.try_acquire().reason == REASON_RPD
    clock.advance(seconds_to_pacific_midnight(clock()) + 61)
    assert guard.try_acquire().allowed is True


# ---- reactive sync: provider 429 opens a proactive cooldown ----

def test_provider_rejection_opens_cooldown_then_clears():
    clock = FakeClock()
    guard = make_guard(clock)
    assert guard.try_acquire().allowed
    guard.record_provider_rejection(retry_after_s=30.0)
    denied = guard.try_acquire()
    assert denied.reason == REASON_COOLDOWN
    assert denied.retry_after_s == pytest.approx(30.0, abs=0.5)
    clock.advance(31)
    assert guard.try_acquire().allowed is True


# ---- end-to-end: the app degrades instead of hard-failing ----

CHUNK = RetrievedChunk(
    "doc::c0",
    "The token bucket algorithm refills tokens at a fixed rate. "
    "Requests consume tokens and are rejected when the bucket is empty.",
    1.0, "rerank",
)


class StubPipeline:
    def retrieve(self, query):
        return [CHUNK], RerankInfo("full", 1, 1, 1.0)


class CountingLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return LLMResponse(
            "The token bucket refills tokens at a fixed rate [1].",
            "fake", 10, 10,
        )


def test_system_at_rpm_boundary_serves_degraded_not_500():
    """DEFINITION OF DONE: drive the service to the exact enforced RPM
    boundary (13 = floor(15 * 0.9) for gemini-2.5-flash-lite per the
    Stage 2.5 limits file). Request 14 must get a clearly-labeled
    retrieval-only degraded response; the LLM must never see it; nothing
    raises."""
    clock = FakeClock()
    guard = make_guard(clock)
    llm = CountingLLM()
    service = GenerationService(StubPipeline(), llm, quota_guard=guard)

    for i in range(guard.enforced_rpm):
        result = service.answer(f"token bucket refill question {i}")
        assert result.status == "ok"
        assert result.degraded is False
    assert llm.calls == guard.enforced_rpm  # 13 real generations

    over = service.answer("token bucket question 14, over budget")
    assert over.status == "degraded_quota_throttled"   # labeled degraded
    assert over.degraded is True
    assert over.extra["throttle_reason"] == REASON_RPM
    assert 0 < over.retry_after_s <= 60                # when to come back
    assert llm.calls == guard.enforced_rpm             # LLM NOT called
    # retrieval-only answer with citations still served
    assert over.answer.startswith("The token bucket algorithm")
    assert over.citations == ["doc::c0"]

    clock.advance(60)                                  # next minute
    assert service.answer("token bucket recovered").status == "ok"


def test_provider_429_and_proactive_throttle_are_distinct_statuses():
    """Operators must be able to tell 'we throttled ourselves' (expected)
    from 'the provider rejected us' (investigate)."""
    clock = FakeClock()
    guard = make_guard(clock)

    class QuotaErrorLLM:
        calls = 0

        def generate(self, prompt):
            self.calls += 1
            raise LLMQuotaError("429", retry_after_s=20.0)

    service = GenerationService(StubPipeline(), QuotaErrorLLM(),
                                quota_guard=guard)
    provider_rejected = service.answer("token bucket rate question one")
    assert provider_rejected.status == "degraded_quota"       # reactive
    assert provider_rejected.retry_after_s == 20.0

    # the 429 opened a cooldown: next request is throttled proactively,
    # with the distinct status and without touching the provider
    throttled = service.answer("token bucket rate question two")
    assert throttled.status == "degraded_quota_throttled"     # proactive
    assert throttled.extra["throttle_reason"] == REASON_COOLDOWN
