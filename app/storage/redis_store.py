"""Redis wrapper: response cache + distributed rate-limit state.

Rate limiting uses a fixed-window counter implemented as a Lua script so
check-and-increment is atomic -- a read-then-write implementation would
let two API workers both pass the check and exceed the limit. Fixed window
(vs. sliding log) is deliberate: one key + one INCR per request fits
Upstash's free-tier command budget, at the cost of allowing up to 2x burst
at window boundaries. That burst is bounded and acceptable at our scale;
revisit if abuse shows up in query logs.

Cache failures are soft: a Redis outage degrades latency (cache misses),
never availability. Errors are logged with full context, then the caller
proceeds without cache. Rate-limit failures are also soft (fail-open):
on Redis outage we serve rather than 429 everyone; the tradeoff is that an
attacker who can take down Redis removes rate limits. Fail-closed would
turn every Redis blip into a full API outage -- worse on free-tier infra
where blips are routine. This is a real tradeoff; flagged for review.
"""

from __future__ import annotations

from dataclasses import dataclass

import redis

from app.logging_config import get_logger

logger = get_logger(__name__)

# KEYS[1] = counter key, ARGV[1] = limit, ARGV[2] = window seconds.
# Returns {allowed, remaining, ttl}.
_RATE_LIMIT_LUA = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[2])
end
local ttl = redis.call('TTL', KEYS[1])
local limit = tonumber(ARGV[1])
if current > limit then
    return {0, 0, ttl}
end
return {1, limit - current, ttl}
"""


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after_s: int


class RedisStore:
    def __init__(self, redis_url: str, namespace: str = "ragp",
                 command_budget=None) -> None:
        """command_budget: optional app.reliability.ResourceBudget
        metering the Upstash daily command allowance (Stage 7.7). When
        the budget opens, every operation BYPASSES Redis instead of
        spending commands we no longer have: cache degrades to misses,
        rate limiting fails open, counters return None (callers fall
        back to local accounting) -- all logged, never raised."""
        self.ns = namespace
        self.command_budget = command_budget
        self._client = redis.Redis.from_url(
            redis_url, decode_responses=False, socket_timeout=2.0,
            socket_connect_timeout=2.0,
        )
        # All scripts registered here, not lazily: a lazy hasattr-guarded
        # registration was a (benign) check-then-act under threads
        # (Stage 6.5 audit). Registration is local hashing, no network.
        self._rate_limit = self._client.register_script(_RATE_LIMIT_LUA)
        self._bounded_incr_script = self._client.register_script(
            self._BOUNDED_INCR_LUA
        )

    def _spend(self, commands: int = 1) -> bool:
        """Meter the command budget; False => bypass Redis entirely."""
        if self.command_budget is None:
            return True
        decision = self.command_budget.record_and_check(commands)
        if not decision.allowed:
            logger.warning(
                "redis_command_budget_open_bypassing",
                used=decision.used, enforced=decision.enforced,
            )
        return decision.allowed

    def ping(self) -> bool:
        if not self._spend():
            return False
        try:
            return bool(self._client.ping())
        except redis.RedisError as exc:
            logger.warning("redis_ping_failed", error=str(exc))
            return False

    # ---- cache ----

    def cache_get(self, key: str) -> bytes | None:
        if not self._spend():
            return None  # budget open: cache degrades to misses
        try:
            return self._client.get(f"{self.ns}:cache:{key}")
        except redis.RedisError as exc:
            logger.warning("cache_get_failed", key=key, error=str(exc))
            return None

    def cache_set(self, key: str, value: bytes, ttl_s: int) -> None:
        if not self._spend():
            return
        try:
            self._client.set(f"{self.ns}:cache:{key}", value, ex=ttl_s)
        except redis.RedisError as exc:
            logger.warning("cache_set_failed", key=key, error=str(exc))

    def cache_delete(self, key: str) -> None:
        if not self._spend():
            return
        try:
            self._client.delete(f"{self.ns}:cache:{key}")
        except redis.RedisError as exc:
            logger.warning("cache_delete_failed", key=key, error=str(exc))

    # ---- bounded counters (LLM quota accounting) ----

    _BOUNDED_INCR_LUA = """
    local current = redis.call('INCR', KEYS[1])
    if current == 1 then
        redis.call('EXPIRE', KEYS[1], ARGV[1])
    end
    return current
    """

    def bounded_incr(self, key: str, ttl_s: int) -> int | None:
        """Atomically increment a counter that expires ttl_s after its
        first increment. Returns the post-increment value, or None when
        Redis is unreachable OR the command budget is open (caller
        decides the fallback policy)."""
        if not self._spend():
            return None
        try:
            value = self._bounded_incr_script(
                keys=[f"{self.ns}:ctr:{key}"], args=[ttl_s]
            )
            return int(value)
        except redis.RedisError as exc:
            logger.warning("bounded_incr_failed", key=key, error=str(exc))
            return None

    # ---- rate limiting ----

    def check_rate_limit(self, client_id: str, limit: int,
                         window_s: int) -> RateLimitDecision:
        """Atomic fixed-window rate limit check. Fail-open on Redis
        errors AND when the command budget is open (documented Stage 2
        tradeoff: a budget/outage blip must not 429 everyone)."""
        if not self._spend():
            return RateLimitDecision(allowed=True, remaining=0, retry_after_s=0)
        key = f"{self.ns}:rl:{client_id}:{window_s}"
        try:
            allowed, remaining, ttl = self._rate_limit(
                keys=[key], args=[limit, window_s]
            )
            return RateLimitDecision(
                allowed=bool(allowed),
                remaining=int(remaining),
                retry_after_s=max(0, int(ttl)),
            )
        except redis.RedisError as exc:
            logger.error(
                "rate_limit_check_failed_failing_open",
                client_id=client_id, error=str(exc),
            )
            return RateLimitDecision(allowed=True, remaining=0, retry_after_s=0)
