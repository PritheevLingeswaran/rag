"""Integration tests for Redis cache + rate limiting against real Redis."""

from __future__ import annotations

from app.storage.redis_store import RedisStore


def test_cache_roundtrip_and_delete(redis_store):
    assert redis_store.cache_get("k1") is None
    redis_store.cache_set("k1", b"payload", ttl_s=60)
    assert redis_store.cache_get("k1") == b"payload"
    redis_store.cache_delete("k1")
    assert redis_store.cache_get("k1") is None


def test_cache_entries_expire(redis_store):
    redis_store.cache_set("short", b"x", ttl_s=1)
    ttl = redis_store._client.ttl("ragp_test:cache:short")
    assert 0 < ttl <= 1


def test_rate_limit_allows_up_to_limit_then_blocks(redis_store):
    decisions = [
        redis_store.check_rate_limit("client-a", limit=5, window_s=60)
        for _ in range(7)
    ]
    assert [d.allowed for d in decisions] == [True] * 5 + [False] * 2
    assert decisions[0].remaining == 4
    assert decisions[4].remaining == 0
    assert decisions[5].retry_after_s > 0


def test_rate_limit_isolated_per_client(redis_store):
    for _ in range(5):
        redis_store.check_rate_limit("client-b", limit=5, window_s=60)
    assert redis_store.check_rate_limit("client-b", 5, 60).allowed is False
    assert redis_store.check_rate_limit("client-c", 5, 60).allowed is True


def test_rate_limit_fails_open_when_redis_down():
    down = RedisStore("redis://127.0.0.1:59999/0", namespace="ragp_test")
    decision = down.check_rate_limit("anyone", limit=1, window_s=60)
    assert decision.allowed is True  # documented fail-open policy


def test_cache_fails_soft_when_redis_down():
    down = RedisStore("redis://127.0.0.1:59999/0", namespace="ragp_test")
    down.cache_set("k", b"v", ttl_s=5)   # no exception
    assert down.cache_get("k") is None   # miss, not crash
