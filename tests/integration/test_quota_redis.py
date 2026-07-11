"""Quota accounting over real Redis: the boundary must hold ACROSS
worker processes, which is exactly what the in-process fallback cannot
guarantee. Two QuotaGuard instances = two simulated workers."""

from __future__ import annotations

import time

from app.generation.quota import REASON_RPM, QuotaGuard, load_model_limits

MODEL = "gemini-2.5-flash-lite"


def test_rpm_boundary_shared_across_workers(redis_store):
    limits = load_model_limits(MODEL)
    clock = lambda: 1_800_000_000.0  # frozen: all requests in one minute
    worker_a = QuotaGuard(limits, redis_store=redis_store, now_fn=clock)
    worker_b = QuotaGuard(limits, redis_store=redis_store, now_fn=clock)

    allowed = 0
    for i in range(worker_a.enforced_rpm + 3):
        guard = worker_a if i % 2 == 0 else worker_b   # alternate workers
        if guard.try_acquire().allowed:
            allowed += 1
    # the two workers together got exactly the enforced budget, not 2x it
    assert allowed == worker_a.enforced_rpm == 13

    denied = worker_b.try_acquire()
    assert denied.allowed is False
    assert denied.reason == REASON_RPM


def test_counter_keys_expire(redis_store):
    limits = load_model_limits(MODEL)
    guard = QuotaGuard(limits, redis_store=redis_store,
                       now_fn=lambda: time.time())
    guard.try_acquire()
    minute = int(time.time() // 60)
    key = f"ragp_test:ctr:llmq:{MODEL}:rpm:{minute}"
    ttl = redis_store._client.ttl(key)
    assert 0 < ttl <= 120


def test_redis_down_falls_back_to_local_counting():
    from app.storage.redis_store import RedisStore

    limits = load_model_limits(MODEL)
    dead = RedisStore("redis://127.0.0.1:59999/0", namespace="ragp_test")
    guard = QuotaGuard(limits, redis_store=dead,
                       now_fn=lambda: 1_800_000_000.0)
    # still enforces (locally) instead of failing open entirely
    for _ in range(guard.enforced_rpm):
        assert guard.try_acquire().allowed
    assert guard.try_acquire().reason == REASON_RPM
