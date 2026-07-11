"""Stage 6.5 concurrency audit -- local shared-state items (no external
services, no models). Each test hammers one enumerated piece of shared
mutable state from many threads/tasks and asserts EXACT accounting, not
absence of crashes.

sys.setswitchinterval(1e-6) forces aggressive GIL handoffs; the harness
was validated to catch >88% lost updates when a race window exists
(see docs/stage6_5_concurrency.md)."""

from __future__ import annotations

import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pytest

sys.setswitchinterval(1e-6)

THREADS = 16


# ---- 1. HybridPipeline rerank-cost EWMA (the audit's fix) ----

def make_tiny_pipeline(sleep_s: float = 0.0):
    from app.core.bm25 import BM25Index
    from app.core.dense import DenseIndex
    from app.core.hybrid import HybridPipeline

    texts = {f"d{i}::c0": f"passage number {i} about topic {i}"
             for i in range(4)}
    bm25 = BM25Index()
    bm25.build(list(texts.items()))
    dense = DenseIndex.from_vectors(
        np.eye(4, dtype=np.float32), list(texts.keys())
    )

    class Embedder:
        def embed_batch(self, batch):
            out = np.zeros((len(batch), 4), dtype=np.float32)
            out[:, 0] = 1.0
            return out

    class Reranker:
        micro_batch = 2

        def score(self, query, passages):
            if sleep_s:
                import time

                time.sleep(sleep_s)
            return np.arange(len(passages), dtype=np.float32)

    return HybridPipeline(bm25, dense, Embedder(), Reranker(), texts,
                          rerank_depth=4, final_top_k=4)


def test_ewma_update_accounting_is_exact_under_contention():
    p = make_tiny_pipeline()
    per_thread = 250
    values = [float(i + 1) for i in range(THREADS)]
    barrier = threading.Barrier(THREADS)

    def worker(v):
        barrier.wait()
        for _ in range(per_thread):
            p._update_rerank_cost(v)

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        list(pool.map(worker, values))

    assert p.rerank_cost_updates == THREADS * per_thread  # exactly 4000
    assert min(values) <= p._read_rerank_cost() <= max(values)


def test_full_rerank_path_update_accounting_under_concurrent_runs():
    p = make_tiny_pipeline()
    runs_per_thread = 20
    # 4 candidates, micro_batch 2 => exactly 2 EWMA updates per run
    barrier = threading.Barrier(8)

    def worker(_):
        barrier.wait()
        for _ in range(runs_per_thread):
            result = p.run("passage topic")
            assert result.rerank.status == "full"

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))
    assert p.rerank_cost_updates == 8 * runs_per_thread * 2


# ---- 2. QuotaGuard local fallback backend ----

def test_quota_local_backend_exact_at_boundary_under_threads():
    from app.generation.quota import QuotaGuard, load_model_limits

    guard = QuotaGuard(load_model_limits("gemini-2.5-flash-lite"),
                       redis_store=None,
                       now_fn=lambda: 1_800_000_000.0)  # frozen minute
    allowed = []
    barrier = threading.Barrier(8)

    def worker(_):
        barrier.wait()
        got = 0
        for _ in range(10):
            if guard.try_acquire().allowed:
                got += 1
        return got

    with ThreadPoolExecutor(max_workers=8) as pool:
        allowed = list(pool.map(worker, range(8)))
    # 80 contended attempts, enforced budget 13: EXACTLY 13 succeed
    assert sum(allowed) == guard.enforced_rpm == 13


# ---- 3. AdmissionController (event-loop state) ----

def test_admission_accounting_conserved_under_stress():
    from app.api.admission import AdmissionController, QueueFullError

    async def scenario():
        ac = AdmissionController(max_concurrency=4, max_queue_depth=8)
        peak = 0
        current = 0
        outcomes = {"admitted": 0, "rejected": 0}

        async def job(i):
            nonlocal peak, current
            try:
                async with ac.admit():
                    current += 1
                    peak = max(peak, current)
                    await asyncio.sleep(0.001 * (i % 3))
                    current -= 1
                    outcomes["admitted"] += 1
            except QueueFullError:
                outcomes["rejected"] += 1

        await asyncio.gather(*(job(i) for i in range(300)))
        return ac, peak, outcomes

    ac, peak, outcomes = asyncio.new_event_loop().run_until_complete(
        scenario()
    )
    snap = ac.snapshot()
    assert outcomes["admitted"] + outcomes["rejected"] == 300
    assert snap["admitted_total"] == outcomes["admitted"]
    assert snap["rejected_total"] == outcomes["rejected"]
    assert snap["in_flight"] == 0 and snap["waiting"] == 0  # fully drained
    assert peak <= 4                                        # bound held


# ---- 4. Prometheus counters ----

def test_prometheus_counter_exact_under_threads():
    from prometheus_client import Counter

    c = Counter("ragp_test_concurrency_probe_total", "audit probe")
    barrier = threading.Barrier(8)

    def worker(_):
        barrier.wait()
        for _ in range(1000):
            c.inc()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))
    assert c._value.get() == 8000.0


# ---- 5. GeminiClient shared httpx.Client ----

def test_gemini_client_thread_safe_over_shared_httpx_client():
    from app.generation.llm_client import GeminiClient

    def handler(req):
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "ok [1]."}]},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 1,
                              "candidatesTokenCount": 1},
        })

    client = GeminiClient(api_key="k",
                          transport=httpx.MockTransport(handler))
    barrier = threading.Barrier(THREADS)

    def worker(_):
        barrier.wait()
        return [client.generate("p").text for _ in range(25)]

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        results = list(pool.map(worker, range(THREADS)))
    assert all(t == "ok [1]." for batch in results for t in batch)


# ---- 6. get_settings lru_cache ----

def test_settings_singleton_under_threads(monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        ids = set(pool.map(lambda _: id(get_settings()), range(64)))
    assert len(ids) == 1
