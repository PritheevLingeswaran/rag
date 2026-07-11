"""Stage 6.5 concurrency audit -- heavy shared state at TARGET SCALE:
FAISS (50k vectors), BM25 (50k chunks), real ONNX model sessions, the
ingestion-vs-serving isolation mechanism, and real-Redis atomicity.

The correctness bar is determinism-equality: concurrent results must be
byte-identical to serial results, not merely crash-free."""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

sys.setswitchinterval(1e-6)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

N_CHUNKS = 50_000
DIM = 384


@pytest.fixture(scope="module")
def components():
    from scripts.load_test_retrieval import build_components

    comps, queries = build_components(N_CHUNKS, "english")
    return comps, queries[:40]


# ---- FAISS index: concurrent reads at 50k ----

def test_faiss_concurrent_search_identical_to_serial(components):
    (bm25, dense, embedder, reranker, texts), queries = components
    qvecs = [embedder.embed_batch([q])[0] for q in queries]
    serial = [dense.search(v, 20) for v in qvecs]

    def worker(_):
        return [dense.search(v, 20) for v in qvecs]

    with ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(worker, range(8)):
            assert result == serial


# ---- FAISS: query-during-ingestion (the actual isolation mechanism) ----

def test_queries_stable_while_ingestion_writes_new_versions(
    components, tmp_path
):
    """The mechanism is versioned immutable directories + atomic rename +
    explicit activation (Stage 2), NOT a lock: a serving index loaded
    from version A shares nothing writable with an ingestion job staging
    version B. This test hammers searches on A while three new versions
    are written and GC'd next to it, asserting bit-identical results
    throughout -- and that A's files remain intact afterward."""
    import faiss

    from app.core.dense import DenseIndex
    from app.ingest.faiss_store import FaissStore

    (_, dense, embedder, _, _), queries = components
    store = FaissStore(tmp_path / "indexes")

    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((N_CHUNKS, DIM), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"s{i}" for i in range(N_CHUNKS)]
    manifest_a = store.write_version("vA", "audit-embedder", vectors, "sha-a")
    serving = DenseIndex.from_files(
        store.version_dir("vA") / "index.faiss", ids
    )

    qvecs = [embedder.embed_batch([q])[0] for q in queries[:10]]
    serial = [serving.search(v, 10) for v in qvecs]

    stop = threading.Event()
    errors: list[Exception] = []

    def hammer():
        try:
            while not stop.is_set():
                for v, expected in zip(qvecs, serial):
                    assert serving.search(v, 10) == expected
        except Exception as exc:  # noqa: BLE001 - collected for the test
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        for i in range(3):  # ingestion writes new versions alongside
            fresh = rng.standard_normal((N_CHUNKS, DIM), dtype=np.float32)
            fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
            store.write_version(f"vB{i}", "audit-embedder", fresh, f"sha-b{i}")
            store.gc(keep_version_ids={"vA", f"vB{i}"})
    finally:
        stop.set()
        for t in threads:
            t.join()

    assert errors == []
    # version A untouched: hash-verified reload still succeeds
    reloaded = store.load_index("vA", expected_sha256=manifest_a.faiss_sha256)
    assert reloaded.ntotal == N_CHUNKS


# ---- BM25: concurrent reads at 50k ----

def test_bm25_concurrent_search_identical_to_serial(components):
    (bm25, _, _, _, _), queries = components
    serial = [bm25.search(q, 30) for q in queries]

    def worker(_):
        return [bm25.search(q, 30) for q in queries]

    with ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(worker, range(8)):
            assert result == serial


# ---- ONNX sessions: concurrent inference determinism ----

def test_embedder_concurrent_identical_to_serial(components):
    (_, _, embedder, _, _), queries = components
    serial = embedder.embed_batch(queries[:16])

    def worker(_):
        return embedder.embed_batch(queries[:16])

    with ThreadPoolExecutor(max_workers=8) as pool:
        for out in pool.map(worker, range(8)):
            assert np.array_equal(out, serial)


def test_reranker_concurrent_identical_to_serial(components):
    (_, _, _, reranker, texts), queries = components
    passages = list(texts.values())[:10]
    serial = reranker.score(queries[0], passages)

    def worker(_):
        return reranker.score(queries[0], passages)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for out in pool.map(worker, range(8)):
            assert np.array_equal(out, serial)


# ---- full pipeline hammer (all shared state at once) ----

def test_full_pipeline_concurrent_hammer(components):
    from app.core.hybrid import HybridPipeline

    (bm25, dense, embedder, reranker, texts), queries = components
    pipeline = HybridPipeline(bm25, dense, embedder, reranker, texts,
                              rerank_depth=10, final_top_k=10,
                              rerank_budget_ms=None)
    serial = {q: pipeline.run(q).retrieved_chunk_ids for q in queries[:8]}

    def worker(q):
        for _ in range(5):
            result = pipeline.run(q)
            assert result.retrieved_chunk_ids == serial[q]
            assert result.rerank.status == "full"
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(worker, queries[:8]))
    assert pipeline.rerank_cost_updates == (8 + 8 * 5) * 2  # exact: 2/run


# ---- Redis: atomicity under real concurrent clients ----

def test_redis_bounded_incr_exact_under_threads(redis_store):
    def worker(_):
        for _ in range(25):
            redis_store.bounded_incr("audit:ctr", 120)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))
    assert redis_store.bounded_incr("audit:ctr", 120) == 201  # 200 + this


def test_redis_rate_limit_exactly_limit_allowed_under_threads(redis_store):
    def worker(_):
        allowed = 0
        for _ in range(5):
            if redis_store.check_rate_limit("audit-client", 20, 60).allowed:
                allowed += 1
        return allowed

    with ThreadPoolExecutor(max_workers=8) as pool:
        total = sum(pool.map(worker, range(8)))
    assert total == 20  # 40 contended attempts, EXACTLY the limit pass


def test_redis_cache_concurrent_writers_never_torn(redis_store):
    import json

    payloads = [
        json.dumps({"writer": i, "data": "x" * 500}).encode() for i in range(8)
    ]

    def writer(i):
        for _ in range(20):
            redis_store.cache_set("audit:dogpile", payloads[i], 60)
            raw = redis_store.cache_get("audit:dogpile")
            body = json.loads(raw)          # parse must never fail
            assert body["data"] == "x" * 500  # value intact, no tearing

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(writer, range(8)))
    final = json.loads(redis_store.cache_get("audit:dogpile"))
    assert final in [json.loads(p) for p in payloads]
