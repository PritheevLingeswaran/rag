"""Load test for the Stage 3 retrieval core at target scale.

Measures the retrieval path in isolation -- embed(query) -> FAISS search +
BM25 search -> RRF -> cross-encoder rerank -> result assembly -- with
concurrent workers hammering a single pipeline instance, exactly as a
FastAPI worker would drive it. Deliberately excludes HTTP/network overhead
(that gets measured when the API layer exists); includes everything else.

Corpus: synthetic 50k chunks (Zipfian text, seed 42) for BM25 + chunk
texts; random normalized vectors for the dense index (FAISS latency is
independent of vector values). Query embedding at request time uses the
REAL ONNX embedder; reranking scores REAL (query, passage) pairs, so the
expensive stages are fully real.

Queries: 6-word prefixes of corpus chunks (seed-42 sample), a lexically
biased but latency-realistic workload.

Report: per-concurrency p50/p95/p99 wall latency per request + throughput.
Latency methodology is fixed; absolute values are machine-dependent.

Usage (the exact command is recorded in the report):
    python scripts/load_test_retrieval.py --chunks 50000 \
        --requests 200 --concurrency 1,4,8 --json eval/results/loadtest.json
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.core.bm25 import BM25Index  # noqa: E402
from app.core.dense import DenseIndex  # noqa: E402
from app.core.hybrid import HybridPipeline  # noqa: E402
from app.core.bootstrap import load_onnx_models  # noqa: E402
from scripts.measure_memory import synth_corpus  # noqa: E402

DIM = 384


def english_corpus(n_chunks: int, seed: int = 42) -> list[tuple[str, str]]:
    """Synthetic chunks built from the REAL corpus vocabulary, so subword
    tokenization ratios (and therefore reranker cost) match production
    text. The 'worstcase' style (synth_corpus) uses wordNNNNN tokens that
    explode to 4-5 subtokens each and pin every passage at the 256-token
    truncation cap -- an upper bound, not a typical load."""
    words: list[str] = []
    corpus_file = REPO_ROOT / "data" / "corpus_v1.jsonl"
    for line in corpus_file.read_text(encoding="utf-8").splitlines():
        doc = json.loads(line)
        words.extend(w.lower() for w in doc["text"].split())
    vocab = sorted(set(words))
    rng = random.Random(seed)
    weights = [1.0 / (i + 1) for i in range(len(vocab))]
    rng.shuffle(vocab)  # decouple Zipf weight from alphabetical order
    docs = []
    for i in range(n_chunks):
        n_tokens = rng.randint(80, 120)
        tokens = rng.choices(vocab, weights=weights, k=n_tokens)
        docs.append((f"synth{i}::c0", " ".join(tokens)))
    return docs


def build_components(n_chunks: int, style: str):
    if style == "worstcase":
        corpus = synth_corpus(n_chunks)
    elif style == "english":
        corpus = english_corpus(n_chunks)
    else:
        raise ValueError(f"unknown corpus style {style!r}")
    texts = dict(corpus)
    bm25 = BM25Index()
    bm25.build(corpus)

    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n_chunks, DIM), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    dense = DenseIndex.from_vectors(vectors, [cid for cid, _ in corpus])
    del vectors

    embedder, reranker = load_onnx_models()

    q_rng = random.Random(42)
    sample = q_rng.sample(corpus, 500)
    queries = [" ".join(text.split()[:6]) for _, text in sample]
    return (bm25, dense, embedder, reranker, texts), queries


def make_pipeline(components, rerank_depth: int) -> HybridPipeline:
    bm25, dense, embedder, reranker, texts = components
    return HybridPipeline(bm25, dense, embedder, reranker, texts,
                          rerank_depth=rerank_depth,
                          final_top_k=min(10, rerank_depth) or 10)


def build_pipeline(n_chunks: int, style: str,
                   rerank_depth: int) -> tuple[HybridPipeline, list[str]]:
    components, queries = build_components(n_chunks, style)
    return make_pipeline(components, rerank_depth), queries


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = max(1, int(round(pct / 100.0 * len(ordered))))
    return ordered[rank - 1]


def run_level(pipeline: HybridPipeline, queries: list[str],
              n_requests: int, concurrency: int) -> dict:
    reqs = [queries[i % len(queries)] for i in range(n_requests)]
    latencies: list[float] = []

    def one(query: str) -> float:
        t0 = time.perf_counter()
        result = pipeline.run(query)
        elapsed = (time.perf_counter() - t0) * 1000.0
        assert result.answer  # sanity: full path executed
        return elapsed

    wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        latencies = list(pool.map(one, reqs))
    wall = time.perf_counter() - wall0

    return {
        "concurrency": concurrency,
        "requests": n_requests,
        "p50_ms": round(percentile(latencies, 50), 1),
        "p95_ms": round(percentile(latencies, 95), 1),
        "p99_ms": round(percentile(latencies, 99), 1),
        "mean_ms": round(statistics.mean(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "throughput_rps": round(n_requests / wall, 2),
        "wall_s": round(wall, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=50_000)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=str, default="1,4,8")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--style", choices=["worstcase", "english"],
                        default="english")
    parser.add_argument("--rerank-depth", type=str, default="20",
                        help="comma-separated depths to sweep; 0 = RRF only")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    levels = [int(x) for x in args.concurrency.split(",")]
    depths = [int(x) for x in args.rerank_depth.split(",")]

    print(f"building components over {args.chunks} chunks "
          f"(style={args.style}) ...")
    t0 = time.perf_counter()
    components, queries = build_components(args.chunks, args.style)
    print(f"build took {time.perf_counter() - t0:.1f}s")

    results = []
    header = (f"{'depth':>5} {'conc':>4} {'reqs':>5} {'p50':>8} {'p95':>8} "
              f"{'p99':>8} {'mean':>8} {'max':>8} {'rps':>7}")
    print(header)
    print("-" * len(header))
    pipeline = None
    for depth in depths:
        pipeline = make_pipeline(components, depth)
        for q in queries[:args.warmup]:
            pipeline.run(q)
        for level in levels:
            r = run_level(pipeline, queries, args.requests, level)
            r["rerank_depth"] = depth
            results.append(r)
            print(f"{depth:>5} {r['concurrency']:>4} {r['requests']:>5} "
                  f"{r['p50_ms']:>7.1f}ms {r['p95_ms']:>7.1f}ms "
                  f"{r['p99_ms']:>7.1f}ms {r['mean_ms']:>7.1f}ms "
                  f"{r['max_ms']:>7.1f}ms {r['throughput_rps']:>7.2f}")

    if args.json:
        args.json.write_text(json.dumps({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "chunks": args.chunks,
            "corpus_style": args.style,
            "rerank_depths": depths,
            "pipeline_config": {
                "bm25_top_n": pipeline.bm25_top_n,
                "dense_top_n": pipeline.dense_top_n,
                "final_top_k": pipeline.final_top_k,
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": __import__("os").cpu_count(),
            },
            "levels": results,
        }, indent=2), encoding="utf-8")
        print(f"json written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
