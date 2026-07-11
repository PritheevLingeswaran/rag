"""Stage 7.5: full-stack regression load test.

Re-runs the EXACT Stage 3 load pattern -- 50k english-style chunks,
seed-42 6-word-prefix queries, 200 requests per level at concurrency
1/4/8 -- but through the entire deployed-locally stack over real HTTP:

    auth (x-api-key) -> per-key rate limit + daily quota (real Redis)
    -> response cache (flushed between levels so every request is a
    MISS: worst-case numbers, cache effect reported separately)
    -> admission control (production bounds) -> hybrid retrieval
    (depth-10 rerank under the 700ms budget) -> generation ->
    citation validation -> response.

LLM honesty: no API key exists, so generation uses a zero-latency
scripted LLM that returns a grounded, cited answer built from the
prompt's [1] source -- the full generation + citation-validation CODE
PATH runs on every request, but provider network latency (~0.5-2s for
Gemini Flash per provider docs) is NOT in these numbers and is stated
in the report. QuotaGuard runs with effectively-unlimited test limits
so the quota code path executes without throttling the measurement.

Layer attribution: /metrics histogram sums are diffed per level, giving
measured per-stage time (embed/bm25/dense/rerank) vs total HTTP time.

Usage:
    ENVIRONMENT=development REDIS_URL=redis://127.0.0.1:56379/0 \
        python scripts/load_test_fullstack.py --chunks 50000 \
        --requests 200 --concurrency 1,4,8 --json out.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402
import uvicorn  # noqa: E402

API_KEY = "loadtest-key-0001"
PORT = 8123


class GroundedScriptedLLM:
    """Zero-latency stand-in returning a grounded cited answer."""

    def generate(self, prompt, **kwargs):
        from app.generation.llm_client import LLMResponse

        m = re.search(r"\[1\] (.+)", prompt)
        grounded = m.group(1).split(". ")[0] if m else "The source says this"
        return LLMResponse(f"{grounded} [1].", "scripted-grounded", 100, 20)


def build_app_and_queries(n_chunks: int):
    from app.core.hybrid import HybridPipeline
    from app.generation.quota import ModelLimits, QuotaGuard
    from app.generation.service import GenerationService
    from app.main import create_app
    from scripts.load_test_retrieval import build_components

    (bm25, dense, embedder, reranker, texts), queries = build_components(
        n_chunks, "english"
    )
    from app.config import get_settings

    settings = get_settings()
    pipeline = HybridPipeline(
        bm25, dense, embedder, reranker, texts,
        rerank_depth=settings.rerank_depth,
        rerank_budget_ms=settings.rerank_budget_ms,
    )
    guard = QuotaGuard(
        ModelLimits("loadtest-unlimited", rpm=10**6, rpd=10**9)
    )
    service = GenerationService(pipeline, GroundedScriptedLLM(),
                                quota_guard=guard)
    app = create_app()
    app.state.service = service
    return app, queries


def percentile(values, pct):
    ordered = sorted(values)
    rank = max(1, int(round(pct / 100.0 * len(ordered))))
    return ordered[rank - 1]


def scrape_stage_sums(metrics_text: str) -> dict[str, float]:
    sums = {}
    for line in metrics_text.splitlines():
        m = re.match(
            r'ragp_retrieval_duration_seconds_sum\{stage="(\w+)"\} ([\d.e+-]+)',
            line,
        )
        if m:
            sums[f"retrieval_{m.group(1)}"] = float(m.group(2))
        m = re.match(
            r'ragp_http_request_duration_seconds_sum\{method="POST",'
            r'path="/v1/query",status="200"\} ([\d.e+-]+)', line,
        )
        if m:
            sums["http_total"] = float(m.group(1))
    return sums


async def run_level(url, queries, n_requests, concurrency):
    results = []
    counter = {"i": 0}
    lock = asyncio.Lock()

    async def worker(client):
        while True:
            async with lock:
                i = counter["i"]
                if i >= n_requests:
                    return
                counter["i"] += 1
            q = queries[i % len(queries)]
            t0 = time.perf_counter()
            resp = await client.post(f"{url}/v1/query", json={"query": q})
            elapsed = (time.perf_counter() - t0) * 1000.0
            body = resp.json() if resp.status_code == 200 else {}
            results.append((resp.status_code, elapsed,
                            body.get("status"), body.get("rerank_status"),
                            body.get("cached")))
            if resp.status_code not in (200, 503):
                raise RuntimeError(f"{resp.status_code}: {resp.text[:200]}")

    async with httpx.AsyncClient(
        timeout=120.0, headers={"x-api-key": API_KEY}
    ) as client:
        wall0 = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
        wall = time.perf_counter() - wall0

    oks = [ms for code, ms, *_ in results if code == 200]
    shed = sum(1 for code, *_ in results if code == 503)
    statuses = {}
    rerank_statuses = {}
    cached_count = 0
    for code, _, status, rst, cached in results:
        if code == 200:
            statuses[status] = statuses.get(status, 0) + 1
            rerank_statuses[rst] = rerank_statuses.get(rst, 0) + 1
            cached_count += 1 if cached else 0
    return {
        "concurrency": concurrency,
        "sent": len(results),
        "ok": len(oks),
        "shed_503": shed,
        "p50_ms": round(percentile(oks, 50), 1),
        "p95_ms": round(percentile(oks, 95), 1),
        "p99_ms": round(percentile(oks, 99), 1),
        "mean_ms": round(statistics.mean(oks), 1),
        "throughput_ok_rps": round(len(oks) / wall, 2),
        "statuses": statuses,
        "rerank_statuses": rerank_statuses,
        "cache_hits": cached_count,
    }


async def main_async(args):
    levels = [int(x) for x in args.concurrency.split(",")]
    url = f"http://127.0.0.1:{PORT}"

    import redis as redis_lib

    redis_client = redis_lib.Redis.from_url(os.environ["REDIS_URL"])

    rows = []
    async with httpx.AsyncClient(timeout=30.0,
                                 headers={"x-api-key": API_KEY}) as probe:
        for _ in range(600):
            try:
                if (await probe.get(f"{url}/health")).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
        # warmup (seeds rerank EWMA + model sessions)
        await probe.post(f"{url}/v1/query", json={"query": "warmup"})

        header = (f"{'conc':>4} {'sent':>5} {'ok':>4} {'503':>4} {'p50':>9} "
                  f"{'p95':>9} {'p99':>9} {'rps':>6}  statuses")
        print(header)
        print("-" * len(header))
        for level in levels:
            redis_client.flushdb()  # every request in a level is a cache MISS
            before = scrape_stage_sums((await probe.get(f"{url}/metrics")).text)
            row = await run_level(url, QUERIES, args.requests, level)
            after = scrape_stage_sums((await probe.get(f"{url}/metrics")).text)
            row["stage_seconds"] = {
                k: round(after.get(k, 0) - before.get(k, 0), 2)
                for k in after
            }
            rows.append(row)
            print(f"{row['concurrency']:>4} {row['sent']:>5} {row['ok']:>4} "
                  f"{row['shed_503']:>4} {row['p50_ms']:>8.1f}m "
                  f"{row['p95_ms']:>8.1f}m {row['p99_ms']:>8.1f}m "
                  f"{row['throughput_ok_rps']:>6.2f}  "
                  f"{row['statuses']} rerank={row['rerank_statuses']} "
                  f"cache_hits={row['cache_hits']}")
            print(f"     stage sums (s): {row['stage_seconds']}")
    return rows


QUERIES: list[str] = []


def main() -> int:
    global QUERIES
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=50_000)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    os.environ.setdefault("API_KEYS", API_KEY)
    os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "1000000")
    os.environ.setdefault("DAILY_QUOTA_PER_KEY", "1000000000")
    os.environ.setdefault("SERVE_PIPELINE", "false")
    from app.config import get_settings

    get_settings.cache_clear()

    print(f"building full stack over {args.chunks} chunks ...")
    app, QUERIES = build_app_and_queries(args.chunks)

    config = uvicorn.Config(app, host="127.0.0.1", port=PORT,
                            log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    rows = asyncio.run(main_async(args))

    server.should_exit = True
    thread.join(timeout=10)

    if args.json:
        args.json.write_text(json.dumps({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "chunks": args.chunks,
            "pipeline_config": "production defaults (rerank_depth=10, budget=700ms, admission 2+4)",
            "llm": "zero-latency scripted grounded (provider latency excluded, stated)",
            "levels": rows,
        }, indent=2), encoding="utf-8")
        print(f"json written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
