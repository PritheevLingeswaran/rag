"""HTTP load test against the running API, through admission control.

Purpose: determine the REAL concurrency ceiling empirically -- the
offered-concurrency level at which the bounded queue starts shedding
(503s) and what latency admitted requests actually see -- rather than
estimating it from component numbers.

Each worker sends requests back-to-back (closed-loop). 503s are counted
as shed load, not failures; anything else non-200 fails the run loudly.

Usage:
    python scripts/load_test_api.py --url http://127.0.0.1:8000 \
        --concurrency 1,2,4,8,16 --requests 40 \
        --json eval/results/api_loadtest.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

QUERIES = [
    "how does raft elect a leader",
    "what is the difference between rdb and aof persistence",
    "why does postgres need vacuum",
    "when should i use a token bucket rate limiter",
    "what does nprobe control in faiss",
    "how do kafka consumer groups work",
    "what problem do virtual nodes solve",
    "why are cross encoders slow",
]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, int(round(pct / 100.0 * len(ordered))))
    return ordered[rank - 1]


async def worker(client: httpx.AsyncClient, url: str, n: int,
                 offset: int, out: list) -> None:
    for i in range(n):
        q = QUERIES[(offset + i) % len(QUERIES)]
        t0 = time.perf_counter()
        resp = await client.post(f"{url}/query", json={"query": q})
        elapsed = (time.perf_counter() - t0) * 1000.0
        out.append((resp.status_code, elapsed,
                    resp.headers.get("retry-after")))
        if resp.status_code not in (200, 503):
            raise RuntimeError(
                f"unexpected status {resp.status_code}: {resp.text[:200]}"
            )


async def run_level(url: str, concurrency: int, requests_per_worker: int,
                    api_key: str | None) -> dict:
    headers = {"x-api-key": api_key} if api_key else {}
    results: list = []
    async with httpx.AsyncClient(timeout=120.0, headers=headers) as client:
        wall0 = time.perf_counter()
        await asyncio.gather(*(
            worker(client, url, requests_per_worker, w * 3, results)
            for w in range(concurrency)
        ))
        wall = time.perf_counter() - wall0

    oks = [ms for status, ms, _ in results if status == 200]
    rejected = [r for r in results if r[0] == 503]
    return {
        "offered_concurrency": concurrency,
        "sent": len(results),
        "ok": len(oks),
        "shed_503": len(rejected),
        "shed_rate": round(len(rejected) / len(results), 3),
        "ok_p50_ms": round(percentile(oks, 50), 1),
        "ok_p95_ms": round(percentile(oks, 95), 1),
        "ok_p99_ms": round(percentile(oks, 99), 1),
        "ok_mean_ms": round(statistics.mean(oks), 1) if oks else None,
        "retry_after_sample": rejected[0][2] if rejected else None,
        "throughput_ok_rps": round(len(oks) / wall, 2),
        "wall_s": round(wall, 1),
    }


async def main_async(args) -> dict:
    levels = [int(x) for x in args.concurrency.split(",")]
    async with httpx.AsyncClient(timeout=30.0) as client:
        health = await client.get(f"{args.url}/health")
        health.raise_for_status()
        admission_before = (
            await client.get(f"{args.url}/admin/admission")
        ).json()

    print(f"admission config: {admission_before['max_concurrency']} "
          f"executing + {admission_before['max_queue_depth']} queued")
    header = (f"{'conc':>4} {'sent':>5} {'ok':>4} {'503':>4} {'shed%':>6} "
              f"{'p50':>8} {'p95':>8} {'p99':>8} {'ok rps':>7}")
    print(header)
    print("-" * len(header))
    rows = []
    for level in levels:
        row = await run_level(args.url, level, args.requests, args.api_key)
        rows.append(row)
        print(f"{row['offered_concurrency']:>4} {row['sent']:>5} "
              f"{row['ok']:>4} {row['shed_503']:>4} "
              f"{row['shed_rate']*100:>5.1f}% {row['ok_p50_ms']:>7.1f}ms "
              f"{row['ok_p95_ms']:>7.1f}ms {row['ok_p99_ms']:>7.1f}ms "
              f"{row['throughput_ok_rps']:>7.2f}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        admission_after = (
            await client.get(f"{args.url}/admin/admission")
        ).json()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "url": args.url,
        "requests_per_worker": args.requests,
        "admission_config": admission_before,
        "admission_after": admission_after,
        "levels": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", default="1,2,4,8,16")
    parser.add_argument("--requests", type=int, default=40,
                        help="requests per worker (closed loop)")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    report = asyncio.run(main_async(args))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"json written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
