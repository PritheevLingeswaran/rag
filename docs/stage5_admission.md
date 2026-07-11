# Stage 5 Report — API Serving & Admission Control

Date: 2026-07-11. Code: `app/api/query.py`, `app/api/admission.py`,
`app/main.py`. Raw runs: `eval/results/api_loadtest_local.json`,
`api_loadtest_throttled.json`, `api_loadtest_throttled_depth4.json`.

## What was built

`POST /query` behind three ordered gates, each rejecting before the next
spends work:

1. **Auth** — `x-api-key`, constant-time compare against `API_KEYS`;
   401 on failure. Production refuses to boot without keys; development
   allows anonymous (logged).
2. **Per-client rate limit** — Redis fixed-window (Stage 2 Lua
   primitive), 429 + Retry-After; fail-open on Redis outage (documented
   Stage 2 tradeoff).
3. **Admission control** — bounded queue: `max_concurrency` requests
   executing, `max_queue_depth` waiting, everything else **immediate
   503 + Retry-After** where Retry-After = backlog × live service-time
   EWMA / concurrency. No request ever waits in an unbounded line —
   this is the structural answer to the Stage 3/4 pathology where
   offered concurrency 4 at 0.1 CPU produced 77s p50 (unbounded
   queueing), and to the discarded host-stall row.

Pipeline execution runs in a worker thread; the event loop stays free to
serve health checks and shed load while requests compute. Single uvicorn
worker **by design**: 512MB cannot hold two model copies, and admission
state is per-process. Startup performs one warmup query, which also
seeds the rerank-budget EWMA (closing the Stage 4 first-request-overshoot
caveat).

## Empirical concurrency ceiling (measured through the mechanism)

Commands (exact):
```
python scripts/load_test_api.py --url http://127.0.0.1:8000  --concurrency 1,2,4,8,16 --requests 30 --json eval/results/api_loadtest_local.json
docker run -d --name ragp-api-throttled --cpus=0.1 --memory=512m -p 18000:8000 -v "$PWD:/app" -v ragp-hf-cache:/root/.cache/huggingface -e SERVE_PIPELINE=true ragp-loadtest python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
python scripts/load_test_api.py --url http://127.0.0.1:18000 --concurrency 1,2,4,8,16 --requests 8 --json eval/results/api_loadtest_throttled.json
# re-measured after tuning queue depth 6 -> 4:
python scripts/load_test_api.py --url http://127.0.0.1:18000 --concurrency 4,6,8,16 --requests 8 --json eval/results/api_loadtest_throttled_depth4.json
```

### Local (8-core laptop, real models, closed-loop workers)

| offered conc | shed % | admitted p50 | admitted p95 | ok rps |
|---|---|---|---|---|
| 1 | 0% | 327 ms | 355 ms | 3.0 |
| 2 | 0% | 355 ms | 381 ms | 5.6 |
| 4 | 0% | 715 ms | 773 ms | 5.5 |
| 8 (= capacity 2+6) | 0% | 1423 ms | 1499 ms | 5.6 |
| 16 | **54.6%** | 1308 ms | 1501 ms | 5.6 |

The property the mechanism buys: **admitted-request p95 is flat
(~1.5 s) no matter how much load is offered**; excess becomes clean
503s, not tail latency.

### 0.1-CPU / 512MB container (Render-like), cold start ~53 s (cached models)

First sweep (queue depth 6):

| offered conc | shed % | admitted p50 | admitted p95 | ok rps |
|---|---|---|---|---|
| 1 | 0% | 465 ms | 506 ms | 2.2 |
| 2 | 0% | 904 ms | 1094 ms | 2.1 |
| 4 | 0% | 1803 ms | 2198 ms | 2.1 |
| 8 (= capacity) | 0% | 3901 ms | 4398 ms | 2.0 |
| 16 | 62.5% | 3496 ms | 5395 ms | 1.9 |

The conc-1 row also validates the Stage 4 rerank budget end-to-end in
production conditions: 465 ms ≈ the measured RRF-only floor — the EWMA
gate learned throttled rerank cost during warmup and skipped, exactly as
designed (`rerank_status: skipped_budget` on responses).

Depth 6 was then REJECTED by its own data: a queue slot that buys a
~4.4 s wait is worse than telling the client to retry. Re-measured at
depth 4:

| offered conc | shed % | admitted p50 | admitted p95 | ok rps |
|---|---|---|---|---|
| 4 | 0% | 1902 ms | 2603 ms | 2.0 |
| 6 (= capacity 2+4) | 0% | 2998 ms | 3503 ms | 2.0 |
| 8 | 25.0% | 2999 ms | 3597 ms | 2.0 |
| 16 | 78.1% | 2705 ms | 5005 ms | 1.7 |

## The honest ceiling, stated plainly (0.1 CPU, current corpus)

- **Sustainable throughput: ~2 requests/second.** That is the machine.
- **p95 ≤ 1 s holds only up to ~2 concurrent closed-loop clients.**
- **Up to 6 concurrent clients (admission capacity 2+4): nothing is
  shed; admitted p95 ≈ 3.5 s.**
- **Beyond 6: requests are shed with 503 + honest Retry-After; admitted
  p95 stays bounded ≤ ~5 s even at 16 offered** — versus the 77–100 s+
  unbounded-queueing regime this replaces.
- "Dozens of concurrent users" therefore means dozens of *humans*, whose
  think-time keeps offered concurrency low and whose repeated questions
  hit the (Stage 2.5-mandated) response cache — not dozens of
  closed-loop request generators. That distinction is now measured, on
  the record, and enforced by the admission bound instead of hoped for.
- Cold start at 0.1 CPU: ~53 s app-level (models cached on disk),
  additive with Render's own ~30–60 s platform spin-up.

Caveats: 60-chunk corpus behind the API (retrieval core at 50k measured
separately in Stages 3–4; the delta is ~10 ms local since embed+rerank
dominate); Docker-on-Windows CFS throttling approximates but does not
equal Render's scheduler; 8-requests-per-worker throttled samples give
coarse percentiles. Re-measure on the real Render container at deploy.

## Standing eval + tests

Eval re-run vs baseline (hybrid): P@1 0.9000 (+0.05), MRR@10 0.9417
(+0.0167), hallucination 0.0 — serving layer does not touch retrieval,
as expected. Suite: **126 passing**, including deterministic saturation
tests: 503-at-exact-boundary with Retry-After header while the pipeline
sees only the admitted requests (`test_queue_full_returns_503_with_retry_after`),
bounded true concurrency, retry-after scaling with backlog, and a fixed
free-slot bug at `max_queue_depth=0` found by its own test.
