# Stage 6 Report — Observability

Date: 2026-07-11. Code: `app/observability.py` (all metric definitions),
instrumentation in `app/core/hybrid.py`, `app/generation/service.py`,
`app/api/query.py`, `app/main.py`. Demo: `scripts/observability_demo.py`.

## Metrics (requirement → metric, all live-verified below)

| Requirement | Metric |
|---|---|
| request latency | `ragp_http_request_duration_seconds{method,path,status}` (route templates as labels, never raw URLs) |
| retrieval latency | `ragp_retrieval_duration_seconds{stage}` — embed / bm25 / dense / rerank / total |
| error rate by type | `ragp_errors_total{type}` — unhandled, llm_quota_429, llm_timeout, llm_server_error, llm_malformed, llm_auth, auth_failed, rate_limited, daily_quota, queue_full |
| cache hit rate | `ragp_cache_requests_total{result=hit\|miss\|bypass}` |
| citation-validation failures | `ragp_citation_sentences_total{verdict}` + `ragp_citation_rejected_answers_total` |
| quota-throttle rate | `ragp_quota_throttled_total{reason}` |
| supporting | `ragp_llm_requests_total{outcome}`, `ragp_rerank_status_total{status}`, `ragp_admission_rejected_total` |

Histogram buckets bracket the SLOs measured in Stages 3–5 (0.5s
throttled retrieval floor, 3.5s admitted p95, 20s LLM ceiling). Label
cardinality is closed-set only; no user data ever becomes a label
(tested: query text absent from /metrics).

**Scope addition, on the record**: a cache-hit-rate metric requires a
cache. The Redis response cache mandated by Stage 2.5's capacity math is
now implemented (`app/api/query.py`): normalized-query key, 1h TTL, only
stable outcomes cached (transient degradations — quota, capacity,
provider errors — are never replayed; tested). A hit spends no admission
slot, no pipeline compute, no LLM quota, and returns `cached: true`.

## Definition of done 1 — one request traced end-to-end by its ID alone

Demo: `ENVIRONMENT=development REDIS_URL=redis://127.0.0.1:56379/0
python scripts/observability_demo.py` — real pipeline + models + Redis +
QuotaGuard; scripted LLM (no live key exists) driving each failure mode
deterministically. Grepping the captured logs for ONE id
(`2974547c190b4af1`, the partial-fabrication request) yields the full
lifecycle, verbatim:

```
[info] anonymous_request_dev_mode        client=testclient          request_id=2974547c190b4af1
[info] retrieval_completed  bm25_ms=0.3 candidates=10 dense_ms=0.1 embed_ms=42.5
                            rerank_ms=278.0 rerank_scored=10 rerank_status=full
                            total_ms=321.0                         request_id=2974547c190b4af1
[info] generation_completed llm_model=scripted llm_ms=0.0 output_tokens=30
                            sentences_kept=1 sentences_rejected=1  request_id=2974547c190b4af1
[info] citation_validation_rejected_some kept=1 rejected=1         request_id=2974547c190b4af1
[info] request_completed    citations=1 degraded=False outcome=served
                            rerank_status=full status=ok_partial_rejected request_id=2974547c190b4af1
```

auth → retrieval (per-stage timings) → generation → citation validation
→ response, one grep. The id propagates into pipeline worker threads via
contextvars (anyio copies the context into `to_thread` workers), is
returned in `X-Request-ID`, and appears in error bodies — the same id a
client would report.

## Definition of done 2 — live nonzero values for every listed metric

`/metrics` after the demo's 6-request traffic mix (verbatim, `_created`
series omitted):

```
ragp_http_request_duration_seconds_count{method="POST",path="/v1/query",status="200"} 6.0
ragp_retrieval_duration_seconds_count{stage="embed"} 5.0
ragp_retrieval_duration_seconds_count{stage="bm25"} 5.0
ragp_retrieval_duration_seconds_count{stage="dense"} 5.0
ragp_retrieval_duration_seconds_count{stage="rerank"} 5.0
ragp_retrieval_duration_seconds_count{stage="total"} 5.0
ragp_errors_total{type="llm_quota_429"} 1.0
ragp_cache_requests_total{result="miss"} 5.0
ragp_cache_requests_total{result="hit"} 1.0
ragp_citation_sentences_total{verdict="supported"} 2.0
ragp_citation_sentences_total{verdict="unsupported"} 3.0
ragp_citation_rejected_answers_total 1.0
ragp_quota_throttled_total{reason="provider_cooldown"} 1.0
ragp_llm_requests_total{outcome="ok"} 3.0
ragp_llm_requests_total{outcome="quota_429"} 1.0
ragp_rerank_status_total{status="full"} 5.0
```

Cross-checks that these are measuring reality: 6 HTTP requests but only
5 retrievals — request 6 was the cache hit; 5 misses + 1 hit = 6 cache
lookups; 3 unsupported sentences = 1 (partial fabrication) + 2 (full
fabrication); exactly 1 rejected answer; the provider 429 shows in BOTH
`errors_total{llm_quota_429}` (reactive) and the subsequent
`quota_throttled_total{provider_cooldown}` (proactive) — the Stage 4.5
operator distinction, now visible in metrics.

## Standing eval + suite

Eval re-run vs baseline (hybrid): P@1 0.9000 (+0.05), MRR@10 0.9417
(+0.0167), hallucination 0.0, latency p50 666ms (instrumentation
overhead is within run-to-run noise vs Stage 5's 638–659ms range).
Suite: **143 passing** (6 new observability tests: /metrics format,
cache hit/normalization, transient-status cache exclusion, route-template
labels + no-user-data-in-labels, unhandled-error counter, citation/LLM
counter movement through the real service).

Honesty notes: the demo LLM is scripted because no API key exists in
this environment — retrieval, models, cache, quota accounting, and the
HTTP stack are real; the scripted part is exactly the component whose
failure modes needed deterministic driving. Metrics are per-process
(acceptable: single worker by design); Prometheus must scrape the single
instance, and counters reset on restart/redeploy (normal for the model).
