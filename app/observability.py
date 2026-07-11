"""Prometheus metrics: the single place every metric is defined.

Naming: ragp_* prefix, seconds for durations, _total for counters.
Buckets are chosen to bracket the SLOs measured in Stages 3-5 (retrieval
floor ~0.5s throttled, admitted p95 ~3.5s, LLM calls 1-20s) -- per our
own corpus doc on Prometheus histograms, quantile accuracy depends on
bucket boundaries bracketing the target.

Metric map (Stage 6 requirement -> metric):
    request latency            ragp_http_request_duration_seconds{method,path,status}
    retrieval latency          ragp_retrieval_duration_seconds{stage}
    error rate by type         ragp_errors_total{type}
    cache hit rate             ragp_cache_requests_total{result}
    citation-validation fails  ragp_citation_sentences_total{verdict}
                               ragp_citation_rejected_answers_total
    quota-throttle rate        ragp_quota_throttled_total{reason}
    (supporting)               ragp_llm_requests_total{outcome}
                               ragp_rerank_status_total{status}
                               ragp_admission_rejected_total

Cardinality discipline: labels are small closed sets (status codes,
enum-like stages/outcomes); nothing user-controlled (no query text, no
client ids) ever becomes a label.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

HTTP_REQUEST_DURATION = Histogram(
    "ragp_http_request_duration_seconds",
    "End-to-end HTTP request duration",
    ["method", "path", "status"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.5, 5.0, 10.0, 30.0),
)

RETRIEVAL_DURATION = Histogram(
    "ragp_retrieval_duration_seconds",
    "Retrieval pipeline stage durations",
    ["stage"],  # embed | bm25 | dense | rerank | total
    buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 15.0),
)

ERRORS = Counter(
    "ragp_errors_total",
    "Errors by type (unhandled, llm_*, auth_failed, rate_limited, "
    "daily_quota, queue_full)",
    ["type"],
)

CACHE_REQUESTS = Counter(
    "ragp_cache_requests_total",
    "Response cache lookups",
    ["result"],  # hit | miss | bypass (no redis)
)

CITATION_SENTENCES = Counter(
    "ragp_citation_sentences_total",
    "Citation validator per-sentence verdicts",
    ["verdict"],  # supported | supported_uncited | invalid_citation | unsupported
)

CITATION_REJECTED_ANSWERS = Counter(
    "ragp_citation_rejected_answers_total",
    "Answers fully rejected by citation validation (fell back to extractive)",
)

QUOTA_THROTTLED = Counter(
    "ragp_quota_throttled_total",
    "Proactive LLM quota throttles by reason",
    ["reason"],  # rpm_exhausted | rpd_exhausted | provider_cooldown
)

LLM_REQUESTS = Counter(
    "ragp_llm_requests_total",
    "LLM call outcomes",
    ["outcome"],  # ok | quota_429 | timeout | server_error | malformed | auth
)

RERANK_STATUS = Counter(
    "ragp_rerank_status_total",
    "Rerank outcomes per request",
    ["status"],  # full | partial | skipped_budget | disabled | no_candidates
)

ADMISSION_REJECTED = Counter(
    "ragp_admission_rejected_total",
    "Requests shed by admission control (503)",
)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
