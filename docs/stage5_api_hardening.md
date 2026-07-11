# Stage 5 (API Layer) Report — Versioning, Validation, Hardening

Date: 2026-07-11. Builds on the admission-control work in
`docs/stage5_admission.md`; this report covers the API contract layer.

## What was built

- **Versioned endpoints**: `POST /v1/query` (OpenAPI at `/docs` /
  `/openapi.json`; the admin introspection endpoint is authenticated and
  excluded from public docs). `/health` stays unversioned for platform
  probes.
- **Strict schemas** (`app/api/query.py`): `extra="forbid"` (unknown
  fields → 422), `query` bounded 1–2000 chars, `max_tokens` bounded
  16–1024, types enforced; documented response model with the explicit
  degradation fields (`status` / `degraded` / `rerank_status`).
- **Auth**: `x-api-key`, constant-time compare, per-key identity used
  for limits. Production refuses to boot without keys.
- **Per-key limits (cost guardrails)**: requests/day quota (default
  500/key/UTC-day, Redis-backed, 429 + Retry-After to UTC midnight)
  checked before the per-minute limit (default 30/min); `max_tokens`
  can only LOWER the server's generation ceiling, never raise it —
  clamped in the Gemini client itself.
- **CORS lockdown**: deny-by-default (no CORS middleware unless
  `CORS_ORIGINS` set → browsers get no `Access-Control-Allow-Origin` at
  all); exact origins only; `*` in production is a boot-time
  `ConfigurationError`.
- **Request size limit**: bodies over 16 KB rejected 413 before
  parsing; missing Content-Length on body methods rejected 411 (chunked
  uploads would bypass the size check).
- **Request IDs**: every request gets an id, bound into every log line
  (structlog contextvars), echoed in `X-Request-ID` and in error bodies
  — errors are diagnosable without leaking anything.
- **Leak-proof errors**: a global handler logs the full traceback
  server-side with the request id and returns exactly
  `{"error": "internal server error", "request_id": ...}`. Nothing
  else, ever.

## Definition of done — actual response body

A service deliberately rigged to throw
`RuntimeError("psycopg2.OperationalError: FATAL password 'hunter2' for
user ragp_admin at 10.0.3.17:5432 -- traceback follows")` was triggered
through the real app stack. The complete client-visible response:

```
HTTP 500
content-type: application/json
x-request-id: 366a14cc1cb642bd
BODY: {"error":"internal server error","request_id":"366a14cc1cb642bd"}
```

The full traceback (including the planted credentials) appeared only in
the server log, keyed by the same request id. The regression test
(`test_internal_exception_leaks_nothing_to_client`) additionally asserts
none of `hunter2 / psycopg / OperationalError / 10.0.3.17 / ragp_admin /
Traceback / RuntimeError / .py` appear anywhere in the response, and that
the body has exactly the two expected keys. Malformed JSON bodies get a
clean 422 describing the client's input, not our internals.

## Guardrail matrix (each row is a passing test)

| Attack / mistake | Response | Test |
|---|---|---|
| Internal exception w/ secrets | 500, generic body + request id only | `test_internal_exception_leaks_nothing_to_client` |
| Unknown/injected fields | 422 | `test_unknown_fields_rejected` |
| Empty / 2001-char / non-string query | 422 | `test_query_length_bounds` |
| `max_tokens` 999999 or 4 | 422; valid values clamped server-side | `test_max_tokens_bounds_and_passthrough` |
| 20 KB body | 413 before parsing | `test_oversized_body_rejected_413_before_parsing` |
| Cross-origin by default | no CORS headers at all | `test_no_cors_headers_by_default` |
| Unlisted origin with CORS on | no allow-origin header | `test_cors_allows_only_configured_origin` |
| `CORS_ORIGINS=*` in production | refuses to boot | `test_wildcard_cors_refused_in_production` |
| 501st request of the day | 429 + Retry-After ≤ 24h | `test_daily_quota_429_with_retry_after` |
| Missing/wrong API key | 401 | `test_missing_api_key_is_401_when_keys_configured` |
| Overload | 503 + Retry-After (admission) | `test_queue_full_returns_503_with_retry_after` |

## Standing eval + suite

Eval re-run vs baseline (hybrid): P@1 0.9000 (+0.05), MRR@10 0.9417
(+0.0167), hallucination 0.0 — API layer does not touch retrieval.
Suite: **137 passing** (22 API tests).

Notes on scope: per-key quotas require Redis; without `REDIS_URL`
(development) they are skipped and logged — Upstash provides them in
production per Stage 2.5. Daily API quota (500/key) is deliberately
above the global Gemini budget (900/day enforced, Stage 4.5): the
per-key quota bounds a single abusive key, the global quota guard
bounds total LLM spend, and beyond it users get labeled retrieval-only
degradation rather than denial.
