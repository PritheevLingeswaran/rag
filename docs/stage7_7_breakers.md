# Stage 7.7 Report — Quota & Cost Circuit Breakers

Date: 2026-07-11. Code: `app/reliability.py` (AlertManager,
ResourceBudget, PostgresStorageBreaker) + wiring in RedisStore,
QuotaGuard, IngestionPipeline, app lifespan, ingestion CLI. Tests:
`tests/reliability/test_circuit_breakers.py` (12, all passing —
verbatim run output below).

## The model (uniform across quotas)

| usage | state | behavior |
|---|---|---|
| < 80% of enforced budget | CLOSED | normal |
| ≥ 80% | ALERT | still serving; **one alert per resource per UTC day**: webhook JSON POST (works with ntfy.sh / Discord / Slack / healthchecks — set `ALERT_WEBHOOK_URL`) + CRITICAL log always |
| ≥ enforced (= 90% of provider hard limit) | OPEN | guarded operation refused/bypassed **before** the provider limit; labeled degradation, never a crash |
| past provider hard limit | reactive | provider errors absorbed by the typed paths built in Stages 2–4.5 |

Alerting is fire-and-forget: a dead webhook receiver is logged and never
propagates (tested: `test_alert_failure_never_breaks_the_caller`).
Budgets reset at UTC midnight (tested), except the LLM budget which
follows the provider's Pacific-midnight reset (Stage 4.5).

## Three evidenced states per quota (definition of done)

### Quota 1 — Gemini RPD (1,000/day hard; 900 enforced; from the Stage 2.5 file)
- **80%**: request #720 of the day fires exactly one captured webhook
  (`resource=gemini_rpd:gemini-2.5-flash-lite, pct≈0.8`); requests 721–770
  keep serving; alert deduped. (`test_llm_rpd_state1...`)
- **Trip (900)**: request #901 → `degraded_quota_throttled`, retrieval-only
  answer with citations, `retry_after_s` to Pacific midnight, **LLM call
  counter still 0** — the provider is never touched. (`..._state2...`)
- **Past limit**: provider 429 → `degraded_quota` with the provider's
  retry-after, cooldown opens, next request throttled proactively
  (`provider_cooldown`); both answers grounded+cited, nothing raised.
  (`..._state3...`)

### Quota 2 — Neon Postgres storage (0.5 GB hard; 450 MB enforced)
- **80%** (377 MB): `check_writable()` allows, state ALERT, one webhook
  (`resource=postgres_storage`), deduped on re-check. (`test_pg_state1...`)
- **Trip (450 MB)**: ingestion refuses **before touching the database**
  (the test's connection object asserts if any attribute is accessed):
  `status=aborted_storage_budget` with an actionable message. Reads are
  never gated — Neon keeps reads alive past its limit, so only writes are
  guarded. (`test_pg_state2...`)
- **Past limit**: a write failing server-side ("No space left on
  device") produces `status=failed_storage`, typed and clean — no stack
  trace, no partial index. (`test_pg_state3...`)
- Design note on record: if the size *check itself* fails, the breaker
  fails OPEN with an ERROR log — a monitoring bug must not become a
  write outage; the provider limit remains the reactive backstop.

### Quota 3 — Upstash Redis commands (500K/month ⇒ 16,129/UTC-day; enforced 14,516)
RedisStore now meters every command it issues (test uses a 100/day
budget; production value from config, derived from the Stage 2.5 file):
- **80%**: command #72 (= 0.8 × 90 enforced) fires one webhook; store
  keeps operating. (`test_redis_state1...`)
- **Trip**: with the budget open, **zero further commands are issued**
  (proven by counting commands at the fake client): cache degrades to
  misses, `cache_set` no-ops, counters return None (LLM quota guard falls
  back to its local backend), rate limiting fails open per the Stage 2
  policy. An end-to-end `/v1/query` still returns 200. (`..._state2...`,
  `..._state2b...`)
- **Past limit**: Upstash-style throttling errors on every command are
  absorbed by the Stage 2 soft-fail paths — miss/no-op/fail-open, no
  exception. (`..._state3...`)

### Not app-enforceable, stated rather than skipped
Render instance-hours/bandwidth and Neon CU-hours cannot be measured
from inside the process. They are covered by: single always-on service
≤ 744 h < 750 h by construction; ops runbook checks (Stage 8 deploy
checklist); and the platform's own suspension behavior, which the
admission/degradation layers turn into downtime rather than corruption.

## Test run (verbatim)

```
tests/reliability/test_circuit_breakers.py::test_llm_rpd_state1_alert_at_80_percent PASSED
tests/reliability/test_circuit_breakers.py::test_llm_rpd_state2_breaker_trips_at_enforced_900 PASSED
tests/reliability/test_circuit_breakers.py::test_llm_rpd_state3_past_provider_limit_429_absorbed PASSED
tests/reliability/test_circuit_breakers.py::test_pg_state1_alert_at_80_percent PASSED
tests/reliability/test_circuit_breakers.py::test_pg_state2_breaker_open_refuses_ingestion_before_writing PASSED
tests/reliability/test_circuit_breakers.py::test_pg_state3_past_hard_limit_write_failure_is_typed_not_a_crash PASSED
tests/reliability/test_circuit_breakers.py::test_redis_state1_alert_at_80_percent PASSED
tests/reliability/test_circuit_breakers.py::test_redis_state2_breaker_open_bypasses_all_redis_traffic PASSED
tests/reliability/test_circuit_breakers.py::test_redis_state2b_end_to_end_request_still_serves PASSED
tests/reliability/test_circuit_breakers.py::test_redis_state3_past_hard_limit_provider_throttling_absorbed PASSED
tests/reliability/test_circuit_breakers.py::test_budget_resets_at_utc_midnight PASSED
tests/reliability/test_circuit_breakers.py::test_alert_failure_never_breaks_the_caller PASSED
============================= 12 passed in 1.70s ==============================
```

One semantics bug found by its own test while building: the budget's
alert originally classified the PRE-event count, firing one event late;
fixed to admit on pre-count (exactly `enforced` events pass) and
classify on post-count (the crossing event itself alerts).

## Standing eval + suite

Suite: **200 passing.** Standing eval unchanged vs baseline (P@1 0.9000
+0.05, MRR@10 0.9417 +0.0167, hallucination 0.0) — breakers gate writes
and budgets, not retrieval.

To receive alerts personally before Stage 8: set `ALERT_WEBHOOK_URL`
(e.g. a free ntfy.sh topic — `https://ntfy.sh/<your-secret-topic>` —
delivers straight to your phone/email with zero signup).
