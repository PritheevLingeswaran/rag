# Stage 4.5 Report — Quota-Aware Generation Guardrails

Date: 2026-07-11. Code: `app/generation/quota.py`, integration in
`app/generation/service.py`. Tests: `tests/generation/test_quota.py`,
`tests/integration/test_quota_redis.py`.

## Design

**Single source of limit truth.** `QuotaGuard` loads RPM/RPD from
`configs/free_tier_limits.json` — the Stage 2.5 artifact — keyed by the
configured model. An unrecorded model raises `ConfigurationError`:
quota numbers are sourced, never guessed. Current model
(`gemini-2.5-flash-lite`): provider 15 RPM / 1,000 RPD.

**Proactive throttling.** Every LLM call must first pass
`try_acquire()`, which enforces `floor(provider_limit × 0.9)` —
13 RPM / 900 RPD — via atomic counters. We hit *our* wall before
Google's, so degradation happens on our terms with a clean label and an
accurate `retry_after_s` (seconds to next minute for RPM; seconds to
**midnight US Pacific** for RPD, matching the provider's documented
reset). Counters live in Redis (Lua `INCR`+`EXPIRE`, shared across
workers — verified with two guard instances alternating against real
Redis: together they get exactly 13, not 26). When Redis is down the
guard falls back to in-process counting rather than failing open
entirely (tested against a dead Redis).

**Reactive sync.** If the provider 429s anyway (multi-worker
undercount on the local fallback, or another consumer sharing the
Google project), `record_provider_rejection()` opens a cooldown for
the provider's retry-after, so subsequent requests are throttled
proactively without touching the API. The reactive path corrects the
proactive one.

**Degraded response contract** (extends the Stage 4 table):

| situation | status | client receives |
|---|---|---|
| our budget exhausted (LLM never called) | `degraded_quota_throttled` | retrieval-only extractive answer + citations, `degraded: true`, `retry_after_s`, `throttle_reason` ∈ {rpm_exhausted, rpd_exhausted, provider_cooldown} |
| provider 429 despite budget | `degraded_quota` | same shape; opens cooldown |

**Operator distinction (logs route the response):**

| signal | level | meaning | operator action |
|---|---|---|---|
| `llm_quota_throttled_proactive` | INFO | expected budget management | none — working as designed |
| `llm_quota_exhausted` (provider 429) | WARNING | accounting slipped or shared project quota | check for other consumers / margin |
| `llm_server_error_after_retry`, `llm_malformed_response`, `llm_auth_failure_check_config` | ERROR | actual API failure | provider incident / config bug |

## Definition of done — boundary test (passing)

`test_system_at_rpm_boundary_serves_degraded_not_500` drives the full
`GenerationService` to the exact enforced boundary from the Stage 2.5
file: 13 requests (= floor(15 × 0.9)) generate normally; **request 14
returns `degraded_quota_throttled` with the retrieval-only answer,
citations intact, `retry_after_s ≤ 60`, and the LLM call counter still
at 13** — the provider is never touched and nothing raises. One minute
later (fake clock) service returns to `ok`. The RPD twin drives 900
requests across simulated minutes and shows request 901 denied with
retry-after ≈ seconds-to-Pacific-midnight, resetting after it.

```
tests/generation/test_quota.py::test_limits_loaded_from_stage25_file PASSED
tests/generation/test_quota.py::test_unknown_model_refuses_to_guess PASSED
tests/generation/test_quota.py::test_enforced_budget_is_90_percent_of_provider PASSED
tests/generation/test_quota.py::test_rpm_boundary_allows_exactly_enforced_then_denies PASSED
tests/generation/test_quota.py::test_rpm_window_resets_next_minute PASSED
tests/generation/test_quota.py::test_rpd_boundary_denies_with_retry_until_pacific_midnight PASSED
tests/generation/test_quota.py::test_rpd_resets_after_pacific_midnight PASSED
tests/generation/test_quota.py::test_provider_rejection_opens_cooldown_then_clears PASSED
tests/generation/test_quota.py::test_system_at_rpm_boundary_serves_degraded_not_500 PASSED
tests/generation/test_quota.py::test_provider_429_and_proactive_throttle_are_distinct_statuses PASSED
tests/integration/test_quota_redis.py::test_rpm_boundary_shared_across_workers PASSED
tests/integration/test_quota_redis.py::test_counter_keys_expire PASSED
tests/integration/test_quota_redis.py::test_redis_down_falls_back_to_local_counting PASSED
```

## Standing eval (unchanged, as expected)

`python eval/run_eval.py --pipeline generation --baseline
eval/results/baseline.json --tag stage4_5-quota` — P@1 0.9000 (+0.05),
MRR@10 0.9417 (+0.0167), hallucination 0.0, p50 638.0 ms. The guard
only gates live-LLM calls; the keyless eval path is untouched. Suite:
116 tests passing.

## Notes / known limits

- RPD counting deliberately increments before RPM checking, so an
  RPM-denied request costs one RPD slot; at 900/day vs ≤13/min this is
  negligible and keeps the counters trivially simple (documented in code).
- TPM (250K tokens/min) is not yet enforced: at ≤13 RPM with ~2K-token
  prompts we peak at ~26K TPM, 10% of the limit — RPM binds first by an
  order of magnitude. Becomes relevant only if prompts grow ~10x.
- Quota accounting counts *attempts we authorize*, not provider-confirmed
  spend; a timed-out request may or may not have counted provider-side.
  The 10% margin absorbs this asymmetry.
