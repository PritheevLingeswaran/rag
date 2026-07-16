# Stage 8.9 Report — Privacy Policy & Data Practices

Date: 2026-07-11. Policy: `PRIVACY.md`, served at `GET /privacy`
(verified from the production container: 200), linked from the API's
own OpenAPI description. Conformance tests: `tests/api/test_privacy.py`
(5, passing) — the policy's claims are ENFORCED by the suite so code
and policy cannot silently drift.

## Recommended (and implemented) retention stance

**Raw user queries are never persisted, and nothing stored is tied to
identity.** There is no stated reason to keep them today, so the
default is zero retention. The pre-existing `query_logs` capability
(Stage 2 schema) remains deliberately unwired, now with a guard
docstring requiring a PRIVACY.md update *before* any wiring — and a
test that fails if anyone wires it without noticing.

## Policy claim ↔ code, side by side

| Policy claim | Code that makes it true | Enforced by |
|---|---|---|
| "We do not store your queries in any database" | `QueryLogRepo` (the only query-text writer) referenced nowhere in `app/` outside its definition | `test_policy_claim_queries_never_persisted` (greps the tree) |
| "logs never contain query text, answers, or IP addresses written by the app" | every `logger.*` call in the serving path passes only timings/statuses/counts (`request_completed`, `retrieval_completed`, `generation_completed`); the single client-address log is the dev-only anonymous branch, unreachable in production (keys required to boot) | `test_policy_claim_no_query_text_or_ip_in_app_logs` |
| app writes no per-request IP logs | `--no-access-log` in the production CMD (change made this stage — uvicorn's default access log DID log IP+path, the one real gap the audit found) | `test_policy_claim_access_log_disabled_in_image` + live check: 2 requests to the prod container produced **0** access-log lines |
| cached responses ≤ 1 hour, not linked to you | `cache_key = sha256(normalized query)` — no client id in key or value; `cache_ttl_s = 3600` | `test_policy_claim_cache_ttl_one_hour`; cache code in `app/api/query.py` |
| rate-limit counters keyed to a truncated key tag, ≤ 25 h | `client_id = f"key:{sha256(key)[:12]}"` (truncated digest — non-reversible, and collision-free unlike the earlier `key[:4]` prefix); TTLs 60 s / 86 400+3 600 s | `app/api/deps.py`, `app/api/query.py` |
| no user data in metrics | closed-set labels only | Stage 6 test (`test_http_request_duration_labels_use_route_template`) |
| Gemini disclosure | queries + snippets go to Google only when `GEMINI_API_KEY` set; free-tier "Google may use content" stated plainly in the policy | `app/generation/llm_client.py` is the only egress |

## Honest notes

- Render's edge logs requests (incl. IP) under Render's own policy —
  disclosed rather than pretended away; the app-side log is silent.
- "Published and linked from the live app": ~~gated on the Stage 8
  account handoff~~ — **closed 2026-07-16**: `GET
  https://ragp-pwf2.onrender.com/privacy` → 200 (full policy, 2,871
  bytes) on the live service, and the live `/openapi.json` description
  links it ("Privacy policy: [/privacy](/privacy) — no accounts,
  queries never stored tied to identity"). Conformance tests re-run
  same day: 5/5 passing locally and within CI's 206 on master.
- Policy contact is "the project repository's issue tracker" — swap in
  a preferred contact channel if you want one.

Suite: **205 passing** (5 new). Standing eval unchanged vs baseline.
