# Stage 9.9 Report — Production hardening

Date: 2026-08-08. Three defects found by probing the *deployed* instance
rather than reading the code. Each was invisible to the test suite
because each concerned how the app behaves toward the platform and the
browser, not toward a caller.

## 1. A broken instance reported itself healthy

`/health` returned `{"status": "ok"}` whenever the process was alive and
settings parsed — its own docstring recorded dependency checks as
deferred. Render's `healthCheckPath` pointed at it.

Consequence, reproduced locally: an instance configured to serve but with
no pipeline answered `ok` to every health probe while every `/v1/query`
raised `AttributeError` → 500 **and fired an operator page**. A config
mistake would have produced an alert storm from an instance the platform
believed was fine and would never have restarted.

Fixed by splitting the two questions that were conflated:

| endpoint | question | on failure |
|---|---|---|
| `/health` | is the process alive and configured? | never fails — a restart would not help, so it reports `ready: false` in a field instead |
| `/health/ready` | can this instance serve a query *now*? | **503**, so the platform takes it out of rotation |

`render.yaml` now gates on `/health/ready`. Safe because uvicorn accepts
connections only after the lifespan — and therefore the pipeline build —
completes, so a healthy instance is never briefly unready.

Readiness deliberately does **not** probe Redis or Postgres. Both degrade
gracefully by design (cache misses, fail-open rate limiting, logged-out
sessions); failing readiness on them would restart an instance that is
serving correctly. `/v1/query` on an unready instance now answers a clean
**503 + Retry-After: 30** instead of a 500 and a page.

## 2. The write endpoint had no authentication

`POST /v1/documents` checked only `is_production`. Production was safe
(403), but any staging or development deployment exposed an
**unauthenticated endpoint that mutates the served index and spends
CPU** — a strictly easier target than the read path beside it, which has
required an API key since Stage 5.

Now behind the same `get_client_id` dependency as `/v1/query`. A write
path must never be the soft entrance.

## 3. No browser-facing security headers

The app serves HTML and carries a session cookie from Google sign-in, and
shipped with no CSP, no `nosniff`, no framing policy, no referrer policy.

Added in middleware — deliberately here rather than in a proxy this
project does not control:

```
Content-Security-Policy: default-src 'self'; script-src 'self';
  style-src 'self'; img-src 'self' https: data:; connect-src 'self';
  form-action 'self'; frame-ancestors 'none'; base-uri 'none';
  object-src 'none'
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Referrer-Policy: no-referrer
Permissions-Policy: geolocation=(), microphone=(), camera=()
Strict-Transport-Security: max-age=31536000; includeSubDomains   [prod only]
```

The policy has **no `'unsafe-inline'`**, which the frontend earned:
Stage 9.6's zero-dependency, no-inline-script, no-CDN choice pays off
here as a policy with no escape hatch. HSTS is production-only — sending
it from a local `http://` dev server would pin the browser to https for
`localhost` and break unrelated projects on that origin.

Swagger UI is CDN-served with an inline bootstrap and cannot live under
that policy, so `/docs` gets a **scoped exception** rather than the app's
policy being weakened for it. A test asserts the app's own CSP never
gains the CDN or `'unsafe-inline'`.

## Verified

- 236 tests pass, 19 skipped (8 new, one per defect and per policy claim).
- Real browser under the live CSP (Playwright, dark mode): stylesheet and
  fonts apply, suggested-question chip → **VERIFIED** cited answer,
  5-row retrieval trace, upload indexes a file, cache hit on repeat.
  **Zero CSP violations**; the only console error is the expected 401
  from `/auth/me` when signed out.
- `/docs` still renders Swagger with no console errors under its scoped
  policy.
