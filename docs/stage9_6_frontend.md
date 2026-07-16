# Stage 9.6 Report — Production Frontend with Google Login

Date: 2026-07-16. Implements docs/stage9_5_frontend_auth.md exactly:
same-origin static frontend, server-side OAuth code flow, opaque
HttpOnly session cookie, Redis sessions, `0002_users.sql`, per-user
limits under the unchanged global quota walls, no CORS anywhere.

## What shipped

| Piece | Where | Notes |
|---|---|---|
| Frontend (no framework, no build step) | `frontend/{index.html,style.css,app.js}` | served same-origin: `/` = app, assets under `/app/` — routes always win over the static mount |
| Auth routes | `app/api/auth.py` | `/auth/google/{login,callback}` (code flow, `state` CSRF cookie), `/auth/me`, `POST /auth/logout` |
| id_token handling | `verify_id_token_claims()` | aud/iss/exp/sub checks (tested x5); signature relies on the direct-TLS token-endpoint channel per OIDC 3.1.3.7 — no JWT dependency; Google tokens discarded after verification |
| Sessions | `RedisStore.session_{set,get,delete}` | opaque 256-bit id, profile snapshot inside (per-request path never touches Postgres), fixed 7-day TTL; any Redis failure = logged out, never access |
| Users | `migrations/0002_users.sql` | applied to Neon 2026-07-16 (`schema_migrations` records 0001+0002); DB touched at login only |
| Identity | `get_client_id` session branch | `user:{id}` alongside `key:{digest}` / dev `anon:{ip}`; API-key surface byte-for-byte unchanged when a key header is present |
| Per-user limits | `app/api/query.py` | `user:*` → 10/min + 50/day (Settings); keys keep 30/500; global Gemini→Groq walls unchanged (Stage 9.5 burst math) |
| Proxy correctness | `Dockerfile` CMD | `--proxy-headers --forwarded-allow-ips '*'` so `request.base_url` is https behind Render's edge (exact-match OAuth redirect URIs) |
| Privacy | `PRIVACY.md` | amended in THIS commit (policy-before-behavior rule): sign-in section, storage table rows, deletion path; core "queries never stored / never linked" invariant restated and untouched |

## The UI is honest about every backend state

Each documented state from Stages 4/4.5/5/7.7 has an explicit rendering
(`app.js` `renderResult`/`startCountdown`/`showError` — one rendering
path for live and demo traffic):

- `ok` → "✓ every sentence verified against the cited sources" + Sources list
- `ok_partial_rejected` → "◐ unverifiable sentences were removed" (never hidden)
- `ok_no_answer` / `no_results` → explicit empty states
- every `degraded_*` → "⚠ Retrieval-only answer — <reason>" with a
  plain-language reason per status; `degraded_quota_throttled`
  additionally starts a countdown from the server's `retry_after_s`
- HTTP 429 (rate limit vs daily quota distinguished) and 503 (shed) →
  wait panel with a live countdown from the SERVER's Retry-After, submit
  disabled until it reaches zero
- 401 → logged-out landing with "session expired" message
- network failure / 5xx → error panel with retry button and `request_id`
- loading → elapsed-seconds counter; after 5 s it tells the truth about
  free-tier cold starts (~2 min worst case), instead of a lying spinner
- `cached: true` → "Served from cache (≤1 h old)"

Accessibility/responsiveness: skip link, labelled input, `aria-live`
result region, `role=alert/status` panels, text+symbol badges (never
color alone), `:focus-visible` outlines, `prefers-reduced-motion`
honored, single column that works at 320 px (input row stacks).
No console leaks: `window.onerror`/`unhandledrejection` render the
error panel — a blank screen is defined as a bug.

**State gallery**: `/#demo:ok`, `#demo:partial`, `#demo:throttled`,
`#demo:degraded`, `#demo:ratelimited`, `#demo:shed`, `#demo:error`,
`#demo:loading` render recorded real payloads through the same
rendering functions — the four degradation states are demonstrable in
any browser in under a minute without forcing the backend into them.

## Static-host tradeoff (stage requirement: state it)

Same-origin from the existing service, not a separate static host:
zero new infrastructure, zero CORS, `SameSite=Lax` cookies just work,
one pipeline. Cost: frontend deploys ride backend deploys and share
instance bandwidth (trivial at these asset sizes). A separate free
static host would buy independent frontend releases at the price of
real CORS + `SameSite=None` cookies + a second origin — rejected at
this scale (full reasoning in the Stage 9.5 doc).

## Verified

- Suite: 50 api tests incl. 14 new (claims x5, session identity → user
  limits, anon fallback, garbage cookie = logged out, /auth/me, logout,
  unconfigured login → 503 naming the missing config, forged state →
  400, frontend at `/` + assets at `/app/*`); full suite green (see CI).
- Local live checks: `/` serves the app, `/auth/me` → 401 logged out,
  `/auth/google/login` → 503 with exact missing-config message,
  anonymous dev query unchanged.

## Remaining for the operator (the two things code cannot mint)

1. **Google OAuth client** (runbook step 7b, docs/stage8_deploy.md):
   create in Google Cloud Console with both redirect URIs, then set
   `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET` on the service (env API or
   dashboard). Until then: frontend serves, sign-in button returns a
   clear 503, API keys unaffected.
2. **Definition-of-done evidence**: the live login → query → logout
   cycle and the four degradation-state screenshots need a browser +
   the OAuth client. The state-gallery URLs make the degradation half
   a 60-second job; the login half needs step 1 first.
