# Stage 9.5 Report — Frontend + Google Auth: Architecture Decisions

Date: 2026-07-16. Status: **decisions only — nothing implemented**, per
the stage's definition of done. Each point states the decision, the
alternatives weighed, and what tips it. The recurring theme: the
Stage 2–8 architecture was built identity-agnostic, so most of these
decisions are "reuse the existing mechanism with a new key prefix,"
not new machinery.

## 0. One prior decision that shapes everything: serve the frontend same-origin

The SPA/static frontend is served **by the existing FastAPI service**
(a `/app` mount of static files in the same container). Not a separate
static host.

- Why: zero new infrastructure, no CORS in production at all (point 4
  collapses), no cross-site cookie problems (`SameSite=Lax` just
  works), one deploy pipeline, one TLS cert, one origin.
- Tradeoff accepted: frontend and backend deploy together (fine: one
  repo, CI already gates both), and static bytes consume the same
  instance's bandwidth (fine: 100 GB/month free tier vs a few KB of
  assets).
- Rejected alternative: Cloudflare Pages / Render static site (also
  free) — buys independent frontend deploys at the cost of real CORS,
  `SameSite=None` cookies, and a second origin to secure. Revisit only
  if the frontend grows its own release cadence.

## 1. How web users and API keys coexist

**Decision: parallel identity branches in the existing auth dependency;
the browser never sees any key or token.**

- Programmatic clients: unchanged. `x-api-key` header → Stage 5 branch
  in `get_client_id` → identity `key:{sha256[:12]}`.
- Web users: Google OAuth **authorization-code flow, server-side**
  (`/auth/google/login` → Google → `/auth/google/callback`). The
  backend (confidential client) exchanges the code, verifies the
  `id_token`, upserts the user row, creates a server-side session, and
  sets an **opaque session id in an `HttpOnly; Secure; SameSite=Lax`
  cookie**. `get_client_id` gains a session branch → identity
  `user:{user_id}`.
- Google's access/refresh tokens are **discarded immediately after
  id_token verification** — we need identity, not Google API access.
  Nothing to leak, rotate, or store.
- No internal API key is minted per user (rejected: a per-user raw key
  in the browser is the thing this design exists to avoid; the session
  IS the scoped credential, revocable server-side).
- CSRF: `state` parameter on the OAuth flow; `SameSite=Lax` blocks
  cross-site POSTs to `/v1/query`; CORS stays deny-by-default. No
  token-in-JS anywhere, so XSS cannot exfiltrate a credential.

## 2. Database schema

**Decision: one new forward migration, `0002_users.sql`. Sessions do
NOT get a table (point 6 puts them in Redis).**

```sql
CREATE TABLE users (
    user_id       TEXT PRIMARY KEY,          -- internal "u_<random>"; NOT the google sub
    google_sub    TEXT UNIQUE NOT NULL,      -- stable Google account id
    email         TEXT NOT NULL,
    display_name  TEXT,
    avatar_url    TEXT,                      -- URL only; the image is never fetched/stored
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

- Extends Stage 2 without breaking anything: the migration runner
  (`app/storage/db.py`) applies `*.sql` in filename order and records
  them in `schema_migrations` — append-only by construction; `0001`
  is never touched. Rollback of schema remains "a new forward
  migration" (existing policy).
- Internal `user_id` (not `google_sub`) is the identity used in Redis
  keys and logs, so Google's identifier never spreads through the
  system.
- **Deliberately absent**: any FK or column linking `users` to
  `query_logs`. The Stage 8.9 guard test (query-text never persisted,
  `QueryLogRepo` unwired) continues to enforce this; point 7 restates
  it as policy.
- Serving path stays DB-free per request: Postgres is touched at
  login/signup only (Neon's ~300 ms wake is acceptable there, and the
  free-tier compute budget stays safe — the per-request path never
  wakes it).

## 3. Per-user rate limits and the sign-up-burst problem

**Decision: reuse the existing identity-keyed Redis fixed-window
mechanism with LOWER per-user defaults; the global Stage 4.5/7.7
guards remain the aggregate wall. No new mechanism.**

- The Stage 5 limiter is already identity-agnostic (`client_id`
  string). Web users get `user:{user_id}` counters with defaults
  below API keys — proposed `10/min` and `50/day` (vs 30/500) —
  because a Google sign-up is free for an attacker: any Google
  account = a fresh quota grant, so individual grants must be small.
  Both values become Settings fields (measured adjustment later, like
  every other limit in this project).
- **The burst math, explicitly** (the Stage 7.7 question): per-user
  quotas bound *individual* consumption; they do NOT multiply LLM
  capacity. The aggregate wall is unchanged: Gemini QuotaGuard
  13 RPM / 450 RPD enforced → Groq fallback 27 RPM / 12,960 RPD
  enforced → extractive answers. So 100 sign-ups in an hour behave as:
  first ~450 LLM calls that Pacific day served by Gemini, next
  ~12,960 by Groq, everyone after that gets the labeled
  `degraded_quota_throttled` extractive path with `retry_after_s` —
  degradation on our terms, exactly the Stage 4.5 contract, now
  benefiting from the Stage "make it production" fallback chain. The
  80% alerts (Stage 7.7 → ntfy) fire long before the wall.
- Auth endpoints themselves get the existing per-IP fixed-window
  limit (`anon:{ip}`) so the login/callback routes can't be hammered.
- Upstash command budget check (Stage 2.5 math): sessions add ~1
  command/request (point 6) on top of the current ~4 → ≈5/request →
  ~2,900 authenticated requests/day inside the enforced 14,516
  commands. The Stage 7.7 command-budget breaker already meters this
  and alerts at 80%; on budget-open, Redis bypass logs web users out
  (session lookups return None) rather than crashing — stated
  honestly rather than hidden.

## 4. CORS policy, exactly

**Decision: no CORS in production. Same-origin (point 0) makes CORS
headers unnecessary — `CORS_ORIGINS` stays UNSET, which the existing
code treats as deny-by-default (no CORS middleware mounted at all).**

- Production: frontend and API share `https://ragp-pwf2.onrender.com`
  (or the future custom domain). No cross-origin requests exist, so
  no origin is ever allowed. The existing boot-refusal on
  `CORS_ORIGINS='*'` in production stays as the tested backstop.
- Local dev, two supported modes:
  1. Same as prod (default): uvicorn serves API + built frontend on
     `http://localhost:8000` — no CORS needed, closest to production.
  2. Frontend dev server (e.g. Vite on `:5173`): set
     `CORS_ORIGINS=http://localhost:5173` in the local gitignored
     `.env` only. Exact origin, never `*`; `ENVIRONMENT=development`
     is already required for that file's values to matter.
- If the frontend ever moves cross-origin (rejected alternative in
  point 0), the policy is: exactly one production origin, set via
  Render env, and cookies move to `SameSite=None; Secure` — recorded
  now so the future change is a checklist, not a redesign.

## 5. OAuth secret handling

**Decision: identical standard to every other secret in this project.**

- `GOOGLE_CLIENT_ID` + `GOOGLE_CLIENT_SECRET`: set in Render's env
  (dashboard or API), `sync: false` entries in `render.yaml`, staged
  commented-out in the local gitignored `.env`, never committed —
  the same lifecycle as `GEMINI_API_KEY`/`GROQ_API_KEY`/
  `METRICS_TOKEN`. Deploy runbook (`docs/stage8_deploy.md`) gains one
  step: create the OAuth client in Google Cloud Console with BOTH
  authorized redirect URIs (`https://<prod>/auth/google/callback`,
  `http://localhost:8000/auth/google/callback`).
- One OAuth client shared by dev and prod (tradeoff: simpler, vs a
  separate dev client isolating the prod secret from laptops — at one
  operator, simplicity wins; revisit if a team forms).
- **No session-signing secret exists at all**: opaque random session
  ids (256-bit) in Redis need no signature — one fewer secret to
  rotate, a direct consequence of the point 6 decision.

## 6. Session storage: Redis vs stateless JWT

**Decision: Redis-backed opaque sessions (`ragp:sess:{id}` →
`{user_id}`), fixed 7-day TTL, no sliding refresh.**

| | Redis sessions | Stateless JWT |
|---|---|---|
| Upstash commands | +1 GET/request, +1 SET/login (≈5/request total; ~2,900 req/day in budget) | 0 |
| Revocation (logout, abuse ban, compromised account) | instant `DEL` | impossible without a denylist — which needs Redis anyway, forfeiting the only advantage |
| Secrets | none | signing key + rotation story |
| Failure mode | Redis down/budget-open → web users logged out (API keys unaffected); alerts at 80% | none, but bans don't work |
| Cookie size | 32 bytes | ~1 KB |

The command-budget cost is real but affordable (the math above), and
point 3 *requires* the ability to cut a user off — an abuse ban that
takes effect "when the JWT expires" is not a ban. Revocation beats
zero-command purity. No sliding TTL: re-login once a week is the
cheapest acceptable UX and saves an `EXPIRE` per request.

## 7. Privacy policy update (Stage 8.9 amendment)

**Decision: PRIVACY.md is amended in the SAME commit that lands the
auth code — policy before behavior, the standing Stage 8.9 rule.**
The amendment, decided now:

- **What's collected at sign-in**: Google account id (`sub`), email,
  display name, avatar URL — from the verified id_token only.
  Google's tokens are discarded at login; we never gain or retain
  access to the user's Google data beyond those profile claims.
- **Retention**: for the life of the account. Sign-in is optional —
  API-key and (dev-mode) anonymous usage keep working with zero
  profile data.
- **Deletion**: on request (repo issue tracker, the existing contact
  channel), everything keyed to the account is removed: the `users`
  row and any live sessions/counters (which self-expire ≤ 25 h / ≤ 7
  days anyway). The current policy line "there is nothing to delete"
  becomes "accounts are the one thing we store about you, and you can
  have them deleted."
- **The load-bearing invariant, unchanged**: queries are never stored,
  and nothing links query content to identity. `query_logs` stays
  unwired (guard test continues to enforce this); per-user rate
  counters carry the opaque internal `user_id`, TTL-bounded, never
  query text. Sign-in changes *who we know you are*, not *what we
  record about what you ask*.

## What implementation will look like (sizing, not commitment)

`0002_users.sql`; `/auth/google/{login,callback,logout}` routes; a
session branch in `get_client_id`; `RedisStore.session_{set,get,del}`;
two Settings fields for per-user limits; static mount for the
frontend; PRIVACY.md amendment; tests mirroring the Stage 5 auth
suite. No changes to retrieval, generation, quota guards, admission,
or CI.

Suite impact of this stage: none (no code). Stopping for review, per
the definition of done.
