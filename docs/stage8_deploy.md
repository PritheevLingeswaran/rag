# Stage 8 — Live Deployment (Render free tier)

Status: **DEPLOYED AND DRILLED 2026-07-13.** Live at
https://ragp-pwf2.onrender.com — HTTPS `/health` 200 through Render's
health gate, auth enforced (401 keyless), `/v1/query` serving cited
answers. Rollback drill performed on the live service (timestamps
below). Creating accounts and handling credentials was deliberately
not automated.

## What is verified (evidence in this repo / report)

| Property | Evidence |
|---|---|
| Multi-stage image, models baked at build (no HF download at cold start, `HF_HUB_OFFLINE=1`) | `Dockerfile` |
| Non-root | `whoami` → `ragp`, uid 999 |
| No secrets in image | all secrets read from env at runtime; `.dockerignore` excludes `.env`; keyless `ENVIRONMENT=production` boot **refuses to serve** (`ConfigurationError: API_KEYS must be set in production`) — drilled |
| Ready-means-ready health check | uvicorn accepts connections only after lifespan completes (models loaded, index built, pipeline warmed, rerank EWMA seeded) → `/health` 200 = genuinely ready; `healthCheckPath: /health` in `render.yaml` |
| Cold start at 0.1 CPU / 512MB (production image, production env) | **~46 s** container-start → `/health` 200 (= fully ready); models baked so zero network at boot (`HF_HUB_OFFLINE=1`) |
| First request after ready | 200 in **451 ms**, correct cited answer (`raft::c1`), through auth + Redis |
| Auth enforced in prod container | 401 without `x-api-key` — drilled |
| Memory in prod container | 156 MiB idle post-boot (60-chunk corpus; 50k-scale projection: 394 MB, Stage 4 Linux measurement) |

## Cold start / sleep behavior — what a user experiences

Render free spins the service down after **15 min without traffic**
(Stage 2.5, official). First request after idle:

    Render platform spin-up (~30–60 s, their number)
  + app cold start (**measured: ~46 s** at 0.1 CPU with the production
    image — model sessions + 60-chunk index build + warmup; zero
    network, models baked)
  ≈ **~1.5–2 minutes worst case**, during which the request hangs or
    times out client-side.

Verdict: **not acceptable for a live service; mitigation required.**
Mitigation: external keepalive ping to `/health` every 10 minutes
(UptimeRobot / cron-job.org, both free). One always-on service = 744 h
< the 750 h monthly budget, so keepalive fits by construction. ToS
note from Stage 2.5 stands: Render documents spin-down as behavior,
not pinging as a violation — re-check on deploy day. Residual risk:
free-tier restarts/deploys still cold-start; users in that window get
the wait. Acceptable at this stage; the fix that removes it is money.

## Operator runbook (account-side, ~20 min)

1. **GitHub**: push this repo (`git remote add origin … && git push`).
2. **Neon** (free): create project → copy the **pooled** connection
   string. Run migrations once from your machine:
   `DATABASE_URL=... python -m app.ingest.cli migrate`
3. **Upstash** (free): create Redis → copy `rediss://` URL.
4. **ntfy.sh** (no signup): pick a secret topic name; your
   `ALERT_WEBHOOK_URL` is `https://ntfy.sh/<topic>`; subscribe on phone.
5. **Render**: New → Blueprint → point at the repo (`render.yaml` is
   picked up). In the service's **Environment** tab (their secret
   manager) set: `API_KEYS` (generate: `python -c "import secrets;
   print(secrets.token_urlsafe(32))"`), `REDIS_URL`, `DATABASE_URL`,
   `ALERT_WEBHOOK_URL`, optionally `GEMINI_API_KEY` (absent ⇒ explicit
   `degraded_no_llm` serving) and `CORS_ORIGINS`.
6. **Domain + HTTPS**: the default `https://ragp.onrender.com` is real
   HTTPS out of the box. Custom subdomain: service → Settings → Custom
   Domains → add `rag.<yourdomain>` → create the shown CNAME at your
   DNS → Render provisions TLS automatically.
7. **Keepalive**: UptimeRobot monitor, GET `https://<url>/health`,
   interval 10 min. (Doubles as your uptime alerting.)
8. **Verify**: `curl https://<url>/health` → 200; a `/v1/query` with
   your key; check the Gemini numbers in AI Studio and update
   `configs/free_tier_limits.json` if they differ (Stage 2.5 action
   item).

## Rollback approach + drill

Render keeps every previous deploy's image. Rollback = service →
Deploys → previous deploy → **Rollback** (or
`render deploys rollback <service>` via CLI): the old image starts,
health check must pass, traffic flips — the working version is never
lost because images are immutable and the DB schema only changes by
forward migration (Stage 2 policy). Index versions are likewise
immutable with a pointer flip (Stage 2), so app rollback never corrupts
retrieval state.

**Drill (do once, day of deploy):** push a trivial commit (e.g. bump
`app/__init__.py` version) → wait for deploy → dashboard Rollback to
the prior deploy → confirm `/health` 200 and `/openapi.json` shows the
prior version string. Record the timestamps in this file.

**Drill record (performed 2026-07-13, all times UTC):**

| Step | Time | Evidence |
|---|---|---|
| Trivial commit pushed (`3781272`, 0.1.0 → 0.1.1) | 10:05:06Z | `git push` |
| New deploy live (through Render health gate) | 10:06:21Z | `/openapi.json` → 0.1.1; `/health` 200 |
| Dashboard Rollback clicked (to `b62789b` / 0.1.0) | ~10:13Z | operator |
| Rollback live, version reverted | 10:13:08Z | `/openapi.json` → 0.1.0; `/health` 200 `{"status":"ok","version":"0.1.0","environment":"production"}` |

Deploy of the trivial commit took ~75 s (Docker layer cache: only the
source `COPY` layer rebuilt, baked models stayed cached); rollback
flip took under a minute of instance boot. Pushing this record
re-deploys `master` (autoDeploy), returning the service to 0.1.1.

## Remaining for the operator (cannot be done from this machine)

Done 2026-07-13: accounts, secrets (Render env tab), deploy, HTTPS
health verification, rollback drill. Still open: keepalive monitor
(step 7, UptimeRobot GET /health every 10 min) if not yet created,
and the Gemini path is serving `degraded_llm_malformed` — the LLM
call is not producing parseable output with the current key; the
extractive fallback is serving correctly meanwhile. Investigate
post-Stage 8.

Note (migrations): port 5432 was blocked on the operator network, so
`0001_init.sql` was applied via Neon's SQL-over-HTTP endpoint in one
transaction, including the `schema_migrations` bookkeeping row —
`python -m app.ingest.cli migrate` sees it as applied.
