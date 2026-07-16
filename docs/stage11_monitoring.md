# Stage 11 Report — Monitoring & Alerting

Date: 2026-07-12. Artifacts: `monitoring/alerts.yml` (rules),
`monitoring/alerts_test.yml` (promtool proof), `monitoring/grafana_dashboard.json`,
`/metrics` bearer-token auth. Account-side setup is the runbook below —
same boundary as Stages 8/8.5: Grafana Cloud / UptimeRobot accounts and
the "alert received by me" evidence are operator-side by nature.

## Threshold alerts — every number traces to a measurement

| Alert | Threshold | Why this number (source) |
|---|---|---|
| `UnhandledExceptions` | > 0 in 10m → **page** | Stage 5 hardening: handler catches all; acceptable background rate is zero |
| `ErrorRateHigh` | hard LLM failures > 5% of requests for 10m → **page** | measured normal = 0% (Stage 7.5: 416/416 served); 5% sustained = provider incident or config break |
| `P95LatencyBeyondEnvelope` | p95 > 4 s for 10m → **page** | measured admitted p95 at full capacity = 3.5 s (Stage 5, 0.1 CPU, queue depth 4); overload sheds via 503 rather than stretching latency, so p95 past the envelope means host degradation or regression |
| `QuotaThrottleSustained` | > 60 throttled answers/hour → **warn** | throttling is designed behavior (Stage 4.5); 60/h ≈ sustained pressure against the 13 RPM enforced budget — capacity planning signal, not incident |
| `ProviderQuotaRejections` | ≥ 3 provider 429s in 15m → **warn** | Stage 4.5 operator contract: 429s despite the guard = shared-project consumer or accounting slip |
| `SheddingSustained` | > 6 shed/min for 15m → **warn** | measured capacity = ~2 rps / 6 clients (Stage 5); sustained shedding = demand outgrew the tier |

**Machine-verified**: `promtool test rules monitoring/alerts_test.yml`
→ `SUCCESS` (promtool 3.5.0; 9 scenario tests: all six alerts proven to
FIRE past threshold; the three subtle ones — error rate, p95,
unhandled — also proven QUIET below it). This is what "the alert works"
means before an account exists; the Grafana contact-point test is the
last-mile delivery check.

## /metrics auth (prerequisite for cloud scraping)

Public metrics are recon material, so scraping a public URL required
auth first: `METRICS_TOKEN` env → `/metrics` demands
`Authorization: Bearer <token>` (constant-time compare; 401 otherwise).
Grafana Cloud's *Metrics Endpoint* integration supports exactly this.
Tested (`test_metrics_token_auth_when_configured`). Unset = open, for
local/dev. **LIVE 2026-07-16**: token set on the Render service via
API and verified on production (bare → 401, bearer → 200); the
Grafana scrape config just needs the token from the operator's `.env`.

## IMPLEMENTED 2026-07-16 — account-free monitoring core

The uptime/keepalive/paging core now runs on infrastructure the
project already had, no new accounts:

| Concern | Implementation | Contact point |
|---|---|---|
| Uptime + keyword check | `.github/workflows/healthcheck.yml`: GET `/health` every 10 min, requires `"status":"ok"` | ntfy topic (phone/email), repeats every 10 min while down |
| Keepalive (Stage 8 item) | same workflow — 10-min pings beat the 15-min spin-down | — |
| Unhandled exceptions → page | `app/main.py` exception handler fires AlertManager (deduped 1/day; no exception detail leaves the box) | ntfy topic |
| Quota thresholds (Gemini RPD, Upstash commands, Neon storage) | in-app breakers, 80% alerts (Stage 7.7, pre-existing) | ntfy topic |

Honest limits: GitHub cron can lag minutes under load and schedules
pause after 60 days of repo inactivity; in-app alerts can't fire if
the process is down (that's what the external workflow is for). p95 /
error-rate / shedding trend alerts still want real metric evaluation —
that remains the Grafana Cloud runbook below, now optional rather than
blocking.

## Operator runbook (~10 min, after Stage 8's deploy — now OPTIONAL, for dashboards + trend alerts)

1. **Grafana Cloud** (free tier: 10k series, 14-day retention — far
   above our ~200 series): create stack →
   **Connections → Metrics Endpoint** → scrape URL
   `https://<app>/metrics`, auth type *Bearer*, paste the value you set
   as `METRICS_TOKEN` on Render → scrape interval 60 s.
2. **Dashboard**: Dashboards → Import → upload
   `monitoring/grafana_dashboard.json` → pick the datasource.
3. **Alerts**: Alerting → Alert rules → create the six rules from
   `monitoring/alerts.yml` (copy the PromQL verbatim; `for:` and
   severity label as listed). Contact point: **Webhook** →
   `https://ntfy.sh/<your-topic>` (same topic as the Stage 7.7 quota
   alerts — one alert channel, phone + email via ntfy).
4. **Test the threshold alert** (definition-of-done evidence): open the
   contact point → **Test** → notification arrives on your ntfy topic;
   screenshot it. For a *real* threshold firing: run
   `python scripts/load_test_api.py --url https://<app> --api-key <key>
   --concurrency 16 --requests 40` — 16 closed-loop workers exceed the
   measured capacity, `SheddingSustained` crosses within ~15 min.
5. **UptimeRobot** (free: 50 monitors, 5-min interval): add HTTP(s)
   monitor → `https://<app>/health`, keyword `ok` (catches a 200 that
   isn't healthy) → alert contacts: your email + webhook
   `https://ntfy.sh/<your-topic>`. This monitor doubles as the Stage 8
   keepalive (5-min interval beats the 10-min requirement).
6. **Test the uptime alert**: Render dashboard → Suspend service for
   ~6 min → UptimeRobot fires "down" to email+ntfy; resume → "up"
   notification; screenshot both.

## Honest notes

- Grafana Cloud scrapes at best every 60 s and evaluates rules ~1 min;
  with `for: 10m` windows, worst-case detection ≈ 12 min. Fine for a
  free-tier single-instance service; not an SLA machine.
- Metrics are per-process and reset on deploy/restart (Stage 6 note);
  `increase()`/`rate()` handle counter resets natively, so the rules
  survive restarts without false pages.
- Render free sleeps on idle; the UptimeRobot 5-min ping keeps it warm,
  which also means "down" alerts are real downs, not sleeps.
- The `up == 0` style target-down alert is deliberately left to
  UptimeRobot (external view beats self-report for uptime).
