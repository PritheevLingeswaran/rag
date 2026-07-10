# Free-Tier Infrastructure Selection (Stage 2.5)

Decision date: **2026-07-10**. Machine-readable limits live in
[`configs/free_tier_limits.json`](../configs/free_tier_limits.json) —
rate-limiting and degradation logic in later stages reads budgets from that
file, never hardcodes them. Re-verify all numbers before deploy day.

## Selections

| Component | Choice | Rejected alternatives (why) |
|---|---|---|
| App hosting | **Render** free web service | Fly.io: free tier discontinued (trial credit only, ~$2-5/mo minimum). Railway: $5 one-time 30-day trial, then $1/mo credit — cannot run an always-on service. |
| Postgres | **Neon** free plan | Supabase: pauses projects after **7 days** of API inactivity and requires a **manual dashboard unpause** — fatal for a low-traffic live service. Neon autosuspends after 5 min but wakes **automatically in ~hundreds of ms**. Render's free Postgres **expires after 30 days**: disqualified. |
| Redis | **Upstash** free | Render free Redis (25 MB) is tied to the same workspace hours budget and is not serverless; Upstash 256 MB + 500K commands/month is strictly better for our pattern. |
| Vector store | **FAISS in-process** (already built, Stage 2) | Hosted vector DBs (Pinecone/Qdrant free tiers) add a network hop per query, a second quota system to manage, and another cold-start surface. See tradeoff below. |
| Index artifact storage | **Cloudflare R2** free (10 GB) | Needed because Render free disk is **ephemeral** — index files vanish on restart. No-card fallbacks: Backblaze B2 (10 GB) or Supabase Storage (1 GB). |
| LLM | **Gemini API** free tier (gemini-3-flash), **Groq** free as fallback provider | OpenRouter free models: 50 req/day, too low as primary. Groq's llama-3.1-8b at 14,400 RPD is the overflow/degradation path. |

## The hard-limits table

Every number the system is now bound by. "Official" = provider docs fetched
2026-07-10; "secondary" = consistent recent third-party sources where the
provider no longer publishes exact numbers publicly.

| Limit | Value | On breach | Source |
|---|---|---|---|
| Render RAM / CPU | 512 MB / 0.1 CPU | OOM kill / throttle | secondary |
| Render idle spin-down | 15 min without inbound traffic | next request waits for spin-up | official |
| Render spin-up (cold start) | ~30–60 s | user's first request hangs that long (or times out client-side) | official |
| Render instance hours | 750 h/month/workspace | ALL free services suspended until next month | official |
| Render bandwidth | 100 GB/month | suspended for rest of month (no card on file) | official |
| Render disk | ephemeral | index lost on every restart/deploy | official |
| Neon storage | 0.5 GB/project | INSERT/UPDATE/DELETE **fail**; data kept | official |
| Neon compute | 100 CU-h/month (~400 h at 0.25 CU) | compute suspended — **DB unreachable until next month** | official |
| Neon autosuspend | after 5 min idle (mandatory) | ~100s-of-ms wake on next query | official |
| Neon connections | ~104 direct / 10,000 pooled (PgBouncer) | connection refused past cap → always use pooled URL | secondary |
| Neon egress | 5 GB/month | throttle/overage | official |
| Upstash data | 256 MB | writes blocked | official |
| Upstash commands | 500K/month (~16.1K/day budget) | database throttled (rate-limited); email notice; paid auto-upgrade only if opted in | official |
| Upstash max request | 10 MB | request rejected | official |
| R2 storage / writes / reads | 10 GB / 1M Class A / 10M Class B per month | billed overage (card required for R2) | official |
| Gemini 3 Flash | **10 RPM / 250K TPM / 1,500 RPD** | HTTP 429 `RESOURCE_EXHAUSTED` + retry-after | secondary¹ |
| Gemini 2.5 Flash (fallback) | 10 RPM / 250K TPM / **250 RPD** | same | secondary¹ |
| Gemini 2.5 Flash-Lite (fallback) | 15 RPM / 250K TPM / 1,000 RPD | same | secondary¹ |
| Gemini quota scope | per Google Cloud **project**, resets midnight Pacific | extra API keys do NOT add quota | official |
| Groq llama-3.1-8b-instant | 30 RPM / 6K TPM / 14,400 RPD | HTTP 429 + retry-after | official/secondary |

¹ Google now shows exact per-project quotas only in the AI Studio dashboard
(`aistudio.google.com/rate-limit`); these are the consistently reported
mid-2026 free-tier values. **Action item: confirm in AI Studio the day the
key is created and update `free_tier_limits.json`.** The 429-handling code
must treat dashboard numbers as advisory and live 429s as ground truth
regardless.

## What these numbers force on the design

1. **Gemini RPD (1,500/day) is the system's binding constraint** — roughly
   1 generation/minute sustained. Consequences: response caching isn't an
   optimization, it's capacity (cache hits don't spend RPD); the API must
   enforce a **global** LLM budget (Redis counter) at ~90% of quota
   (10 RPM → enforce 8; 1,500 RPD → enforce ~1,350) so we degrade on our
   terms (retrieval-only answers with citations) instead of Google's 429.
2. **Cold starts are real UX**: idle→request = Render spin-up (30–60 s) +
   Neon wake (~0.5 s) + index download from R2. Mitigations: external
   uptime pinger every ~10 min keeps Render warm and stays far inside the
   750 h budget (one service running 24/7 = 744 h); `/health` ping touches
   Postgres to keep Neon's 5-min window irrelevant during active hours.
   Honest note: keepalive pinging a free service is a ToS gray area on some
   platforms; Render's docs describe spin-down as a behavior, not pinging
   as a violation, but this gets re-checked before launch.
3. **Upstash budget arithmetic**: ~16.1K commands/day; at ~3 commands per
   request (rate-limit Lua + cache GET + occasional SET), that sustains
   ~5K requests/day — comfortably above what Gemini RPD lets us generate
   anyway. Redis is not the bottleneck; still, the rate-limit key TTLs and
   cache TTLs must be tuned so idle traffic doesn't burn commands.
4. **Neon CU-hours require autosuspend to work in our favor**: bursty
   dozens-of-users traffic with 5-min suspend windows fits easily in 100
   CU-h; an always-awake DB does not (182 CU-h). The connection layer must
   use the **pooled** connection string and tolerate first-query wake
   latency. A monthly CU-hours check belongs in the ops runbook — compute
   exhaustion mid-month means a hard outage.
5. **512 MB RAM bounds the vector index**: 50K chunks × 384-dim float32 ≈
   74 MB — fine. It also means **no local embedding/reranking models in
   the serving process**; embedding at query time must come from an API or
   stay lexical (BM25). This constraint lands on Stage 3's design.
6. **Ephemeral disk means index artifacts live in R2** and are pulled at
   boot (adds seconds to cold start; 10 GB holds ~130 index versions at
   74 MB — gc keeps 3).

## FAISS in-process vs hosted vector DB — the tradeoff on record

Kept: FAISS as a library inside the API process.
**Wins**: zero extra service/quota/cold-start, no network hop per query
(~ms-level retrieval), exact search at our scale, index versioning already
built and tested (Stage 2).
**Costs we accept**: index must fit in app RAM (74 MB at target scale —
fits); index rebuild/versioning is our code, not a managed service; no
scaling beyond one process without adding artifact-sync machinery; every
app restart re-downloads the index (ephemeral disk).
**Trigger to revisit**: corpus beyond ~200K chunks, multi-instance serving,
or RAM pressure from other components.
