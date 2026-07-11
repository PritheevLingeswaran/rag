# Stage 6.5 Report — Concurrency & Shared-State Safety Audit

Date: 2026-07-11. Tests: `tests/core/test_concurrency_local.py` (7),
`tests/integration/test_concurrency.py` (9, at 50k-chunk target scale).
Correctness bar everywhere: EXACT accounting or bit-identical
serial-vs-concurrent equality — never mere absence of crashes.
Harness validity: `sys.setswitchinterval(1e-6)`; the lost-update
detector was validated by deliberately inserting a call boundary into an
RMW window — it then catches an **88.8% loss rate** (355,182 of 400,000
updates lost), so passes below are meaningful.

## The three headline questions

**FAISS index vs ingestion — the actual mechanism.** There is no lock
and no read-replica; the mechanism is Stage 2's **versioned immutable
swap**: ingestion builds a NEW version in a staging dir, fsyncs,
atomically renames, and never opens a served file for write; serving
loads a version at boot and the active pointer flips in Postgres
(partial unique index guarantees one active). Proof, not assumption:
`test_queries_stable_while_ingestion_writes_new_versions` hammers a
served 50k index from 4 threads while three new 50k versions are written
and GC'd beside it — every result bit-identical to serial throughout,
and version A hash-verifies afterward. **PASS.** Stated limitation: the
serving process picks up a newly activated version only on restart;
in-process hot-swap does not exist yet (it would need an atomic
pipeline-reference swap; recorded as future work, not pretended away).

**Redis — atomic or check-then-act?** Rate-limit counters, quota
counters, and daily quotas are single Lua scripts (atomic by Redis's
execution model) — proven under 8 concurrent threaded clients:
exactly 20 of 40 contended rate-limit attempts pass (never 21), and 200
contended `bounded_incr`s land exactly. Cache set/get is NOT
check-then-act-safe by design: two concurrent misses both compute and
both write (thundering herd) — accepted and bounded (admission caps
concurrent pipeline work at 2; values are idempotent; last-writer-wins;
tested that concurrent writers never produce a torn/unparseable value).
One check-then-act found and fixed: lazy `hasattr`-guarded Lua script
registration → moved to `__init__` (was benign — duplicate registration
— but the pattern is the disease).

## Full shared-state enumeration

| # | Shared state | Writers/Readers | Mechanism | Test @ scale | Result |
|---|---|---|---|---|---|
| 1 | FAISS served index | ingestion writes NEW versions only; N query threads read | immutable version dirs + atomic rename + DB-pointer activation | 8-thread search = serial, 50k; search-during-ingestion | **PASS** |
| 2 | BM25 posting arrays | read-only after `build()` (double-build raises) | immutability | 8-thread search = serial, 50k | **PASS** |
| 3 | ONNX sessions (embed + rerank) | N threads run inference on shared sessions | onnxruntime sessions are internally thread-safe; tokenizer configured once at init | 8-thread outputs bit-equal serial | **PASS** |
| 4 | `HybridPipeline` rerank-cost EWMA | N worker threads RMW | **AUDIT FIX**: was unlocked; now lock + update counter | 16×250 contended updates == exactly 4000; full-path runs == exact 2/run; value within sample range | **PASS (fixed)** |
| 5 | `get_settings()` singleton | all threads, first-call race | **AUDIT FIX**: `lru_cache` → double-checked lock | 64 concurrent calls | **FAIL→PASS** (before: 11 distinct instances; after: 1) |
| 6 | AdmissionController counters + EWMA | event-loop coroutines only | single-threaded asyncio by construction | 300-task stress: admitted+rejected==300, drains to 0/0, peak ≤ bound | **PASS** |
| 7 | QuotaGuard local fallback counters | N threads | `threading.Lock` in `_LocalCounters` + locked cooldown | 80 contended acquires → exactly 13 allowed | **PASS** |
| 8 | QuotaGuard Redis counters | N processes | atomic Lua | two-worker exactness (Stage 4.5) + threaded exactness here | **PASS** |
| 9 | Redis rate-limit / daily-quota counters | N processes | atomic Lua | exactly-limit-allowed under 8 threads | **PASS** |
| 10 | Redis response cache | N writers/readers | atomic single-value SET/GET; dog-pile accepted & bounded | concurrent writers: never torn, final value = one writer's intact payload | **PASS (accepted risk documented)** |
| 11 | redis-py connection pool | all threads | library-internal locking (exercised by every Redis test above under 8 threads) | via tests 8–10 | **PASS** |
| 12 | httpx.Client in GeminiClient | N worker threads | httpx thread-safe client (exercised, not assumed) | 16 threads × 25 calls over one client | **PASS** |
| 13 | Prometheus counters/histograms | all threads | prometheus_client internal locks (exercised) | 8×1000 incs == exactly 8000 | **PASS** |
| 14 | Postgres connections | ingestion CLI only, single-threaded | psycopg `Connection` is NOT thread-safe; **no concurrent use exists anywhere** (serving path has no DB connection) | n/a — nothing concurrent to test | **N/A**; recorded requirement: psycopg_pool is mandatory before any DB use enters the serving path |
| 15 | Structlog contextvars (request_id) | per-task/thread | contextvars isolation (copied into `to_thread` workers) | exercised by every API test + Stage 6 trace | **PASS** |

## Races found → before/after

**Race 1 (real, fixed): `get_settings()` lru_cache first-call race.**
Before (verbatim test failure): `assert 11 == 1` — 64 concurrent calls
constructed 11 distinct Settings instances (`functools.lru_cache` does
not lock around the wrapped call). Benign only while Settings is
immutable. After (double-checked locking): same test, 1 instance,
passes.

**Race 2 (latent, fixed): rerank-cost EWMA unlocked RMW.** The honest
empirical result cuts the other way: 400,000 contended updates through
the verbatim pre-fix logic lost **zero** updates on CPython 3.11 —
the statement is straight-line bytecode, and the eval breaker cannot
interrupt it mid-RMW. The same harness loses 88.8% when a call boundary
exists in the window, so this is a validated property, not a weak test.
Classification: de-facto safe under the current GIL, guaranteed by
nothing — free-threaded CPython or any refactor that puts a call inside
that statement breaks it silently. Fixed with a lock + exact update
counter; post-fix tests assert exact accounting (4000/4000 contended;
2-per-run through the real rerank path).

**Race 3 (benign, fixed): lazy Redis script registration**
(check-then-act on `hasattr`) → registered in `__init__`.

## Standing eval + suite

Suite: **159 passing** (16 new audit tests). Standing eval unchanged vs
baseline: P@1 0.9000 (+0.05), MRR@10 0.9417 (+0.0167), hallucination
0.0 — the locks added are on paths whose behavior the eval already
pinned down, and the EWMA fix is behavior-identical single-threaded.

Known accepted risks, restated: cache thundering herd (bounded by
admission=2, idempotent values); no in-process index hot-swap (restart
picks up activations); metrics are per-process (single worker by
design).
