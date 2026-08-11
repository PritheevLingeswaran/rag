# Hybrid RAG Platform

Production-grade hybrid RAG system (FastAPI + FAISS + BM25 + cross-encoder
reranking + Prometheus + PostgreSQL + Redis) built to run live on free-tier
infrastructure. Design target: 10k–50k documents, dozens–low hundreds of
concurrent users, p95 retrieval+generation latency under 500ms–1s.

## The metric rule

**No metric claim exists in this project unless `eval/run_eval.py` produced
it.** Every stage re-runs the exact same harness and reports the diff against
the committed baseline:

```
python eval/run_eval.py --baseline eval/results/baseline.json --tag stageN
```

Runs are seeded (`RANDOM_SEED = 42`). Quality metrics (P@1, MRR@10,
hallucination rate) are bit-for-bit reproducible; latency numbers are
reproducible in *methodology* (3 warmups, median of 5 timed repeats per
query, p50/p95 over per-query medians) but naturally vary by machine.

## Versioned eval artifacts

- `data/corpus_v1.jsonl` — 30 documents, chunked deterministically
  (1 paragraph = 1 chunk, IDs `{doc_id}::c{n}`). The chunking rule is part
  of the versioned contract: changing it requires `corpus_v2`, not an edit.
- `eval/dataset_v1.jsonl` — 20 queries with gold chunk IDs and expected
  answers. Both files' SHA-256 hashes are recorded in every results file.
- `eval/results/baseline.json` — the committed Stage 0 baseline. Timestamped
  per-run files live alongside it.

## Metric definitions (harness contract v1.0)

| Metric | Definition |
|---|---|
| P@1 | top-1 retrieved chunk is in the gold set |
| MRR@10 | reciprocal rank of first gold chunk in top 10, else 0 |
| Hallucination rate | fraction of answers with ≥1 sentence whose content-token grounding in retrieved context is < 0.7 |
| Unsupported-token rate | fraction of answer content tokens absent from retrieved context |
| Latency p50/p95 | over per-query median of 5 timed runs (after 3 warmups) |

The hallucination metric is a **deterministic lexical proxy**, not an LLM
judge. It cannot detect fluent-but-wrong paraphrase, but it is reproducible,
free, and strict against verbatim fabrication. When abstractive generation
lands, an LLM-judge metric can be added *alongside* it (never replacing it)
under a new harness version.

### That proxy is measured, not assumed (Stage 10)

Scored against RAGBench (`galileo-ai/ragbench`), which ships support
labels and the scores three LLM judges gave the same rows:

| method | best F1 | AUC | cost per evaluation |
|---|---|---|---|
| **this project's lexical proxy** | 0.383 | **0.639** | none |
| RAGAS faithfulness | 0.302 | 0.567 | LLM calls |
| TruLens groundedness | 0.384 | 0.635 | LLM calls |
| GPT adherence judge | 0.389 | 0.628 | LLM calls |

All four land in the same band — but the per-domain split is the real
result. The proxy **beats every LLM judge on technical documentation**
(techqa F1 0.758 vs 0.696–0.753; emanual 0.483 vs 0.261–0.400) and
**collapses on multi-hop web QA and finance** (hotpotqa 0.095, finqa
0.130), where faithful answers synthesise rather than reuse source
wording. This corpus is technical documentation, so the cheap metric is
fit for it — and would be the wrong metric for a paraphrase-heavy one.

RAGBench's labels are model-generated, not human adjudication, which
bounds the claim; the full caveats, the operating-point analysis (the
check deliberately over-rejects, because dropping a true sentence beats
shipping a fabricated one), and reproduction steps are in
[`docs/stage10_grounding_validation.md`](docs/stage10_grounding_validation.md).

```
python eval/validate_grounding.py --tag stage10-grounding
```

## Retrieval core (Stage 3)

Hybrid retrieval: BM25 (numpy posting lists, `app/core/bm25.py`) and dense
FAISS retrieval (`app/core/dense.py`) run as isolated modules, fused with
Reciprocal Rank Fusion (`app/core/rrf.py`, k=60), then reranked by a
cross-encoder (`app/core/onnx_text.py`). Embedding and reranking are LOCAL
int8 ONNX models (MiniLM-L6 family) running on raw onnxruntime with memory
arenas and weight prepacking disabled — required to fit Render's 512MB cap
(measured gate: `scripts/measure_memory.py`, results in
`eval/results/memory_stage3.json`: 407.6MB peak at 50k chunks, FITS).

Load-test results and the latency consequences for free-tier serving are
in `docs/loadtest_stage3.md`.

## Generation & citations (Stage 4)

`app/generation/` adds source-grounded generation over the hybrid
retriever: prompt with numbered sources → Gemini (REST, typed error
taxonomy in `app/errors.py`) → **chunk-level citation validation before
anything is returned**. Every sentence is checked against the chunks it
cites using the same grounding definition as the eval harness
(`app/core/grounding.py` — measurement and enforcement can never drift);
fabricated or mis-cited sentences are removed, and if nothing survives the
service falls back to a deterministic extractive answer. Every LLM failure
mode (quota 429, timeout, 5xx, malformed, auth, no key) maps to an
explicit `degraded_*` status with an extractive answer — the exact
client-visible contract is the table in `app/generation/service.py`.

Reranking degradation is equally explicit: every response carries
`rerank_status` (`full` / `partial` / `skipped_budget` / `disabled`), with
an adaptive per-request budget that predicts micro-batch cost from a
learned EWMA and falls back to RRF order rather than blowing the latency
target (defaults set from CPU-throttled measurements, not laptop numbers:
`docs/loadtest_stage4.md`).

## API serving & admission control (Stage 5)

`POST /v1/query` serves the full pipeline behind three ordered gates:
API-key auth (401; anonymous only outside production), per-client Redis
rate limiting (429 + Retry-After, fail-open on Redis outage), and a
**bounded admission queue** (`app/api/admission.py`): at most
`ADMISSION_MAX_CONCURRENCY` requests execute while
`ADMISSION_MAX_QUEUE_DEPTH` wait; anything beyond gets an immediate
**503 + Retry-After** (estimated from a live service-time EWMA) instead
of unbounded queueing. Measured consequence: admitted-request p95 stays
flat regardless of offered load; excess load is shed early and honestly.
The empirically measured concurrency ceilings (local + 0.1-CPU
container) are documented in `docs/stage5_admission.md`. Single-process
by design — the 512MB cap cannot hold two model copies.

### Response cache

A repeat of an identical query is served from cache, spending no pipeline
compute and no LLM quota — on this infrastructure the cache *is* capacity
(Stage 2.5). Redis backs it where configured; deployments without Redis
use an equivalent in-process cache with the same key, the same
cacheable-status set (transient degradations are never replayed) and the
same TTL. Cache keys are namespaced by a corpus version, so any corpus
change makes prior entries unaddressable rather than stale. Measured:
1.45 s → 0.007 s on a repeat query.

## Web app & document upload (Stage 9.6–9.10)

`frontend/` is a zero-dependency web app served same-origin (no CORS, no
build step): Google sign-in, a chat surface, and a **retrieval trace** on
every answer showing all retrieved chunks in rank order with the cited
ones marked — the retriever's ranking next to the generator's usage.
Every documented backend state has an explicit rendering; blank screens
and raw console errors are defined as bugs.

`POST /v1/documents` accepts a `.txt`/`.md`/`.pdf` and indexes it into
the **live session** — chunked by the same corpus v1 paragraph rule, so
an uploaded chunk cites exactly like a corpus chunk. It is deliberately
**dev/staging only (403 in production)**: arbitrary writes to a shared
corpus would invalidate the eval contract above, and the authorization
model for that is unbuilt, so the feature is not pretended. Production
ingestion remains the versioned CLI (`python -m app.ingest.cli`).
Rebuilds run through the same admission controller as queries — it is
the same CPU-bound work on the same core.

## Stage 0 skeleton

`src/ragp/` contains the earliest working pipeline: a dependency-free BM25
index (`bm25.py`) over paragraph chunks (`corpus.py`) with an extractive
answer stub (`pipeline.py`) that returns the first two sentences of the top
chunk. It exists so the harness exercises a full query→retrieve→answer path
from day one. Hallucination rate is ~0 by construction for an extractive
system — that is the honest baseline, and the number becomes informative
once generation is abstractive.

## Storage layer (Stage 2)

PostgreSQL holds documents, chunks, index-version records, chunk→FAISS-row
mappings, and query/citation logs (`migrations/0001_init.sql`). Embedding
vectors live in FAISS files on disk, not in Postgres — free-tier Postgres
storage is capped and vectors are the bulk of the data; Postgres stores the
metadata and hashes needed to verify them. Redis provides the response
cache and atomic (Lua) fixed-window rate limiting; both fail soft/open on
Redis outage — a cache blip degrades latency, never availability (tradeoff
documented in `app/storage/redis_store.py`).

### Index versioning & rollback

Every successful ingestion run produces an immutable version directory
`indexes/{version_id}/` (index.faiss + manifest.json) and an
`index_versions` row. Nothing is ever mutated in place:

- **Build ≠ activate.** A new index goes live only via an explicit
  `activate`, a single transactional status flip. A partial unique index in
  Postgres guarantees at most one `active` version exists.
- **Writes are atomic.** Indexes are staged in a temp dir, fsynced, then
  renamed — a version directory either fully exists or doesn't. Disk-full
  mid-write leaves the active index untouched.
- **Rollback** (`python -m app.ingest.cli rollback`) transactionally marks
  the active version `rolled_back` and re-activates the most recent prior
  `ready` version, whose files are still on disk. `gc` retains the active
  + last N ready versions and sweeps the rest plus orphaned staging dirs.
- **Integrity.** index.faiss SHA-256 is recorded in both the manifest and
  Postgres; loading verifies it and refuses to serve a corrupt file.

### Ingestion failure policy (each behavior integration-tested)

| Failure | Behavior |
|---|---|
| Malformed doc | Skipped + recorded in run report; run aborts before any write if >10% malformed (systematic input breakage) |
| Embedding failure mid-batch | Batch retried 3x with backoff; then the run aborts, version marked `failed`, **no index written** — a partially-embedded index is silent corruption |
| Disk full / index write error | Staging dir cleaned up, version marked `failed`, active index unaffected |
| Re-run of identical corpus | Detected via corpus SHA-256 + embedder id; existing version reused, no rebuild |

## Layout

```
app/                      application source
  api/                    routes: query, documents, auth, health, admission
  core/                   retrieval: bm25, dense, rrf, hybrid, grounding
  generation/             LLM clients, quota guards, citation validation
  ingest/                 versioned ingestion CLI + FAISS store
  storage/                Postgres + Redis adapters
frontend/                 zero-dependency web app (html/css/js, no build)
data/corpus_v1.jsonl      versioned corpus
eval/dataset_v1.jsonl     versioned eval queries + gold labels
eval/run_eval.py          THE eval harness (stdlib-only)
eval/results/             committed baseline + per-run results
eval/validate_grounding.py  grounding metric vs RAGBench (Stage 10)
docs/                     one design report per stage
```

## Design reports

Every stage is written up: what was built, why that way, what was
measured, and what was deliberately not done. Read in this order.

| # | Report | What it decides |
|---|---|---|
| 2.5 | [infrastructure](docs/infrastructure.md) | free-tier component choice and the limits each imposes |
| 3 | [loadtest_stage3](docs/loadtest_stage3.md) | retrieval core under load; the measured concurrency ceiling |
| 4 | [loadtest_stage4](docs/loadtest_stage4.md) | adaptive rerank budget, generation + citation layer |
| 4.5 | [stage4_5_quota](docs/stage4_5_quota.md) | quota-aware generation guardrails |
| 5 | [stage5_admission](docs/stage5_admission.md) | bounded admission queue — why shedding beats queueing |
| 5 | [stage5_api_hardening](docs/stage5_api_hardening.md) | API versioning, strict validation, size limits |
| 6 | [stage6_observability](docs/stage6_observability.md) | metrics and structured logging |
| 6.5 | [stage6_5_concurrency](docs/stage6_5_concurrency.md) | shared-state safety audit |
| 7 | [stage7_testing](docs/stage7_testing.md) | the testing strategy and named integration scenarios |
| 7.5 | [stage7_5_fullstack](docs/stage7_5_fullstack.md) | full-stack regression load test |
| 7.7 | [stage7_7_breakers](docs/stage7_7_breakers.md) | quota and cost circuit breakers |
| 8 | [stage8_deploy](docs/stage8_deploy.md) | live deployment on Render free tier |
| 8.5 | [stage8_5_cicd](docs/stage8_5_cicd.md) | CI/CD and the merge-blocking drill |
| 8.9 | [stage8_9_privacy](docs/stage8_9_privacy.md) | privacy policy and data practices |
| 9.5 | [stage9_5_frontend_auth](docs/stage9_5_frontend_auth.md) | frontend + Google auth architecture decisions |
| 9.6 | [stage9_6_frontend](docs/stage9_6_frontend.md) | production frontend with Google login |
| 9.7 | [stage9_7_chat_ui](docs/stage9_7_chat_ui.md) | chat UI, and what was deliberately not ported |
| 9.8 | [stage9_8](docs/stage9_8.md) | visual identity, dev-mode upload, review pass |
| 9.9 | [stage9_9_hardening](docs/stage9_9_hardening.md) | readiness vs liveness, write auth, security headers |
| 9.10 | [stage9_10_app_shell](docs/stage9_10_app_shell.md) | app shell, sign-in surface, client-side threads |
| 10 | [stage10_grounding_validation](docs/stage10_grounding_validation.md) | **the grounding metric measured against RAGBench** |
| 11 | [stage11_monitoring](docs/stage11_monitoring.md) | monitoring and alerting |

If you read only two: [stage5_admission](docs/stage5_admission.md) for
how the system behaves under load it cannot serve, and
[stage10_grounding_validation](docs/stage10_grounding_validation.md) for
the only metric in this project measured against data it did not create.
