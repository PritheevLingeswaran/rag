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
data/corpus_v1.jsonl      versioned corpus
eval/dataset_v1.jsonl     versioned eval queries + gold labels
eval/run_eval.py          THE eval harness (stdlib-only)
eval/results/             committed baseline + per-run results
src/ragp/                 pipeline source
```
