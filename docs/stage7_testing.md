# Stage 7 Report — Testing

Date: 2026-07-11. Suite: **188 tests, all passing.**

## Definition of done — clean checkout, one command

Fresh `git clone` → fresh venv → `pip install -r requirements.txt` →
`python -m pytest`:

```
tests\test_config.py ....                                                [100%]
======================= 188 passed in 124.07s (0:02:04) =======================
```

This validated requirements.txt completeness against an empty venv.
Environment notes, stated: integration tests need the two disposable
docker containers (Postgres 16 + Redis 7 on test ports) and skip with a
visible reason when absent; ONNX models come from the user-level HF cache
(a network-fresh machine downloads ~50MB once). One environment artifact
found while doing this: faiss's DLL fails to load from very long Windows
paths ("filename or extension is too long") — not a packaging bug of
ours, but worth knowing; the clean run used a short path.

## Named integration scenarios (the required list)

| Scenario | Named test | What it protects against |
|---|---|---|
| Happy path | `test_api_e2e.py::test_happy_path_returns_grounded_cited_answer` | end-to-end wiring rot: real models + corpus through real HTTP, citations + explicit degradation labels intact |
| Empty index | `test_api_e2e.py::test_empty_corpus_fails_loudly_at_build_not_at_runtime` | a serving process silently existing with nothing to serve; policy: refuse at build, never 500 at query time |
| (empty-result cousin) | `test_api_e2e.py::test_query_with_no_lexical_overlap_still_answers` | BM25-empty crashing the fusion path |
| Malformed query | `test_api_e2e.py::test_malformed_query_payloads_rejected_cleanly` (+ `test_hardening.py::test_malformed_json_body_is_clean_422`, `test_unknown_fields_rejected`) | parser/validator leaks, injected fields |
| Oversized query | `test_api_e2e.py::test_oversized_query_rejected` (+ `test_hardening.py::test_oversized_body_rejected_413_before_parsing`) | memory abuse; body read before size check |
| Concurrent @ target scale | `test_concurrency.py::test_full_pipeline_concurrent_hammer` (50k chunks, 8 threads, bit-identical results + exact EWMA accounting; plus the FAISS/BM25/ONNX equality tests) | result corruption & lost updates under real load |
| Quota exhaustion (Stage 4.5) | `test_quota.py::test_system_at_rpm_boundary_serves_degraded_not_500` (+ Redis two-worker exactness) | hard-failing at the provider boundary instead of labeled degradation |

## Unit-test inventory for isolated modules (Stages 2–4)

Every isolated module has dedicated tests; gaps found by this stage's
inventory and filled: `grounding.py` (9 tests — pins the eval-contract
semantics incl. the exact 0.7 boundary), `faiss_store.py` (9 isolated
filesystem tests — atomic write, integrity, gc), `HashingEmbedder`
(6 tests). Already covered: bm25, corpus, dense, rrf, hybrid (incl.
budget/fallback), citations, llm_client (transport taxonomy), service
(failure table), quota, admission, middleware/deps (via API tests),
redis_store + repositories + migrations (integration, service-bound by
nature).

## Coverage on the critical path — with meaning, not vanity

`pytest --cov=app` (full table in CI output; highlights):

| Module (critical query path) | Cover | The uncovered lines are |
|---|---|---|
| core/hybrid.py | 98% | defensive no-candidate branch |
| generation/service.py | 98% | two log-only lines |
| generation/citations.py | 98% | one validation guard |
| ingest/faiss_store.py | 98% | fsync error re-raise arms |
| core/bm25.py / rrf.py | 96% | input-validation raises |
| api (deps/middleware/query/main) | 91–96% | 411 content-length arm, prod-guard redundancy |
| generation/quota.py | 94% | unparseable-retry-after log arm |
| core/onnx_text.py | 93% | model-download helper (network) |

Honest zeros and lows, with reasons:
- `core/pipeline.py` **0%** — the Stage 0 skeleton; exercised by the
  eval harness (`--pipeline skeleton` reproduces the committed baseline),
  not by pytest. Kept as the eval reference implementation.
- `ingest/cli.py` **0%** — argparse wiring around fully-tested pieces;
  exercised live in the Stage 2 demo. Risk: flag typos, accepted.
- `storage/repositories.py` 85% — the uncovered block is
  `QueryLogRepo.log`, which is **not wired into serving yet**; the
  coverage gap is a truthful signal of an unfinished feature, recorded.

Overall: 90% — reported for completeness, not as the headline; the
claim that matters is the table above plus 16 concurrency tests whose
assertions are exactness, not coverage.
