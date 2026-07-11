# Stage 4 Report — Adaptive Rerank Budget + Generation & Citation Layer

Date: 2026-07-11. Raw artifacts: `eval/results/loadtest_stage4_throttled.json`,
`eval/results/memory_stage4_linux.json`, `eval/results/stage4-*.json`.

## Part A — CPU-throttled load test (sets the production defaults)

Exact command (Linux container, hard limits approximating Render free tier):

```
docker build -f docker/loadtest.Dockerfile -t ragp-loadtest .
docker run -d --name ragp-throttled-lt --cpus=0.1 --memory=512m \
  -v "$PWD:/app" -v ragp-hf-cache:/root/.cache/huggingface ragp-loadtest \
  python scripts/load_test_retrieval.py --chunks 50000 --requests 20 \
  --warmup 3 --concurrency 1,4 --style english --rerank-depth 20,10,5,0 \
  --json eval/results/loadtest_stage4_throttled.json
```

(20 requests/level rather than 200: at ~10x throttling the full matrix
would take hours; percentiles are correspondingly coarser — stated, not
hidden.)

### Results (english corpus, 50k chunks, 0.1 CPU / 512MB)

| rerank depth | conc | p50 | p95 | p99 | rps |
|---|---|---|---|---|---|
| 20 | 1 | 14104 ms | 15998 ms | 16197 ms | 0.07 |
| 20 | 4 | 76696 ms | 100009 ms | 118199 ms | 0.05 |
| 10 | 1 | 7606 ms | 8498 ms | 8498 ms | 0.13 |
| 10 | 4 | 39701 ms | 44396 ms | 48901 ms | 0.10 |
| 5 | 1 | 4198 ms | 4697 ms | 4898 ms | 0.24 |
| 5 | 4 | 18901 ms | 26795 ms | 27197 ms | 0.19 |
| **0 (RRF only)** | 1 | **500 ms** | **595 ms** | 595 ms | 1.96 |
| 0 (RRF only) | 4 | 2598 ms | 3300 ms | 3799 ms | 1.54 |

Derived, consistent across depths: **rerank ≈ 700 ms per passage at 0.1
CPU** ((14104−500)/20=680, (7606−500)/10=711, (4198−500)/5=740) vs ~65
ms/passage unthrottled. Retrieval floor without rerank: 500 ms p50.

Bonus verification: the same image ran the Stage 3 RAM gate under a hard
512MB limit on Linux — **393.8 MB peak, no OOM kill**
(`eval/results/memory_stage4_linux.json`), closing Stage 3's
Windows-only caveat. During the throttled run the container held ~353MiB.

### Defaults set from this data (app/config.py)

- `rerank_depth = 10` (candidate ceiling)
- `rerank_budget_ms = 700`

Mechanism: before each micro-batch the pipeline predicts its cost from an
EWMA of measured per-passage rerank time; a batch that would exceed the
remaining budget never starts, and remaining candidates keep RRF order.
Prediction (not just an elapsed-time check) is required by these numbers:
one micro-batch of 5 costs ~3.5 s at 0.1 CPU, so a check-after design
would overshoot a 700 ms budget 5x. Consequences:
- capable host (~65 ms/passage): depth 10 fully reranks in ~650 ms → `full`
- 0.1-CPU host (~700 ms/passage): first batch predicted 3.5 s → rerank
  skipped, RRF order served at ~500 ms p50 → `skipped_budget`, explicit
- until the first batch ever runs in a process there is no estimate, so
  the first request may overshoot once; the serving startup must seed the
  EWMA with a warmup query (API-stage task, noted in code).

### Honest limits at 0.1 CPU (unchanged by the budget)

Even RRF-only at concurrency 4 is 2.6 s p50 — a single 0.1-CPU worker
cannot hold p95 ≤ 1 s for concurrent retrieval, budget or not. Real
concurrency on Render must come from the response cache (already forced
by Gemini's RPD math, Stage 2.5) and honest queueing. LLM generation
latency comes on top of all numbers here.

## Part B — Rerank quality: full vs fallback (standing eval format)

Commands:
```
python eval/run_eval.py --pipeline hybrid          --baseline eval/results/baseline.json --tag stage4-full-rerank
python eval/run_eval.py --pipeline hybrid-fallback --baseline eval/results/baseline.json --tag stage4-fallback
```

| metric | full rerank (d20, no budget) | fallback (budget=0, pure RRF) | Stage 0 baseline |
|---|---|---|---|
| P@1 | 0.9000 | **0.9000** | 0.8500 |
| MRR@10 | 0.9417 | **0.9500** | 0.9250 |
| Hallucination rate | 0.0000 | 0.0000 | 0.0000 |
| Latency p50 (60-chunk corpus, laptop) | 639.6 ms | 42.5 ms | 0.077 ms |
| P@1 misses | q07, q11 | q01, q07 | q01, q07, q11 |

**Correction to the Stage 3 report, on the record**: I claimed the
reranker's value as "+0.05 P@1 (fused-only was 0.85)" — an estimate, not
a measurement. Measured now: fused-only (RRF) P@1 is **0.90**, equal to
the reranked path, with slightly better MRR. On this 20-query eval set
the cross-encoder provides no net quality gain (it fixes q01 but demotes
q11; RRF does the reverse); the +0.05 over baseline comes from hybrid
dense+lexical fusion. Caveat both ways: 20 queries ⇒ each query is 0.05
of P@1; treat the full-vs-fallback difference as noise-level, not as
proof the reranker is useless. The budget architecture keeps rerank
where it is cheap and skips it where it is ruinous — consistent with
this data.

## Part C — Fallback logic test output (verbatim)

```
tests/core/test_hybrid.py::test_full_rerank_orders_by_reranker_score PASSED
tests/core/test_hybrid.py::test_fused_candidates_include_both_retrievers PASSED
tests/core/test_hybrid.py::test_zero_budget_forces_pure_rrf_order PASSED
tests/core/test_hybrid.py::test_budget_exceeded_mid_way_yields_partial PASSED
tests/core/test_hybrid.py::test_predictive_gate_skips_after_learning_cost PASSED
tests/core/test_hybrid.py::test_generous_budget_reranks_fully PASSED
tests/core/test_hybrid.py::test_unlimited_budget_is_default PASSED
tests/core/test_hybrid.py::test_depth_zero_disables_rerank_entirely PASSED
tests/core/test_hybrid.py::test_every_result_carries_rerank_info PASSED
tests/core/test_hybrid.py::test_negative_depth_rejected PASSED
tests/core/test_hybrid.py::test_depth_below_final_top_k_rejected PASSED
```

Every response carries `rerank_status` (`full` / `partial` /
`skipped_budget` / `disabled` / `no_candidates`) plus scored/candidate
counts; degraded paths are logged (`rerank_degraded`). Same explicit
pattern as the generation layer's `degraded_*` statuses below.

## Part D — Generation & citation layer (Stage 4 proper)

Serving path: hybrid retrieve → prompt with numbered sources → Gemini
(typed failure taxonomy) → **citation validation before return**. The
validator shares its grounding definition with the harness's
hallucination metric (`app/core/grounding.py`); the refactor was verified
bit-identical against the committed baseline before anything else landed.

Client-visible failure contract (each row is a passing test in
`tests/generation/test_service.py`): quota-429 → `degraded_quota` (+
retry_after), timeout → `degraded_timeout`, 5xx/network after one retry →
`degraded_llm_error`, malformed/empty/safety-blocked →
`degraded_llm_malformed`, auth → `degraded_llm_auth` (ERROR-logged),
no key → `degraded_no_llm`, all sentences rejected →
`degraded_citation_rejected`. Every degraded path returns a
deterministic extractive answer with citations, never a 500.

**Definition-of-done test** (passing; `tests/generation/test_citations.py`):
a fabricated "Raft clusters are hard-limited to a maximum of 42 nodes"
claim with a legitimate-looking `[2]` citation is rejected
(`unsupported`, coverage < 0.7) and removed from the returned answer,
while the grounded sentence survives. Also caught: citations of
nonexistent sources, and correct facts citing the *wrong* chunk.

### Eval (standing format), full serving path vs baseline

Command: `python eval/run_eval.py --pipeline generation --baseline eval/results/baseline.json --tag stage4-generation`

Run keyless, so the harness measured the real `degraded_no_llm` path —
exactly what a deployment without `GEMINI_API_KEY` serves:

| metric | generation path (keyless) | baseline | delta |
|---|---|---|---|
| P@1 | 0.9000 | 0.8500 | +0.0500 |
| MRR@10 | 0.9417 | 0.9250 | +0.0167 |
| Hallucination rate | 0.0000 | 0.0000 | +0.0000 |
| Latency p50 | 647.6 ms | 0.077 ms | +647.5 ms |

Stated plainly: hallucination rate 0.0 here reflects the extractive
degraded path (grounded by construction) — it is NOT evidence about live
LLM output. The harness's `--pipeline generation` mode runs the real
Gemini path the day a key exists; that run must be tagged and reported
before any live-quality claim is made. Suite: 103 tests passing.
