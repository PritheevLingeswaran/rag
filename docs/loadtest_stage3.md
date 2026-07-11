# Stage 3 Load Test Report — Retrieval Core

Date: 2026-07-11. Raw per-run JSON (exact command, config, environment
embedded in each): `eval/results/loadtest_stage3_*.json`,
`eval/results/memory_stage3.json`.

**Test machine**: AMD Ryzen 5 7520U (8 logical cores), Windows 11,
Python 3.11.9. This is substantially faster than Render's free-tier
0.1-CPU container — every number below is a LOWER bound on production
latency. See "Free-tier consequences".

**What is measured**: the in-process retrieval path — ONNX query embed →
FAISS + BM25 search → RRF → ONNX cross-encoder rerank → assembly — driven
by N concurrent threads against one pipeline instance at 50,000 chunks.
HTTP/network overhead is deliberately excluded (it belongs to the API
stage); everything compute-heavy is fully real (real ONNX inference on
every request). Dense index vectors are random-normalized (FAISS latency
is independent of vector values); BM25/rerank text is synthetic — Zipfian
over real-corpus vocabulary for realistic subword-token lengths
("english"), or `wordNNNNN` gibberish that pins every passage at the
256-token cap ("worstcase" = upper bound).

## RAM gate (prerequisite, measured before implementation)

Command: `python scripts/measure_memory.py --chunks 50000 --json eval/results/memory_stage3.json`

| Step (cumulative) | RSS | Peak |
|---|---|---|
| bare interpreter | 19.2 MB | 19.2 MB |
| + FastAPI app imported | 49.9 MB | 49.9 MB |
| + ONNX embedder AND reranker sessions | 168.7 MB | 171.3 MB |
| + both warmed (rerank max_len 256, micro_batch 5) | 171.2 MB | 171.3 MB |
| + FAISS 50k×384 read from disk (prod boot path) | 247.6 MB | 247.6 MB |
| + BM25 over 50k chunks + chunk text store | 363.5 MB | 375.5 MB |
| + 25-query burst, full hybrid path | 367.3 MB | **407.6 MB** |

**Verdict: FITS — 407.6 MB peak vs 512 MB cap (~20% headroom).**

It did NOT fit on the first attempt; getting here required three measured
design changes, all now defaults in `app/core/onnx_text.py`:
1. fastembed → raw onnxruntime with `enable_cpu_mem_arena=False` and
   `session.disable_prepacking=1` (fastembed config peaked at **544 MB**);
2. quantized `Xenova/all-MiniLM-L6-v2` (int8) instead of bge-small ONNX,
   whose file ballooned ~150 MB at session load;
3. reranker `max_length` 512→256 + micro-batches of 5: cross-encoder
   attention activations scale with batch×seq², and a 20×512 batch
   transiently allocated ~250 MB (measured peak 648 MB → 408 MB after).

Caveats: Windows RSS; re-verify on the Linux container at deploy. Peak was
measured with all-256-token passages (worst case), so the burst figure is
conservative. int8 dynamic quantization makes rerank logits shift slightly
(~0.1–0.3) with batch composition; fixed micro_batch ⇒ bit-deterministic
(tested in `tests/integration/test_onnx_models.py`).

## Load test results

200 requests per level, 10 warmup, seed-42 query set (6-word prefixes of
corpus chunks). Exact commands as run:

```
python scripts/load_test_retrieval.py --chunks 50000 --requests 200 --concurrency 1,4,8 --style english --rerank-depth 20 --json eval/results/loadtest_stage3_english_d20.json
python scripts/load_test_retrieval.py --chunks 50000 --requests 200 --concurrency 8  --style english --rerank-depth 20 --json eval/results/loadtest_stage3_english_d20_c8.json
python scripts/load_test_retrieval.py --chunks 50000 --requests 200 --concurrency 1,4,8 --style english --rerank-depth 10 --json eval/results/loadtest_stage3_english_d10.json
python scripts/load_test_retrieval.py --chunks 50000 --requests 200 --concurrency 1,4,8 --json eval/results/loadtest_stage3_worstcase_d20.json
```

### english, rerank_depth=20 (default config)

| conc | p50 | p95 | p99 | mean | rps |
|---|---|---|---|---|---|
| 1 | 1429.4 ms | 1598.0 ms | 1763.1 ms | 1441.1 ms | 0.69 |
| 4 | 2163.0 ms | 2499.1 ms | 2917.2 ms | 2189.0 ms | 1.80 |
| 8 | 3550.6 ms | 5028.5 ms | 5817.1 ms | 3775.2 ms | 2.10 |

Integrity note: the first conc-8 run of this config produced physically
impossible values (p95 ≈ 10⁷ ms in a run that completed in minutes —
consistent with a host stall/sleep during measurement). That row was
discarded and the level rerun in isolation (`_c8.json`, table above); the
raw discarded file is kept in `loadtest_stage3_english_d20.json`.

### english, rerank_depth=10

| conc | p50 | p95 | p99 | mean | rps |
|---|---|---|---|---|---|
| 1 | 723.9 ms | 898.7 ms | 976.7 ms | 743.9 ms | 1.34 |
| 4 | 1069.6 ms | 1209.8 ms | 1296.9 ms | 1074.4 ms | 3.71 |
| 8 | 1739.7 ms | 2070.3 ms | 2363.4 ms | 1761.6 ms | 4.49 |

### worstcase (all passages at 256-token cap), rerank_depth=20

| conc | p50 | p95 | p99 | mean | rps |
|---|---|---|---|---|---|
| 1 | 2113.2 ms | 2285.6 ms | 2338.9 ms | 2131.3 ms | 0.47 |
| 4 | 3621.2 ms | 4435.9 ms | 5165.6 ms | 3814.6 ms | 1.04 |
| 8 | 5564.2 ms | 8243.8 ms | 9060.0 ms | 5886.7 ms | 1.35 |

### Per-stage breakdown (english, depth 20, median over 30 queries)

| stage | median |
|---|---|
| query embed (ONNX) | 41.6 ms |
| BM25 top-30 over 50k | 1.7 ms |
| FAISS top-30 over 50k | 4.8 ms |
| RRF fusion | 0.1 ms |
| **cross-encoder rerank (20 passages)** | **1427.8 ms** |

## Analysis — read this before Stage 4/5

1. **The cross-encoder is 97% of retrieval latency.** Everything else —
   both retrievers, fusion, embedding — totals ~50 ms at 50k chunks.
   Rerank cost is linear in depth and roughly quadratic in passage token
   length (depth 20→10 halves it; english→worstcase roughly doubles it).
2. **The p95 < 500ms–1s end-to-end target is NOT achievable on Render
   0.1 CPU with always-on rerank at depth 20.** 1.4 s p50 on an 8-core
   laptop can only get worse on a 0.1-CPU container — plausibly by an
   order of magnitude — and LLM generation isn't in these numbers yet.
   This is a measured conclusion, not a guess about exact prod numbers;
   the deploy stage must re-measure on the real container.
3. **Options this forces (decision needed, my recommendation is (a)):**
   (a) rerank_depth 5–8 by default + a latency budget: rerank only within
   a per-request time budget, fall back to RRF order when exceeded (RRF
   order is already good: P@1 0.90 came from rerank@20 on the real
   corpus, but fused-only P@1 was 0.85 — the cheap path is not garbage);
   (b) drop the cross-encoder entirely and rely on RRF (loses its P@1
   gain); (c) pay for compute (out of scope by project constraint).
4. **Throughput ceiling with rerank@10 was ~4.5 rps on 8 cores** —
   "dozens of concurrent users" only works if most requests hit the
   response cache (which Gemini's 1,500 RPD already forces, Stage 2.5).

## Eval harness (standing format), hybrid vs Stage 0 baseline

Command: `python eval/run_eval.py --pipeline hybrid --baseline eval/results/baseline.json --tag stage3-hybrid`

| metric | current | baseline | delta |
|---|---|---|---|
| P@1 | 0.9000 | 0.8500 | **+0.0500** |
| MRR@10 | 0.9417 | 0.9250 | **+0.0167** |
| Hallucination rate | 0.0000 | 0.0000 | +0.0000 |
| Unsupported-token rate | 0.0000 | 0.0000 | +0.0000 |
| Latency p50 (real 60-chunk corpus) | 658.7 ms | 0.077 ms | +658.6 ms |
| Latency p95 | 725.9 ms | 0.142 ms | +725.8 ms |

q01 (Raft leader election) fixed by dense retrieval; q07/q11 remain
(rank 2 and 3). Latency delta is the honest price of real inference vs the
Stage 0 extractive stub.
