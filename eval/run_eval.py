"""Fixed, reproducible evaluation harness. THE only source of metric claims.

Usage:
    python eval/run_eval.py                          # run, print, save results
    python eval/run_eval.py --baseline eval/results/baseline.json
                                                     # also diff vs baseline
    python eval/run_eval.py --tag stage1             # label the results file

Metrics (definitions are part of the harness contract; changing any of them
invalidates comparison with prior runs and requires a version bump):

- P@1: fraction of queries whose top-1 retrieved chunk is in the gold set.
- MRR@10: mean reciprocal rank of the first gold chunk within the top 10;
  0 if no gold chunk appears in the top 10.
- Hallucination rate: fraction of answers containing at least one unsupported
  sentence. A sentence is unsupported when < GROUNDING_THRESHOLD of its
  content tokens (alphanumeric, lowercased, stopwords removed) appear in the
  union of retrieved chunk texts. This is a deterministic lexical proxy, not
  an LLM judge; see eval/README note in repo README.
- Unsupported-token rate: secondary, finer-grained view of the same check.
- Latency: per query, LATENCY_REPEATS timed runs after LATENCY_WARMUP
  warmups; the query's latency is the median of those runs. Reported p50/p95
  are over the per-query medians. Wall-clock latency is inherently
  machine-dependent; the *methodology* is what is fixed.

Determinism: RANDOM_SEED fixes query execution order. The Stage 0 pipeline
itself is fully deterministic. Quality metrics (P@1, MRR, hallucination) are
bit-for-bit reproducible; latency is reproducible in method, not in value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ragp import __version__ as ragp_version  # noqa: E402
from ragp.bm25 import tokenize  # noqa: E402
from ragp.pipeline import PipelineResult, SkeletonPipeline  # noqa: E402

# ---- Harness contract constants (bump HARNESS_VERSION if any change) ----
HARNESS_VERSION = "1.0"
RANDOM_SEED = 42
CORPUS_PATH = REPO_ROOT / "data" / "corpus_v1.jsonl"
DATASET_PATH = REPO_ROOT / "eval" / "dataset_v1.jsonl"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
MRR_CUTOFF = 10
GROUNDING_THRESHOLD = 0.7
LATENCY_WARMUP = 3
LATENCY_REPEATS = 5

STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or "
    "that the to was were what when where which who why with does do did "
    "not no can could should would".split()
)

import re  # noqa: E402

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


def sentence_support(answer: str, context: str) -> tuple[int, int, int, int]:
    """Return (unsupported_sentences, total_sentences,
    unsupported_tokens, total_tokens) for an answer against context."""
    ctx_tokens = set(content_tokens(context))
    sentences = [s for s in _SENTENCE_RE.split(answer) if content_tokens(s)]
    unsupported_sents = 0
    unsupported_toks = 0
    total_toks = 0
    for sent in sentences:
        toks = content_tokens(sent)
        missing = sum(1 for t in toks if t not in ctx_tokens)
        total_toks += len(toks)
        unsupported_toks += missing
        coverage = (len(toks) - missing) / len(toks)
        if coverage < GROUNDING_THRESHOLD:
            unsupported_sents += 1
    return unsupported_sents, len(sentences), unsupported_toks, total_toks


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = {"query_id", "query", "relevant_chunk_ids",
                       "expected_answer"} - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no}: missing {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty eval dataset")
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "UNCOMMITTED_OR_NO_GIT"


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; deterministic for a fixed value list."""
    if not values:
        raise ValueError("percentile of empty list")
    ordered = sorted(values)
    rank = max(1, int(round(pct / 100.0 * len(ordered))))
    return ordered[rank - 1]


def evaluate() -> dict:
    pipeline = SkeletonPipeline(CORPUS_PATH, top_k=MRR_CUTOFF)
    dataset = load_dataset(DATASET_PATH)

    # Validate gold chunk ids against the actual chunker output up front,
    # so a chunking-contract drift fails loudly instead of scoring as zeros.
    known = set(pipeline._chunks.keys())
    for row in dataset:
        unknown = [c for c in row["relevant_chunk_ids"] if c not in known]
        if unknown:
            raise ValueError(
                f"{row['query_id']}: gold chunk ids not in corpus: {unknown}"
            )

    order = list(range(len(dataset)))
    random.Random(RANDOM_SEED).shuffle(order)

    per_query = []
    for idx in order:
        row = dataset[idx]
        gold = set(row["relevant_chunk_ids"])

        # Warmups (untimed), then timed repeats; median is the query latency.
        for _ in range(LATENCY_WARMUP):
            pipeline.run(row["query"])
        timings = []
        result: PipelineResult | None = None
        for _ in range(LATENCY_REPEATS):
            t0 = time.perf_counter()
            result = pipeline.run(row["query"])
            timings.append((time.perf_counter() - t0) * 1000.0)
        assert result is not None
        latency_ms = statistics.median(timings)

        retrieved = result.retrieved_chunk_ids
        p_at_1 = 1.0 if retrieved and retrieved[0] in gold else 0.0
        rr = 0.0
        for rank, cid in enumerate(retrieved[:MRR_CUTOFF], start=1):
            if cid in gold:
                rr = 1.0 / rank
                break
        context = "\n".join(result.retrieved_texts)
        u_sents, n_sents, u_toks, n_toks = sentence_support(
            result.answer, context
        )
        per_query.append({
            "query_id": row["query_id"],
            "p_at_1": p_at_1,
            "reciprocal_rank": rr,
            "first_gold_rank": (1.0 / rr) if rr > 0 else None,
            "top1": retrieved[0] if retrieved else None,
            "gold": sorted(gold),
            "hallucinated": u_sents > 0,
            "unsupported_sentences": u_sents,
            "answer_sentences": n_sents,
            "unsupported_tokens": u_toks,
            "answer_tokens": n_toks,
            "latency_ms": round(latency_ms, 3),
            "answer": result.answer,
        })

    per_query.sort(key=lambda r: r["query_id"])
    n = len(per_query)
    latencies = [r["latency_ms"] for r in per_query]
    total_utoks = sum(r["unsupported_tokens"] for r in per_query)
    total_toks = sum(r["answer_tokens"] for r in per_query)
    metrics = {
        "p_at_1": round(sum(r["p_at_1"] for r in per_query) / n, 4),
        "mrr_at_10": round(
            sum(r["reciprocal_rank"] for r in per_query) / n, 4
        ),
        "hallucination_rate": round(
            sum(1 for r in per_query if r["hallucinated"]) / n, 4
        ),
        "unsupported_token_rate": round(
            total_utoks / total_toks if total_toks else 0.0, 4
        ),
        "latency_p50_ms": round(percentile(latencies, 50), 3),
        "latency_p95_ms": round(percentile(latencies, 95), 3),
        "num_queries": n,
    }
    return {
        "harness_version": HARNESS_VERSION,
        "ragp_version": ragp_version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "random_seed": RANDOM_SEED,
        "corpus_sha256": sha256_file(CORPUS_PATH),
        "dataset_sha256": sha256_file(DATASET_PATH),
        "config": {
            "mrr_cutoff": MRR_CUTOFF,
            "grounding_threshold": GROUNDING_THRESHOLD,
            "latency_warmup": LATENCY_WARMUP,
            "latency_repeats": LATENCY_REPEATS,
            "top_k": MRR_CUTOFF,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "metrics": metrics,
        "per_query": per_query,
    }


def print_report(report: dict, baseline: dict | None) -> None:
    m = report["metrics"]
    print("=" * 62)
    print(f"EVAL HARNESS v{report['harness_version']}  "
          f"seed={report['random_seed']}  queries={m['num_queries']}")
    print(f"timestamp : {report['timestamp_utc']}")
    print(f"git commit: {report['git_commit']}")
    print(f"corpus    : sha256:{report['corpus_sha256'][:12]}  "
          f"dataset: sha256:{report['dataset_sha256'][:12]}")
    print("-" * 62)
    rows = [
        ("P@1", "p_at_1", "{:.4f}"),
        ("MRR@10", "mrr_at_10", "{:.4f}"),
        ("Hallucination rate", "hallucination_rate", "{:.4f}"),
        ("Unsupported-token rate", "unsupported_token_rate", "{:.4f}"),
        ("Latency p50 (ms)", "latency_p50_ms", "{:.3f}"),
        ("Latency p95 (ms)", "latency_p95_ms", "{:.3f}"),
    ]
    if baseline is None:
        for label, key, fmt in rows:
            print(f"{label:<26} {fmt.format(m[key])}")
    else:
        bm = baseline["metrics"]
        print(f"{'metric':<26} {'current':>10} {'baseline':>10} {'delta':>10}")
        for label, key, fmt in rows:
            cur, base = m[key], bm[key]
            delta = cur - base
            print(f"{label:<26} {fmt.format(cur):>10} {fmt.format(base):>10} "
                  f"{'+' if delta >= 0 else ''}{delta:.4f}")
        print(f"(baseline: {baseline['timestamp_utc']}, "
              f"commit {baseline['git_commit'][:12]})")
    print("-" * 62)
    misses = [r for r in report["per_query"] if r["p_at_1"] == 0.0]
    if misses:
        print("P@1 misses:")
        for r in misses:
            print(f"  {r['query_id']}: top1={r['top1']}  gold={r['gold']}  "
                  f"first_gold_rank={r['first_gold_rank']}")
    else:
        print("P@1 misses: none")
    print("=" * 62)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=None,
                        help="prior results JSON to diff against")
    parser.add_argument("--tag", type=str, default="run",
                        help="label for the results filename")
    args = parser.parse_args()

    baseline = None
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        if baseline.get("harness_version") != HARNESS_VERSION:
            print(f"WARNING: baseline harness v{baseline.get('harness_version')} "
                  f"!= current v{HARNESS_VERSION}; deltas may not be comparable.")

    report = evaluate()
    print_report(report, baseline)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{args.tag}_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"results written: {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
