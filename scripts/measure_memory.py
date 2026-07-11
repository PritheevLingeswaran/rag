"""RAM feasibility measurement for Stage 3 (Render free tier = 512MB).

Measures serving-process RSS and OS peak (high-water) step by step while
assembling everything the process must hold at TARGET SCALE (50k chunks),
using the app's real modules (app.core.onnx_text / bm25 / dense):

  1. Python + FastAPI app imported
  2. ONNX embedder loaded + warmed (Xenova/all-MiniLM-L6-v2, int8)
  3. ONNX reranker loaded + warmed (Xenova/ms-marco-MiniLM-L-6-v2, int8)
  4. FAISS index (50k x 384) READ FROM DISK -- the production boot path;
     the file is built in a throwaway subprocess, mimicking ingestion
     happening in a different process/machine
  5. BM25 index built over 50k synthetic chunks + chunk text store
  6. 25-query burst through the full hybrid path, then peak RSS

History (why the code is shaped like this): the first fastembed-based
attempt peaked at 544MB (DOES NOT FIT); raw onnxruntime sessions with
arena+prepacking disabled brought steady state down but inference still
spiked ~300MB because cross-encoder attention activations scale with
batch x seq_len^2 (20x512 batch ~= 250MB). Rerank max_length=256 +
micro_batch=5 (now defaults in app.core.onnx_text) bound that to ~16MB.

Synthetic corpus: Zipfian text, seed 42. Dense vectors are random
normalized -- FAISS memory/latency are independent of vector values.
Windows RSS caveat: Render runs Linux; same order, re-verify on the real
container at deploy stage.

Usage:
    python scripts/measure_memory.py --chunks 50000 [--json out.json]
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DIM = 384
RENDER_CAP_MB = 512


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def peak_mb() -> float:
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / (1024 * 1024)


def synth_corpus(n_chunks: int, seed: int = 42) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    vocab = [f"word{i}" for i in range(30_000)]
    weights = [1.0 / (i + 1) for i in range(len(vocab))]  # Zipf
    docs = []
    for i in range(n_chunks):
        n_tokens = rng.randint(80, 120)
        tokens = rng.choices(vocab, weights=weights, k=n_tokens)
        docs.append((f"synth{i}::c0", " ".join(tokens)))
    return docs


def build_index_file_in_subprocess(n_chunks: int, path: Path) -> None:
    code = (
        "import numpy as np, faiss\n"
        "rng = np.random.default_rng(42)\n"
        f"v = rng.standard_normal(({n_chunks}, {DIM}), dtype=np.float32)\n"
        "v /= np.linalg.norm(v, axis=1, keepdims=True)\n"
        f"ix = faiss.IndexFlatIP({DIM}); ix.add(v)\n"
        f"faiss.write_index(ix, r'{path}')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=50_000)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    index_file = Path(tempfile.gettempdir()) / f"ragp_memgate_{args.chunks}.faiss"
    build_index_file_in_subprocess(args.chunks, index_file)

    steps: list[tuple[str, float, float]] = []

    def record(label: str) -> None:
        gc.collect()
        steps.append((label, rss_mb(), peak_mb()))
        print(f"{label:<62} RSS = {steps[-1][1]:7.1f} MB  peak = {steps[-1][2]:7.1f} MB")

    record("0. bare interpreter + stdlib")

    from app.main import app  # noqa: F401
    record("1. + FastAPI app imported (app.main)")

    from app.core.bootstrap import load_onnx_models
    embedder, reranker = load_onnx_models()
    record("2. + ONNX embedder AND reranker sessions loaded")

    embedder.embed_batch(["warmup query for the embedder"])
    reranker.rerank("warmup", [("c", "a candidate passage to score")])
    record("3. + both models warmed (rerank max_len=256, micro_batch=5)")

    from app.core.dense import DenseIndex
    corpus_ids = [f"synth{i}::c0" for i in range(args.chunks)]
    dense = DenseIndex.from_files(index_file, corpus_ids)
    record(f"4. + FAISS index {args.chunks} x {DIM} read from disk (prod boot)")

    from app.core.bm25 import BM25Index
    from app.core.hybrid import HybridPipeline
    corpus = synth_corpus(args.chunks)
    bm25 = BM25Index()
    bm25.build(corpus)
    queries = [" ".join(text.split()[:6]) for _, text in corpus[:25]]
    texts = dict(corpus)
    del corpus
    record(f"5. + BM25 over {args.chunks} chunks + chunk text store")

    pipeline = HybridPipeline(bm25, dense, embedder, reranker, texts)
    for q in queries:
        result = pipeline.run(q)
        assert result.answer
    record("6. after 25-query burst through full hybrid path")

    peak = peak_mb()
    verdict = "FITS" if peak <= RENDER_CAP_MB * 0.9 else (
        "TIGHT" if peak <= RENDER_CAP_MB else "DOES NOT FIT"
    )
    print("-" * 92)
    print(f"{'peak RSS (OS high-water mark)':<62}                {peak:7.1f} MB")
    print(f"Render free-tier cap: {RENDER_CAP_MB} MB  ->  verdict: {verdict} "
          f"(fits if peak <= 90% of cap)")

    index_file.unlink(missing_ok=True)

    if args.json:
        args.json.write_text(json.dumps({
            "chunks": args.chunks,
            "architecture": "raw onnxruntime, arena off, prepacking off, 1 thread; rerank max_len 256 micro_batch 5; faiss loaded from file",
            "steps": [
                {"label": l, "rss_mb": round(r, 1), "peak_mb": round(p, 1)}
                for l, r, p in steps
            ],
            "peak_rss_mb": round(peak, 1),
            "render_cap_mb": RENDER_CAP_MB,
            "verdict": verdict,
            "platform_caveat": "measured on Windows; re-verify on Linux container before deploy",
        }, indent=2), encoding="utf-8")
        print(f"json written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
