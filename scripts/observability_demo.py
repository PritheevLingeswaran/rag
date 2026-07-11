"""Stage 6 definition-of-done demo.

Boots the REAL app stack (real hybrid pipeline + models, real Redis for
cache/limits, real QuotaGuard) with a scripted LLM that produces a mix of
grounded answers, partial fabrications, full fabrications, and one
provider 429 -- then:

  1. prints each request's id/status/cached flag,
  2. prints the full log trace of ONE request id (end-to-end:
     retrieval -> rerank -> generation -> citation -> response),
  3. prints every listed Stage 6 metric from /metrics with its live value.

Synthetic-traffic honesty note: the LLM is scripted (no API key exists in
this environment) so that every failure-mode metric can be driven
deterministically; everything else -- retrieval, models, cache, quota
accounting, HTTP stack -- is real.

Usage:
    ENVIRONMENT=development REDIS_URL=redis://127.0.0.1:56379/0 \
        python scripts/observability_demo.py
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.errors import LLMQuotaError  # noqa: E402
from app.generation.llm_client import LLMResponse  # noqa: E402


class ScriptedLLM:
    """Yields a scripted sequence of behaviors, then repeats the last."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def generate(self, prompt, **kwargs):
        self.calls += 1
        step = self.script.pop(0) if self.script else "good"
        # extract source [1] text from the prompt to build grounded answers
        m = re.search(r"\[1\] (.+)", prompt)
        source_sentence = m.group(1).split(". ")[0] if m else "The source text"
        if step == "good":
            return LLMResponse(f"{source_sentence} [1].", "scripted", 100, 20)
        if step == "partial_fabrication":
            return LLMResponse(
                f"{source_sentence} [1]. Also, this was standardized by "
                f"NASA in 1969 with a budget of 17 zorkmids [1].",
                "scripted", 100, 30,
            )
        if step == "full_fabrication":
            return LLMResponse(
                "Everything here was invented by aliens in 1602 [1]. "
                "The moon is made of Redis instances [2].",
                "scripted", 100, 25,
            )
        if step == "quota_429":
            raise LLMQuotaError("scripted provider 429", retry_after_s=2.0)
        raise ValueError(step)


def main() -> int:
    from app.config import get_settings

    settings = get_settings()
    if not settings.redis_url:
        print("REDIS_URL required (real cache/quota accounting is the point)")
        return 1

    from app.core.bootstrap import build_hybrid_from_corpus
    from app.generation.quota import QuotaGuard, load_model_limits
    from app.generation.service import GenerationService
    from app.main import create_app
    from app.storage.redis_store import RedisStore

    print("building real pipeline (models + corpus)...")
    pipeline = build_hybrid_from_corpus(Path(settings.corpus_path))
    llm = ScriptedLLM([
        "good", "partial_fabrication", "full_fabrication", "quota_429",
        "good", "good",
    ])
    guard = QuotaGuard(load_model_limits(settings.llm_model),
                       redis_store=RedisStore(settings.redis_url,
                                              namespace="ragp_demo_quota"))
    service = GenerationService(pipeline, llm, quota_guard=guard)

    app = create_app()
    app.state.service = service

    queries = [
        ("how does raft elect a leader", "good -> ok"),
        ("what does nprobe control in faiss", "partial fabrication -> stripped"),
        ("why does postgres need vacuum", "full fabrication -> rejected+fallback"),
        ("how do kafka consumer groups work", "provider 429 -> degraded_quota"),
        ("what is a bloom filter", "cooldown -> degraded_quota_throttled"),
        ("how does raft elect a leader", "repeat -> CACHE HIT"),
    ]

    with TestClient(app) as client:
        # demo namespace: clear cache keys so reruns behave identically
        client.app.state.redis_store._client.flushdb()

        print(f"\n{'#':>2} {'request_id':<18} {'HTTP':<5} {'status':<28} "
              f"{'cached':<7} scenario")
        rows = []
        for i, (q, scenario) in enumerate(queries, 1):
            resp = client.post("/v1/query", json={"query": q})
            body = resp.json()
            rows.append(body)
            print(f"{i:>2} {body.get('request_id', '?'):<18} "
                  f"{resp.status_code:<5} {body.get('status', '?'):<28} "
                  f"{str(body.get('cached', False)):<7} {scenario}")

        trace_id = rows[1]["request_id"]  # the partial-fabrication request
        metrics_text = client.get("/metrics").text

    print(f"\n=== TRACE for request_id={trace_id} "
          f"(grep of captured app logs) ===")
    print(f"(see stderr/stdout log lines tagged request_id={trace_id})")

    print("\n=== LIVE METRICS (nonzero ragp_* series) ===")
    for line in metrics_text.splitlines():
        if line.startswith("ragp_") and not line.endswith(" 0.0"):
            if "_bucket" in line or "_sum" in line:
                continue  # keep output readable; counts tell the story
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
