"""End-to-end integration scenarios through the real app stack (real
hybrid pipeline + ONNX models over the real corpus; scripted LLM where a
generation-path behavior is under test, keyless degraded path otherwise).

These are the named Stage 7 integration scenarios; concurrent-at-scale
and quota-exhaustion live in test_concurrency.py / test_quota*.py and
are cross-referenced in docs/stage7_testing.md."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="module")
def real_client():
    """App with the REAL pipeline (60-chunk corpus + ONNX models) built
    by the actual lifespan path, keyless => degraded_no_llm serving."""
    import os

    os.environ["SERVE_PIPELINE"] = "true"
    from app.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        yield client
    os.environ["SERVE_PIPELINE"] = "false"
    get_settings.cache_clear()


def test_happy_path_returns_grounded_cited_answer(real_client):
    resp = real_client.post(
        "/v1/query", json={"query": "how does raft elect a leader"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded_no_llm"   # keyless: explicit
    assert body["citations"] == ["raft::c0"]
    assert "leader election" in body["answer"]
    assert body["rerank_status"] in ("full", "partial", "skipped_budget")
    assert body["retrieved_chunk_ids"][0].startswith("raft")


def test_generation_path_with_llm_validates_citations(real_client):
    """Same stack with a scripted LLM: a partially fabricated answer is
    stripped in-flight and the response says so."""
    from app.generation.llm_client import LLMResponse
    from app.generation.service import GenerationService

    original = real_client.app.state.service

    class PartialFabricator:
        def generate(self, prompt, **kwargs):
            m = re.search(r"\[1\] (.+)", prompt)
            grounded = m.group(1).split(". ")[0]
            return LLMResponse(
                f"{grounded} [1]. The protocol was licensed to Oracle "
                f"for nine million dollars in 1988 [1].",
                "scripted", 100, 30,
            )

    real_client.app.state.service = GenerationService(
        original.pipeline, PartialFabricator()
    )
    try:
        resp = real_client.post(
            "/v1/query",
            json={"query": "when is a raft log entry committed"},
        )
    finally:
        real_client.app.state.service = original

    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "ok_partial_rejected"
    assert "Oracle" not in body["answer"]
    assert "nine million" not in body["answer"]
    assert body["citations"]


def test_query_with_no_lexical_overlap_still_answers(real_client):
    """BM25 finds nothing; dense retrieval still returns nearest
    chunks; the request never 500s."""
    resp = real_client.post(
        "/v1/query", json={"query": "zzzqx wvvbn kkjhg mmnbv"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieved_chunk_ids"]  # dense neighbors served
    assert body["answer"]


def test_empty_corpus_fails_loudly_at_build_not_at_runtime(tmp_path):
    """'Empty index' policy: an empty corpus is refused at BUILD time
    (fail-loud) so a serving process can never exist with an empty
    index and silently answer nothing."""
    from app.core.bootstrap import build_hybrid_from_corpus
    from app.core.corpus import CorpusFormatError

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(CorpusFormatError, match="zero chunks|no documents"):
        build_hybrid_from_corpus(empty)


def test_malformed_query_payloads_rejected_cleanly(real_client):
    # broken JSON
    resp = real_client.post(
        "/v1/query", content=b'{"query": not json',
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    # wrong type
    assert real_client.post(
        "/v1/query", json={"query": ["a", "list"]}
    ).status_code == 422
    # unknown extra field
    assert real_client.post(
        "/v1/query", json={"query": "x", "hack": 1}
    ).status_code == 422


def test_oversized_query_rejected(real_client):
    # over the 2000-char schema bound but under the body-size cap
    assert real_client.post(
        "/v1/query", json={"query": "x" * 2001}
    ).status_code == 422
    # over the 16KB body cap: rejected before parsing
    assert real_client.post(
        "/v1/query", json={"query": "x" * 20_000}
    ).status_code == 413
