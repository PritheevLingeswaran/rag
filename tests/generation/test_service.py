"""GenerationService tests: every row of the failure-mode table in
app/generation/service.py is exercised here with fake LLMs and a stub
retrieval pipeline -- no network, no models."""

import pytest

from app.errors import (
    LLMAuthError,
    LLMConfigError,
    LLMMalformedError,
    LLMQuotaError,
    LLMServerError,
    LLMTimeoutError,
)  # noqa: F401 - all raised in the failure-taxonomy parametrize
from app.core.hybrid import RerankInfo, RetrievedChunk
from app.generation.llm_client import LLMResponse
from app.generation.service import GenerationResult, GenerationService

CHUNKS = [
    RetrievedChunk(
        "raft::c0",
        "A follower that hears no heartbeat becomes a candidate and "
        "requests votes; a candidate that receives votes from a majority "
        "becomes leader.",
        9.0, "rerank",
    ),
    RetrievedChunk(
        "raft::c1",
        "The Raft leader replicates log entries to followers. An entry "
        "is committed once replicated on a majority of servers.",
        7.0, "rerank",
    ),
]


class StubPipeline:
    def __init__(self, chunks=CHUNKS, rerank_status="full"):
        self.chunks = chunks
        self.rerank_status = rerank_status

    def retrieve(self, query):
        return self.chunks, RerankInfo(self.rerank_status,
                                       len(self.chunks), len(self.chunks), 1.0)


class FakeLLM:
    def __init__(self, text=None, errors=None, model="fake-model"):
        self.text = text
        self.errors = list(errors or [])
        self.model = model
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return LLMResponse(self.text, self.model, 100, 20)


def make_service(llm, pipeline=None) -> GenerationService:
    return GenerationService(pipeline or StubPipeline(), llm)


# ---- happy path ----

def test_happy_path_returns_validated_answer_with_citations():
    llm = FakeLLM("A follower becomes a candidate and requests votes [1].")
    result = make_service(llm).answer("how does raft elect a leader?")
    assert result.status == "ok"
    assert result.degraded is False
    assert result.citations == ["raft::c0"]
    assert result.answer.startswith("A follower becomes a candidate")
    assert result.rerank_status == "full"
    assert result.llm_model == "fake-model"


def test_fabricated_sentence_is_stripped_status_partial():
    llm = FakeLLM(
        "A follower becomes a candidate and requests votes [1]. "
        "Raft clusters are limited to exactly 42 nodes maximum [2]."
    )
    result = make_service(llm).answer("q")
    assert result.status == "ok_partial_rejected"
    assert result.degraded is False
    assert "42" not in result.answer
    assert result.validation.rejected == 1


def test_fully_fabricated_answer_falls_back_to_extractive():
    llm = FakeLLM("Raft was invented by aliens in 1847 [1].")
    result = make_service(llm).answer("q")
    assert result.status == "degraded_citation_rejected"
    assert result.degraded is True
    # extractive fallback comes verbatim from the top retrieved chunk
    assert result.answer.startswith("A follower that hears no heartbeat")
    assert result.citations == ["raft::c0"]
    assert result.validation.all_rejected


# ---- LLM failure taxonomy: what the client receives ----

@pytest.mark.parametrize("error,expected_status", [
    (LLMQuotaError("quota", retry_after_s=30.0), "degraded_quota"),
    (LLMTimeoutError("slow"), "degraded_timeout"),
    (LLMMalformedError("garbage"), "degraded_llm_malformed"),
    (LLMConfigError("model not found (HTTP 404)"), "degraded_llm_config"),
    (LLMAuthError("bad key"), "degraded_llm_auth"),
])
def test_llm_failures_degrade_to_extractive_with_explicit_status(
    error, expected_status
):
    result = make_service(FakeLLM(errors=[error])).answer("q")
    assert result.status == expected_status
    assert result.degraded is True
    assert result.answer.startswith("A follower that hears no heartbeat")
    assert result.citations == ["raft::c0"]
    assert result.retrieved_chunk_ids  # retrieval still delivered


# ---- secondary-provider fallback ----

class DenyingGuard:
    def try_acquire(self):
        from app.generation.quota import REASON_RPD, QuotaDecision
        return QuotaDecision(False, REASON_RPD, 0, 0, 3600.0)

    def record_provider_rejection(self, retry_after_s):
        pass


def test_primary_failure_served_by_fallback_as_ok():
    primary = FakeLLM(errors=[LLMServerError("boom"), LLMServerError("boom")])
    fallback = FakeLLM("A follower becomes a candidate and requests votes [1].",
                       model="fallback-model")
    service = GenerationService(StubPipeline(), primary,
                                fallback_llm=fallback)
    result = service.answer("q")
    assert result.status == "ok"
    assert result.degraded is False
    assert result.llm_model == "fallback-model"
    assert fallback.calls == 1


def test_primary_quota_exhausted_served_by_fallback():
    """The main capacity win: primary budget gone (450 RPD) must not
    mean a degraded day when the fallback has 14,400 RPD spare."""
    primary = FakeLLM("never called")
    fallback = FakeLLM("A follower becomes a candidate and requests votes [1].",
                       model="fallback-model")
    service = GenerationService(StubPipeline(), primary,
                                quota_guard=DenyingGuard(),
                                fallback_llm=fallback)
    result = service.answer("q")
    assert result.status == "ok"
    assert result.llm_model == "fallback-model"
    assert primary.calls == 0          # guard denied before the call
    assert fallback.calls == 1


def test_both_providers_failing_degrades_with_primary_status():
    primary = FakeLLM(errors=[LLMTimeoutError("slow")])
    fallback = FakeLLM(errors=[LLMServerError("down"), LLMServerError("down")])
    service = GenerationService(StubPipeline(), primary,
                                fallback_llm=fallback)
    result = service.answer("q")
    assert result.status == "degraded_timeout"   # PRIMARY's failure, truthfully
    assert result.degraded is True
    assert result.answer.startswith("A follower that hears no heartbeat")


def test_fallback_guard_denial_degrades_like_no_fallback():
    primary = FakeLLM(errors=[LLMTimeoutError("slow")])
    fallback = FakeLLM("unused")
    service = GenerationService(StubPipeline(), primary,
                                fallback_llm=fallback,
                                fallback_quota_guard=DenyingGuard())
    result = service.answer("q")
    assert result.status == "degraded_timeout"
    assert fallback.calls == 0


def test_no_fallback_behavior_unchanged():
    result = make_service(FakeLLM(errors=[LLMTimeoutError("slow")])).answer("q")
    assert result.status == "degraded_timeout"


def test_quota_error_surfaces_retry_after():
    err = LLMQuotaError("quota", retry_after_s=42.0)
    result = make_service(FakeLLM(errors=[err])).answer("q")
    assert result.retry_after_s == 42.0


def test_server_error_retried_once_then_succeeds():
    llm = FakeLLM(
        "An entry is committed once replicated on a majority of servers [2].",
        errors=[LLMServerError("blip")],
    )
    result = make_service(llm).answer("q")
    assert llm.calls == 2
    assert result.status == "ok"


def test_server_error_twice_degrades():
    llm = FakeLLM(errors=[LLMServerError("down"), LLMServerError("down")])
    result = make_service(llm).answer("q")
    assert llm.calls == 2
    assert result.status == "degraded_llm_error"
    assert result.degraded is True


def test_quota_error_is_never_retried():
    llm = FakeLLM(errors=[LLMQuotaError("quota")])
    make_service(llm).answer("q")
    assert llm.calls == 1


def test_no_llm_configured_serves_extractive_explicitly():
    result = make_service(llm=None).answer("q")
    assert result.status == "degraded_no_llm"
    assert result.degraded is True
    assert result.answer.startswith("A follower that hears no heartbeat")


# ---- edge paths ----

def test_no_retrieval_results():
    result = make_service(FakeLLM("x"), StubPipeline(chunks=[])).answer("q")
    assert result.status == "no_results"
    assert result.citations == []
    assert "No relevant documents" in result.answer


def test_idk_answer_passes_through_unvalidated():
    result = make_service(FakeLLM("I don't know.")).answer("q")
    assert result.status == "ok_no_answer"
    assert result.answer == "I don't know."
    assert result.citations == []


def test_rerank_status_propagates_to_response():
    pipeline = StubPipeline(rerank_status="skipped_budget")
    llm = FakeLLM("A follower becomes a candidate and requests votes [1].")
    result = make_service(llm, pipeline).answer("q")
    assert result.rerank_status == "skipped_budget"
