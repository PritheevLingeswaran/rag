"""GenerationService: retrieve -> prompt -> LLM -> citation-validate.

WHAT THE CLIENT RECEIVES IN EACH FAILURE CASE (the contract; each row is
covered by a test in tests/generation/test_service.py):

| Failure                       | status                      | answer the client gets      |
|-------------------------------|-----------------------------|-----------------------------|
| none (happy path)             | ok                          | validated LLM answer + citations |
| some sentences unsupported    | ok_partial_rejected         | LLM answer minus rejected sentences; rejection counts included |
| ALL sentences rejected        | degraded_citation_rejected  | extractive answer from top chunks + citations |
| OUR quota budget exhausted (proactive, LLM never called) | degraded_quota_throttled | extractive answer; retry_after_s + throttle reason (rpm/rpd/cooldown) |
| provider 429 despite budget   | degraded_quota              | extractive answer; retry_after_s surfaced when known; opens proactive cooldown |
| LLM timeout                   | degraded_timeout            | extractive answer           |
| LLM 5xx / network (after 1 retry) | degraded_llm_error      | extractive answer           |
| LLM malformed/empty response  | degraded_llm_malformed      | extractive answer           |
| LLM config rejected (404 bad model, 400) | degraded_llm_config | extractive answer (ERROR log: our config bug, fix llm_model) |
| LLM auth failure              | degraded_llm_auth           | extractive answer (and an ERROR log: this is a config bug, not weather) |
| no LLM configured             | degraded_no_llm             | extractive answer           |
| nothing retrieved             | no_results                  | explicit "no relevant documents" |

Degradation is always explicit (status + degraded flag on every result)
and never an exception to the caller: retrieval-backed extractive answers
with citations are strictly better than a 500. LLMAuthError is
deliberately NOT retried and logged at ERROR -- it cannot fix itself.

Log-level contract (operators route on these, Stage 4.5):
    INFO  llm_quota_throttled_proactive  expected budget management; no
                                         action needed, capacity working
                                         as designed
    WARN  llm_quota_exhausted            provider 429'd us: accounting
                                         slipped or another consumer
                                         shares the project quota --
                                         investigate consumers/margin
    ERROR llm_server_error_after_retry / llm_malformed_response /
          llm_auth_failure_check_config  actual API failure -- provider
                                         incident or our config bug

The extractive fallback is the first two sentences of the top retrieved
chunk -- the same deterministic path the eval harness has measured since
Stage 0 (hallucination rate ~0 by construction).

Retry policy: exactly one retry, only for LLMServerError (transient by
definition). Quota/timeout are not retried: retrying a 429 burns quota we
do not have, retrying a timeout doubles worst-case latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.grounding import split_sentences
from app.core.hybrid import HybridPipeline, RetrievedChunk
from app.errors import (
    LLMAuthError,
    LLMConfigError,
    LLMMalformedError,
    LLMQuotaError,
    LLMServerError,
    LLMTimeoutError,
)
from app.generation.citations import CitationValidator, ValidationResult
from app.generation.llm_client import LLMClient
from app.logging_config import get_logger

logger = get_logger(__name__)

PROMPT_TEMPLATE = """\
You are a precise assistant answering strictly from the provided sources.

Rules:
- Use ONLY information from the numbered sources below.
- After every sentence, cite the source(s) it came from as [n].
- If the sources do not contain the answer, reply exactly: I don't know.
- Be concise: 1-4 sentences.

Sources:
{sources}

Question: {query}

Answer:"""

IDK_MARKER = "i don't know"


@dataclass(frozen=True)
class GenerationResult:
    query: str
    answer: str
    status: str
    degraded: bool
    citations: list[str]                      # chunk_ids backing the answer
    retrieved_chunk_ids: list[str]
    retrieved_texts: list[str]
    rerank_status: str
    validation: ValidationResult | None = None
    retry_after_s: float | None = None
    llm_model: str | None = None
    extra: dict = field(default_factory=dict)


class GenerationService:
    def __init__(self, pipeline: HybridPipeline,
                 llm: LLMClient | None,
                 validator: CitationValidator | None = None,
                 max_context_chunks: int = 5,
                 quota_guard=None) -> None:
        if max_context_chunks <= 0:
            raise ValueError("max_context_chunks must be positive")
        self.pipeline = pipeline
        self.llm = llm
        self.validator = validator or CitationValidator()
        self.max_context_chunks = max_context_chunks
        self.quota_guard = quota_guard

    # ---- helpers ----

    @staticmethod
    def _extractive_answer(chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
        top = chunks[0]
        sentences = split_sentences(top.text)[:2]
        return " ".join(sentences).strip(), [top.chunk_id]

    def _build_prompt(self, query: str,
                      chunks: list[RetrievedChunk]) -> str:
        sources = "\n".join(
            f"[{i}] {c.text}" for i, c in enumerate(chunks, start=1)
        )
        return PROMPT_TEMPLATE.format(sources=sources, query=query)

    def _degraded(self, query: str, chunks: list[RetrievedChunk],
                  rerank_status: str, status: str,
                  retry_after_s: float | None = None) -> GenerationResult:
        answer, citations = self._extractive_answer(chunks)
        return GenerationResult(
            query=query, answer=answer, status=status, degraded=True,
            citations=citations,
            retrieved_chunk_ids=[c.chunk_id for c in chunks],
            retrieved_texts=[c.text for c in chunks],
            rerank_status=rerank_status,
            retry_after_s=retry_after_s,
        )

    def _call_llm_once_retrying_5xx(self, prompt: str,
                                    max_output_tokens: int | None = None):
        kwargs = {}
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        try:
            return self.llm.generate(prompt, **kwargs)
        except LLMServerError as exc:
            logger.warning("llm_server_error_retrying", error=str(exc))
            return self.llm.generate(prompt, **kwargs)  # second failure propagates

    # ---- entrypoint ----

    def answer(self, query: str,
               max_output_tokens: int | None = None) -> GenerationResult:
        chunks, rerank_info = self.pipeline.retrieve(query)
        if not chunks:
            return GenerationResult(
                query=query,
                answer="No relevant documents were found for this question.",
                status="no_results", degraded=False, citations=[],
                retrieved_chunk_ids=[], retrieved_texts=[],
                rerank_status=rerank_info.status,
            )
        context = chunks[:self.max_context_chunks]

        if self.llm is None:
            return self._degraded(query, context, rerank_info.status,
                                  "degraded_no_llm")

        from app.observability import QUOTA_THROTTLED

        if self.quota_guard is not None:
            decision = self.quota_guard.try_acquire()
            if not decision.allowed:
                QUOTA_THROTTLED.labels(reason=decision.reason).inc()
                # Expected budget management, NOT a failure: INFO level.
                logger.info(
                    "llm_quota_throttled_proactive",
                    reason=decision.reason,
                    retry_after_s=decision.retry_after_s,
                    remaining_rpd=decision.remaining_rpd,
                )
                result = self._degraded(
                    query, context, rerank_info.status,
                    "degraded_quota_throttled",
                    retry_after_s=decision.retry_after_s,
                )
                result.extra["throttle_reason"] = decision.reason
                return result

        from app.observability import ERRORS, LLM_REQUESTS

        prompt = self._build_prompt(query, context)
        import time as _time

        llm_t0 = _time.perf_counter()
        try:
            llm_resp = self._call_llm_once_retrying_5xx(
                prompt, max_output_tokens
            )
        except LLMQuotaError as exc:
            # Provider rejected despite proactive accounting: WARNING,
            # and open a cooldown so we stop asking until it clears.
            LLM_REQUESTS.labels(outcome="quota_429").inc()
            ERRORS.labels(type="llm_quota_429").inc()
            logger.warning("llm_quota_exhausted",
                           retry_after_s=exc.retry_after_s)
            if self.quota_guard is not None:
                self.quota_guard.record_provider_rejection(exc.retry_after_s)
            return self._degraded(query, context, rerank_info.status,
                                  "degraded_quota",
                                  retry_after_s=exc.retry_after_s)
        except LLMTimeoutError as exc:
            LLM_REQUESTS.labels(outcome="timeout").inc()
            ERRORS.labels(type="llm_timeout").inc()
            logger.warning("llm_timeout", error=str(exc))
            return self._degraded(query, context, rerank_info.status,
                                  "degraded_timeout")
        except LLMServerError as exc:
            LLM_REQUESTS.labels(outcome="server_error").inc()
            ERRORS.labels(type="llm_server_error").inc()
            logger.error("llm_server_error_after_retry", error=str(exc))
            return self._degraded(query, context, rerank_info.status,
                                  "degraded_llm_error")
        except LLMMalformedError as exc:
            LLM_REQUESTS.labels(outcome="malformed").inc()
            ERRORS.labels(type="llm_malformed").inc()
            logger.error("llm_malformed_response", error=str(exc))
            return self._degraded(query, context, rerank_info.status,
                                  "degraded_llm_malformed")
        except LLMConfigError as exc:
            LLM_REQUESTS.labels(outcome="config").inc()
            ERRORS.labels(type="llm_config").inc()
            logger.error("llm_config_rejected_check_llm_model", error=str(exc))
            return self._degraded(query, context, rerank_info.status,
                                  "degraded_llm_config")
        except LLMAuthError as exc:
            LLM_REQUESTS.labels(outcome="auth").inc()
            ERRORS.labels(type="llm_auth").inc()
            logger.error("llm_auth_failure_check_config", error=str(exc))
            return self._degraded(query, context, rerank_info.status,
                                  "degraded_llm_auth")
        LLM_REQUESTS.labels(outcome="ok").inc()
        llm_ms = round((_time.perf_counter() - llm_t0) * 1000, 1)

        if llm_resp.text.strip().lower().rstrip(".") == IDK_MARKER:
            return GenerationResult(
                query=query, answer="I don't know.",
                status="ok_no_answer", degraded=False, citations=[],
                retrieved_chunk_ids=[c.chunk_id for c in context],
                retrieved_texts=[c.text for c in context],
                rerank_status=rerank_info.status,
                llm_model=llm_resp.model,
            )

        validation = self.validator.validate(
            llm_resp.text, [(c.chunk_id, c.text) for c in context]
        )
        from app.observability import (
            CITATION_REJECTED_ANSWERS,
            CITATION_SENTENCES,
        )

        for verdict in validation.verdicts:
            CITATION_SENTENCES.labels(verdict=verdict.verdict).inc()
        logger.info(
            "generation_completed",
            llm_ms=llm_ms, llm_model=llm_resp.model,
            output_tokens=llm_resp.output_tokens,
            sentences_kept=validation.kept,
            sentences_rejected=validation.rejected,
        )
        if validation.all_rejected:
            CITATION_REJECTED_ANSWERS.inc()
            logger.warning(
                "citation_validation_rejected_all",
                rejected=validation.rejected,
                verdicts=[v.verdict for v in validation.verdicts],
            )
            answer, citations = self._extractive_answer(context)
            return GenerationResult(
                query=query, answer=answer,
                status="degraded_citation_rejected", degraded=True,
                citations=citations,
                retrieved_chunk_ids=[c.chunk_id for c in context],
                retrieved_texts=[c.text for c in context],
                rerank_status=rerank_info.status,
                validation=validation, llm_model=llm_resp.model,
            )

        status = "ok" if validation.rejected == 0 else "ok_partial_rejected"
        if validation.rejected:
            logger.info(
                "citation_validation_rejected_some",
                kept=validation.kept, rejected=validation.rejected,
            )
        return GenerationResult(
            query=query, answer=validation.validated_answer,
            status=status, degraded=False,
            citations=validation.citations,
            retrieved_chunk_ids=[c.chunk_id for c in context],
            retrieved_texts=[c.text for c in context],
            rerank_status=rerank_info.status,
            validation=validation, llm_model=llm_resp.model,
        )
