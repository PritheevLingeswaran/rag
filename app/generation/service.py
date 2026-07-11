"""GenerationService: retrieve -> prompt -> LLM -> citation-validate.

WHAT THE CLIENT RECEIVES IN EACH FAILURE CASE (the contract; each row is
covered by a test in tests/generation/test_service.py):

| Failure                       | status                      | answer the client gets      |
|-------------------------------|-----------------------------|-----------------------------|
| none (happy path)             | ok                          | validated LLM answer + citations |
| some sentences unsupported    | ok_partial_rejected         | LLM answer minus rejected sentences; rejection counts included |
| ALL sentences rejected        | degraded_citation_rejected  | extractive answer from top chunks + citations |
| LLM quota exhausted (429)     | degraded_quota              | extractive answer; retry_after_s surfaced when known |
| LLM timeout                   | degraded_timeout            | extractive answer           |
| LLM 5xx / network (after 1 retry) | degraded_llm_error      | extractive answer           |
| LLM malformed/empty response  | degraded_llm_malformed      | extractive answer           |
| LLM auth failure              | degraded_llm_auth           | extractive answer (and an ERROR log: this is a config bug, not weather) |
| no LLM configured             | degraded_no_llm             | extractive answer           |
| nothing retrieved             | no_results                  | explicit "no relevant documents" |

Degradation is always explicit (status + degraded flag on every result)
and never an exception to the caller: retrieval-backed extractive answers
with citations are strictly better than a 500. LLMAuthError is
deliberately NOT retried and logged at ERROR -- it cannot fix itself.

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
                 max_context_chunks: int = 5) -> None:
        if max_context_chunks <= 0:
            raise ValueError("max_context_chunks must be positive")
        self.pipeline = pipeline
        self.llm = llm
        self.validator = validator or CitationValidator()
        self.max_context_chunks = max_context_chunks

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

    def _call_llm_once_retrying_5xx(self, prompt: str):
        try:
            return self.llm.generate(prompt)
        except LLMServerError as exc:
            logger.warning("llm_server_error_retrying", error=str(exc))
            return self.llm.generate(prompt)  # second failure propagates

    # ---- entrypoint ----

    def answer(self, query: str) -> GenerationResult:
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

        prompt = self._build_prompt(query, context)
        try:
            llm_resp = self._call_llm_once_retrying_5xx(prompt)
        except LLMQuotaError as exc:
            logger.warning("llm_quota_exhausted",
                           retry_after_s=exc.retry_after_s)
            return self._degraded(query, context, rerank_info.status,
                                  "degraded_quota",
                                  retry_after_s=exc.retry_after_s)
        except LLMTimeoutError as exc:
            logger.warning("llm_timeout", error=str(exc))
            return self._degraded(query, context, rerank_info.status,
                                  "degraded_timeout")
        except LLMServerError as exc:
            logger.error("llm_server_error_after_retry", error=str(exc))
            return self._degraded(query, context, rerank_info.status,
                                  "degraded_llm_error")
        except LLMMalformedError as exc:
            logger.error("llm_malformed_response", error=str(exc))
            return self._degraded(query, context, rerank_info.status,
                                  "degraded_llm_malformed")
        except LLMAuthError as exc:
            logger.error("llm_auth_failure_check_config", error=str(exc))
            return self._degraded(query, context, rerank_info.status,
                                  "degraded_llm_auth")

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
        if validation.all_rejected:
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
