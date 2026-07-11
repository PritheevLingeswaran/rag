"""Chunk-level citation validation.

The LLM is instructed to cite retrieved chunks as [1], [2], ... after each
sentence. Before an answer is returned, EVERY sentence is checked against
the chunks it cites (or, if uncited, against all retrieved chunks) using
the same lexical grounding definition as the eval harness's hallucination
metric (app.core.grounding -- one definition, measured and enforced).

Per-sentence verdicts:
    supported          cited chunks lexically support the sentence
    supported_uncited  no citation markers, but supported by the union of
                       retrieved chunks (kept; flagged for transparency)
    invalid_citation   cites a chunk index that does not exist (rejected:
                       a fabricated citation is worse than none)
    unsupported        cited/available context does not support the
                       sentence (rejected -- this is the fabrication case)

Rejected sentences are REMOVED from the answer. The validator never
rewrites text; it only drops and reports. If every sentence is rejected,
the caller (GenerationService) falls back to an extractive answer.

Known tradeoff, on record: grounding is lexical (threshold 0.7 over
content tokens). It is strict against fabricated entities/numbers and
verbatim inventions, but a heavily paraphrased TRUE sentence can be
falsely rejected, and a fabricated sentence built entirely from context
vocabulary can slip through. Deterministic and free beats perfect here;
an LLM-judge pass can be added alongside later (harness version bump).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.grounding import (
    GROUNDING_THRESHOLD,
    content_tokens,
    sentence_coverage,
    split_sentences,
)

_MARKER_RE = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class SentenceVerdict:
    sentence: str
    verdict: str              # supported | supported_uncited | invalid_citation | unsupported
    cited_indices: tuple[int, ...]
    coverage: float


@dataclass(frozen=True)
class ValidationResult:
    validated_answer: str          # only kept sentences, markers preserved
    verdicts: list[SentenceVerdict]
    kept: int
    rejected: int
    citations: list[str] = field(default_factory=list)  # chunk_ids actually cited by kept sentences

    @property
    def all_rejected(self) -> bool:
        return self.kept == 0


class CitationValidator:
    def __init__(self, threshold: float = GROUNDING_THRESHOLD) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold

    def validate(self, answer: str,
                 retrieved: list[tuple[str, str]]) -> ValidationResult:
        """answer: LLM text with [n] markers (1-based, ordered as the
        chunks were presented in the prompt). retrieved: (chunk_id, text)
        in prompt order. Never trusts the answer: every sentence is
        checked before anything is returned."""
        if not retrieved:
            raise ValueError("validate called with no retrieved chunks")

        chunk_tokens = [set(content_tokens(text)) for _, text in retrieved]
        all_tokens: set[str] = set().union(*chunk_tokens)

        verdicts: list[SentenceVerdict] = []
        kept_sentences: list[str] = []
        cited_chunk_ids: list[str] = []

        for sentence in split_sentences(answer):
            markers = tuple(
                int(m) for m in _MARKER_RE.findall(sentence)
            )
            # Sentences with no content tokens (e.g. bare "[1]") are
            # structural, not claims; keep without verdict inflation.
            if not content_tokens(_MARKER_RE.sub("", sentence)):
                continue

            invalid = [m for m in markers if not 1 <= m <= len(retrieved)]
            if invalid:
                verdicts.append(SentenceVerdict(
                    sentence, "invalid_citation", markers, 0.0
                ))
                continue

            if markers:
                support_tokens: set[str] = set().union(
                    *(chunk_tokens[m - 1] for m in markers)
                )
                verdict_if_ok = "supported"
            else:
                support_tokens = all_tokens
                verdict_if_ok = "supported_uncited"

            coverage = sentence_coverage(
                _MARKER_RE.sub("", sentence), support_tokens
            )
            if coverage >= self.threshold:
                verdicts.append(SentenceVerdict(
                    sentence, verdict_if_ok, markers, round(coverage, 4)
                ))
                kept_sentences.append(sentence)
                for m in markers:
                    cid = retrieved[m - 1][0]
                    if cid not in cited_chunk_ids:
                        cited_chunk_ids.append(cid)
            else:
                verdicts.append(SentenceVerdict(
                    sentence, "unsupported", markers, round(coverage, 4)
                ))

        rejected = sum(
            1 for v in verdicts
            if v.verdict in ("unsupported", "invalid_citation")
        )
        return ValidationResult(
            validated_answer=" ".join(kept_sentences).strip(),
            verdicts=verdicts,
            kept=len(kept_sentences),
            rejected=rejected,
            citations=cited_chunk_ids,
        )
