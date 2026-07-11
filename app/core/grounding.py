"""Lexical grounding: is a sentence supported by a context text?

THE single definition of "supported", shared by the eval harness's
hallucination metric and the citation validator (app/generation). Moving
it here changed no behavior -- the harness re-run after the refactor is
bit-identical (see eval/results/). Changing anything in this module
invalidates comparison with prior eval runs and requires a harness
version bump (eval/run_eval.py contract).

Definition (harness contract v1.0): a sentence is supported by a context
iff >= GROUNDING_THRESHOLD of its content tokens (alphanumeric,
lowercased, stopwords removed) appear in the context's content-token set.
This is a deterministic lexical proxy: strict against verbatim
fabrication, blind to fluent paraphrase; the tradeoff is documented in
README and revisited when an LLM-judge metric can run alongside.
"""

from __future__ import annotations

import re

from app.core.bm25 import tokenize

GROUNDING_THRESHOLD = 0.7

STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or "
    "that the to was were what when where which who why with does do did "
    "not no can could should would".split()
)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_RE.split(text) if s.strip()]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


def sentence_coverage(sentence: str, context_tokens: set[str]) -> float:
    """Fraction of the sentence's content tokens present in the context.
    Sentences with no content tokens return 1.0 (nothing to fabricate)."""
    toks = content_tokens(sentence)
    if not toks:
        return 1.0
    present = sum(1 for t in toks if t in context_tokens)
    return present / len(toks)


def is_supported(sentence: str, context: str,
                 threshold: float = GROUNDING_THRESHOLD) -> bool:
    return sentence_coverage(sentence, set(content_tokens(context))) >= threshold
