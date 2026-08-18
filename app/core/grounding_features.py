"""Features for a learned grounding classifier.

Every feature is a pure function of (sentence, context, position) and uses
only the tokenizer already shared by retrieval and the citation validator,
so a model trained on these can run in the serving path with no new
dependency and no extra model download.

Feature 0 is the shipped lexical coverage. The learned model therefore
strictly generalises the current heuristic: it can reproduce it by
weighting that feature alone.
"""

from __future__ import annotations

import re

from app.core.grounding import content_tokens

FEATURE_NAMES = (
    "coverage",
    "bigram_coverage",
    "longest_shared_run",
    "numeral_coverage",
    "has_numeral",
    "long_token_coverage",
    "capitalised_coverage",
    "missing_count",
    "token_count",
    "position",
    "is_first",
    "context_size",
)

_NUMERAL_RE = re.compile(r"\d[\d,.]*")
_CAP_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}")


def _longest_shared_run(tokens: list[str], ctx: set[str]) -> float:
    best = run = 0
    for t in tokens:
        run = run + 1 if t in ctx else 0
        best = max(best, run)
    return best / len(tokens) if tokens else 0.0


def _ratio(present: int, total: int, default: float = 1.0) -> float:
    return present / total if total else default


def extract(sentence: str, context: str, index: int = 0,
            total: int = 1, ctx_tokens: set[str] | None = None,
            ctx_bigrams: set[tuple[str, str]] | None = None) -> list[float]:
    """Feature vector for one sentence against its context.

    ctx_tokens / ctx_bigrams may be passed in when scoring many sentences
    against the same context, which is the serving case.
    """
    if ctx_tokens is None:
        ctx_tokens = set(content_tokens(context))
    toks = content_tokens(sentence)
    n = len(toks)
    if n == 0:
        return [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                index / max(total, 1), 1.0 if index == 0 else 0.0,
                min(len(ctx_tokens) / 1000.0, 1.0)]

    present = sum(1 for t in toks if t in ctx_tokens)
    missing = n - present

    if ctx_bigrams is None:
        ctx_list = content_tokens(context)
        ctx_bigrams = set(zip(ctx_list, ctx_list[1:]))
    bigrams = list(zip(toks, toks[1:]))
    bigram_present = sum(1 for b in bigrams if b in ctx_bigrams)

    sent_nums = _NUMERAL_RE.findall(sentence)
    ctx_nums = set(_NUMERAL_RE.findall(context))
    num_present = sum(1 for x in sent_nums if x in ctx_nums)

    long_toks = [t for t in toks if len(t) >= 8]
    long_present = sum(1 for t in long_toks if t in ctx_tokens)

    caps = [c.lower() for c in _CAP_RE.findall(sentence)]
    cap_present = sum(1 for c in caps if c in ctx_tokens)

    return [
        present / n,
        _ratio(bigram_present, len(bigrams)),
        _longest_shared_run(toks, ctx_tokens),
        _ratio(num_present, len(sent_nums)),
        1.0 if sent_nums else 0.0,
        _ratio(long_present, len(long_toks)),
        _ratio(cap_present, len(caps)),
        min(missing / 10.0, 1.0),
        min(n / 40.0, 1.0),
        index / max(total, 1),
        1.0 if index == 0 else 0.0,
        min(len(ctx_tokens) / 1000.0, 1.0),
    ]


def extract_response(sentences: list[str], context: str) -> list[list[float]]:
    """Feature vectors for every sentence of one response, sharing the
    context tokenisation across them."""
    ctx_tokens = set(content_tokens(context))
    ctx_list = content_tokens(context)
    ctx_bigrams = set(zip(ctx_list, ctx_list[1:]))
    total = len(sentences)
    return [
        extract(s, context, i, total, ctx_tokens, ctx_bigrams)
        for i, s in enumerate(sentences)
    ]
