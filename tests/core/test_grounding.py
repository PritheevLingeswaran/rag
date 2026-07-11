"""Unit tests for app/core/grounding.py -- the shared support
definition. These pin the harness-contract semantics: any behavior
change here invalidates eval comparability, so every edge is nailed."""

import pytest

from app.core.grounding import (
    GROUNDING_THRESHOLD,
    content_tokens,
    is_supported,
    sentence_coverage,
    split_sentences,
)


def test_threshold_is_the_contract_value():
    assert GROUNDING_THRESHOLD == 0.7


def test_split_sentences_on_terminators():
    assert split_sentences("One. Two! Three? Four.") == [
        "One.", "Two!", "Three?", "Four.",
    ]


def test_split_sentences_ignores_blank_fragments():
    assert split_sentences("Only one sentence.") == ["Only one sentence."]
    assert split_sentences("") == []


def test_content_tokens_drop_stopwords_and_case():
    assert content_tokens("The Cat IS on a mat") == ["cat", "mat"]


def test_coverage_fraction_exact():
    ctx = set(content_tokens("the token bucket refills at a fixed rate"))
    # 'token bucket refills quickly': 4 content tokens, 3 in context
    assert sentence_coverage("token bucket refills quickly", ctx) == 0.75


def test_no_content_tokens_counts_as_supported():
    # nothing claimable => nothing fabricated
    assert sentence_coverage("of the and", set()) == 1.0


def test_is_supported_threshold_boundary():
    ctx = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    # 7 of 10 tokens present = exactly 0.7 -> supported (>= threshold)
    s7 = "alpha beta gamma delta epsilon zeta eta xxx yyy zzz"
    assert is_supported(s7, ctx) is True
    # 6 of 10 = 0.6 -> not supported
    s6 = "alpha beta gamma delta epsilon zeta www xxx yyy zzz"
    assert is_supported(s6, ctx) is False


def test_custom_threshold_respected():
    assert is_supported("alpha zzz", "alpha beta", threshold=0.5) is True
    assert is_supported("alpha zzz", "alpha beta", threshold=0.51) is False
