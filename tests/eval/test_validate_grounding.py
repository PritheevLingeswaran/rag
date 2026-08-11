"""Scoring maths for the grounding validation harness.

The headline numbers in docs/stage10_grounding_validation.md are only
worth as much as these four functions, so they get exact-value tests
rather than smoke tests.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "validate_grounding", REPO_ROOT / "eval" / "validate_grounding.py"
)
vg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vg)


def test_confusion_counts_unsupported_as_the_positive_class():
    # (predicted_positive, actual_positive)
    pairs = [(True, True), (True, True), (True, False),
             (False, True), (False, False), (False, False)]
    c = vg.confusion(pairs)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 2)
    assert c["precision"] == round(2 / 3, 4)
    assert c["recall"] == round(2 / 3, 4)
    assert c["f1"] == round(2 / 3, 4)
    assert c["accuracy"] == round(4 / 6, 4)


def test_confusion_is_zero_not_nan_when_nothing_is_predicted():
    c = vg.confusion([(False, True), (False, False)])
    assert c["precision"] == 0.0 and c["f1"] == 0.0


def test_auc_is_one_for_perfect_separation_and_half_for_ties():
    perfect = [(0.9, True), (0.8, True), (0.2, False), (0.1, False)]
    assert vg.auc(perfect) == 1.0
    inverted = [(0.1, True), (0.2, True), (0.8, False), (0.9, False)]
    assert vg.auc(inverted) == 0.0
    all_tied = [(0.5, True), (0.5, False)]
    assert vg.auc(all_tied) == 0.5


def test_auc_is_none_without_both_classes():
    assert vg.auc([(0.4, True), (0.6, True)]) is None


def test_best_f1_finds_the_separating_threshold():
    scored = [(0.9, True), (0.8, True), (0.2, False), (0.1, False)]
    best = vg.best_f1(scored)
    assert best["f1"] == 1.0
    assert 0.2 <= best["threshold"] < 0.8


def test_sentence_coverage_ratio_matches_the_shipped_definition():
    ctx = set(vg.content_tokens("raft elects a leader using randomized timeouts"))
    # every content token present
    assert vg.sentence_coverage_ratio("raft elects a leader", ctx) == 1.0
    # no content tokens at all -> None (nothing to check, not a failure)
    assert vg.sentence_coverage_ratio("the of and", ctx) is None
    # a fabricated entity drags coverage below the shipped threshold
    low = vg.sentence_coverage_ratio(
        "Oracle licensed nine million dollars quarterly revenue Zanzibar", ctx)
    assert low is not None and low < vg.GROUNDING_THRESHOLD


def test_evaluate_end_to_end_on_a_tiny_hand_built_sample():
    rows = [
        {   # response reuses source vocabulary => predicted supported
            "documents": ["Raft elects a leader using randomized timeouts."],
            "response_sentences": [["a", "Raft elects a leader using randomized timeouts."]],
            "unsupported_response_sentence_keys": [],
            "adherence_score": True,
            "dataset_name": "toy",
            "ragas_faithfulness": 1.0,
        },
        {   # fabricated content => predicted unsupported, labelled unsupported
            "documents": ["Raft elects a leader using randomized timeouts."],
            "response_sentences": [["a", "Oracle purchased Raft for nine million dollars in 1988."]],
            "unsupported_response_sentence_keys": ["a"],
            "adherence_score": False,
            "dataset_name": "toy",
            "ragas_faithfulness": 0.0,
        },
    ]
    r = vg.evaluate(rows)
    assert r["sentence_level"]["n"] == 2
    assert r["sentence_level"]["positives"] == 1
    assert r["sentence_level"]["at_shipped_threshold"]["tp"] == 1
    assert r["sentence_level"]["at_shipped_threshold"]["fn"] == 0
    ex = r["example_level"]
    assert ex["n"] == 2 and ex["positives"] == 1
    assert ex["methods"]["ragp_lexical_proxy"]["at_shipped_threshold"]["f1"] == 1.0
    # a baseline present on every row is scored on every row
    assert ex["methods"]["ragas_faithfulness"]["n"] == 2
