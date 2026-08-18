"""Feature extraction and the trained grounding classifier.

The Stage 12 claim is that a learned model beats the shipped heuristic on
held-out data. These tests cover the two things that claim rests on: the
features behave as described, and the serialised model scores identically
to the numpy training code that produced it.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from app.core.grounding_features import FEATURE_NAMES, extract, extract_response

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "eval" / "grounding_model_v1.json"

_spec = importlib.util.spec_from_file_location(
    "validate_grounding", REPO_ROOT / "eval" / "validate_grounding.py"
)
vg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vg)

CONTEXT = (
    "Raft elects a leader using randomized election timeouts. "
    "A candidate wins once it collects votes from a majority of the cluster."
)


def test_feature_vector_length_matches_names():
    assert len(extract("Raft elects a leader.", CONTEXT)) == len(FEATURE_NAMES)


def test_coverage_feature_reproduces_the_shipped_heuristic():
    verbatim = extract("Raft elects a leader using randomized timeouts", CONTEXT)
    invented = extract("Oracle paid nine million dollars in 1988", CONTEXT)
    assert verbatim[0] == 1.0
    assert invented[0] < 0.5


def test_numeral_features_flag_unsupported_figures():
    idx = FEATURE_NAMES.index("numeral_coverage")
    has = FEATURE_NAMES.index("has_numeral")
    invented = extract("The cluster held 4096 nodes", CONTEXT)
    assert invented[has] == 1.0
    assert invented[idx] == 0.0
    none = extract("Raft elects a leader", CONTEXT)
    assert none[has] == 0.0
    assert none[idx] == 1.0


def test_longest_shared_run_rewards_verbatim_spans():
    idx = FEATURE_NAMES.index("longest_shared_run")
    verbatim = extract("Raft elects a leader using randomized election timeouts",
                       CONTEXT)
    scattered = extract("Leader zebra timeouts zebra cluster", CONTEXT)
    assert verbatim[idx] > scattered[idx]


def test_sentence_without_content_tokens_is_not_flagged():
    v = extract("The of and.", CONTEXT)
    assert v[0] == 1.0


def test_extract_response_matches_per_sentence_extraction():
    sents = ["Raft elects a leader.", "Oracle paid nine million dollars."]
    batched = extract_response(sents, CONTEXT)
    for i, s in enumerate(sents):
        assert batched[i] == pytest.approx(
            extract(s, CONTEXT, i, len(sents)), abs=1e-12)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained")
def test_model_file_is_well_formed_and_trained_off_the_test_split():
    m = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    assert m["feature_names"] == list(FEATURE_NAMES)
    assert len(m["coefficients"]) == len(FEATURE_NAMES)
    assert len(m["mean"]) == len(m["std"]) == len(FEATURE_NAMES)
    assert 0.0 < m["threshold"] < 1.0
    assert all(s > 0 for s in m["std"])
    # The held-out sample must not appear anywhere in the training record.
    assert "ragbench_sample_v1" not in json.dumps(m["training"])


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained")
def test_scoring_is_deterministic_and_ranks_fabrication_higher():
    m = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    sents = [
        "Raft elects a leader using randomized election timeouts.",
        "Oracle licensed the protocol for nine million dollars in 1988.",
    ]
    a = vg.score_learned(m, sents, CONTEXT)
    b = vg.score_learned(m, sents, CONTEXT)
    assert a == b
    assert all(0.0 <= p <= 1.0 for p in a)
    assert a[1] > a[0]


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained")
def test_serialised_model_matches_a_hand_computed_logistic_score():
    m = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    sent = "Raft elects a leader using randomized election timeouts."
    vec = extract_response([sent], CONTEXT)[0]
    z = sum(((v - mu) / sd) * w
            for v, mu, sd, w in zip(vec, m["mean"], m["std"],
                                    m["coefficients"])) + m["intercept"]
    expected = 1.0 / (1.0 + math.exp(-z))
    assert vg.score_learned(m, [sent], CONTEXT)[0] == pytest.approx(expected)
