"""RRF unit tests: pure function, hand-computed expectations."""

import pytest

from app.core.rrf import rrf_fuse


def test_hand_computed_scores():
    fused = rrf_fuse([["a", "b"], ["b", "a"]], k=60)
    # a: 1/61 + 1/62 ; b: 1/62 + 1/61  -> tie
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(scores["a"])


def test_doc_in_both_lists_beats_doc_in_one():
    fused = rrf_fuse([["x", "y"], ["x", "z"]])
    assert fused[0][0] == "x"


def test_tie_breaks_by_best_rank_then_arrival():
    # 'a' and 'b' tie on score; 'a' has best rank 1, 'b' best rank 1 too,
    # but 'a' arrived first -> deterministic order a, b.
    fused = rrf_fuse([["a"], ["b"]], k=60)
    assert [cid for cid, _ in fused] == ["a", "b"]


def test_rank_positions_matter():
    fused = rrf_fuse([["a", "b", "c"], ["c", "b", "a"]], k=60)
    scores = dict(fused)
    # b is rank 2 in both; a and c are rank 1 + rank 3
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["b"] == pytest.approx(2 / 62)
    assert fused[0][0] in ("a", "c")  # 1/61+1/63 > 2/62


def test_empty_rankings_allowed():
    assert rrf_fuse([[], []]) == []
    assert rrf_fuse([["a"], []])[0][0] == "a"


def test_top_k_truncates():
    fused = rrf_fuse([["a", "b", "c", "d"]], top_k=2)
    assert len(fused) == 2


def test_duplicate_in_single_ranking_raises():
    with pytest.raises(ValueError, match="duplicate"):
        rrf_fuse([["a", "a"]])


def test_invalid_k_raises():
    with pytest.raises(ValueError, match="k must be positive"):
        rrf_fuse([["a"]], k=0)
