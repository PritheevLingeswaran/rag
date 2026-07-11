"""Reciprocal Rank Fusion.

Isolated module: pure functions, stdlib only.

RRF score of a document d over ranking lists R:
    score(d) = sum over r in R where d appears:  1 / (k + rank_r(d))
with rank starting at 1. k=60 is the constant from the original paper
(Cormack et al., 2009); it dampens the dominance of top ranks. RRF is
rank-based, so it needs no score normalization between BM25 and cosine
similarities -- which is exactly why it is the standard fusion choice for
hybrid lexical+dense retrieval.

Deterministic tie-breaking: equal fused scores order by (best single rank,
then first-list-order) so results never depend on dict iteration order.
"""

from __future__ import annotations

DEFAULT_RRF_K = 60


def rrf_fuse(
    rankings: list[list[str]],
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into one list of (id, rrf_score), best first.

    Empty input lists are allowed (contribute nothing). Duplicate ids
    within one ranking are an error: they signal an upstream bug.
    """
    if k <= 0:
        raise ValueError("rrf k must be positive")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    arrival: dict[str, int] = {}
    counter = 0
    for ranking in rankings:
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:
                raise ValueError(
                    f"duplicate id {doc_id!r} within a single ranking"
                )
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in best_rank or rank < best_rank[doc_id]:
                best_rank[doc_id] = rank
            if doc_id not in arrival:
                arrival[doc_id] = counter
                counter += 1
    fused = sorted(
        scores.items(),
        key=lambda item: (-item[1], best_rank[item[0]], arrival[item[0]]),
    )
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        fused = fused[:top_k]
    return fused
