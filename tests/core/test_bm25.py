import pytest

from app.core.bm25 import BM25Index, tokenize


def test_tokenize_lowercases_and_splits_on_non_alnum():
    assert tokenize("Hello, World! v2") == ["hello", "world", "v2"]


def test_search_ranks_matching_docs_above_unrelated():
    idx = BM25Index()
    idx.build([
        ("a", "cats are great pets"),
        ("b", "dogs are loyal companions"),
        ("c", "cats and dogs both make great pets"),
    ])
    results = idx.search("cats pets", top_k=3)
    result_ids = [doc_id for doc_id, _ in results]
    assert result_ids[0] in ("a", "c")
    assert "b" not in result_ids
    assert all(score > 0 for _, score in results)


def test_search_returns_empty_for_unknown_terms():
    idx = BM25Index()
    idx.build([("a", "hello world")])
    assert idx.search("zzz nonexistent") == []


def test_build_called_twice_raises():
    idx = BM25Index()
    idx.build([("a", "x")])
    with pytest.raises(RuntimeError):
        idx.build([("b", "y")])


def test_search_before_build_raises():
    idx = BM25Index()
    with pytest.raises(RuntimeError):
        idx.search("x")
