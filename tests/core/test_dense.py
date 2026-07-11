"""DenseIndex unit tests: isolated, synthetic vectors only, no app deps."""

import numpy as np
import pytest

from app.core.dense import DenseIndex


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture()
def index() -> DenseIndex:
    vectors = np.stack([
        unit([1.0, 0.0, 0.0]),
        unit([0.0, 1.0, 0.0]),
        unit([0.0, 0.0, 1.0]),
        unit([1.0, 1.0, 0.0]),
    ])
    return DenseIndex.from_vectors(vectors, ["a", "b", "c", "d"])


def test_exact_nearest_neighbor(index):
    results = index.search(unit([1.0, 0.1, 0.0]), top_k=2)
    assert results[0][0] == "a"
    assert results[1][0] == "d"
    assert results[0][1] > results[1][1]


def test_scores_are_cosine_similarities(index):
    results = index.search(unit([0.0, 1.0, 0.0]), top_k=4)
    by_id = dict(results)
    assert by_id["b"] == pytest.approx(1.0, abs=1e-6)
    assert by_id["d"] == pytest.approx(1.0 / np.sqrt(2), abs=1e-6)
    assert by_id["a"] == pytest.approx(0.0, abs=1e-6)


def test_deterministic_across_repeated_searches(index):
    q = unit([0.3, 0.5, 0.2])
    assert index.search(q, top_k=4) == index.search(q, top_k=4)


def test_top_k_larger_than_index_returns_all(index):
    assert len(index.search(unit([1.0, 0.0, 0.0]), top_k=100)) == 4


def test_rejects_mismatched_ids():
    vecs = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match="chunk ids"):
        DenseIndex.from_vectors(vecs, ["only-one"])


def test_rejects_wrong_dtype():
    with pytest.raises(ValueError, match="float32"):
        DenseIndex.from_vectors(np.eye(3, dtype=np.float64), ["a", "b", "c"])


def test_rejects_empty():
    with pytest.raises(ValueError, match="zero vectors"):
        DenseIndex.from_vectors(np.zeros((0, 3), dtype=np.float32), [])


def test_rejects_wrong_query_dim(index):
    with pytest.raises(ValueError, match="does not match"):
        index.search(np.ones(5, dtype=np.float32), top_k=1)


def test_roundtrip_through_file(tmp_path, index):
    import faiss

    path = tmp_path / "ix.faiss"
    faiss.write_index(index._index, str(path))
    loaded = DenseIndex.from_files(path, ["a", "b", "c", "d"])
    q = unit([1.0, 0.2, 0.3])
    assert loaded.search(q, top_k=4) == index.search(q, top_k=4)
