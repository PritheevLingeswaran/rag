"""Isolated unit tests for FaissStore (filesystem only, no DB/services).
Protects the atomic-write/integrity/gc contract that the concurrency
audit and index versioning both depend on."""

from __future__ import annotations

import numpy as np
import pytest

from app.errors import IndexIntegrityError, IndexWriteError
from app.ingest.faiss_store import FaissStore


def unit_vectors(n: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(3)
    v = rng.standard_normal((n, dim), dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture()
def store(tmp_path) -> FaissStore:
    return FaissStore(tmp_path / "indexes")


def test_write_then_load_roundtrip(store):
    manifest = store.write_version("v1", "emb", unit_vectors(10), "corp-sha")
    index = store.load_index("v1", expected_sha256=manifest.faiss_sha256)
    assert index.ntotal == 10
    assert manifest.chunk_count == 10
    assert manifest.embedding_dim == 8


def test_manifest_contents_persisted(store):
    store.write_version("v1", "emb-x", unit_vectors(4), "corp-sha")
    m = store.load_manifest("v1")
    assert m.embedder_id == "emb-x"
    assert m.corpus_sha256 == "corp-sha"


def test_no_staging_dir_left_after_success(store):
    store.write_version("v1", "emb", unit_vectors(4), "s")
    assert not any(p.name.startswith(".tmp-") for p in store.root.iterdir())


def test_duplicate_version_refused(store):
    store.write_version("v1", "emb", unit_vectors(4), "s")
    with pytest.raises(IndexWriteError, match="already exists"):
        store.write_version("v1", "emb", unit_vectors(4), "s")


def test_wrong_dtype_refused(store):
    with pytest.raises(IndexWriteError, match="float32"):
        store.write_version("v1", "emb",
                            unit_vectors(4).astype(np.float64), "s")


def test_tampered_file_fails_integrity(store):
    store.write_version("v1", "emb", unit_vectors(6), "s")
    path = store.version_dir("v1") / "index.faiss"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(IndexIntegrityError, match="sha256"):
        store.load_index("v1")


def test_db_hash_mismatch_fails_integrity(store):
    store.write_version("v1", "emb", unit_vectors(6), "s")
    with pytest.raises(IndexIntegrityError, match="database record"):
        store.load_index("v1", expected_sha256="0" * 64)


def test_missing_manifest_is_integrity_error(store):
    with pytest.raises(IndexIntegrityError, match="manifest"):
        store.load_manifest("never-written")


def test_gc_keeps_only_named_versions_and_sweeps_staging(store):
    store.write_version("v1", "emb", unit_vectors(4), "s")
    store.write_version("v2", "emb", unit_vectors(4), "s")
    (store.root / ".tmp-crashed").mkdir()
    removed = store.gc(keep_version_ids={"v2"})
    assert sorted(removed) == [".tmp-crashed", "v1"]
    assert store.load_index("v2").ntotal == 4
    assert not store.version_dir("v1").exists()
