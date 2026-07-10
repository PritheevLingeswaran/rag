"""Integration tests for the ingestion pipeline against real Postgres.

Every failure policy documented in app/ingest/pipeline.py is exercised
here: malformed docs, systematic input breakage, embedding failure
mid-batch, disk full during index write, idempotent re-runs, versioning,
activation, rollback, and integrity verification.
"""

from __future__ import annotations

import errno
import json
from pathlib import Path

import numpy as np
import pytest

import app.ingest.pipeline as pipeline_mod
from app.errors import EmbeddingError, IndexIntegrityError
from app.ingest.embedder import HashingEmbedder
from app.ingest.faiss_store import FaissStore
from app.ingest.pipeline import IngestionPipeline
from app.storage.repositories import IndexVersionRepo

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS = REPO_ROOT / "data" / "corpus_v1.jsonl"


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "EMBED_BACKOFF_BASE_S", 0.0)


@pytest.fixture()
def store(tmp_path) -> FaissStore:
    return FaissStore(tmp_path / "indexes")


def make_pipeline(conn, store, embedder=None) -> IngestionPipeline:
    return IngestionPipeline(conn, embedder or HashingEmbedder(dim=64), store)


class FailingEmbedder:
    """Fails permanently on the second batch -- mid-batch failure."""

    def __init__(self, dim: int = 64) -> None:
        self._inner = HashingEmbedder(dim=dim)
        self._batches = 0

    @property
    def embedder_id(self) -> str:
        return "failing-test-embedder"

    @property
    def dim(self) -> int:
        return self._inner.dim

    def embed_batch(self, texts):
        self._batches += 1
        if self._batches >= 2:
            raise EmbeddingError("simulated model failure (e.g. OOM/API down)")
        return self._inner.embed_batch(texts)


def test_full_ingest_builds_version_with_mapping(conn, store):
    report = make_pipeline(conn, store).run(CORPUS)
    assert report.status == "built"
    assert report.docs_ingested == 30
    assert report.chunk_count == 60
    assert report.docs_malformed == 0

    v = IndexVersionRepo(conn).get(report.version_id)
    assert v.status == "ready"
    assert v.chunk_count == 60

    # chunk_embeddings maps every chunk to a unique faiss row
    n_rows = conn.execute(
        "SELECT count(*), count(DISTINCT faiss_row) FROM chunk_embeddings "
        "WHERE version_id = %s", (report.version_id,)
    ).fetchone()
    assert n_rows == (60, 60)

    # index file loads and passes integrity check against the DB hash
    index = store.load_index(report.version_id, expected_sha256=v.faiss_sha256)
    assert index.ntotal == 60


def test_reingest_identical_corpus_is_idempotent(conn, store):
    p = make_pipeline(conn, store)
    first = p.run(CORPUS)
    second = p.run(CORPUS)
    assert first.status == "built"
    assert second.status == "reused"
    assert second.version_id == first.version_id
    assert second.docs_ingested == 0
    assert second.docs_unchanged == 30
    n_versions = conn.execute("SELECT count(*) FROM index_versions").fetchone()[0]
    assert n_versions == 1


def test_changed_document_creates_new_version(conn, store, tmp_path):
    p = make_pipeline(conn, store)
    first = p.run(CORPUS)

    lines = CORPUS.read_text(encoding="utf-8").splitlines()
    doc = json.loads(lines[0])
    doc["text"] = doc["text"] + "\n\nA newly appended paragraph of content."
    lines[0] = json.dumps(doc)
    modified = tmp_path / "corpus_modified.jsonl"
    modified.write_text("\n".join(lines) + "\n", encoding="utf-8")

    second = p.run(modified)
    assert second.status == "built"
    assert second.version_id != first.version_id
    assert second.docs_ingested == 1
    assert second.docs_unchanged == 29
    # old version untouched on disk (nothing mutated in place)
    assert store.load_index(first.version_id).ntotal == 60
    assert store.load_index(second.version_id).ntotal == 61


def test_malformed_docs_are_skipped_and_recorded(conn, store, tmp_path):
    lines = CORPUS.read_text(encoding="utf-8").splitlines()
    lines.insert(5, "{this is not json")
    lines.insert(10, '{"doc_id": "no-text", "title": "missing text field"}')
    bad = tmp_path / "corpus_bad_rows.jsonl"
    bad.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = make_pipeline(conn, store).run(bad)
    assert report.status == "built"          # 2/32 < 10% threshold: continue
    assert report.docs_malformed == 2
    assert len(report.malformed) == 2
    assert report.chunk_count == 60          # only valid docs ingested


def test_mostly_malformed_input_aborts_before_any_write(conn, store, tmp_path):
    bad = tmp_path / "corpus_broken.jsonl"
    bad.write_text(
        '{"doc_id": "ok", "title": "t", "text": "fine."}\n'
        "{garbage\n{garbage\n{garbage\n",
        encoding="utf-8",
    )
    report = make_pipeline(conn, store).run(bad)
    assert report.status == "aborted_input"
    assert "systematically broken" in report.error
    assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM index_versions").fetchone()[0] == 0


def test_embedding_failure_mid_batch_aborts_without_index(conn, store):
    report = make_pipeline(conn, store, FailingEmbedder()).run(CORPUS)
    assert report.status == "failed"
    assert "simulated model failure" in report.error

    v = IndexVersionRepo(conn).get(report.version_id)
    assert v.status == "failed"
    assert "simulated model failure" in v.error
    # no index directory under the final version name, no orphan mapping rows
    assert not store.version_dir(report.version_id).exists()
    assert conn.execute(
        "SELECT count(*) FROM chunk_embeddings WHERE version_id = %s",
        (report.version_id,),
    ).fetchone()[0] == 0
    # doc/chunk upserts are kept (idempotent, correct data)
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 60


def test_disk_full_marks_run_failed_and_active_index_survives(
    conn, store, monkeypatch, tmp_path
):
    p = make_pipeline(conn, store)
    versions = IndexVersionRepo(conn)
    first = p.run(CORPUS)
    versions.activate(first.version_id)

    import faiss as faiss_mod

    def explode(index, path):
        raise OSError(errno.ENOSPC, "No space left on device", path)

    monkeypatch.setattr(faiss_mod, "write_index", explode)

    lines = CORPUS.read_text(encoding="utf-8").splitlines()
    doc = json.loads(lines[0])
    doc["text"] += "\n\nMore."
    lines[0] = json.dumps(doc)
    modified = tmp_path / "corpus_m.jsonl"
    modified.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = p.run(modified)
    assert report.status == "failed"
    assert "No space left on device" in report.error

    # active version unchanged, its files intact, no staging junk left
    active = versions.get_active()
    assert active.version_id == first.version_id
    assert store.load_index(first.version_id).ntotal == 60
    assert not any(
        d.name.startswith(".tmp-") for d in store.root.iterdir()
    )


def test_activate_and_rollback(conn, store, tmp_path):
    p = make_pipeline(conn, store)
    versions = IndexVersionRepo(conn)

    v1 = p.run(CORPUS).version_id
    versions.activate(v1)

    lines = CORPUS.read_text(encoding="utf-8").splitlines()
    doc = json.loads(lines[0])
    doc["text"] += "\n\nExtra paragraph for v2."
    lines[0] = json.dumps(doc)
    modified = tmp_path / "c2.jsonl"
    modified.write_text("\n".join(lines) + "\n", encoding="utf-8")
    v2 = p.run(modified).version_id
    versions.activate(v2)

    assert versions.get_active().version_id == v2
    assert versions.get(v1).status == "ready"

    rolled_back, now_active = versions.rollback_active()
    assert rolled_back == v2
    assert now_active == v1
    assert versions.get_active().version_id == v1
    assert versions.get(v2).status == "rolled_back"
    # rollback target's files still on disk and loadable
    assert store.load_index(v1).ntotal == 60


def test_rollback_with_no_prior_version_refuses(conn, store):
    versions = IndexVersionRepo(conn)
    v1 = make_pipeline(conn, store).run(CORPUS).version_id
    versions.activate(v1)
    with pytest.raises(ValueError, match="no prior 'ready' version"):
        versions.rollback_active()


def test_tampered_index_file_fails_integrity_check(conn, store):
    report = make_pipeline(conn, store).run(CORPUS)
    index_path = store.version_dir(report.version_id) / "index.faiss"
    data = bytearray(index_path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    index_path.write_bytes(bytes(data))
    with pytest.raises(IndexIntegrityError, match="sha256"):
        store.load_index(report.version_id)


def test_embeddings_are_deterministic():
    a = HashingEmbedder(dim=64).embed_batch(["hello world", "bm25 ranking"])
    b = HashingEmbedder(dim=64).embed_batch(["hello world", "bm25 ranking"])
    assert np.array_equal(a, b)
    norms = np.linalg.norm(a, axis=1)
    assert np.allclose(norms, 1.0)
