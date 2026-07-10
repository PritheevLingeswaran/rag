"""FAISS index file management: atomic writes, manifests, integrity checks.

Layout on disk (INDEX_ROOT from settings):

    indexes/
      v20260710T183000Z-3f2a91/
        index.faiss
        manifest.json        # version_id, embedder_id, dim, count, sha256s

Write protocol (crash/disk-full safe):
  1. Build the index fully in memory.
  2. Write index + manifest into a ".tmp-{version}" staging directory.
  3. fsync both files, then os.replace()-rename the staging dir to its
     final name. Rename is atomic on the same filesystem, so a version
     directory either fully exists or doesn't -- readers can never observe
     a half-written index under a final version name.
  4. Any OSError (ENOSPC/disk full included) aborts: staging dir is
     removed best-effort, IndexWriteError raised, and -- critically -- the
     previously active index files were never touched, so serving is
     unaffected.

Load protocol: recompute sha256 of index.faiss and compare with both the
manifest and the Postgres index_versions row; mismatch raises
IndexIntegrityError rather than serving from a corrupt/tampered file.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app.errors import IndexIntegrityError, IndexWriteError
from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class IndexManifest:
    version_id: str
    embedder_id: str
    embedding_dim: int
    chunk_count: int
    corpus_sha256: str
    faiss_sha256: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fsync_file(path: Path) -> None:
    # O_RDWR, not O_RDONLY: Windows os.fsync requires a writable handle
    # (fsync on a read-only fd fails with EBADF).
    fd = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class FaissStore:
    def __init__(self, index_root: Path) -> None:
        self.root = index_root

    def version_dir(self, version_id: str) -> Path:
        return self.root / version_id

    def write_version(self, version_id: str, embedder_id: str,
                      vectors: np.ndarray, corpus_sha256: str) -> IndexManifest:
        """Build and atomically persist a new index version directory."""
        if vectors.ndim != 2 or vectors.dtype != np.float32:
            raise IndexWriteError(
                f"expected float32 matrix, got {vectors.dtype} ndim={vectors.ndim}"
            )
        count, dim = vectors.shape
        # Inner product over L2-normalized vectors == cosine similarity.
        # Flat (exhaustive) index: exact, zero recall loss, and at <=50k
        # vectors x 384 dims it is ~75MB and a few ms per query -- IVF/HNSW
        # approximation is not worth its recall cost at this corpus size.
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        final_dir = self.version_dir(version_id)
        if final_dir.exists():
            raise IndexWriteError(f"version dir already exists: {final_dir}")
        staging = self.root / f".tmp-{version_id}"
        try:
            staging.mkdir(parents=True, exist_ok=False)
            index_path = staging / "index.faiss"
            faiss.write_index(index, str(index_path))
            _fsync_file(index_path)
            faiss_sha = _sha256_file(index_path)
            manifest = IndexManifest(
                version_id=version_id,
                embedder_id=embedder_id,
                embedding_dim=dim,
                chunk_count=count,
                corpus_sha256=corpus_sha256,
                faiss_sha256=faiss_sha,
            )
            manifest_path = staging / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest.__dict__, indent=2), encoding="utf-8"
            )
            _fsync_file(manifest_path)
            os.replace(staging, final_dir)
        except OSError as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise IndexWriteError(
                f"failed writing index version {version_id}: {exc}"
            ) from exc
        logger.info(
            "index_version_written", version_id=version_id,
            chunk_count=count, dim=dim, sha256=faiss_sha[:12],
        )
        return manifest

    def load_manifest(self, version_id: str) -> IndexManifest:
        path = self.version_dir(version_id) / "manifest.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IndexIntegrityError(
                f"cannot read manifest for {version_id}: {exc}"
            ) from exc
        return IndexManifest(**data)

    def load_index(self, version_id: str,
                   expected_sha256: str | None = None) -> faiss.Index:
        """Load an index, verifying file hash against manifest (and, if
        provided, against the hash recorded in Postgres)."""
        manifest = self.load_manifest(version_id)
        index_path = self.version_dir(version_id) / "index.faiss"
        actual = _sha256_file(index_path)
        if actual != manifest.faiss_sha256:
            raise IndexIntegrityError(
                f"{version_id}: index.faiss sha256 {actual[:12]} != "
                f"manifest {manifest.faiss_sha256[:12]}"
            )
        if expected_sha256 is not None and actual != expected_sha256:
            raise IndexIntegrityError(
                f"{version_id}: index.faiss sha256 {actual[:12]} != "
                f"database record {expected_sha256[:12]}"
            )
        return faiss.read_index(str(index_path))

    def gc(self, keep_version_ids: set[str]) -> list[str]:
        """Delete version directories not in keep_version_ids.
        Also sweeps orphaned staging dirs from crashed runs."""
        removed = []
        if not self.root.exists():
            return removed
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            if child.name.startswith(".tmp-") or child.name not in keep_version_ids:
                shutil.rmtree(child, ignore_errors=True)
                removed.append(child.name)
                logger.info("index_version_gc", version_id=child.name)
        return removed
