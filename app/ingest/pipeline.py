"""Ingestion pipeline: load -> chunk -> embed -> FAISS write -> activate.

Failure policy, per stage (each is tested in tests/integration):

  Malformed document
      The document is skipped and recorded (doc_id/line + reason) in the
      run report; the rest of the batch proceeds. If more than
      MALFORMED_ABORT_FRACTION of documents are malformed, or zero valid
      documents remain, the run aborts before writing anything -- a mostly
      broken input file signals an upstream bug, not a few bad rows.

  Embedding failure mid-batch
      Each batch is retried up to EMBED_MAX_RETRIES with exponential
      backoff. If a batch still fails, the run ABORTS: the index_versions
      row is marked 'failed' with the error, and no FAISS file is written
      under a final version name. We never build an index from partial
      embeddings -- an index silently missing 30% of chunks *looks* healthy
      and degrades recall invisibly, which is corruption. Document/chunk
      upserts already committed are kept: they are idempotent and correct.

  Disk full / any index write failure
      FaissStore stages into a temp dir and atomically renames on success,
      so an ENOSPC (or any OSError) mid-write leaves only a staging dir
      that is swept by gc(); the version row is marked 'failed'. The
      currently active index files are never opened for write, so serving
      is unaffected.

Idempotency: a run's identity is (corpus_sha256 of all chunk texts,
embedder_id). Re-ingesting identical content returns the existing version
without rebuilding. Changed corpora produce a NEW version directory; nothing
is ever mutated in place, which is what makes rollback trivial.

Activation is explicit and separate from building: a new index goes live
only via activate(), a single-row transactional flip that rollback()
reverses. Ingestion alone can never change what users are served.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import psycopg

from app.errors import EmbeddingError, IndexWriteError, MalformedDocumentError
from app.ingest.embedder import Embedder
from app.ingest.faiss_store import FaissStore
from app.logging_config import get_logger
from app.storage.repositories import DocumentRepo, IndexVersionRepo

logger = get_logger(__name__)

MALFORMED_ABORT_FRACTION = 0.10
EMBED_BATCH_SIZE = 32
EMBED_MAX_RETRIES = 3
EMBED_BACKOFF_BASE_S = 0.5


@dataclass
class RunReport:
    version_id: str | None
    status: str                       # 'built' | 'reused' | 'failed' | 'aborted_input'
    docs_seen: int = 0
    docs_ingested: int = 0
    docs_unchanged: int = 0
    docs_malformed: int = 0
    chunk_count: int = 0
    malformed: list[dict] = field(default_factory=list)
    error: str | None = None


def _parse_doc(line_no: int, line: str) -> dict:
    try:
        doc = json.loads(line)
    except json.JSONDecodeError as exc:
        raise MalformedDocumentError(f"line {line_no}: invalid JSON: {exc}") from exc
    missing = {"doc_id", "title", "text"} - doc.keys()
    if missing:
        raise MalformedDocumentError(
            f"line {line_no}: missing fields {sorted(missing)}"
        )
    if not isinstance(doc["text"], str) or not doc["text"].strip():
        raise MalformedDocumentError(
            f"line {line_no}: doc {doc.get('doc_id')!r} has empty text"
        )
    return doc


def _chunk_doc(doc: dict) -> list[tuple[str, int, str, str]]:
    paragraphs = [p.strip() for p in doc["text"].split("\n\n") if p.strip()]
    return [
        (
            f"{doc['doc_id']}::c{i}",
            i,
            para,
            hashlib.sha256(para.encode("utf-8")).hexdigest(),
        )
        for i, para in enumerate(paragraphs)
    ]


class IngestionPipeline:
    def __init__(self, conn: psycopg.Connection, embedder: Embedder,
                 faiss_store: FaissStore, storage_breaker=None) -> None:
        """storage_breaker: optional app.reliability.PostgresStorageBreaker.
        When open (DB size >= enforced budget), ingestion refuses BEFORE
        writing anything -- Neon fails INSERT/UPDATE/DELETE past its hard
        limit, so tripping early keeps us the ones choosing what breaks
        (a new corpus can wait; a corrupted half-write cannot)."""
        self.conn = conn
        self.embedder = embedder
        self.store = faiss_store
        self.storage_breaker = storage_breaker
        self.docs = DocumentRepo(conn)
        self.versions = IndexVersionRepo(conn)

    # ---- stage 1: load + validate ----

    def _load_documents(self, corpus_path: Path,
                        report: RunReport) -> list[dict]:
        docs: list[dict] = []
        with corpus_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                report.docs_seen += 1
                try:
                    docs.append(_parse_doc(line_no, line))
                except MalformedDocumentError as exc:
                    report.docs_malformed += 1
                    report.malformed.append({"line": line_no, "reason": str(exc)})
                    logger.warning("malformed_document_skipped", reason=str(exc))
        if report.docs_seen == 0:
            raise MalformedDocumentError(f"{corpus_path}: no documents found")
        frac = report.docs_malformed / report.docs_seen
        if not docs or frac > MALFORMED_ABORT_FRACTION:
            raise MalformedDocumentError(
                f"{report.docs_malformed}/{report.docs_seen} documents malformed "
                f"(> {MALFORMED_ABORT_FRACTION:.0%} threshold); aborting run "
                f"before any write -- input looks systematically broken"
            )
        return docs

    # ---- stage 2: upsert docs + chunks (idempotent) ----

    def _upsert_documents(self, docs: list[dict], report: RunReport) -> None:
        with self.conn.transaction():
            for doc in docs:
                content_sha = hashlib.sha256(
                    doc["text"].encode("utf-8")
                ).hexdigest()
                if self.docs.get_content_hash(doc["doc_id"]) == content_sha:
                    report.docs_unchanged += 1
                    continue
                self.docs.upsert(
                    doc["doc_id"], doc["title"], doc.get("source"), content_sha
                )
                self.docs.replace_chunks(doc["doc_id"], _chunk_doc(doc))
                report.docs_ingested += 1

    # ---- stage 3: embed with retry ----

    def _embed_all(self, texts: list[str]):
        import numpy as np

        batches = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start:start + EMBED_BATCH_SIZE]
            last_error: Exception | None = None
            for attempt in range(1, EMBED_MAX_RETRIES + 1):
                try:
                    batches.append(self.embedder.embed_batch(batch))
                    last_error = None
                    break
                except EmbeddingError as exc:
                    last_error = exc
                    logger.warning(
                        "embed_batch_failed", batch_start=start,
                        attempt=attempt, max_retries=EMBED_MAX_RETRIES,
                        error=str(exc),
                    )
                    if attempt < EMBED_MAX_RETRIES:
                        time.sleep(EMBED_BACKOFF_BASE_S * 2 ** (attempt - 1))
            if last_error is not None:
                raise EmbeddingError(
                    f"batch at offset {start} failed after "
                    f"{EMBED_MAX_RETRIES} attempts: {last_error}"
                ) from last_error
        return np.vstack(batches)

    # ---- orchestration ----

    def run(self, corpus_path: Path) -> RunReport:
        report = RunReport(version_id=None, status="failed")

        if self.storage_breaker is not None:
            decision = self.storage_breaker.check_writable()
            if not decision.allowed:
                report.status = "aborted_storage_budget"
                report.error = (
                    f"postgres storage breaker OPEN: {decision.used} bytes "
                    f">= enforced budget {decision.enforced} (hard limit "
                    f"{decision.hard_limit}); refusing to write. Free space "
                    f"or raise the plan before ingesting."
                )
                logger.error("ingestion_refused_storage_budget",
                             used=decision.used, enforced=decision.enforced)
                return report

        try:
            docs = self._load_documents(corpus_path, report)
        except MalformedDocumentError as exc:
            report.status = "aborted_input"
            report.error = str(exc)
            logger.error("ingestion_aborted_bad_input", error=str(exc))
            return report

        try:
            self._upsert_documents(docs, report)
        except psycopg.Error as exc:
            # Past the provider hard limit (or any DB write failure):
            # typed, clean abort -- never a stack trace to the operator,
            # never a partial index (nothing index-side has happened yet).
            report.status = "failed_storage"
            report.error = f"database write failed: {exc}"
            logger.error("ingestion_db_write_failed", error=str(exc))
            return report

        chunk_rows = self.docs.all_chunks_ordered()
        report.chunk_count = len(chunk_rows)
        corpus_sha = hashlib.sha256(
            "\x00".join(f"{cid}\x01{text}" for cid, text in chunk_rows)
            .encode("utf-8")
        ).hexdigest()

        existing = self.versions.find_reusable(
            corpus_sha, self.embedder.embedder_id
        )
        if existing is not None:
            report.version_id = existing.version_id
            report.status = "reused"
            logger.info(
                "ingestion_reused_existing_version",
                version_id=existing.version_id, corpus_sha256=corpus_sha[:12],
            )
            return report

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        version_id = f"v{stamp}-{uuid.uuid4().hex[:6]}"
        report.version_id = version_id
        with self.conn.transaction():
            self.versions.create_building(
                version_id, self.embedder.embedder_id, self.embedder.dim
            )

        try:
            vectors = self._embed_all([text for _, text in chunk_rows])
            manifest = self.store.write_version(
                version_id, self.embedder.embedder_id, vectors, corpus_sha
            )
            with self.conn.transaction():
                self.versions.mark_ready(
                    version_id, len(chunk_rows), corpus_sha, manifest.faiss_sha256
                )
                self.versions.write_chunk_mapping(
                    version_id,
                    [(cid, row) for row, (cid, _) in enumerate(chunk_rows)],
                )
        except (EmbeddingError, IndexWriteError) as exc:
            with self.conn.transaction():
                self.versions.mark_failed(version_id, str(exc))
            report.status = "failed"
            report.error = str(exc)
            logger.error(
                "ingestion_run_failed", version_id=version_id,
                error=str(exc), active_index_unaffected=True,
            )
            return report

        report.status = "built"
        logger.info(
            "ingestion_run_complete", version_id=version_id,
            chunks=len(chunk_rows), docs_ingested=report.docs_ingested,
            docs_unchanged=report.docs_unchanged,
        )
        return report
