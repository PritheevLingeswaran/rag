"""Repository layer: all SQL for documents, chunks, index versions,
embedding metadata, and query/citation logs lives here.

Repositories take an open psycopg connection and never commit on their own
unless stated -- transaction boundaries belong to the caller (the ingestion
pipeline groups multiple repo calls into one atomic unit).
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg


@dataclass(frozen=True)
class IndexVersion:
    version_id: str
    status: str
    embedder_id: str
    embedding_dim: int
    chunk_count: int | None
    corpus_sha256: str | None
    faiss_sha256: str | None
    error: str | None


def _to_version(row: tuple) -> IndexVersion:
    return IndexVersion(*row)


_VERSION_COLS = (
    "version_id, status, embedder_id, embedding_dim, chunk_count, "
    "corpus_sha256, faiss_sha256, error"
)


class DocumentRepo:
    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    def get_content_hash(self, doc_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT content_sha256 FROM documents WHERE doc_id = %s", (doc_id,)
        ).fetchone()
        return row[0] if row else None

    def upsert(self, doc_id: str, title: str, source: str | None,
               content_sha256: str) -> None:
        self.conn.execute(
            """
            INSERT INTO documents (doc_id, title, source, content_sha256)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET
                title = EXCLUDED.title,
                source = EXCLUDED.source,
                content_sha256 = EXCLUDED.content_sha256,
                updated_at = now()
            """,
            (doc_id, title, source, content_sha256),
        )

    def replace_chunks(self, doc_id: str,
                       chunks: list[tuple[str, int, str, str]]) -> None:
        """Delete and re-insert a document's chunks.

        chunks: (chunk_id, chunk_index, text, text_sha256). Called only for
        new/changed documents, inside the caller's transaction, so a crash
        mid-replace rolls back to the previous consistent state.
        """
        self.conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (chunk_id, doc_id, chunk_index, text, text_sha256)
                VALUES (%s, %s, %s, %s, %s)
                """,
                [(cid, doc_id, idx, text, sha) for cid, idx, text, sha in chunks],
            )

    def all_chunks_ordered(self) -> list[tuple[str, str]]:
        """All (chunk_id, text) ordered deterministically by (doc_id, index)."""
        return self.conn.execute(
            "SELECT chunk_id, text FROM chunks ORDER BY doc_id, chunk_index"
        ).fetchall()


class IndexVersionRepo:
    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    def create_building(self, version_id: str, embedder_id: str,
                        embedding_dim: int) -> None:
        self.conn.execute(
            """
            INSERT INTO index_versions (version_id, status, embedder_id, embedding_dim)
            VALUES (%s, 'building', %s, %s)
            """,
            (version_id, embedder_id, embedding_dim),
        )

    def mark_ready(self, version_id: str, chunk_count: int,
                   corpus_sha256: str, faiss_sha256: str) -> None:
        self.conn.execute(
            """
            UPDATE index_versions
            SET status = 'ready', chunk_count = %s, corpus_sha256 = %s,
                faiss_sha256 = %s
            WHERE version_id = %s AND status = 'building'
            """,
            (chunk_count, corpus_sha256, faiss_sha256, version_id),
        )

    def mark_failed(self, version_id: str, error: str) -> None:
        self.conn.execute(
            "UPDATE index_versions SET status = 'failed', error = %s "
            "WHERE version_id = %s",
            (error[:2000], version_id),
        )

    def get(self, version_id: str) -> IndexVersion | None:
        row = self.conn.execute(
            f"SELECT {_VERSION_COLS} FROM index_versions WHERE version_id = %s",
            (version_id,),
        ).fetchone()
        return _to_version(row) if row else None

    def get_active(self) -> IndexVersion | None:
        row = self.conn.execute(
            f"SELECT {_VERSION_COLS} FROM index_versions WHERE status = 'active'"
        ).fetchone()
        return _to_version(row) if row else None

    def find_reusable(self, corpus_sha256: str,
                      embedder_id: str) -> IndexVersion | None:
        """An existing non-failed version built from identical content with
        the same embedder -- the idempotency check for a whole run."""
        row = self.conn.execute(
            f"""
            SELECT {_VERSION_COLS} FROM index_versions
            WHERE corpus_sha256 = %s AND embedder_id = %s
              AND status IN ('ready', 'active')
            ORDER BY created_at DESC LIMIT 1
            """,
            (corpus_sha256, embedder_id),
        ).fetchone()
        return _to_version(row) if row else None

    def list_versions(self) -> list[IndexVersion]:
        rows = self.conn.execute(
            f"SELECT {_VERSION_COLS} FROM index_versions ORDER BY created_at"
        ).fetchall()
        return [_to_version(r) for r in rows]

    def activate(self, version_id: str) -> None:
        """Atomically make version_id the single active version.

        The partial unique index on status='active' makes a double-activate
        race impossible: the second transaction would violate the index and
        roll back.
        """
        with self.conn.transaction():
            self.conn.execute(
                "UPDATE index_versions SET status = 'ready' "
                "WHERE status = 'active'"
            )
            cur = self.conn.execute(
                """
                UPDATE index_versions
                SET status = 'active', activated_at = now()
                WHERE version_id = %s AND status = 'ready'
                """,
                (version_id,),
            )
            if cur.rowcount != 1:
                raise ValueError(
                    f"cannot activate {version_id!r}: not found or not in "
                    f"'ready' state"
                )

    def rollback_active(self) -> tuple[str, str]:
        """Deactivate the active version and activate the most recent prior
        'ready' version. Returns (rolled_back_id, new_active_id)."""
        with self.conn.transaction():
            active = self.get_active()
            if active is None:
                raise ValueError("no active index version to roll back")
            prev = self.conn.execute(
                """
                SELECT version_id FROM index_versions
                WHERE status = 'ready' AND version_id <> %s
                ORDER BY created_at DESC LIMIT 1
                """,
                (active.version_id,),
            ).fetchone()
            if prev is None:
                raise ValueError(
                    "no prior 'ready' version exists to roll back to"
                )
            self.conn.execute(
                "UPDATE index_versions SET status = 'rolled_back' "
                "WHERE version_id = %s",
                (active.version_id,),
            )
            self.conn.execute(
                "UPDATE index_versions SET status = 'active', activated_at = now() "
                "WHERE version_id = %s",
                (prev[0],),
            )
            return active.version_id, prev[0]

    def write_chunk_mapping(self, version_id: str,
                            rows: list[tuple[str, int]]) -> None:
        """rows: (chunk_id, faiss_row) for every vector in the index."""
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunk_embeddings (chunk_id, version_id, faiss_row)
                VALUES (%s, %s, %s)
                """,
                [(cid, version_id, row) for cid, row in rows],
            )


class QueryLogRepo:
    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    def log(self, query_text: str, answer_text: str | None,
            index_version: str | None, latency_ms: float,
            citations: list[tuple[str, int, float]]) -> int:
        """citations: (chunk_id, rank, score). Commits its own transaction --
        query logging must never be left dangling on an open connection."""
        with self.conn.transaction():
            row = self.conn.execute(
                """
                INSERT INTO query_logs (query_text, answer_text, index_version, latency_ms)
                VALUES (%s, %s, %s, %s) RETURNING id
                """,
                (query_text, answer_text, index_version, latency_ms),
            ).fetchone()
            assert row is not None
            log_id: int = row[0]
            with self.conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO citations (query_log_id, chunk_id, rank, score)
                    VALUES (%s, %s, %s, %s)
                    """,
                    [(log_id, cid, rank, score) for cid, rank, score in citations],
                )
        return log_id
