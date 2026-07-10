-- 0001_init.sql
-- Core schema: documents, chunks, index versions, embedding metadata,
-- query/citation logs.
--
-- Conventions:
--   * All timestamps are timestamptz, set server-side.
--   * Embeddings themselves live in FAISS index files on disk, NOT in
--     Postgres (free-tier Postgres storage is capped; vectors are the
--     bulk of the data). Postgres stores the *metadata* needed to map a
--     chunk to its row in a specific FAISS index version, and to verify
--     integrity.

CREATE TABLE documents (
    doc_id          TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    source          TEXT,
    content_sha256  TEXT NOT NULL,          -- idempotency key: unchanged docs are skipped on re-ingest
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    chunk_id     TEXT PRIMARY KEY,          -- "{doc_id}::c{n}", the versioned chunking contract
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index  INT  NOT NULL,
    text         TEXT NOT NULL,
    text_sha256  TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_id, chunk_index)
);

-- One row per FAISS index build. Exactly one row may be 'active' at a time
-- (enforced by the partial unique index below); the serving layer reads the
-- active version. Rollback = flip which row is active, never mutate files.
CREATE TABLE index_versions (
    version_id     TEXT PRIMARY KEY,        -- e.g. "v20260710T180000Z-a1b2c3"
    status         TEXT NOT NULL CHECK (status IN ('building', 'ready', 'active', 'rolled_back', 'failed')),
    embedder_id    TEXT NOT NULL,           -- identifies model + config; versions are not comparable across embedders
    embedding_dim  INT  NOT NULL,
    chunk_count    INT,
    corpus_sha256  TEXT,                    -- hash of the exact ingested content (idempotency key for a run)
    faiss_sha256   TEXT,                    -- hash of the written index file (integrity check on load)
    error          TEXT,                    -- populated when status = 'failed'
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    activated_at   TIMESTAMPTZ
);

CREATE UNIQUE INDEX index_versions_single_active
    ON index_versions ((TRUE)) WHERE status = 'active';

-- Maps a chunk to its row number in a specific FAISS index version.
CREATE TABLE chunk_embeddings (
    chunk_id    TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    version_id  TEXT NOT NULL REFERENCES index_versions(version_id) ON DELETE CASCADE,
    faiss_row   INT  NOT NULL,
    PRIMARY KEY (chunk_id, version_id),
    UNIQUE (version_id, faiss_row)
);

CREATE TABLE query_logs (
    id             BIGSERIAL PRIMARY KEY,
    query_text     TEXT NOT NULL,
    answer_text    TEXT,
    index_version  TEXT,                    -- which index served this query (nullable: BM25-only serving has none)
    latency_ms     DOUBLE PRECISION,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE citations (
    id            BIGSERIAL PRIMARY KEY,
    query_log_id  BIGINT NOT NULL REFERENCES query_logs(id) ON DELETE CASCADE,
    chunk_id      TEXT NOT NULL,            -- no FK: a cited chunk may be deleted later; logs are immutable history
    rank          INT NOT NULL,
    score         DOUBLE PRECISION,
    UNIQUE (query_log_id, rank)
);

CREATE INDEX query_logs_created_at ON query_logs (created_at);
CREATE INDEX chunks_doc_id ON chunks (doc_id);
