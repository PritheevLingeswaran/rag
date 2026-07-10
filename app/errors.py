"""Shared exception types.

Every failure mode in the ingestion/storage layer maps to one of these, so
callers can handle classes of failure explicitly instead of catching broad
Exception. None of these are ever silently swallowed; they either abort the
operation or are recorded per-item in the run report.
"""

from __future__ import annotations


class RagpError(Exception):
    """Base class for all application errors."""


class ConfigurationError(RagpError):
    """A required setting is missing for the component being used."""


class MalformedDocumentError(RagpError):
    """A single input document violates the corpus schema."""


class EmbeddingError(RagpError):
    """Embedding a batch failed after retries; the ingestion run must abort."""


class IndexWriteError(RagpError):
    """Writing/renaming the FAISS index failed (disk full, permissions...)."""


class IndexIntegrityError(RagpError):
    """A loaded index does not match its recorded manifest/hash."""


class MigrationError(RagpError):
    """A schema migration failed to apply."""
