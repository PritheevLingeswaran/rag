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


class LLMError(RagpError):
    """Base for LLM API failures. GenerationService catches these and
    degrades explicitly; they never propagate to the client raw."""


class LLMAuthError(LLMError):
    """Invalid/missing API key (HTTP 401/403). Not retryable; a config
    problem, logged loudly."""


class LLMQuotaError(LLMError):
    """Rate limit / quota exhausted (HTTP 429 RESOURCE_EXHAUSTED)."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class LLMConfigError(LLMError):
    """The provider rejected the request AS WE CONFIGURED IT (HTTP 404
    unknown/retired model, 400 bad request shape, ...). Not retryable and
    not the provider's fault; logged loudly like auth failures. Kept
    distinct from LLMMalformedError so a wrong model name is never
    misdiagnosed as a provider response-shape problem (this exact
    confusion happened in production with a retired model id)."""


class LLMTimeoutError(LLMError):
    """The request exceeded our client-side timeout."""


class LLMServerError(LLMError):
    """Provider-side 5xx or network failure. Retryable once."""


class LLMMalformedError(LLMError):
    """Response was not in the documented shape (unparseable JSON,
    no candidates, empty text, safety-blocked with no content)."""
