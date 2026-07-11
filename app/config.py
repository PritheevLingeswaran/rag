"""Application configuration.

Settings are loaded from environment variables / a .env file via
pydantic-settings. Fields with no default are REQUIRED: if they are absent
from both the environment and .env, instantiating Settings() raises
pydantic.ValidationError immediately, which is exactly what we want -- a
misconfigured deployment should fail at startup, not serve requests with a
guessed default (e.g. silently defaulting ENVIRONMENT to "production" would
be actively dangerous).

get_settings() is process-wide cached (lru_cache) so environment parsing and
validation happen once per process, not per request.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Required: no default. Forces every deployment to state explicitly
    # which environment it is, rather than inheriting a default that is
    # wrong more often than it's right.
    environment: Literal["development", "staging", "production"] = Field(
        ...,
        description="Deployment environment; must be set explicitly, no default.",
    )

    app_name: str = "ragp"

    # Storage endpoints. Optional at Settings level because the bare API
    # skeleton (/health) must boot without them; any component that needs
    # one calls require_setting() and fails loudly at the point of use.
    database_url: str | None = None
    redis_url: str | None = None
    index_root: str = "indexes"

    # Retrieval defaults, set from the CPU-THROTTLED (0.1 CPU, 512MB)
    # load test in docs/loadtest_stage4.md -- NOT from laptop numbers.
    # Measured there: RRF-only retrieval floor ~500ms p50, rerank cost
    # ~700ms PER PASSAGE at 0.1 CPU (vs ~65ms/passage unthrottled).
    #
    # rerank_depth=10 is the candidate ceiling; the adaptive budget
    # governs how much of it actually runs. rerank_budget_ms=700 means:
    # on capable hosts (~65ms/passage) depth 10 fully reranks (~650ms);
    # on 0.1-CPU hosts the EWMA cost predictor sees one micro-batch
    # (5 x ~700ms = 3.5s) over budget and serves RRF order instead
    # (explicit rerank_status='skipped_budget' on the response).
    rerank_depth: int = 10
    rerank_budget_ms: float = 700.0

    # LLM generation (Stage 4). No key => the service serves the explicit
    # 'degraded_no_llm' extractive path rather than failing to boot.
    gemini_api_key: str | None = None
    llm_model: str = "gemini-2.5-flash-lite"
    llm_timeout_s: float = 20.0
    llm_max_output_tokens: int = 1024

    # Fraction of the provider's documented free-tier RPM/RPD we allow
    # ourselves (Stage 4.5). We hit our own wall before Google's.
    quota_safety_margin: float = 0.9

    # API serving. api_keys is comma-separated; REQUIRED in production
    # (startup fails loudly without it), optional in development where
    # anonymous access is allowed and logged.
    api_keys: str | None = None
    rate_limit_per_minute: int = 30
    daily_quota_per_key: int = 500   # cost guardrail: requests/key/UTC-day
    corpus_path: str = "data/corpus_v1.jsonl"
    serve_pipeline: bool = True   # tests boot the app without models

    # CORS: comma-separated exact origins. Unset => NO CORS headers are
    # emitted at all (deny-by-default); "*" is deliberately rejected in
    # production by create_app.
    cors_origins: str | None = None
    max_request_bytes: int = 16_384

    @property
    def cors_origin_list(self) -> list[str]:
        if not self.cors_origins:
            return []
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # Admission control (bounded queue in front of the pipeline).
    # Values measured, not estimated (docs/stage5_admission.md):
    # at 0.1 CPU throughput saturates at ~2.1 rps with 2 executing
    # slots, and queue depth 4 caps admitted-request p95 at ~3s while
    # anything beyond is shed with 503 + Retry-After. Depth 6 was
    # measured first and rejected: it bought p95 ~4.4s waits, worse
    # than telling the client to retry.
    admission_max_concurrency: int = 2
    admission_max_queue_depth: int = 4

    @property
    def api_key_list(self) -> list[str]:
        if not self.api_keys:
            return []
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]
    log_level: Literal[
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    ] = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def require_setting(value: str | None, env_name: str) -> str:
    """Fail loudly when an optional-at-boot setting is needed but unset."""
    from app.errors import ConfigurationError

    if value is None or not value.strip():
        raise ConfigurationError(
            f"{env_name} must be set (in the environment or .env) to use "
            f"this component"
        )
    return value
