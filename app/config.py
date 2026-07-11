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
