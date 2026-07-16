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

import threading
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
    llm_model: str = "gemini-3.1-flash-lite"
    llm_timeout_s: float = 20.0
    llm_max_output_tokens: int = 1024

    # Secondary LLM provider (Stage 2.5's planned Groq fallback). When
    # set, a Gemini failure or exhausted Gemini budget falls back to
    # Groq (its own quota guard) before the extractive path. Absent =>
    # single-provider behavior, unchanged.
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"

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

    # Google OAuth (Stage 9.6, per the Stage 9.5 decisions). Both unset
    # => the /auth/google/* routes return 503 with a clear error; the
    # API-key surface is unaffected. Web sessions additionally need
    # REDIS_URL (session store) and DATABASE_URL (users table at login).
    google_client_id: str | None = None
    google_client_secret: str | None = None
    session_ttl_s: int = 7 * 86_400   # fixed 7-day TTL, no sliding refresh

    # Per-user limits for Google-authenticated web users (Stage 9.5
    # point 3): deliberately below the API-key limits because a Google
    # sign-up is a free quota grant to an attacker -- individual grants
    # stay small; the global QuotaGuards remain the aggregate wall.
    user_rate_limit_per_minute: int = 10
    user_daily_quota: int = 50

    # CORS: comma-separated exact origins. Unset => NO CORS headers are
    # emitted at all (deny-by-default); "*" is deliberately rejected in
    # production by create_app.
    cors_origins: str | None = None
    max_request_bytes: int = 16_384

    # Response cache (Stage 2.5 capacity math: cache hits spend no LLM
    # quota and no pipeline compute -- on this infra the cache IS
    # capacity). Requires Redis; bypassed (and counted) without it.
    cache_ttl_s: int = 3600

    # Quota circuit breakers + alerting (Stage 7.7). Budgets derive from
    # configs/free_tier_limits.json: 500K Upstash commands/month ~= 16129
    # per UTC day; Neon storage 0.5GB. Webhook: any JSON-POST receiver
    # (ntfy.sh / Discord / Slack / healthchecks -- all free).
    alert_webhook_url: str | None = None
    redis_daily_command_budget: int = 16_129
    postgres_storage_limit_mb: int = 500

    # Stage 11: when set, GET /metrics requires
    # 'Authorization: Bearer <token>' (Grafana Cloud scraper supports
    # this). Unset => /metrics is open (fine locally, not on a public
    # URL: operational metrics are recon material).
    metrics_token: str | None = None

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


# NOT lru_cache: functools.lru_cache does not lock around the wrapped
# call, so concurrent first-calls each construct (and receive) their own
# Settings instance -- measured in the Stage 6.5 audit: 64 concurrent
# calls yielded 11 distinct instances. Benign only while Settings stays
# immutable; double-checked locking makes the singleton actual.
_settings_lock = threading.Lock()
_settings_instance: Settings | None = None


def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        with _settings_lock:
            if _settings_instance is None:
                _settings_instance = Settings()
    return _settings_instance


def _clear_settings_cache() -> None:
    global _settings_instance
    with _settings_lock:
        _settings_instance = None


# tests use get_settings.cache_clear(), matching the old lru_cache API
get_settings.cache_clear = _clear_settings_cache


def require_setting(value: str | None, env_name: str) -> str:
    """Fail loudly when an optional-at-boot setting is needed but unset."""
    from app.errors import ConfigurationError

    if value is None or not value.strip():
        raise ConfigurationError(
            f"{env_name} must be set (in the environment or .env) to use "
            f"this component"
        )
    return value
