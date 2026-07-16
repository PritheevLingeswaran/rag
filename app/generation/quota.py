"""Proactive LLM quota accounting against the Stage 2.5 provider limits.

"Quota exceeded" is an expected, planned-for state, not an error. This
module makes the system hit OUR budget wall (a deliberate safety margin
below the provider's) before it can ever hit Google's, so degradation
happens on our terms with a clean label instead of a hard 429.

Limits come from configs/free_tier_limits.json -- the same file Stage 2.5
sourced from provider docs. An unknown model is a ConfigurationError:
guessing quota numbers defeats the point.

Enforcement:
    enforced_rpm = floor(provider_rpm * safety_margin)     (default 0.9)
    enforced_rpd = floor(provider_rpd * safety_margin)
RPM uses per-epoch-minute windows; RPD uses a window keyed to the
provider's documented reset (midnight US Pacific), so our day rolls over
exactly when Google's does.

State backends: Redis first (atomic bounded counters shared across
workers -- correct under concurrency and across restarts), with an
in-process thread-safe fallback when Redis is unconfigured or down. The
fallback under-counts across multiple workers; if the provider rejects us
anyway, record_provider_rejection() opens a cooldown that proactively
blocks until retry-after expires, so the reactive path corrects the
proactive one.

Operator-facing distinction (also in GenerationService logs):
    quota throttled (expected)  -> status degraded_quota_throttled, INFO
    provider 429 (accounting slipped or shared project quota) -> WARNING
    provider 5xx/timeout (actual API failure) -> ERROR
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from app.errors import ConfigurationError
from app.logging_config import get_logger

logger = get_logger(__name__)

LIMITS_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "free_tier_limits.json"
PACIFIC = ZoneInfo("America/Los_Angeles")
DEFAULT_SAFETY_MARGIN = 0.9

REASON_OK = "ok"
REASON_RPM = "rpm_exhausted"
REASON_RPD = "rpd_exhausted"
REASON_COOLDOWN = "provider_cooldown"


@dataclass(frozen=True)
class ModelLimits:
    model: str
    rpm: int
    rpd: int


@dataclass(frozen=True)
class QuotaDecision:
    allowed: bool
    reason: str
    remaining_rpm: int
    remaining_rpd: int
    retry_after_s: float


def load_model_limits(model: str, limits_path: Path = LIMITS_PATH) -> ModelLimits:
    data = json.loads(limits_path.read_text(encoding="utf-8"))
    gemini = data["llm_gemini_free"]
    if model == gemini["primary_model"]:
        return ModelLimits(model, int(gemini["primary_rpm"]),
                           int(gemini["primary_rpd"]))
    fallback = gemini.get("fallback_models", {})
    if model in fallback:
        return ModelLimits(model, int(fallback[model]["rpm"]),
                           int(fallback[model]["rpd"]))
    raise ConfigurationError(
        f"no free-tier limits recorded for model {model!r} in "
        f"{limits_path.name}; add them (sourced, not guessed) before use"
    )


def seconds_to_pacific_midnight(now_utc: float) -> float:
    now_pacific = datetime.fromtimestamp(now_utc, tz=PACIFIC)
    next_midnight = (now_pacific + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return (next_midnight - now_pacific).total_seconds()


def pacific_day_stamp(now_utc: float) -> str:
    return datetime.fromtimestamp(now_utc, tz=PACIFIC).strftime("%Y%m%d")


class _LocalCounters:
    """In-process fallback backend. Thread-safe; same window semantics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def incr(self, key: str) -> int:
        with self._lock:
            self._counts[key] = self._counts.get(key, 0) + 1
            # opportunistic GC: prune only per-minute (:rpm:) keys, whose
            # fixed-width epoch-minute suffix makes lexicographic == numeric
            # order. Daily (:rpd:) keys are NEVER pruned: ':rpd:' sorts
            # before ':rpm:', so a naive keep-last-N evicted the LIVE daily
            # counter mid-day, resetting the day's count (quota over-spend).
            # At most a handful of rpd keys ever exist (one per day).
            if len(self._counts) > 1000:
                stale_rpm = sorted(
                    k for k in self._counts if ":rpm:" in k
                )[:-100]
                for k in stale_rpm:
                    del self._counts[k]
            return self._counts[key]


class QuotaGuard:
    """Call try_acquire() BEFORE every LLM request; call
    record_provider_rejection() whenever the provider 429s anyway."""

    def __init__(self, limits: ModelLimits,
                 redis_store=None,
                 safety_margin: float = DEFAULT_SAFETY_MARGIN,
                 now_fn: Callable[[], float] = time.time,
                 alerts=None,
                 alert_pct: float = 0.8) -> None:
        self.alerts = alerts
        self.alert_pct = alert_pct
        if not 0.0 < safety_margin <= 1.0:
            raise ValueError("safety_margin must be in (0, 1]")
        self.limits = limits
        self.enforced_rpm = math.floor(limits.rpm * safety_margin)
        self.enforced_rpd = math.floor(limits.rpd * safety_margin)
        if self.enforced_rpm < 1 or self.enforced_rpd < 1:
            raise ConfigurationError(
                f"safety margin {safety_margin} leaves no usable quota for "
                f"{limits.model} (rpm {limits.rpm}, rpd {limits.rpd})"
            )
        self._redis = redis_store
        self._local = _LocalCounters()
        self._now = now_fn
        self._cooldown_until: float = 0.0
        self._lock = threading.Lock()

    # ---- accounting ----

    def _count(self, key: str, ttl_s: int) -> int:
        if self._redis is not None:
            value = self._redis.bounded_incr(key, ttl_s)
            if value is not None:
                return value
            logger.warning("quota_redis_unavailable_using_local", key=key)
        return self._local.incr(key)

    def try_acquire(self) -> QuotaDecision:
        now = self._now()

        with self._lock:
            cooldown_left = self._cooldown_until - now
        if cooldown_left > 0:
            return QuotaDecision(
                allowed=False, reason=REASON_COOLDOWN,
                remaining_rpm=0, remaining_rpd=0,
                retry_after_s=round(cooldown_left, 1),
            )

        minute = int(now // 60)
        day = pacific_day_stamp(now)
        rpm_key = f"llmq:{self.limits.model}:rpm:{minute}"
        rpd_key = f"llmq:{self.limits.model}:rpd:{day}"

        # Count the MINUTE first: a request denied on RPM must not burn
        # the scarce daily budget. (Counting the day first drained a
        # whole day's RPD in minutes under sustained over-RPM load: at
        # the 30/min API rate limit vs the 13/min LLM budget, every
        # denied request still consumed an RPD slot.) The residual leak
        # now points the harmless direction: a request denied on RPD has
        # consumed one RPM slot in that minute, which costs nothing real
        # -- RPD-denied means no LLM calls happen today anyway.
        rpm_used = self._count(rpm_key, 120)
        if rpm_used > self.enforced_rpm:
            return QuotaDecision(
                allowed=False, reason=REASON_RPM,
                remaining_rpm=0,
                remaining_rpd=self.enforced_rpd,  # not consulted: RPD was deliberately not counted
                retry_after_s=round(60.0 - (now % 60), 1),
            )

        rpd_used = self._count(rpd_key, int(seconds_to_pacific_midnight(now)) + 60)
        # 80% daily alert (Stage 7.7): one alert per resource per day,
        # dedupe inside AlertManager.
        if self.alerts is not None and rpd_used >= self.alert_pct * self.enforced_rpd:
            self.alerts.fire(
                f"gemini_rpd:{self.limits.model}",
                rpd_used / self.enforced_rpd,
                f"LLM daily quota at {rpd_used}/{self.enforced_rpd} enforced "
                f"(provider hard limit {self.limits.rpd}/day)",
            )
        if rpd_used > self.enforced_rpd:
            return QuotaDecision(
                allowed=False, reason=REASON_RPD,
                remaining_rpm=max(0, self.enforced_rpm - rpm_used),
                remaining_rpd=0,
                retry_after_s=round(seconds_to_pacific_midnight(now), 0),
            )

        return QuotaDecision(
            allowed=True, reason=REASON_OK,
            remaining_rpm=self.enforced_rpm - rpm_used,
            remaining_rpd=self.enforced_rpd - rpd_used,
            retry_after_s=0.0,
        )

    def record_provider_rejection(self, retry_after_s: float | None) -> None:
        """The provider 429'd despite our accounting (multi-worker local
        fallback, or another consumer on the same Google project). Open a
        proactive cooldown so we stop asking until retry-after passes."""
        cooldown = retry_after_s if retry_after_s is not None else 60.0
        with self._lock:
            self._cooldown_until = max(
                self._cooldown_until, self._now() + cooldown
            )
        logger.warning(
            "quota_provider_rejection_cooldown",
            model=self.limits.model, cooldown_s=cooldown,
            note="provider rejected despite proactive accounting; "
                 "check for other consumers on this project or local-"
                 "fallback under-counting across workers",
        )

    def snapshot(self) -> dict:
        """Current accounting state for /health-style introspection."""
        now = self._now()
        return {
            "model": self.limits.model,
            "enforced_rpm": self.enforced_rpm,
            "enforced_rpd": self.enforced_rpd,
            "provider_rpm": self.limits.rpm,
            "provider_rpd": self.limits.rpd,
            "cooldown_active": self._cooldown_until > now,
            "backend": "redis" if self._redis is not None else "local",
        }
