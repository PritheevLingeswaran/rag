"""Quota/cost circuit breakers + alerting (Stage 7.7).

Model shared by every breaker (limits sourced from Stage 2.5):

    usage < 80% of enforced budget      CLOSED   normal operation
    usage >= 80%                        ALERT    still serving; one alert
                                                 per resource per day fires
                                                 (webhook + CRITICAL log)
    usage >= enforced (margin * hard)   OPEN     the guarded operation is
                                                 refused/bypassed BEFORE
                                                 the provider limit is hit;
                                                 the app degrades, labeled,
                                                 never crashes
    past the provider hard limit        reactive paths (already built and
                                                 tested) absorb provider
                                                 errors: 429 cooldown, soft
                                                 cache fails, typed abort

Alerting is fire-and-forget BY DESIGN: an alert failure must never take
down serving (it is logged). The webhook is a plain JSON POST -- works
with ntfy.sh, Discord, Slack, healthchecks.io, all free.

Breakers here protect the three quotas that can kill a live deployment
silently: Gemini RPD (via QuotaGuard + alerts), Neon Postgres storage
(gates every write path), Upstash Redis daily command budget (RedisStore
self-metering with graceful bypass). Render instance-hours/bandwidth
cannot be measured in-process and are an ops-runbook concern, stated in
the report, not silently skipped.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable

import httpx

from app.logging_config import get_logger

logger = get_logger(__name__)

STATE_CLOSED = "closed"
STATE_ALERT = "alert"
STATE_OPEN = "open"

DEFAULT_MARGIN = 0.9
DEFAULT_ALERT_PCT = 0.8


def utc_day(now: float) -> str:
    return time.strftime("%Y%m%d", time.gmtime(now))


class AlertManager:
    """Sends at most one alert per (resource, UTC day). Never raises."""

    def __init__(self, webhook_url: str | None = None,
                 transport: httpx.BaseTransport | None = None,
                 now_fn: Callable[[], float] = time.time) -> None:
        self.webhook_url = webhook_url
        self._client = httpx.Client(timeout=5.0, transport=transport)
        self._now = now_fn
        self._sent: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def fire(self, resource: str, pct: float, message: str,
             details: dict | None = None) -> bool:
        """Returns True if an alert was (newly) emitted for this window."""
        key = (resource, utc_day(self._now()))
        with self._lock:
            if key in self._sent:
                return False
            self._sent.add(key)
        # CRITICAL log always -- even with no webhook configured, the
        # operator's log stream carries the alert.
        logger.critical(
            "quota_alert", resource=resource, pct=round(pct, 3),
            message=message, **(details or {}),
        )
        if self.webhook_url:
            try:
                self._client.post(self.webhook_url, json={
                    "resource": resource,
                    "pct_of_budget": round(pct, 3),
                    "message": message,
                    "details": details or {},
                    "ts": self._now(),
                })
            except httpx.HTTPError as exc:
                logger.error("quota_alert_webhook_failed", error=str(exc))
        return True


@dataclass(frozen=True)
class BreakerDecision:
    allowed: bool
    state: str          # closed | alert | open
    used: float
    enforced: float
    hard_limit: float


class ResourceBudget:
    """Counted budget with a daily window (UTC): counts events, alerts at
    alert_pct of the enforced budget, opens at the enforced budget."""

    def __init__(self, name: str, hard_limit: int,
                 margin: float = DEFAULT_MARGIN,
                 alert_pct: float = DEFAULT_ALERT_PCT,
                 alerts: AlertManager | None = None,
                 now_fn: Callable[[], float] = time.time) -> None:
        if hard_limit < 1:
            raise ValueError("hard_limit must be >= 1")
        if not 0 < alert_pct < margin <= 1.0:
            raise ValueError("need 0 < alert_pct < margin <= 1")
        self.name = name
        self.hard_limit = hard_limit
        self.enforced = math.floor(hard_limit * margin)
        self.alert_at = alert_pct * self.enforced
        self.alerts = alerts
        self._now = now_fn
        self._lock = threading.Lock()
        self._window: str | None = None
        self._used = 0

    def _roll_window(self) -> None:
        day = utc_day(self._now())
        if day != self._window:
            self._window = day
            self._used = 0

    def _decide(self, used: int) -> BreakerDecision:
        if used >= self.enforced:
            state, allowed = STATE_OPEN, False
        elif used >= self.alert_at:
            state, allowed = STATE_ALERT, True
        else:
            state, allowed = STATE_CLOSED, True
        return BreakerDecision(allowed, state, used, self.enforced,
                               self.hard_limit)

    def record_and_check(self, n: int = 1) -> BreakerDecision:
        """Admission uses the PRE-event count (exactly `enforced` events
        pass); the returned state classifies the POST-event count (the
        80% alert fires during the event that crosses the line)."""
        with self._lock:
            self._roll_window()
            if self._used >= self.enforced:
                decision = BreakerDecision(False, STATE_OPEN, self._used,
                                           self.enforced, self.hard_limit)
            else:
                self._used += n
                state = (STATE_ALERT if self._used >= self.alert_at
                         else STATE_CLOSED)
                decision = BreakerDecision(True, state, self._used,
                                           self.enforced, self.hard_limit)
        if decision.state == STATE_ALERT and self.alerts is not None:
            self.alerts.fire(
                self.name, decision.used / self.enforced,
                f"{self.name} at {decision.used}/{self.enforced} of "
                f"enforced daily budget (hard limit {self.hard_limit})",
            )
        return decision

    def peek(self) -> BreakerDecision:
        with self._lock:
            self._roll_window()
            return self._decide(self._used)


class PostgresStorageBreaker:
    """Measured (not counted) budget: gates WRITE paths on database size.

    size_fn returns current DB bytes (production:
    SELECT pg_database_size(current_database()); tests: a fake). The
    reading is cached for cache_s to avoid a size query per write.
    Reads are never gated: Neon keeps reads working past the storage
    limit; only writes fail there, so only writes are protected here.
    """

    def __init__(self, size_fn: Callable[[], int], limit_bytes: int,
                 margin: float = DEFAULT_MARGIN,
                 alert_pct: float = DEFAULT_ALERT_PCT,
                 alerts: AlertManager | None = None,
                 cache_s: float = 60.0,
                 now_fn: Callable[[], float] = time.time) -> None:
        if limit_bytes < 1:
            raise ValueError("limit_bytes must be >= 1")
        self.size_fn = size_fn
        self.limit_bytes = limit_bytes
        self.enforced = math.floor(limit_bytes * margin)
        self.alert_at = alert_pct * self.enforced
        self.alerts = alerts
        self.cache_s = cache_s
        self._now = now_fn
        self._lock = threading.Lock()
        self._cached_size: int | None = None
        self._cached_at = 0.0

    def check_writable(self) -> BreakerDecision:
        now = self._now()
        with self._lock:
            if (self._cached_size is None
                    or now - self._cached_at >= self.cache_s):
                try:
                    self._cached_size = int(self.size_fn())
                except Exception as exc:  # noqa: BLE001 - any failure here
                    # Fail-open with a loud log: refusing all writes
                    # because the SIZE CHECK broke would turn a monitoring
                    # bug into an outage. The provider's own hard limit
                    # remains the reactive backstop.
                    logger.error("storage_size_check_failed", error=str(exc))
                    self._cached_size = 0
                self._cached_at = now
            size = self._cached_size

        if size >= self.enforced:
            state, allowed = STATE_OPEN, False
        elif size >= self.alert_at:
            state, allowed = STATE_ALERT, True
        else:
            state, allowed = STATE_CLOSED, True
        if state == STATE_ALERT and self.alerts is not None:
            self.alerts.fire(
                "postgres_storage", size / self.enforced,
                f"postgres storage at {size}/{self.enforced} bytes of "
                f"enforced budget (hard limit {self.limit_bytes})",
            )
        return BreakerDecision(allowed, state, size, self.enforced,
                               self.limit_bytes)
