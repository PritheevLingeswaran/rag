"""Admission control: a bounded queue in front of the retrieval pipeline.

Why this exists (measured, not theoretical): the pipeline is CPU-bound,
so under overload every extra concurrent request makes ALL requests
slower -- Stage 3/4 load tests showed depth-20 rerank going from 14s
(conc 1) to 77s p50 (conc 4) at 0.1 CPU, and the discarded host-stall
row showed what unbounded queueing does to tail latency. The honest
alternative to unbounded queueing is refusing early: a request that
would wait behind a deep queue gets an immediate 503 + Retry-After
instead of a multi-minute hang the client has usually abandoned anyway.

Semantics:
    max_concurrency  requests executing in the pipeline simultaneously
    max_queue_depth  requests allowed to WAIT for a slot
    beyond that      QueueFullError -> HTTP 503 + Retry-After

Retry-After is estimated from a live EWMA of service time:
    (in_flight + waiting) * ewma_service_time / max_concurrency
rounded up, floored at 1s -- i.e. "when the current backlog should have
drained", not a magic constant.

Counters are plain ints: this controller lives on the event loop and is
only touched from async context (single-threaded), which uvicorn with a
single worker guarantees. The 512MB cap forces a single process anyway
(the models cannot fit twice), so admission state is process-global by
construction; a multi-instance future needs a shared-state redesign and
says so here.
"""

from __future__ import annotations

import asyncio
import math
import time
from contextlib import asynccontextmanager

from app.logging_config import get_logger

logger = get_logger(__name__)


class QueueFullError(Exception):
    def __init__(self, retry_after_s: int) -> None:
        super().__init__(f"admission queue full; retry after {retry_after_s}s")
        self.retry_after_s = retry_after_s


class AdmissionController:
    _EWMA_ALPHA = 0.2

    def __init__(self, max_concurrency: int, max_queue_depth: int,
                 initial_service_time_s: float = 1.0) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if max_queue_depth < 0:
            raise ValueError("max_queue_depth must be >= 0")
        self.max_concurrency = max_concurrency
        self.max_queue_depth = max_queue_depth
        self._sem = asyncio.Semaphore(max_concurrency)
        self._waiting = 0
        self._in_flight = 0
        self._rejected_total = 0
        self._admitted_total = 0
        self._service_time_ewma_s = initial_service_time_s

    def _retry_after_s(self) -> int:
        backlog = self._in_flight + self._waiting
        estimate = backlog * self._service_time_ewma_s / self.max_concurrency
        return max(1, math.ceil(estimate))

    @asynccontextmanager
    async def admit(self):
        # Reject only requests that would actually have to WAIT while the
        # queue is at capacity. A free execution slot admits immediately
        # even with max_queue_depth=0 (waiting>0 means no slot is truly
        # free: earlier waiters own the next releases).
        would_wait = (
            self._in_flight >= self.max_concurrency or self._waiting > 0
        )
        if would_wait and self._waiting >= self.max_queue_depth:
            self._rejected_total += 1
            retry_after = self._retry_after_s()
            logger.warning(
                "admission_rejected_queue_full",
                waiting=self._waiting, in_flight=self._in_flight,
                max_queue_depth=self.max_queue_depth,
                retry_after_s=retry_after,
            )
            raise QueueFullError(retry_after)
        self._waiting += 1
        try:
            await self._sem.acquire()
        finally:
            self._waiting -= 1
        self._in_flight += 1
        self._admitted_total += 1
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._service_time_ewma_s = (
                self._EWMA_ALPHA * elapsed
                + (1 - self._EWMA_ALPHA) * self._service_time_ewma_s
            )
            self._in_flight -= 1
            self._sem.release()

    def snapshot(self) -> dict:
        return {
            "in_flight": self._in_flight,
            "waiting": self._waiting,
            "max_concurrency": self.max_concurrency,
            "max_queue_depth": self.max_queue_depth,
            "admitted_total": self._admitted_total,
            "rejected_total": self._rejected_total,
            "service_time_ewma_s": round(self._service_time_ewma_s, 3),
        }
