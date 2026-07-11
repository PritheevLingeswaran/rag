"""AdmissionController unit tests (pure asyncio, no HTTP)."""

import asyncio

import pytest

from app.api.admission import AdmissionController, QueueFullError


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_rejects_exactly_beyond_queue_depth():
    async def scenario():
        ac = AdmissionController(max_concurrency=1, max_queue_depth=2)
        release = asyncio.Event()
        started = asyncio.Event()

        async def occupant():
            async with ac.admit():
                started.set()
                await release.wait()

        async def waiter():
            async with ac.admit():
                pass

        occupant_task = asyncio.create_task(occupant())
        await started.wait()
        # fill the queue to exactly max_queue_depth
        waiters = [asyncio.create_task(waiter()) for _ in range(2)]
        await asyncio.sleep(0)  # let them reach the semaphore
        assert ac.snapshot()["waiting"] == 2

        # one past the bound: immediate rejection with retry-after
        with pytest.raises(QueueFullError) as exc_info:
            async with ac.admit():
                pass
        assert exc_info.value.retry_after_s >= 1

        release.set()
        await asyncio.gather(occupant_task, *waiters)
        snap = ac.snapshot()
        assert snap["rejected_total"] == 1
        assert snap["admitted_total"] == 3
        assert snap["in_flight"] == 0 and snap["waiting"] == 0

    run(scenario())


def test_concurrency_is_actually_bounded():
    async def scenario():
        ac = AdmissionController(max_concurrency=2, max_queue_depth=10)
        peak = 0
        current = 0

        async def job():
            nonlocal peak, current
            async with ac.admit():
                current += 1
                peak = max(peak, current)
                await asyncio.sleep(0.01)
                current -= 1

        await asyncio.gather(*(job() for _ in range(8)))
        assert peak == 2   # never more than max_concurrency in flight

    run(scenario())


def test_retry_after_scales_with_backlog_and_service_time():
    async def scenario():
        ac = AdmissionController(max_concurrency=1, max_queue_depth=1,
                                 initial_service_time_s=2.0)
        release = asyncio.Event()

        async def occupant():
            async with ac.admit():
                await release.wait()

        task = asyncio.create_task(occupant())
        await asyncio.sleep(0)

        async def waiter():
            async with ac.admit():
                pass
        wtask = asyncio.create_task(waiter())
        await asyncio.sleep(0)

        with pytest.raises(QueueFullError) as exc_info:
            async with ac.admit():
                pass
        # backlog = 1 in flight + 1 waiting, ewma 2s, conc 1 => ~4s
        assert exc_info.value.retry_after_s == 4
        release.set()
        await asyncio.gather(task, wtask)

    run(scenario())


def test_queue_depth_zero_means_no_waiting():
    async def scenario():
        ac = AdmissionController(max_concurrency=1, max_queue_depth=0)
        release = asyncio.Event()

        async def occupant():
            async with ac.admit():
                await release.wait()

        task = asyncio.create_task(occupant())
        await asyncio.sleep(0)
        with pytest.raises(QueueFullError):
            async with ac.admit():
                pass
        release.set()
        await task

    run(scenario())


def test_invalid_bounds_rejected():
    with pytest.raises(ValueError):
        AdmissionController(max_concurrency=0, max_queue_depth=1)
    with pytest.raises(ValueError):
        AdmissionController(max_concurrency=1, max_queue_depth=-1)
