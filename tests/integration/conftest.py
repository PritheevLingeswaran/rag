"""Integration fixtures: real Postgres + Redis (docker containers).

Connection targets default to the disposable local test containers
(ports 55432/56379) and can be overridden with RAGP_TEST_DATABASE_URL /
RAGP_TEST_REDIS_URL. If Postgres is unreachable the whole integration
suite SKIPS (visibly, with a reason) rather than failing or silently
faking the database.
"""

from __future__ import annotations

import os
import time

import psycopg
import pytest

TEST_DATABASE_URL = os.environ.get(
    "RAGP_TEST_DATABASE_URL",
    "postgresql://ragp:ragp@127.0.0.1:55432/ragp_test",
)
TEST_REDIS_URL = os.environ.get(
    "RAGP_TEST_REDIS_URL", "redis://127.0.0.1:56379/0"
)


def _wait_for_postgres(url: str, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(url, connect_timeout=2):
                return True
        except psycopg.OperationalError:
            time.sleep(0.5)
    return False


@pytest.fixture(scope="session")
def pg_available() -> bool:
    if not _wait_for_postgres(TEST_DATABASE_URL):
        pytest.skip(f"integration Postgres unreachable at {TEST_DATABASE_URL}")
    return True


@pytest.fixture()
def conn(pg_available):
    """Fresh schema per test: drop and recreate public, re-run migrations."""
    from app.storage.db import run_migrations

    with psycopg.connect(TEST_DATABASE_URL) as c:
        with c.transaction():
            c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        run_migrations(c)
        yield c


@pytest.fixture()
def redis_store():
    from app.storage.redis_store import RedisStore

    store = RedisStore(TEST_REDIS_URL, namespace="ragp_test")
    if not store.ping():
        pytest.skip(f"integration Redis unreachable at {TEST_REDIS_URL}")
    store._client.flushdb()
    return store
