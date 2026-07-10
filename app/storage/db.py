"""Postgres connection helper and SQL migration runner.

Migrations are plain SQL files in migrations/, applied in filename order,
each inside its own transaction, recorded in schema_migrations. Plain SQL
(vs. an ORM/alembic) keeps the schema reviewable in one file and the
dependency count down for free-tier deploys; the cost is no auto-generated
downgrades -- rollback of schema is a new forward migration, which is the
safer production habit anyway.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

from app.errors import MigrationError
from app.logging_config import get_logger

logger = get_logger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"


def connect(database_url: str) -> psycopg.Connection:
    """Open a connection with autocommit OFF; callers own transactions."""
    return psycopg.connect(database_url)


def run_migrations(conn: psycopg.Connection) -> list[str]:
    """Apply pending migrations in order. Returns the names applied."""
    with conn.transaction():
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name       TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
    applied = {
        row[0]
        for row in conn.execute("SELECT name FROM schema_migrations").fetchall()
    }
    pending = sorted(
        p for p in MIGRATIONS_DIR.glob("*.sql") if p.name not in applied
    )
    done: list[str] = []
    for path in pending:
        sql = path.read_text(encoding="utf-8")
        try:
            with conn.transaction():
                conn.execute(sql)
                conn.execute(
                    "INSERT INTO schema_migrations (name) VALUES (%s)",
                    (path.name,),
                )
        except psycopg.Error as exc:
            raise MigrationError(f"migration {path.name} failed: {exc}") from exc
        logger.info("migration_applied", migration=path.name)
        done.append(path.name)
    return done
