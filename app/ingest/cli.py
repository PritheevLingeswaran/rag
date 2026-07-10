"""Ingestion/index-management CLI.

    python -m app.ingest.cli migrate
    python -m app.ingest.cli ingest data/corpus_v1.jsonl [--activate]
    python -m app.ingest.cli versions
    python -m app.ingest.cli activate <version_id>
    python -m app.ingest.cli rollback
    python -m app.ingest.cli gc [--keep N]

Requires DATABASE_URL (and INDEX_ROOT, defaulting to ./indexes) in the
environment or .env; exits non-zero with the validation error otherwise.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from app.config import get_settings, require_setting
from app.ingest.embedder import HashingEmbedder
from app.ingest.faiss_store import FaissStore
from app.ingest.pipeline import IngestionPipeline
from app.logging_config import configure_logging, get_logger
from app.storage.db import connect, run_migrations
from app.storage.repositories import IndexVersionRepo

logger = get_logger(__name__)

DEFAULT_GC_KEEP = 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("migrate")
    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("corpus", type=Path)
    p_ingest.add_argument("--activate", action="store_true",
                          help="activate the version if the run succeeds")
    sub.add_parser("versions")
    p_activate = sub.add_parser("activate")
    p_activate.add_argument("version_id")
    sub.add_parser("rollback")
    p_gc = sub.add_parser("gc")
    p_gc.add_argument("--keep", type=int, default=DEFAULT_GC_KEEP,
                      help="ready versions to retain besides active")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings)
    db_url = require_setting(settings.database_url, "DATABASE_URL")
    store = FaissStore(Path(settings.index_root))

    with connect(db_url) as conn:
        versions = IndexVersionRepo(conn)

        if args.command == "migrate":
            applied = run_migrations(conn)
            print(f"applied: {applied or 'nothing pending'}")

        elif args.command == "ingest":
            pipeline = IngestionPipeline(conn, HashingEmbedder(), store)
            report = pipeline.run(args.corpus)
            print(json.dumps(dataclasses.asdict(report), indent=2))
            if report.status in ("failed", "aborted_input"):
                return 1
            if args.activate and report.status == "built":
                versions.activate(report.version_id)
                print(f"activated: {report.version_id}")

        elif args.command == "versions":
            active = versions.get_active()
            for v in versions.list_versions():
                marker = " <-- ACTIVE" if active and v.version_id == active.version_id else ""
                print(f"{v.version_id}  {v.status:<12} chunks={v.chunk_count} "
                      f"embedder={v.embedder_id}{marker}")

        elif args.command == "activate":
            versions.activate(args.version_id)
            print(f"activated: {args.version_id}")

        elif args.command == "rollback":
            rolled_back, now_active = versions.rollback_active()
            print(f"rolled back: {rolled_back}\nnow active : {now_active}")

        elif args.command == "gc":
            keep: set[str] = set()
            ready_seen = 0
            for v in reversed(versions.list_versions()):
                if v.status == "active":
                    keep.add(v.version_id)
                elif v.status == "ready" and ready_seen < args.keep:
                    keep.add(v.version_id)
                    ready_seen += 1
            removed = store.gc(keep)
            print(f"kept: {sorted(keep)}\nremoved: {removed or 'nothing'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
