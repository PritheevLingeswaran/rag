"""Stage 8.9: the privacy policy is served, linked, and its claims are
enforced by tests so code and policy cannot silently drift."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app

REPO = Path(__file__).resolve().parent.parent.parent


def make_client() -> TestClient:
    from app.config import get_settings

    get_settings.cache_clear()
    return TestClient(create_app())


def test_privacy_policy_served_and_linked():
    with make_client() as client:
        resp = client.get("/privacy")
        assert resp.status_code == 200
        assert "independently operated" in resp.text
        assert "never stored" in resp.text
        # linked from the app's own API description
        spec = client.get("/openapi.json").json()
        assert "/privacy" in spec["info"]["description"]


def test_policy_claim_queries_never_persisted():
    """Policy: 'We do not store your queries in any database.'
    Proof: QueryLogRepo (the only query-text writer) is referenced
    nowhere in the app outside its own definition."""
    hits = []
    for py in (REPO / "app").rglob("*.py"):
        if py.name == "repositories.py":
            continue
        if "QueryLogRepo" in py.read_text(encoding="utf-8"):
            hits.append(str(py))
    assert hits == [], f"QueryLogRepo wired into: {hits} -- update PRIVACY.md first"


def test_policy_claim_no_query_text_or_ip_in_app_logs():
    """Policy: app logs never contain query text, answers, or IPs.
    Proof: no logger call in the serving path passes query/answer text;
    the one client-address log is the dev-only anonymous branch."""
    log_call = re.compile(r"logger\.(info|warning|error|critical)\(([^)]*)\)",
                          re.DOTALL)
    offenders = []
    for py in (REPO / "app").rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for match in log_call.finditer(text):
            args = match.group(2)
            if re.search(r"\b(query|answer)\s*=\s*(body|result|request)\b",
                         args) or "query_text" in args:
                offenders.append(f"{py.name}: {args[:80]}")
    assert offenders == [], offenders
    # the only client-address log is behind the non-production branch
    deps = (REPO / "app" / "api" / "deps.py").read_text(encoding="utf-8")
    assert "anonymous_request_dev_mode" in deps
    assert deps.index("is_production") < deps.index("anonymous_request_dev_mode")


def test_policy_claim_access_log_disabled_in_image():
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert "--no-access-log" in dockerfile
    assert "PRIVACY.md" in dockerfile


def test_policy_claim_cache_ttl_one_hour():
    from app.config import Settings

    assert Settings(_env_file=None, environment="development").cache_ttl_s <= 3600
