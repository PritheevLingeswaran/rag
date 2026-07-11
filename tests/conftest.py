"""Shared test fixtures.

ENVIRONMENT has no default in app.config.Settings by design (see
app/config.py) -- it must be set explicitly for any real run. Tests set a
default here so importing app.main during collection doesn't fail, while
tests/test_config.py still exercises the fail-loud behavior directly with
monkeypatch + Settings(_env_file=None).
"""

import os

import pytest

os.environ.setdefault("ENVIRONMENT", "development")
# Tests never boot the real models through the app lifespan; endpoints
# under test inject stub services onto app.state instead.
os.environ.setdefault("SERVE_PIPELINE", "false")


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
