import pytest
from pydantic import ValidationError

from app.config import Settings


def test_environment_is_required(monkeypatch):
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_environment_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "bogus")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_environment_accepts_valid_value(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "staging")
    settings = Settings(_env_file=None)
    assert settings.environment == "staging"
    assert settings.is_production is False


def test_defaults_apply_when_optional_vars_absent(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    settings = Settings(_env_file=None)
    assert settings.log_level == "INFO"
    assert settings.is_production is True
