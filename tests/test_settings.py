"""Tests for environment-backed application settings."""
import pytest

from config.settings import get_settings


def test_llm_temperature_can_be_configured_with_environment(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")

    assert get_settings().llm_temperature == 0.7


@pytest.mark.parametrize("value", ["not-a-number", "-0.1", "2.1"])
def test_invalid_llm_temperature_is_reported(monkeypatch, value):
    monkeypatch.setenv("LLM_TEMPERATURE", value)

    with pytest.raises(ValueError, match="LLM_TEMPERATURE"):
        get_settings()
