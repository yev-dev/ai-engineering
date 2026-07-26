"""Tests for provider/model environment resolution in models factory."""
from __future__ import annotations

from financial_time_series_construction.models import ModelRequestFactory


def test_describe_environment_uses_provider_default_model(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("LLM_DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")

    config = ModelRequestFactory.describe_environment()

    assert config["provider"] == "deepseek"
    assert config["model"] == "deepseek/deepseek-chat"
    assert config["provider_kwargs"]["api_key"] == "sk-test"
    assert config["provider_kwargs"]["api_base"] == "https://api.deepseek.com"


def test_describe_environment_respects_full_model_override(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "github/gpt-4.1-mini")

    config = ModelRequestFactory.describe_environment()

    assert config["provider"] == "ollama"
    assert config["model"] == "github/gpt-4.1-mini"
