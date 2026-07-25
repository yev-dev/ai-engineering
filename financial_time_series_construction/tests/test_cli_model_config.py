"""CLI tests for model config listing and framework runtime selection."""
from __future__ import annotations

import json

import pytest


def test_cli_list_model_config_prints_and_exits(monkeypatch, capsys) -> None:
    from financial_time_series_construction import cli

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "ollama/qwen2.5:1.5b")
    monkeypatch.setattr("sys.argv", ["cli.py", "--list-model-config"])

    cli.main()

    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["provider"] == "ollama"
    assert payload["model"] == "ollama/qwen2.5:1.5b"


def test_cli_framework_choice_defaults_to_autogen(monkeypatch) -> None:
    from financial_time_series_construction.runtime import build_runtime

    runtime = build_runtime("autogen", session_id="test_session", factory=None)
    assert runtime.session_id == "test_session"


def test_runtime_factory_crawl_not_wired_yet() -> None:
    from financial_time_series_construction.runtime import build_runtime

    with pytest.raises(NotImplementedError):
        build_runtime("crawl", session_id="test_session", factory=None)
