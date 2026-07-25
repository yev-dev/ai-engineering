"""Deterministic CLI smoke tests for follow-up progression behavior."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest_plugins = ["financial_time_series_construction.tests.test_workflow_int"]


class _SequenceFactory:
    """Deterministic LLM response sequence used for CLI smoke tests."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses[:]

    def chat(self, request: Any) -> str:
        if not self._responses:
            raise AssertionError("No more mocked LLM responses available.")
        return self._responses.pop(0)


class TestCLISmoke:
    """Validate CLI behavior for the reported no-progress loop scenario."""

    def test_follow_up_dates_progress_beyond_orchestrator(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_data_dir: Path,
    ) -> None:
        """After follow-up date input, the next step should not pause at Orchestrator."""
        from financial_time_series_construction import cli
        import financial_time_series_construction.processor as processor_module

        session_id = "smoke_followup_progress"
        output_root = tmp_path / "artifacts"
        monkeypatch.setenv("TIME_SERIES_OUTPUT_DIR", str(output_root))

        responses = [
            (
                "Thought: I need a date range before proceeding.\n"
                "Action: request_human_input\n"
                "Action Input: {\"prompt\": \"Please provide the start and end dates for the historical data you'd like to retrieve (e.g. 2023-01-01 to 2023-12-31).\"}"
            ),
            (
                "Thought: Resolve the provided ticker first.\n"
                "Action: get_instrument_details\n"
                "Action Input: {\"query\": \"AAPL\"}"
            ),
            "Final Answer: Instrument resolved and ready for market-data retrieval.",
        ]
        factory = _SequenceFactory(responses)
        monkeypatch.setattr(
            processor_module.ModelRequestFactory,
            "from_environment",
            classmethod(lambda cls: factory),
        )

        provided_inputs = iter(["AAPL between 2023-01-01 to 2023-12-31"])

        def _fake_input(_prompt: str) -> str:
            try:
                return next(provided_inputs)
            except StopIteration as exc:
                raise AssertionError("CLI requested more follow-up inputs than expected.") from exc

        monkeypatch.setattr("builtins.input", _fake_input)
        monkeypatch.setattr(
            "sys.argv",
            [
                "cli.py",
                "--request",
                "AAPL stock",
                "--session-id",
                session_id,
            ],
        )

        cli.main()

        events_path = output_root / session_id / "events.json"
        trace_path = output_root / session_id / "react_trace.json"
        assert events_path.exists(), "Expected CLI to persist events.json artifact"
        assert trace_path.exists(), "Expected CLI to persist react_trace.json artifact"

        events = json.loads(events_path.read_text())
        trace_records = json.loads(trace_path.read_text())
        delegated_targets = [
            entry["payload"].get("to_agent", "")
            for entry in events
            if entry.get("type") == "delegated"
        ]

        assert "ReferenceDataAgent" in delegated_targets, (
            "Expected follow-up to progress by delegating beyond Orchestrator."
        )
        completed_agents = [
            entry["payload"].get("agent", "")
            for entry in events
            if entry.get("type") == "agent_completed"
        ]
        assert "ReferenceDataAgent" in completed_agents, (
            "Expected workflow to progress beyond Orchestrator after follow-up."
        )
        assert any(record.get("type") == "llm_response" for record in trace_records)
        assert any(
            record.get("type") == "tool_call"
            and record.get("payload", {}).get("description")
            for record in trace_records
        )
