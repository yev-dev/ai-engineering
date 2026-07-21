"""CLI integration test for the LangGraph time series construction workflow.

Tests the end-to-end flow mimicking user interactions with the CLI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from time_series_construction_langgraph.agents_definition import CallbackEventType
from time_series_construction_langgraph.graph import TimeSeriesConstructionGraph
from time_series_construction_langgraph.tools import (
    apply_gap_filling,
    available_data_sources,
    build_timeseries,
    check_data_quality,
    generate_report,
    get_instrument_details,
    historical_prices,
    recommend_gap_methods,
    visualize_timeseries,
)

# Import test fixtures from the main test file
pytest_plugins = ["time_series_construction_langgraph.tests.test_workflow_int"]


# ---------------------------------------------------------------------------
# Test: Full CLI workflow simulation
# ---------------------------------------------------------------------------


class TestCLIIntegration:
    """End-to-end CLI workflow - simulates user typing requests and responses."""

    def test_full_workflow_with_mocked_llm(
        self, mock_data_dir: Path, mock_output_dir: Path
    ) -> None:
        """Simulate workflow: LLM executes tools and reaches Final Answer."""
        # Create a mock factory that simulates tool execution
        class MockSingleStepFactory:
            def __init__(self) -> None:
                self.call_count = 0

            def chat(self, request: Any) -> str:
                self.call_count += 1
                messages = request.messages
                # Check if we have tool results in the messages
                has_tool_result = any("Tool result:" in m.get("content", "") for m in messages)
                if has_tool_result:
                    # After tool execution, return final answer
                    return "Final Answer: Time series construction completed for AAPL."
                # First call: resolve and execute tools
                return (
                    "Thought: Looking up AAPL.\n"
                    "Action: get_instrument_details\n"
                    "Action Input: {\"query\": \"AAPL\"}"
                )

        # Inject mock factory
        import time_series_construction_langgraph.graph as graph_module
        original_factory = graph_module._factory
        graph_module._factory = MockSingleStepFactory()

        try:
            graph = TimeSeriesConstructionGraph()
            
            # Process request - should execute tool and get final answer
            events = graph.process_user_request("Build AAPL from 2023 to 2024")
            
            # Should have completed with a final answer
            completed = [e for e in events if e.type == CallbackEventType.AGENT_COMPLETED]
            assert len(completed) >= 1, f"Expected AGENT_COMPLETED, got {events}"
            
            # No errors should occur
            errors = [e for e in events if e.type == CallbackEventType.ERROR]
            assert not errors, f"Unexpected errors: {errors}"
        finally:
            graph_module._factory = original_factory

    def test_cli_handles_empty_and_finishes(
        self, mock_data_dir: Path, mock_output_dir: Path
    ) -> None:
        """CLI should handle empty input and request clarification."""
        graph = TimeSeriesConstructionGraph()
        events = graph.process_user_request("")
        awaiting = [e for e in events if e.type == CallbackEventType.AWAITING_USER_INPUT]
        assert len(awaiting) == 1
        assert "What financial time series" in awaiting[0].payload.get("prompt", "")

    def test_cli_handles_invalid_ticker(
        self, mock_data_dir: Path, mock_output_dir: Path
    ) -> None:
        """CLI should handle an unknown ticker gracefully."""
        class MockFailFactory:
            def chat(self, request: Any) -> str:
                return "Final Answer: I cannot process that request."

        import time_series_construction_langgraph.graph as graph_module
        original_factory = graph_module._factory
        graph_module._factory = MockFailFactory()

        try:
            graph = TimeSeriesConstructionGraph()
            events = graph.process_user_request("Build NOPE from 2023 to 2024")
            completed = [e for e in events if e.type == CallbackEventType.AGENT_COMPLETED]
            assert len(completed) >= 1
        finally:
            graph_module._factory = original_factory

    def test_cli_quit_command(
        self, mock_data_dir: Path, mock_output_dir: Path
    ) -> None:
        """CLI should handle quit command during paused state."""
        graph = TimeSeriesConstructionGraph()
        graph.waiting = True  # Simulate we're in a paused state
        
        events = graph.process_user_response("quit")
        errors = [e for e in events if e.type == CallbackEventType.ERROR]
        assert len(errors) == 1
        assert "cancelled" in errors[0].payload.get("message", "").lower()