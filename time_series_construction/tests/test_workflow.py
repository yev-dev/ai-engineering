"""Workflow and data-contract tests for the time series construction application."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from time_series_v2.agents_definition import CallbackEventType, get_agent
from time_series_v2.processor import TimeSeriesConstructionProcessor
from time_series_v2.tools import (
    apply_gap_filling,
    check_data_quality,
    get_instrument_details,
    historical_prices,
)


def test_agent_specs_have_goals_and_tools() -> None:
    orchestrator = get_agent("orchestrator")
    assert orchestrator is not None
    assert orchestrator.goal
    assert orchestrator.tools
    assert orchestrator.guardrails


def test_apple_data_workflow_uses_fixture_data() -> None:
    instrument = get_instrument_details("Apple")
    assert instrument["found"] is True
    assert instrument["symbol"] == "AAPL"

    prices = historical_prices("AAPL", "2023-10-02", "2023-10-10", "yahoo")
    quality = check_data_quality(prices["prices"], "yahoo", "AAPL")
    filled = apply_gap_filling(prices, "none")

    assert len(prices["dates"]) == 7
    assert quality["completeness_pct"] == 100.0
    assert filled["prices"] == prices["prices"]


@pytest.mark.parametrize("query", ["AAPL", "Apple", "Apple Inc."])
def test_instrument_lookup_accepts_ticker_symbol_and_full_name(query: str) -> None:
    result = get_instrument_details(query=query)
    assert result["found"] is True
    assert result["symbol"] == "AAPL"


def test_instrument_lookup_accepts_legacy_symbol_argument() -> None:
    result = get_instrument_details(symbol="AAPL")
    assert result["found"] is True
    assert result["symbol"] == "AAPL"


def test_instrument_lookup_suggests_correction_for_appl_typo() -> None:
    result = get_instrument_details(query="APPL")
    assert result["found"] is False
    assert "AAPL" in result["suggestions"]


def test_missing_instrument_is_reported_to_user() -> None:
    result = get_instrument_details("not-a-real-security")
    assert result["found"] is False


def test_empty_date_range_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="No historical data is available"):
        historical_prices("AAPL", "2021-01-01", "2021-01-31", "yahoo")


def test_orchestrator_delegates_to_reference_agent() -> None:
    responses = iter([
        "Thought: route.\nAction: delegate_to_agent\nAction Input: "
        '{"agent_name": "ReferenceDataAgent", "request": "resolve Apple"}',
        "Final Answer: instrument lookup reached",
    ])
    factory = SimpleNamespace(chat=lambda request: next(responses))
    processor = TimeSeriesConstructionProcessor(factory=factory)

    events = processor.process_user_request("create a time series for Apple")

    assert any(
        event.type == CallbackEventType.AGENT_COMPLETED
        and event.payload.get("agent") == "ReferenceDataAgent"
        for event in events
    )


def test_orchestrator_recovers_when_model_mentions_agent_in_final_answer() -> None:
    responses = iter([
        "Final Answer: ReferenceDataAgent was not recognized. Please provide more details.",
        "Final Answer: reference lookup reached",
    ])
    factory = SimpleNamespace(chat=lambda request: next(responses))
    processor = TimeSeriesConstructionProcessor(factory=factory)

    events = processor.process_user_request("create time series for Apple between 2023 and 2025")

    assert any(
        event.type == CallbackEventType.AGENT_COMPLETED
        and event.payload.get("agent") == "ReferenceDataAgent"
        for event in events
    )


def test_orchestrator_recovers_when_delegation_input_is_fenced_json() -> None:
    responses = iter([
        "Thought: route.\nAction: delegate_to_agent\nAction Input:\n"
        "```json\n{\"agent_name\": \"ReferenceDataAgent\", "
        "\"request\": \"resolve Apple\"}\n```",
        "Final Answer: reference lookup reached",
    ])
    factory = SimpleNamespace(chat=lambda request: next(responses))
    processor = TimeSeriesConstructionProcessor(factory=factory)

    events = processor.process_user_request("create time series for Apple between 2023 and 2025")

    assert any(
        event.type == CallbackEventType.AGENT_COMPLETED
        and event.payload.get("agent") == "ReferenceDataAgent"
        for event in events
    )


def test_orchestrator_recovers_from_unparseable_delegation() -> None:
    responses = iter([
        "Thought: route. Action: delegate_to_agent Action Input: malformed",
        "Final Answer: reference lookup reached",
    ])
    factory = SimpleNamespace(chat=lambda request: next(responses))
    processor = TimeSeriesConstructionProcessor(factory=factory)

    events = processor.process_user_request("create time series for Apple between 2023 and 2025")

    assert any(
        event.type == CallbackEventType.AGENT_COMPLETED
        and event.payload.get("agent") == "ReferenceDataAgent"
        for event in events
    )


def test_reference_agent_recovers_when_model_reports_failed_market_delegation() -> None:
    responses = iter([
        "Thought: resolve.\nAction: get_instrument_details\nAction Input: {\"query\": \"Apple\"}",
        "Final Answer: I identified Apple as AAPL, but MarketDataAgent was reported as unknown.",
        "Final Answer: market data loaded",
    ])
    factory = SimpleNamespace(chat=lambda request: next(responses))
    processor = TimeSeriesConstructionProcessor(factory=factory)

    events = processor.process_user_request("create time series for Apple between 2023 and 2025")

    assert any(
        event.type == CallbackEventType.AGENT_COMPLETED
        and event.payload.get("agent") == "MarketDataAgent"
        for event in events
    )


def test_list_valued_tool_result_is_not_treated_as_callback_events() -> None:
    responses = iter([
        "Thought: inspect sources.\nAction: available_data_sources\nAction Input: {}",
        "Final Answer: sources inspected",
    ])
    factory = SimpleNamespace(chat=lambda request: next(responses))
    processor = TimeSeriesConstructionProcessor(factory=factory)

    events = processor.process_user_request("create a stock time series for Apple in 2023")

    assert any(
        event.type == CallbackEventType.AGENT_COMPLETED
        and event.payload.get("agent") == "Orchestrator"
        for event in events
    )


def test_unavailable_tool_becomes_user_facing_error() -> None:
    responses = iter([
        "Thought: use an unavailable ticker.\nAction: historical_prices\n"
        'Action Input: {"symbol": "NOPE", "start_date": "2023-01-01", '
        '"end_date": "2023-12-31", "source": "yahoo"}',
    ])
    factory = SimpleNamespace(chat=lambda request: next(responses))
    processor = TimeSeriesConstructionProcessor(factory=factory)

    events = processor.process_user_request("create a stock time series for NOPE 2023")

    errors = [event for event in events if event.type == CallbackEventType.ERROR]
    assert errors
    assert errors[0].payload["recoverable"] is True
    assert errors[0].payload["user_action"]
