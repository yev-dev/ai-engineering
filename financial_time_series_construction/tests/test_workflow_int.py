"""Integration tests for the time series construction autogen workflow.

Tests cover:
- Instrument resolution (fuzzy matching, exact match, by name)
- Historical data loading from all sources
- Data quality metrics and gap filling
- Artifact generation (CSV, reports, visualizations)
- Full ReAct workflow with mocked LLM (delegation chain, pause/resume)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from financial_time_series_construction.agents_definition import CallbackEvent, CallbackEventType
from financial_time_series_construction.processor import TimeSeriesConstructionProcessor
from financial_time_series_construction.tools import (
    apply_gap_filling,
    available_data_sources,
    build_timeseries,
    check_data_quality,
    extract_date_range,
    extract_requested_date_range,
    generate_report,
    get_instrument_details,
    historical_prices,
    normalize_date_range,
    normalize_requested_dates,
    parse_flexible_date,
    recommend_gap_methods,
    visualize_timeseries,
)

# ---------------------------------------------------------------------------
# Fixtures – replicate the data files that tools.py depends on
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create mock CSV data files and patch ``tools.DATA_DIR``."""
    import financial_time_series_construction.tools as tools_module

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # -- instruments.csv ---------------------------------------------------
    instruments = pd.DataFrame(
        {
            "symbol": ["AAPL", "GOOGL", "MSFT"],
            "security_name": [
                "Apple Inc.",
                "Alphabet Inc.",
                "Microsoft Corporation",
            ],
            "sector": [
                "Information Technology",
                "Communication Services",
                "Information Technology",
            ],
            "sub_industry": [
                "Technology Hardware, Storage & Peripherals",
                "Interactive Media & Services",
                "Systems Software",
            ],
            "date_added": ["1982-11-30", "2014-04-03", "1994-06-01"],
        }
    )
    instruments.to_csv(data_dir / "instruments.csv", index=False)

    # -- source CSVs (wide-format, Date column + ticker columns) -----------
    dates = pd.bdate_range("2023-01-01", "2024-12-31")
    n = len(dates)

    # Yahoo – inject a few NaN gaps to exercise quality / gap-filling
    yahoo_prices: list[float | None] = [
        150.0 + i * 0.05 + (i % 7) * 0.5 for i in range(n)
    ]
    for idx in (5, 6, 7):
        yahoo_prices[idx] = None
    yahoo_df = pd.DataFrame(
        {"Date": dates.strftime("%Y-%m-%d"), "AAPL": yahoo_prices}
    )
    yahoo_df.to_csv(data_dir / "yahoo_stock_data.csv", index=False)

    # Bloomberg
    bloomberg_prices = [151.0 + i * 0.05 + (i % 5) * 0.3 for i in range(n)]
    bloomberg_df = pd.DataFrame(
        {"Date": dates.strftime("%Y-%m-%d"), "AAPL": bloomberg_prices}
    )
    bloomberg_df.to_csv(data_dir / "bloomberg_stock_data.csv", index=False)

    # Reuters
    reuters_prices = [149.0 + i * 0.05 + (i % 6) * 0.4 for i in range(n)]
    reuters_df = pd.DataFrame(
        {"Date": dates.strftime("%Y-%m-%d"), "AAPL": reuters_prices}
    )
    reuters_df.to_csv(data_dir / "reuters_stock_data.csv", index=False)

    monkeypatch.setattr(tools_module, "DATA_DIR", data_dir)
    return data_dir


@pytest.fixture
def mock_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect artifact output to a temporary directory."""
    import financial_time_series_construction.tools as tools_module

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr(tools_module, "OUTPUT_ROOT", output_dir)
    return output_dir


@pytest.fixture
def mock_processor() -> TimeSeriesConstructionProcessor:
    """Return a processor wired to a dummy LLM factory.

    The caller must set ``factory.chat_sequence`` (a list of strings) before
    calling ``process_user_request`` / ``process_user_response``.
    """

    class _SequenceFactory:
        def __init__(self) -> None:
            self.chat_sequence: list[str] = []

        def chat(self, request: Any) -> str:
            return self.chat_sequence.pop(0)

    factory = _SequenceFactory()
    return TimeSeriesConstructionProcessor(factory=factory)


# ---------------------------------------------------------------------------
# Tests – instrument resolution
# ---------------------------------------------------------------------------


class TestInstrumentResolution:
    """Verify that the fuzzy-matching logic resolves various query formats."""

    def test_apl_resolves_to_aapl(self, mock_data_dir: Path) -> None:
        """APL (typo) should fuzzy-match to AAPL."""
        result = get_instrument_details(query="APL")
        assert result["found"] is True
        assert result["symbol"] == "AAPL"

    def test_apl_resolves_via_symbol_arg(self, mock_data_dir: Path) -> None:
        """APL via symbol parameter should resolve."""
        result = get_instrument_details(symbol="APL")
        assert result["found"] is True
        assert result["symbol"] == "AAPL"

    def test_aapl_direct_lookup(self, mock_data_dir: Path) -> None:
        """Exact ticker lookup."""
        result = get_instrument_details(query="AAPL")
        assert result["found"] is True
        assert result["symbol"] == "AAPL"

    def test_apple_inc_full_name(self, mock_data_dir: Path) -> None:
        """Full security name lookup."""
        result = get_instrument_details(query="Apple Inc.")
        assert result["found"] is True
        assert result["symbol"] == "AAPL"

    def test_freeform_query_with_embedded_ticker(self, mock_data_dir: Path) -> None:
        """Free-form requests with embedded ticker should resolve the symbol."""
        result = get_instrument_details(
            query="build financial time series for apple with AAPL ticker start 2023-01-01 and 2024-01-01"
        )
        assert result["found"] is True
        assert result["symbol"] == "AAPL"

    def test_freeform_query_with_company_fragment(self, mock_data_dir: Path) -> None:
        """Company-name fragments in natural language should still resolve."""
        result = get_instrument_details(query="construct time series for apple for 2023")
        assert result["found"] is True
        assert result["symbol"] == "AAPL"

    def test_unknown_instrument_returns_suggestions(self, mock_data_dir: Path) -> None:
        """Unknown instrument should return suggestions."""
        result = get_instrument_details(query="NONEXISTENT")
        assert result["found"] is False
        assert "suggestions" in result

    def test_empty_query_returns_message(self, mock_data_dir: Path) -> None:
        """Empty query should return a helpful message."""
        result = get_instrument_details(query="")
        assert result["found"] is False
        assert "No instrument query" in result.get("message", "")


# ---------------------------------------------------------------------------
# Tests – historical data loading
# ---------------------------------------------------------------------------


class TestHistoricalData:
    """Fetch and inspect AAPL prices across the 2023–2024 window."""

    def test_available_sources(self, mock_data_dir: Path) -> None:
        """All three sources should be available."""
        sources = available_data_sources()
        assert "yahoo" in sources
        assert "bloomberg" in sources
        assert "reuters" in sources

    def test_fetch_yahoo_full_range(self, mock_data_dir: Path) -> None:
        """Yahoo data should return prices for the full range."""
        prices = historical_prices("AAPL", "2023-01-03", "2024-12-30", "yahoo")
        assert prices["symbol"] == "AAPL"
        assert prices["source"] == "yahoo"
        assert len(prices["dates"]) > 0
        assert len(prices["prices"]) > 0
        assert prices["dates"][0] >= "2023-01-03"
        assert prices["dates"][-1] <= "2024-12-30"

    def test_fetch_bloomberg(self, mock_data_dir: Path) -> None:
        """Bloomberg data should return prices."""
        prices = historical_prices("AAPL", "2023-06-01", "2023-06-30", "bloomberg")
        assert prices["symbol"] == "AAPL"
        assert prices["source"] == "bloomberg"
        assert len(prices["dates"]) > 0

    def test_fetch_reuters(self, mock_data_dir: Path) -> None:
        """Reuters data should return prices."""
        prices = historical_prices("AAPL", "2024-01-02", "2024-01-31", "reuters")
        assert prices["symbol"] == "AAPL"
        assert prices["source"] == "reuters"
        assert len(prices["dates"]) > 0

    def test_empty_date_range_raises(self, mock_data_dir: Path) -> None:
        """Empty date range should raise ValueError."""
        with pytest.raises(ValueError, match="No historical data is available"):
            historical_prices("AAPL", "2021-01-01", "2021-01-31", "yahoo")

    def test_unknown_ticker_raises(self, mock_data_dir: Path) -> None:
        """Unknown ticker should raise ValueError."""
        with pytest.raises(ValueError, match="is not available"):
            historical_prices("NOPE", "2023-01-01", "2023-12-31", "yahoo")

    def test_unsupported_source_raises(self, mock_data_dir: Path) -> None:
        """Unsupported source should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported source"):
            historical_prices("AAPL", "2023-01-01", "2023-12-31", "unknown_source")


class TestFlexibleDateParsing:
    """Verify support for word- and number-based date formats."""

    def test_parse_year_to_boundaries(self) -> None:
        assert parse_flexible_date("2023", "start").strftime("%Y-%m-%d") == "2023-01-01"
        assert parse_flexible_date("2023", "end").strftime("%Y-%m-%d") == "2023-12-31"

    def test_parse_month_word_range(self) -> None:
        start, end = normalize_date_range("January 2023", "March 2023")
        assert start == "2023-01-01"
        assert end == "2023-03-31"

    def test_parse_quarter_wording(self) -> None:
        start, end = normalize_date_range("Q1 2023", "Q2 2023")
        assert start == "2023-01-01"
        assert end == "2023-06-30"

    def test_extract_between_expression(self) -> None:
        extracted = extract_date_range("AAPL between January 2023 and December 2023")
        assert extracted == ("2023-01-01", "2023-12-31")

    def test_extract_start_end_date_expression(self) -> None:
        extracted = extract_date_range(
            "construct Apple time series for start date 2023-01-01 and end date 2024-01-01"
        )
        assert extracted == ("2023-01-01", "2024-01-01")

    def test_extract_requested_date_range_accepts_start_end_fields(self) -> None:
        result = extract_requested_date_range(start_date="2023-01-01", end_date="2024-01-01")
        assert result == {"start_date": "2023-01-01", "end_date": "2024-01-01"}

    def test_normalize_requested_dates_accepts_request_text(self) -> None:
        result = normalize_requested_dates(request="from January 2023 to January 2024")
        assert result == {"start_date": "2023-01-01", "end_date": "2024-01-31"}

    def test_historical_prices_accepts_word_dates(self, mock_data_dir: Path) -> None:
        prices = historical_prices("AAPL", "January 2023", "March 2023", "yahoo")
        assert prices["dates"][0].startswith("2023-01")
        assert prices["dates"][-1].startswith("2023-03") or prices["dates"][-1].startswith("2023-02")


# ---------------------------------------------------------------------------
# Tests – data quality and gap filling
# ---------------------------------------------------------------------------


class TestDataQuality:
    """Quality metrics detect injected gaps; gap filling repairs them."""

    def test_quality_detects_missing_values(self, mock_data_dir: Path) -> None:
        """Yahoo data has injected NaN gaps that should be detected."""
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "yahoo")
        quality = check_data_quality(prices["prices"], "yahoo", "AAPL")
        assert quality["missing_count"] > 0
        assert quality["completeness_pct"] < 100.0

    def test_quality_bloomberg_no_missing(self, mock_data_dir: Path) -> None:
        """Bloomberg data has no gaps."""
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "bloomberg")
        quality = check_data_quality(prices["prices"], "bloomberg", "AAPL")
        assert quality["missing_count"] == 0
        assert quality["completeness_pct"] == 100.0

    def test_linear_interpolation_fills_gaps(self, mock_data_dir: Path) -> None:
        """Linear interpolation should fill all NaN gaps."""
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "yahoo")
        filled = apply_gap_filling(prices, "linear_interpolation")
        assert filled["method"] == "linear_interpolation"
        assert filled["symbol"] == "AAPL"
        assert len(filled["dates"]) == len(prices["dates"])
        assert all(p is not None for p in filled["prices"])

    def test_forward_fill(self, mock_data_dir: Path) -> None:
        """Forward fill should fill gaps with previous values."""
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "yahoo")
        filled = apply_gap_filling(prices, "forward_fill")
        assert filled["method"] == "forward_fill"
        assert all(p is not None for p in filled["prices"])

    def test_backward_fill(self, mock_data_dir: Path) -> None:
        """Backward fill should fill gaps with next values."""
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "yahoo")
        filled = apply_gap_filling(prices, "backward_fill")
        assert filled["method"] == "backward_fill"
        assert all(p is not None for p in filled["prices"])

    def test_no_gap_method_preserves_nans(self, mock_data_dir: Path) -> None:
        """'none' method should preserve NaN values."""
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "yahoo")
        filled = apply_gap_filling(prices, "none")
        assert filled["method"] == "none"
        assert any(p is None for p in filled["prices"])

    def test_recommend_methods_with_gaps(self, mock_data_dir: Path) -> None:
        """When gaps exist, recommend interpolation methods."""
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "yahoo")
        quality = check_data_quality(prices["prices"], "yahoo", "AAPL")
        methods = recommend_gap_methods(quality, prices)
        assert "linear_interpolation" in methods
        assert "forward_fill" in methods
        assert "backward_fill" in methods

    def test_recommend_methods_no_gaps(self, mock_data_dir: Path) -> None:
        """When no gaps exist, recommend 'none'."""
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "bloomberg")
        quality = check_data_quality(prices["prices"], "bloomberg", "AAPL")
        methods = recommend_gap_methods(quality, prices)
        assert methods == ["none"]

    def test_unsupported_method_raises(self, mock_data_dir: Path) -> None:
        """Unsupported method should raise ValueError."""
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "yahoo")
        with pytest.raises(ValueError, match="Unsupported gap method"):
            apply_gap_filling(prices, "invalid_method")


# ---------------------------------------------------------------------------
# Tests – artifact generation
# ---------------------------------------------------------------------------


class TestArtifacts:
    """CSV reports, final series, and charts are written to the output dir."""

    def test_build_timeseries_csv(self, mock_data_dir: Path, mock_output_dir: Path) -> None:
        """Final time series CSV should be created with correct columns."""
        prices = historical_prices("AAPL", "2023-01-03", "2024-12-30", "yahoo")
        filled = apply_gap_filling(prices, "linear_interpolation")
        path = build_timeseries(filled, filename="AAPL_timeseries.csv", run_id="int_test")
        csv_path = Path(path)
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["date", "price"]
        assert len(df) == len(filled["dates"])

    def test_generate_report_csv(self, mock_data_dir: Path, mock_output_dir: Path) -> None:
        """Quality report CSV should be created with correct columns."""
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "yahoo")
        quality = check_data_quality(prices["prices"], "yahoo", "AAPL")
        path = generate_report(quality, filename="quality_report.csv", run_id="int_test")
        csv_path = Path(path)
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert "source" in df.columns
        assert df["source"].iloc[0] == "yahoo"

    def test_visualize_timeseries_png(self, mock_data_dir: Path, mock_output_dir: Path) -> None:
        """Visualization PNG should be created."""
        prices = historical_prices("AAPL", "2023-01-03", "2024-12-30", "yahoo")
        filled = apply_gap_filling(prices, "linear_interpolation")
        path = visualize_timeseries(filled, title="AAPL 2023-2024", run_id="int_test")
        png_path = Path(path)
        assert png_path.exists()
        assert png_path.suffix == ".png"

    def test_build_timeseries_with_run_id(self, mock_data_dir: Path, mock_output_dir: Path) -> None:
        """Run ID should create a subdirectory structure."""
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "yahoo")
        filled = apply_gap_filling(prices, "linear_interpolation")
        path = build_timeseries(filled, filename="test_series.csv", run_id="custom_run")
        csv_path = Path(path)
        assert csv_path.exists()
        assert "custom_run" in str(csv_path)


# ---------------------------------------------------------------------------
# Tests – full ReAct workflow with mocked LLM
# ---------------------------------------------------------------------------


class TestFullWorkflow:
    """Simulate the end-to-end ReAct loop with a dummy LLM factory."""

    def test_workflow_completes_for_apl(self, mock_data_dir: Path, mock_processor: TimeSeriesConstructionProcessor) -> None:
        """The delegation chain should route APL through all agents.

        Because the ReAct loop pauses at ``request_human_input``, we verify that
        the processor reaches the expected agent delegation steps and emits the
        correct events before pausing.
        """
        mock_processor.factory.chat_sequence = [
            # ReferenceDataAgent → resolve APL
            (
                "Thought: Look up APL in the instrument catalog.\n"
                "Action: get_instrument_details\n"
                "Action Input: {\"query\": \"APL\"}"
            ),
            # ReferenceDataAgent → delegate to MarketDataAgent
            (
                "Thought: APL resolved to AAPL. Delegate to MarketDataAgent.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"MarketDataAgent\", "
                "\"request\": \"load AAPL from 2023-01-01 to 2024-12-31 from all sources\"}"
            ),
            # MarketDataAgent → list sources
            (
                "Thought: Check available sources.\n"
                "Action: available_data_sources\n"
                "Action Input: {}"
            ),
            # MarketDataAgent → load yahoo
            (
                "Thought: Load yahoo data for AAPL.\n"
                "Action: historical_prices\n"
                "Action Input: {\"symbol\": \"AAPL\", "
                "\"start_date\": \"2023-01-03\", "
                "\"end_date\": \"2024-12-30\", "
                "\"source\": \"yahoo\"}"
            ),
            # MarketDataAgent → delegate to DataQualityAgent
            (
                "Thought: Data loaded. Delegate to DataQualityAgent.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"DataQualityAgent\", "
                "\"request\": \"check quality of AAPL from yahoo\"}"
            ),
            # DataQualityAgent → check quality
            (
                "Thought: Check data quality.\n"
                "Action: check_data_quality\n"
                "Action Input: {\"prices\": [150.0, 150.5, 151.0], "
                "\"source\": \"yahoo\", \"symbol\": \"AAPL\"}"
            ),
            # DataQualityAgent → delegate to ReportingAgent
            (
                "Thought: Quality checked. Delegate to ReportingAgent.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"ReportingAgent\", "
                "\"request\": \"present quality report for AAPL from yahoo\"}"
            ),
            # ReportingAgent → ask user to select source (pauses the loop)
            (
                "Thought: Present report and ask user to select source.\n"
                "Action: request_human_input\n"
                "Action Input: {\"prompt\": \"Select a data source for AAPL:\", "
                "\"options\": [\"yahoo\", \"bloomberg\", \"reuters\"]}"
            ),
        ]

        events = mock_processor.process_user_request(
            "create a time series for APL between 2023 and 2024"
        )

        # The workflow should have completed agents up to ReportingAgent
        completed = [e for e in events if e.type == CallbackEventType.AGENT_COMPLETED]
        agent_names = {e.payload.get("agent") for e in completed}
        assert "ReferenceDataAgent" in agent_names, (
            f"Expected ReferenceDataAgent in completed agents, got {agent_names}"
        )
        assert "MarketDataAgent" in agent_names
        assert "DataQualityAgent" in agent_names
        assert "ReportingAgent" not in agent_names  # paused before completion

        # Should have an AWAITING_USER_INPUT event from ReportingAgent
        awaiting = [e for e in events if e.type == CallbackEventType.AWAITING_USER_INPUT]
        assert len(awaiting) == 1
        assert awaiting[0].payload["agent"] == "ReportingAgent"

        # No error events should be emitted
        errors = [e for e in events if e.type == CallbackEventType.ERROR]
        assert not errors, f"Unexpected error events: {errors}"

    def test_workflow_resume_completes_reporting_agent(self, mock_data_dir: Path, mock_processor: TimeSeriesConstructionProcessor) -> None:
        """After the user selects a source, ``process_user_response`` resumes the
        ReportingAgent which completes with a Final Answer."""
        mock_processor.factory.chat_sequence = [
            # ReferenceDataAgent → resolve APL → AAPL
            (
                "Thought: Look up APL.\n"
                "Action: get_instrument_details\n"
                "Action Input: {\"query\": \"APL\"}"
            ),
            # ReferenceDataAgent → delegate to MarketDataAgent
            (
                "Thought: Resolved to AAPL. Delegate.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"MarketDataAgent\", "
                "\"request\": \"load AAPL from yahoo for 2023-2024\"}"
            ),
            # MarketDataAgent → load yahoo data
            (
                "Thought: Load yahoo data.\n"
                "Action: historical_prices\n"
                "Action Input: {\"symbol\": \"AAPL\", "
                "\"start_date\": \"2023-01-03\", "
                "\"end_date\": \"2024-12-30\", "
                "\"source\": \"yahoo\"}"
            ),
            # MarketDataAgent → delegate to DataQualityAgent
            (
                "Thought: Data loaded. Delegate.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"DataQualityAgent\", "
                "\"request\": \"check quality of AAPL from yahoo\"}"
            ),
            # DataQualityAgent → check quality
            (
                "Thought: Check quality.\n"
                "Action: check_data_quality\n"
                "Action Input: {\"prices\": [150.0, 150.5], "
                "\"source\": \"yahoo\", \"symbol\": \"AAPL\"}"
            ),
            # DataQualityAgent → delegate to ReportingAgent
            (
                "Thought: Checked. Delegate.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"ReportingAgent\", "
                "\"request\": \"present report\"}"
            ),
            # ReportingAgent → pause for source selection
            (
                "Thought: Ask user for source.\n"
                "Action: request_human_input\n"
                "Action Input: {\"prompt\": \"Select source:\", "
                "\"options\": [\"yahoo\", \"bloomberg\"]}"
            ),
            # ReportingAgent (resumed after user says "yahoo") → final answer
            "Final Answer: User selected yahoo as the data source for AAPL.",
        ]

        # Phase 1 – initial request pauses at ReportingAgent
        events = mock_processor.process_user_request(
            "create a time series for APL between 2023 and 2024"
        )
        awaiting = [e for e in events if e.type == CallbackEventType.AWAITING_USER_INPUT]
        assert len(awaiting) == 1
        assert awaiting[0].payload["agent"] == "ReportingAgent"

        # Phase 2 – resume with user's source selection, ReportingAgent completes
        events2 = mock_processor.process_user_response("yahoo")
        completed = [e for e in events2 if e.type == CallbackEventType.AGENT_COMPLETED]
        agent_names = {e.payload.get("agent") for e in completed}
        assert "ReportingAgent" in agent_names, (
            f"Expected ReportingAgent in completed agents, got {agent_names}"
        )
        errors = [e for e in events2 if e.type == CallbackEventType.ERROR]
        assert not errors, f"Unexpected error events after resume: {errors}"

    def test_workflow_recovers_from_apel_typo(self, mock_data_dir: Path, mock_processor: TimeSeriesConstructionProcessor) -> None:
        """A typo like APEL should still resolve via fuzzy matching."""
        mock_processor.factory.chat_sequence = [
            # Orchestrator → delegate
            (
                "Thought: Resolve the instrument.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"ReferenceDataAgent\", "
                "\"request\": \"create a time series for APEL between 2023 and 2024\"}"
            ),
            # ReferenceDataAgent → resolve (will fail, but we simulate recovery)
            (
                "Thought: APEL not found. Suggest AAPL.\n"
                "Action: get_instrument_details\n"
                "Action Input: {\"query\": \"APEL\"}"
            ),
            # ReferenceDataAgent → final answer with suggestion
            "Final Answer: Instrument APEL was not found. Did you mean AAPL?",
        ]

        events = mock_processor.process_user_request(
            "create a time series for APEL between 2023 and 2024"
        )

        # Should get an error about the instrument not being found
        errors = [e for e in events if e.type == CallbackEventType.ERROR]
        assert errors, "Expected an error for unresolved instrument APEL"

    def test_workflow_rejects_environment_command(self, mock_processor: TimeSeriesConstructionProcessor) -> None:
        """Environment commands should be rejected with a clarification prompt."""
        events = mock_processor.process_user_request("conda activate myenv")
        awaiting = [e for e in events if e.type == CallbackEventType.AWAITING_USER_INPUT]
        assert len(awaiting) == 1
        assert "conda environment" in awaiting[0].payload.get("prompt", "").lower()

    def test_workflow_rejects_non_financial_request(self, mock_processor: TimeSeriesConstructionProcessor) -> None:
        """Non-financial requests should trigger a clarification prompt."""
        events = mock_processor.process_user_request("what is the weather today?")
        awaiting = [e for e in events if e.type == CallbackEventType.AWAITING_USER_INPUT]
        assert len(awaiting) == 1
        assert "financial time series" in awaiting[0].payload.get("prompt", "").lower()

    def test_workflow_cancel_during_pause(self, mock_data_dir: Path, mock_processor: TimeSeriesConstructionProcessor) -> None:
        """User cancellation during pause should emit an error event."""
        mock_processor.factory.chat_sequence = [
            (
                "Thought: Delegate.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"ReferenceDataAgent\", "
                "\"request\": \"load AAPL\"}"
            ),
            (
                "Thought: Ask user.\n"
                "Action: request_human_input\n"
                "Action Input: {\"prompt\": \"Select source:\", "
                "\"options\": [\"yahoo\"]}"
            ),
        ]

        events = mock_processor.process_user_request("load AAPL")
        awaiting = [e for e in events if e.type == CallbackEventType.AWAITING_USER_INPUT]
        assert len(awaiting) == 1

        # Cancel the workflow
        events2 = mock_processor.process_user_response("exit")
        errors = [e for e in events2 if e.type == CallbackEventType.ERROR]
        assert errors, "Expected error event on cancellation"
        assert "cancelled" in errors[0].payload.get("message", "").lower()

    def test_workflow_delegation_events_emitted(self, mock_data_dir: Path, mock_processor: TimeSeriesConstructionProcessor) -> None:
        """Delegation between agents should emit DELEGATED events."""
        mock_processor.factory.chat_sequence = [
            (
                "Thought: Delegate to ReferenceDataAgent.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"ReferenceDataAgent\", "
                "\"request\": \"resolve AAPL stock\"}"
            ),
            "Final Answer: AAPL is Apple Inc.",
        ]

        events = mock_processor.process_user_request("resolve AAPL stock")
        delegated = [e for e in events if e.type == CallbackEventType.DELEGATED]
        assert len(delegated) >= 1
        assert delegated[0].payload.get("from_agent") == "Orchestrator"
        assert delegated[0].payload.get("to_agent") == "ReferenceDataAgent"

    def test_orchestrator_clarification_with_dates_progresses(
        self,
        mock_data_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """When user provides dates in follow-up, workflow should progress instead of looping."""
        mock_processor.factory.chat_sequence = [
            (
                "Thought: Missing required inputs.\n"
                "Action: request_human_input\n"
                "Action Input: {\"prompt\": \"Please provide ticker and date range\"}"
            ),
            (
                "Thought: Resolve the provided asset.\n"
                "Action: get_instrument_details\n"
                "Action Input: {\"query\": \"AAPL\"}"
            ),
            "Final Answer: Instrument resolved. Proceeding with AAPL from 2023-01-01 to 2023-12-31.",
        ]

        initial_events = mock_processor.process_user_request("Build a time series")
        pause_events = [e for e in initial_events if e.type == CallbackEventType.AWAITING_USER_INPUT]
        assert pause_events, "Expected initial clarification pause"

        resumed_events = mock_processor.process_user_response(
            "AAPL between January 2023 and December 2023"
        )
        delegated = [e for e in resumed_events if e.type == CallbackEventType.DELEGATED]
        assert delegated, "Expected orchestrator to delegate after clarification"
        assert delegated[0].payload.get("to_agent") == "ReferenceDataAgent"
        second_pause = [e for e in resumed_events if e.type == CallbackEventType.AWAITING_USER_INPUT]
        assert not second_pause, "Did not expect orchestrator to ask for dates again"

    def test_initial_request_with_word_dates_bypasses_orchestrator_pause(
        self,
        mock_data_dir: Path,
    ) -> None:
        """Request containing instrument + date range should delegate immediately."""

        class _RecordingFactory:
            def __init__(self) -> None:
                self.seen_system_prompts: list[str] = []

            def chat(self, request: Any) -> str:
                self.seen_system_prompts.append(request.system_prompt)
                return "Final Answer: Reference data accepted."

        factory = _RecordingFactory()
        processor = TimeSeriesConstructionProcessor(factory=factory)

        events = processor.process_user_request("build AAPL from january 2023 to january 2024")
        delegated = [e for e in events if e.type == CallbackEventType.DELEGATED]
        pauses = [e for e in events if e.type == CallbackEventType.AWAITING_USER_INPUT]

        assert delegated, "Expected direct delegation from Orchestrator"
        assert delegated[0].payload.get("to_agent") == "ReferenceDataAgent"
        assert not pauses, "Did not expect initial Orchestrator clarification pause"
        assert factory.seen_system_prompts, "Expected downstream agent LLM call"
        assert "Resolve the instrument" in factory.seen_system_prompts[0]

    def test_initial_request_with_start_end_labels_bypasses_orchestrator_pause(
        self,
        mock_data_dir: Path,
    ) -> None:
        """Requests with 'start date ... end date ...' should also bypass initial loop."""

        class _RecordingFactory:
            def __init__(self) -> None:
                self.seen_system_prompts: list[str] = []

            def chat(self, request: Any) -> str:
                self.seen_system_prompts.append(request.system_prompt)
                return "Final Answer: Reference data accepted."

        factory = _RecordingFactory()
        processor = TimeSeriesConstructionProcessor(factory=factory)

        events = processor.process_user_request(
            "construct Apple time series for start date 2023-01-01 and end date 2024-01-01"
        )
        delegated = [e for e in events if e.type == CallbackEventType.DELEGATED]
        pauses = [e for e in events if e.type == CallbackEventType.AWAITING_USER_INPUT]

        assert delegated, "Expected direct delegation from Orchestrator"
        assert delegated[0].payload.get("to_agent") == "ReferenceDataAgent"
        assert not pauses, "Did not expect initial Orchestrator clarification pause"

    def test_explicit_agent_response_delegates_to_market_data(
        self,
        mock_data_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """If user replies with MarketDataAgent, workflow should delegate directly."""
        mock_processor.factory.chat_sequence = [
            (
                "Thought: Need user decision.\n"
                "Action: request_human_input\n"
                "Action Input: {\"prompt\": \"Confirm target agent\", \"options\": [\"Confirm\", \"Cancel\"]}"
            ),
            "Final Answer: Market data flow started.",
        ]

        first_events = mock_processor.process_user_request("build AAPL stock")
        pauses = [event for event in first_events if event.type == CallbackEventType.AWAITING_USER_INPUT]
        assert pauses, "Expected pause before explicit target response"

        resumed_events = mock_processor.process_user_response("MarketDataAgent")
        delegated = [event for event in resumed_events if event.type == CallbackEventType.DELEGATED]
        assert delegated, "Expected direct delegation after explicit agent selection"
        assert delegated[0].payload.get("to_agent") == "MarketDataAgent"

    def test_reference_reask_is_bypassed_after_instrument_resolution(
        self,
        mock_data_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """ReferenceDataAgent should not loop on request_human_input after resolving instrument."""
        mock_processor.factory.chat_sequence = [
            (
                "Thought: Resolve instrument.\n"
                "Action: get_instrument_details\n"
                "Action Input: {\"query\": \"AAPL\"}"
            ),
            (
                "Thought: Ask to confirm MarketDataAgent availability.\n"
                "Action: request_human_input\n"
                "Action Input: {\"prompt\": \"Please confirm that you want to use MarketDataAgent despite its unavailability.\", "
                "\"options\": [\"Confirm\", \"Cancel\"]}"
            ),
            "Final Answer: Market data stage reached.",
        ]

        events = mock_processor.process_user_request(
            "construct Apple time series for start date 2023-01-01 and end date 2024-01-01"
        )

        delegated = [event for event in events if event.type == CallbackEventType.DELEGATED]
        assert delegated, "Expected deterministic delegation chain"
        delegated_targets = {event.payload.get("to_agent") for event in delegated}
        assert "MarketDataAgent" in delegated_targets
        pauses = [event for event in events if event.type == CallbackEventType.AWAITING_USER_INPUT]
        assert not pauses, "Expected bypass of ReferenceDataAgent confirmation loop"

    def test_reference_agent_recovers_from_bool_symbol_argument(
        self,
        mock_data_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """Malformed symbol=true payload should not crash tool validation.

        After ReferenceDataAgent resolves the instrument, the processor
        auto-delegates to MarketDataAgent, so the test provides a MarketDataAgent
        response to complete the continuation.
        """
        mock_processor.factory.chat_sequence = [
            (
                "Thought: Resolve instrument using available context.\n"
                "Action: get_instrument_details\n"
                "Action Input: {\"symbol\": true}"
            ),
            # ReferenceDataAgent completes → auto-continuation delegates to MarketDataAgent
            "Final Answer: Instrument resolved from context.",
            # MarketDataAgent response for the auto-continuation
            "Final Answer: Market data retrieval started for AAPL.",
        ]

        events = mock_processor.process_user_request("Build AAPL from 2023-01-01 to 2023-12-31")
        errors = [event for event in events if event.type == CallbackEventType.ERROR]
        assert not errors, f"Unexpected validation error events: {errors}"

        completed = [event for event in events if event.type == CallbackEventType.AGENT_COMPLETED]
        completed_agents = {event.payload.get("agent") for event in completed}
        assert "ReferenceDataAgent" in completed_agents
        assert "MarketDataAgent" in completed_agents, (
            "Expected MarketDataAgent to be auto-delegated after ReferenceDataAgent completion"
        )

    def test_market_data_source_selection_final_is_bypassed(
        self,
        mock_data_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """MarketDataAgent should not stop at 'select one source' final answers."""
        mock_processor.factory.chat_sequence = [
            (
                "Thought: Resolve instrument first.\n"
                "Action: get_instrument_details\n"
                "Action Input: {\"query\": \"AAPL\"}"
            ),
            (
                "Thought: Delegate to market data collection.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"MarketDataAgent\", "
                "\"request\": \"load AAPL from 2023-01-01 to 2023-12-31\"}"
            ),
            (
                "Final Answer: The available data sources for retrieving historical prices "
                "are Yahoo, Bloomberg, and Reuters. Please select one of these options to "
                "continue with the request."
            ),
            (
                "Thought: List available sources before loading data.\n"
                "Action: available_data_sources\n"
                "Action Input: {}"
            ),
            (
                "Thought: Load yahoo data.\n"
                "Action: historical_prices\n"
                "Action Input: {\"symbol\": \"AAPL\", \"start_date\": \"2023-01-01\", "
                "\"end_date\": \"2023-12-31\", \"source\": \"yahoo\"}"
            ),
            (
                "Thought: Load bloomberg data.\n"
                "Action: historical_prices\n"
                "Action Input: {\"symbol\": \"AAPL\", \"start_date\": \"2023-01-01\", "
                "\"end_date\": \"2023-12-31\", \"source\": \"bloomberg\"}"
            ),
            (
                "Thought: Load reuters data.\n"
                "Action: historical_prices\n"
                "Action Input: {\"symbol\": \"AAPL\", \"start_date\": \"2023-01-01\", "
                "\"end_date\": \"2023-12-31\", \"source\": \"reuters\"}"
            ),
            (
                "Thought: Delegate to data quality with all sources collected.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"DataQualityAgent\", "
                "\"request\": \"check quality for AAPL across yahoo, bloomberg, and reuters\"}"
            ),
            "Final Answer: Data quality stage started for all sources.",
        ]

        events = mock_processor.process_user_request(
            "Build Time Series for Apple (AAPL) from 2023-01-01 to 2023-12-31"
        )

        pauses = [event for event in events if event.type == CallbackEventType.AWAITING_USER_INPUT]
        assert not pauses, "Did not expect a source-selection pause from MarketDataAgent"

        final_answers = [
            str(event.payload.get("result", {}).get("final_answer", ""))
            for event in events
            if event.type == CallbackEventType.AGENT_COMPLETED
        ]
        assert all("please select one" not in answer.casefold() for answer in final_answers)

        delegated_targets = [
            event.payload.get("to_agent")
            for event in events
            if event.type == CallbackEventType.DELEGATED
        ]
        assert "DataQualityAgent" in delegated_targets


# ---------------------------------------------------------------------------
# Tests – handler and callback processor
# ---------------------------------------------------------------------------


class TestHandler:
    """Verify the TimeSeriesConstructionHandler works correctly."""

    def test_handler_emit_and_poll(self) -> None:
        """Events should be emitted and polled in FIFO order."""
        from financial_time_series_construction.handler import TimeSeriesConstructionHandler

        handler = TimeSeriesConstructionHandler(session_id="test")
        handler.emit(CallbackEvent(CallbackEventType.USER_REQUEST, {"request": "test"}))
        handler.emit(CallbackEvent(CallbackEventType.AGENT_COMPLETED, {"agent": "TestAgent"}))

        assert handler.has_events()
        event1 = handler.poll()
        assert event1 is not None
        assert event1.type == CallbackEventType.USER_REQUEST

        event2 = handler.poll()
        assert event2 is not None
        assert event2.type == CallbackEventType.AGENT_COMPLETED

        assert not handler.has_events()

    def test_handler_reset(self) -> None:
        """Reset should clear all state."""
        from financial_time_series_construction.handler import TimeSeriesConstructionHandler

        handler = TimeSeriesConstructionHandler(session_id="test")
        handler.emit(CallbackEvent(CallbackEventType.USER_REQUEST, {"request": "test"}))
        handler.waiting_for_input = True
        handler.paused_state = {"agent": "Test"}
        handler.current_agent = "TestAgent"
        handler.add_to_trace("test trace")

        handler.reset()
        assert not handler.has_events()
        assert not handler.waiting_for_input
        assert handler.paused_state is None
        assert handler.current_agent is None
        assert handler.get_trace() == ""

    def test_handler_trace(self) -> None:
        """Trace entries should be accumulated."""
        from financial_time_series_construction.handler import TimeSeriesConstructionHandler

        handler = TimeSeriesConstructionHandler(session_id="test")
        handler.add_to_trace("line1")
        handler.add_to_trace("line2")
        assert handler.get_trace() == "line1\nline2"

    def test_callback_processor(self) -> None:
        """CallbackProcessor should dispatch events to multiple handlers."""
        from financial_time_series_construction.handler import (
            CallbackProcessor,
            TimeSeriesConstructionHandler,
        )

        handler1 = TimeSeriesConstructionHandler(session_id="h1")
        handler2 = TimeSeriesConstructionHandler(session_id="h2")
        processor = CallbackProcessor([handler1])
        processor.add_handler(handler2)

        event = CallbackEvent(CallbackEventType.USER_REQUEST, {"request": "test"})
        processor.on_event(event)

        assert handler1.has_events()
        assert handler2.has_events()


class TestFrontToBackWorkflow:
    """Front-to-back workflow tests with multi-step human-in-the-loop checkpoints."""

    def test_full_workflow_with_two_human_checkpoints(
        self,
        mock_data_dir: Path,
        mock_output_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """Run orchestrator -> specialists -> source selection -> gap filling -> final output."""
        mock_processor.factory.chat_sequence = [
            (
                "Thought: Route this request to reference resolution.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"ReferenceDataAgent\", "
                "\"request\": \"build AAPL from 2023-01-03 to 2023-01-31\"}"
            ),
            (
                "Thought: Resolve instrument details first.\n"
                "Action: get_instrument_details\n"
                "Action Input: {\"query\": \"AAPL\"}"
            ),
            (
                "Thought: Delegate to market data collection.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"MarketDataAgent\", "
                "\"request\": \"load AAPL from 2023-01-03 to 2023-01-31\"}"
            ),
            (
                "Thought: Discover all available sources.\n"
                "Action: available_data_sources\n"
                "Action Input: {}"
            ),
            (
                "Thought: Pull yahoo prices for the requested range.\n"
                "Action: historical_prices\n"
                "Action Input: {\"symbol\": \"AAPL\", \"start_date\": \"2023-01-03\", "
                "\"end_date\": \"2023-01-31\", \"source\": \"yahoo\"}"
            ),
            (
                "Thought: Pass prices to data quality agent.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"DataQualityAgent\", "
                "\"request\": \"run quality checks for yahoo AAPL\"}"
            ),
            (
                "Thought: Compute quality metrics.\n"
                "Action: check_data_quality\n"
                "Action Input: {\"prices\": [150.0, null, 151.0, 151.4], "
                "\"source\": \"yahoo\", \"symbol\": \"AAPL\"}"
            ),
            (
                "Thought: Ask reporting to present summary and collect source choice.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"ReportingAgent\", "
                "\"request\": \"present source quality summary for AAPL\"}"
            ),
            (
                "Thought: Pause for user source selection.\n"
                "Action: request_human_input\n"
                "Action Input: {\"prompt\": \"Choose preferred source for AAPL\", "
                "\"options\": [\"yahoo\", \"bloomberg\", \"reuters\"]}"
            ),
            (
                "Thought: User selected source, continue with gap filling.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"GapFillingAgent\", "
                "\"request\": \"apply gap filling to yahoo AAPL\"}"
            ),
            (
                "Thought: Recommend methods based on missing values.\n"
                "Action: recommend_gap_methods\n"
                "Action Input: {\"quality_report\": {\"missing_count\": 2}, "
                "\"prices\": {\"symbol\": \"AAPL\", \"prices\": [150.0, null, 151.0]}}"
            ),
            (
                "Thought: Ask user to choose a gap filling method.\n"
                "Action: request_human_input\n"
                "Action Input: {\"prompt\": \"Select gap filling method\", "
                "\"options\": [\"linear_interpolation\", \"forward_fill\", \"backward_fill\"]}"
            ),
            (
                "Thought: Apply chosen method before constructing final output.\n"
                "Action: apply_gap_filling\n"
                "Action Input: {\"prices\": {\"symbol\": \"AAPL\", \"dates\": [\"2023-01-03\", \"2023-01-04\", \"2023-01-05\"], "
                "\"prices\": [150.0, null, 151.0]}, \"method\": \"linear_interpolation\"}"
            ),
            (
                "Thought: Delegate to final series construction.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"TimeSeriesConstructionAgent\", "
                "\"request\": \"build final AAPL continuous series\"}"
            ),
            (
                "Thought: Persist final series artifact.\n"
                "Action: build_timeseries\n"
                "Action Input: {\"series\": {\"symbol\": \"AAPL\", "
                "\"dates\": [\"2023-01-03\", \"2023-01-04\", \"2023-01-05\"], "
                "\"prices\": [150.0, 150.5, 151.0]}, \"filename\": \"final_timeseries.csv\", "
                "\"run_id\": \"it_full_flow\"}"
            ),
            (
                "Thought: Finish and summarize result.\n"
                "Final Answer: Continuous AAPL series generated at it_full_flow/final_timeseries.csv."
            ),
        ]

        first_pass_events = mock_processor.process_user_request(
            "Build AAPL from 2023-01-03 to 2023-01-31 and help me fill data gaps."
        )
        first_pause = [e for e in first_pass_events if e.type == CallbackEventType.AWAITING_USER_INPUT]
        assert first_pause, "Expected source-selection pause from ReportingAgent"
        assert first_pause[-1].payload["agent"] == "ReportingAgent"

        second_pass_events = mock_processor.process_user_response("yahoo")
        second_pause = [e for e in second_pass_events if e.type == CallbackEventType.AWAITING_USER_INPUT]
        assert second_pause, "Expected gap-method pause from GapFillingAgent"
        assert second_pause[-1].payload["agent"] == "GapFillingAgent"

        final_pass_events = mock_processor.process_user_response("linear_interpolation")
        completed = [e for e in final_pass_events if e.type == CallbackEventType.AGENT_COMPLETED]
        completed_agents = {e.payload.get("agent") for e in completed}
        assert "TimeSeriesConstructionAgent" in completed_agents

        errors = [e for e in final_pass_events if e.type == CallbackEventType.ERROR]
        assert not errors, f"Unexpected errors in final pass: {errors}"

        output_file = mock_output_dir / "it_full_flow" / "final_timeseries.csv"
        assert output_file.exists(), "Expected final output file to be persisted"

    def test_user_exits_after_source_comparison(
        self,
        mock_data_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """User can stop the workflow at first checkpoint after quality summary."""
        mock_processor.factory.chat_sequence = [
            (
                "Thought: Route request.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"ReferenceDataAgent\", "
                "\"request\": \"build AAPL from 2023-01-03 to 2023-01-31\"}"
            ),
            (
                "Thought: Ask user whether to continue after comparison.\n"
                "Action: request_human_input\n"
                "Action Input: {\"prompt\": \"Select source or exit\", "
                "\"options\": [\"yahoo\", \"bloomberg\", \"reuters\", \"exit\"]}"
            ),
        ]

        events = mock_processor.process_user_request("build AAPL for Jan 2023")
        pauses = [e for e in events if e.type == CallbackEventType.AWAITING_USER_INPUT]
        assert pauses, "Expected pause at source-selection checkpoint"

        exit_events = mock_processor.process_user_response("exit")
        exit_errors = [e for e in exit_events if e.type == CallbackEventType.ERROR]
        assert exit_errors, "Expected cancellation event when user exits"
        assert "cancelled" in exit_errors[0].payload.get("message", "").lower()