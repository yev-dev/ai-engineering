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


class TestSymbolRecoveryHelpers:
    """Processor recovery helpers should prefer real tickers and avoid false positives."""

    def test_parenthesized_ticker_is_recovered(self) -> None:
        """Parenthesized tickers like Apple (AAPL) should recover the ticker symbol."""
        text = "Historical prices for Apple Inc. (AAPL) from 2023-01-01 to 2024-01-01"
        symbol = TimeSeriesConstructionProcessor._extract_symbol_candidate_from_text(text)
        assert symbol == "AAPL"

    def test_one_letter_uppercase_token_is_ignored(self) -> None:
        """Single-letter uppercase tokens should not be treated as ticker symbols."""
        text = "Historical summary: I think the data looks complete."
        symbol = TimeSeriesConstructionProcessor._extract_symbol_candidate_from_text(text)
        assert symbol is None


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

    def test_empty_date_range_uses_closest_available_dates(self, mock_data_dir: Path) -> None:
        """Out-of-range requests should snap to closest available dates."""
        prices = historical_prices("AAPL", "2021-01-01", "2021-01-31", "yahoo")
        assert prices["symbol"] == "AAPL"
        assert prices["source"] == "yahoo"
        assert len(prices["dates"]) > 0

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

    def test_quality_includes_available_and_date_range(self, mock_data_dir: Path) -> None:
        """Quality metrics should include available record count and observed date range."""
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "yahoo")
        quality = check_data_quality(data=prices)
        assert quality["available_record_count"] == quality["total_values"] - quality["missing_count"]
        assert quality["min_date"] is not None
        assert quality["max_date"] is not None

    def test_linear_interpolation_fills_gaps(self, mock_data_dir: Path) -> None:
        """Linear interpolation should fill all NaN gaps."""
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "yahoo")
        filled = apply_gap_filling(prices, "linear_interpolation")
        assert filled["method"] == "linear_interpolation"
        assert filled["symbol"] == "AAPL"
        assert filled["source"] == "yahoo"
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
        assert list(df.columns) == ["date", "price", "source", "gap_filling_method"]
        assert df["source"].iloc[0] == "yahoo"
        assert df["gap_filling_method"].iloc[0] == "linear_interpolation"
        assert len(df) == len(filled["dates"])

    def test_build_timeseries_prefers_filled_series_payload(self, mock_output_dir: Path) -> None:
        """build_timeseries should persist the full filled population when both raw and filled fields exist."""
        series = {
            "symbol": "AAPL",
            "source": "yahoo",
            "method": "linear_interpolation",
            "dates": ["2023-01-03", "2023-01-04"],
            "prices": [150.0, 151.0],
            "original_dates": ["2023-01-03", "2023-01-04"],
            "original_prices": [150.0, None],
            "filled_dates": ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
            "filled_prices": [150.0, 150.5, 151.0, 151.5],
        }
        path = build_timeseries(series, filename="filled_population.csv", run_id="int_test")
        csv_path = Path(path)
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 4
        assert list(df["price"]) == [150.0, 150.5, 151.0, 151.5]

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

    def test_visualize_timeseries_before_after_payload(self, mock_data_dir: Path, mock_output_dir: Path) -> None:
        """Visualization should accept before/after payloads and persist a chart."""
        prices = historical_prices("AAPL", "2023-01-03", "2024-12-30", "yahoo")
        filled = apply_gap_filling(prices, "linear_interpolation")
        path = visualize_timeseries(filled, title="AAPL before and after gap filling", run_id="int_test")
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

        data_quality_events = [
            event for event in completed
            if event.payload.get("agent") == "DataQualityAgent"
        ]
        assert data_quality_events, "Expected DataQualityAgent completion event"
        quality_result = data_quality_events[-1].payload.get("result", {})
        quality_report = quality_result.get("data_quality_report")
        assert isinstance(quality_report, dict)
        assert quality_report.get("report_type") == "data_quality_summary"
        assert isinstance(quality_report.get("rows"), list)
        assert quality_report.get("rows")[0]["source"] == "yahoo"
        assert quality_report.get("rows")[0].get("available_record_count") is not None
        assert quality_report.get("rows")[0].get("min_date") is not None
        assert quality_report.get("rows")[0].get("max_date") is not None
        assert "summary" in quality_report
        assert quality_report["summary"].get("total_available_records") is not None
        assert quality_report["summary"].get("min_date") is not None
        assert quality_report["summary"].get("max_date") is not None

        # Should have an AWAITING_USER_INPUT event from ReportingAgent
        awaiting = [e for e in events if e.type == CallbackEventType.AWAITING_USER_INPUT]
        assert len(awaiting) == 1
        assert awaiting[0].payload["agent"] == "ReportingAgent"

        # No error events should be emitted
        errors = [e for e in events if e.type == CallbackEventType.ERROR]
        assert not errors, f"Unexpected error events: {errors}"

    def test_market_data_loads_all_sources_after_source_listing(
        self,
        mock_data_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """MarketDataAgent should not need one LLM turn per source once the source list is known."""
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
            # MarketDataAgent → discover sources only once
            (
                "Thought: Check available sources.\n"
                "Action: available_data_sources\n"
                "Action Input: {}"
            ),
            # DataQualityAgent → final answer after deterministic source loading
            "Final Answer: Quality reviewed for all market sources.",
            # ReportingAgent → pause for source selection
            (
                "Thought: Present the comparison and ask the user to choose a source.\n"
                "Action: request_human_input\n"
                "Action Input: {\"prompt\": \"Select a data source for AAPL:\", "
                "\"options\": [\"yahoo\", \"bloomberg\", \"reuters\"]}"
            ),
            # Downstream reporting/gap-filling turns may continue depending on the model
            "Final Answer: User selected yahoo as the data source for AAPL.",
            (
                "Thought: Ask for the gap-filling method.\n"
                "Action: request_human_input\n"
                "Action Input: {\"prompt\": \"Select gap method:\", "
                "\"options\": [\"linear_interpolation\", \"forward_fill\", \"backward_fill\", \"none\"]}"
            ),
        ]

        events = mock_processor.process_user_request(
            "create a time series for APL between 2023 and 2024"
        )

        completed = [e for e in events if e.type == CallbackEventType.AGENT_COMPLETED]
        market_events = [e for e in completed if e.payload.get("agent") == "MarketDataAgent"]
        assert market_events, "Expected MarketDataAgent completion event"

        market_result = market_events[-1].payload.get("result", {})
        assert sorted(market_result.get("loaded_sources", [])) == ["bloomberg", "reuters", "yahoo"]

        data_quality_events = [
            event for event in completed
            if event.payload.get("agent") == "DataQualityAgent"
        ]
        assert data_quality_events, "Expected DataQualityAgent completion event"
        quality_report = data_quality_events[-1].payload.get("result", {}).get("data_quality_report", {})
        assert quality_report.get("summary", {}).get("source_count") == 3

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
            # GapFillingAgent → narrative final answer triggers forced method-selection pause
            "Final Answer: Gap filling can now proceed once a method is selected.",
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
        """Environment commands should be rejected with an explicit error."""
        events = mock_processor.process_user_request("conda activate myenv")
        errors = [e for e in events if e.type == CallbackEventType.ERROR]
        assert len(errors) == 1
        assert "conda environment" in errors[0].payload.get("message", "").lower()

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
        assert all(event.payload.get("agent") != "Orchestrator" for event in pauses), (
            "Did not expect initial Orchestrator clarification pause"
        )
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
        assert all(event.payload.get("agent") != "Orchestrator" for event in pauses), (
            "Did not expect initial Orchestrator clarification pause"
        )


    def test_construction_truncated_series_payload_is_recovered(
        self,
        mock_output_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """A truncated build_timeseries payload should be replaced with the full filled series from context."""
        mock_processor.factory.chat_sequence = [
            (
                "Thought: Build the final artifact now.\n"
                "Action: build_timeseries\n"
                "Action Input: {\"series\": {\"symbol\": \"AAPL\", "
                "\"dates\": [\"2023-01-03\", \"2023-01-04\"], "
                "\"prices\": [150.0, 150.5]}, \"filename\": \"final_timeseries.csv\", "
                "\"run_id\": \"it_construction_truncated\"}"
            ),
            (
                "Thought: Summarize final result for the user.\n"
                "Final Answer: Final summary complete with constructed CSV and visualization artifacts."
            ),
        ]

        mock_processor.handler.paused_state = {
            "agent": "TimeSeriesConstructionAgent",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Build and persist the final time series for AAPL using linear_interpolation gap-filling. "
                        "Filled data: {\"symbol\": \"AAPL\", \"method\": \"linear_interpolation\", "
                        "\"dates\": [\"2023-01-03\", \"2023-01-04\", \"2023-01-05\", \"2023-01-06\"], "
                        "\"prices\": [150.0, 150.5, 151.0, 151.5]}. "
                        "Original request: Build AAPL from 2023-01-03 to 2023-01-31"
                    ),
                }
            ],
            "iteration": 0,
        }

        final_events = mock_processor.process_user_response("continue")

        delegated_edges = {
            (e.payload.get("from_agent"), e.payload.get("to_agent"))
            for e in final_events
            if e.type == CallbackEventType.DELEGATED
        }
        assert ("TimeSeriesConstructionAgent", "ReportingAgent") in delegated_edges

        output_file = mock_output_dir / "it_construction_truncated" / "final_timeseries.csv"
        assert output_file.exists(), "Expected final output file to be persisted"
        df = pd.read_csv(output_file)
        assert len(df) == 4
        assert list(df["price"]) == [150.0, 150.5, 151.0, 151.5]
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
            # DataQualityAgent response for the next auto-continuation
            "Final Answer: Data quality checks completed for AAPL across all sources.",
            # ReportingAgent response then pauses for source selection (no explicit choice)
            "Final Answer: Quality comparison prepared. Please select one source to continue.",
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

    def test_reference_final_answer_without_tool_still_continues_to_market(
        self,
        mock_data_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """ReferenceDataAgent final answers should not terminate the workflow early.

        If the model returns a final resolution sentence (without any tool call),
        the processor should still continue to MarketDataAgent.
        """
        mock_processor.factory.chat_sequence = [
            "Final Answer: Instrument resolved successfully. Symbol: AAPL.",
            "Final Answer: Market data retrieval started for AAPL.",
        ]

        events = mock_processor.process_user_request("Build AAPL from 2023-01-01 to 2023-12-31")

        delegated_targets = [
            event.payload.get("to_agent")
            for event in events
            if event.type == CallbackEventType.DELEGATED
        ]
        assert "MarketDataAgent" in delegated_targets

        completed_agents = [
            event.payload.get("agent")
            for event in events
            if event.type == CallbackEventType.AGENT_COMPLETED
        ]
        assert "ReferenceDataAgent" in completed_agents
        assert "MarketDataAgent" in completed_agents

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

    def test_market_final_answer_without_tool_still_continues_to_quality(
        self,
        mock_data_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """MarketDataAgent final answers should not terminate workflow early.

        If MarketDataAgent returns a narrative final answer without explicit
        historical_prices tool outputs, processor should still continue to
        DataQualityAgent using deterministic fallback routing.
        """
        mock_processor.factory.chat_sequence = [
            (
                "Thought: Resolve instrument.\n"
                "Action: get_instrument_details\n"
                "Action Input: {\"query\": \"AAPL\"}"
            ),
            "Final Answer: Instrument resolved successfully. Symbol: AAPL.",
            (
                "Final Answer: Historical prices for Apple Inc. (AAPL) from 2023-01-01 "
                "to 2024-01-01 have been retrieved successfully."
            ),
            "Final Answer: Data quality stage started.",
        ]

        events = mock_processor.process_user_request("AAPL from 2023-01-01 to 2024-01-01")

        delegated_targets = [
            event.payload.get("to_agent")
            for event in events
            if event.type == CallbackEventType.DELEGATED
        ]
        assert "MarketDataAgent" in delegated_targets
        assert "DataQualityAgent" in delegated_targets

        completed_agents = [
            event.payload.get("agent")
            for event in events
            if event.type == CallbackEventType.AGENT_COMPLETED
        ]
        assert "MarketDataAgent" in completed_agents
        assert "DataQualityAgent" in completed_agents

    def test_reporting_final_answer_without_explicit_source_requires_pause(
        self,
        mock_data_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """ReportingAgent should pause when no explicit source choice is provided.

        A final answer that mentions multiple sources must not be treated as
        a user selection.
        """
        mock_processor.factory.chat_sequence = [
            (
                "Thought: Resolve instrument.\n"
                "Action: get_instrument_details\n"
                "Action Input: {\"query\": \"AAPL\"}"
            ),
            "Final Answer: Instrument resolved successfully. Symbol: AAPL.",
            (
                "Final Answer: Historical prices for AAPL have been retrieved from "
                "Yahoo, Bloomberg, and Reuters."
            ),
            "Final Answer: Quality metrics computed for yahoo, bloomberg, and reuters.",
            (
                "Thought: Present report and request explicit user source selection.\n"
                "Action: request_human_input\n"
                "Action Input: {\"prompt\": \"Choose source\", \"options\": [\"yahoo\", \"bloomberg\", \"reuters\"]}"
            ),
        ]

        events = mock_processor.process_user_request("AAPL from 2023-01-01 to 2024-01-01")

        delegated_targets = [
            event.payload.get("to_agent")
            for event in events
            if event.type == CallbackEventType.DELEGATED
        ]
        assert "ReportingAgent" in delegated_targets
        assert "GapFillingAgent" not in delegated_targets

        pauses = [
            event for event in events
            if event.type == CallbackEventType.AWAITING_USER_INPUT
        ]
        assert pauses, "Expected explicit pause for source selection"
        assert pauses[-1].payload.get("agent") == "ReportingAgent"

    def test_reporting_narrative_final_answer_forces_source_selection_pause(
        self,
        mock_data_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """Narrative ReportingAgent final answer must still trigger HITL pause.

        Reproduces the case where ReportingAgent claims quality report is ready
        but does not call request_human_input.
        """
        mock_processor.factory.chat_sequence = [
            (
                "Thought: Resolve instrument.\n"
                "Action: get_instrument_details\n"
                "Action Input: {\"query\": \"AAPL\"}"
            ),
            "Final Answer: Instrument resolved successfully. Symbol: AAPL.",
            "Final Answer: Historical prices for AAPL have been retrieved successfully.",
            "Final Answer: Data quality stage started.",
            (
                "Final Answer: A quality report has been generated for AAPL from Yahoo, "
                "Bloomberg, and Reuters. The data is consistent across all sources."
            ),
        ]

        events = mock_processor.process_user_request("AAPL from 2023-01-01 to 2024-01-01")

        pauses = [
            event for event in events
            if event.type == CallbackEventType.AWAITING_USER_INPUT
        ]
        assert pauses, "Expected forced source-selection pause at ReportingAgent"
        assert pauses[-1].payload.get("agent") == "ReportingAgent"

        reporting_completed = [
            event for event in events
            if event.type == CallbackEventType.AGENT_COMPLETED
            and event.payload.get("agent") == "ReportingAgent"
        ]
        assert not reporting_completed, "ReportingAgent should pause, not complete"

    def test_gapfilling_narrative_final_answer_forces_method_selection_pause(
        self,
        mock_data_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """GapFillingAgent narrative final answers must trigger HITL method pause."""
        mock_processor.factory.chat_sequence = [
            (
                "Thought: Resolve instrument.\n"
                "Action: get_instrument_details\n"
                "Action Input: {\"query\": \"AAPL\"}"
            ),
            "Final Answer: Instrument resolved successfully. Symbol: AAPL.",
            "Final Answer: Historical prices for AAPL have been retrieved successfully.",
            "Final Answer: Data quality stage started.",
            "Final Answer: Quality report is ready. Please choose a source.",
            "Final Answer: Source selection acknowledged. Delegating to gap-filling.",
            "Final Answer: Gap filling can now proceed once a method is selected.",
        ]

        # Initial pass pauses at ReportingAgent for source selection.
        first_events = mock_processor.process_user_request("AAPL from 2023-01-01 to 2024-01-01")
        first_pauses = [e for e in first_events if e.type == CallbackEventType.AWAITING_USER_INPUT]
        assert first_pauses and first_pauses[-1].payload.get("agent") == "ReportingAgent"

        # Resume with explicit source; GapFillingAgent should then force method-selection pause.
        second_events = mock_processor.process_user_response("yahoo")
        second_pauses = [e for e in second_events if e.type == CallbackEventType.AWAITING_USER_INPUT]
        assert second_pauses, "Expected forced method-selection pause at GapFillingAgent"
        assert second_pauses[-1].payload.get("agent") == "GapFillingAgent"

    def test_data_quality_malformed_delegate_payload_recovers_to_reporting(
        self,
        mock_data_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """DataQualityAgent malformed delegate payload should not loop indefinitely.

        When local model returns delegate_to_agent with empty target/request,
        processor should recover deterministically and continue to ReportingAgent
        source-selection checkpoint.
        """
        mock_processor.factory.chat_sequence = [
            (
                "Thought: Resolve instrument.\n"
                "Action: get_instrument_details\n"
                "Action Input: {\"query\": \"AAPL\"}"
            ),
            "Final Answer: Instrument resolved successfully. Symbol: AAPL.",
            "Final Answer: Historical prices for AAPL have been retrieved successfully.",
            (
                "Thought: Continue to reporting.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"\", \"request\": \"\"}"
            ),
            "Final Answer: Quality report generated across sources.",
        ]

        events = mock_processor.process_user_request("AAPL from 2023-01-01 to 2024-01-01")

        delegated_targets = [
            event.payload.get("to_agent")
            for event in events
            if event.type == CallbackEventType.DELEGATED
        ]
        assert "ReportingAgent" in delegated_targets

        pauses = [
            event for event in events
            if event.type == CallbackEventType.AWAITING_USER_INPUT
        ]
        assert pauses and pauses[-1].payload.get("agent") == "ReportingAgent"

    def test_data_quality_final_answer_fallback_computes_metrics(
        self,
        mock_data_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """DataQuality fallback should compute metrics, not placeholder None values."""
        mock_processor.factory.chat_sequence = [
            "Final Answer: Instrument resolved successfully. Symbol: AAPL.",
            "Final Answer: Historical prices for AAPL have been retrieved from yahoo, bloomberg, and reuters.",
            "Final Answer: Data quality checks completed for AAPL across sources.",
            "Final Answer: Quality comparison prepared. Please choose one source to continue.",
        ]

        events = mock_processor.process_user_request("Build AAPL from 2023-01-03 to 2023-01-31")

        data_quality_event = next(
            (
                event
                for event in events
                if event.type == CallbackEventType.AGENT_COMPLETED
                and event.payload.get("agent") == "DataQualityAgent"
            ),
            None,
        )
        assert data_quality_event is not None, "Expected DataQualityAgent completion event"

        quality_report = data_quality_event.payload.get("result", {}).get("data_quality_report")
        assert isinstance(quality_report, dict)
        rows = quality_report.get("rows", [])
        assert rows, "Expected quality report rows"

        # Ensure deterministic fallback computed actual metrics.
        for row in rows:
            assert row.get("completeness_pct") is not None
            assert row.get("available_record_count") is not None
            assert row.get("min_date") is not None
            assert row.get("max_date") is not None

        summary = quality_report.get("summary", {})
        assert summary.get("total_available_records") is not None
        assert summary.get("average_completeness_pct") is not None


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

    def test_handler_structured_trace_records(self) -> None:
        """Structured trace records should be accumulated separately."""
        from financial_time_series_construction.handler import TimeSeriesConstructionHandler

        handler = TimeSeriesConstructionHandler(session_id="test")
        handler.add_trace_record("llm_response", {"content": "Thought: test"}, agent="AgentA", iteration=0)
        handler.add_trace_record("tool_call", {"tool": "check_data_quality"}, agent="AgentA", iteration=0)

        records = handler.get_trace_records()
        assert len(records) == 2
        assert records[0]["type"] == "llm_response"
        assert records[1]["payload"]["tool"] == "check_data_quality"
        trace_text = handler.get_trace()
        assert '"type": "llm_response"' in trace_text
        assert '"type": "tool_call"' in trace_text

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
                "Thought: Create final visualization artifact.\n"
                "Action: visualize_timeseries\n"
                "Action Input: {\"prices\": {\"symbol\": \"AAPL\", "
                "\"dates\": [\"2023-01-03\", \"2023-01-04\", \"2023-01-05\"], "
                "\"prices\": [150.0, 150.5, 151.0]}, \"title\": \"AAPL Continuous Series\", "
                "\"run_id\": \"it_full_flow\"}"
            ),
            (
                "Thought: Delegate to reporting for final summary.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"ReportingAgent\", "
                "\"request\": \"summarize final artifacts for AAPL\"}"
            ),
            (
                "Thought: Finish with a concise workflow summary.\n"
                "Final Answer: Completed workflow for AAPL with CSV and chart artifacts generated."
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
        assert "ReportingAgent" in completed_agents

        delegated = [e for e in final_pass_events if e.type == CallbackEventType.DELEGATED]
        delegated_edges = {
            (e.payload.get("from_agent"), e.payload.get("to_agent"))
            for e in delegated
        }
        assert ("TimeSeriesConstructionAgent", "ReportingAgent") in delegated_edges

        errors = [e for e in final_pass_events if e.type == CallbackEventType.ERROR]
        assert not errors, f"Unexpected errors in final pass: {errors}"

        output_file = mock_output_dir / "it_full_flow" / "final_timeseries.csv"
        assert output_file.exists(), "Expected final output file to be persisted"
        chart_file = mock_output_dir / "it_full_flow" / "timeseries.png"
        assert chart_file.exists(), "Expected final visualization file to be persisted"

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

    def test_dataquality_malformed_check_payload_recovers_to_reporting_pause(
        self,
        mock_data_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """Malformed check_data_quality args should recover without hard error.

        Regression target: stronger models may emit check_data_quality with
        instrument_symbol/sources but without prices or data payload.
        Processor should recover from prior historical_prices tool results,
        continue to ReportingAgent, and pause for source selection.
        """
        mock_processor.factory.chat_sequence = [
            (
                "Thought: Compute quality metrics for all sources.\n"
                "Action: check_data_quality\n"
                "Action Input: {\"instrument_symbol\": \"AAPL\", \"sources\": [\"yahoo\", \"bloomberg\", \"reuters\"], "
                "\"start_date\": \"2023-01-03\", \"end_date\": \"2023-01-31\"}"
            ),
            (
                "Thought: Ask user to select preferred source.\n"
                "Action: request_human_input\n"
                "Action Input: {\"prompt\": \"Choose preferred source for AAPL\", "
                "\"options\": [\"yahoo\", \"bloomberg\", \"reuters\"]}"
            ),
        ]

        mock_processor.handler.paused_state = {
            "agent": "DataQualityAgent",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "run quality checks for AAPL across all sources. "
                        "Original request: Build AAPL from 2023-01-03 to 2023-01-31"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Tool result: {\"symbol\": \"AAPL\", \"source\": \"yahoo\", "
                        "\"dates\": [\"2023-01-03\", \"2023-01-04\", \"2023-01-05\"], "
                        "\"prices\": [150.0, null, 151.0]}"
                    ),
                },
            ],
            "iteration": 0,
        }

        events = mock_processor.process_user_response("continue")

        errors = [e for e in events if e.type == CallbackEventType.ERROR]
        assert not errors, f"Unexpected errors during malformed payload recovery: {errors}"

        delegated_edges = {
            (e.payload.get("from_agent"), e.payload.get("to_agent"))
            for e in events
            if e.type == CallbackEventType.DELEGATED
        }
        assert ("DataQualityAgent", "ReportingAgent") in delegated_edges

        pauses = [
            e for e in events
            if e.type == CallbackEventType.AWAITING_USER_INPUT
            and e.payload.get("agent") == "ReportingAgent"
        ]
        assert pauses, "Expected source-selection pause from ReportingAgent"

    def test_gapfilling_selected_method_does_not_repause(
        self,
        mock_data_dir: Path,
        mock_output_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """GapFilling should not loop on request_human_input after user picks a method."""
        mock_processor.factory.chat_sequence = [
            # Looping model behavior: asks again even though user already selected.
            (
                "Thought: Please confirm your selected method again.\n"
                "Action: request_human_input\n"
                "Action Input: {\"prompt\": \"Please choose a gap-filling method\", \"options\": [\"linear_interpolation\", \"forward_fill\", \"backward_fill\", \"none\"]}"
            ),
            (
                "Thought: Delegate to final series construction.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"TimeSeriesConstructionAgent\", \"request\": \"build final AAPL continuous series\"}"
            ),
            (
                "Thought: Persist final series artifact.\n"
                "Action: build_timeseries\n"
                "Action Input: {\"series\": {\"symbol\": \"AAPL\", \"dates\": [\"2023-01-03\", \"2023-01-04\", \"2023-01-05\"], \"prices\": [150.0, 150.5, 151.0]}, \"filename\": \"final_timeseries.csv\", \"run_id\": \"it_gap_loop\"}"
            ),
            (
                "Thought: Complete construction stage.\n"
                "Final Answer: Constructed final AAPL time series artifact."
            ),
            (
                "Thought: Provide final report to the user.\n"
                "Final Answer: Final summary complete with constructed CSV and visualization artifacts."
            ),
        ]

        # Resume directly at GapFilling pause to isolate loop behavior.
        mock_processor.handler.paused_state = {
            "agent": "GapFillingAgent",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "apply gap filling to yahoo AAPL. "
                        "Original request: Build AAPL from 2023-01-03 to 2023-01-31"
                    ),
                }
            ],
            "iteration": 1,
        }

        final_events = mock_processor.process_user_response("1")
        repause_events = [
            e for e in final_events
            if e.type == CallbackEventType.AWAITING_USER_INPUT
            and e.payload.get("agent") == "GapFillingAgent"
        ]
        assert not repause_events, "GapFillingAgent should not re-pause after method selection"

        delegated_edges = {
            (e.payload.get("from_agent"), e.payload.get("to_agent"))
            for e in final_events
            if e.type == CallbackEventType.DELEGATED
        }
        assert ("GapFillingAgent", "TimeSeriesConstructionAgent") in delegated_edges

        completed_agents = {
            e.payload.get("agent")
            for e in final_events
            if e.type == CallbackEventType.AGENT_COMPLETED
        }
        assert "TimeSeriesConstructionAgent" in completed_agents

    def test_gapfilling_method_resume_auto_continues_to_construction(
        self,
        mock_data_dir: Path,
        mock_output_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """Numeric method response at GapFilling pause should auto-continue.

        Regression target: avoid reliance on an extra GapFilling LLM turn after
        user method selection. Processor should apply method from context and
        delegate directly to TimeSeriesConstructionAgent.
        """
        mock_processor.factory.chat_sequence = [
            (
                "Thought: Persist final series artifact.\n"
                "Action: build_timeseries\n"
                "Action Input: {\"series\": {\"symbol\": \"AAPL\", "
                "\"dates\": [\"2023-01-03\", \"2023-01-04\", \"2023-01-05\"], "
                "\"prices\": [150.0, 150.5, 151.0]}, \"filename\": \"final_timeseries.csv\", "
                "\"run_id\": \"it_gap_resume_auto\"}"
            ),
            (
                "Thought: Final series is generated.\n"
                "Final Answer: Time series construction completed for AAPL."
            ),
            (
                "Thought: Provide final report to the user.\n"
                "Final Answer: Final summary complete with constructed CSV and visualization artifacts."
            ),
        ]

        mock_processor.handler.paused_state = {
            "agent": "GapFillingAgent",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "apply gap filling to yahoo AAPL. "
                        "Original request: Build AAPL from 2023-01-03 to 2023-01-31"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Tool result: {\"symbol\": \"AAPL\", \"source\": \"yahoo\", "
                        "\"dates\": [\"2023-01-03\", \"2023-01-04\", \"2023-01-05\"], "
                        "\"prices\": [150.0, null, 151.0]}"
                    ),
                },
            ],
            "iteration": 1,
            "checkpoint": "gap_method_selection",
        }

        final_events = mock_processor.process_user_response("1")

        repause_events = [
            e for e in final_events
            if e.type == CallbackEventType.AWAITING_USER_INPUT
            and e.payload.get("agent") == "GapFillingAgent"
        ]
        assert not repause_events, "GapFillingAgent should not re-pause after numeric method input"

        delegated_edges = {
            (e.payload.get("from_agent"), e.payload.get("to_agent"))
            for e in final_events
            if e.type == CallbackEventType.DELEGATED
        }
        assert ("GapFillingAgent", "TimeSeriesConstructionAgent") in delegated_edges

        completed_agents = {
            e.payload.get("agent")
            for e in final_events
            if e.type == CallbackEventType.AGENT_COMPLETED
        }
        assert "TimeSeriesConstructionAgent" in completed_agents

        output_file = mock_output_dir / "it_gap_resume_auto" / "final_timeseries.csv"
        assert output_file.exists(), "Expected final output file to be persisted"

    def test_gapfilling_resume_uses_source_selection_marker_and_skips_repause(
        self,
        mock_data_dir: Path,
        mock_output_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """Method-selected resumes should use [SOURCE_SELECTION] marker for source recovery.

        Regression target: when quality context lists multiple sources, source
        extraction from free-form text may fail and trigger GapFilling re-pause.
        The explicit [SOURCE_SELECTION] marker must drive deterministic recovery
        and continue directly to construction/reporting.
        """
        mock_processor.factory.chat_sequence = [
            (
                "Thought: Persist final series artifact.\n"
                "Action: build_timeseries\n"
                "Action Input: {\"series\": {\"symbol\": \"AAPL\", "
                "\"dates\": [\"2023-01-03\", \"2023-01-04\", \"2023-01-05\"], "
                "\"prices\": [150.0, 150.5, 151.0]}, \"filename\": \"final_timeseries.csv\", "
                "\"run_id\": \"it_gap_marker_resume\"}"
            ),
            (
                "Thought: Final series is generated.\n"
                "Final Answer: Time series construction completed for AAPL."
            ),
            (
                "Thought: Provide final report to the user.\n"
                "Final Answer: Final summary complete with constructed CSV and visualization artifacts."
            ),
        ]

        mock_processor.handler.paused_state = {
            "agent": "GapFillingAgent",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Apply gap filling for AAPL. Quality comparison includes yahoo, bloomberg, reuters. "
                        "Original request: Build AAPL from 2023-01-03 to 2023-01-31"
                    ),
                },
                {
                    "role": "user",
                    "content": "[SOURCE_SELECTION] bloomberg",
                },
            ],
            "iteration": 2,
            "checkpoint": "gap_method_selection",
        }

        final_events = mock_processor.process_user_response("1")

        repause_events = [
            e for e in final_events
            if e.type == CallbackEventType.AWAITING_USER_INPUT
            and e.payload.get("agent") == "GapFillingAgent"
        ]
        assert not repause_events, "GapFillingAgent should not re-pause after method selection"

        delegated_edges = {
            (e.payload.get("from_agent"), e.payload.get("to_agent"))
            for e in final_events
            if e.type == CallbackEventType.DELEGATED
        }
        assert ("GapFillingAgent", "TimeSeriesConstructionAgent") in delegated_edges

        completed_agents = {
            e.payload.get("agent")
            for e in final_events
            if e.type == CallbackEventType.AGENT_COMPLETED
        }
        assert "TimeSeriesConstructionAgent" in completed_agents

        output_file = mock_output_dir / "it_gap_marker_resume" / "final_timeseries.csv"
        assert output_file.exists(), "Expected final output file to be persisted"

    def test_construction_final_answer_auto_continues_to_reporting(
        self,
        mock_data_dir: Path,
        mock_output_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """Construction narrative final answers should still delegate to ReportingAgent.

        Regression: local models may stop at TimeSeriesConstructionAgent after
        build_timeseries without calling visualize_timeseries or delegate_to_agent.
        The processor must complete missing artifact generation and finish with
        a final reporting summary.
        """
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
                "\"run_id\": \"it_missing_visual\"}"
            ),
            (
                "Thought: Final series is generated.\n"
                "Final Answer: Time series construction completed for AAPL."
            ),
            (
                "Thought: Summarize final result for the user.\n"
                "Final Answer: Completed end-to-end workflow for AAPL with final CSV and chart artifacts."
            ),
            (
                "Thought: Provide final report to the user.\n"
                "Final Answer: Final summary complete with constructed CSV and visualization artifacts."
            ),
        ]

        first_pass_events = mock_processor.process_user_request(
            "Build AAPL from 2023-01-03 to 2023-01-31 and help me fill data gaps."
        )
        first_pause = [e for e in first_pass_events if e.type == CallbackEventType.AWAITING_USER_INPUT]
        assert first_pause and first_pause[-1].payload["agent"] == "ReportingAgent"

        second_pass_events = mock_processor.process_user_response("yahoo")
        second_pause = [e for e in second_pass_events if e.type == CallbackEventType.AWAITING_USER_INPUT]
        assert second_pause and second_pause[-1].payload["agent"] == "GapFillingAgent"

        final_events = mock_processor.process_user_response("linear_interpolation")
        completed = [e for e in final_events if e.type == CallbackEventType.AGENT_COMPLETED]
        completed_agents = {e.payload.get("agent") for e in completed}
        assert "TimeSeriesConstructionAgent" in completed_agents
        assert "ReportingAgent" in completed_agents

        delegated = [e for e in final_events if e.type == CallbackEventType.DELEGATED]
        delegated_edges = {
            (e.payload.get("from_agent"), e.payload.get("to_agent"))
            for e in delegated
        }
        assert ("TimeSeriesConstructionAgent", "ReportingAgent") in delegated_edges

        output_file = mock_output_dir / "it_missing_visual" / "final_timeseries.csv"
        assert output_file.exists(), "Expected final output file to be persisted"

        png_files = list(mock_output_dir.rglob("timeseries.png"))
        assert png_files, "Expected visualization artifact to be persisted"

        errors = [e for e in final_events if e.type == CallbackEventType.ERROR]
        assert not errors, f"Unexpected errors in final pass: {errors}"

    def test_construction_unparseable_action_input_still_continues(
        self,
        mock_data_dir: Path,
        mock_output_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """Malformed TimeSeries action payload should not block final continuation.

        Reproduces local-model behavior where build_timeseries Action Input is
        truncated/invalid JSON. Processor should recover deterministically from
        filled-data context, persist artifacts, and delegate to ReportingAgent.
        """
        mock_processor.factory.chat_sequence = [
            (
                "Thought: Build the final artifact now.\n"
                "Action: build_timeseries\n"
                "Action Input: {\"series\": {\"symbol\": \"AAPL\", "
                "\"dates\": [\"2023-01-03\", \"2023-01-04\", \"2023-01-05\"], "
                "\"prices\": [150.0, 150.5, 151.0]}, \"filename\": \"final_timeseries.csv\", "
                "\"run_id\": \"it_construction_unparseable\""
            ),
            (
                "Thought: Summarize final result for the user.\n"
                "Final Answer: Final summary complete with constructed CSV and visualization artifacts."
            ),
        ]

        mock_processor.handler.paused_state = {
            "agent": "TimeSeriesConstructionAgent",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Build and persist the final time series for AAPL using linear_interpolation gap-filling. "
                        "Filled data: {\"symbol\": \"AAPL\", \"method\": \"linear_interpolation\", "
                        "\"dates\": [\"2023-01-03\", \"2023-01-04\", \"2023-01-05\"], "
                        "\"prices\": [150.0, 150.5, 151.0]}. "
                        "Original request: Build AAPL from 2023-01-03 to 2023-01-31"
                    ),
                }
            ],
            "iteration": 0,
        }

        final_events = mock_processor.process_user_response("continue")

        delegated_edges = {
            (e.payload.get("from_agent"), e.payload.get("to_agent"))
            for e in final_events
            if e.type == CallbackEventType.DELEGATED
        }
        assert ("TimeSeriesConstructionAgent", "ReportingAgent") in delegated_edges

        completed_agents = {
            e.payload.get("agent")
            for e in final_events
            if e.type == CallbackEventType.AGENT_COMPLETED
        }
        assert "TimeSeriesConstructionAgent" in completed_agents
        assert "ReportingAgent" in completed_agents

        output_file = mock_output_dir / "default" / "final_timeseries.csv"
        assert output_file.exists(), "Expected deterministic construction CSV artifact"
        chart_file = mock_output_dir / "default" / "timeseries.png"
        assert chart_file.exists(), "Expected deterministic construction chart artifact"

    def test_gapfilling_final_answer_with_method_still_continues_to_construction(
        self,
        mock_data_dir: Path,
        mock_output_dir: Path,
        mock_processor: TimeSeriesConstructionProcessor,
    ) -> None:
        """GapFilling narrative completion with explicit method must not terminate flow.

        Reproduces qwen behavior where GapFillingAgent returns a Final Answer
        mentioning linear_interpolation without calling apply_gap_filling.
        Processor should recover prices, apply gap-filling deterministically,
        then continue to TimeSeriesConstructionAgent and final Reporting summary.
        """
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
                "Thought: Pull bloomberg prices for the requested range.\n"
                "Action: historical_prices\n"
                "Action Input: {\"symbol\": \"AAPL\", \"start_date\": \"2023-01-03\", "
                "\"end_date\": \"2023-01-31\", \"source\": \"bloomberg\"}"
            ),
            (
                "Thought: Pass prices to data quality agent.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"DataQualityAgent\", "
                "\"request\": \"run quality checks for bloomberg AAPL\"}"
            ),
            (
                "Thought: Compute quality metrics.\n"
                "Action: check_data_quality\n"
                "Action Input: {\"prices\": [150.0, null, 151.0, 151.4], "
                "\"source\": \"bloomberg\", \"symbol\": \"AAPL\"}"
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
                "Thought: Continue with gap filling.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"GapFillingAgent\", "
                "\"request\": \"apply gap filling to bloomberg AAPL. Original request: Build AAPL from 2023-01-03 to 2023-01-31 and help me fill data gaps.\"}"
            ),
            (
                "Thought: Ask user to choose a gap filling method.\n"
                "Action: request_human_input\n"
                "Action Input: {\"prompt\": \"Select gap filling method\", "
                "\"options\": [\"linear_interpolation\", \"forward_fill\", \"backward_fill\"]}"
            ),
            (
                "Thought: Gap-filling applied using linear_interpolation to the AAPL instrument from Bloomberg.\n"
                "Final Answer: Gap-filling applied using linear_interpolation to the AAPL instrument from Bloomberg."
            ),
            (
                "Thought: Persist final series artifact.\n"
                "Action: build_timeseries\n"
                "Action Input: {\"series\": {\"symbol\": \"AAPL\", "
                "\"dates\": [\"2023-01-03\", \"2023-01-04\", \"2023-01-05\"], "
                "\"prices\": [150.0, 150.5, 151.0]}, \"filename\": \"final_timeseries.csv\", "
                "\"run_id\": \"it_gap_method_final\"}"
            ),
            (
                "Thought: Summarize final result for the user.\n"
                "Final Answer: Completed end-to-end workflow for AAPL with final CSV and chart artifacts."
            ),
            (
                "Thought: Provide final report to the user.\n"
                "Final Answer: Final summary complete with constructed CSV and visualization artifacts."
            ),
        ]

        first_pass_events = mock_processor.process_user_request(
            "Build AAPL from 2023-01-03 to 2023-01-31 and help me fill data gaps."
        )
        first_pause = [e for e in first_pass_events if e.type == CallbackEventType.AWAITING_USER_INPUT]
        assert first_pause and first_pause[-1].payload["agent"] == "ReportingAgent"

        second_pass_events = mock_processor.process_user_response("bloomberg")
        second_pause = [e for e in second_pass_events if e.type == CallbackEventType.AWAITING_USER_INPUT]
        assert second_pause and second_pause[-1].payload["agent"] == "GapFillingAgent"

        final_events = mock_processor.process_user_response("linear_interpolation")
        completed = [e for e in final_events if e.type == CallbackEventType.AGENT_COMPLETED]
        completed_agents = {e.payload.get("agent") for e in completed}
        assert "GapFillingAgent" in completed_agents
        assert "TimeSeriesConstructionAgent" in completed_agents
        assert "ReportingAgent" in completed_agents

        delegated_edges = {
            (e.payload.get("from_agent"), e.payload.get("to_agent"))
            for e in final_events
            if e.type == CallbackEventType.DELEGATED
        }
        assert ("GapFillingAgent", "TimeSeriesConstructionAgent") in delegated_edges
        assert ("TimeSeriesConstructionAgent", "ReportingAgent") in delegated_edges

        output_file = mock_output_dir / "it_gap_method_final" / "final_timeseries.csv"
        assert output_file.exists(), "Expected final output file to be persisted"
        png_files = list(mock_output_dir.rglob("timeseries.png"))
        assert png_files, "Expected visualization artifact to be persisted"

        errors = [e for e in final_events if e.type == CallbackEventType.ERROR]
        assert not errors, f"Unexpected errors in final pass: {errors}"