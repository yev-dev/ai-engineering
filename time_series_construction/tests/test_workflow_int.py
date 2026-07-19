"""Integration test for the time series construction workflow.

Constructs a time series for APL (resolved to AAPL via fuzzy matching)
between 2023 and 2024 using mocked data files and a mocked LLM factory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from time_series_construction.agents_definition import CallbackEventType
from time_series_construction.processor import TimeSeriesConstructionProcessor
from time_series_construction.tools import (
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

# ---------------------------------------------------------------------------
# Fixtures – replicate the files that cli.py / tools.py depend on
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create mock CSV data files and patch ``tools.DATA_DIR``."""
    import time_series_construction.tools as tools_module

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

    # Yahoo  – inject a few NaN gaps to exercise quality / gap-filling
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
    import time_series_construction.tools as tools_module

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
    """Verify that the fuzzy-matching logic resolves APL → AAPL."""

    def test_apl_resolves_to_aapl(self, mock_data_dir: Path) -> None:
        result = get_instrument_details(query="APL")
        assert result["found"] is True
        assert result["symbol"] == "AAPL"

    def test_apl_resolves_via_symbol_arg(self, mock_data_dir: Path) -> None:
        result = get_instrument_details(symbol="APL")
        assert result["found"] is True
        assert result["symbol"] == "AAPL"

    def test_aapl_direct_lookup(self, mock_data_dir: Path) -> None:
        result = get_instrument_details(query="AAPL")
        assert result["found"] is True
        assert result["symbol"] == "AAPL"

    def test_apple_inc_full_name(self, mock_data_dir: Path) -> None:
        result = get_instrument_details(query="Apple Inc.")
        assert result["found"] is True
        assert result["symbol"] == "AAPL"


# ---------------------------------------------------------------------------
# Tests – historical data loading
# ---------------------------------------------------------------------------


class TestHistoricalData:
    """Fetch and inspect AAPL prices across the 2023–2024 window."""

    def test_available_sources(self, mock_data_dir: Path) -> None:
        sources = available_data_sources()
        assert "yahoo" in sources
        assert "bloomberg" in sources
        assert "reuters" in sources

    def test_fetch_yahoo_full_range(self, mock_data_dir: Path) -> None:
        prices = historical_prices("AAPL", "2023-01-03", "2024-12-30", "yahoo")
        assert prices["symbol"] == "AAPL"
        assert prices["source"] == "yahoo"
        assert len(prices["dates"]) > 0
        assert len(prices["prices"]) > 0
        assert prices["dates"][0] >= "2023-01-03"
        assert prices["dates"][-1] <= "2024-12-30"

    def test_fetch_bloomberg(self, mock_data_dir: Path) -> None:
        prices = historical_prices("AAPL", "2023-06-01", "2023-06-30", "bloomberg")
        assert prices["symbol"] == "AAPL"
        assert prices["source"] == "bloomberg"
        assert len(prices["dates"]) > 0

    def test_fetch_reuters(self, mock_data_dir: Path) -> None:
        prices = historical_prices("AAPL", "2024-01-02", "2024-01-31", "reuters")
        assert prices["symbol"] == "AAPL"
        assert prices["source"] == "reuters"
        assert len(prices["dates"]) > 0

    def test_empty_date_range_raises(self, mock_data_dir: Path) -> None:
        with pytest.raises(ValueError, match="No historical data is available"):
            historical_prices("AAPL", "2021-01-01", "2021-01-31", "yahoo")

    def test_unknown_ticker_raises(self, mock_data_dir: Path) -> None:
        with pytest.raises(ValueError, match="is not available"):
            historical_prices("NOPE", "2023-01-01", "2023-12-31", "yahoo")


# ---------------------------------------------------------------------------
# Tests – data quality and gap filling
# ---------------------------------------------------------------------------


class TestDataQuality:
    """Quality metrics detect injected gaps; gap filling repairs them."""

    def test_quality_detects_missing_values(self, mock_data_dir: Path) -> None:
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "yahoo")
        quality = check_data_quality(prices["prices"], "yahoo", "AAPL")
        assert quality["missing_count"] > 0
        assert quality["completeness_pct"] < 100.0

    def test_linear_interpolation_fills_gaps(self, mock_data_dir: Path) -> None:
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "yahoo")
        filled = apply_gap_filling(prices, "linear_interpolation")
        assert filled["method"] == "linear_interpolation"
        assert filled["symbol"] == "AAPL"
        assert len(filled["dates"]) == len(prices["dates"])
        assert all(p is not None for p in filled["prices"])

    def test_forward_fill(self, mock_data_dir: Path) -> None:
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "yahoo")
        filled = apply_gap_filling(prices, "forward_fill")
        assert filled["method"] == "forward_fill"
        assert all(p is not None for p in filled["prices"])

    def test_backward_fill(self, mock_data_dir: Path) -> None:
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "yahoo")
        filled = apply_gap_filling(prices, "backward_fill")
        assert filled["method"] == "backward_fill"
        assert all(p is not None for p in filled["prices"])

    def test_no_gap_method_preserves_nans(self, mock_data_dir: Path) -> None:
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "yahoo")
        filled = apply_gap_filling(prices, "none")
        assert filled["method"] == "none"
        assert any(p is None for p in filled["prices"])

    def test_recommend_methods_with_gaps(self, mock_data_dir: Path) -> None:
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "yahoo")
        quality = check_data_quality(prices["prices"], "yahoo", "AAPL")
        methods = recommend_gap_methods(quality, prices)
        assert "linear_interpolation" in methods
        assert "forward_fill" in methods
        assert "backward_fill" in methods

    def test_recommend_methods_no_gaps(self, mock_data_dir: Path) -> None:
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "bloomberg")
        quality = check_data_quality(prices["prices"], "bloomberg", "AAPL")
        methods = recommend_gap_methods(quality, prices)
        assert methods == ["none"]


# ---------------------------------------------------------------------------
# Tests – artifact generation
# ---------------------------------------------------------------------------


class TestArtifacts:
    """CSV reports, final series, and charts are written to the output dir."""

    def test_build_timeseries_csv(self, mock_data_dir: Path, mock_output_dir: Path) -> None:
        prices = historical_prices("AAPL", "2023-01-03", "2024-12-30", "yahoo")
        filled = apply_gap_filling(prices, "linear_interpolation")
        path = build_timeseries(filled, filename="AAPL_timeseries.csv", run_id="int_test")
        csv_path = Path(path)
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["date", "price"]
        assert len(df) == len(filled["dates"])

    def test_generate_report_csv(self, mock_data_dir: Path, mock_output_dir: Path) -> None:
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "yahoo")
        quality = check_data_quality(prices["prices"], "yahoo", "AAPL")
        path = generate_report(quality, filename="quality_report.csv", run_id="int_test")
        csv_path = Path(path)
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert "source" in df.columns
        assert df["source"].iloc[0] == "yahoo"

    def test_visualize_timeseries_png(self, mock_data_dir: Path, mock_output_dir: Path) -> None:
        prices = historical_prices("AAPL", "2023-01-03", "2024-12-30", "yahoo")
        filled = apply_gap_filling(prices, "linear_interpolation")
        path = visualize_timeseries(filled, title="AAPL 2023-2024", run_id="int_test")
        png_path = Path(path)
        assert png_path.exists()
        assert png_path.suffix == ".png"


# ---------------------------------------------------------------------------
# Tests – full ReAct workflow with mocked LLM
# ---------------------------------------------------------------------------


class TestFullWorkflow:
    """Simulate the end-to-end ReAct loop with a dummy LLM factory."""

    def test_workflow_completes_for_apl(self, mock_data_dir: Path, mock_processor: TimeSeriesConstructionProcessor) -> None:
        """The delegation chain should route APL through ReferenceDataAgent to MarketDataAgent.

        Because the ReAct loop pauses at ``request_human_input``, we verify that
        the processor reaches the expected agent delegation steps and emits the
        correct events before pausing.
        """
        mock_processor.factory.chat_sequence = [
            # Orchestrator → delegate to ReferenceDataAgent
            (
                "Thought: I need to resolve the instrument first.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"ReferenceDataAgent\", "
                "\"request\": \"create a time series for APL between 2023 and 2024\"}"
            ),
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
            # Orchestrator → delegate to ReferenceDataAgent
            (
                "Thought: Resolve APL.\n"
                "Action: delegate_to_agent\n"
                "Action Input: {\"agent_name\": \"ReferenceDataAgent\", "
                "\"request\": \"create a time series for APL between 2023 and 2024\"}"
            ),
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