"""Integration tests for the time series construction autogen workflow.

Tests cover real tool functions with mock data files (no LLM mocking):
- Instrument resolution (fuzzy matching, exact match, by name)
- Historical data loading from all sources
- Data quality metrics and gap filling
- Artifact generation (CSV, reports, visualizations)
- Flexible date parsing
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

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

    # Set a run_id so tools can use the DataStore
    tools_module.set_run_id("test_run")

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

    # Set a run_id so tools can use the DataStore
    tools_module.set_run_id("test_run")

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr(tools_module, "OUTPUT_ROOT", output_dir)
    return output_dir


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
        quality = check_data_quality(prices=prices["prices"], source="yahoo", symbol="AAPL")
        assert quality["missing_count"] > 0
        assert quality["completeness_pct"] < 100.0

    def test_quality_bloomberg_no_missing(self, mock_data_dir: Path) -> None:
        """Bloomberg data has no gaps."""
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "bloomberg")
        quality = check_data_quality(prices=prices["prices"], source="bloomberg", symbol="AAPL")
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
        quality = check_data_quality(prices=prices["prices"], source="yahoo", symbol="AAPL")
        methods = recommend_gap_methods(quality, prices)
        assert "linear_interpolation" in methods
        assert "forward_fill" in methods
        assert "backward_fill" in methods

    def test_recommend_methods_no_gaps(self, mock_data_dir: Path) -> None:
        """When no gaps exist, recommend 'none'."""
        prices = historical_prices("AAPL", "2023-01-03", "2023-01-15", "bloomberg")
        quality = check_data_quality(prices=prices["prices"], source="bloomberg", symbol="AAPL")
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
        quality = check_data_quality(prices=prices["prices"], source="yahoo", symbol="AAPL")
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