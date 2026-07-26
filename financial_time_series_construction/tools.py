"""Deterministic domain tools exposed to ReAct agents via langchain annotations."""
from __future__ import annotations

import concurrent.futures
import logging
import os
import re
import uuid
from datetime import timedelta
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from langchain_core.tools import StructuredTool

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_ROOT = Path(os.getenv("TIME_SERIES_OUTPUT_DIR", Path.home() / "time_series_construction"))
SOURCES = ("yahoo", "bloomberg", "reuters")
logger = logging.getLogger(__name__)


def _debug_flow_enabled() -> bool:
    """Return True when lightweight flow debugging is enabled."""
    return str(os.getenv("TSC_DEBUG_FLOW", "")).strip().casefold() in {
        "1", "true", "yes", "on",
    }


def _debug_tool_event(tool: str, phase: str, **kwargs: Any) -> None:
    """Emit one-line debug flow events for tool-level tracing."""
    if not _debug_flow_enabled():
        return
    details = " ".join(f"{key}={value}" for key, value in kwargs.items())
    logger.info("debug_flow component=tool tool=%s phase=%s %s", tool, phase, details)


def _log_tool_progress(tool_name: str, phase: str, **kwargs: Any) -> None:
    """Emit structured tool progress logs for workflow diagnostics."""
    details = " ".join(f"{key}={value}" for key, value in kwargs.items())
    logger.info("tool_progress tool=%s phase=%s %s", tool_name, phase, details)

_MONTH_WORD_RE = re.compile(
    r"^(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\s+(\d{4})$",
    re.IGNORECASE,
)
_MONTH_NUM_RE = re.compile(r"^(\d{1,2})[/-](\d{4})$")
_YEAR_RE = re.compile(r"^\d{4}$")
_QUARTER_RE = re.compile(r"^(?:q([1-4])\s*(\d{4})|(\d{4})\s*q([1-4]))$", re.IGNORECASE)
# Pre-compute common typos for "between" so users don't need exact spelling.
_BETWEEN_TYPOS = "|".join([
    "between",       # correct
    "betwne",        # missing one 'e'
    "betwen",        # missing one 'e'
    "betwene",       # extra 'e'
    "betwee",        # missing 'n'
    "betweem",       # 'm' instead of 'n'
    "betweeen",      # triple 'e'
    "beteen",        # missing 'w'
    "beteween",      # swapped 'we'
    "btween",        # missing 'e'
    "beetween",      # extra 'e'
    "bwtween",       # swapped 'we'
])

_RANGE_PATTERNS = (
    re.compile(
        r"\bfrom\s+(?P<start>.+?)\s+(?:to|until|through|thru|till)\s+(?P<end>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:{_BETWEEN_TYPOS})\s+(?P<start>.+?)\s+(?:and|to|until|through|thru|till)\s+(?P<end>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bstart\s+date\s*(?:is|=|:)?\s*(?P<start>.+?)\s+"
        r"(?:and\s+)?end\s+date\s*(?:is|=|:)?\s*(?P<end>.+)$",
        re.IGNORECASE,
    ),
)


def _run_dir(run_id: str | None = None) -> Path:
    """Create and return a run-specific output directory."""
    directory = OUTPUT_ROOT / (run_id or f"run_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}")
    directory.mkdir(parents=True, exist_ok=True)
    logger.debug("artifact_directory path=%s", directory)
    return directory


def parse_flexible_date(value: str, boundary: str = "start") -> pd.Timestamp:
    """Parse numeric or word-based dates into a normalized timestamp.

    Supported examples:
    - 2023-01-01
    - 01/31/2023
    - January 2023
    - 2023
    - Q1 2023 / 2023 Q1
    - today / yesterday / tomorrow
    """
    text = str(value or "").strip()
    if not text:
        raise ValueError("Date value is empty.")

    lowered = text.casefold().replace(",", "").strip()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    if lowered in {"today", "now"}:
        return now
    if lowered == "yesterday":
        return now - timedelta(days=1)
    if lowered == "tomorrow":
        return now + timedelta(days=1)

    match = _YEAR_RE.match(lowered)
    if match:
        year = int(lowered)
        return pd.Timestamp(f"{year}-01-01") if boundary == "start" else pd.Timestamp(f"{year}-12-31")

    match = _QUARTER_RE.match(lowered)
    if match:
        quarter = int(match.group(1) or match.group(4))
        year = int(match.group(2) or match.group(3))
        month_start = 1 + (quarter - 1) * 3
        start = pd.Timestamp(year=year, month=month_start, day=1)
        if boundary == "start":
            return start
        return (start + pd.offsets.QuarterEnd()).normalize()

    match = _MONTH_WORD_RE.match(lowered)
    if match:
        month_name, year = match.groups()
        start = pd.to_datetime(f"1 {month_name} {year}", errors="raise")
        if boundary == "start":
            return pd.Timestamp(start).normalize().tz_localize(None)
        return (pd.Timestamp(start) + pd.offsets.MonthEnd()).normalize().tz_localize(None)

    match = _MONTH_NUM_RE.match(lowered)
    if match:
        month, year = match.groups()
        start = pd.Timestamp(year=int(year), month=int(month), day=1)
        if boundary == "start":
            return start
        return (start + pd.offsets.MonthEnd()).normalize()

    # Generic parser fallback for full dates.
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        raise ValueError(f"Unsupported date format: {value}")
    return pd.Timestamp(parsed).normalize().tz_localize(None)


def _strip_date_prefix(text: str) -> str:
    """Remove common leading connector words from a date string.

    Handles cases like 'from from january 2023' -> 'january 2023'
    by iteratively stripping leading connector words.
    """
    text = text.strip()
    while True:
        new_text = re.sub(
            r"^(?:from|between|starting|beginning|the)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        if new_text == text:
            break
        text = new_text
    return text


def extract_date_range(text: str) -> tuple[str, str] | None:
    """Extract and normalize a date range from free-form user text.

    Returns ISO start/end date strings when successful.
    """
    raw = str(text or "").strip()
    if not raw:
        return None
    for pattern in _RANGE_PATTERNS:
        match = pattern.search(raw)
        if not match:
            continue
        start_raw = _strip_date_prefix(match.group("start").strip(" ."))
        end_raw = _strip_date_prefix(match.group("end").strip(" ."))
        start = parse_flexible_date(start_raw, "start")
        end = parse_flexible_date(end_raw, "end")
        if end < start:
            raise ValueError("End date must be on or after start date.")
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    return None


def normalize_date_range(start_date: str, end_date: str) -> tuple[str, str]:
    """Normalize start and end date inputs to ISO yyyy-mm-dd strings."""
    start = parse_flexible_date(start_date, "start")
    end = parse_flexible_date(end_date, "end")
    if end < start:
        raise ValueError("End date must be on or after start date.")
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def normalize_requested_dates(
    start_date: str | None = None,
    end_date: str | None = None,
    request: str | None = None,
) -> dict[str, str]:
    """Tool wrapper that normalizes date inputs into ISO boundaries.

    Accepts either explicit start/end fields or a free-form request string.
    """
    if request and (not start_date or not end_date):
        extracted = extract_date_range(request)
        if extracted is not None:
            start_date, end_date = extracted
    if not start_date or not end_date:
        raise ValueError(
            "Please provide both start_date and end_date, or provide request containing a date range."
        )
    start, end = normalize_date_range(start_date, end_date)
    return {"start_date": start, "end_date": end}


def extract_requested_date_range(
    request: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, str]:
    """Tool wrapper to extract/normalize date range from flexible inputs."""
    if start_date and end_date:
        start, end = normalize_date_range(start_date, end_date)
        return {"start_date": start, "end_date": end}

    extracted = extract_date_range(request or "")
    if extracted is None:
        raise ValueError(
            "Could not extract a date range. Use wording like 'from Jan 2023 to Mar 2023'."
        )
    start, end = extracted
    return {"start_date": start, "end_date": end}


def get_instrument_details(
    query: str | None = None,
    identifier: str = "auto",
    symbol: str | None = None,
) -> dict[str, Any]:
    """Resolve an instrument by symbol, ticker, short name, or full name.

    Args:
        query: The search query (ticker, symbol, or security name).
        identifier: How to interpret the query ('auto', 'ticker', 'symbol', 'name', 'security_name').
        symbol: Alternative parameter for the query.

    Returns:
        A dict with 'found' boolean and instrument details if resolved.
    """
    value_input = query or symbol or ""
    _debug_tool_event("get_instrument_details", "start", query=value_input, identifier=identifier)
    logger.info("tool_get_instrument_details query=%s identifier=%s", value_input, identifier)
    frame = pd.read_csv(DATA_DIR / "instruments.csv")
    value = value_input.strip().casefold()
    if not value:
        return {"found": False, "query": value_input, "message": "No instrument query was provided."}

    # Free-form requests may contain a ticker plus extra context (for example
    # "apple with AAPL ticker start ..."). Prefer exact symbol tokens first.
    symbol_values = frame["symbol"].astype(str).str.strip()
    symbol_lookup = {item.casefold(): item for item in symbol_values}
    token_candidates = re.findall(r"\b[A-Za-z]{1,8}(?:-[A-Za-z])?\b", value_input)
    for token in token_candidates:
        normalized_token = token.casefold()
        if normalized_token in symbol_lookup:
            matches = frame[symbol_values.str.casefold() == normalized_token]
            if not matches.empty:
                record = matches.iloc[0].to_dict()
                logger.info(
                    "instrument_resolved_from_token query=%s token=%s symbol=%s",
                    value_input,
                    token,
                    record.get("symbol"),
                )
                _debug_tool_event("get_instrument_details", "resolved", mode="token", symbol=record.get("symbol"))
                return {"found": True, "query": value_input, **record}

    # If query includes company-name fragments, match on meaningful word tokens.
    security_names = frame["security_name"].astype(str)
    query_tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-z]{3,}", value_input)
        if token.casefold() not in {
            "build",
            "create",
            "construct",
            "generate",
            "series",
            "time",
            "ticker",
            "stock",
            "financial",
            "start",
            "date",
            "end",
            "from",
            "for",
            "and",
            "with",
        }
    }
    if query_tokens:
        scores: list[tuple[int, int]] = []
        for row_index, security_name in enumerate(security_names):
            name = str(security_name).casefold()
            score = sum(1 for token in query_tokens if token in name)
            if score > 0:
                scores.append((score, row_index))
        if scores:
            _, best_index = max(scores, key=lambda item: item[0])
            record = frame.iloc[best_index].to_dict()
            logger.info(
                "instrument_resolved_from_name_tokens query=%s symbol=%s security_name=%s tokens=%s",
                value_input,
                record.get("symbol"),
                record.get("security_name"),
                sorted(query_tokens),
            )
            _debug_tool_event("get_instrument_details", "resolved", mode="name_tokens", symbol=record.get("symbol"))
            return {"found": True, "query": value_input, **record}

    ticker_candidate = (
        value_input.strip().upper() == value_input.strip()
        and value_input.strip().isalpha()
        and len(value_input.strip()) <= 6
    )
    columns = (
        ["symbol"]
        if identifier.casefold() == "auto" and ticker_candidate
        else ["symbol", "security_name"]
    )
    if identifier.casefold() in {"ticker", "symbol"}:
        columns = ["symbol"]
    elif identifier.casefold() in {"name", "asset", "security", "security_name"}:
        columns = ["security_name"]

    normalized = frame[columns].astype(str).apply(lambda col: col.str.strip().str.casefold())
    matches = frame[normalized.eq(value).any(axis=1)]
    if matches.empty:
        matches = frame[normalized.apply(lambda col: col.str.contains(value, regex=False)).any(axis=1)]
    if matches.empty:
        choices = [item for col in columns for item in frame[col].dropna().astype(str)]
        suggestions = get_close_matches(value_input, choices, n=3, cutoff=0.65)
        logger.warning(
            "instrument_not_found query=%s identifier=%s suggestions=%s",
            value_input, identifier, suggestions,
        )
        message = "Instrument was not found."
        if suggestions:
            message += f" Did you mean: {', '.join(suggestions)}?"
        _debug_tool_event("get_instrument_details", "not_found", query=value_input)
        return {"found": False, "query": value_input, "suggestions": suggestions, "message": message}
    record = matches.iloc[0].to_dict()
    logger.info(
        "instrument_resolved query=%s symbol=%s security_name=%s",
        value_input, record.get("symbol"), record.get("security_name"),
    )
    _debug_tool_event("get_instrument_details", "resolved", mode="direct", symbol=record.get("symbol"))
    return {"found": True, "query": value_input, **record}


def available_data_sources() -> list[str]:
    """List configured historical data sources."""
    logger.info("tool_available_data_sources count=%d", len(SOURCES))
    return list(SOURCES)


def historical_prices(symbol: str, start_date: str, end_date: str, source: str) -> dict[str, Any]:
    """Load historical prices for a ticker from a given source.

    Args:
        symbol: The ticker symbol.
        start_date: Start date (YYYY-MM-DD or parseable format).
        end_date: End date (YYYY-MM-DD or parseable format).
        source: Data source name ('yahoo', 'bloomberg', 'reuters').

    Returns:
        Dict with symbol, source, dates list, and prices list.
    """
    logger.info(
        "tool_historical_prices_start symbol=%s source=%s start=%s end=%s",
        symbol, source, start_date, end_date,
    )
    _debug_tool_event("historical_prices", "start", symbol=symbol, source=source, start=start_date, end=end_date)
    source = source.casefold()
    if source not in SOURCES:
        raise ValueError(f"Unsupported source: {source}")
    normalized_start, normalized_end = normalize_date_range(start_date, end_date)
    frame = pd.read_csv(DATA_DIR / f"{source}_stock_data.csv", index_col="Date", parse_dates=True)
    if symbol not in frame.columns:
        raise ValueError(f"Ticker {symbol} is not available in {source} data.")
    series = pd.to_numeric(frame[symbol], errors="coerce").sort_index()
    series = series.loc[pd.Timestamp(normalized_start):pd.Timestamp(normalized_end)]
    if series.empty:
        full_series = pd.to_numeric(frame[symbol], errors="coerce").sort_index()
        if full_series.empty:
            raise ValueError(
                f"No historical data is available for {symbol} from {normalized_start} to {normalized_end} in {source}."
            )

        requested_start = pd.Timestamp(normalized_start)
        requested_end = pd.Timestamp(normalized_end)
        index = full_series.index

        # Snap to the closest available dates when the requested window has no rows
        # (for example, weekends/holidays or out-of-range boundaries).
        left = max(0, min(index.searchsorted(requested_start, side="left"), len(index) - 1))
        right = max(0, min(index.searchsorted(requested_end, side="right") - 1, len(index) - 1))
        if right < left:
            if left >= len(index):
                left = len(index) - 1
            right = left

        series = full_series.iloc[left : right + 1]
        logger.warning(
            "tool_historical_prices_fallback_closest_dates symbol=%s source=%s requested_start=%s requested_end=%s actual_start=%s actual_end=%s observations=%d",
            symbol,
            source,
            normalized_start,
            normalized_end,
            series.index.min().strftime("%Y-%m-%d") if not series.empty else None,
            series.index.max().strftime("%Y-%m-%d") if not series.empty else None,
            len(series),
        )
        _debug_tool_event(
            "historical_prices",
            "fallback_closest_dates",
            symbol=symbol,
            source=source,
            actual_start=series.index.min().strftime("%Y-%m-%d") if not series.empty else None,
            actual_end=series.index.max().strftime("%Y-%m-%d") if not series.empty else None,
            observations=len(series),
        )
    logger.info(
        "tool_historical_prices_completed symbol=%s source=%s observations=%d missing=%d",
        symbol, source, len(series), int(series.isna().sum()),
    )
    _debug_tool_event("historical_prices", "completed", symbol=symbol, source=source, observations=len(series))
    return {
        "symbol": symbol,
        "source": source,
        "dates": [d.strftime("%Y-%m-%d") for d in series.index],
        "prices": [None if pd.isna(value) else float(value) for value in series],
    }


def check_data_quality(
    prices: list[Any] | None = None,
    dates: list[str] | None = None,
    source: str | None = None,
    symbol: str | None = None,
    data: dict[str, Any] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Calculate completeness and common price-quality metrics.

    Accepts either individual parameters (prices, source, symbol) or a
    single data dict from historical_prices output (with 'prices', 'source',
    'symbol' keys). If only source and symbol are provided (without prices),
    the tool will automatically fetch historical data using historical_prices.

    Args:
        prices: List of price values (may contain None for missing).
        source: Data source name.
        symbol: Ticker symbol.
        data: Optional dict from historical_prices output containing
              'prices', 'source', and 'symbol' keys.
        start_date: Start date for auto-fetching data (YYYY-MM-DD).
        end_date: End date for auto-fetching data (YYYY-MM-DD).

    Returns:
        Dict with quality metrics including missing_count, completeness_pct, issues.
    """
    # Support passing the full historical_prices result dict as a single argument
    if data is not None:
        prices = data.get("prices", prices)
        dates = data.get("dates", dates)
        source = data.get("source", source)
        symbol = data.get("symbol", symbol)

    # Auto-fetch data if only source and symbol are provided (no prices list)
    if prices is None and source is not None and symbol is not None:
        try:
            hp_result = historical_prices(
                symbol=symbol,
                start_date=start_date or "2010-01-01",
                end_date=end_date or "2025-12-31",
                source=source,
            )
            prices = hp_result.get("prices")
            dates = hp_result.get("dates", dates)
            logger.info(
                "tool_check_data_quality_auto_fetched symbol=%s source=%s observations=%d",
                symbol, source, len(prices) if prices else 0,
            )
        except Exception as error:
            raise ValueError(
                f"Could not auto-fetch data for {symbol} from {source}: {error}"
            )

    if prices is None or source is None or symbol is None:
        raise ValueError(
            "check_data_quality requires prices, source, and symbol. "
            "Pass them individually or provide a data dict from historical_prices output."
        )

    logger.info(
        "tool_check_data_quality_start symbol=%s source=%s observations=%d",
        symbol, source, len(prices),
    )
    _debug_tool_event("check_data_quality", "start", symbol=symbol, source=source, observations=len(prices))
    values = pd.Series(prices, dtype="float64")
    if dates is not None and len(dates) == len(values):
        date_index = pd.to_datetime(pd.Series(dates, dtype="string"), errors="coerce")
    else:
        date_index = pd.Series([pd.NaT] * len(values), dtype="datetime64[ns]")
    missing = int(values.isna().sum())
    available = int(values.notna().sum())
    non_positive = int((values.dropna() <= 0).sum())
    min_value = float(values.min(skipna=True)) if available else None
    max_value = float(values.max(skipna=True)) if available else None

    # Use date index only where a price observation exists.
    observed_dates = date_index[values.notna()] if len(date_index) == len(values) else pd.Series(dtype="datetime64[ns]")
    observed_dates = observed_dates.dropna()
    min_date = observed_dates.min().strftime("%Y-%m-%d") if not observed_dates.empty else None
    max_date = observed_dates.max().strftime("%Y-%m-%d") if not observed_dates.empty else None

    issues = []
    if missing:
        issues.append("missing_or_nan_values")
    if non_positive:
        issues.append("non_positive_prices")
    result = {
        "source": source,
        "symbol": symbol,
        "total_values": len(values),
        "available_record_count": available,
        "missing_count": missing,
        "nan_count": missing,
        "completeness_pct": round((1 - missing / len(values)) * 100, 2) if len(values) else 0.0,
        "min_value": min_value,
        "max_value": max_value,
        "min_date": min_date,
        "max_date": max_date,
        "duplicate_count": 0,
        "issues": issues,
    }
    logger.info(
        "tool_check_data_quality_completed symbol=%s source=%s missing=%d issues=%d",
        symbol, source, missing, len(issues),
    )
    _debug_tool_event(
        "check_data_quality",
        "completed",
        symbol=symbol,
        source=source,
        available=available,
        missing=missing,
    )
    return result


def recommend_gap_methods(quality_report: dict[str, Any], prices: dict[str, Any]) -> list[str]:
    """Recommend methods for missing observations based on quality report.

    Args:
        quality_report: Output from check_data_quality.
        prices: Output from historical_prices.

    Returns:
        List of recommended gap-filling method names.
    """
    methods = (
        ["linear_interpolation", "forward_fill", "backward_fill"]
        if quality_report.get("missing_count")
        else ["none"]
    )
    logger.info("tool_gap_methods_recommended symbol=%s methods=%s", prices.get("symbol"), methods)
    return methods


def apply_gap_filling(
    prices: dict[str, Any],
    method: str,
    dates: list[str] | None = None,
) -> dict[str, Any]:
    """Apply a supported gap-filling method to price data.

    Args:
        prices: Output from historical_prices with 'prices' and 'dates' keys.
        method: One of 'linear_interpolation', 'forward_fill', 'backward_fill', 'none'.
        dates: Optional override for date index.

    Returns:
        Dict with filled prices, dates, and method metadata.
    """
    logger.info("tool_gap_filling_start symbol=%s method=%s", prices.get("symbol"), method)
    _debug_tool_event("apply_gap_filling", "start", symbol=prices.get("symbol"), method=method)
    source_dates = [str(value) for value in (dates or prices["dates"])]
    normalized_dates = [parse_flexible_date(value, "start").strftime("%Y-%m-%d") for value in source_dates]
    original_prices = list(prices["prices"])
    series = pd.Series(
        original_prices,
        index=pd.to_datetime(normalized_dates),
        dtype="float64",
    )
    if method == "linear_interpolation":
        filled = series.interpolate(method="time").ffill().bfill()
    elif method == "forward_fill":
        filled = series.ffill()
    elif method == "backward_fill":
        filled = series.bfill()
    elif method == "none":
        filled = series
    else:
        raise ValueError(f"Unsupported gap method: {method}")
    result = {
        "symbol": prices["symbol"],
        "source": prices.get("source"),
        "method": method,
        "original_dates": source_dates,
        "original_prices": original_prices,
        "filled_dates": [d.strftime("%Y-%m-%d") for d in filled.index],
        "filled_prices": [None if pd.isna(value) else float(value) for value in filled],
        "dates": [d.strftime("%Y-%m-%d") for d in filled.index],
        "prices": [None if pd.isna(value) else float(value) for value in filled],
    }
    logger.info(
        "tool_gap_filling_completed symbol=%s method=%s observations=%d remaining_missing=%d",
        prices.get("symbol"), method, len(filled), int(filled.isna().sum()),
    )
    _debug_tool_event("apply_gap_filling", "completed", symbol=prices.get("symbol"), method=method, observations=len(filled))
    return result


def build_timeseries(
    series: dict[str, Any],
    filename: str = "final_timeseries.csv",
    run_id: str | None = None,
) -> str:
    """Persist a final time series CSV artifact.

    Args:
        series: Dict with 'dates' and 'prices' keys.
        filename: Output filename.
        run_id: Optional run identifier for directory structure.

    Returns:
        Path to the saved CSV file.
    """
    output_dates = series.get("filled_dates") or series.get("dates") or []
    output_prices = series.get("filled_prices") or series.get("prices") or []
    logger.info("tool_build_timeseries_start symbol=%s filename=%s prices_length=%d", series.get("symbol"), filename, len(output_prices))
    _debug_tool_event("build_timeseries", "start", symbol=series.get("symbol"), filename=filename, run_id=run_id)
    output = _run_dir(run_id) / filename
    frame_data: dict[str, Any] = {
        "date": output_dates,
        "price": output_prices,
    }
    if series.get("source"):
        frame_data["source"] = [series.get("source")] * len(output_dates)
    if series.get("method"):
        frame_data["gap_filling_method"] = [series.get("method")] * len(output_dates)
    pd.DataFrame(frame_data).to_csv(output, index=False)
    logger.info("tool_build_timeseries_completed path=%s", output)
    _debug_tool_event("build_timeseries", "completed", path=output)
    return str(output)


def generate_report(
    data: dict[str, Any] | list[dict[str, Any]],
    filename: str = "quality_report.csv",
    run_id: str | None = None,
) -> str:
    """Persist a CSV quality report artifact.

    Args:
        data: Dict or list of dicts with quality metrics.
        filename: Output filename.
        run_id: Optional run identifier.

    Returns:
        Path to the saved CSV file.
    """
    logger.info("tool_generate_report_start filename=%s", filename)
    output = _run_dir(run_id) / filename
    pd.DataFrame(data if isinstance(data, list) else [data]).to_csv(output, index=False)
    logger.info("tool_generate_report_completed path=%s", output)
    return str(output)


def visualize_timeseries(
    prices: dict[str, Any],
    title: str = "Time series",
    run_id: str | None = None,
) -> str:
    """Create a seaborn time series chart and save as PNG.

    Args:
        prices: Dict with 'dates' and 'prices' keys.
        title: Chart title.
        run_id: Optional run identifier.

    Returns:
        Path to the saved PNG file.
    """
    logger.info("tool_visualize_timeseries_start symbol=%s title=%s", prices.get("symbol"), title)
    output = _run_dir(run_id) / "timeseries.png"
    filled_dates = prices.get("filled_dates") or prices.get("dates") or []
    filled_prices = prices.get("filled_prices") or prices.get("prices") or []
    filled_frame = pd.DataFrame({
        "date": pd.to_datetime(filled_dates),
        "price": pd.to_numeric(pd.Series(filled_prices), errors="coerce"),
    })
    original_frame: pd.DataFrame | None = None
    if prices.get("original_dates") and prices.get("original_prices"):
        original_frame = pd.DataFrame({
            "date": pd.to_datetime(prices["original_dates"]),
            "price": pd.to_numeric(pd.Series(prices["original_prices"]), errors="coerce"),
        })
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(11, 5))
    if original_frame is not None:
        axis.plot(
            original_frame["date"],
            original_frame["price"],
            label="Before gap filling",
            color="0.7",
            linestyle="--",
            linewidth=1.4,
            alpha=0.95,
        )
    axis.plot(
        filled_frame["date"],
        filled_frame["price"],
        label="After gap filling",
        color="tab:blue",
        linewidth=2.0,
    )
    if original_frame is not None:
        aligned = filled_frame.merge(
            original_frame.rename(columns={"price": "original_price"}),
            on="date",
            how="left",
        )
        gap_mask = aligned["original_price"].isna() & aligned["price"].notna()
        if gap_mask.any():
            axis.scatter(
                aligned.loc[gap_mask, "date"],
                aligned.loc[gap_mask, "price"],
                label="Gap filled",
                color="tab:orange",
                s=28,
                zorder=4,
            )
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Price")
    if original_frame is not None:
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)
    logger.info("tool_visualize_timeseries_completed path=%s", output)
    return str(output)


def delegate_to_agent(agent_name: str, request: str) -> dict[str, str]:
    """Delegate work to a named specialist agent.

    Args:
        agent_name: Name of the target agent.
        request: The request/prompt to pass to the target agent.

    Returns:
        Dict with delegation status.
    """
    _log_tool_progress(
        "delegate_to_agent",
        "completed",
        agent_name=agent_name,
        request_chars=len(request or ""),
    )
    return {"status": "delegating", "agent_name": agent_name, "request": request}


def request_human_input(
    prompt: str,
    options: list[str] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Request input from the human user.

    Args:
        prompt: The question or prompt to display.
        options: Optional list of valid choices.
        context: Optional additional context data.

    Returns:
        Dict with the prompt and options for the human-in-the-loop handler.
    """
    _log_tool_progress(
        "request_human_input",
        "completed",
        options=len(options or []),
        has_context=bool(context),
        prompt_chars=len(prompt or ""),
    )
    result: dict[str, Any] = {"prompt": prompt, "requires_input": True}
    if options:
        result["options"] = options
    if context:
        result["context"] = context
    return result


def _tool(function: Any, name: str, description: str) -> StructuredTool:
    """Wrap a function as a langchain StructuredTool."""
    return StructuredTool.from_function(func=function, name=name, description=description)


TOOL_REGISTRY: dict[str, StructuredTool] = {
    "get_instrument_details": _tool(
        get_instrument_details,
        "get_instrument_details",
        "Resolve a ticker or security name from the instrument catalog.",
    ),
    "available_data_sources": _tool(
        available_data_sources,
        "available_data_sources",
        "List configured historical data sources (yahoo, bloomberg, reuters).",
    ),
    "historical_prices": _tool(
        historical_prices,
        "historical_prices",
        "Load a ticker's historical prices for a date range from a specific source.",
    ),
    "check_data_quality": _tool(
        check_data_quality,
        "check_data_quality",
        "Calculate completeness and common price-quality metrics for a series.",
    ),
    "recommend_gap_methods": _tool(
        recommend_gap_methods,
        "recommend_gap_methods",
        "Recommend gap-filling methods based on data quality report.",
    ),
    "normalize_requested_dates": _tool(
        normalize_requested_dates,
        "normalize_requested_dates",
        "Normalize start and end dates into yyyy-mm-dd, supporting word and numeric formats.",
    ),
    "extract_requested_date_range": _tool(
        extract_requested_date_range,
        "extract_requested_date_range",
        "Extract a date range from free-form text such as 'between January 2023 and December 2023'.",
    ),
    "apply_gap_filling": _tool(
        apply_gap_filling,
        "apply_gap_filling",
        "Apply a supported gap-filling method (linear_interpolation, forward_fill, backward_fill).",
    ),
    "build_timeseries": _tool(
        build_timeseries,
        "build_timeseries",
        "Persist a final time series CSV artifact to the output directory.",
    ),
    "generate_report": _tool(
        generate_report,
        "generate_report",
        "Persist a CSV quality report artifact to the output directory.",
    ),
    "visualize_timeseries": _tool(
        visualize_timeseries,
        "visualize_timeseries",
        "Create a seaborn time series chart and save as PNG.",
    ),
    "delegate_to_agent": _tool(
        delegate_to_agent,
        "delegate_to_agent",
        "Delegate work to a named specialist agent.",
    ),
    "request_human_input": _tool(
        request_human_input,
        "request_human_input",
        "Request input from the human user with optional choices.",
    ),
}


def get_tool(name: str) -> StructuredTool | None:
    """Retrieve a tool by name from the registry."""
    return TOOL_REGISTRY.get(name)


def get_tool_description(name: str) -> str | None:
    """Return a tool description from the registry if available."""
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return None
    return str(getattr(tool, "description", "") or "").strip() or None