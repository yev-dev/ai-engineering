"""Deterministic domain tools exposed to ReAct agents via langchain annotations."""
from __future__ import annotations

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

_MONTH_WORD_RE = re.compile(
    r"^(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\s+(\d{4})$",
    re.IGNORECASE,
)
_MONTH_NUM_RE = re.compile(r"^(\d{1,2})[/-](\d{4})$")
_YEAR_RE = re.compile(r"^\d{4}$")
_QUARTER_RE = re.compile(r"^(?:q([1-4])\s*(\d{4})|(\d{4})\s*q([1-4]))$", re.IGNORECASE)
_RANGE_PATTERNS = (
    re.compile(
        r"\bfrom\s+(?P<start>.+?)\s+(?:to|until|through|thru|till)\s+(?P<end>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bbetween\s+(?P<start>.+?)\s+(?:and|to|until|through|thru|till)\s+(?P<end>.+)$",
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
        return {"found": False, "query": value_input, "suggestions": suggestions, "message": message}
    record = matches.iloc[0].to_dict()
    logger.info(
        "instrument_resolved query=%s symbol=%s security_name=%s",
        value_input, record.get("symbol"), record.get("security_name"),
    )
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
        raise ValueError(
            f"No historical data is available for {symbol} from {normalized_start} to {normalized_end} in {source}."
        )
    logger.info(
        "tool_historical_prices_completed symbol=%s source=%s observations=%d missing=%d",
        symbol, source, len(series), int(series.isna().sum()),
    )
    return {
        "symbol": symbol,
        "source": source,
        "dates": [d.strftime("%Y-%m-%d") for d in series.index],
        "prices": [None if pd.isna(value) else float(value) for value in series],
    }


def check_data_quality(
    prices: list[Any] | None = None,
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
    values = pd.Series(prices, dtype="float64")
    missing = int(values.isna().sum())
    non_positive = int((values.dropna() <= 0).sum())
    issues = []
    if missing:
        issues.append("missing_or_nan_values")
    if non_positive:
        issues.append("non_positive_prices")
    result = {
        "source": source,
        "symbol": symbol,
        "total_values": len(values),
        "missing_count": missing,
        "nan_count": missing,
        "completeness_pct": round((1 - missing / len(values)) * 100, 2) if len(values) else 0.0,
        "duplicate_count": 0,
        "issues": issues,
    }
    logger.info(
        "tool_check_data_quality_completed symbol=%s source=%s missing=%d issues=%d",
        symbol, source, missing, len(issues),
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
    normalized_dates = [parse_flexible_date(str(value), "start").strftime("%Y-%m-%d") for value in (dates or prices["dates"])]
    series = pd.Series(
        prices["prices"],
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
        "method": method,
        "dates": [d.strftime("%Y-%m-%d") for d in filled.index],
        "prices": [None if pd.isna(value) else float(value) for value in filled],
    }
    logger.info(
        "tool_gap_filling_completed symbol=%s method=%s observations=%d remaining_missing=%d",
        prices.get("symbol"), method, len(filled), int(filled.isna().sum()),
    )
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
    logger.info("tool_build_timeseries_start symbol=%s filename=%s", series.get("symbol"), filename)
    output = _run_dir(run_id) / filename
    pd.DataFrame({"date": series["dates"], "price": series["prices"]}).to_csv(output, index=False)
    logger.info("tool_build_timeseries_completed path=%s", output)
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
    frame = pd.DataFrame({"date": pd.to_datetime(prices["dates"]), "price": prices["prices"]})
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=frame, x="date", y="price", ax=axis)
    axis.set_title(title)
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