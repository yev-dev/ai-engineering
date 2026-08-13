"""Deterministic domain tools exposed to ReAct agents via langchain annotations."""
from __future__ import annotations

import logging
import re
import uuid
from datetime import timedelta
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import matplotlib

# Use a non-interactive backend so that figures can be created from worker
# threads.  The default macOS backend ("macosx") requires the main thread and
# raises ``RuntimeError: Cannot create a GUI FigureManager outside the main
# thread`` when tools are invoked from a ReAct agent's thread pool.
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
import pandas as pd
import seaborn as sns
from langchain_core.tools import StructuredTool

from market_data_ai.configuration import get_config
from market_data_ai.database import (
    DataStore,
    get_datastore,
    init_datastore,
)

ROOT = Path(__file__).resolve().parent
_config = get_config()
DATA_DIR = _config.paths.data_dir
OUTPUT_ROOT = _config.paths.output_dir
DATABASE_DIR = _config.paths.database_dir
SOURCES = tuple(_config.data_sources)
logger = logging.getLogger(__name__)

# ── DataStore integration ────────────────────────────────────────────────
# The processor injects the run_id via set_run_id() before invoking any tool.
# This allows tools to store/load time series data without the LLM needing to
# pass run_id, and without serializing full payloads into LLM conversation.
# The DataStore is a global singleton created under the
# ``time_series_database/database/`` subfolder of the output root.
_current_run_id: str | None = None


def set_run_id(run_id: str) -> None:
    """Set the current run_id for DataStore operations.

    Called by the processor before invoking tools so that time series data
    can be stored/loaded from the correct database.

    Args:
        run_id: The run/session identifier.
    """
    global _current_run_id
    _current_run_id = run_id
    # Ensure DataStore is initialised with the correct output root.
    # Idempotent: subsequent calls with the same OUTPUT_ROOT are no-ops.
    try:
        get_datastore()
    except RuntimeError:
        init_datastore(DATABASE_DIR / "datastore.db")


def _get_data_store() -> DataStore:
    """Return the global DataStore singleton.

    Returns:
        The global DataStore instance (created under ``time_series_database/database/``).

    Raises:
        RuntimeError: If ``set_run_id()`` has not been called.
    """
    if _current_run_id is None:
        raise RuntimeError(
            "run_id not set. Call set_run_id() before using tools "
            "that require DataStore."
        )
    try:
        return get_datastore()
    except RuntimeError:
        init_datastore(DATABASE_DIR / "datastore.db")
        return get_datastore()


def _resolve_timeseries(
    prices: dict[str, Any] | None = None,
    data_ref: str | None = None,
    symbol: str | None = None,
    source: str | None = None,
    method: str | None = None,
    prefer_filled: bool = True,
) -> dict[str, Any]:
    """Resolve a time series from explicit prices, data_ref, or identifiers.

    Priority:
    1. ``prices`` dict (backward compatibility) – used as-is.
    2. ``data_ref`` – loaded from the DataStore (supports both raw and
       gap-filled references).
    3. ``symbol`` + ``source`` – looked up in the DataStore for the current
       run. When *prefer_filled* is True, gap-filled series are tried first,
       then the raw series.

    The time series data is always loaded from the database – never from the
    LLM-provided conversation.

    Args:
        prices: Backward-compat inline payload with 'dates'/'prices'.
        data_ref: Reference key from the DataStore.
        symbol: Ticker symbol.
        source: Data source name.
        method: Gap-filling method (used for filled lookups).
        prefer_filled: When True (and symbol+source are used), try
            gap-filled series first, then raw.

    Returns:
        A dict with ``symbol``, ``source``, ``dates``, ``prices`` plus any
        method/filled metadata when a filled series is loaded.

    Raises:
        ValueError: If no series can be resolved.
    """
    if prices is not None and isinstance(prices, dict):
        return prices

    store = None
    try:
        store = _get_data_store()
    except RuntimeError:
        pass

    if data_ref is not None and store is not None:
        try:
            if str(data_ref).endswith(":filled"):
                return store.get_gap_filled_series(data_ref)
            return store.get_timeseries(data_ref)
        except KeyError:
            pass

    if store is not None and symbol and source:
        run_id = _current_run_id or ""
        if run_id:
            if prefer_filled:
                try:
                    filled_list = store.list_gap_filled_series(
                        run_id, symbol=symbol, source=source
                    )
                    if filled_list:
                        item = filled_list[0]
                        item_ref = item.get("data_ref")
                        if item_ref:
                            loaded = store.get_gap_filled_series(item_ref)
                            if loaded and loaded.get("filled_dates") and loaded.get("filled_prices"):
                                logger.info(
                                    "series_resolved_from_store mode=filled symbol=%s source=%s method=%s",
                                    symbol, source, loaded.get("method"),
                                )
                                return loaded
                except KeyError:
                    pass
            try:
                raw_list = store.list_timeseries(run_id, symbol=symbol, source=source)
                if raw_list:
                    item = raw_list[0]
                    item_ref = item.get("data_ref")
                    if item_ref:
                        loaded = store.get_timeseries(item_ref)
                        logger.info(
                            "series_resolved_from_store mode=raw symbol=%s source=%s observations=%d",
                            symbol, source, len(loaded.get("dates", [])),
                        )
                        return loaded
            except KeyError:
                pass

    raise ValueError(
        "Could not resolve time series data from the database. Provide "
        "'prices', 'data_ref', or 'symbol' + 'source'."
    )


def _build_summary(series: pd.Series) -> dict[str, Any]:
    """Build a compact summary dict from a pandas Series.

    Args:
        series: A pandas Series of price values indexed by date.

    Returns:
        A dict with count, start_date, end_date, available, missing, min, max.
    """
    return {
        "count": len(series),
        "start_date": series.index.min().strftime("%Y-%m-%d") if len(series) else None,
        "end_date": series.index.max().strftime("%Y-%m-%d") if len(series) else None,
        "available": int(series.notna().sum()),
        "missing": int(series.isna().sum()),
        "min": float(series.min(skipna=True)) if series.notna().any() else None,
        "max": float(series.max(skipna=True)) if series.notna().any() else None,
    }


def _debug_flow_enabled() -> bool:
    """Return True when lightweight flow debugging is enabled."""
    return bool(get_config().runtime.debug_flow)


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

    # Store the full time series in the DataStore so downstream tools can
    # load it via data_ref without the LLM needing to see the full payload.
    dates_list = [d.strftime("%Y-%m-%d") for d in series.index]
    prices_list = [None if pd.isna(value) else float(value) for value in series]
    data_ref = _get_data_store().put_timeseries(
        run_id=_current_run_id or "",
        symbol=symbol,
        source=source,
        dates=dates_list,
        prices=prices_list,
    )
    return {
        "symbol": symbol,
        "source": source,
        "data_ref": data_ref,
        "summary": _build_summary(series),
        # Keep dates/prices for backward compatibility with tools that
        # receive inline data. The processor strips these before sending
        # to the LLM to avoid token bloat.
        "dates": dates_list,
        "prices": prices_list,
    }


def check_data_quality(
    prices: list[Any] | None = None,
    dates: list[str] | None = None,
    source: str | None = None,
    symbol: str | None = None,
    data: dict[str, Any] | None = None,
    data_ref: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Calculate completeness and common price-quality metrics.

    Accepts either individual parameters (prices, source, symbol) or a
    single data dict from historical_prices output (with 'prices', 'source',
    'symbol' keys). If only source and symbol are provided (without prices),
    the tool will automatically fetch historical data using historical_prices.
    Alternatively, pass a data_ref to load data from the DataStore.

    Args:
        prices: List of price values (may contain None for missing).
        source: Data source name.
        symbol: Ticker symbol.
        data: Optional dict from historical_prices output containing
              'prices', 'source', and 'symbol' keys.
        data_ref: Optional reference key to load data from the DataStore.
        start_date: Start date for auto-fetching data (YYYY-MM-DD).
        end_date: End date for auto-fetching data (YYYY-MM-DD).

    Returns:
        Dict with quality metrics including missing_count, completeness_pct, issues.
    """
    # Load from DataStore if data_ref is provided
    if data_ref is not None and data is None:
        try:
            data = _get_data_store().get_timeseries(data_ref)
            logger.info(
                "tool_check_data_quality_loaded_from_store data_ref=%s symbol=%s source=%s",
                data_ref, data.get("symbol"), data.get("source"),
            )
        except KeyError as error:
            raise ValueError(f"Could not load data from DataStore: {error}")

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
    prices: dict[str, Any] | None = None,
    method: str = "linear_interpolation",
    dates: list[str] | None = None,
    data_ref: str | None = None,
    symbol: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Apply a supported gap-filling method to price data.

    The time series is always loaded from the database – never from the
    LLM.  Accepts either a ``prices`` dict (backward compatibility), a
    ``data_ref``, or ``symbol`` + ``source`` identifiers to look up the
    series in the DataStore for the current run.

    Args:
        prices: Backward-compat output from historical_prices with 'prices' and 'dates' keys.
        method: One of 'linear_interpolation', 'forward_fill', 'backward_fill', 'none'.
        dates: Optional override for date index.
        data_ref: Reference key to load prices from the DataStore.
        symbol: Ticker symbol (used with ``source`` to look up the series
            from the database).
        source: Data source name (used with ``symbol``).

    Returns:
        Dict with filled prices, dates, method metadata, and a ``data_ref``
        if the result was stored in the DataStore.
    """
    # Resolve the input series from the database. The LLM only needs to pass
    # identifiers (data_ref or symbol+source) – never the full time series.
    prices = _resolve_timeseries(
        prices=prices,
        data_ref=data_ref,
        symbol=symbol,
        source=source,
        method=method,
        prefer_filled=False,
    )
    if "dates" not in prices or "prices" not in prices:
        raise ValueError(
            "apply_gap_filling requires a series with 'dates' and 'prices'. "
            "Provide 'data_ref' or 'symbol' + 'source' so data can be loaded "
            "from the database."
        )

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

    filled_dates = [d.strftime("%Y-%m-%d") for d in filled.index]
    filled_prices = [None if pd.isna(value) else float(value) for value in filled]

    result = {
        "symbol": prices["symbol"],
        "source": prices.get("source"),
        "method": method,
        "original_dates": source_dates,
        "original_prices": original_prices,
        "filled_dates": filled_dates,
        "filled_prices": filled_prices,
        "dates": filled_dates,
        "prices": filled_prices,
    }

    # Store the gap-filled result in the DataStore so downstream tools can
    # load it via data_ref without the LLM needing to see the full payload.
    try:
        store = _get_data_store()
        filled_data_ref = store.put_gap_filled_series(
            run_id=_current_run_id or "",
            symbol=str(prices.get("symbol", "UNKNOWN")),
            source=str(prices.get("source", "unknown")),
            method=method,
            original_dates=source_dates,
            original_prices=original_prices,
            filled_dates=filled_dates,
            filled_prices=filled_prices,
            original_data_ref=data_ref,
        )
        result["data_ref"] = filled_data_ref
        logger.debug(
            "gap_filled_series_stored data_ref=%s symbol=%s method=%s",
            filled_data_ref, prices.get("symbol"), method,
        )
    except Exception as error:
        logger.warning("gap_filled_series_store_failed error=%s", error)

    logger.info(
        "tool_gap_filling_completed symbol=%s method=%s observations=%d remaining_missing=%d",
        prices.get("symbol"), method, len(filled), int(filled.isna().sum()),
    )
    _debug_tool_event("apply_gap_filling", "completed", symbol=prices.get("symbol"), method=method, observations=len(filled))
    return result


def build_timeseries(
    series: dict[str, Any] | None = None,
    filename: str = "final_timeseries.csv",
    run_id: str | None = None,
    data_ref: str | None = None,
    symbol: str | None = None,
    source: str | None = None,
) -> str:
    """Persist a final time series CSV artifact.

    The time series is always loaded from the database – never from the
    LLM.  Accepts either a ``series`` dict (backward compatibility), a
    ``data_ref``, or ``symbol`` + ``source`` identifiers to look up the
    series in the DataStore for the current run.

    Args:
        series: Backward-compat dict with 'dates' and 'prices' keys.
        filename: Output filename.
        run_id: Optional run identifier for directory structure.
        data_ref: Reference key to load series from DataStore.
        symbol: Ticker symbol (used with ``source``).
        source: Data source name (used with ``symbol``).

    Returns:
        Path to the saved CSV file.
    """
    # Resolve the input series from the database. The LLM only needs to pass
    # identifiers (data_ref or symbol+source) – never the full time series.
    loaded_series = _resolve_timeseries(
        prices=series,
        data_ref=data_ref,
        symbol=symbol,
        source=source,
        prefer_filled=True,
    )
    if "dates" not in loaded_series and "filled_dates" not in loaded_series:
        raise ValueError(
            "build_timeseries requires a series with dates/prices. "
            "Provide 'data_ref' or 'symbol' + 'source'."
        )
    if loaded_series is not series:
        series = loaded_series

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

    # Record artifact in DataStore
    try:
        _get_data_store().put_artifact(
            run_id=run_id or _current_run_id or "",
            artifact_type="csv",
            path=str(output),
            symbol=series.get("symbol"),
            source=series.get("source"),
        )
    except Exception as error:
        logger.warning("artifact_store_failed path=%s error=%s", output, error)

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
    prices: dict[str, Any] | None = None,
    title: str = "Time series",
    run_id: str | None = None,
    data_ref: str | None = None,
    symbol: str | None = None,
    source: str | None = None,
) -> str:
    """Create a seaborn time series chart and save as PNG.

    The time series is always loaded from the database – never from the
    LLM.  Accepts either a ``prices`` dict (backward compatibility), a
    ``data_ref``, or ``symbol`` + ``source`` identifiers to look up the
    series in the DataStore for the current run.

    Args:
        prices: Backward-compat dict with 'dates' and 'prices' keys.
        title: Chart title.
        run_id: Optional run identifier.
        data_ref: Reference key to load series from the DataStore.
        symbol: Ticker symbol (used with ``source`` to look up the series).
        source: Data source name (used with ``symbol``).

    Returns:
        Path to the saved PNG file.
    """
    prices = _resolve_timeseries(
        prices=prices,
        data_ref=data_ref,
        symbol=symbol,
        source=source,
        prefer_filled=True,
    )
    if "dates" not in prices and "filled_dates" not in prices:
        raise ValueError(
            "visualize_timeseries requires a series with dates/prices. "
            "Provide 'data_ref' or 'symbol' + 'source'."
        )
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


def get_populated_timeseries(
    run_id: str | None = None,
    symbol: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Load populated time series data for a run from the DataStore.

    Sources data from both the ``raw_timeseries`` (populated before) and
    ``filled_timeseries`` (populated after) tables.  Returns a comparison
    view with both the original (pre-fill) series and the gap-filled
    (post-fill) series.

    Args:
        run_id: The run/session identifier.  Defaults to the current run
            set by ``set_run_id()``.
        symbol: Optional ticker symbol filter.
        source: Optional data source name filter.

    Returns:
        Dict with ``run_id``, ``symbols``, ``sources``, ``before``
        and ``after`` series lists.  Each series entry contains
        ``symbol``, ``source``, ``data_ref``, ``dates``, ``prices``
        (and ``method`` / ``filled_dates`` / ``filled_prices`` for
        the after table).

    Raises:
        ValueError: If no populated series are found.
    """
    store = _get_data_store()
    run_id = run_id or _current_run_id
    if not run_id:
        raise ValueError("No run_id available; pass run_id or call set_run_id().")

    logger.info(
        "tool_get_populated_timeseries_start run_id=%s symbol=%s source=%s",
        run_id, symbol or "*", source or "*",
    )
    _debug_tool_event(
        "get_populated_timeseries", "start",
        run_id=run_id, symbol=symbol or "*", source=source or "*",
    )

    before_series = store.list_timeseries(
        run_id, symbol=symbol, source=source
    )
    after_series = store.list_gap_filled_series(
        run_id, symbol=symbol, source=source
    )

    # Normalise "before" entries: list_timeseries already returns
    # data_ref/symbol/source/dates/prices.
    before: list[dict[str, Any]] = []
    for item in before_series:
        loaded = store.get_timeseries(item["data_ref"])
        before.append({
            "symbol": loaded.get("symbol"),
            "source": loaded.get("source"),
            "data_ref": item["data_ref"],
            "dates": loaded.get("dates", []),
            "prices": loaded.get("prices", []),
        })
        logger.debug(
            "populated_before_series symbol=%s source=%s observations=%d",
            loaded.get("symbol"), loaded.get("source"), len(loaded.get("dates", [])),
        )

    # Normalise "after" entries: list_gap_filled_series returns
    # filled_dates/filled_prices but we also want original_dates/prices.
    after: list[dict[str, Any]] = []
    for item in after_series:
        loaded = store.get_gap_filled_series(item["data_ref"])
        after.append({
            "symbol": loaded.get("symbol"),
            "source": loaded.get("source"),
            "data_ref": item["data_ref"],
            "method": loaded.get("method"),
            "original_dates": loaded.get("original_dates", []),
            "original_prices": loaded.get("original_prices", []),
            "filled_dates": loaded.get("filled_dates", []),
            "filled_prices": loaded.get("filled_prices", []),
        })
        logger.debug(
            "populated_after_series symbol=%s source=%s method=%s observations=%d",
            loaded.get("symbol"), loaded.get("source"), loaded.get("method"),
            len(loaded.get("filled_dates", [])),
        )

    if not before and not after:
        raise ValueError(
            f"No populated time series found for run_id={run_id!r} "
            f"symbol={symbol!r} source={source!r}. "
            "Run a workflow first to populate the DataStore."
        )

    symbols = sorted({item["symbol"] for item in before + after if item.get("symbol")})
    sources = sorted({item["source"] for item in before + after if item.get("source")})

    result = {
        "run_id": run_id,
        "symbols": symbols,
        "sources": sources,
        "before": before,
        "after": after,
    }
    logger.info(
        "tool_get_populated_timeseries_completed run_id=%s before=%d after=%d",
        run_id, len(before), len(after),
    )
    _debug_tool_event(
        "get_populated_timeseries", "completed",
        run_id=run_id, before=len(before), after=len(after),
    )
    return result


def find_run_by_id(run_id: str) -> dict[str, Any]:
    """Find a run record in the DataStore by its ``run_id``.

    Args:
        run_id: The run/session identifier to look up.

    Returns:
        Dict with ``found`` boolean plus run metadata when resolved:
        ``run_id``, ``start_date``, ``end_date``, ``created_at``,
        ``updated_at``, ``timeseries_count``, ``filled_count``,
        ``artifact_count``.

    Raises:
        ValueError: If the DataStore is not initialised (``set_run_id()``
            has not been called).
    """
    store = _get_data_store()

    logger.info("tool_find_run_by_id_start run_id=%s", run_id)
    _debug_tool_event("find_run_by_id", "start", run_id=run_id)

    try:
        run_record = store.get_run(run_id)
    except KeyError:
        logger.warning("tool_find_run_by_id_not_found run_id=%s", run_id)
        _debug_tool_event("find_run_by_id", "not_found", run_id=run_id)
        return {"found": False, "run_id": run_id}

    logger.info("tool_find_run_by_id_found run_id=%s", run_id)
    _debug_tool_event("find_run_by_id", "found", run_id=run_id)
    return {"found": True, **run_record}


# ── DataStore tools ────────────────────────────────────────────────────────
# These tools expose the DataStore API (database.py) to ReAct agents.  They
# are registered in DATASTORE_TOOL_REGISTRY and can be used independently of
# the domain tools in TOOL_REGISTRY.  All tools operate on the global
# DataStore singleton and require set_run_id() to have been called first.


def datastore_put_run_metadata(
    run_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Register or update a run with optional date metadata in the DataStore.

    Args:
        run_id: The run/session identifier.  Defaults to the current run
            set by ``set_run_id()``.
        start_date: Optional ISO start date.
        end_date: Optional ISO end date.

    Returns:
        Dict with ``run_id`` and the numeric ``run_id_num`` of the row.
    """
    store = _get_data_store()
    run_id = run_id or _current_run_id
    if not run_id:
        raise ValueError("No run_id available; pass run_id or call set_run_id().")
    run_id_num = store.put_run_metadata(run_id, start_date=start_date, end_date=end_date)
    logger.info("datastore_put_run_metadata run_id=%s run_id_num=%d", run_id, run_id_num)
    _debug_tool_event("datastore_put_run_metadata", "completed", run_id=run_id)
    return {"run_id": run_id, "run_id_num": run_id_num}


def datastore_get_run(run_id: str | None = None) -> dict[str, Any]:
    """Return a single run record from the DataStore by its ``run_id``.

    Args:
        run_id: The run/session identifier.  Defaults to the current run
            set by ``set_run_id()``.

    Returns:
        Dict with ``found`` boolean plus run metadata when resolved:
        ``run_id``, ``start_date``, ``end_date``, ``created_at``,
        ``updated_at``, ``timeseries_count``, ``filled_count``,
        ``artifact_count``.
    """
    store = _get_data_store()
    run_id = run_id or _current_run_id
    if not run_id:
        raise ValueError("No run_id available; pass run_id or call set_run_id().")
    try:
        run_record = store.get_run(run_id)
    except KeyError:
        logger.warning("datastore_get_run_not_found run_id=%s", run_id)
        return {"found": False, "run_id": run_id}
    logger.info("datastore_get_run_found run_id=%s", run_id)
    return {"found": True, **run_record}


def datastore_list_runs() -> dict[str, Any]:
    """List all runs in the DataStore with summary statistics.

    Returns:
        Dict with ``runs`` list.  Each entry contains ``run_id``,
        ``start_date``, ``end_date``, ``created_at``,
        ``timeseries_count``, ``filled_count``, ``artifact_count``.
    """
    store = _get_data_store()
    runs = store.list_runs_with_stats()
    logger.info("datastore_list_runs count=%d", len(runs))
    _debug_tool_event("datastore_list_runs", "completed", count=len(runs))
    return {"runs": runs}


def datastore_list_instruments() -> dict[str, Any]:
    """List all known instrument symbols in the DataStore.

    Returns:
        Dict with ``instruments`` list of symbol strings.
    """
    store = _get_data_store()
    instruments = store.get_instruments()
    logger.info("datastore_list_instruments count=%d", len(instruments))
    _debug_tool_event("datastore_list_instruments", "completed", count=len(instruments))
    return {"instruments": instruments}


def datastore_list_sources() -> dict[str, Any]:
    """List all known data source names in the DataStore.

    Returns:
        Dict with ``sources`` list of source name strings.
    """
    store = _get_data_store()
    sources = store.get_sources()
    logger.info("datastore_list_sources count=%d", len(sources))
    _debug_tool_event("datastore_list_sources", "completed", count=len(sources))
    return {"sources": sources}


def datastore_list_gap_filling_methods() -> dict[str, Any]:
    """List all known gap-filling method names in the DataStore.

    Returns:
        Dict with ``methods`` list of method name strings.
    """
    store = _get_data_store()
    methods = store.get_gap_filling_methods()
    logger.info("datastore_list_gap_filling_methods count=%d", len(methods))
    _debug_tool_event("datastore_list_gap_filling_methods", "completed", count=len(methods))
    return {"methods": methods}


def datastore_put_timeseries(
    symbol: str,
    source: str,
    dates: list[str],
    prices: list[Any],
    run_id: str | None = None,
) -> dict[str, Any]:
    """Store a raw time series in the DataStore and return a reference key.

    Args:
        symbol: Ticker symbol.
        source: Data source name.
        dates: List of ISO date strings.
        prices: List of price values (float or None for missing).
        run_id: The run/session identifier.  Defaults to the current run
            set by ``set_run_id()``.

    Returns:
        Dict with ``data_ref``, ``symbol``, ``source``, ``observations``.
    """
    store = _get_data_store()
    run_id = run_id or _current_run_id
    if not run_id:
        raise ValueError("No run_id available; pass run_id or call set_run_id().")
    if len(dates) != len(prices):
        raise ValueError("dates and prices must have the same length.")
    data_ref = store.put_timeseries(
        run_id=run_id,
        symbol=symbol,
        source=source,
        dates=dates,
        prices=prices,
    )
    logger.info(
        "datastore_put_timeseries data_ref=%s symbol=%s source=%s observations=%d",
        data_ref, symbol, source, len(dates),
    )
    _debug_tool_event("datastore_put_timeseries", "completed", data_ref=data_ref, observations=len(dates))
    return {"data_ref": data_ref, "symbol": symbol, "source": source, "observations": len(dates)}


def datastore_get_timeseries(data_ref: str) -> dict[str, Any]:
    """Load a raw time series from the DataStore by its reference key.

    Args:
        data_ref: The reference key returned by ``datastore_put_timeseries``
            (form ``<run_id>:<symbol>:<source>``).

    Returns:
        Dict with ``found`` boolean plus ``run_id``, ``symbol``, ``source``,
        ``dates``, ``prices`` when resolved.
    """
    store = _get_data_store()
    try:
        series = store.get_timeseries(data_ref)
    except KeyError:
        logger.warning("datastore_get_timeseries_not_found data_ref=%s", data_ref)
        return {"found": False, "data_ref": data_ref}
    logger.info(
        "datastore_get_timeseries_found data_ref=%s symbol=%s observations=%d",
        data_ref, series.get("symbol"), len(series.get("dates", [])),
    )
    return {"found": True, **series}


def datastore_list_timeseries(
    run_id: str | None = None,
    symbol: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """List stored raw time series in the DataStore, optionally filtered.

    Args:
        run_id: The run/session identifier.  Defaults to the current run
            set by ``set_run_id()``.
        symbol: If provided, filter by symbol.
        source: If provided, filter by source.

    Returns:
        Dict with ``series`` list.  Each entry contains ``data_ref``,
        ``symbol``, ``source``, ``dates``, ``prices``.
    """
    store = _get_data_store()
    run_id = run_id or _current_run_id
    if not run_id:
        raise ValueError("No run_id available; pass run_id or call set_run_id().")
    series = store.list_timeseries(run_id, symbol=symbol, source=source)
    logger.info(
        "datastore_list_timeseries run_id=%s symbol=%s source=%s count=%d",
        run_id, symbol or "*", source or "*", len(series),
    )
    _debug_tool_event("datastore_list_timeseries", "completed", count=len(series))
    return {"run_id": run_id, "series": series}


def datastore_delete_timeseries(data_ref: str) -> dict[str, Any]:
    """Delete a raw time series from the DataStore by its reference key.

    Args:
        data_ref: The reference key to delete.

    Returns:
        Dict with ``deleted`` boolean and ``data_ref``.
    """
    store = _get_data_store()
    deleted = store.delete_timeseries(data_ref)
    logger.info("datastore_delete_timeseries data_ref=%s deleted=%s", data_ref, deleted)
    _debug_tool_event("datastore_delete_timeseries", "completed", data_ref=data_ref, deleted=deleted)
    return {"deleted": deleted, "data_ref": data_ref}


def datastore_put_gap_filled_series(
    symbol: str,
    source: str,
    method: str,
    filled_dates: list[str],
    filled_prices: list[Any],
    original_dates: list[str] | None = None,
    original_prices: list[Any] | None = None,
    original_data_ref: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Store a gap-filled time series in the DataStore and return a reference key.

    Args:
        symbol: Ticker symbol.
        source: Data source name.
        method: Gap-filling method used (e.g. ``linear_interpolation``).
        filled_dates: Date strings after filling.
        filled_prices: Price values after filling.
        original_dates: Optional original (pre-filling) date strings.
        original_prices: Optional original (pre-filling) price values.
        original_data_ref: Optional reference to the raw source series.
        run_id: The run/session identifier.  Defaults to the current run
            set by ``set_run_id()``.

    Returns:
        Dict with ``data_ref``, ``symbol``, ``source``, ``method``,
        ``observations``.
    """
    store = _get_data_store()
    run_id = run_id or _current_run_id
    if not run_id:
        raise ValueError("No run_id available; pass run_id or call set_run_id().")
    if len(filled_dates) != len(filled_prices):
        raise ValueError("filled_dates and filled_prices must have the same length.")
    data_ref = store.put_gap_filled_series(
        run_id=run_id,
        symbol=symbol,
        source=source,
        method=method,
        filled_dates=filled_dates,
        filled_prices=filled_prices,
        original_dates=original_dates,
        original_prices=original_prices,
        original_data_ref=original_data_ref,
    )
    logger.info(
        "datastore_put_gap_filled_series data_ref=%s symbol=%s source=%s method=%s observations=%d",
        data_ref, symbol, source, method, len(filled_dates),
    )
    _debug_tool_event("datastore_put_gap_filled_series", "completed", data_ref=data_ref, method=method)
    return {
        "data_ref": data_ref,
        "symbol": symbol,
        "source": source,
        "method": method,
        "observations": len(filled_dates),
    }


def datastore_get_gap_filled_series(data_ref: str) -> dict[str, Any]:
    """Load a gap-filled series from the DataStore by its reference key.

    Args:
        data_ref: The reference key returned by
            ``datastore_put_gap_filled_series``
            (form ``<run_id>:<symbol>:<source>:filled``).

    Returns:
        Dict with ``found`` boolean plus ``run_id``, ``symbol``, ``source``,
        ``method``, ``original_dates``, ``original_prices``,
        ``filled_dates``, ``filled_prices`` when resolved.
    """
    store = _get_data_store()
    try:
        series = store.get_gap_filled_series(data_ref)
    except KeyError:
        logger.warning("datastore_get_gap_filled_series_not_found data_ref=%s", data_ref)
        return {"found": False, "data_ref": data_ref}
    logger.info(
        "datastore_get_gap_filled_series_found data_ref=%s symbol=%s method=%s",
        data_ref, series.get("symbol"), series.get("method"),
    )
    return {"found": True, **series}


def datastore_list_gap_filled_series(
    run_id: str | None = None,
    symbol: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """List stored gap-filled series in the DataStore, optionally filtered.

    Args:
        run_id: The run/session identifier.  Defaults to the current run
            set by ``set_run_id()``.
        symbol: If provided, filter by symbol.
        source: If provided, filter by source.

    Returns:
        Dict with ``series`` list.  Each entry contains ``data_ref``,
        ``symbol``, ``source``, ``method``, ``filled_dates``,
        ``filled_prices``.
    """
    store = _get_data_store()
    run_id = run_id or _current_run_id
    if not run_id:
        raise ValueError("No run_id available; pass run_id or call set_run_id().")
    series = store.list_gap_filled_series(run_id, symbol=symbol, source=source)
    logger.info(
        "datastore_list_gap_filled_series run_id=%s symbol=%s source=%s count=%d",
        run_id, symbol or "*", source or "*", len(series),
    )
    _debug_tool_event("datastore_list_gap_filled_series", "completed", count=len(series))
    return {"run_id": run_id, "series": series}


def datastore_put_quality_report(
    report: dict[str, Any],
    symbol: str | None = None,
    source: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Store a quality report in the DataStore and return a reference key.

    Args:
        report: The quality report dict.
        symbol: Optional symbol associated with the report.
        source: Optional source associated with the report.
        run_id: The run/session identifier.  Defaults to the current run
            set by ``set_run_id()``.

    Returns:
        Dict with ``report_id``, ``symbol``, ``source``.
    """
    store = _get_data_store()
    run_id = run_id or _current_run_id
    if not run_id:
        raise ValueError("No run_id available; pass run_id or call set_run_id().")
    report_id = store.put_quality_report(
        run_id=run_id,
        report=report,
        symbol=symbol,
        source=source,
    )
    logger.info(
        "datastore_put_quality_report report_id=%s symbol=%s source=%s",
        report_id, symbol or "unknown", source or "unknown",
    )
    _debug_tool_event("datastore_put_quality_report", "completed", report_id=report_id)
    return {"report_id": report_id, "symbol": symbol, "source": source}


def datastore_get_quality_report(report_id: str) -> dict[str, Any]:
    """Load a quality report from the DataStore by its reference key.

    Args:
        report_id: The reference key returned by
            ``datastore_put_quality_report``.

    Returns:
        Dict with ``found`` boolean plus the report dict when resolved.
    """
    store = _get_data_store()
    try:
        report = store.get_quality_report(report_id)
    except KeyError:
        logger.warning("datastore_get_quality_report_not_found report_id=%s", report_id)
        return {"found": False, "report_id": report_id}
    logger.info("datastore_get_quality_report_found report_id=%s", report_id)
    return {"found": True, "report_id": report_id, "report": report}


def datastore_list_quality_reports(
    run_id: str | None = None,
    symbol: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """List quality reports in the DataStore for a run, optionally filtered.

    Args:
        run_id: The run/session identifier.  Defaults to the current run
            set by ``set_run_id()``.
        symbol: If provided, filter by symbol.
        source: If provided, filter by source.

    Returns:
        Dict with ``reports`` list.  Each entry contains ``report_id``,
        ``symbol``, ``source``.
    """
    store = _get_data_store()
    run_id = run_id or _current_run_id
    if not run_id:
        raise ValueError("No run_id available; pass run_id or call set_run_id().")
    reports = store.list_quality_reports(run_id, symbol=symbol, source=source)
    logger.info(
        "datastore_list_quality_reports run_id=%s symbol=%s source=%s count=%d",
        run_id, symbol or "*", source or "*", len(reports),
    )
    _debug_tool_event("datastore_list_quality_reports", "completed", count=len(reports))
    return {"run_id": run_id, "reports": reports}


def datastore_put_artifact(
    artifact_type: str,
    path: str,
    symbol: str | None = None,
    source: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Record an artifact path in the DataStore.

    Args:
        artifact_type: One of 'csv', 'png', 'report'.
        path: File path to the artifact.
        symbol: Optional symbol.
        source: Optional source.
        run_id: The run/session identifier.  Defaults to the current run
            set by ``set_run_id()``.

    Returns:
        Dict with ``artifact_id``, ``artifact_type``, ``path``.
    """
    store = _get_data_store()
    run_id = run_id or _current_run_id
    if not run_id:
        raise ValueError("No run_id available; pass run_id or call set_run_id().")
    artifact_id = store.put_artifact(
        run_id=run_id,
        artifact_type=artifact_type,
        path=path,
        symbol=symbol,
        source=source,
    )
    logger.info(
        "datastore_put_artifact artifact_id=%s type=%s path=%s",
        artifact_id, artifact_type, path,
    )
    _debug_tool_event("datastore_put_artifact", "completed", artifact_id=artifact_id, type=artifact_type)
    return {"artifact_id": artifact_id, "artifact_type": artifact_type, "path": path}


def datastore_list_artifacts(
    run_id: str | None = None,
    artifact_type: str | None = None,
) -> dict[str, Any]:
    """List stored artifacts in the DataStore for a run.

    Args:
        run_id: The run/session identifier.  Defaults to the current run
            set by ``set_run_id()``.
        artifact_type: If provided, filter by type.

    Returns:
        Dict with ``artifacts`` list.  Each entry contains ``artifact_id``,
        ``artifact_type``, ``path``, ``symbol``, ``source``.
    """
    store = _get_data_store()
    run_id = run_id or _current_run_id
    if not run_id:
        raise ValueError("No run_id available; pass run_id or call set_run_id().")
    artifacts = store.list_artifacts(run_id, artifact_type=artifact_type)
    logger.info(
        "datastore_list_artifacts run_id=%s type=%s count=%d",
        run_id, artifact_type or "*", len(artifacts),
    )
    _debug_tool_event("datastore_list_artifacts", "completed", count=len(artifacts))
    return {"run_id": run_id, "artifacts": artifacts}


def datastore_vacuum() -> dict[str, Any]:
    """Recover disk space and defragment the DataStore database.

    Returns:
        Dict with ``vacuumed`` boolean and ``db_path``.
    """
    store = _get_data_store()
    store.vacuum()
    logger.info("datastore_vacuum db_path=%s", store.db_path)
    _debug_tool_event("datastore_vacuum", "completed")
    return {"vacuumed": True, "db_path": str(store.db_path)}


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
    "get_populated_timeseries": _tool(
        get_populated_timeseries,
        "get_populated_timeseries",
        "Load populated time series data for a run from the DataStore, "
        "including both the raw (before) and gap-filled (after) series.",
    ),
    "find_run_by_id": _tool(
        find_run_by_id,
        "find_run_by_id",
        "Find a run record in the DataStore by its run_id.",
    ),
}


# ── DataStore tool registry ────────────────────────────────────────────────
# Exposes the DataStore API (database.py) to ReAct agents.  These tools are
# kept separate from TOOL_REGISTRY so that agents can opt in to direct
# database access without exposing the full domain tool surface.
DATASTORE_TOOL_REGISTRY: dict[str, StructuredTool] = {
    "datastore_put_run_metadata": _tool(
        datastore_put_run_metadata,
        "datastore_put_run_metadata",
        "Register or update a run with optional date metadata in the DataStore.",
    ),
    "datastore_get_run": _tool(
        datastore_get_run,
        "datastore_get_run",
        "Return a single run record from the DataStore by its run_id.",
    ),
    "datastore_list_runs": _tool(
        datastore_list_runs,
        "datastore_list_runs",
        "List all runs in the DataStore with summary statistics.",
    ),
    "datastore_list_instruments": _tool(
        datastore_list_instruments,
        "datastore_list_instruments",
        "List all known instrument symbols in the DataStore.",
    ),
    "datastore_list_sources": _tool(
        datastore_list_sources,
        "datastore_list_sources",
        "List all known data source names in the DataStore.",
    ),
    "datastore_list_gap_filling_methods": _tool(
        datastore_list_gap_filling_methods,
        "datastore_list_gap_filling_methods",
        "List all known gap-filling method names in the DataStore.",
    ),
    "datastore_put_timeseries": _tool(
        datastore_put_timeseries,
        "datastore_put_timeseries",
        "Store a raw time series in the DataStore and return a reference key.",
    ),
    "datastore_get_timeseries": _tool(
        datastore_get_timeseries,
        "datastore_get_timeseries",
        "Load a raw time series from the DataStore by its reference key.",
    ),
    "datastore_list_timeseries": _tool(
        datastore_list_timeseries,
        "datastore_list_timeseries",
        "List stored raw time series in the DataStore, optionally filtered by symbol/source.",
    ),
    "datastore_delete_timeseries": _tool(
        datastore_delete_timeseries,
        "datastore_delete_timeseries",
        "Delete a raw time series from the DataStore by its reference key.",
    ),
    "datastore_put_gap_filled_series": _tool(
        datastore_put_gap_filled_series,
        "datastore_put_gap_filled_series",
        "Store a gap-filled time series in the DataStore and return a reference key.",
    ),
    "datastore_get_gap_filled_series": _tool(
        datastore_get_gap_filled_series,
        "datastore_get_gap_filled_series",
        "Load a gap-filled series from the DataStore by its reference key.",
    ),
    "datastore_list_gap_filled_series": _tool(
        datastore_list_gap_filled_series,
        "datastore_list_gap_filled_series",
        "List stored gap-filled series in the DataStore, optionally filtered by symbol/source.",
    ),
    "datastore_put_quality_report": _tool(
        datastore_put_quality_report,
        "datastore_put_quality_report",
        "Store a quality report in the DataStore and return a reference key.",
    ),
    "datastore_get_quality_report": _tool(
        datastore_get_quality_report,
        "datastore_get_quality_report",
        "Load a quality report from the DataStore by its reference key.",
    ),
    "datastore_list_quality_reports": _tool(
        datastore_list_quality_reports,
        "datastore_list_quality_reports",
        "List quality reports in the DataStore for a run, optionally filtered by symbol/source.",
    ),
    "datastore_put_artifact": _tool(
        datastore_put_artifact,
        "datastore_put_artifact",
        "Record an artifact path in the DataStore.",
    ),
    "datastore_list_artifacts": _tool(
        datastore_list_artifacts,
        "datastore_list_artifacts",
        "List stored artifacts in the DataStore for a run, optionally filtered by type.",
    ),
    "datastore_vacuum": _tool(
        datastore_vacuum,
        "datastore_vacuum",
        "Recover disk space and defragment the DataStore database.",
    ),
}


def get_datastore_tool(name: str) -> StructuredTool | None:
    """Retrieve a DataStore tool by name from the registry."""
    return DATASTORE_TOOL_REGISTRY.get(name)


def get_datastore_tool_description(name: str) -> str | None:
    """Return a DataStore tool description from the registry if available."""
    tool = DATASTORE_TOOL_REGISTRY.get(name)
    if tool is None:
        return None
    return str(getattr(tool, "description", "") or "").strip() or None


def get_tool(name: str) -> StructuredTool | None:
    """Retrieve a tool by name from the registry."""
    return TOOL_REGISTRY.get(name)


def get_tool_description(name: str) -> str | None:
    """Return a tool description from the registry if available."""
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return None
    return str(getattr(tool, "description", "") or "").strip() or None
