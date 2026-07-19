"""Deterministic domain tools exposed to ReAct agents."""
from __future__ import annotations

import json
import logging
import os
import uuid
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


def _run_dir(run_id: str | None = None) -> Path:
    directory = OUTPUT_ROOT / (run_id or f"run_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}")
    directory.mkdir(parents=True, exist_ok=True)
    logger.debug("artifact_directory path=%s", directory)
    return directory


def get_instrument_details(
    query: str | None = None,
    identifier: str = "auto",
    symbol: str | None = None,
) -> dict[str, Any]:
    """Resolve an instrument by symbol, ticker, short name, or full name."""
    value_input = query or symbol or ""
    logger.info("tool_get_instrument_details query=%s identifier=%s", value_input, identifier)
    frame = pd.read_csv(DATA_DIR / "instruments.csv")
    value = value_input.strip().casefold()
    if not value:
        return {"found": False, "query": value_input, "message": "No instrument query was provided."}

    ticker_candidate = value_input.strip().upper() == value_input.strip() and value_input.strip().isalpha() and len(value_input.strip()) <= 6
    columns = ["symbol"] if identifier.casefold() == "auto" and ticker_candidate else ["symbol", "security_name"]
    if identifier.casefold() in {"ticker", "symbol"}:
        columns = ["symbol"]
    elif identifier.casefold() in {"name", "asset", "security", "security_name"}:
        columns = ["security_name"]

    normalized = frame[columns].astype(str).apply(lambda column: column.str.strip().str.casefold())
    matches = frame[normalized.eq(value).any(axis=1)]
    if matches.empty:
        matches = frame[normalized.apply(lambda column: column.str.contains(value, regex=False)).any(axis=1)]
    if matches.empty:
        choices = [item for column in columns for item in frame[column].dropna().astype(str)]
        suggestions = get_close_matches(value_input, choices, n=3, cutoff=0.65)
        logger.warning("instrument_not_found query=%s identifier=%s suggestions=%s", value_input, identifier, suggestions)
        message = "Instrument was not found."
        if suggestions:
            message += f" Did you mean: {', '.join(suggestions)}?"
        return {"found": False, "query": value_input, "suggestions": suggestions, "message": message}
    record = matches.iloc[0].to_dict()
    logger.info("instrument_resolved query=%s symbol=%s security_name=%s", value_input, record.get("symbol"), record.get("security_name"))
    return {"found": True, "query": value_input, **record}


def available_data_sources() -> list[str]:
    logger.info("tool_available_data_sources count=%d", len(SOURCES))
    return list(SOURCES)


def historical_prices(symbol: str, start_date: str, end_date: str, source: str) -> dict[str, Any]:
    logger.info("tool_historical_prices_start symbol=%s source=%s start=%s end=%s", symbol, source, start_date, end_date)
    source = source.casefold()
    if source not in SOURCES:
        raise ValueError(f"Unsupported source: {source}")
    frame = pd.read_csv(DATA_DIR / f"{source}_stock_data.csv", index_col="Date", parse_dates=True)
    if symbol not in frame.columns:
        raise ValueError(f"Ticker {symbol} is not available in {source} data.")
    series = pd.to_numeric(frame[symbol], errors="coerce").sort_index()
    series = series.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
    if series.empty:
        raise ValueError(f"No historical data is available for {symbol} from {start_date} to {end_date} in {source}.")
    logger.info("tool_historical_prices_completed symbol=%s source=%s observations=%d missing=%d", symbol, source, len(series), int(series.isna().sum()))
    return {"symbol": symbol, "source": source, "dates": [d.strftime("%Y-%m-%d") for d in series.index],
            "prices": [None if pd.isna(value) else float(value) for value in series]}


def check_data_quality(prices: list[Any], source: str, symbol: str) -> dict[str, Any]:
    logger.info("tool_check_data_quality_start symbol=%s source=%s observations=%d", symbol, source, len(prices))
    values = pd.Series(prices, dtype="float64")
    missing = int(values.isna().sum())
    non_positive = int((values.dropna() <= 0).sum())
    issues = []
    if missing:
        issues.append("missing_or_nan_values")
    if non_positive:
        issues.append("non_positive_prices")
    result = {"source": source, "symbol": symbol, "total_values": len(values),
              "missing_count": missing, "nan_count": missing,
              "completeness_pct": round((1 - missing / len(values)) * 100, 2) if len(values) else 0.0,
              "duplicate_count": 0, "issues": issues}
    logger.info("tool_check_data_quality_completed symbol=%s source=%s missing=%d issues=%d", symbol, source, missing, len(issues))
    return result


def recommend_gap_methods(quality_report: dict[str, Any], prices: dict[str, Any]) -> list[str]:
    methods = ["linear_interpolation", "forward_fill", "backward_fill"] if quality_report.get("missing_count") else ["none"]
    logger.info("tool_gap_methods_recommended symbol=%s methods=%s", prices.get("symbol"), methods)
    return methods


def apply_gap_filling(prices: dict[str, Any], method: str, dates: list[str] | None = None) -> dict[str, Any]:
    logger.info("tool_gap_filling_start symbol=%s method=%s", prices.get("symbol"), method)
    series = pd.Series(prices["prices"], index=pd.to_datetime(dates or prices["dates"]), dtype="float64")
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
    result = {"symbol": prices["symbol"], "method": method,
              "dates": [d.strftime("%Y-%m-%d") for d in filled.index],
              "prices": [None if pd.isna(value) else float(value) for value in filled]}
    logger.info("tool_gap_filling_completed symbol=%s method=%s observations=%d remaining_missing=%d", prices.get("symbol"), method, len(filled), int(filled.isna().sum()))
    return result


def build_timeseries(series: dict[str, Any], filename: str = "final_timeseries.csv", run_id: str | None = None) -> str:
    logger.info("tool_build_timeseries_start symbol=%s filename=%s", series.get("symbol"), filename)
    output = _run_dir(run_id) / filename
    pd.DataFrame({"date": series["dates"], "price": series["prices"]}).to_csv(output, index=False)
    logger.info("tool_build_timeseries_completed path=%s", output)
    return str(output)


def generate_report(data: dict[str, Any], filename: str = "quality_report.csv", run_id: str | None = None) -> str:
    logger.info("tool_generate_report_start filename=%s", filename)
    output = _run_dir(run_id) / filename
    pd.DataFrame(data if isinstance(data, list) else [data]).to_csv(output, index=False)
    logger.info("tool_generate_report_completed path=%s", output)
    return str(output)


def visualize_timeseries(prices: dict[str, Any], title: str = "Time series", run_id: str | None = None) -> str:
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
    return {"status": "delegating", "agent_name": agent_name, "request": request}


def _tool(function: Any, name: str, description: str) -> StructuredTool:
    return StructuredTool.from_function(func=function, name=name, description=description)


TOOL_REGISTRY = {
    "get_instrument_details": _tool(get_instrument_details, "get_instrument_details", "Resolve a ticker or security name from the instrument catalog."),
    "available_data_sources": _tool(available_data_sources, "available_data_sources", "List configured historical data sources."),
    "historical_prices": _tool(historical_prices, "historical_prices", "Load a ticker's historical prices for a date range."),
    "check_data_quality": _tool(check_data_quality, "check_data_quality", "Calculate completeness and common price-quality metrics."),
    "recommend_gap_methods": _tool(recommend_gap_methods, "recommend_gap_methods", "Recommend methods for missing observations."),
    "apply_gap_filling": _tool(apply_gap_filling, "apply_gap_filling", "Apply a supported gap-filling method."),
    "build_timeseries": _tool(build_timeseries, "build_timeseries", "Persist a final time series CSV artifact."),
    "generate_report": _tool(generate_report, "generate_report", "Persist a CSV quality report artifact."),
    "visualize_timeseries": _tool(visualize_timeseries, "visualize_timeseries", "Create a seaborn time series chart."),
    "delegate_to_agent": _tool(delegate_to_agent, "delegate_to_agent", "Delegate work to a named specialist agent."),
}


def get_tool(name: str) -> StructuredTool | None:
    return TOOL_REGISTRY.get(name)
