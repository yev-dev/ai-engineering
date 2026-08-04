"""Pydantic models for tool input validation and transformation.

These models validate and normalize LLM-generated tool arguments before
they reach the underlying tool functions. This ensures consistent parameter
naming, type coercion, and early error detection regardless of which LLM
provider generated the call.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ── Helper ────────────────────────────────────────────────────────────────

_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y")


def _coerce_date(value: Any) -> str:
    """Coerce a value to an ISO date string (yyyy-mm-dd)."""
    if not value:
        raise ValueError("Date value is empty")
    if isinstance(value, str):
        value = value.strip()
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        # Try pandas-style flexible parsing as last resort
        try:
            import pandas as pd
            ts = pd.Timestamp(value)
            return ts.strftime("%Y-%m-%d")
        except Exception:
            raise ValueError(f"Cannot parse date: {value}")
    if isinstance(value, (int, float)):
        # Could be a timestamp
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            raise ValueError(f"Cannot parse numeric date: {value}")
    raise ValueError(f"Unsupported date type: {type(value).__name__}")


# ── Tool Input Models ─────────────────────────────────────────────────────


class HistoricalPricesInput(BaseModel):
    """Validated input for the ``historical_prices`` tool."""

    symbol: str = Field(..., min_length=1, max_length=20, description="Ticker symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD or parseable)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD or parseable)")
    source: str = Field(..., pattern=r"^(yahoo|bloomberg|reuters)$", description="Data source name")

    @field_validator("start_date", "end_date")
    @classmethod
    def normalize_date(cls, value: str) -> str:
        return _coerce_date(value)

    @model_validator(mode="after")
    def validate_date_range(self) -> "HistoricalPricesInput":
        if self.end_date < self.start_date:
            raise ValueError("End date must be on or after start date")
        return self


class CheckDataQualityInput(BaseModel):
    """Validated input for the ``check_data_quality`` tool.

    Accepts either a ``data_ref`` (to load from DataStore) or inline
    ``prices``/``dates``/``source``/``symbol`` parameters.
    """

    prices: list[Any] | None = Field(None, description="List of price values")
    dates: list[str] | None = Field(None, description="List of ISO date strings")
    source: str | None = Field(None, description="Data source name")
    symbol: str | None = Field(None, min_length=1, max_length=20, description="Ticker symbol")
    data: dict[str, Any] | None = Field(None, description="Dict from historical_prices output")
    data_ref: str | None = Field(None, description="Reference key to load from DataStore")
    start_date: str | None = Field(None, description="Start date for auto-fetching")
    end_date: str | None = Field(None, description="End date for auto-fetching")

    @model_validator(mode="after")
    def resolve_data(self) -> "CheckDataQualityInput":
        """Ensure at least one data source is available."""
        has_inline = self.prices is not None or self.data is not None
        has_ref = self.data_ref is not None
        has_auto = self.source is not None and self.symbol is not None
        if not has_inline and not has_ref and not has_auto:
            raise ValueError(
                "check_data_quality requires prices, data, data_ref, or source+symbol"
            )
        return self


class ApplyGapFillingInput(BaseModel):
    """Validated input for the ``apply_gap_filling`` tool.

    The LLM only needs to provide identifiers (``data_ref`` or
    ``symbol`` + ``source``) – the full time series is always loaded
    from the database by the tool.
    """

    prices: dict[str, Any] | None = Field(None, description="Backward-compat dict with 'prices' and 'dates' keys")
    data_ref: str | None = Field(None, description="Reference key to load prices from DataStore")
    method: str = Field(
        ...,
        pattern=r"^(linear_interpolation|forward_fill|backward_fill|none)$",
        description="Gap-filling method",
    )
    dates: list[str] | None = Field(None, description="Optional override for date index")
    symbol: str | None = Field(None, min_length=1, max_length=20, description="Ticker symbol (with source)")
    source: str | None = Field(None, pattern=r"^(yahoo|bloomberg|reuters)$", description="Data source name (with symbol)")

    @model_validator(mode="after")
    def resolve_prices(self) -> "ApplyGapFillingInput":
        has_inline = self.prices is not None
        has_ref = self.data_ref is not None
        has_identifiers = self.symbol is not None and self.source is not None
        if not has_inline and not has_ref and not has_identifiers:
            raise ValueError(
                "apply_gap_filling requires 'prices', 'data_ref', or 'symbol' + 'source'"
            )
        return self


class BuildTimeseriesInput(BaseModel):
    """Validated input for the ``build_timeseries`` tool.

    The LLM only needs to provide identifiers (``data_ref`` or
    ``symbol`` + ``source``) – the full time series is always loaded
    from the database by the tool.
    """

    series: dict[str, Any] | None = Field(None, description="Backward-compat dict with 'dates' and 'prices' keys")
    data_ref: str | None = Field(None, description="Reference key to load series from DataStore")
    filename: str = Field("final_timeseries.csv", description="Output filename")
    run_id: str | None = Field(None, description="Optional run identifier")
    symbol: str | None = Field(None, min_length=1, max_length=20, description="Ticker symbol (with source)")
    source: str | None = Field(None, pattern=r"^(yahoo|bloomberg|reuters)$", description="Data source name (with symbol)")


class VisualizeTimeseriesInput(BaseModel):
    """Validated input for the ``visualize_timeseries`` tool.

    The LLM only needs to provide identifiers (``data_ref`` or
    ``symbol`` + ``source``) – the full time series is always loaded
    from the database by the tool.
    """

    prices: dict[str, Any] | None = Field(None, description="Backward-compat dict with 'dates' and 'prices' keys")
    title: str = Field("Time series", description="Chart title")
    run_id: str | None = Field(None, description="Optional run identifier")
    data_ref: str | None = Field(None, description="Reference key to load series from DataStore")
    symbol: str | None = Field(None, min_length=1, max_length=20, description="Ticker symbol (with source)")
    source: str | None = Field(None, pattern=r"^(yahoo|bloomberg|reuters)$", description="Data source name (with symbol)")

    @model_validator(mode="after")
    def resolve_prices(self) -> "VisualizeTimeseriesInput":
        has_inline = self.prices is not None
        has_ref = self.data_ref is not None
        has_identifiers = self.symbol is not None and self.source is not None
        if not has_inline and not has_ref and not has_identifiers:
            raise ValueError(
                "visualize_timeseries requires 'prices', 'data_ref', or 'symbol' + 'source'"
            )
        return self


class DelegateToAgentInput(BaseModel):
    """Validated input for the ``delegate_to_agent`` tool."""

    agent_name: str = Field(..., min_length=1, description="Name of the target agent")
    request: str = Field(..., min_length=1, description="The request to pass to the target agent")


class RequestHumanInput(BaseModel):
    """Validated input for the ``request_human_input`` tool."""

    prompt: str = Field(..., min_length=1, description="The question or prompt to display")
    options: list[str] | None = Field(None, description="Optional list of valid choices")
    context: dict[str, Any] | None = Field(None, description="Optional additional context data")


class GetInstrumentDetailsInput(BaseModel):
    """Validated input for the ``get_instrument_details`` tool."""

    query: str | None = Field(None, min_length=1, description="The search query")
    identifier: str = Field("auto", pattern=r"^(auto|ticker|symbol|name|security_name)$")
    symbol: str | None = Field(None, min_length=1, max_length=20, description="Alternative query param")


class NormalizeDatesInput(BaseModel):
    """Validated input for date normalization tools."""

    start_date: str | None = Field(None, description="Start date")
    end_date: str | None = Field(None, description="End date")
    request: str | None = Field(None, description="Free-form request containing date range")


# ── Tool Input Registry ───────────────────────────────────────────────────

TOOL_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "historical_prices": HistoricalPricesInput,
    "check_data_quality": CheckDataQualityInput,
    "apply_gap_filling": ApplyGapFillingInput,
    "build_timeseries": BuildTimeseriesInput,
    "visualize_timeseries": VisualizeTimeseriesInput,
    "delegate_to_agent": DelegateToAgentInput,
    "request_human_input": RequestHumanInput,
    "get_instrument_details": GetInstrumentDetailsInput,
    "normalize_requested_dates": NormalizeDatesInput,
    "extract_requested_date_range": NormalizeDatesInput,
}


def validate_tool_input(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize tool arguments using the registered Pydantic model.

    Args:
        tool_name: The tool name.
        args: Raw arguments from the LLM.

    Returns:
        Validated and normalized arguments dict.

    Raises:
        ValueError: If validation fails.
    """
    model_cls = TOOL_INPUT_MODELS.get(tool_name)
    if model_cls is None:
        # No model registered — pass through as-is
        return args
    try:
        validated = model_cls(**args)
        return validated.model_dump(exclude_none=True)
    except Exception as exc:
        logger.warning(
            "tool_input_validation_failed tool=%s error=%s args_keys=%s",
            tool_name,
            exc,
            sorted(args.keys()),
        )
        raise ValueError(f"Invalid arguments for {tool_name}: {exc}") from exc