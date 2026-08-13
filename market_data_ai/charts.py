"""Seaborn chart builders for the Time Series Construction dashboard.

This module is deliberately free of ``streamlit`` so the figure-building logic
can be tested and reused independently.  The dashboard (``dashboard.py``)
calls :func:`build_run_graphs`, which retrieves the "before" (raw) and "after"
(gap-filled) time series for a run from the DataStore and renders each pair
through every registered graph builder.

Extending the graph set is a purely additive two-step change:

    1. Write a builder ``def my_builder(symbol, source, before_df, after_df)
       -> matplotlib.figure.Figure``.
    2. Append it to :data:`_GRAPH_BUILDERS`.

No other code needs to change; both the before/after graph and any future
graphs are picked up automatically for every series pair.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import matplotlib

# Use a non-interactive backend so figures can be produced outside the GUI
# main thread (mirrors tools.py) and in headless/CI environments.
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

logger = logging.getLogger(__name__)

# Type alias for a graph builder: (symbol, source, before_df, after_df) -> Figure.
GraphBuilder = Callable[
    [str, str, "pd.DataFrame | None", "pd.DataFrame | None"], "plt.Figure"
]


def series_to_frame(series: dict[str, Any] | None) -> pd.DataFrame | None:
    """Convert a DataStore series dict into a tidy ``DataFrame``.

    Handles both "before" (``dates``/``prices``) and "after"
    (``filled_dates``/``filled_prices`` or ``original_dates``/``original_prices``)
    payloads.  Returns ``None`` when no usable observations exist.

    Args:
        series: A series dict returned by the DataStore ``list_*_series`` /
            ``get_*_series`` methods, or the normalised ``before``/``after``
            entries from ``get_populated_timeseries``.

    Returns:
        A frame with ``date`` (datetime) and ``price`` (float) columns, or
        ``None`` if there is nothing to plot.
    """
    if series is None:
        return None
    dates = (
        series.get("dates")
        or series.get("filled_dates")
        or series.get("original_dates")
        or []
    )
    prices = (
        series.get("prices")
        or series.get("filled_prices")
        or series.get("original_prices")
        or []
    )
    if not dates:
        return None
    frame = pd.DataFrame(
        {"date": pd.to_datetime(dates, errors="coerce"), "price": prices}
    )
    return frame.dropna(subset=["date"])


def _pair_series(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> list[tuple[str, str, pd.DataFrame | None, pd.DataFrame | None]]:
    """Pair each "after" series with its matching "before" series.

    Series are matched on ``(symbol, source)``.  Returns a list of
    ``(symbol, source, before_df, after_df)`` tuples for every unique pair,
    including before-only and after-only series so nothing is dropped.

    Args:
        before: Normalised "before" series list (raw observations).
        after: Normalised "after" series list (gap-filled series).

    Returns:
        A list of pairing tuples.
    """
    before_map = {(b.get("symbol"), b.get("source")): b for b in before}
    after_map = {(a.get("symbol"), a.get("source")): a for a in after}

    pairs: list[tuple[str, str, pd.DataFrame | None, pd.DataFrame | None]] = []
    for key, after_series in after_map.items():
        before_df = series_to_frame(before_map.get(key))
        after_df = series_to_frame(after_series)
        pairs.append((key[0], key[1], before_df, after_df))
    for key, before_series in before_map.items():
        if key not in after_map:
            pairs.append((key[0], key[1], series_to_frame(before_series), None))
    return pairs


# ── Individual graph builders ───────────────────────────────────────────────
# Each builder renders one Figure for a single (symbol, source) series pair.
# Append new builders to _GRAPH_BUILDERS to add graphs dashboard-wide.


def graph_before_after_overlay(
    symbol: str,
    source: str,
    before_df: pd.DataFrame | None,
    after_df: pd.DataFrame | None,
) -> plt.Figure:
    """Overlay the raw ("before") and gap-filled ("after") price series."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if before_df is not None and not before_df.empty:
        sns.lineplot(
            data=before_df, x="date", y="price", ax=ax,
            label="Before (raw)", marker="o", markersize=4, alpha=0.65,
        )
    if after_df is not None and not after_df.empty:
        sns.lineplot(
            data=after_df, x="date", y="price", ax=ax,
            label="After (filled)", marker="s", markersize=4,
        )
    ax.set_title(f"{symbol} ({source}) — Before vs After")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def graph_filled_gaps(
    symbol: str,
    source: str,
    before_df: pd.DataFrame | None,
    after_df: pd.DataFrame | None,
) -> plt.Figure:
    """Highlight the filled series and overlay the originally observed points.

    Compact visualisation of where gap-filling reconstructed the series: the
    continuous filled line with the raw (non-missing) observations highlighted.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    if after_df is not None and not after_df.empty:
        sns.lineplot(
            data=after_df, x="date", y="price", ax=ax,
            label="Filled series", marker="o", markersize=3,
        )
    if before_df is not None and not before_df.empty:
        known = before_df.dropna(subset=["price"])
        if not known.empty:
            sns.scatterplot(
                data=known, x="date", y="price", ax=ax,
                color="red", s=32, label="Original observations",
            )
    ax.set_title(f"{symbol} ({source}) — Filled gaps")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


#: Registry of graph builders.  Append new builder functions here to add graphs.
_GRAPH_BUILDERS: list[GraphBuilder] = [
    graph_before_after_overlay,
    graph_filled_gaps,
]


def build_run_graphs(
    run_id: str,
    populated: dict[str, Any] | None = None,
    active_sources: list[str] | None = None,
) -> list[tuple[str, plt.Figure]]:
    """Build the configured set of graphs for a run's populated series.

    Retrieves the before/after time series for ``run_id`` from the DataStore
    (via :func:`dashboard_app.get_populated_timeseries`) unless pre-supplied,
    pairs them by ``(symbol, source)``, and runs every registered graph builder
    over each pair.

    Args:
        run_id: The run/session identifier used to look up the series.
        populated: Optional pre-loaded ``get_populated_timeseries`` result
            (used for testing / to avoid re-querying).
        active_sources: Optional restriction to a subset of data sources.
            Only pairs whose ``source`` is in this list (case-insensitive) are
            graphed.  When ``None`` or empty no filtering is applied.

    Returns:
        A list of ``(title, figure)`` tuples ready for the dashboard to render.

    Raises:
        ValueError: If no populated time series exist for the run when
            ``populated`` is not supplied.
    """
    if populated is None:
        from market_data_ai.dashboard_app import (
            get_populated_timeseries,
        )

        populated = get_populated_timeseries(run_id)

    pairs = _pair_series(populated.get("before", []), populated.get("after", []))
    if active_sources:
        allowed = {str(source).casefold() for source in active_sources}
        pairs = [
            (symbol, source, before_df, after_df)
            for symbol, source, before_df, after_df in pairs
            if str(source).casefold() in allowed
        ]

    graphs: list[tuple[str, plt.Figure]] = []
    for symbol, source, before_df, after_df in pairs:
        if before_df is None and after_df is None:
            continue
        # Missing sides render as None so builders must tolerate them; prevent
        # one failing graph from killing the whole tab.
        for builder in _GRAPH_BUILDERS:
            try:
                fig = builder(symbol, source, before_df, after_df)
            except Exception:  # noqa: BLE001 - one bad graph shouldn't kill the tab
                logger.exception(
                    "Graph builder failed symbol=%s source=%s builder=%s",
                    symbol, source, getattr(builder, "__name__", builder),
                )
                continue
            graphs.append((f"{symbol} ({source})", fig))

    logger.info("Built %d graph(s) for run_id=%s", len(graphs), run_id)
    return graphs


