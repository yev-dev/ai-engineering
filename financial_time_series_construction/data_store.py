"""SQLite-backed data store for time series and intermediate results.

This module is a thin compatibility wrapper.  All new code should import from
``time_series_database`` directly:

    from financial_time_series_construction.time_series_construction import (
        DataStore, get_datastore, init_datastore, reset_datastore,
    )

The schema is fully normalised with dimension tables
(``runs``, ``instruments``, ``sources``) and fact tables
(``raw_timeseries``, ``filled_timeseries``, ``quality_reports``, ``artifacts``).
Each row in the fact tables represents one price observation
``(trading_date, price)`` – no JSON columns are used for series data.
"""
from __future__ import annotations

import logging


# Re-export the canonical implementation from the new package.
from financial_time_series_construction.time_series_construction import (
    DataStore,
    close_datastore,
    get_datastore,
    init_datastore,
    reset_datastore,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DataStore",
    "close_datastore",
    "get_datastore",
    "init_datastore",
    "reset_datastore",
]