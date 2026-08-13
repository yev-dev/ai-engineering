"""Tests for the enhanced SQLite-backed DataStore with normalised schema."""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Generator

import pytest

from market_data_ai.database import (
    DataStore,
    get_datastore,
    init_datastore,
    reset_datastore,
    close_datastore,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_global() -> Generator[None, None, None]:
    """Reset the global singleton before and after each test."""
    reset_datastore()
    yield
    close_datastore()


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Provide a temporary database path for each test."""
    return tmp_path / "test_datastore.db"


@pytest.fixture
def store(tmp_db_path: Path) -> Generator[DataStore, None, None]:
    """Create a fresh DataStore backed by a temporary SQLite file."""
    ds = DataStore.create(tmp_db_path)
    yield ds
    ds.close()
    if tmp_db_path.exists():
        tmp_db_path.unlink()


@pytest.fixture
def sample_dates() -> list[str]:
    return ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]


@pytest.fixture
def sample_prices() -> list[float | None]:
    return [150.0, 151.5, None, 152.0, 153.25]


@pytest.fixture
def sample_report() -> dict[str, Any]:
    return {
        "completeness_pct": 80.0,
        "available_record_count": 4,
        "missing_count": 1,
        "min_date": "2024-01-01",
        "max_date": "2024-01-05",
        "issues": ["gap_detected"],
    }


# ── Initialisation ────────────────────────────────────────────────────


class TestDataStoreInit:
    """Tests for DataStore construction and schema creation."""

    def test_creates_db_file(self, tmp_db_path: Path) -> None:
        assert not tmp_db_path.exists()
        DataStore.create(tmp_db_path)
        assert tmp_db_path.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "nested.db"
        assert not nested.parent.exists()
        DataStore.create(nested)
        assert nested.exists()

    def test_schema_tables_exist(self, tmp_db_path: Path) -> None:
        DataStore.create(tmp_db_path)
        with sqlite3.connect(tmp_db_path) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        table_names = [row[0] for row in tables]
        # New normalised tables (dimensions + facts)
        assert "runs" in table_names
        assert "instruments" in table_names
        assert "sources" in table_names
        assert "raw_timeseries" in table_names
        assert "filled_timeseries" in table_names
        assert "quality_reports" in table_names
        assert "artifacts" in table_names
        # The new schema is static – no migrations table
        assert "migrations" not in table_names

    def test_normalised_observation_rows(self, store: DataStore) -> None:
        """Each (date, price) pair should be one row – not a JSON blob."""
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        prices = [150.0, None, 152.0]
        store.put_timeseries("run1", "AAPL", "yahoo", dates, prices)
        with sqlite3.connect(store.db_path) as conn:
            rows = conn.execute(
                "SELECT date, price FROM raw_timeseries ORDER BY date"
            ).fetchall()
        assert [(r[0], r[1]) for r in rows] == [
            ("2024-01-01", 150.0),
            ("2024-01-02", None),
            ("2024-01-03", 152.0),
        ]

    def test_dimension_tables_use_numeric_fks(self, store: DataStore) -> None:
        """Fact tables reference dimension tables via numeric foreign keys."""
        store.put_timeseries("run1", "AAPL", "yahoo", ["2024-01-01"], [150.0])
        with sqlite3.connect(store.db_path) as conn:
            row = conn.execute(
                "SELECT run_id, instrument_id, source_id, date, price "
                "FROM raw_timeseries"
            ).fetchone()
        assert isinstance(row[0], int)
        assert isinstance(row[1], int)
        assert isinstance(row[2], int)
        assert row[3] == "2024-01-01"
        assert row[4] == 150.0

    def test_no_backward_compat_views(self, tmp_db_path: Path) -> None:
        """The new schema has no backward-compat views."""
        DataStore.create(tmp_db_path)
        with sqlite3.connect(tmp_db_path) as conn:
            views = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name"
            ).fetchall()
        view_names = [row[0] for row in views]
        assert "timeseries" not in view_names
        assert "gap_filled_series" not in view_names

    def test_schema_indexes_exist(self, tmp_db_path: Path) -> None:
        DataStore.create(tmp_db_path)
        with sqlite3.connect(tmp_db_path) as conn:
            indexes = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
            ).fetchall()
        index_names = [row[0] for row in indexes]
        assert "idx_raw_ts_run" in index_names
        assert "idx_raw_ts_run_instrument" in index_names
        assert "idx_raw_ts_run_instrument_source" in index_names
        assert "idx_raw_ts_date" in index_names
        assert "idx_filled_ts_run" in index_names
        assert "idx_filled_ts_run_instrument" in index_names
        assert "idx_filled_ts_run_instrument_source" in index_names
        assert "idx_filled_ts_method" in index_names
        assert "idx_filled_ts_date" in index_names
        assert "idx_qr_run" in index_names
        assert "idx_art_run" in index_names
        assert "idx_art_run_type" in index_names

    def test_idempotent_schema_creation(self, tmp_db_path: Path) -> None:
        """Calling _init_schema twice should not raise."""
        ds = DataStore.create(tmp_db_path)
        ds._init_schema()  # no error expected

    def test_for_run_factory(self, tmp_path: Path) -> None:
        run_id = "test_run_001"
        ds = DataStore.for_run(tmp_path, run_id)
        expected = tmp_path / "time_series_database" / "database" / "datastore.db"
        assert ds.db_path == expected
        assert expected.exists()

    def test_for_run_returns_singleton(self, tmp_path: Path) -> None:
        """for_run should return the same singleton instance."""
        ds1 = DataStore.for_run(tmp_path, "run1")
        ds2 = DataStore.for_run(tmp_path, "run2")
        assert ds1 is ds2  # Same singleton

    def test_repr(self, tmp_db_path: Path) -> None:
        ds = DataStore.create(tmp_db_path)
        assert repr(ds) == f"DataStore(db_path={tmp_db_path!r})"

    def test_no_migrations_table(self, tmp_db_path: Path) -> None:
        """The new schema has no migrations table – schema is static."""
        DataStore.create(tmp_db_path)
        with sqlite3.connect(tmp_db_path) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        table_names = [row[0] for row in tables]
        assert "migrations" not in table_names


# ── Global Singleton ──────────────────────────────────────────────────


class TestDataStoreSingleton:
    """Tests for the global singleton pattern."""

    def test_get_datastore_raises_before_init(self) -> None:
        with pytest.raises(RuntimeError, match="DataStore has not been initialised"):
            get_datastore()

    def test_init_datastore(self, tmp_db_path: Path) -> None:
        ds = init_datastore(tmp_db_path)
        assert get_datastore() is ds

    def test_init_datastore_idempotent(self, tmp_db_path: Path) -> None:
        ds1 = init_datastore(tmp_db_path)
        ds2 = init_datastore(tmp_db_path)
        assert ds1 is ds2

    def test_reset_datastore(self, tmp_db_path: Path) -> None:
        init_datastore(tmp_db_path)
        reset_datastore()
        with pytest.raises(RuntimeError):
            get_datastore()

    def test_close_datastore(self, tmp_db_path: Path) -> None:
        init_datastore(tmp_db_path)
        close_datastore()
        with pytest.raises(RuntimeError):
            get_datastore()

    def test_create_replaces_singleton(self, tmp_db_path: Path) -> None:
        ds1 = DataStore.create(tmp_db_path)
        ds2 = DataStore.create(tmp_db_path)
        # create() resets and re-creates, so ds1 and ds2 are different instances
        # but both point to the same file
        assert ds1 is not ds2
        assert ds1.db_path == ds2.db_path


# ── Dimension Tables ─────────────────────────────────────────────────


class TestDataStoreDimensions:
    """Tests for dimension table operations."""

    def test_put_run_metadata(self, store: DataStore) -> None:
        run_id = store.put_run_metadata("run1", start_date="2024-01-01", end_date="2024-12-31")
        assert isinstance(run_id, int)
        assert run_id > 0

    def test_get_run_ids(self, store: DataStore) -> None:
        store.put_run_metadata("run1")
        store.put_run_metadata("run2")
        ids = store.get_run_ids()
        assert "run1" in ids
        assert "run2" in ids

    def test_get_instruments(self, store: DataStore) -> None:
        store.put_timeseries("run1", "AAPL", "yahoo", ["2024-01-01"], [150.0])
        store.put_timeseries("run1", "MSFT", "yahoo", ["2024-01-01"], [300.0])
        instruments = store.get_instruments()
        assert "AAPL" in instruments
        assert "MSFT" in instruments

    def test_get_sources(self, store: DataStore) -> None:
        store.put_timeseries("run1", "AAPL", "yahoo", ["2024-01-01"], [150.0])
        store.put_timeseries("run1", "AAPL", "bloomberg", ["2024-01-01"], [151.0])
        sources = store.get_sources()
        assert "yahoo" in sources
        assert "bloomberg" in sources


# ── Raw Timeseries ───────────────────────────────────────────────────


class TestDataStoreTimeseries:
    """Tests for put/get/list timeseries operations."""

    def test_put_and_get_timeseries(
        self, store: DataStore, sample_dates: list[str], sample_prices: list[float | None]
    ) -> None:
        data_ref = store.put_timeseries("run1", "AAPL", "yahoo", sample_dates, sample_prices)
        assert data_ref == "run1:AAPL:yahoo"

        result = store.get_timeseries(data_ref)
        assert result["run_id"] == "run1"
        assert result["symbol"] == "AAPL"
        assert result["source"] == "yahoo"
        assert result["dates"] == sample_dates
        assert result["prices"] == sample_prices

    def test_get_timeseries_raises_key_error(self, store: DataStore) -> None:
        with pytest.raises(KeyError, match="No timeseries found for data_ref: nonexistent"):
            store.get_timeseries("nonexistent")

    def test_put_timeseries_upserts(self, store: DataStore) -> None:
        """INSERT OR REPLACE should allow overwriting the same data_ref."""
        dates_a = ["2024-01-01"]
        prices_a = [100.0]
        dates_b = ["2024-01-01", "2024-01-02"]
        prices_b = [100.0, 101.0]

        data_ref = store.put_timeseries("run1", "AAPL", "yahoo", dates_a, prices_a)
        result_a = store.get_timeseries(data_ref)
        assert len(result_a["dates"]) == 1

        # Overwrite with same data_ref
        store.put_timeseries("run1", "AAPL", "yahoo", dates_b, prices_b)
        result_b = store.get_timeseries(data_ref)
        assert len(result_b["dates"]) == 2

    def test_list_timeseries_all(self, store: DataStore) -> None:
        store.put_timeseries("run1", "AAPL", "yahoo", ["2024-01-01"], [150.0])
        store.put_timeseries("run1", "MSFT", "yahoo", ["2024-01-01"], [300.0])
        store.put_timeseries("run1", "AAPL", "bloomberg", ["2024-01-01"], [151.0])

        results = store.list_timeseries("run1")
        assert len(results) == 3

    def test_list_timeseries_filter_by_symbol(self, store: DataStore) -> None:
        store.put_timeseries("run1", "AAPL", "yahoo", ["2024-01-01"], [150.0])
        store.put_timeseries("run1", "MSFT", "yahoo", ["2024-01-01"], [300.0])

        results = store.list_timeseries("run1", symbol="AAPL")
        assert len(results) == 1
        assert results[0]["symbol"] == "AAPL"

    def test_list_timeseries_filter_by_source(self, store: DataStore) -> None:
        store.put_timeseries("run1", "AAPL", "yahoo", ["2024-01-01"], [150.0])
        store.put_timeseries("run1", "AAPL", "bloomberg", ["2024-01-01"], [151.0])

        results = store.list_timeseries("run1", source="bloomberg")
        assert len(results) == 1
        assert results[0]["source"] == "bloomberg"

    def test_list_timeseries_filter_by_symbol_and_source(self, store: DataStore) -> None:
        store.put_timeseries("run1", "AAPL", "yahoo", ["2024-01-01"], [150.0])
        store.put_timeseries("run1", "AAPL", "bloomberg", ["2024-01-01"], [151.0])
        store.put_timeseries("run1", "MSFT", "yahoo", ["2024-01-01"], [300.0])

        results = store.list_timeseries("run1", symbol="AAPL", source="yahoo")
        assert len(results) == 1
        assert results[0]["symbol"] == "AAPL"
        assert results[0]["source"] == "yahoo"

    def test_list_timeseries_empty(self, store: DataStore) -> None:
        results = store.list_timeseries("nonexistent_run")
        assert results == []

    def test_list_timeseries_different_run_isolation(self, store: DataStore) -> None:
        store.put_timeseries("run1", "AAPL", "yahoo", ["2024-01-01"], [150.0])
        store.put_timeseries("run2", "MSFT", "yahoo", ["2024-01-01"], [300.0])

        results_run1 = store.list_timeseries("run1")
        results_run2 = store.list_timeseries("run2")
        assert len(results_run1) == 1
        assert len(results_run2) == 1
        assert results_run1[0]["symbol"] == "AAPL"
        assert results_run2[0]["symbol"] == "MSFT"

    def test_put_timeseries_with_none_prices(
        self, store: DataStore
    ) -> None:
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        prices: list[float | None] = [100.0, None, 102.0]
        data_ref = store.put_timeseries("run1", "AAPL", "yahoo", dates, prices)
        result = store.get_timeseries(data_ref)
        assert result["prices"] == [100.0, None, 102.0]

    def test_put_timeseries_empty_lists(self, store: DataStore) -> None:
        data_ref = store.put_timeseries("run1", "AAPL", "yahoo", [], [])
        result = store.get_timeseries(data_ref)
        assert result["dates"] == []
        assert result["prices"] == []

    def test_delete_timeseries(self, store: DataStore) -> None:
        data_ref = store.put_timeseries("run1", "AAPL", "yahoo", ["2024-01-01"], [150.0])
        assert store.delete_timeseries(data_ref) is True
        assert store.delete_timeseries(data_ref) is False  # Already deleted

    def test_dimensions_auto_populated(self, store: DataStore) -> None:
        """Putting a timeseries should auto-populate dimension tables."""
        store.put_timeseries("run1", "AAPL", "yahoo", ["2024-01-01"], [150.0])
        assert "run1" in store.get_run_ids()
        assert "AAPL" in store.get_instruments()
        assert "yahoo" in store.get_sources()


# ── Quality Reports ───────────────────────────────────────────────────


class TestDataStoreQualityReports:
    """Tests for put/get quality report operations."""

    def test_put_and_get_quality_report(
        self, store: DataStore, sample_report: dict[str, Any]
    ) -> None:
        report_id = store.put_quality_report("run1", sample_report, symbol="AAPL", source="yahoo")
        assert "run1:" in report_id
        assert "AAPL" in report_id
        assert "yahoo" in report_id

        result = store.get_quality_report(report_id)
        assert result["completeness_pct"] == 80.0
        assert result["issues"] == ["gap_detected"]

    def test_get_quality_report_raises_key_error(self, store: DataStore) -> None:
        with pytest.raises(KeyError, match="No quality report found for report_id: bad_id"):
            store.get_quality_report("bad_id")

    def test_put_quality_report_without_symbol_source(
        self, store: DataStore, sample_report: dict[str, Any]
    ) -> None:
        report_id = store.put_quality_report("run1", sample_report)
        assert "unknown" in report_id

        result = store.get_quality_report(report_id)
        assert result["completeness_pct"] == 80.0

    def test_put_quality_report_nested_dict(
        self, store: DataStore
    ) -> None:
        report = {
            "metrics": {"mean": 0.05, "std": 0.02},
            "flags": {"anomaly": False},
        }
        report_id = store.put_quality_report("run1", report)
        result = store.get_quality_report(report_id)
        assert result["metrics"]["mean"] == 0.05
        assert result["flags"]["anomaly"] is False

    def test_list_quality_reports(self, store: DataStore, sample_report: dict[str, Any]) -> None:
        store.put_quality_report("run1", sample_report, symbol="AAPL", source="yahoo")
        store.put_quality_report("run1", sample_report, symbol="MSFT", source="yahoo")
        reports = store.list_quality_reports("run1")
        assert len(reports) == 2


# ── Artifacts ─────────────────────────────────────────────────────────


class TestDataStoreArtifacts:
    """Tests for put/list artifact operations."""

    def test_put_artifact(self, store: DataStore) -> None:
        artifact_id = store.put_artifact("run1", "csv", "/path/to/file.csv", symbol="AAPL")
        assert "run1:csv:" in artifact_id

    def test_list_artifacts(self, store: DataStore) -> None:
        store.put_artifact("run1", "csv", "/path/to/file1.csv", symbol="AAPL")
        store.put_artifact("run1", "png", "/path/to/chart.png", symbol="MSFT")
        store.put_artifact("run1", "report", "/path/to/report.txt")

        results = store.list_artifacts("run1")
        assert len(results) == 3

    def test_list_artifacts_filter_by_type(self, store: DataStore) -> None:
        store.put_artifact("run1", "csv", "/path/to/file1.csv")
        store.put_artifact("run1", "png", "/path/to/chart.png")
        store.put_artifact("run1", "report", "/path/to/report.txt")

        results = store.list_artifacts("run1", artifact_type="csv")
        assert len(results) == 1
        assert results[0]["artifact_type"] == "csv"

    def test_list_artifacts_empty(self, store: DataStore) -> None:
        results = store.list_artifacts("nonexistent_run")
        assert results == []

    def test_list_artifacts_returns_all_fields(self, store: DataStore) -> None:
        store.put_artifact("run1", "csv", "/data/file.csv", symbol="AAPL", source="yahoo")
        results = store.list_artifacts("run1")
        assert len(results) == 1
        item = results[0]
        assert "artifact_id" in item
        assert item["artifact_type"] == "csv"
        assert item["path"] == "/data/file.csv"
        assert item["symbol"] == "AAPL"
        assert item["source"] == "yahoo"


# ── Gap-Filled Series ─────────────────────────────────────────────────


class TestDataStoreGapFilledSeries:
    """Tests for put/get gap-filled series operations."""

    def test_put_and_get_gap_filled_series(self, store: DataStore) -> None:
        original_dates = ["2024-01-01", "2024-01-03", "2024-01-05"]
        original_prices = [100.0, None, 102.0]
        filled_dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        filled_prices = [100.0, 101.0, 101.5, 101.75, 102.0]

        # First insert the raw timeseries so the FK constraint is satisfied
        store.put_timeseries("run1", "AAPL", "yahoo", original_dates, original_prices)

        data_ref = store.put_gap_filled_series(
            run_id="run1",
            symbol="AAPL",
            source="yahoo",
            method="linear_interpolation",
            original_dates=original_dates,
            original_prices=original_prices,
            filled_dates=filled_dates,
            filled_prices=filled_prices,
            original_data_ref="run1:AAPL:yahoo",
        )
        assert data_ref == "run1:AAPL:yahoo:filled"

        result = store.get_gap_filled_series(data_ref)
        assert result["run_id"] == "run1"
        assert result["symbol"] == "AAPL"
        assert result["source"] == "yahoo"
        assert result["method"] == "linear_interpolation"
        assert result["original_dates"] == original_dates
        assert result["original_prices"] == original_prices
        assert result["filled_dates"] == filled_dates
        assert result["filled_prices"] == filled_prices
        assert result["original_data_ref"] == "run1:AAPL:yahoo"

    def test_get_gap_filled_series_raises_key_error(self, store: DataStore) -> None:
        with pytest.raises(KeyError, match="No gap_filled_series found for data_ref: bad"):
            store.get_gap_filled_series("bad")

    def test_put_gap_filled_series_without_original_ref(self, store: DataStore) -> None:
        # Insert raw timeseries first for FK constraint
        store.put_timeseries("run1", "AAPL", "yahoo", ["2024-01-01"], [100.0])
        data_ref = store.put_gap_filled_series(
            run_id="run1",
            symbol="AAPL",
            source="yahoo",
            method="ffill",
            original_dates=["2024-01-01"],
            original_prices=[100.0],
            filled_dates=["2024-01-01", "2024-01-02"],
            filled_prices=[100.0, 100.0],
        )
        result = store.get_gap_filled_series(data_ref)
        # The original_data_ref is reconstructed from run_id/symbol/source
        # since it's derivable from the filled series' own identifiers.
        assert result["original_data_ref"] == "run1:AAPL:yahoo"

    def test_put_gap_filled_series_upserts(self, store: DataStore) -> None:
        """Same data_ref should be replaceable."""
        # Insert raw timeseries first for FK constraint
        store.put_timeseries("run1", "AAPL", "yahoo", ["2024-01-01"], [100.0])
        store.put_gap_filled_series(
            run_id="run1",
            symbol="AAPL",
            source="yahoo",
            method="ffill",
            original_dates=["2024-01-01"],
            original_prices=[100.0],
            filled_dates=["2024-01-01"],
            filled_prices=[100.0],
        )
        store.put_gap_filled_series(
            run_id="run1",
            symbol="AAPL",
            source="yahoo",
            method="linear",
            original_dates=["2024-01-01"],
            original_prices=[100.0],
            filled_dates=["2024-01-01", "2024-01-02"],
            filled_prices=[100.0, 101.0],
        )
        result = store.get_gap_filled_series("run1:AAPL:yahoo:filled")
        assert result["method"] == "linear"
        assert len(result["filled_dates"]) == 2

    def test_list_gap_filled_series(self, store: DataStore) -> None:
        # Insert raw timeseries first for FK constraint
        store.put_timeseries("run1", "AAPL", "yahoo", ["2024-01-01"], [100.0])
        store.put_timeseries("run1", "MSFT", "yahoo", ["2024-01-01"], [200.0])
        store.put_gap_filled_series(
            run_id="run1", symbol="AAPL", source="yahoo", method="linear",
            original_dates=["2024-01-01"], original_prices=[100.0],
            filled_dates=["2024-01-01"], filled_prices=[100.0],
        )
        store.put_gap_filled_series(
            run_id="run1", symbol="MSFT", source="yahoo", method="ffill",
            original_dates=["2024-01-01"], original_prices=[200.0],
            filled_dates=["2024-01-01"], filled_prices=[200.0],
        )
        results = store.list_gap_filled_series("run1")
        assert len(results) == 2


# ── Cross-Run Queries ────────────────────────────────────────────────


class TestDataStoreCrossRun:
    """Tests for cross-run query operations."""

    def test_list_runs_with_stats(self, store: DataStore) -> None:
        store.put_timeseries("run1", "AAPL", "yahoo", ["2024-01-01"], [150.0])
        store.put_gap_filled_series(
            run_id="run1", symbol="AAPL", source="yahoo", method="linear",
            original_dates=["2024-01-01"], original_prices=[150.0],
            filled_dates=["2024-01-01"], filled_prices=[150.0],
        )
        store.put_artifact("run1", "csv", "/path/to/file.csv")

        stats = store.list_runs_with_stats()
        assert len(stats) >= 1
        run1_stats = [s for s in stats if s["run_id"] == "run1"][0]
        assert run1_stats["timeseries_count"] >= 1
        assert run1_stats["filled_count"] >= 1
        assert run1_stats["artifact_count"] >= 1


# ── Edge Cases ────────────────────────────────────────────────────────


class TestDataStoreEdgeCases:
    """Tests for edge cases and error handling."""

    def test_close_is_noop(self, store: DataStore) -> None:
        """close() should not break subsequent operations."""
        store.close()
        # Should still work after close (re-opens connection)
        data_ref = store.put_timeseries("run1", "AAPL", "yahoo", ["2024-01-01"], [150.0])
        result = store.get_timeseries(data_ref)
        assert result["symbol"] == "AAPL"

    def test_concurrent_data_refs_different_runs(self, store: DataStore) -> None:
        """Same symbol/source in different runs should not conflict."""
        ref1 = store.put_timeseries("run1", "AAPL", "yahoo", ["2024-01-01"], [150.0])
        ref2 = store.put_timeseries("run2", "AAPL", "yahoo", ["2024-01-01"], [155.0])
        assert ref1 != ref2
        assert store.get_timeseries(ref1)["prices"] == [150.0]
        assert store.get_timeseries(ref2)["prices"] == [155.0]

    def test_large_payload(self, store: DataStore) -> None:
        """Store and retrieve a large time series."""
        from datetime import date, timedelta

        start = date(2024, 1, 1)
        dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(365)]
        prices = [float(i + 1) for i in range(365)]
        data_ref = store.put_timeseries("run1", "SPY", "yahoo", dates, prices)
        result = store.get_timeseries(data_ref)
        assert len(result["dates"]) == 365
        assert result["prices"][0] == 1.0
        assert result["prices"][-1] == 365.0

    def test_special_characters_in_symbol(self, store: DataStore) -> None:
        """Symbols with special characters should be handled."""
        data_ref = store.put_timeseries("run1", "BRK.B", "yahoo", ["2024-01-01"], [350.0])
        result = store.get_timeseries(data_ref)
        assert result["symbol"] == "BRK.B"

    def test_unicode_in_report(self, store: DataStore) -> None:
        """Unicode characters in report should be handled."""
        report = {"description": "Café au Lait — 100%"}
        report_id = store.put_quality_report("run1", report)
        result = store.get_quality_report(report_id)
        assert result["description"] == "Café au Lait — 100%"

    def test_vacuum(self, store: DataStore) -> None:
        """vacuum() should not raise."""
        store.put_timeseries("run1", "AAPL", "yahoo", ["2024-01-01"], [150.0])
        store.vacuum()  # no error expected


# ── Logging ───────────────────────────────────────────────────────────


class TestDataStoreLogging:
    """Tests that DataStore emits expected log messages."""

    def test_init_logs_info(self, tmp_db_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            DataStore.create(tmp_db_path)
        assert "DataStore initialised at" in caplog.text
        assert str(tmp_db_path) in caplog.text

    def test_for_run_logs_info(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            DataStore.for_run(tmp_path, "test_run")
        assert "DataStore initialised at" in caplog.text

    def test_put_timeseries_logs_info(
        self, store: DataStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            store.put_timeseries("run1", "AAPL", "yahoo", ["2024-01-01"], [150.0])
        assert "Timeseries stored: data_ref=run1:AAPL:yahoo" in caplog.text

    def test_get_timeseries_missing_logs_error(
        self, store: DataStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.ERROR):
            with pytest.raises(KeyError):
                store.get_timeseries("nonexistent")
        assert "Timeseries not found: data_ref=nonexistent" in caplog.text

    def test_put_quality_report_logs_info(
        self, store: DataStore, sample_report: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            store.put_quality_report("run1", sample_report, symbol="AAPL")
        assert "Quality report stored:" in caplog.text
        assert "AAPL" in caplog.text

    def test_get_quality_report_missing_logs_error(
        self, store: DataStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.ERROR):
            with pytest.raises(KeyError):
                store.get_quality_report("bad_id")
        assert "Quality report not found: report_id=bad_id" in caplog.text

    def test_put_artifact_logs_info(self, store: DataStore, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            store.put_artifact("run1", "csv", "/path/to/file.csv")
        assert "Artifact stored:" in caplog.text
        assert "/path/to/file.csv" in caplog.text

    def test_put_gap_filled_series_logs_info(
        self, store: DataStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            store.put_gap_filled_series(
                run_id="run1",
                symbol="AAPL",
                source="yahoo",
                method="linear",
                original_dates=["2024-01-01"],
                original_prices=[100.0],
                filled_dates=["2024-01-01"],
                filled_prices=[100.0],
            )
        assert "Gap-filled series stored:" in caplog.text
        assert "linear" in caplog.text

    def test_get_gap_filled_series_missing_logs_error(
        self, store: DataStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.ERROR):
            with pytest.raises(KeyError):
                store.get_gap_filled_series("bad_ref")
        assert "Gap-filled series not found: data_ref=bad_ref" in caplog.text

    def test_list_timeseries_logs_info(
        self, store: DataStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        store.put_timeseries("run1", "AAPL", "yahoo", ["2024-01-01"], [150.0])
        with caplog.at_level(logging.INFO):
            store.list_timeseries("run1")
        assert "Listed timeseries: run_id=run1" in caplog.text

    def test_list_artifacts_logs_info(
        self, store: DataStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        store.put_artifact("run1", "csv", "/path/to/file.csv")
        with caplog.at_level(logging.INFO):
            store.list_artifacts("run1")
        assert "Listed artifacts: run_id=run1" in caplog.text