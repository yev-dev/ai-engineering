"""SQLite-backed DataStore with a fully normalised schema and singleton management.

Schema overview
---------------

Dimension tables (look-up references):

* ``runs``               – one row per logical run/session (``run_id`` text unique).
* ``instruments``         – one row per ticker symbol.
* ``sources``             – one row per data source (yahoo, bloomberg, reuters).
* ``gap_filling_method``  – one row per gap-filling method name (e.g. ``linear_interpolation``).

Fact tables (one row **per price observation** – no JSON blobs):

* ``raw_timeseries``    – ``(date, price)`` for the original data.
* ``filled_timeseries`` – ``(date, price)`` after gap-filling.  References
  ``gap_filling_method`` via a numeric ``method_id`` foreign key.  The link
  back to the original (raw) series is established by ``run_id``,
  ``instrument_id`` and ``source_id`` – no ``original_data_ref`` text column
  is stored.

Metadata tables:

* ``quality_reports`` – quality report document per (run, optional symbol/source).
* ``artifacts``       – file artifact produced during a run.

There is no migration table – this project keeps the schema simple by design.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Global singleton ────────────────────────────────────────────────────────

_DEFAULT_DATASTORE: DataStore | None = None


def get_datastore() -> DataStore:
    """Return the global singleton DataStore instance.

    Raises:
        RuntimeError: If ``init_datastore()`` has not been called yet.
    """
    if _DEFAULT_DATASTORE is None:
        raise RuntimeError(
            "DataStore has not been initialised. Call init_datastore(db_path) "
            "or DataStore.create(db_path) first."
        )
    return _DEFAULT_DATASTORE


def init_datastore(db_path: Path) -> DataStore:
    """Initialise the global singleton DataStore.

    Subsequent calls return the same instance (idempotent).  To force
    re-initialisation call ``reset_datastore()`` first.

    Args:
        db_path: Path to the SQLite database file.
    """
    global _DEFAULT_DATASTORE
    if _DEFAULT_DATASTORE is None:
        _DEFAULT_DATASTORE = DataStore(db_path)
        logger.info("Global DataStore initialised at %s", db_path)
    else:
        logger.debug("Global DataStore already initialised at %s", _DEFAULT_DATASTORE.db_path)
    return _DEFAULT_DATASTORE


def reset_datastore() -> None:
    """Reset the global singleton (useful for testing)."""
    global _DEFAULT_DATASTORE
    _DEFAULT_DATASTORE = None


def close_datastore() -> None:
    """Close the global singleton and reset it."""
    global _DEFAULT_DATASTORE
    if _DEFAULT_DATASTORE is not None:
        _DEFAULT_DATASTORE.close()
        _DEFAULT_DATASTORE = None


# ── Dimension helpers ────────────────────────────────────────────────────


def _ensure_run(conn: sqlite3.Connection, run_id: str) -> int:
    """Get or create a run row, returning its numeric id.

    If the run already exists its ``updated_at`` column is touched.
    """
    cursor = conn.execute("SELECT id FROM runs WHERE run_id = ?", (run_id,))
    row = cursor.fetchone()
    if row is not None:
        conn.execute(
            "UPDATE runs SET updated_at = datetime('now') WHERE id = ?",
            (row[0],),
        )
        return row[0]
    cursor = conn.execute(
        "INSERT INTO runs (run_id) VALUES (?)", (run_id,)
    )
    return cursor.lastrowid  # type: ignore[return-value]


def _ensure_instrument(conn: sqlite3.Connection, symbol: str) -> int:
    """Get or create an instrument row, returning its numeric id."""
    cursor = conn.execute("SELECT id FROM instruments WHERE symbol = ?", (symbol,))
    row = cursor.fetchone()
    if row is not None:
        return row[0]
    cursor = conn.execute(
        "INSERT INTO instruments (symbol) VALUES (?)", (symbol,)
    )
    return cursor.lastrowid  # type: ignore[return-value]


def _ensure_source(conn: sqlite3.Connection, source: str) -> int:
    """Get or create a source row, returning its numeric id."""
    cursor = conn.execute("SELECT id FROM sources WHERE name = ?", (source,))
    row = cursor.fetchone()
    if row is not None:
        return row[0]
    cursor = conn.execute(
        "INSERT INTO sources (name) VALUES (?)", (source,)
    )
    return cursor.lastrowid  # type: ignore[return-value]


def _ensure_run_source(conn: sqlite3.Connection, run_id: str, source: str) -> None:
    """Record that ``source`` is an active data source for ``run_id``.

    This is the denormalised "run_id -> active data sources" reference the
    dashboard and HITL use to know which sources are available for a run.  It
    is an idempotent upsert, so re-storing a series for the same source has no
    side effects.
    """
    run_num = _ensure_run(conn, run_id)
    src_num = _ensure_source(conn, source)
    conn.execute(
        "INSERT OR IGNORE INTO run_sources (run_id, source_id) VALUES (?, ?)",
        (run_num, src_num),
    )


def _ensure_method(conn: sqlite3.Connection, method: str) -> int:
    """Get or create a gap_filling_method row, returning its numeric id.

    Args:
        conn: An active SQLite connection.
        method: The gap-filling method name (e.g. ``linear_interpolation``).

    Returns:
        The numeric id of the method row.
    """
    cursor = conn.execute("SELECT id FROM gap_filling_method WHERE name = ?", (method,))
    row = cursor.fetchone()
    if row is not None:
        return row[0]
    cursor = conn.execute(
        "INSERT INTO gap_filling_method (name) VALUES (?)", (method,)
    )
    return cursor.lastrowid  # type: ignore[return-value]


def _find_run(conn: sqlite3.Connection, run_id: str) -> int | None:
    """Return the numeric id of a run row, or None if it does not exist."""
    row = conn.execute(
        "SELECT id FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    return row[0] if row is not None else None


def _find_instrument(conn: sqlite3.Connection, symbol: str) -> int | None:
    """Return the numeric id of an instrument row, or None if it does not exist."""
    row = conn.execute(
        "SELECT id FROM instruments WHERE symbol = ?", (symbol,)
    ).fetchone()
    return row[0] if row is not None else None


def _find_source(conn: sqlite3.Connection, source: str) -> int | None:
    """Return the numeric id of a source row, or None if it does not exist."""
    row = conn.execute(
        "SELECT id FROM sources WHERE name = ?", (source,)
    ).fetchone()
    return row[0] if row is not None else None


def _find_method(conn: sqlite3.Connection, method: str) -> int | None:
    """Return the numeric id of a gap_filling_method row, or None if it does not exist."""
    row = conn.execute(
        "SELECT id FROM gap_filling_method WHERE name = ?", (method,)
    ).fetchone()
    return row[0] if row is not None else None


# ── DataStore ───────────────────────────────────────────────────────────────


class DataStore:
    """Centralised SQLite-backed data store with a fully normalised schema.

    Time-series data is stored in fully normalised form – each row in the
    fact tables (``raw_timeseries`` / ``filled_timeseries``) represents one
    price observation ``(date, price)`` and references the dimension
    tables via numeric foreign keys.  No JSON columns are used for series.

    The singleton helpers ``get_datastore()`` / ``init_datastore()`` ensure
    that only **one** database file is ever opened during a process lifetime.
    """

    def __init__(self, db_path: Path) -> None:
        """Initialise the data store at *db_path*.

        Args:
            db_path: Path to the SQLite database file.  Parent directories
                are created automatically.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._init_schema()
        logger.info("DataStore initialised at %s", self.db_path)
        logger.debug("datastore_initialized path=%s", self.db_path)

    # ── Connection management ──────────────────────────────────────────────

    @property
    def connection(self) -> sqlite3.Connection:
        """Return a persistent connection (created lazily).

        The connection is created with ``check_same_thread=False`` so it can
        be shared across threads (e.g. the dashboard's background worker
        thread and the main Streamlit thread).  All public methods acquire
        ``self._lock`` to serialise access and prevent concurrent writes.
        """
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
        return self._connection

    def close(self) -> None:
        """Close the persistent database connection."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __del__(self) -> None:
        self.close()

    # ── Factory ────────────────────────────────────────────────────────────

    @classmethod
    def for_run(cls, output_root: Path, run_id: str) -> "DataStore":
        """Create (or return the singleton) DataStore for a specific run.

        When called for the first time the database is created in the
        one-and-only ``time_series_database/database/`` subfolder of
        *output_root*, ensuring only one database is ever used.

        Args:
            output_root: Root output directory (e.g. ``~/time_series_construction``).
            run_id: The run/session identifier (used to track metadata,
                not to create separate databases).

        Returns:
            A DataStore instance backed by
            ``<output_root>/time_series_database/database/datastore.db``.
        """
        db_path = output_root / "time_series_database" / "database" / "datastore.db"
        try:
            return init_datastore(db_path)
        except RuntimeError:
            return DataStore(db_path)

    @classmethod
    def create(cls, db_path: Path) -> "DataStore":
        """Create a new DataStore and register it as the global singleton.

        Unlike ``init_datastore`` this will **replace** an existing singleton.
        Useful for testing.
        """
        reset_datastore()
        return init_datastore(db_path)

    # ── Schema ────────────────────────────────────────────────────────────
    #
    # Dimension tables:
    #   runs                 – id, run_id, start_date, end_date, created_at, updated_at
    #   instruments          – id, symbol, created_at
    #   sources              – id, name, created_at
    #   gap_filling_method   – id, name, description, created_at
    #
    # Fact tables (one row per observation):
    #   raw_timeseries       – id, run_id(FK), instrument_id(FK), source_id(FK),
    #                          date, price, created_at
    #   filled_timeseries    – id, run_id(FK), instrument_id(FK), source_id(FK),
    #                          method_id(FK→gap_filling_method), date, price, created_at
    #
    # Metadata tables:
    #   quality_reports      – report_id, run_id(FK), instrument_id(FK), source_id(FK),
    #                          report, created_at
    #   artifacts            – artifact_id, run_id(FK), instrument_id(FK), source_id(FK),
    #                          artifact_type, path, created_at
    #
    # ────────────────────────────────────────────────────────────────────────

    _SCHEMA_SQL = """
        CREATE TABLE IF NOT EXISTS runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT NOT NULL UNIQUE,
            start_date  TEXT,
            end_date    TEXT,
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS instruments (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol     TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS sources (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS gap_filling_method (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL UNIQUE,
            description TEXT,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS raw_timeseries (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id        INTEGER NOT NULL,
            instrument_id INTEGER NOT NULL,
            source_id     INTEGER NOT NULL,
            date          TEXT NOT NULL,
            price         REAL,
            created_at    TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE (run_id, instrument_id, source_id, date),
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE,
            FOREIGN KEY (instrument_id) REFERENCES instruments(id),
            FOREIGN KEY (source_id) REFERENCES sources(id)
        );

        CREATE TABLE IF NOT EXISTS filled_timeseries (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id        INTEGER NOT NULL,
            instrument_id INTEGER NOT NULL,
            source_id     INTEGER NOT NULL,
            method_id     INTEGER NOT NULL,
            date          TEXT NOT NULL,
            price         REAL NOT NULL,
            created_at    TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE (run_id, instrument_id, source_id, date),
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE,
            FOREIGN KEY (instrument_id) REFERENCES instruments(id),
            FOREIGN KEY (source_id) REFERENCES sources(id),
            FOREIGN KEY (method_id) REFERENCES gap_filling_method(id)
        );

        CREATE TABLE IF NOT EXISTS quality_reports (
            report_id     TEXT PRIMARY KEY,
            run_id        INTEGER NOT NULL,
            instrument_id INTEGER,
            source_id     INTEGER,
            report        TEXT NOT NULL,
            created_at    TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE,
            FOREIGN KEY (instrument_id) REFERENCES instruments(id),
            FOREIGN KEY (source_id) REFERENCES sources(id)
        );

        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id   TEXT PRIMARY KEY,
            run_id        INTEGER NOT NULL,
            instrument_id INTEGER,
            source_id     INTEGER,
            artifact_type TEXT NOT NULL,
            path          TEXT NOT NULL,
            created_at    TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE,
            FOREIGN KEY (instrument_id) REFERENCES instruments(id),
            FOREIGN KEY (source_id) REFERENCES sources(id)
        );

        CREATE TABLE IF NOT EXISTS files (
            file_id      TEXT PRIMARY KEY,
            filename     TEXT NOT NULL,
            description  TEXT,
            path         TEXT NOT NULL,
            created_at   TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS run_sources (
            run_id    INTEGER NOT NULL,
            source_id INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (run_id, source_id),
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE,
            FOREIGN KEY (source_id) REFERENCES sources(id)
        );

        CREATE TABLE IF NOT EXISTS run_input_metadata (
            run_id     INTEGER PRIMARY KEY,
            metadata   TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_raw_ts_run
            ON raw_timeseries (run_id);
        CREATE INDEX IF NOT EXISTS idx_raw_ts_run_instrument
            ON raw_timeseries (run_id, instrument_id);
        CREATE INDEX IF NOT EXISTS idx_raw_ts_run_instrument_source
            ON raw_timeseries (run_id, instrument_id, source_id);
        CREATE INDEX IF NOT EXISTS idx_raw_ts_date
            ON raw_timeseries (date);
        CREATE INDEX IF NOT EXISTS idx_filled_ts_run
            ON filled_timeseries (run_id);
        CREATE INDEX IF NOT EXISTS idx_filled_ts_run_instrument
            ON filled_timeseries (run_id, instrument_id);
        CREATE INDEX IF NOT EXISTS idx_filled_ts_run_instrument_source
            ON filled_timeseries (run_id, instrument_id, source_id);
        CREATE INDEX IF NOT EXISTS idx_filled_ts_method
            ON filled_timeseries (method_id);
        CREATE INDEX IF NOT EXISTS idx_filled_ts_date
            ON filled_timeseries (date);
        CREATE INDEX IF NOT EXISTS idx_qr_run
            ON quality_reports (run_id);
        CREATE INDEX IF NOT EXISTS idx_art_run
            ON artifacts (run_id);
        CREATE INDEX IF NOT EXISTS idx_art_run_type
            ON artifacts (run_id, artifact_type);
        CREATE INDEX IF NOT EXISTS idx_files_created
            ON files (created_at);
        CREATE INDEX IF NOT EXISTS idx_run_sources_run
            ON run_sources (run_id);
    """

    def _init_schema(self) -> None:
        """Create the full schema (idempotent – ``CREATE IF NOT EXISTS``).

        If a legacy or incompatible schema is detected (e.g. a database
        created by a previous version that used denormalised ``symbol``/
        ``source`` text columns instead of numeric ``instrument_id``/
        ``source_id`` foreign keys, or still uses ``trading_date`` instead
        of ``date``, or still has the ``original_data_ref`` column),
        **all** existing tables, views, and indexes are dropped and the
        schema is rebuilt from scratch.

        This follows the project's "no migrations table" philosophy: the
        schema is kept static and a full rebuild is preferred over
        incremental migration.
        """
        conn = self.connection

        if not self._schema_is_compatible(conn):
            logger.warning(
                "Incompatible legacy schema detected at %s; "
                "dropping all objects and recreating with the current schema.",
                self.db_path,
            )
            self._drop_all_objects(conn)

        conn.executescript(self._SCHEMA_SQL)
        conn.commit()

    # ── Schema compatibility ────────────────────────────────────────────

    def _schema_is_compatible(self, conn: sqlite3.Connection) -> bool:
        """Check whether the existing database schema matches the current one.

        Returns ``True`` for an empty database or one whose ``raw_timeseries``
        and ``filled_timeseries`` tables already conform to the current
        normalised layout (including the ``gap_filling_method`` dimension
        table, ``date`` columns, ``method_id`` foreign key, and no
        ``original_data_ref`` column).  Any legacy layout is treated as
        incompatible.

        Args:
            conn: An active SQLite connection.

        Returns:
            ``True`` if the schema is compatible (or the DB is fresh),
            ``False`` otherwise.
        """
        table_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }

        if not table_names:
            return True  # Fresh database – nothing to migrate.

        # The legacy schema had a 'migrations' table; the current one never does.
        if "migrations" in table_names:
            return False

        # The new schema requires the gap_filling_method dimension table.
        if "gap_filling_method" not in table_names:
            return False

        # Spot-check that fact tables use the current column layout.
        if "raw_timeseries" in table_names:
            columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(raw_timeseries)"
                ).fetchall()
            }
            if "date" not in columns or "trading_date" in columns:
                return False

        if "filled_timeseries" in table_names:
            columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(filled_timeseries)"
                ).fetchall()
            }
            if "date" not in columns or "trading_date" in columns:
                return False
            # original_data_ref was removed – its presence means legacy schema.
            if "original_data_ref" in columns:
                return False
            # method_id FK replaces the old text-based 'method' column.
            if "method_id" not in columns:
                return False

        return True

    def _drop_all_objects(self, conn: sqlite3.Connection) -> None:
        """Drop every user-defined table, view, and index.

        Called when a legacy/incompatible schema is detected.  Foreign-key
        enforcement is temporarily disabled so that tables can be dropped in
        any order.  After this call the database is empty and ready for the
        new schema to be applied.
        """
        # PRAGMA foreign_keys must be set outside a transaction.
        conn.commit()
        conn.execute("PRAGMA foreign_keys=OFF")

        # Drop views first (they may reference tables).
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name NOT LIKE 'sqlite_%'"
        ).fetchall():
            conn.execute(f'DROP VIEW IF EXISTS "{name}"')

        # Drop indexes.
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        ).fetchall():
            conn.execute(f'DROP INDEX IF EXISTS "{name}"')

        # Drop tables last (fact tables reference dimension tables).
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall():
            conn.execute(f'DROP TABLE IF EXISTS "{name}"')

        conn.commit()
        conn.execute("PRAGMA foreign_keys=ON")

    # ── Dimension helpers (public) ─────────────────────────────────────────

    def put_run_metadata(
        self,
        run_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> int:
        """Register or update a run with optional date metadata.

        Args:
            run_id: The run/session identifier.
            start_date: Optional ISO start date.
            end_date: Optional ISO end date.

        Returns:
            The numeric id of the run row.
        """
        conn = self.connection
        run_id_num = _ensure_run(conn, run_id)
        if start_date is not None or end_date is not None:
            updates: list[str] = []
            params: list[Any] = []
            if start_date is not None:
                updates.append("start_date = ?")
                params.append(start_date)
            if end_date is not None:
                updates.append("end_date = ?")
                params.append(end_date)
            updates.append("updated_at = datetime('now')")
            params.append(run_id_num)
            conn.execute(
                f"UPDATE runs SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()
        return run_id_num

    def put_run_input_metadata(self, run_id: str, metadata: dict[str, Any]) -> int:
        """Store a JSON blob of input metadata for a run (selected files, asset, dates).

        Args:
            run_id: The run/session identifier.
            metadata: Dict of input metadata to persist.

        Returns:
            The numeric id of the run row.
        """
        conn = self.connection
        run_id_num = _ensure_run(conn, run_id)
        json_text = json.dumps(metadata or {})
        # Use INSERT OR REPLACE to create or update the single-row metadata record.
        conn.execute(
            "INSERT OR REPLACE INTO run_input_metadata (run_id, metadata, created_at, updated_at) VALUES (?, ?, datetime('now'), datetime('now'))",
            (run_id_num, json_text),
        )
        conn.commit()
        return run_id_num

    def get_run_input_metadata(self, run_id: str) -> dict[str, Any]:
        """Retrieve the input metadata JSON blob for a run.

        Returns an empty dict when no metadata exists for the run.
        """
        conn = self.connection
        run_num = _find_run(conn, run_id)
        if run_num is None:
            return {}
        row = conn.execute(
            "SELECT metadata FROM run_input_metadata WHERE run_id = ?",
            (run_num,),
        ).fetchone()
        if row is None or row[0] is None:
            return {}
        try:
            return json.loads(row[0])
        except Exception:
            return {}

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Return a single run record by its ``run_id``.

        Args:
            run_id: The run/session identifier.

        Returns:
            Dict with ``run_id``, ``start_date``, ``end_date``,
            ``created_at``, ``updated_at`` plus summary stats
            (``timeseries_count``, ``filled_count``, ``artifact_count``).

        Raises:
            KeyError: If no run with the given ``run_id`` exists.
        """
        conn = self.connection
        row = conn.execute(
            """
            SELECT r.run_id, r.start_date, r.end_date,
                   r.created_at, r.updated_at,
                   (SELECT COUNT(*) FROM raw_timeseries rt
                    WHERE rt.run_id = r.id) AS ts_count,
                   (SELECT COUNT(*) FROM filled_timeseries ft
                    WHERE ft.run_id = r.id) AS filled_count,
                   (SELECT COUNT(*) FROM artifacts a
                    WHERE a.run_id = r.id) AS art_count
            FROM runs r
            WHERE r.run_id = ?
            """,
            (run_id,),
        ).fetchone()

        if row is None:
            raise KeyError(f"No run found for run_id: {run_id}")

        logger.debug("run_retrieved run_id=%s", run_id)
        return {
            "run_id": row[0],
            "start_date": row[1],
            "end_date": row[2],
            "created_at": row[3],
            "updated_at": row[4],
            "timeseries_count": row[5],
            "filled_count": row[6],
            "artifact_count": row[7],
        }

    def get_run_ids(self) -> list[str]:
        """Return all known run IDs ordered by creation time."""
        conn = self.connection
        return [
            row[0]
            for row in conn.execute(
                "SELECT run_id FROM runs ORDER BY created_at"
            ).fetchall()
        ]

    def get_run_sources(self, run_id: str) -> list[str]:
        """Return the active data sources recorded for a run.

        Falls back to deriving sources from the run's stored series when the
        explicit ``run_sources`` reference is empty (e.g. for runs created by
        older code paths that never populated the reference).

        Args:
            run_id: The run/session identifier.

        Returns:
            Sorted list of active source names for the run (e.g. ``yahoo``).
        """
        conn = self.connection
        run_num = _find_run(conn, run_id)
        if run_num is None:
            return []

        rows = conn.execute(
            """
            SELECT s.name
            FROM run_sources rs
            JOIN sources s ON s.id = rs.source_id
            WHERE rs.run_id = ?
            ORDER BY s.name
            """,
            (run_num,),
        ).fetchall()
        stored = [row[0] for row in rows]
        if stored:
            return stored

        # Fallback: derive active sources from the series stored for the run.
        fallback = conn.execute(
            """
            SELECT s.name
            FROM (
                SELECT source_id FROM raw_timeseries WHERE run_id = ?
                UNION
                SELECT source_id FROM filled_timeseries WHERE run_id = ?
            ) t
            JOIN sources s ON s.id = t.source_id
            ORDER BY s.name
            """,
            (run_num, run_num),
        ).fetchall()
        return [row[0] for row in fallback]

    def get_instruments(self) -> list[str]:
        """Return all known instrument symbols."""
        conn = self.connection
        return [
            row[0]
            for row in conn.execute(
                "SELECT symbol FROM instruments ORDER BY symbol"
            ).fetchall()
        ]

    def get_sources(self) -> list[str]:
        """Return all known source names."""
        conn = self.connection
        return [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sources ORDER BY name"
            ).fetchall()
        ]

    def get_gap_filling_methods(self) -> list[str]:
        """Return all known gap-filling method names.

        Returns:
            A sorted list of method name strings.
        """
        conn = self.connection
        return [
            row[0]
            for row in conn.execute(
                "SELECT name FROM gap_filling_method ORDER BY name"
            ).fetchall()
        ]

    # ── Raw Timeseries ──────────────────────────────────────────────────────

    def put_timeseries(
        self,
        run_id: str,
        symbol: str,
        source: str,
        dates: list[str],
        prices: list[Any],
    ) -> str:
        """Store a raw time series and return a reference key.

        The series is stored as one normalised row per ``(date, price)``
        observation.  Reinserting the same ``run_id``/``symbol``/``source``
        replaces the previous series.

        Args:
            run_id: The run/session identifier.
            symbol: Ticker symbol.
            source: Data source name.
            dates: List of ISO date strings.
            prices: List of price values (float or None for missing).

        Returns:
            A ``data_ref`` string of the form ``<run_id>:<symbol>:<source>``.
        """
        data_ref = f"{run_id}:{symbol}:{source}"
        conn = self.connection
        run_num = _ensure_run(conn, run_id)
        inst_num = _ensure_instrument(conn, symbol)
        src_num = _ensure_source(conn, source)

        # Replace the existing series (if any) with the new observations.
        conn.execute(
            "DELETE FROM raw_timeseries "
            "WHERE run_id = ? AND instrument_id = ? AND source_id = ?",
            (run_num, inst_num, src_num),
        )
        conn.executemany(
            """
            INSERT INTO raw_timeseries
                (run_id, instrument_id, source_id, date, price)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (run_num, inst_num, src_num, date_str, price)
                for date_str, price in zip(dates, prices)
            ],
        )
        conn.commit()
        # Record this source as an active data source for the run so run_id
        # carries a persistent reference to the sources used by this run.
        _ensure_run_source(conn, run_id, source)
        logger.info(
            "Timeseries stored: data_ref=%s symbol=%s source=%s count=%d",
            data_ref, symbol, source, len(dates),
        )
        logger.debug(
            "timeseries_stored data_ref=%s symbol=%s source=%s count=%d",
            data_ref, symbol, source, len(dates),
        )
        return data_ref

    def get_timeseries(self, data_ref: str) -> dict[str, Any]:
        """Load a raw time series by its reference key.

        Args:
            data_ref: The reference key returned by ``put_timeseries``.

        Returns:
            A dict with ``run_id``, ``symbol``, ``source``, ``dates``, ``prices``.

        Raises:
            KeyError: If no time series is found for the given reference.
        """
        try:
            run_id, symbol, source = _parse_data_ref(data_ref)
        except KeyError:
            logger.error("Timeseries not found: data_ref=%s", data_ref)
            raise KeyError(f"No timeseries found for data_ref: {data_ref}")
        conn = self.connection

        run_num = _find_run(conn, run_id)
        inst_num = _find_instrument(conn, symbol)
        src_num = _find_source(conn, source)
        if run_num is None or inst_num is None or src_num is None:
            logger.error("Timeseries not found: data_ref=%s", data_ref)
            raise KeyError(f"No timeseries found for data_ref: {data_ref}")

        rows = conn.execute(
            """
            SELECT date, price
            FROM raw_timeseries
            WHERE run_id = ? AND instrument_id = ? AND source_id = ?
            ORDER BY date
            """,
            (run_num, inst_num, src_num),
        ).fetchall()

        logger.debug("timeseries_retrieved data_ref=%s symbol=%s", data_ref, symbol)
        return {
            "run_id": run_id,
            "symbol": symbol,
            "source": source,
            "dates": [row[0] for row in rows],
            "prices": [row[1] for row in rows],
        }

    def list_timeseries(
        self,
        run_id: str,
        symbol: str | None = None,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        """List stored raw time series, optionally filtered.

        Args:
            run_id: The run/session identifier.
            symbol: If provided, filter by symbol.
            source: If provided, filter by source.

        Returns:
            List of dicts with ``data_ref``, ``symbol``, ``source``, ``dates``, ``prices``.
        """
        query = """
            SELECT r.run_id, i.symbol, s.name,
                   rt.date, rt.price
            FROM raw_timeseries rt
            JOIN runs r        ON r.id = rt.run_id
            JOIN instruments i ON i.id = rt.instrument_id
            JOIN sources s     ON s.id = rt.source_id
            WHERE r.run_id = ?
        """
        params: list[Any] = [run_id]
        if symbol:
            query += " AND i.symbol = ?"
            params.append(symbol)
        if source:
            query += " AND s.name = ?"
            params.append(source)
        query += " ORDER BY i.symbol, s.name, rt.date"

        conn = self.connection
        rows = conn.execute(query, params).fetchall()

        # Group observation rows back into list-based series payloads.
        series_map: dict[tuple[str, str], dict[str, Any]] = {}
        for row_run_id, row_symbol, row_source, date_str, price in rows:
            key = (row_symbol, row_source)
            if key not in series_map:
                series_map[key] = {
                    "data_ref": f"{row_run_id}:{row_symbol}:{row_source}",
                    "symbol": row_symbol,
                    "source": row_source,
                    "dates": [],
                    "prices": [],
                }
            series_map[key]["dates"].append(date_str)
            series_map[key]["prices"].append(price)

        results = list(series_map.values())
        logger.info(
            "Listed timeseries: run_id=%s symbol=%s source=%s count=%d",
            run_id, symbol or "*", source or "*", len(results),
        )
        return results

    def delete_timeseries(self, data_ref: str) -> bool:
        """Delete a raw time series by its reference key.

        Args:
            data_ref: The reference key to delete.

        Returns:
            True if rows were deleted, False otherwise.
        """
        try:
            run_id, symbol, source = _parse_data_ref(data_ref)
        except KeyError:
            return False

        conn = self.connection
        run_num = _find_run(conn, run_id)
        inst_num = _find_instrument(conn, symbol)
        src_num = _find_source(conn, source)
        if run_num is None or inst_num is None or src_num is None:
            return False

        cursor = conn.execute(
            "DELETE FROM raw_timeseries "
            "WHERE run_id = ? AND instrument_id = ? AND source_id = ?",
            (run_num, inst_num, src_num),
        )
        conn.commit()
        return cursor.rowcount > 0

    # ── Filled (Gap-Filled) Timeseries ───────────────────────────────────────

    def put_gap_filled_series(
        self,
        run_id: str,
        symbol: str,
        source: str,
        method: str,
        filled_dates: list[str],
        filled_prices: list[Any],
        original_dates: list[str] | None = None,
        original_prices: list[Any] | None = None,
        original_data_ref: str | None = None,
    ) -> str:
        """Store a gap-filled time series and return a reference key.

        The series is stored as one normalised row per ``(date, price)``
        observation.  Reinserting the same ``run_id``/``symbol``/``source``
        replaces the previous filled series.  The ``method`` name is
        resolved to a numeric ``method_id`` foreign key referencing the
        ``gap_filling_method`` dimension table.

        Args:
            run_id: The run/session identifier.
            symbol: Ticker symbol.
            source: Data source name.
            method: Gap-filling method used (e.g.
                ``linear_interpolation``).
            filled_dates: Date strings after filling.
            filled_prices: Price values after filling.
            original_dates: Optional original (pre-filling) date strings.
            original_prices: Optional original (pre-filling) price values.
            original_data_ref: Optional reference to the raw source series.

        Returns:
            A ``data_ref`` string of the form ``<run_id>:<symbol>:<source>:filled``.
        """
        data_ref = f"{run_id}:{symbol}:{source}:filled"
        conn = self.connection
        run_num = _ensure_run(conn, run_id)
        inst_num = _ensure_instrument(conn, symbol)
        src_num = _ensure_source(conn, source)
        method_num = _ensure_method(conn, method)

        # Replace the existing filled series for this run/symbol/source.
        conn.execute(
            "DELETE FROM filled_timeseries "
            "WHERE run_id = ? AND instrument_id = ? AND source_id = ?",
            (run_num, inst_num, src_num),
        )
        conn.executemany(
            """
            INSERT INTO filled_timeseries
                (run_id, instrument_id, source_id, method_id, date, price)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_num,
                    inst_num,
                    src_num,
                    method_num,
                    date_str,
                    price,
                )
                for date_str, price in zip(filled_dates, filled_prices)
            ],
        )
        conn.commit()
        # Keep the run's active-source reference up to date for this run.
        _ensure_run_source(conn, run_id, source)
        logger.info(
            "Gap-filled series stored: data_ref=%s symbol=%s source=%s method=%s observations=%d",
            data_ref, symbol, source, method, len(filled_dates),
        )
        logger.debug(
            "gap_filled_series_stored data_ref=%s symbol=%s source=%s method=%s original_ref=%s",
            data_ref, symbol, source, method, original_data_ref,
        )
        return data_ref

    def get_gap_filled_series(self, data_ref: str) -> dict[str, Any]:
        """Load a gap-filled series by its reference key.

        Args:
            data_ref: The reference key returned by ``put_gap_filled_series``.

        Returns:
            A dict with ``run_id``, ``symbol``, ``source``, ``method``,
            ``original_dates``, ``original_prices``, ``filled_dates``,
            ``filled_prices``.

        Raises:
            KeyError: If no gap-filled series is found.
        """
        try:
            run_id, symbol, source = _parse_filled_data_ref(data_ref)
        except KeyError:
            logger.error("Gap-filled series not found: data_ref=%s", data_ref)
            raise KeyError(f"No gap_filled_series found for data_ref: {data_ref}")
        conn = self.connection

        run_num = _find_run(conn, run_id)
        inst_num = _find_instrument(conn, symbol)
        src_num = _find_source(conn, source)
        if run_num is None or inst_num is None or src_num is None:
            logger.error("Gap-filled series not found: data_ref=%s", data_ref)
            raise KeyError(f"No gap_filled_series found for data_ref: {data_ref}")

        rows = conn.execute(
            """
            SELECT gfm.name AS method, ft.date, ft.price
            FROM filled_timeseries ft
            JOIN gap_filling_method gfm ON gfm.id = ft.method_id
            WHERE ft.run_id = ? AND ft.instrument_id = ? AND ft.source_id = ?
            ORDER BY ft.date
            """,
            (run_num, inst_num, src_num),
        ).fetchall()

        if not rows:
            logger.error("Gap-filled series not found: data_ref=%s", data_ref)
            raise KeyError(f"No gap_filled_series found for data_ref: {data_ref}")

        method = rows[0][0]
        filled_dates = [row[1] for row in rows]
        filled_prices = [row[2] for row in rows]

        # The original dates/prices are stored in the raw series for the
        # same run/symbol/source; reconstruct them here for API parity.
        original_dates: list[str] = []
        original_prices: list[Any] = []
        raw_rows = conn.execute(
            """
            SELECT date, price
            FROM raw_timeseries
            WHERE run_id = ? AND instrument_id = ? AND source_id = ?
            ORDER BY date
            """,
            (run_num, inst_num, src_num),
        ).fetchall()
        if raw_rows:
            original_dates = [row[0] for row in raw_rows]
            original_prices = [row[1] for row in raw_rows]

        logger.debug("gap_filled_series_retrieved data_ref=%s method=%s", data_ref, method)
        return {
            "run_id": run_id,
            "symbol": symbol,
            "source": source,
            "method": method,
            "original_dates": original_dates,
            "original_prices": original_prices,
            "filled_dates": filled_dates,
            "filled_prices": filled_prices,
            "original_data_ref": f"{run_id}:{symbol}:{source}",
        }

    def list_gap_filled_series(
        self,
        run_id: str,
        symbol: str | None = None,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        """List stored gap-filled series, optionally filtered.

        Args:
            run_id: The run/session identifier.
            symbol: If provided, filter by symbol.
            source: If provided, filter by source.

        Returns:
            List of dicts with series metadata.
        """
        query = """
            SELECT r.run_id, i.symbol, s.name,
                   gfm.name AS method, ft.date, ft.price
            FROM filled_timeseries ft
            JOIN runs r        ON r.id = ft.run_id
            JOIN instruments i ON i.id = ft.instrument_id
            JOIN sources s     ON s.id = ft.source_id
            JOIN gap_filling_method gfm ON gfm.id = ft.method_id
            WHERE r.run_id = ?
        """
        params: list[Any] = [run_id]
        if symbol:
            query += " AND i.symbol = ?"
            params.append(symbol)
        if source:
            query += " AND s.name = ?"
            params.append(source)
        query += " ORDER BY i.symbol, s.name, ft.date"

        conn = self.connection
        rows = conn.execute(query, params).fetchall()

        series_map: dict[tuple[str, str], dict[str, Any]] = {}
        for row_run_id, row_symbol, row_source, method, date_str, price in rows:
            key = (row_symbol, row_source)
            if key not in series_map:
                series_map[key] = {
                    "data_ref": f"{row_run_id}:{row_symbol}:{row_source}:filled",
                    "symbol": row_symbol,
                    "source": row_source,
                    "method": method,
                    "filled_dates": [],
                    "filled_prices": [],
                }
            series_map[key]["filled_dates"].append(date_str)
            series_map[key]["filled_prices"].append(price)

        return list(series_map.values())

    # ── Quality Reports ──────────────────────────────────────────────────────

    def put_quality_report(
        self,
        run_id: str,
        report: dict[str, Any],
        symbol: str | None = None,
        source: str | None = None,
    ) -> str:
        """Store a quality report and return a reference key.

        Args:
            run_id: The run/session identifier.
            report: The quality report dict.
            symbol: Optional symbol associated with the report.
            source: Optional source associated with the report.

        Returns:
            A ``report_id`` string of the form ``<run_id>:<timestamp>:<symbol>:<source>``.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        report_id = f"{run_id}:{timestamp}:{symbol or 'unknown'}:{source or 'unknown'}"
        conn = self.connection
        run_num = _ensure_run(conn, run_id)
        inst_num = _ensure_instrument(conn, symbol) if symbol else None
        src_num = _ensure_source(conn, source) if source else None
        conn.execute(
            """
            INSERT INTO quality_reports
                (report_id, run_id, instrument_id, source_id, report)
            VALUES (?, ?, ?, ?, ?)
            """,
            (report_id, run_num, inst_num, src_num, json.dumps(report, default=str)),
        )
        conn.commit()
        logger.info(
            "Quality report stored: report_id=%s symbol=%s source=%s",
            report_id, symbol or "unknown", source or "unknown",
        )
        logger.debug("quality_report_stored report_id=%s", report_id)
        return report_id

    def get_quality_report(self, report_id: str) -> dict[str, Any]:
        """Load a quality report by its reference key.

        Args:
            report_id: The reference key returned by ``put_quality_report``.

        Returns:
            The quality report dict.

        Raises:
            KeyError: If no report is found.
        """
        conn = self.connection
        row = conn.execute(
            "SELECT report FROM quality_reports WHERE report_id = ?",
            (report_id,),
        ).fetchone()

        if row is None:
            logger.error("Quality report not found: report_id=%s", report_id)
            raise KeyError(f"No quality report found for report_id: {report_id}")

        logger.debug("quality_report_retrieved report_id=%s", report_id)
        return json.loads(row[0])

    def list_quality_reports(
        self,
        run_id: str,
        symbol: str | None = None,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        """List quality reports for a run, optionally filtered.

        Args:
            run_id: The run/session identifier.
            symbol: If provided, filter by symbol.
            source: If provided, filter by source.

        Returns:
            List of report metadata dicts.
        """
        query = """
            SELECT qr.report_id, i.symbol, s.name
            FROM quality_reports qr
            JOIN runs r ON r.id = qr.run_id
            LEFT JOIN instruments i ON i.id = qr.instrument_id
            LEFT JOIN sources s ON s.id = qr.source_id
            WHERE r.run_id = ?
        """
        params: list[Any] = [run_id]
        if symbol:
            query += " AND i.symbol = ?"
            params.append(symbol)
        if source:
            query += " AND s.name = ?"
            params.append(source)

        conn = self.connection
        rows = conn.execute(query, params).fetchall()
        return [
            {
                "report_id": row[0],
                "symbol": row[1],
                "source": row[2],
            }
            for row in rows
        ]

    # ── Artifacts ───────────────────────────────────────────────────────────

    def put_artifact(
        self,
        run_id: str,
        artifact_type: str,
        path: str,
        symbol: str | None = None,
        source: str | None = None,
    ) -> str:
        """Record an artifact path in the data store.

        Args:
            run_id: The run/session identifier.
            artifact_type: One of 'csv', 'png', 'report'.
            path: File path to the artifact.
            symbol: Optional symbol.
            source: Optional source.

        Returns:
            An ``artifact_id`` string.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        artifact_id = f"{run_id}:{artifact_type}:{timestamp}"
        conn = self.connection
        run_num = _ensure_run(conn, run_id)
        inst_num = _ensure_instrument(conn, symbol) if symbol else None
        src_num = _ensure_source(conn, source) if source else None
        conn.execute(
            """
            INSERT INTO artifacts
                (artifact_id, run_id, instrument_id, source_id, artifact_type, path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (artifact_id, run_num, inst_num, src_num, artifact_type, path),
        )
        conn.commit()
        logger.info(
            "Artifact stored: artifact_id=%s type=%s path=%s",
            artifact_id, artifact_type, path,
        )
        return artifact_id

    def list_artifacts(
        self,
        run_id: str,
        artifact_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """List stored artifacts for a run.

        Args:
            run_id: The run/session identifier.
            artifact_type: If provided, filter by type.

        Returns:
            List of dicts with ``artifact_id``, ``artifact_type``, ``path``,
            ``symbol``, ``source``.
        """
        query = """
            SELECT a.artifact_id, a.artifact_type, a.path,
                   i.symbol, s.name
            FROM artifacts a
            JOIN runs r ON r.id = a.run_id
            LEFT JOIN instruments i ON i.id = a.instrument_id
            LEFT JOIN sources s ON s.id = a.source_id
            WHERE r.run_id = ?
        """
        params: list[Any] = [run_id]
        if artifact_type:
            query += " AND a.artifact_type = ?"
            params.append(artifact_type)

        conn = self.connection
        rows = conn.execute(query, params).fetchall()

        logger.info(
            "Listed artifacts: run_id=%s type=%s count=%d",
            run_id, artifact_type or "*", len(rows),
        )
        return [
            {
                "artifact_id": row[0],
                "artifact_type": row[1],
                "path": row[2],
                "symbol": row[3],
                "source": row[4],
            }
            for row in rows
        ]

    # ── Files (user-uploaded) ───────────────────────────────────────────

    def put_file(
        self,
        file_id: str,
        filename: str,
        description: str | None,
        path: str,
    ) -> str:
        """Register an uploaded file in the datastore.

        Args:
            file_id: Unique identifier for the file (caller-generated).
            filename: Original filename.
            description: Optional human readable description.
            path: Filesystem path where the file was saved.

        Returns:
            The stored `file_id`.
        """
        conn = self.connection
        conn.execute(
            """
            INSERT OR REPLACE INTO files
                (file_id, filename, description, path)
            VALUES (?, ?, ?, ?)
            """,
            (file_id, filename, description, path),
        )
        conn.commit()
        logger.info("File registered: file_id=%s filename=%s path=%s", file_id, filename, path)
        return file_id

    def list_files(self) -> list[dict[str, Any]]:
        """Return all registered user-uploaded files.

        Returns a list of dicts with `file_id`, `filename`, `description`,
        `path`, and `created_at`.
        """
        conn = self.connection
        rows = conn.execute(
            "SELECT file_id, filename, description, path, created_at FROM files ORDER BY created_at DESC"
        ).fetchall()
        return [
            {
                "file_id": row[0],
                "filename": row[1],
                "description": row[2],
                "path": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]

    def get_file(self, file_id: str) -> dict[str, Any]:
        """Return metadata for a single registered file.

        Raises KeyError if not found.
        """
        conn = self.connection
        row = conn.execute(
            "SELECT file_id, filename, description, path, created_at FROM files WHERE file_id = ?",
            (file_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"No file found for file_id: {file_id}")
        return {
            "file_id": row[0],
            "filename": row[1],
            "description": row[2],
            "path": row[3],
            "created_at": row[4],
        }

    def delete_file(self, file_id: str) -> bool:
        """Delete a registered file entry.

        Note: this only removes the DB record. Deleting the underlying
        filesystem object is the caller's responsibility.
        """
        conn = self.connection
        cursor = conn.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
        conn.commit()
        return cursor.rowcount > 0

    # ── Cross-run queries ───────────────────────────────────────────────────

    def list_runs_with_stats(
        self,
    ) -> list[dict[str, Any]]:
        """List all runs with summary statistics.

        Returns:
            List of dicts with ``run_id``, ``start_date``, ``end_date``,
            ``timeseries_count``, ``filled_count``, ``artifact_count``.
        """
        conn = self.connection
        rows = conn.execute(
            """
            SELECT
                r.run_id,
                r.start_date,
                r.end_date,
                r.created_at,
                (SELECT COUNT(*) FROM raw_timeseries rt
                 WHERE rt.run_id = r.id) AS ts_count,
                (SELECT COUNT(*) FROM filled_timeseries ft
                 WHERE ft.run_id = r.id) AS filled_count,
                (SELECT COUNT(*) FROM artifacts a
                 WHERE a.run_id = r.id) AS art_count
            FROM runs r
            ORDER BY r.created_at DESC
            """
        ).fetchall()
        return [
            {
                "run_id": row[0],
                "start_date": row[1],
                "end_date": row[2],
                "created_at": row[3],
                "timeseries_count": row[4],
                "filled_count": row[5],
                "artifact_count": row[6],
            }
            for row in rows
        ]

    # ── Utility ──────────────────────────────────────────────────────────────

    def vacuum(self) -> None:
        """Recover disk space and defragment the database."""
        self.connection.execute("VACUUM")
        logger.info("Database vacuumed: path=%s", self.db_path)

    def __repr__(self) -> str:
        return f"DataStore(db_path={self.db_path!r})"


# ── Reference helpers ────────────────────────────────────────────────────


def _parse_data_ref(data_ref: str) -> tuple[str, str, str]:
    """Split a raw-series reference of the form ``run_id:symbol:source``.

    The source is always the final ``:``-separated token and the run id is
    always the first token; the symbol may itself contain ``:`` characters.

    Raises:
        KeyError: If the reference has fewer than three parts.
    """
    parts = data_ref.split(":")
    if len(parts) < 3:
        raise KeyError(f"Invalid data_ref format: {data_ref}")
    return parts[0], ":".join(parts[1:-1]), parts[-1]


def _parse_filled_data_ref(data_ref: str) -> tuple[str, str, str]:
    """Split a filled-series reference of the form ``run_id:symbol:source:filled``.

    Raises:
        KeyError: If the reference does not end with ``:filled``.
    """
    suffix = ":filled"
    if not data_ref.endswith(suffix):
        raise KeyError(f"Invalid filled data_ref format: {data_ref}")
    return _parse_data_ref(data_ref[: -len(suffix)])
