"""Registry of user-selected data files injected into the workflow.

The ``DataSourceRegistry`` holds the data files the user selected in the
dashboard's "Data Sources" section.  It is populated from the injected file
metadata (name, description, path) and made available to the workflow so the
agents know which registered files are available for time-series construction.

The registry is deliberately framework-agnostic: it is a plain data container
that the processor injects into agent system prompts and can be extended with
lookup helpers without coupling to Streamlit or the LLM layer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .database import DataStore, get_datastore

logger = logging.getLogger(__name__)


def _infer_source(
    filename: str,
    allowed_sources: list[str] | tuple[str, ...] | None = None,
) -> str | None:
    """Infer the data-source name (e.g. ``yahoo``) for a file of a run.

    Inference is driven by the sources *assigned to the active run*
    (``allowed_sources``) — populated dynamically per run from the DataStore's
    ``run_sources`` reference — rather than hard-coded values in ``config.yaml``.
    The first source name that appears (case-insensitively) in ``filename`` is
    returned.

    Args:
        filename: The uploaded file's name.
        allowed_sources: The sources assigned to the active run.  When ``None``
            or empty, no inference is attempted and ``None`` is returned; a
            ``source`` must then be assigned explicitly (e.g. by tests or the
            user).

    Returns:
        The matching source name (lowercased), or ``None``.
    """
    if not filename or not allowed_sources:
        return None
    lowered = filename.casefold()
    for source in allowed_sources:
        name = str(source).casefold()
        if name and name in lowered:
            return name
    return None


@dataclass(frozen=True)
class DataSourceFile:
    """Metadata for a single user-selected data file.

    Backed by the DataStore ``files`` table.  ``source`` records which data
    source the file corresponds to (e.g. ``yahoo``) and is inferred from the
    filename so tools and the dashboard can reason about active sources.
    """

    file_id: str
    filename: str
    description: str | None = None
    path: str | None = None
    source: str | None = None
    assigned_sources: tuple[str, ...] = field(
        default=(), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # Infer the source from the run's assigned sources when not explicit.
        if not self.source and self.assigned_sources:
            object.__setattr__(
                self,
                "source",
                _infer_source(self.filename, self.assigned_sources),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for this file."""
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "description": self.description,
            "path": self.path,
            "source": self.source,
        }


class DataSourceRegistry:
    """Registry of data files available to the workflow.

    Populated from the files the user selected in the dashboard.  The registry
    is injected into the processor so agents can reference the available files
    during time-series construction.

    The registry is backed by the DataStore's ``files`` table: when a new file
    is uploaded its metadata is persisted to the database, and the registry can
    be refreshed from the database so it always reflects the current set of
    registered files.
    """

    def __init__(
        self,
        files: list[dict[str, Any]] | None = None,
        run_id: str | None = None,
    ) -> None:
        self._files: dict[str, DataSourceFile] = {}
        self.run_id: str | None = run_id
        self._assigned_sources_cache: list[str] | None = None
        if files:
            self.add_files(files)

    # ── Database-backed helpers ────────────────────────────────────────────

    @staticmethod
    def _get_store() -> DataStore | None:
        """Return the global DataStore singleton, or None if not initialised."""
        try:
            return get_datastore()
        except RuntimeError:
            return None

    def assigned_sources(self) -> list[str]:
        """Return the sources dynamically assigned to this registry's active run.

        Sources are resolved from the DataStore ``run_sources`` reference for
        ``self.run_id`` (populated whenever series are stored for the run).  When
        no run is bound or no sources have been recorded yet, an empty list is
        returned and files keep an explicit/``None`` source.

        Returns:
            Sorted list of source names assigned to the active run.
        """
        if self._assigned_sources_cache is None:
            self._assigned_sources_cache = self._resolve_assigned_sources()
        return self._assigned_sources_cache

    def _resolve_assigned_sources(self) -> list[str]:
        if not self.run_id:
            return []
        store = self._get_store()
        if store is None:
            return []
        try:
            return store.get_run_sources(self.run_id)
        except Exception as error:  # noqa: BLE001 - no store is a normal state
            logger.warning(
                "assigned_sources_resolve_failed run_id=%s error=%s",
                self.run_id,
                error,
            )
            return []


    def refresh_from_database(self) -> None:
        """Reload the registry from the DataStore ``files`` table.

        Any files persisted in the database (e.g. uploaded via the dashboard)
        are loaded into the registry so the workflow always sees the current
        set of registered files.
        """
        store = self._get_store()
        if store is None:
            return
        try:
            db_files = store.list_files()
        except Exception as error:
            logger.warning("data_source_registry_refresh_failed error=%s", error)
            return
        self._files.clear()
        self.add_files(db_files)
        logger.info(
            "data_source_registry_refreshed count=%d",
            len(self._files),
        )

    def persist_to_database(self) -> None:
        """Persist the registry's files to the DataStore ``files`` table.

        Ensures every file in the registry has a corresponding row in the
        database so the metadata survives across sessions.
        """
        store = self._get_store()
        if store is None:
            return
        for file in self._files.values():
            try:
                store.put_file(
                    file_id=file.file_id,
                    filename=file.filename,
                    description=file.description,
                    path=file.path or "",
                )
            except Exception as error:
                logger.warning(
                    "data_source_registry_persist_failed file_id=%s error=%s",
                    file.file_id,
                    error,
                )
        logger.info(
            "data_source_registry_persisted count=%d",
            len(self._files),
        )

    def add_file(
        self,
        file_id: str,
        filename: str,
        description: str | None = None,
        path: str | None = None,
        source: str | None = None,
    ) -> None:
        """Register a single data file.

        Args:
            file_id: Unique identifier for the file.
            filename: Original file name.
            description: Optional human-readable description.
            path: Filesystem path where the file was saved.
            source: Optional explicit data-source name.  When omitted, the
                source is inferred from the filename against the sources
                assigned to the registry's active run (``self.run_id``).
        """
        self._files[file_id] = DataSourceFile(
            file_id=file_id,
            filename=filename,
            description=description,
            path=path,
            source=source,
            assigned_sources=tuple(self.assigned_sources()),
        )
        logger.debug(
            "data_source_registry_added file_id=%s filename=%s",
            file_id,
            filename,
        )

    def add_files(self, files: list[dict[str, Any]]) -> None:
        """Register multiple data files from dict metadata."""
        for item in files:
            file_id = str(item.get("file_id") or "").strip()
            filename = str(item.get("filename") or "").strip()
            if not file_id or not filename:
                continue
            self.add_file(
                file_id=file_id,
                filename=filename,
                description=item.get("description"),
                path=item.get("path"),
                source=item.get("source"),
            )

    def clear(self) -> None:
        """Remove all registered files."""
        self._files.clear()

    @property
    def is_empty(self) -> bool:
        """Return True when no files are registered."""
        return not self._files

    def __len__(self) -> int:
        return len(self._files)

    def list_files(self) -> list[DataSourceFile]:
        """Return the registered files in insertion order."""
        return list(self._files.values())

    def get_file(self, file_id: str) -> DataSourceFile | None:
        """Return a registered file by its id, or None."""
        return self._files.get(file_id)

    def filenames(self) -> list[str]:
        """Return the list of registered file names."""
        return [file.filename for file in self._files.values()]

    def to_dicts(self) -> list[dict[str, Any]]:
        """Return the registered files as a list of JSON-serialisable dicts."""
        return [file.to_dict() for file in self._files.values()]

    def describe(self) -> str:
        """Return a human-readable summary of the available files.

        This is injected into agent system prompts so the LLM knows which
        registered files are available for the workflow.
        """
        if self.is_empty:
            return ""
        lines = ["[AVAILABLE DATA FILES]"]
        for file in self._files.values():
            description = file.description or "no description"
            lines.append(f"- {file.filename}: {description}")
        return "\n".join(lines)

    # ── Active-source helpers ─────────────────────────────────────────────

    def sources(self) -> list[str]:
        """Return the distinct data-source names across registered files.

        Sources are derived from each :class:`DataSourceFile.source`, which is
        inferred from the file name against the sources assigned to the
        registry's active run (``run_sources``).  This is the registry's view of
        the "active data sources" available to a run.

        Returns:
            Sorted list of unique source names (e.g. ``["bloomberg", "yahoo"]``).
        """
        return sorted(
            {
                file.source
                for file in self._files.values()
                if file.source
            }
        )

    def active_sources(self) -> list[str]:
        """Alias for :meth:`sources` — the active data sources of the registry."""
        return self.sources()

    def filter_by_sources(self, sources: list[str]) -> "DataSourceRegistry":
        """Return a new registry containing only files matching ``sources``.

        Files whose inferred source is ``None`` are dropped.  This lets the
        workflow restrict which data files are visible to a run based on its
        active data sources.

        Args:
            sources: The set of allowed source names (lowercased).

        Returns:
            A new :class:`DataSourceRegistry` with the matching files.
        """
        allowed = {str(source).casefold() for source in sources}
        matched = [
            file.to_dict()
            for file in self._files.values()
            if file.source and file.source.casefold() in allowed
        ]
        return DataSourceRegistry(matched)