from __future__ import annotations

from typing import Iterable

import pandas as pd
from rich.console import Console
from rich.table import Table

from .models import GapFillSuggestion, QualityMetric, ReActEvent


class CLIReporter:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def print_quality_table(self, quality: list[QualityMetric]) -> None:
        table = Table(title="Data Quality Comparison by Source")
        table.add_column("Source")
        table.add_column("Rows", justify="right")
        table.add_column("Missing Close", justify="right")
        table.add_column("Duplicate Dates", justify="right")
        table.add_column("Coverage %", justify="right")
        table.add_column("Start")
        table.add_column("End")

        for q in quality:
            table.add_row(
                q.source,
                str(q.rows),
                str(q.missing_close),
                str(q.duplicate_dates),
                f"{q.coverage_pct:.2f}",
                q.start_date,
                q.end_date,
            )
        self.console.print(table)

    def print_gap_suggestions(self, suggestions: list[GapFillSuggestion]) -> None:
        table = Table(title="Gap Filling Recommendations")
        table.add_column("Method")
        table.add_column("Rationale")
        for s in suggestions:
            table.add_row(s.method, s.rationale)
        self.console.print(table)

    def print_generated_series_summary(self, df: pd.DataFrame, source: str, method: str) -> None:
        table = Table(title="Continuous Time-Series Summary")
        table.add_column("Metric")
        table.add_column("Value")

        table.add_row("Selected Source", source)
        table.add_row("Gap Fill Method", method)
        table.add_row("Rows", str(len(df)))
        table.add_row("Start", str(pd.to_datetime(df["date"]).min().date()))
        table.add_row("End", str(pd.to_datetime(df["date"]).max().date()))
        table.add_row("Missing Close (post-fill)", str(int(df["close"].isna().sum())))
        table.add_row("Min Close", f"{df['close'].min():.4f}")
        table.add_row("Max Close", f"{df['close'].max():.4f}")
        self.console.print(table)

    def print_react_trace(self, events: Iterable[ReActEvent], title: str) -> None:
        table = Table(title=title)
        table.add_column("Agent")
        table.add_column("Thought")
        table.add_column("Action")
        table.add_column("Observation")
        for e in events:
            table.add_row(e.agent, e.thought, e.action, e.observation)
        self.console.print(table)

    def print_artifacts(self, run_id: str, artifact_paths: dict[str, str]) -> None:
        table = Table(title="Audit Artifacts")
        table.add_column("Run ID")
        table.add_column("Artifact")
        table.add_column("Path")
        for name, path in artifact_paths.items():
            table.add_row(run_id, name, path)
        self.console.print(table)
