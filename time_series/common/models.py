from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd


@dataclass
class PipelineRequest:
    ticker: str
    start_date: date
    end_date: date


@dataclass
class DataSourceResult:
    source: str
    data: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityMetric:
    source: str
    rows: int
    missing_close: int
    duplicate_dates: int
    coverage_pct: float
    start_date: str
    end_date: str


@dataclass
class GapFillSuggestion:
    method: str
    rationale: str


@dataclass
class ReActEvent:
    agent: str
    thought: str
    action: str
    observation: str
