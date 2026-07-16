from __future__ import annotations

import pandas as pd

from .models import DataSourceResult, QualityMetric


class DataQualityAgent:
    """Compares data quality across available market data sources."""

    def evaluate(self, source_results: list[DataSourceResult]) -> list[QualityMetric]:
        quality: list[QualityMetric] = []
        for result in source_results:
            df = result.data.copy()
            rows = len(df)
            missing_close = int(df["close"].isna().sum())
            duplicate_dates = int(df.duplicated(subset=["date"]).sum())
            coverage_pct = max(0.0, 100.0 * (1.0 - (missing_close / rows))) if rows else 0.0

            quality.append(
                QualityMetric(
                    source=result.source,
                    rows=rows,
                    missing_close=missing_close,
                    duplicate_dates=duplicate_dates,
                    coverage_pct=coverage_pct,
                    start_date=str(pd.to_datetime(df["date"]).min().date()) if rows else "n/a",
                    end_date=str(pd.to_datetime(df["date"]).max().date()) if rows else "n/a",
                )
            )
        return sorted(quality, key=lambda x: (x.missing_close, x.duplicate_dates, -x.coverage_pct))
