from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import DataSourceResult, PipelineRequest


@dataclass
class MarketDataAgentStub:
    """Stubbed multi-source market data retriever for historical prices."""

    seed: int = 42

    def fetch(self, request: PipelineRequest) -> list[DataSourceResult]:
        sources = ["bloomberg", "reuters", "yahoo"]
        return [self._build_source_data(request, source) for source in sources]

    def _build_source_data(self, request: PipelineRequest, source: str) -> DataSourceResult:
        rng = np.random.default_rng(abs(hash((self.seed, source, request.ticker))) % (2**32))
        dates = pd.date_range(request.start_date, request.end_date, freq="B")

        base = np.linspace(100.0, 130.0, len(dates))
        seasonality = 2.0 * np.sin(np.linspace(0.0, 5.0, len(dates)))
        noise_scale = {"bloomberg": 0.4, "reuters": 0.6, "yahoo": 0.9}[source]
        close = base + seasonality + rng.normal(0.0, noise_scale, len(dates))

        df = pd.DataFrame(
            {
                "date": dates,
                "ticker": request.ticker.upper(),
                "close": close,
                "source": source,
            }
        )

        missing_ratio = {"bloomberg": 0.01, "reuters": 0.03, "yahoo": 0.08}[source]
        missing_idx = rng.choice(df.index, int(len(df) * missing_ratio), replace=False)
        df.loc[missing_idx, "close"] = np.nan

        duplicate_ratio = {"bloomberg": 0.0, "reuters": 0.01, "yahoo": 0.02}[source]
        dup_idx = rng.choice(df.index, int(len(df) * duplicate_ratio), replace=False)
        duplicates = df.loc[dup_idx]
        df = pd.concat([df, duplicates], ignore_index=True).sort_values("date").reset_index(drop=True)

        return DataSourceResult(source=source, data=df)
