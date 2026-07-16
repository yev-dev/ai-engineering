from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from .models import DataSourceResult, PipelineRequest
from .stubs import MarketDataAgentStub


class MarketDataConnector(Protocol):
    source_name: str

    def fetch(self, request: PipelineRequest) -> DataSourceResult:
        ...


@dataclass
class YahooLiveConnector:
    """Live Yahoo connector with graceful fallback to synthetic stub data."""

    source_name: str = "yahoo"
    fallback_seed: int = 42

    def fetch(self, request: PipelineRequest) -> DataSourceResult:
        try:
            import yfinance as yf

            df = yf.download(
                request.ticker,
                start=request.start_date,
                end=request.end_date,
                auto_adjust=False,
                progress=False,
            )
            if df.empty:
                raise ValueError("Yahoo returned empty dataset")

            close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            out = (
                df[[close_col]]
                .rename(columns={close_col: "close"})
                .reset_index()
                .rename(columns={"Date": "date"})
            )
            out["ticker"] = request.ticker.upper()
            out["source"] = self.source_name

            return DataSourceResult(
                source=self.source_name,
                data=out[["date", "ticker", "close", "source"]],
                metadata={"mode": "live", "provider": "yfinance"},
            )
        except Exception as exc:
            stub = MarketDataAgentStub(seed=self.fallback_seed)
            fallback = next(x for x in stub.fetch(request) if x.source == self.source_name)
            fallback.metadata.update(
                {
                    "mode": "stub_fallback",
                    "provider": "yfinance",
                    "reason": str(exc),
                }
            )
            return fallback


@dataclass
class YahooStubConnector:
    """Explicit stub Yahoo connector when live retrieval is disabled."""

    source_name: str = "yahoo"
    seed: int = 42

    def fetch(self, request: PipelineRequest) -> DataSourceResult:
        stub = MarketDataAgentStub(seed=self.seed)
        result = next(x for x in stub.fetch(request) if x.source == self.source_name)
        result.metadata.update(
            {
                "mode": "stub",
                "provider": "yahoo_stub",
                "note": "Live Yahoo disabled by configuration.",
            }
        )
        return result


@dataclass
class BloombergConnectorPlaceholder:
    """Placeholder Bloomberg connector interface."""

    source_name: str = "bloomberg"
    seed: int = 42

    def fetch(self, request: PipelineRequest) -> DataSourceResult:
        stub = MarketDataAgentStub(seed=self.seed)
        result = next(x for x in stub.fetch(request) if x.source == self.source_name)
        result.metadata.update(
            {
                "mode": "placeholder",
                "provider": "bloomberg",
                "note": "Replace with real Bloomberg API integration.",
            }
        )
        return result


@dataclass
class ReutersConnectorPlaceholder:
    """Placeholder Reuters connector interface."""

    source_name: str = "reuters"
    seed: int = 42

    def fetch(self, request: PipelineRequest) -> DataSourceResult:
        stub = MarketDataAgentStub(seed=self.seed)
        result = next(x for x in stub.fetch(request) if x.source == self.source_name)
        result.metadata.update(
            {
                "mode": "placeholder",
                "provider": "reuters",
                "note": "Replace with real Reuters API integration.",
            }
        )
        return result


@dataclass
class MultiSourceMarketDataAgent:
    connectors: list[MarketDataConnector]

    def fetch(self, request: PipelineRequest) -> list[DataSourceResult]:
        results = [connector.fetch(request) for connector in self.connectors]
        # Ensure stable source order in output tables.
        ordering = {"bloomberg": 0, "reuters": 1, "yahoo": 2}
        return sorted(results, key=lambda x: ordering.get(x.source, 99))


def build_default_market_data_agent(use_live_yahoo: bool, seed: int) -> MultiSourceMarketDataAgent:
    yahoo_connector: MarketDataConnector
    if use_live_yahoo:
        yahoo_connector = YahooLiveConnector(fallback_seed=seed)
    else:
        yahoo_connector = YahooStubConnector(seed=seed)

    connectors: list[MarketDataConnector] = [
        BloombergConnectorPlaceholder(seed=seed),
        ReutersConnectorPlaceholder(seed=seed),
        yahoo_connector,
    ]
    return MultiSourceMarketDataAgent(connectors=connectors)


def source_results_to_table(source_results: list[DataSourceResult]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for result in source_results:
        rows.append(
            {
                "source": result.source,
                "mode": str(result.metadata.get("mode", "unknown")),
                "provider": str(result.metadata.get("provider", result.source)),
                "rows": str(len(result.data)),
            }
        )
    return pd.DataFrame(rows)
