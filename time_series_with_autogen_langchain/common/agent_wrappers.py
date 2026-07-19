from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..common.agent_specs import AGENT_REACT_SPECS
from ..common.models import DataSourceResult, PipelineRequest, QualityMetric
from ..common.tools.pipeline_tools import (
    DataQualityTool,
    GapAnalysisTool,
    MarketDataTool,
    TimeSeriesGenerationTool,
)


@dataclass
class BaseAgentWrapper:
    name: str
    react_definition: str


@dataclass
class MarketDataAgentWrapper(BaseAgentWrapper):
    tool: MarketDataTool

    def run(self, request: PipelineRequest) -> list[DataSourceResult]:
        return self.tool.run(request)


@dataclass
class DataQualityAgentWrapper(BaseAgentWrapper):
    tool: DataQualityTool

    def run(self, source_results: list[DataSourceResult]) -> list[QualityMetric]:
        return self.tool.run(source_results)


@dataclass
class GapAnalysisAgentWrapper(BaseAgentWrapper):
    tool: GapAnalysisTool

    def run(self, series: pd.Series):
        return self.tool.run(series)


@dataclass
class TimeSeriesGenerationAgentWrapper(BaseAgentWrapper):
    tool: TimeSeriesGenerationTool

    def run(self, source_df: pd.DataFrame, method: str) -> pd.DataFrame:
        return self.tool.run(source_df, method)


def build_wrapper_specs() -> dict[str, str]:
    return AGENT_REACT_SPECS.copy()
