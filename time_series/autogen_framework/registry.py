from __future__ import annotations

from dataclasses import dataclass

from ..common.agent_specs import AGENT_REACT_SPECS
from ..common.agent_wrappers import (
    DataQualityAgentWrapper,
    GapAnalysisAgentWrapper,
    MarketDataAgentWrapper,
    TimeSeriesGenerationAgentWrapper,
)
from ..common.tools.pipeline_tools import DataQualityTool, GapAnalysisTool, MarketDataTool, TimeSeriesGenerationTool


@dataclass
class AutogenAgentRegistry:
    market_data_agent: MarketDataAgentWrapper
    data_quality_agent: DataQualityAgentWrapper
    gap_analysis_agent: GapAnalysisAgentWrapper
    timeseries_generation_agent: TimeSeriesGenerationAgentWrapper


def build_autogen_registry(
    market_data_tool: MarketDataTool,
    data_quality_tool: DataQualityTool,
    gap_analysis_tool: GapAnalysisTool,
    generation_tool: TimeSeriesGenerationTool,
) -> AutogenAgentRegistry:
    return AutogenAgentRegistry(
        market_data_agent=MarketDataAgentWrapper(
            name="MarketDataAgent",
            react_definition=AGENT_REACT_SPECS["market_data_agent"],
            tool=market_data_tool,
        ),
        data_quality_agent=DataQualityAgentWrapper(
            name="DataQualityAgent",
            react_definition=AGENT_REACT_SPECS["data_quality_agent"],
            tool=data_quality_tool,
        ),
        gap_analysis_agent=GapAnalysisAgentWrapper(
            name="GapAnalysisAgent",
            react_definition=AGENT_REACT_SPECS["gap_analysis_agent"],
            tool=gap_analysis_tool,
        ),
        timeseries_generation_agent=TimeSeriesGenerationAgentWrapper(
            name="TimeSeriesGenerationAgent",
            react_definition=AGENT_REACT_SPECS["timeseries_generation_agent"],
            tool=generation_tool,
        ),
    )
