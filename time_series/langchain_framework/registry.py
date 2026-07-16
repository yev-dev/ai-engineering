from __future__ import annotations

from dataclasses import dataclass

from common.agent_specs import AGENT_REACT_SPECS
from common.agent_wrappers import (
    DataQualityAgentWrapper,
    GapAnalysisAgentWrapper,
    MarketDataAgentWrapper,
    TimeSeriesGenerationAgentWrapper,
)
from common.tools.pipeline_tools import DataQualityTool, GapAnalysisTool, MarketDataTool, TimeSeriesGenerationTool


@dataclass
class LangChainAgentRegistry:
    market_data_node: MarketDataAgentWrapper
    data_quality_node: DataQualityAgentWrapper
    gap_analysis_node: GapAnalysisAgentWrapper
    timeseries_generation_node: TimeSeriesGenerationAgentWrapper


def build_langchain_registry(
    market_data_tool: MarketDataTool,
    data_quality_tool: DataQualityTool,
    gap_analysis_tool: GapAnalysisTool,
    generation_tool: TimeSeriesGenerationTool,
) -> LangChainAgentRegistry:
    return LangChainAgentRegistry(
        market_data_node=MarketDataAgentWrapper(
            name="MarketDataNode",
            react_definition=AGENT_REACT_SPECS["market_data_agent"],
            tool=market_data_tool,
        ),
        data_quality_node=DataQualityAgentWrapper(
            name="DataQualityNode",
            react_definition=AGENT_REACT_SPECS["data_quality_agent"],
            tool=data_quality_tool,
        ),
        gap_analysis_node=GapAnalysisAgentWrapper(
            name="GapAnalysisNode",
            react_definition=AGENT_REACT_SPECS["gap_analysis_agent"],
            tool=gap_analysis_tool,
        ),
        timeseries_generation_node=TimeSeriesGenerationAgentWrapper(
            name="TimeSeriesGenerationNode",
            react_definition=AGENT_REACT_SPECS["timeseries_generation_agent"],
            tool=generation_tool,
        ),
    )
