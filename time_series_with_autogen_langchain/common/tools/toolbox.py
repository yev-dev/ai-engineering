from __future__ import annotations

from typing import Any

import pandas as pd

from ..models import PipelineRequest
from .decorators import autogen_tool, get_decorated_tools, langgraph_tool
from .pipeline_tools import (
    AuditExportTool,
    DataQualityTool,
    GapAnalysisTool,
    MarketDataTool,
    TimeSeriesGenerationTool,
)


class CommonToolbox:
    """Framework-neutral decorator-based toolbox for LLM orchestration."""

    def __init__(
        self,
        market_data_tool: MarketDataTool,
        data_quality_tool: DataQualityTool,
        gap_analysis_tool: GapAnalysisTool,
        generation_tool: TimeSeriesGenerationTool,
        audit_tool: AuditExportTool,
    ) -> None:
        self.market_data_tool = market_data_tool
        self.data_quality_tool = data_quality_tool
        self.gap_analysis_tool = gap_analysis_tool
        self.generation_tool = generation_tool
        self.audit_tool = audit_tool

    @autogen_tool(
        name="fetch_market_data",
        description="Fetch multi-source market data (Bloomberg/Reuters/Yahoo) for request context.",
    )
    @langgraph_tool(
        name="fetch_market_data",
        description="Fetch multi-source market data (Bloomberg/Reuters/Yahoo) for request context.",
    )
    def fetch_market_data(self, context: dict[str, Any], args: dict[str, Any]) -> str:
        request = PipelineRequest(
            ticker=str(context["request"]["ticker"]),
            start_date=context["request"]["start_date"],
            end_date=context["request"]["end_date"],
        )
        source_results = self.market_data_tool.run(request)
        context["source_results"] = source_results
        return f"Fetched {len(source_results)} source datasets."

    @autogen_tool(
        name="evaluate_data_quality",
        description="Evaluate quality metrics for fetched source datasets.",
    )
    @langgraph_tool(
        name="evaluate_data_quality",
        description="Evaluate quality metrics for fetched source datasets.",
    )
    def evaluate_data_quality(self, context: dict[str, Any], args: dict[str, Any]) -> str:
        source_results = context.get("source_results", [])
        if not source_results:
            return "No source data available. Run fetch_market_data first."
        quality = self.data_quality_tool.run(source_results)
        context["quality"] = quality
        return f"Computed quality metrics for {len(quality)} sources."

    @autogen_tool(
        name="select_source",
        description="Select data source from quality-ranked candidates.",
    )
    @langgraph_tool(
        name="select_source",
        description="Select data source from quality-ranked candidates.",
    )
    def select_source(self, context: dict[str, Any], args: dict[str, Any]) -> str:
        quality = context.get("quality", [])
        if not quality:
            return "No quality metrics available. Run evaluate_data_quality first."

        selected = str(args.get("source") or context.get("user_selected_source") or quality[0].source).lower()
        valid_sources = {q.source for q in quality}
        if selected not in valid_sources:
            return f"Invalid source '{selected}'. Valid sources: {sorted(valid_sources)}"

        context["selected_source"] = selected
        return f"Selected source '{selected}'."

    @autogen_tool(
        name="analyze_gaps",
        description="Analyze missing data and recommend gap fill methods for selected source.",
    )
    @langgraph_tool(
        name="analyze_gaps",
        description="Analyze missing data and recommend gap fill methods for selected source.",
    )
    def analyze_gaps(self, context: dict[str, Any], args: dict[str, Any]) -> str:
        source_results = context.get("source_results", [])
        selected_source = context.get("selected_source")
        if not source_results or not selected_source:
            return "Source data and selected_source are required before gap analysis."

        selected_df = next((s.data for s in source_results if s.source == selected_source), pd.DataFrame())
        if selected_df.empty:
            return f"No dataframe available for selected source '{selected_source}'."

        suggestions = self.gap_analysis_tool.run(selected_df["close"])
        context["gap_suggestions"] = suggestions
        return f"Generated {len(suggestions)} gap-fill recommendations."

    @autogen_tool(
        name="select_gap_method",
        description="Choose the gap-fill method to apply for continuous time series construction.",
    )
    @langgraph_tool(
        name="select_gap_method",
        description="Choose the gap-fill method to apply for continuous time series construction.",
    )
    def select_gap_method(self, context: dict[str, Any], args: dict[str, Any]) -> str:
        suggestions = context.get("gap_suggestions", [])
        if not suggestions:
            return "Gap suggestions unavailable. Run analyze_gaps first."

        suggested = suggestions[0].method
        selected = str(args.get("method") or context.get("user_selected_gap_method") or suggested)
        allowed = {s.method for s in suggestions}
        if selected not in allowed:
            return f"Invalid method '{selected}'. Allowed: {sorted(allowed)}"

        context["gap_method"] = selected
        return f"Selected gap method '{selected}'."

    @autogen_tool(
        name="generate_continuous_series",
        description="Generate continuous time series from selected source and gap method.",
    )
    @langgraph_tool(
        name="generate_continuous_series",
        description="Generate continuous time series from selected source and gap method.",
    )
    def generate_continuous_series(self, context: dict[str, Any], args: dict[str, Any]) -> str:
        source_results = context.get("source_results", [])
        selected_source = context.get("selected_source")
        gap_method = context.get("gap_method")

        if not source_results or not selected_source or not gap_method:
            return "Need source_results, selected_source, and gap_method before generation."

        selected_df = next((s.data for s in source_results if s.source == selected_source), pd.DataFrame())
        if selected_df.empty:
            return f"No dataframe for selected source '{selected_source}'."

        continuous = self.generation_tool.run(selected_df, gap_method)
        context["continuous"] = continuous
        return f"Generated continuous series with {len(continuous)} rows."

    @autogen_tool(
        name="export_artifacts",
        description="Export audit artifacts (quality CSV, series CSV, JSON report).",
    )
    @langgraph_tool(
        name="export_artifacts",
        description="Export audit artifacts (quality CSV, series CSV, JSON report).",
    )
    def export_artifacts(self, context: dict[str, Any], args: dict[str, Any]) -> str:
        quality = context.get("quality", [])
        continuous = context.get("continuous")
        if continuous is None:
            return "No continuous series available. Run generate_continuous_series first."

        request = PipelineRequest(
            ticker=str(context["request"]["ticker"]),
            start_date=context["request"]["start_date"],
            end_date=context["request"]["end_date"],
        )
        artifact_paths = self.audit_tool.export(
            framework=str(context.get("framework", "autogen")),
            request=request,
            selected_source=str(context.get("selected_source")),
            gap_method=str(context.get("gap_method")),
            quality=quality,
            continuous=continuous,
            events=context.get("react_events", []),
        )
        context["artifact_paths"] = artifact_paths
        return f"Exported artifacts: {list(artifact_paths.keys())}"

    @autogen_tool(
        name="finish",
        description="Finish orchestration once all required outputs are available.",
    )
    @langgraph_tool(
        name="finish",
        description="Finish orchestration once all required outputs are available.",
    )
    def finish(self, context: dict[str, Any], args: dict[str, Any]) -> str:
        has_required = all(
            [
                context.get("quality") is not None,
                context.get("continuous") is not None,
                context.get("artifact_paths") is not None,
            ]
        )
        if not has_required:
            return "Cannot finish: missing one of quality, continuous, or artifact_paths."
        context["done"] = True
        return "Workflow completed successfully."


def get_autogen_tool_registry(toolbox: CommonToolbox) -> dict[str, tuple[str, Any]]:
    return get_decorated_tools(toolbox, framework="autogen")


def get_langgraph_tool_registry(toolbox: CommonToolbox) -> dict[str, tuple[str, Any]]:
    return get_decorated_tools(toolbox, framework="langgraph")
