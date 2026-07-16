from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from ..audit import AuditArtifactManager
from ..connectors import MultiSourceMarketDataAgent
from ..gap_fill import GapFillAgent
from ..models import DataSourceResult, PipelineRequest, QualityMetric, ReActEvent
from ..quality import DataQualityAgent


class MarketDataTool:
    def __init__(self, connector_agent: MultiSourceMarketDataAgent) -> None:
        self.connector_agent = connector_agent

    def run(self, request: PipelineRequest) -> list[DataSourceResult]:
        return self.connector_agent.fetch(request)


class DataQualityTool:
    def __init__(self, quality_agent: DataQualityAgent) -> None:
        self.quality_agent = quality_agent

    def run(self, source_results: list[DataSourceResult]) -> list[QualityMetric]:
        return self.quality_agent.evaluate(source_results)


class GapAnalysisTool:
    def __init__(self, gap_fill_agent: GapFillAgent) -> None:
        self.gap_fill_agent = gap_fill_agent

    def run(self, series: pd.Series):
        return self.gap_fill_agent.suggest(series)


class TimeSeriesGenerationTool:
    def __init__(self, gap_fill_agent: GapFillAgent) -> None:
        self.gap_fill_agent = gap_fill_agent

    def run(self, source_df: pd.DataFrame, method: str) -> pd.DataFrame:
        deduped = source_df.drop_duplicates(subset=["date"], keep="last").copy()
        return self.gap_fill_agent.apply(deduped, method)


class AuditExportTool:
    def __init__(self, artifact_manager: AuditArtifactManager) -> None:
        self.artifact_manager = artifact_manager

    def export(
        self,
        framework: str,
        request: PipelineRequest,
        selected_source: str,
        gap_method: str,
        quality: list[QualityMetric],
        continuous: pd.DataFrame,
        events: list[ReActEvent],
    ) -> dict[str, str]:
        quality_df = pd.DataFrame([asdict(q) for q in quality]) if quality else pd.DataFrame()
        artifact_paths = {
            "quality_csv": self.artifact_manager.export_dataframe(quality_df, "quality_summary.csv"),
            "continuous_series_csv": self.artifact_manager.export_dataframe(continuous, "continuous_series.csv"),
        }
        report = self.artifact_manager.build_run_report(
            framework=framework,
            request=asdict(request),
            selected_source=selected_source,
            gap_method=gap_method,
            quality=quality,
            react_events=events,
            artifact_paths=artifact_paths,
        )
        artifact_paths["run_report_json"] = self.artifact_manager.export_json(report, "run_report.json")
        return artifact_paths
