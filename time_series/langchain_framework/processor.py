from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from rich.console import Console

from ..common.audit import AuditArtifactManager, generate_run_id
from ..common.connectors import build_default_market_data_agent
from ..common.gap_fill import GapFillAgent
from ..common.llm_clients import build_llm_client
from ..common.orchestration import LLMOrchestrator
from ..common.reporting import CLIReporter
from ..common.tools.external_services import ExternalConnectivityService, ServiceEndpoint
from ..common.tools.pipeline_tools import (
    AuditExportTool,
    DataQualityTool,
    GapAnalysisTool,
    MarketDataTool,
    TimeSeriesGenerationTool,
)
from ..common.tools.toolbox import CommonToolbox
from ..common.quality import DataQualityAgent

from .react_engine import LangChainReActEngine
from .tools import build_langgraph_tool_registry


@dataclass
class LangChainProcessorConfig:
    ticker: str
    start: date
    end: date
    source: str | None
    gap_method: str | None
    yahoo_mode: str
    llm_client: str
    llm_model: str | None
    llm_base_url: str | None
    llm_api_key: str | None
    llm_temperature: float
    llm_max_tokens: int
    export_dir: str
    run_id: str | None
    check_services: bool


class LangChainProcessor:
    def __init__(self, config: LangChainProcessorConfig, console: Console | None = None) -> None:
        self.config = config
        self.console = console or Console()
        self.reporter = CLIReporter(self.console)

    def _build_tools(self, run_id: str) -> CommonToolbox:
        use_live_yahoo = self.config.yahoo_mode == "live"
        market_data_agent = build_default_market_data_agent(use_live_yahoo=use_live_yahoo, seed=2002)
        artifact_mgr = AuditArtifactManager(export_dir=self.config.export_dir, run_id=run_id)

        market_data_tool = MarketDataTool(connector_agent=market_data_agent)
        data_quality_tool = DataQualityTool(quality_agent=DataQualityAgent())
        gap_fill_agent = GapFillAgent()
        gap_analysis_tool = GapAnalysisTool(gap_fill_agent=gap_fill_agent)
        generation_tool = TimeSeriesGenerationTool(gap_fill_agent=gap_fill_agent)
        audit_tool = AuditExportTool(artifact_manager=artifact_mgr)

        return CommonToolbox(
            market_data_tool=market_data_tool,
            data_quality_tool=data_quality_tool,
            gap_analysis_tool=gap_analysis_tool,
            generation_tool=generation_tool,
            audit_tool=audit_tool,
        )

    def execute(self) -> None:
        llm_client = build_llm_client(
            client_name=self.config.llm_client,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
            api_base=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
        )
        if llm_client is None:
            raise ValueError("LLM orchestration requires --llm-client set to 'copilot' or 'ollama'.")

        run_id = self.config.run_id or generate_run_id(prefix="langchain")
        toolbox = self._build_tools(run_id=run_id)
        tool_registry = build_langgraph_tool_registry(toolbox)

        if self.config.check_services:
            services = ExternalConnectivityService()
            services.register_endpoint(ServiceEndpoint(name="yahoo", base_url="https://query1.finance.yahoo.com"))
            if self.config.llm_base_url:
                services.register_endpoint(ServiceEndpoint(name="llm", base_url=self.config.llm_base_url))
            self.console.print("Service connectivity status:", services.health_check_all())

        orchestrator = LLMOrchestrator(llm_client=llm_client, framework="langgraph", max_steps=20)
        react = LangChainReActEngine()

        context: dict[str, Any] = {
            "framework": "langchain",
            "request": {
                "ticker": self.config.ticker,
                "start_date": self.config.start,
                "end_date": self.config.end,
            },
            "user_selected_source": self.config.source,
            "user_selected_gap_method": self.config.gap_method,
            "react_events": react.events,
            "done": False,
        }
        history: list[dict[str, Any]] = []

        for _ in range(orchestrator.max_steps):
            decision = orchestrator.next_action(context=context, history=history, tool_registry=tool_registry)
            _desc, func = tool_registry[decision.tool]
            observation = func(context, decision.args)

            react.node(
                decision.thought,
                f"Tool: {decision.tool} args={decision.args}",
                str(observation),
            )
            history.append(
                {
                    "thought": decision.thought,
                    "tool": decision.tool,
                    "args": decision.args,
                    "observation": str(observation),
                }
            )

            if context.get("quality") is not None:
                self.reporter.print_quality_table(context["quality"])
            if context.get("gap_suggestions") is not None:
                self.reporter.print_gap_suggestions(context["gap_suggestions"])

            if context.get("done"):
                break

        if not context.get("done"):
            raise RuntimeError("LLM orchestrator did not complete workflow within max steps.")

        continuous = context.get("continuous")
        selected_source = str(context.get("selected_source"))
        gap_method = str(context.get("gap_method"))
        artifact_paths = context.get("artifact_paths", {})

        if continuous is None:
            raise RuntimeError("Workflow completed without generated continuous series.")

        self.reporter.print_generated_series_summary(continuous, selected_source, gap_method)
        self.reporter.print_react_trace(react.events, "LangChain ReACT Trace")
        self.reporter.print_artifacts(run_id=run_id, artifact_paths=artifact_paths)

    # LLM-driven only: no deterministic fallback helper here.
