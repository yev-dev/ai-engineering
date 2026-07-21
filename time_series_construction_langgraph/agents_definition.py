"""Declarative agent and callback definitions for the time series workflow."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import logging
import re

logger = logging.getLogger(__name__)


class CallbackEventType(str, Enum):
    USER_REQUEST = "user_request"
    AWAITING_USER_INPUT = "awaiting_user_input"
    DATA_SOURCE_SELECTED = "data_source_selected"
    GAP_METHOD_RECOMMENDED = "gap_method_recommended"
    GAP_METHOD_APPLIED = "gap_method_applied"
    TIMESERIES_GENERATED = "timeseries_generated"
    TIMESERIES_DOWNLOADED = "timeseries_downloaded"
    AGENT_COMPLETED = "agent_completed"
    ERROR = "error"


@dataclass
class CallbackEvent:
    type: CallbackEventType
    payload: dict[str, Any] = field(default_factory=dict)
    session_id: str = "default"


@dataclass(frozen=True)
class Agent:
    name: str
    description: str
    system_prompt: str
    tools: list[str]
    goal: str = ""
    guardrails: tuple[str, ...] = ()


_COMMON = """You are a ReAct financial data construction agent. Use tools for facts and never invent data.
Respond with Thought, Action, Action Input JSON, and finally Final Answer. Ask for human
input whenever a user decision is required. Delegate to specialist agents when appropriate."""

AGENT_REGISTRY: dict[str, Agent] = {
    "Orchestrator": Agent(
        "Orchestrator", "Plans the workflow from a natural-language request.",
        _COMMON + " Plan the request and delegate to ReferenceDataAgent. If the request is not a financial time-series request, use request_human_input with a prompt asking for a ticker or security name and a start/end date. Never interpret shell commands as financial requests.",
        ["delegate_to_agent", "request_human_input"],
        "Understand the request, validate that it concerns a financial time series, and route it to the correct specialist.",
        ("Do not retrieve data yourself.", "Do not invent symbols, dates, prices, or agent names.", "Ask for clarification when required fields are missing."),
    ),
    "ReferenceDataAgent": Agent(
        "ReferenceDataAgent", "Resolves a ticker or security name and dates.",
        _COMMON + " Resolve the instrument with get_instrument_details using query=<ticker, symbol, short name, or full security name>. The tool returns the canonical symbol; use that canonical symbol for later tools. Then delegate to MarketDataAgent.",
        ["get_instrument_details", "delegate_to_agent", "request_human_input"],
        "Resolve the user's asset name or ticker to a catalog instrument and preserve the requested date range.",
        ("Use the instrument catalog as the source of truth.", "Tell the user when the instrument is unavailable.", "Never substitute a similar instrument without confirmation."),
    ),
    "MarketDataAgent": Agent(
        "MarketDataAgent", "Loads comparable historical series from all configured sources.",
        _COMMON + " List sources, load each source, then delegate to DataQualityAgent.",
        ["available_data_sources", "historical_prices", "delegate_to_agent"],
        "Retrieve the requested instrument and date range from every available source.",
        ("Report unavailable sources or empty date ranges explicitly.", "Never fill missing values in this stage.", "Preserve source names in every result."),
    ),
    "DataQualityAgent": Agent(
        "DataQualityAgent", "Measures completeness and quality for each source.",
        _COMMON + " Check every series and delegate to ReportingAgent for source selection.",
        ["check_data_quality", "delegate_to_agent"],
        "Compare source completeness and data-quality issues using measurable metrics.",
        ("Do not call a series complete when it contains missing values.", "Include one report per source.", "Explain when no source has usable observations."),
    ),
    "GapFillingAgent": Agent(
        "GapFillingAgent", "Recommends and applies gap-filling methods after source selection.",
        _COMMON + " Recommend methods, ask the user to choose, and apply the selected method.",
        ["recommend_gap_methods", "apply_gap_filling", "request_human_input"],
        "Offer deterministic gap-filling choices and apply only the method selected by the user.",
        ("Never hide the original missing-value count.", "Do not apply a method without user selection.", "Report remaining missing values after filling."),
    ),
    "TimeSeriesConstructionAgent": Agent(
        "TimeSeriesConstructionAgent", "Builds and persists the final continuous series.",
        _COMMON + " Generate the final CSV and chart artifacts.",
        ["build_timeseries", "visualize_timeseries"],
        "Create reproducible CSV and visualization artifacts for the selected continuous series.",
        ("Do not claim an artifact was created unless the tool returns a path.", "Preserve dates and the selected method in output metadata."),
    ),
    "ReportingAgent": Agent(
        "ReportingAgent", "Presents quality summaries and asks for source selection.",
        _COMMON + " Present a concise report and ask the user to choose a source or exit.",
        ["generate_report", "request_human_input"],
        "Present understandable quality results and obtain the user's source or exit decision.",
        ("State limitations plainly.", "Never select a source on the user's behalf.", "Include artifact paths when available."),
    ),
}


def get_agent(name: str) -> Agent | None:
    normalized = re.sub(r"[^a-z0-9]", "", name.casefold())
    agent = next(
        (candidate for candidate_name, candidate in AGENT_REGISTRY.items()
         if re.sub(r"[^a-z0-9]", "", candidate_name.casefold()) == normalized),
        None,
    )
    logger.debug("agent_lookup name=%s found=%s", name, agent is not None)
    return agent