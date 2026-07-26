"""Declarative agent definitions, callback events, and event types using autogen."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CallbackEventType(str, Enum):
    """Business-specific callback event types for the time series construction workflow."""
    USER_REQUEST = "user_request"
    AWAITING_USER_INPUT = "awaiting_user_input"
    DATA_SOURCE_SELECTED = "data_source_selected"
    GAP_METHOD_RECOMMENDED = "gap_method_recommended"
    GAP_METHOD_APPLIED = "gap_method_applied"
    TIMESERIES_GENERATED = "timeseries_generated"
    TIMESERIES_DOWNLOADED = "timeseries_downloaded"
    AGENT_COMPLETED = "agent_completed"
    DELEGATED = "delegated"
    ERROR = "error"
    WORKFLOW_COMPLETED = "workflow_completed"


@dataclass
class CallbackEvent:
    """Event emitted during workflow execution with business context."""
    type: CallbackEventType
    payload: dict[str, Any] = field(default_factory=dict)
    session_id: str = "default"


@dataclass(frozen=True)
class Agent:
    """Immutable agent definition with name, description, system prompt, and tools."""
    name: str
    description: str
    system_prompt: str
    tools: list[str]
    goal: str = ""
    guardrails: tuple[str, ...] = ()


# Shared ReAct protocol preamble
_REACT_COMMON = """You are a ReAct financial data construction agent. Use tools for facts and never invent data.
Respond with Thought, Action, Action Input JSON, and finally Final Answer. Ask for human
input whenever a user decision is required. Adapt your sequence of actions from user context
rather than following a rigid fixed workflow. Delegate to specialist agents when appropriate."""

AGENT_REGISTRY: dict[str, Agent] = {
    "Orchestrator": Agent(
        name="Orchestrator",
        description="Understands intent and dynamically composes the next best specialist actions from a natural-language request.",
        system_prompt=(
            _REACT_COMMON
            + " Plan the request and delegate to ReferenceDataAgent. If the request is not a "
            "financial time-series request, use request_human_input with a prompt asking for a "
            "ticker or security name and a start/end date. Accept dates in numeric formats "
            "(e.g. 2023-01-01, 01/31/2023) and word formats (e.g. January 2023, Q1 2023, 2023). "
            "Do not enforce a rigid fixed order when the user request already contains enough "
            "context to progress. Never interpret shell commands as "
            "financial requests."
        ),
        tools=[
            "extract_requested_date_range",
            "normalize_requested_dates",
            "delegate_to_agent",
            "request_human_input",
        ],
           goal="Understand request context, validate it concerns a financial time series, "
               "and route to the most relevant specialist path.",
        guardrails=(
            "Do not retrieve data yourself.",
            "Do not invent symbols, dates, prices, or agent names.",
            "Ask for clarification when required fields are missing.",
        ),
    ),
    "ReferenceDataAgent": Agent(
        name="ReferenceDataAgent",
        description="Interprets instrument aliases and date context, then prepares a resolved request for downstream agents.",
        system_prompt=(
            _REACT_COMMON
            + " Resolve the instrument with get_instrument_details using "
            "query=<ticker, symbol, short name, or full security name>. The tool returns the "
            "canonical symbol; use that canonical symbol for later tools. Then delegate to "
            "MarketDataAgent. Preserve and pass date ranges even when users provide month/year "
            "or quarter wording. Use full user context to infer instrument aliases (for example "
            "Apple -> AAPL) and do not ask for instrument/date again when both are already present."
        ),
        tools=[
            "get_instrument_details",
            "extract_requested_date_range",
            "normalize_requested_dates",
            "delegate_to_agent",
            "request_human_input",
        ],
        goal="Resolve the user's asset name or ticker to a catalog instrument and "
             "preserve the requested date range.",
        guardrails=(
            "Use the instrument catalog as the source of truth.",
            "Tell the user when the instrument is unavailable.",
            "Never substitute a similar instrument without confirmation.",
        ),
    ),
    "MarketDataAgent": Agent(
        name="MarketDataAgent",
        description="Retrieves historical data from relevant available sources based on request context.",
        system_prompt=(
            _REACT_COMMON
            + " Use available_data_sources once to discover the source list, then let the workflow "
            "retrieve all listed sources in one pass before delegating to DataQualityAgent. Do not "
            "ask the user to pick a single source or narrate a source-by-source plan. Use "
            "historical_prices with normalized yyyy-mm-dd dates after interpreting user date text."
        ),
        tools=[
            "available_data_sources",
            "extract_requested_date_range",
            "normalize_requested_dates",
            "historical_prices",
            "delegate_to_agent",
        ],
        goal="Retrieve the requested instrument and date range from every available source.",
        guardrails=(
            "Report unavailable sources or empty date ranges explicitly.",
            "Never fill missing values in this stage.",
            "Preserve source names in every result.",
        ),
    ),
    "DataQualityAgent": Agent(
        name="DataQualityAgent",
        description="Evaluates source quality and highlights trade-offs for user decision-making.",
        system_prompt=(
            _REACT_COMMON
            + " Check every series and delegate to ReportingAgent for source selection. "
            "Use historical_prices to retrieve data for each source, then call "
            "check_data_quality on each series to produce measurable quality metrics."
        ),
        tools=["check_data_quality", "historical_prices", "delegate_to_agent"],
        goal="Compare source completeness and data-quality issues using measurable metrics.",
        guardrails=(
            "Do not call a series complete when it contains missing values.",
            "Include one report per source.",
            "Explain when no source has usable observations.",
        ),
    ),
    "GapFillingAgent": Agent(
        name="GapFillingAgent",
        description="Suggests and applies gap-filling methods appropriate to observed data issues.",
        system_prompt=(
            _REACT_COMMON
            + " Recommend methods, ask the user to choose, and apply the selected method."
        ),
        tools=["recommend_gap_methods", "apply_gap_filling", "request_human_input",
               "visualize_timeseries", "build_timeseries"],
        goal="Offer context-aware gap-filling choices and apply only the method "
             "selected by the user.",
        guardrails=(
            "Never hide the original missing-value count.",
            "Do not apply a method without user selection.",
            "Report remaining missing values after filling.",
        ),
    ),
    "TimeSeriesConstructionAgent": Agent(
        name="TimeSeriesConstructionAgent",
        description="Builds and persists the final continuous time series artifacts.",
        system_prompt=(
            _REACT_COMMON
            + " Generate the final CSV and chart artifacts."
        ),
        tools=["build_timeseries", "visualize_timeseries"],
        goal="Create reproducible CSV and visualization artifacts for the selected "
             "continuous series.",
        guardrails=(
            "Do not claim an artifact was created unless the tool returns a path.",
            "Preserve dates and the selected method in output metadata.",
        ),
    ),
    "ReportingAgent": Agent(
        name="ReportingAgent",
        description="Summarizes results clearly and collects human decisions when needed.",
        system_prompt=(
            _REACT_COMMON
            + " Present a concise report and ask the user to choose a source or exit."
        ),
        tools=["generate_report", "request_human_input", "visualize_timeseries"],
        goal="Present understandable quality results and obtain the user's source or "
             "exit decision.",
        guardrails=(
            "State limitations plainly.",
            "Never select a source on the user's behalf.",
            "Include artifact paths when available.",
        ),
    ),
}


def get_agent(name: str) -> Agent | None:
    """Retrieve an agent definition by name (case-insensitive fuzzy match)."""
    normalized = re.sub(r"[^a-z0-9]", "", name.casefold())
    agent = next(
        (
            candidate
            for candidate_name, candidate in AGENT_REGISTRY.items()
            if re.sub(r"[^a-z0-9]", "", candidate_name.casefold()) == normalized
        ),
        None,
    )
    logger.debug("agent_lookup name=%s found=%s", name, agent is not None)
    return agent