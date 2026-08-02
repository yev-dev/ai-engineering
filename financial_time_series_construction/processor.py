"""LLM-driven ReAct processor with callback-based pause and resume using autogen patterns."""
from __future__ import annotations

import json
import logging
import re
import ast
from typing import Any

from financial_time_series_construction.agents_definition import (
    Agent,
    CallbackEvent,
    CallbackEventType,
    get_agent,
)
from financial_time_series_construction.handler import TimeSeriesConstructionHandler
from financial_time_series_construction.models import LLMRequest, ModelRequestFactory
from financial_time_series_construction.tools import (
    SOURCES,
    extract_date_range,
    get_instrument_details,
    get_tool,
    get_tool_description,
    normalize_date_range,
    set_run_id,
    _get_data_store,
)
from financial_time_series_construction.tool_models import validate_tool_input

from financial_time_series_construction.prompts import (
    agent_system_prompt,
    request_prompt,
    unavailable_message,
)
from financial_time_series_construction.debug_logger import (
    LoopDetector,
    log_message_size,
    log_workflow_progress,
    timer,
)

logger = logging.getLogger(__name__)

_MARKETDATA_ALL_SOURCES_NOTE = (
    "[SYSTEM_NOTE] AUTO_RETRIEVE_ALL_SOURCES: Do not ask the user to choose a single "
    "source. Retrieve historical_prices for every source returned by available_data_sources, "
    "then delegate to DataQualityAgent with the aggregated context."
)

_REFERENCE_EXECUTE_NOTE = (
    "[SYSTEM_NOTE] EXECUTE_TOOL_NOW: You must call get_instrument_details with the resolved "
    "query parameter. Do not describe what you will do - call the tool now and return the actual result."
)

_REPORTING_FINAL_SUMMARY_NOTE = (
    "[SYSTEM_NOTE] FINAL_SUMMARY_ONLY: When TimeSeriesConstructionAgent is ready to hand off "
    "to ReportingAgent, keep the request focused on summarizing the completed artifacts."
)


class TimeSeriesConstructionProcessor:
    """Main processor that orchestrates the ReACT workflow.

    Manages the LLM-driven ReACT loop with delegation, human-in-the-loop
    pauses, and callback event emission.
    """

    def __init__(
        self,
        factory: ModelRequestFactory | None = None,
        handler: TimeSeriesConstructionHandler | None = None,
    ) -> None:
        self.factory = factory or ModelRequestFactory.from_environment()
        self.handler = handler or TimeSeriesConstructionHandler()
        self.pending_events: list[CallbackEvent] = []
        logger.info("processor_initialized session_id=%s", self.handler.session_id)

    def _debug_flow_event(
        self,
        component: str,
        stage: str,
        agent: str | None = None,
        iteration: int | None = None,
        detail: str | None = None,
    ) -> None:
        """Emit a structured workflow flow event for debug tracing."""
        parts = [f"component={component}", f"stage={stage}"]
        if agent is not None:
            parts.append(f"agent={agent}")
        if iteration is not None:
            parts.append(f"iteration={iteration}")
        if detail:
            parts.append(f"detail={detail}")
        logger.debug("flow_event %s", " ".join(parts))

    def process_user_request(self, user_input: str) -> list[CallbackEvent]:
        """Process an initial user request through the workflow.

        Args:
            user_input: The user's natural language request.

        Returns:
            List of callback events generated during processing.
        """
        user_input = user_input.strip()
        logger.info("request_received characters=%d", len(user_input))
        self.handler.add_trace_record(
            "user_request",
            {"content": user_input},
            agent="User",
        )

        if self._is_environment_command(user_input):
            logger.warning("request_rejected reason=environment_command")
            return [
                CallbackEvent(
                    CallbackEventType.ERROR,
                    {
                        "agent": "System",
                        "message": (
                            "Shell/conda commands are not supported inside this application. "
                            "Activate your conda environment in your terminal before starting this app, "
                            "then enter a financial request such as "
                            "'Build AAPL from 2023-01-01 to 2023-12-31'."
                        ),
                        "recoverable": True,
                        "user_action": (
                            "Activate conda in your terminal before running this app, "
                            "then enter a valid financial request."
                        ),
                    },
                )
            ]

        if not user_input or not self._looks_like_financial_request(user_input):
            logger.info("request_requires_clarification")
            return [
                CallbackEvent(
                    CallbackEventType.AWAITING_USER_INPUT,
                    {
                        "agent": "Orchestrator",
                        "prompt": request_prompt(),
                        "options": [],
                    },
                )
            ]

        direct_events = self._try_direct_delegate_from_request(user_input)
        if direct_events is not None:
            return direct_events

        self.handler.emit(
            CallbackEvent(CallbackEventType.USER_REQUEST, {"request": user_input})
        )
        logger.info("workflow_started agent=Orchestrator")
        return self._run_agent(
            get_agent("Orchestrator"),
            [{"role": "user", "content": user_input}],
        )

    def _try_direct_delegate_from_request(self, user_input: str) -> list[CallbackEvent] | None:
        """Deterministically delegate when request already includes instrument + date range.

        This prevents an avoidable first-turn clarification loop from the Orchestrator
        for requests such as "build AAPL from January 2023 to January 2024".
        """
        try:
            extracted = extract_date_range(user_input)
        except ValueError:
            return None
        if extracted is None:
            return None

        instrument_query = self._extract_instrument_query(user_input)
        if not instrument_query:
            return None

        start_date, end_date = extracted
        enriched_request = (
            f"instrument={instrument_query}; start_date={start_date}; end_date={end_date}; "
            f"original_request={user_input}"
        )
        target = get_agent("ReferenceDataAgent")
        if target is None:
            return [
                CallbackEvent(
                    CallbackEventType.ERROR,
                    {"message": "ReferenceDataAgent is not registered."},
                    self.handler.session_id,
                )
            ]

        logger.info(
            "request_direct_delegate mode=deterministic instrument=%s start=%s end=%s reason=request_contains_instrument_and_date_range",
            instrument_query,
            start_date,
            end_date,
        )
        pre_events = [
            CallbackEvent(CallbackEventType.USER_REQUEST, {"request": user_input}, self.handler.session_id),
            CallbackEvent(
                CallbackEventType.AGENT_COMPLETED,
                {"agent": "Orchestrator", "result": {"delegated_to": target.name}},
                self.handler.session_id,
            ),
            CallbackEvent(
                CallbackEventType.DELEGATED,
                {
                    "from_agent": "Orchestrator",
                    "to_agent": target.name,
                    "request": enriched_request,
                    "routing_mode": "deterministic",
                    "routing_reason": "bypassed Orchestrator clarification: request already includes instrument and date range",
                },
                self.handler.session_id,
            ),
        ]
        events = self._run_agent(
            target,
            [{"role": "user", "content": enriched_request}],
        )
        return pre_events + (events if self._is_event_list(events) else [])

    @staticmethod
    def _extract_instrument_query(user_input: str) -> str | None:
        """Best-effort extraction of ticker/security query from free-form request."""
        text = user_input.strip()
        if not text:
            return None

        kv_match = re.search(r"\binstrument\s*=\s*([^;\n]+)", text, re.IGNORECASE)
        if kv_match:
            value = kv_match.group(1).strip(" \t,.;")
            if value:
                return value

        range_split = re.split(
            r"\b(?:from|between|start\s+date)\b",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )
        head = range_split[0].strip(" ,.;") if range_split else text
        cleaned = re.sub(
            r"\b(?:build|create|construct|generate|make|for|time\s*series|series|historical|data|prices|price)\b",
            " ",
            head,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\b(?:start\s+date|end\s+date)\b.*$", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;")
        if cleaned:
            return cleaned

        token_match = re.search(r"\b[A-Z]{1,6}\b", text)
        if token_match:
            return token_match.group(0)
        return None

    @staticmethod
    def _extract_original_request_from_messages(messages: list[dict[str, str]]) -> str:
        """Extract the original user request from enriched agent messages."""
        user_texts = [
            msg.get("content", "")
            for msg in messages
            if msg.get("role") == "user"
        ]
        all_user_text = " ".join(user_texts).strip()
        original_match = re.search(r"original_request=(.+)", all_user_text)
        return original_match.group(1).strip() if original_match else all_user_text

    @staticmethod
    def _extract_symbol_candidate_from_text(text: str) -> str | None:
        """Extract a likely ticker symbol token from free-form text."""
        if not text:
            return None
        # Prefer an explicit parenthesized ticker, e.g. "Apple Inc. (AAPL)".
        parenthesized = re.search(r"\(([A-Za-z]{2,8}(?:-[A-Za-z])?)\)", text)
        if parenthesized:
            return parenthesized.group(1).upper()
        # Prefer explicit symbol/ticker patterns before generic uppercase tokens.
        explicit = re.search(
            r"\b(?:symbol|ticker)\s*(?:is|=|:)?\s*([A-Za-z]{2,8}(?:-[A-Za-z])?)\b",
            text,
            re.IGNORECASE,
        )
        if explicit:
            return explicit.group(1).upper()
        generic = re.search(r"\b([A-Z]{2,6})\b", text)
        if generic:
            return generic.group(1)
        return None

    @staticmethod
    def _extract_sources_from_text(text: str) -> list[str]:
        """Extract known market data sources from free-form text."""
        lowered = str(text or "").casefold()
        found: list[str] = []
        for source in ("yahoo", "bloomberg", "reuters"):
            if source in lowered:
                found.append(source)
        return found

    @staticmethod
    def _extract_available_sources_from_messages(messages: list[dict[str, str]]) -> list[str]:
        """Extract a source list from available_data_sources tool results."""
        sources: list[str] = []
        seen: set[str] = set()
        for message in messages:
            if message.get("role") != "user" or "Tool result:" not in message.get("content", ""):
                continue
            try:
                payload = json.loads(message["content"].replace("Tool result: ", "", 1))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, list):
                for item in payload:
                    source = str(item).strip().casefold()
                    if source in SOURCES and source not in seen:
                        seen.add(source)
                        sources.append(source)
            elif isinstance(payload, dict):
                candidate_sources = payload.get("sources")
                if isinstance(candidate_sources, list):
                    for item in candidate_sources:
                        source = str(item).strip().casefold()
                        if source in SOURCES and source not in seen:
                            seen.add(source)
                            sources.append(source)
        return sources

    @staticmethod
    def _extract_market_source_context_from_messages(
        messages: list[dict[str, str]],
    ) -> tuple[list[str], list[dict[str, str]], str | None]:
        """Extract loaded market sources, unavailable sources, and symbol from tool results."""
        loaded_sources: list[str] = []
        unavailable_sources: list[dict[str, str]] = []
        resolved_symbol: str | None = None
        seen_loaded: set[str] = set()
        seen_unavailable: set[tuple[str, str]] = set()

        for message in messages:
            if message.get("role") != "user" or "Tool result:" not in message.get("content", ""):
                continue
            try:
                payload = json.loads(message["content"].replace("Tool result: ", "", 1))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue

            if payload.get("dates") and payload.get("source"):
                source = str(payload.get("source", "")).strip().casefold()
                if source and source not in seen_loaded:
                    seen_loaded.add(source)
                    loaded_sources.append(source)
                if resolved_symbol is None and payload.get("symbol"):
                    resolved_symbol = str(payload.get("symbol"))
                continue

            if payload.get("market_data_available") is False or payload.get("non_fatal") is True:
                source = str(payload.get("source", "")).strip().casefold()
                reason = str(payload.get("error", payload.get("message", "source unavailable"))).strip()
                if not source:
                    continue
                key = (source, reason)
                if key in seen_unavailable:
                    continue
                seen_unavailable.add(key)
                unavailable_sources.append({"source": source, "reason": reason})

        return loaded_sources, unavailable_sources, resolved_symbol

    def _deterministically_load_market_sources(
        self,
        agent: Agent,
        messages: list[dict[str, str]],
        iteration: int,
        symbol: str,
        start_date: str,
        end_date: str,
        candidate_sources: list[str],
        loaded_sources: list[str],
    ) -> None:
        """Load any missing market-data sources without additional LLM turns."""
        historical_tool = get_tool("historical_prices")
        if historical_tool is None:
            return

        loaded = {source.casefold() for source in loaded_sources}
        for source in candidate_sources:
            source_key = str(source).strip().casefold()
            if not source_key or source_key in loaded:
                continue

            tool_args = {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "source": source_key,
            }
            try:
                result = historical_tool.invoke(tool_args)
            except Exception as error:
                result = {
                    "market_data_available": False,
                    "non_fatal": True,
                    "source": source_key,
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "error": str(error),
                }

            self.handler.add_trace_record(
                "tool_call",
                {
                    "tool": "historical_prices",
                    "description": get_tool_description("historical_prices"),
                    "arguments": tool_args,
                },
                agent=agent.name,
                iteration=iteration,
            )
            self.handler.add_trace_record(
                "tool_result",
                {
                    "tool": "historical_prices",
                    "description": get_tool_description("historical_prices"),
                    "result": result,
                },
                agent=agent.name,
                iteration=iteration,
            )
            # Strip full payload from LLM-visible message to avoid token bloat
            llm_result = TimeSeriesConstructionProcessor._strip_payload_for_llm(result)
            messages.append({"role": "user", "content": f"Tool result: {json.dumps(llm_result, default=str)}"})

            if isinstance(result, dict) and result.get("source") and result.get("dates"):
                loaded.add(source_key)

    @staticmethod
    def _extract_explicit_source_selection(text: str) -> str | None:
        """Extract user-explicit source selection from text.

        Returns a source only when exactly one supported source is present.
        This avoids false positives from report text that lists all sources.
        """
        mentioned = TimeSeriesConstructionProcessor._extract_sources_from_text(text)
        unique = sorted(set(mentioned))
        if len(unique) == 1:
            return unique[0]
        return None

    @staticmethod
    def _extract_selected_source_marker_from_messages(messages: list[dict[str, str]]) -> str | None:
        """Extract source from explicit checkpoint marker lines.

        Marker format is injected by deterministic resume logic:
        [SOURCE_SELECTION] <source>
        """
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content", "").strip()
            if not content.startswith("[SOURCE_SELECTION]"):
                continue
            selected = TimeSeriesConstructionProcessor._extract_explicit_source_selection(content)
            if selected:
                return selected
        return None

    @staticmethod
    def _extract_unavailable_market_sources_from_messages(
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Extract structured unavailable-source records from prior tool results."""
        unavailable: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for message in messages:
            if message.get("role") != "user" or "Tool result:" not in message.get("content", ""):
                continue
            try:
                payload = json.loads(message["content"].replace("Tool result: ", "", 1))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("market_data_available") is True:
                continue
            if payload.get("market_data_available") is False or payload.get("non_fatal") is True:
                source = str(payload.get("source", "")).strip().casefold()
                reason = str(payload.get("error", payload.get("message", "source unavailable"))).strip()
                if not source:
                    continue
                key = (source, reason)
                if key in seen:
                    continue
                seen.add(key)
                unavailable.append({"source": source, "reason": reason})
        return unavailable

    def _log_continuation_decision(
        self,
        agent: str,
        iteration: int,
        status: str,
        detail: str,
    ) -> None:
        """Emit structured logs for continuation routing diagnostics."""
        log_workflow_progress(agent, iteration, status, detail=detail)
        logger.info(
            "workflow_continuation agent=%s iteration=%d status=%s detail=%s",
            agent,
            iteration,
            status,
            detail,
        )
        self.handler.add_trace_record(
            "continuation_decision",
            {
                "status": status,
                "detail": detail,
            },
            agent=agent,
            iteration=iteration,
        )

    @staticmethod
    def _extract_quality_rows_from_messages(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Extract quality rows from prior tool results or transfer context."""
        rows: list[dict[str, Any]] = []
        historical_by_source: dict[str, dict[str, Any]] = {}

        # First: direct tool results from check_data_quality.
        for message in messages:
            if message.get("role") != "user" or "Tool result:" not in message.get("content", ""):
                continue
            try:
                payload = json.loads(message["content"].replace("Tool result: ", "", 1))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("source") and "completeness_pct" in payload:
                rows.append(payload)
            if (
                isinstance(payload, dict)
                and payload.get("source")
                and isinstance(payload.get("dates"), list)
                and isinstance(payload.get("prices"), list)
            ):
                source_key = str(payload.get("source", "")).strip().casefold()
                if source_key and source_key not in historical_by_source:
                    historical_by_source[source_key] = payload

        # Fallback: parse delegated transfer context "Quality data: [...]".
        if not rows:
            for message in messages:
                if message.get("role") != "user":
                    continue
                text = message.get("content", "")
                match = re.search(
                    r"Quality data:\s*(\[.*?\])(?:\s*\.\s*Original request:|$)",
                    text,
                    flags=re.DOTALL,
                )
                if not match:
                    continue
                try:
                    parsed = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and item.get("source"):
                            rows.append(item)
                if rows:
                    break

        # Deduplicate by source, preserving first occurrence.
        deduped: list[dict[str, Any]] = []
        seen_sources: set[str] = set()
        for item in rows:
            source = str(item.get("source", "")).strip().casefold()
            if not source or source in seen_sources:
                continue
            seen_sources.add(source)

            # Enrich from historical_prices payload when quality rows are sparse.
            historical = historical_by_source.get(source)
            if historical is not None:
                dates = historical.get("dates")
                prices = historical.get("prices")
                if isinstance(dates, list) and isinstance(prices, list) and len(dates) == len(prices):
                    available_indices = [index for index, value in enumerate(prices) if value is not None]
                    if item.get("available_record_count") is None:
                        item["available_record_count"] = len(available_indices)
                    if available_indices:
                        if item.get("min_date") is None:
                            item["min_date"] = str(dates[min(available_indices)])
                        if item.get("max_date") is None:
                            item["max_date"] = str(dates[max(available_indices)])
                        observed_prices = [prices[index] for index in available_indices]
                        if observed_prices and item.get("min_value") is None:
                            item["min_value"] = float(min(observed_prices))
                        if observed_prices and item.get("max_value") is None:
                            item["max_value"] = float(max(observed_prices))

            deduped.append(item)
        return deduped

    @staticmethod
    def _build_reporting_selection_prompt(quality_rows: list[dict[str, Any]]) -> tuple[str, list[str]]:
        """Build a source-selection prompt from quality rows.

        Returns:
            (prompt_text, source_options)
        """
        if not quality_rows:
            options = ["yahoo", "bloomberg", "reuters"]
            prompt = (
                "Data quality comparison is ready. Please choose one source to continue "
                "to gap-filling: yahoo, bloomberg, or reuters."
            )
            return prompt, options

        lines = [
            "Data quality summary by source:",
        ]
        options: list[str] = []
        for row in quality_rows:
            source = str(row.get("source", "unknown")).strip().casefold()
            options.append(source)
            completeness = row.get("completeness_pct", "n/a")
            missing = row.get("missing_count", row.get("nan_count", "n/a"))
            issues = row.get("issues", [])
            issue_text = ", ".join(str(item) for item in issues) if issues else "none"
            lines.append(
                f"- {source}: completeness={completeness}%, missing={missing}, issues={issue_text}"
            )

        lines.append("Select one source to continue to gap filling.")
        return "\n".join(lines), options

    @staticmethod
    def _extract_explicit_gap_method(text: str) -> str | None:
        """Extract an explicit gap-filling method choice from text."""
        lowered = str(text or "").casefold()
        numeric_match = re.search(r"\b([1-4])\b", lowered)
        if numeric_match:
            numeric_map = {
                "1": "linear_interpolation",
                "2": "forward_fill",
                "3": "backward_fill",
                "4": "none",
            }
            mapped = numeric_map.get(numeric_match.group(1))
            if mapped:
                return mapped
        aliases = {
            "linear_interpolation": ["linear_interpolation", "linear interpolation"],
            "forward_fill": ["forward_fill", "forward fill", "ffill"],
            "backward_fill": ["backward_fill", "backward fill", "bfill"],
            "none": ["none", "no filling", "skip"],
        }
        matched: list[str] = []
        for canonical, values in aliases.items():
            if any(value in lowered for value in values):
                matched.append(canonical)
        unique = sorted(set(matched))
        return unique[0] if len(unique) == 1 else None

    @staticmethod
    def _extract_gap_method_options_from_messages(messages: list[dict[str, str]]) -> list[str]:
        """Extract recommended gap methods from tool results if available."""
        options: list[str] = []
        for message in messages:
            if message.get("role") != "user" or "Tool result:" not in message.get("content", ""):
                continue
            content = message.get("content", "")
            raw = content.replace("Tool result: ", "", 1)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                for item in parsed:
                    method = str(item).strip().casefold()
                    if method in {
                        "linear_interpolation",
                        "forward_fill",
                        "backward_fill",
                        "none",
                    }:
                        options.append(method)

        if not options:
            return ["linear_interpolation", "forward_fill", "backward_fill", "none"]

        return list(dict.fromkeys(options))

    @staticmethod
    def _build_gap_method_prompt(options: list[str]) -> str:
        """Build a reusable method-selection prompt."""
        formatted = ", ".join(options)
        return (
            "Please choose a gap-filling method to continue time-series construction. "
            f"Available methods: {formatted}."
        )

    @staticmethod
    def _build_data_quality_report(
        quality_rows: list[dict[str, Any]],
        unavailable_sources: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Build a JSON-serializable data quality report.

        The rows payload is list-of-dicts so it can be serialized directly to a
        DataFrame by downstream consumers.
        """
        normalized_rows: list[dict[str, Any]] = []
        completeness_candidates: list[tuple[str, float]] = []
        total_missing = 0
        total_available = 0
        symbol = None
        min_dates: list[str] = []
        max_dates: list[str] = []
        min_values: list[float] = []
        max_values: list[float] = []

        for item in quality_rows:
            row = {
                "source": item.get("source"),
                "symbol": item.get("symbol"),
                "total_values": item.get("total_values"),
                "available_record_count": item.get("available_record_count"),
                "missing_count": item.get("missing_count", item.get("nan_count")),
                "completeness_pct": item.get("completeness_pct"),
                "min_value": item.get("min_value"),
                "max_value": item.get("max_value"),
                "min_date": item.get("min_date"),
                "max_date": item.get("max_date"),
                "duplicate_count": item.get("duplicate_count"),
                "issues": item.get("issues", []),
                "note": item.get("note"),
            }
            normalized_rows.append(row)

            if symbol is None and row.get("symbol"):
                symbol = row.get("symbol")

            missing_value = row.get("missing_count")
            if isinstance(missing_value, int):
                total_missing += missing_value

            available_value = row.get("available_record_count")
            if isinstance(available_value, int):
                total_available += available_value

            min_date = row.get("min_date")
            if isinstance(min_date, str) and min_date:
                min_dates.append(min_date)

            max_date = row.get("max_date")
            if isinstance(max_date, str) and max_date:
                max_dates.append(max_date)

            min_value = row.get("min_value")
            if isinstance(min_value, (int, float)):
                min_values.append(float(min_value))

            max_value = row.get("max_value")
            if isinstance(max_value, (int, float)):
                max_values.append(float(max_value))

            completeness_value = row.get("completeness_pct")
            if isinstance(completeness_value, (int, float)) and row.get("source"):
                completeness_candidates.append((str(row["source"]), float(completeness_value)))

        best_source = None
        worst_source = None
        average_completeness = None
        if completeness_candidates:
            best_source = max(completeness_candidates, key=lambda item: item[1])[0]
            worst_source = min(completeness_candidates, key=lambda item: item[1])[0]
            average_completeness = round(
                sum(item[1] for item in completeness_candidates) / len(completeness_candidates),
                2,
            )

        unavailable_sources = unavailable_sources or []
        unavailable_source_names = [
            item.get("source")
            for item in unavailable_sources
            if item.get("source")
        ]

        return {
            "report_type": "data_quality_summary",
            "rows": normalized_rows,
            "summary": {
                "symbol": symbol,
                "source_count": len(normalized_rows),
                "sources": [row.get("source") for row in normalized_rows if row.get("source")],
                "unavailable_source_count": len(unavailable_sources),
                "unavailable_sources": unavailable_source_names,
                "total_available_records": total_available,
                "total_missing_count": total_missing,
                "min_date": min(min_dates) if min_dates else None,
                "max_date": max(max_dates) if max_dates else None,
                "min_value": min(min_values) if min_values else None,
                "max_value": max(max_values) if max_values else None,
                "average_completeness_pct": average_completeness,
                "best_source_by_completeness": best_source,
                "worst_source_by_completeness": worst_source,
            },
            "unavailable_sources": unavailable_sources,
        }

    @staticmethod
    def _is_environment_command(user_input: str) -> bool:
        """Check if input looks like a shell/environment command."""
        command = user_input.casefold().strip()
        return command.startswith(("conda ", "source ", "export ", "pip ", "python ", "cd "))

    @staticmethod
    def _looks_like_financial_request(user_input: str) -> bool:
        """Heuristic check if input looks like a financial request."""
        text = user_input.casefold()
        finance_terms = (
            "ticker", "symbol", "stock", "share", "price", "series",
            "timeseries", "time series", "security", "asset", "market", "data",
        )
        return any(term in text for term in finance_terms) or any(
            char.isdigit() for char in text
        )

    def process_user_response(self, user_input: str) -> list[CallbackEvent]:
        """Process a user response after a human-in-the-loop pause.

        Args:
            user_input: The user's response text.

        Returns:
            List of callback events generated during processing.
        """
        logger.info("user_response_received characters=%d", len(user_input))
        state = self.handler.handle_user_response(user_input)
        if state is None:
            return self._drain()
        explicit_delegate = self._try_delegate_from_explicit_agent_response(state, user_input)
        if explicit_delegate is not None:
            return explicit_delegate
        auto_progress = self._try_auto_progress_orchestrator(state, user_input)
        if auto_progress is not None:
            return auto_progress
        reporting_resume = self._try_resume_reporting_from_source_selection(state, user_input)
        if reporting_resume is not None:
            return reporting_resume
        gapfilling_resume = self._try_resume_gapfilling_from_method_selection(state, user_input)
        if gapfilling_resume is not None:
            return gapfilling_resume
        messages = state["messages"] + [{"role": "user", "content": user_input}]
        return self._run_agent(
            get_agent(state["agent"]),
            messages,
            state.get("iteration", 0),
        )

    def _try_resume_gapfilling_from_method_selection(
        self,
        state: dict[str, Any],
        user_input: str,
    ) -> list[CallbackEvent] | None:
        """Deterministically continue GapFilling after a method response.

        When resuming from a GapFilling pause, local models may ignore the user
        response and re-ask or emit a narrative answer without tool calls. If the
        user provided an explicit method, apply it immediately from context and
        continue to TimeSeriesConstructionAgent.
        """
        if state.get("agent") != "GapFillingAgent":
            return None
        checkpoint = state.get("checkpoint")
        if checkpoint not in (None, "", "gap_method_selection"):
            return None

        selected_method = self._extract_explicit_gap_method(user_input)
        if not selected_method:
            return None

        base_messages = list(state.get("messages", []))
        recovered_prices = self._recover_gapfilling_prices_from_context(
            base_messages,
            final_text=user_input,
        )
        if recovered_prices is None:
            return None

        apply_tool = get_tool("apply_gap_filling")
        if apply_tool is None:
            return None

        apply_args = {
            "prices": recovered_prices,
            "method": selected_method,
        }
        try:
            filled_result = apply_tool.invoke(apply_args)
        except Exception as error:
            self.handler.on_tool_error(error)
            logger.exception("gapfilling_resume_apply_failed")
            return [self._user_error("apply_gap_filling", str(error))]

        if not (isinstance(filled_result, dict) and filled_result.get("prices")):
            return None

        agent = get_agent("GapFillingAgent")
        if agent is None:
            return None

        iteration = int(state.get("iteration", 0) or 0)
        self.handler.add_trace_record(
            "tool_call",
            {
                "tool": "apply_gap_filling",
                "description": get_tool_description("apply_gap_filling"),
                "arguments": apply_args,
            },
            agent=agent.name,
            iteration=iteration,
        )
        self.handler.add_trace_record(
            "tool_result",
            {
                "tool": "apply_gap_filling",
                "description": get_tool_description("apply_gap_filling"),
                "result": filled_result,
            },
            agent=agent.name,
            iteration=iteration,
        )
        logger.info(
            "gapfilling_resume_auto_continue mode=deterministic method=%s symbol=%s",
            selected_method,
            filled_result.get("symbol"),
        )

        resumed_messages = base_messages + [
            {"role": "user", "content": user_input},
            {"role": "user", "content": f"Tool result: {json.dumps(filled_result, default=str)}"},
        ]
        continuation = self._continue_gapfilling_to_construction(
            response=f"Final Answer: Gap-filling applied with {selected_method}.",
            agent=agent,
            messages=resumed_messages,
            iteration=iteration,
            visited=set(),
        )
        return continuation

    def _try_resume_reporting_from_source_selection(
        self,
        state: dict[str, Any],
        user_input: str,
    ) -> list[CallbackEvent] | None:
        """Deterministically continue Reporting after user source selection.

        When resuming from the Reporting source-selection checkpoint, local
        models often replay stale turns (re-asking selection, self-delegating,
        or consuming an old queue entry). If the user explicitly selected a
        source, continue directly to the standard Reporting->GapFilling path.
        """
        if state.get("agent") != "ReportingAgent":
            return None
        if state.get("checkpoint") != "source_selection":
            return None

        selected_source = self._extract_explicit_source_selection(user_input)
        if not selected_source:
            return None

        reporting_agent = get_agent("ReportingAgent")
        if reporting_agent is None:
            return None

        base_messages = list(state.get("messages", []))
        resumed_messages = base_messages + [
            {"role": "user", "content": user_input},
            {"role": "user", "content": f"[SOURCE_SELECTION] {selected_source}"},
        ]
        iteration = int(state.get("iteration", 0) or 0)
        logger.info(
            "reporting_resume_auto_continue mode=deterministic source=%s reason=explicit_source_at_checkpoint",
            selected_source,
        )
        continuation = self._continue_reporting_to_gapfilling(
            response=f"Final Answer: User selected {selected_source} as the data source.",
            agent=reporting_agent,
            messages=resumed_messages,
            iteration=iteration,
            visited=set(),
        )
        return continuation

    def _try_delegate_from_explicit_agent_response(
        self,
        state: dict[str, Any],
        user_input: str,
    ) -> list[CallbackEvent] | None:
        """Delegate immediately when user explicitly names a registered target agent."""
        current_agent_name = str(state.get("agent", "")).strip()
        target = get_agent(user_input.strip())
        if target is None:
            return None
        if not current_agent_name or target.name == current_agent_name:
            return None

        user_messages = [
            message.get("content", "")
            for message in state.get("messages", [])
            if message.get("role") == "user"
        ]
        transfer_request = " ".join(text for text in user_messages if text).strip() or user_input.strip()

        logger.info(
            "explicit_agent_delegate from_agent=%s to_agent=%s reason=user_named_registered_agent",
            current_agent_name,
            target.name,
        )
        pre_events = [
            CallbackEvent(
                CallbackEventType.AGENT_COMPLETED,
                {"agent": current_agent_name, "result": {"delegated_to": target.name}},
                self.handler.session_id,
            ),
            CallbackEvent(
                CallbackEventType.DELEGATED,
                {
                    "from_agent": current_agent_name,
                    "to_agent": target.name,
                    "request": transfer_request,
                    "routing_mode": "deterministic",
                    "routing_reason": "bypassed re-ask loop: user explicitly selected a registered target agent",
                },
                self.handler.session_id,
            ),
        ]
        result = self._run_agent(
            target,
            [{"role": "user", "content": transfer_request}],
        )
        return pre_events + (result if self._is_event_list(result) else [])

    def _try_auto_progress_orchestrator(
        self,
        state: dict[str, Any],
        user_input: str,
    ) -> list[CallbackEvent] | None:
        """Advance from Orchestrator clarification when user already provided date range.

        This avoids repeated clarification loops when the user responds with valid
        ticker/date details in free-form language.
        """
        if state.get("agent") != "Orchestrator":
            return None

        combined_user_context = " ".join(
            msg.get("content", "")
            for msg in state.get("messages", [])
            if msg.get("role") == "user"
        )
        combined_user_context = f"{combined_user_context} {user_input}".strip()

        try:
            normalized_range = extract_date_range(combined_user_context)
        except ValueError as error:
            logger.info("orchestrator_auto_progress_skipped reason=invalid_date_range error=%s", error)
            return None

        if normalized_range is None:
            return None

        start_date, end_date = normalized_range
        request = f"{combined_user_context} (normalized date range: {start_date} to {end_date})"
        logger.info(
            "orchestrator_auto_progress_detected mode=deterministic start=%s end=%s reason=user_followup_contains_date_range",
            start_date,
            end_date,
        )

        target = get_agent("ReferenceDataAgent")
        if target is None:
            return [
                CallbackEvent(
                    CallbackEventType.ERROR,
                    {"message": "ReferenceDataAgent is not registered."},
                    self.handler.session_id,
                )
            ]

        pre_events = [
            CallbackEvent(
                CallbackEventType.AGENT_COMPLETED,
                {"agent": "Orchestrator", "result": {"delegated_to": target.name}},
                self.handler.session_id,
            ),
            CallbackEvent(
                CallbackEventType.DELEGATED,
                {
                    "from_agent": "Orchestrator",
                    "to_agent": target.name,
                    "request": request,
                    "routing_mode": "deterministic",
                    "routing_reason": "bypassed Orchestrator clarification: follow-up already includes date range",
                },
                self.handler.session_id,
            ),
        ]
        result = self._run_agent(
            target,
            [{"role": "user", "content": request}],
        )
        return pre_events + (result if self._is_event_list(result) else [])

    def _run_agent(
        self,
        agent: Agent | None,
        messages: list[dict[str, str]],
        start: int = 0,
        visited: set[str] | None = None,
    ) -> list[CallbackEvent]:
        """Run a ReACT agent loop for a given agent definition.

        Args:
            agent: The agent definition to run.
            messages: The conversation messages so far.
            start: Starting iteration count.
            visited: Set of already-visited agent names (for cycle detection).

        Returns:
            List of callback events generated during agent execution.
        """
        if agent is None:
            return [
                CallbackEvent(
                    CallbackEventType.ERROR,
                    {"message": "Orchestrator is not registered."},
                )
            ]

        visited = visited or set()
        if agent.name in visited:
            return [
                CallbackEvent(
                    CallbackEventType.ERROR,
                    {"message": f"Agent cycle detected at {agent.name}."},
                )
            ]
        visited.add(agent.name)

        self.handler.current_agent = agent.name
        # Inject run_id so tools can store/load time series via DataStore
        set_run_id(self.handler.session_id)
        prompt = self._prompt(agent)
        loop_detector = LoopDetector(max_repeats=3)
        unparseable_count = 0
        log_workflow_progress(agent.name, start, "started")
        logger.info("agent_started agent=%s iteration_start=%d", agent.name, start)

        for iteration in range(start, 8):
            log_workflow_progress(agent.name, iteration, "llm_call")
            logger.info("agent_iteration agent=%s iteration=%d", agent.name, iteration)
            self._debug_flow_event(
                component="processor",
                stage="llm_call",
                agent=agent.name,
                iteration=iteration,
            )

            # Log message size to track growing conversation history
            log_message_size(agent.name, iteration, messages)
            self.handler.add_trace_record(
                "llm_request",
                {
                    "system_prompt": prompt,
                    "messages": list(messages),
                },
                agent=agent.name,
                iteration=iteration,
            )

            # Get LLM response with timing
            try:
                with timer(agent.name, iteration, "LLM call"):
                    response = self.factory.chat(
                        LLMRequest(
                            system_prompt=prompt,
                            messages=messages,
                            callbacks=[self.handler],
                        )
                    )
            except Exception as error:
                self.handler.on_llm_error(error)
                log_workflow_progress(agent.name, iteration, "error", detail=str(error))
                return self._drain()

            # Record trace
            self.handler.add_to_trace(
                f"[{agent.name}:{iteration}] {response}"
            )
            self.handler.add_trace_record(
                "llm_response",
                {"content": response},
                agent=agent.name,
                iteration=iteration,
            )
            messages.append({"role": "assistant", "content": response})

            # Check for response loop (identical response as previous iteration)
            loop_detector.record_response(agent.name, iteration, response)

            if (
                agent.name == "ReportingAgent"
                and any(
                    _REPORTING_FINAL_SUMMARY_NOTE in msg.get("content", "")
                    for msg in messages
                    if msg.get("role") == "user"
                )
            ):
                summary_text = response.split("Final Answer:", 1)[1].strip() if "Final Answer:" in response else response.strip()
                self._log_continuation_decision(
                    agent.name,
                    iteration,
                    "completed",
                    "reporting_final_summary_terminal",
                )
                logger.info(
                    "agent_completed agent=%s iteration=%d reason=reporting_final_summary_terminal",
                    agent.name,
                    iteration,
                )
                self.handler.emit(
                    CallbackEvent(
                        CallbackEventType.AGENT_COMPLETED,
                        {
                            "agent": agent.name,
                            "result": {"final_answer": summary_text},
                        },
                    )
                )
                return self._drain()

            # Check for Final Answer, including models that emit "Action: Final Answer".
            if "Final Answer:" in response or re.search(r"Action:\s*Final\s+Answer\b", response, re.IGNORECASE):
                if "Final Answer:" not in response and re.search(r"Action:\s*Final\s+Answer\b", response, re.IGNORECASE):
                    response = re.sub(
                        r"Action:\s*Final\s+Answer",
                        "Final Answer:",
                        response,
                        count=1,
                        flags=re.IGNORECASE,
                    )
                log_workflow_progress(agent.name, iteration, "final_answer", detail="llm_returned_final_answer")
                if (
                    agent.name == "MarketDataAgent"
                    and self._looks_like_market_source_selection_final(response)
                    and not any(
                        _MARKETDATA_ALL_SOURCES_NOTE in message.get("content", "")
                        for message in messages
                        if message.get("role") == "user"
                    )
                ):
                    logger.warning(
                        "market_source_selection_final_bypassed mode=deterministic iteration=%d",
                        iteration,
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": _MARKETDATA_ALL_SOURCES_NOTE,
                        }
                    )
                    continue
                if (
                    agent.name == "ReferenceDataAgent"
                    and self._looks_like_placeholder_final(response)
                    and not any(
                        _REFERENCE_EXECUTE_NOTE in message.get("content", "")
                        for message in messages
                        if message.get("role") == "user"
                    )
                ):
                    logger.warning(
                        "reference_placeholder_final_bypassed mode=deterministic iteration=%d reason=final_answer_describes_instead_of_retrieving",
                        iteration,
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": _REFERENCE_EXECUTE_NOTE,
                        }
                    )
                    continue
                recovery = self._recover_orchestrator_delegation(
                    response, agent, messages, iteration, visited,
                )
                if recovery is not None:
                    return recovery
                # Auto-delegate from ReferenceDataAgent to MarketDataAgent when
                # the instrument has been resolved. This prevents the workflow
                # from returning to the user after ReferenceDataAgent completes
                # with a Final Answer instead of calling delegate_to_agent.
                continuation = self._continue_after_agent_completion(
                    response, agent, messages, iteration, visited,
                )
                if continuation is not None:
                    self._log_continuation_decision(
                        agent.name,
                        iteration,
                        "delegating",
                        "continued_after_final_answer",
                    )
                    return continuation
                self._log_continuation_decision(
                    agent.name,
                    iteration,
                    "completed",
                    "final_answer_without_continuation",
                )
                logger.info(
                    "agent_completed agent=%s iteration=%d", agent.name, iteration
                )
                self.handler.emit(
                    CallbackEvent(
                        CallbackEventType.AGENT_COMPLETED,
                        {
                            "agent": agent.name,
                            "result": {
                                "final_answer": response.split("Final Answer:", 1)[1].strip()
                            },
                        },
                    )
                )
                return self._drain()

            # Parse tool calls
            calls = self._parse_calls(response)
            logger.info(
                "agent_actions agent=%s iteration=%d count=%d",
                agent.name, iteration, len(calls),
            )
            self._debug_flow_event(
                component="processor",
                stage="parsed_actions",
                agent=agent.name,
                iteration=iteration,
                detail=f"count={len(calls)}",
            )
            if calls:
                logger.info(
                    "agent_actions_detail agent=%s iteration=%d tools=%s",
                    agent.name,
                    iteration,
                    ",".join(call["name"] for call in calls),
                )
            self.handler.add_trace_record(
                "parsed_actions",
                {
                    "count": len(calls),
                    "calls": calls,
                },
                agent=agent.name,
                iteration=iteration,
            )

            if not calls:
                unparseable_count += 1
                if agent.name == "DataQualityAgent":
                    deterministic_quality = self._continue_quality_to_reporting(
                        response,
                        agent,
                        messages,
                        iteration,
                        visited,
                    )
                    if deterministic_quality is not None:
                        self._debug_flow_event(
                            component="processor",
                            stage="deterministic_recovery",
                            agent=agent.name,
                            iteration=iteration,
                            detail="quality_recovered_after_unparseable_actions",
                        )
                        self._log_continuation_decision(
                            agent.name,
                            iteration,
                            "delegating",
                            "quality_recovered_after_unparseable_actions",
                        )
                        return deterministic_quality
                if agent.name == "GapFillingAgent":
                    deterministic_gap = self._continue_gapfilling_to_construction(
                        response,
                        agent,
                        messages,
                        iteration,
                        visited,
                    )
                    if deterministic_gap is not None:
                        self._debug_flow_event(
                            component="processor",
                            stage="deterministic_recovery",
                            agent=agent.name,
                            iteration=iteration,
                            detail="gapfilling_recovered_after_unparseable_actions",
                        )
                        self._log_continuation_decision(
                            agent.name,
                            iteration,
                            "pausing",
                            "gapfilling_recovered_after_unparseable_actions",
                        )
                        return deterministic_gap
                if agent.name == "TimeSeriesConstructionAgent":
                    deterministic_construction = self._continue_construction_to_reporting(
                        response,
                        agent,
                        messages,
                        iteration,
                        visited,
                    )
                    if deterministic_construction is not None:
                        self._debug_flow_event(
                            component="processor",
                            stage="deterministic_recovery",
                            agent=agent.name,
                            iteration=iteration,
                            detail="construction_recovered_after_unparseable_actions",
                        )
                        self._log_continuation_decision(
                            agent.name,
                            iteration,
                            "delegating",
                            "construction_recovered_after_unparseable_actions",
                        )
                        return deterministic_construction
                if (
                    agent.name == "ReportingAgent"
                    and any(
                        _REPORTING_FINAL_SUMMARY_NOTE in msg.get("content", "")
                        for msg in messages
                        if msg.get("role") == "user"
                    )
                ):
                    summary_text = response.split("Final Answer:", 1)[1].strip() if "Final Answer:" in response else response.strip()
                    self._log_continuation_decision(
                        agent.name,
                        iteration,
                        "completed",
                        "reporting_final_summary_terminal",
                    )
                    logger.info(
                        "agent_completed agent=%s iteration=%d reason=reporting_final_summary_terminal",
                        agent.name,
                        iteration,
                    )
                    self.handler.emit(
                        CallbackEvent(
                            CallbackEventType.AGENT_COMPLETED,
                            {
                                "agent": agent.name,
                                "result": {"final_answer": summary_text},
                            },
                        )
                    )
                    return self._drain()
                recovery = self._recover_orchestrator_delegation(
                    response, agent, messages, iteration, visited,
                )
                if recovery is not None:
                    return recovery
                if unparseable_count >= 3:
                    format_override = (
                        "[SYSTEM_NOTE] FORMAT_OVERRIDE: Output exactly this format without any additional text:\n"
                        "Action: get_instrument_details\n"
                        "Action Input: {\"query\": \"AAPL\"}"
                    )
                    messages.append({"role": "user", "content": format_override})
                    logger.warning(
                        "format_override_injected agent=%s iteration=%d reason=unparseable_count_exceeded",
                        agent.name,
                        iteration,
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": "Use the required Action and Action Input JSON format.",
                        }
                    )
                self._debug_flow_event(
                    component="processor",
                    stage="re_prompt_for_action_format",
                    agent=agent.name,
                    iteration=iteration,
                )
                continue
                self.handler.add_trace_record(
                    "final_answer",
                    {"content": response.split("Final Answer:", 1)[1].strip()},
                    agent=agent.name,
                    iteration=iteration,
                )

            # Execute each tool call
            for call in calls:
                if agent.name == "GapFillingAgent":
                    last_user_message = next(
                        (msg.get("content", "") for msg in reversed(messages) if msg.get("role") == "user"),
                        "",
                    )
                    selected_method = self._extract_explicit_gap_method(last_user_message)
                    if (
                        selected_method
                        and call["name"] in {
                            "check_data_quality",
                            "get_instrument_details",
                            "available_data_sources",
                            "extract_requested_date_range",
                            "normalize_requested_dates",
                            "request_human_input",
                            "recommend_gap_methods",
                        }
                    ):
                        logger.warning(
                            "off_contract_tool_recovered mode=deterministic agent=%s iteration=%d tool=%s method=%s",
                            agent.name,
                            iteration,
                            call["name"],
                            selected_method,
                        )
                        gap_continuation = self._continue_gapfilling_to_construction(
                            response="",
                            agent=agent,
                            messages=messages,
                            iteration=iteration,
                            visited=visited,
                        )
                        if gap_continuation is not None:
                            self._debug_flow_event(
                                component="processor",
                                stage="deterministic_recovery",
                                agent=agent.name,
                                iteration=iteration,
                                detail=f"gapfilling_recovered_after_off_contract_{call['name']}",
                            )
                            self._log_continuation_decision(
                                agent.name,
                                iteration,
                                "delegating",
                                f"gapfilling_recovered_after_off_contract_{call['name']}",
                            )
                            return gap_continuation

                if agent.name == "DataQualityAgent" and call["name"] in {
                    "get_instrument_details",
                    "available_data_sources",
                    "extract_requested_date_range",
                    "normalize_requested_dates",
                }:
                    logger.warning(
                        "off_contract_tool_recovered mode=deterministic agent=%s iteration=%d tool=%s",
                        agent.name,
                        iteration,
                        call["name"],
                    )
                    quality_continuation = self._continue_quality_to_reporting(
                        response="",
                        agent=agent,
                        messages=messages,
                        iteration=iteration,
                        visited=visited,
                    )
                    if quality_continuation is not None:
                        self._debug_flow_event(
                            component="processor",
                            stage="deterministic_recovery",
                            agent=agent.name,
                            iteration=iteration,
                            detail=f"quality_recovered_after_off_contract_{call['name']}",
                        )
                        self._log_continuation_decision(
                            agent.name,
                            iteration,
                            "delegating",
                            f"quality_recovered_after_off_contract_{call['name']}",
                        )
                        return quality_continuation

                log_workflow_progress(
                    agent.name,
                    iteration,
                    "tool_call",
                    detail=call["name"],
                )
                result = self._execute(
                    call, agent, messages, prompt, iteration, visited,
                )
                if result is None:
                    return self._drain()
                if self._is_event_list(result):
                    return result
                log_workflow_progress(
                    agent.name,
                    iteration,
                    "tool_result",
                    detail=call["name"],
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"Tool result: {json.dumps(result, default=str)}",
                    }
                )

                if agent.name == "MarketDataAgent" and call["name"] in {"available_data_sources", "historical_prices"}:
                    market_continuation = self._continue_market_to_quality(
                        response="",
                        agent=agent,
                        messages=messages,
                        iteration=iteration,
                        visited=visited,
                    )
                    if market_continuation is not None:
                        self._debug_flow_event(
                            component="processor",
                            stage="deterministic_recovery",
                            agent=agent.name,
                            iteration=iteration,
                            detail=f"market_auto_continued_after_{call['name']}",
                        )
                        self._log_continuation_decision(
                            agent.name,
                            iteration,
                            "delegating",
                            f"market_auto_continued_after_{call['name']}",
                        )
                        return market_continuation

                if agent.name == "DataQualityAgent" and call["name"] == "check_data_quality":
                    quality_continuation = self._continue_quality_to_reporting(
                        response="",
                        agent=agent,
                        messages=messages,
                        iteration=iteration,
                        visited=visited,
                    )
                    if quality_continuation is not None:
                        self._debug_flow_event(
                            component="processor",
                            stage="deterministic_recovery",
                            agent=agent.name,
                            iteration=iteration,
                            detail="quality_auto_continued_after_check_data_quality",
                        )
                        self._log_continuation_decision(
                            agent.name,
                            iteration,
                            "delegating",
                            "quality_auto_continued_after_check_data_quality",
                        )
                        return quality_continuation

        # Iteration limit reached
        self.handler.emit(
            CallbackEvent(
                CallbackEventType.ERROR,
                {"message": f"{agent.name} reached its iteration limit."},
            )
        )
        return self._drain()

    def _recover_orchestrator_delegation(
        self,
        response: str,
        agent: Agent,
        messages: list[dict[str, str]],
        iteration: int,
        visited: set[str],
    ) -> list[CallbackEvent] | None:
        """Recover when an LLM states delegation instead of calling the tool.

        Catches three patterns:
        1. Explicit mention of a registered agent name (e.g. "delegate to ReferenceDataAgent")
        2. Generic delegation language (e.g. "delegate to agent")
        3. "I will proceed" / "I will now" / "proceeding" — the Orchestrator
           describing intent without calling any tool. In this case, delegate
           to ReferenceDataAgent with the original user request.
        """
        target = next(
            (
                candidate
                for name in (
                    "ReferenceDataAgent",
                    "MarketDataAgent",
                    "DataQualityAgent",
                    "GapFillingAgent",
                    "TimeSeriesConstructionAgent",
                    "ReportingAgent",
                )
                for candidate in [get_agent(name)]
                if candidate is not None
                and re.search(
                    r"\b" + r"[ _-]?".join(re.findall(r"[A-Z][a-z]*|[A-Z]+(?=[A-Z]|$)", name)) + r"\b",
                    response,
                    re.IGNORECASE,
                )
            ),
            None,
        )
        if (
            agent.name == "Orchestrator"
            and target is None
            and re.search(r"\bdelegate[_ -]?to[_ -]?agent\b", response, re.IGNORECASE)
        ):
            target = get_agent("ReferenceDataAgent")
        # Catch "I will proceed" / "I will now" / "proceeding" patterns where
        # the Orchestrator describes intent without calling any tool. This is
        # common with deepseek-v2:16b which sometimes summarises instead of
        # executing. Delegate to ReferenceDataAgent with the original request.
        if (
            agent.name == "Orchestrator"
            and target is None
            and re.search(
                r"\b(?:i will now|i will proceed|proceeding|i shall|"
                r"i will begin|starting the|beginning the)\b",
                response,
                re.IGNORECASE,
            )
        ):
            target = get_agent("ReferenceDataAgent")
            logger.warning(
                "orchestrator_proceed_recovered mode=heuristic target=%s iteration=%d "
                "reason=llm_described_intent_without_tool_call",
                target.name, iteration,
            )
        if target is None:
            return None
        logger.warning(
            "orchestrator_delegation_recovered mode=heuristic target=%s iteration=%d reason=llm_mentioned_delegation_without_tool_call",
            target.name, iteration,
        )
        original_request = next(
            (msg["content"] for msg in messages if msg.get("role") == "user"),
            "",
        )
        return self._run_agent(
            target,
            [{"role": "user", "content": original_request}],
            visited=visited.copy(),
        )

    def _continue_after_agent_completion(
        self,
        response: str,
        agent: Agent,
        messages: list[dict[str, str]],
        iteration: int,
        visited: set[str],
    ) -> list[CallbackEvent] | None:
        """Auto-delegate to the next agent in the workflow chain after agent completion.

        When an agent completes with a Final Answer instead of calling delegate_to_agent,
        this method automatically delegates to the next appropriate agent to continue
        the workflow without returning to the user for input.

        Supported continuations:
        - ReferenceDataAgent → MarketDataAgent (when instrument is resolved)
        - MarketDataAgent → DataQualityAgent (when market data has been loaded)
        - DataQualityAgent → ReportingAgent (when quality metrics are computed)
        - ReportingAgent → GapFillingAgent (when user has selected a source)
        - GapFillingAgent → TimeSeriesConstructionAgent (when gap-filling is applied)
        - TimeSeriesConstructionAgent → ReportingAgent (final summary)

        Args:
            response: The LLM response containing the Final Answer.
            agent: The current agent definition.
            messages: The conversation messages.
            iteration: Current iteration number.
            visited: Set of visited agent names.

        Returns:
            List of CallbackEvents if continuation was triggered, None otherwise.
        """
        # --- ReferenceDataAgent → MarketDataAgent ---
        if agent.name == "ReferenceDataAgent":
            return self._continue_reference_to_market(response, agent, messages, iteration, visited)

        # --- MarketDataAgent → DataQualityAgent ---
        if agent.name == "MarketDataAgent":
            return self._continue_market_to_quality(response, agent, messages, iteration, visited)

        # --- DataQualityAgent → ReportingAgent ---
        if agent.name == "DataQualityAgent":
            return self._continue_quality_to_reporting(response, agent, messages, iteration, visited)

        # --- ReportingAgent → GapFillingAgent ---
        if agent.name == "ReportingAgent":
            return self._continue_reporting_to_gapfilling(response, agent, messages, iteration, visited)

        # --- GapFillingAgent → TimeSeriesConstructionAgent ---
        if agent.name == "GapFillingAgent":
            return self._continue_gapfilling_to_construction(response, agent, messages, iteration, visited)

        # --- TimeSeriesConstructionAgent → ReportingAgent ---
        if agent.name == "TimeSeriesConstructionAgent":
            return self._continue_construction_to_reporting(response, agent, messages, iteration, visited)

        self._log_continuation_decision(
            agent.name,
            iteration,
            "completed",
            "no_continuation_rule_for_agent",
        )
        return None

    def _continue_reference_to_market(
        self,
        response: str,
        agent: Agent,
        messages: list[dict[str, str]],
        iteration: int,
        visited: set[str],
    ) -> list[CallbackEvent] | None:
        """Auto-delegate from ReferenceDataAgent to MarketDataAgent after instrument resolution."""
        # Look for a resolved symbol in the tool results in messages
        resolved_symbol = None
        for message in messages:
            if message.get("role") == "user" and "Tool result:" in message.get("content", ""):
                try:
                    tool_result = json.loads(message["content"].replace("Tool result: ", "", 1))
                    if isinstance(tool_result, dict) and tool_result.get("found") and tool_result.get("symbol"):
                        resolved_symbol = tool_result["symbol"]
                        break
                except (json.JSONDecodeError, KeyError):
                    continue

        original_request = self._extract_original_request_from_messages(messages)

        # Fallback: some models emit a terminal "Final Answer" without calling tools.
        # Recover symbol from the final answer or original request to continue the flow.
        if not resolved_symbol:
            final_text = response.split("Final Answer:", 1)[1].strip() if "Final Answer:" in response else response
            recovered = self._extract_symbol_candidate_from_text(final_text)
            if recovered:
                resolved_symbol = recovered
                logger.info(
                    "reference_symbol_recovered source=final_answer symbol=%s iteration=%d",
                    resolved_symbol,
                    iteration,
                )

        if not resolved_symbol:
            inferred_query = self._extract_instrument_query(original_request)
            if inferred_query:
                try:
                    inferred_result = get_instrument_details(query=inferred_query)
                    if inferred_result.get("found") and inferred_result.get("symbol"):
                        resolved_symbol = str(inferred_result["symbol"])
                        logger.info(
                            "reference_symbol_recovered source=deterministic_tool symbol=%s query=%s iteration=%d",
                            resolved_symbol,
                            inferred_query,
                            iteration,
                        )
                except Exception as error:
                    logger.debug(
                        "reference_symbol_recovery_tool_failed query=%s error=%s",
                        inferred_query,
                        error,
                    )

        if not resolved_symbol:
            recovered = self._extract_symbol_candidate_from_text(original_request)
            if recovered:
                resolved_symbol = recovered
                logger.info(
                    "reference_symbol_recovered source=original_request symbol=%s iteration=%d",
                    resolved_symbol,
                    iteration,
                )

        if not resolved_symbol:
            self._log_continuation_decision(
                agent.name,
                iteration,
                "completed",
                "reference_to_market_skipped_missing_symbol",
            )
            return None

        market_agent = get_agent("MarketDataAgent")
        if market_agent is None:
            return None

        transfer_request = (
            f"Retrieve historical prices for {resolved_symbol}. "
            f"Original request: {original_request}"
        )

        logger.info(
            "reference_auto_continue mode=deterministic to_agent=%s symbol=%s reason=reference_agent_completed_with_resolved_instrument",
            market_agent.name,
            resolved_symbol,
        )
        self._log_continuation_decision(
            agent.name,
            iteration,
            "delegating",
            f"reference_to_market symbol={resolved_symbol}",
        )

        continuation_events = [
            CallbackEvent(
                CallbackEventType.AGENT_COMPLETED,
                {"agent": agent.name, "result": {"delegated_to": market_agent.name}},
                self.handler.session_id,
            ),
            CallbackEvent(
                CallbackEventType.DELEGATED,
                {
                    "from_agent": agent.name,
                    "to_agent": market_agent.name,
                    "request": transfer_request,
                    "routing_mode": "deterministic",
                    "routing_reason": "bypassed user prompt: ReferenceDataAgent completed with resolved instrument, auto-delegating to MarketDataAgent",
                },
                self.handler.session_id,
            ),
        ]

        target_result = self._run_agent(
            market_agent,
            [{"role": "user", "content": transfer_request}],
            visited=visited.copy(),
        )

        return continuation_events + (target_result if self._is_event_list(target_result) else [])

    def _continue_market_to_quality(
        self,
        response: str,
        agent: Agent,
        messages: list[dict[str, str]],
        iteration: int,
        visited: set[str],
    ) -> list[CallbackEvent] | None:
        """Auto-delegate from MarketDataAgent to DataQualityAgent after data loading.

        Checks if market data was successfully loaded from at least one source
        by looking for historical_prices tool results in the conversation.
        """
        original_request = self._extract_original_request_from_messages(messages)
        final_text = response.split("Final Answer:", 1)[1].strip() if "Final Answer:" in response else response

        loaded_sources, unavailable_sources, resolved_symbol = self._extract_market_source_context_from_messages(messages)
        if not unavailable_sources:
            unavailable_sources = self._extract_unavailable_market_sources_from_messages(messages)

        if not resolved_symbol:
            resolved_symbol = self._extract_symbol_candidate_from_text(final_text)
        if not resolved_symbol:
            resolved_symbol = self._extract_symbol_candidate_from_text(original_request)

        candidate_sources = list(
            dict.fromkeys(
                self._extract_available_sources_from_messages(messages)
                + self._extract_sources_from_text(final_text)
                + self._extract_sources_from_text(original_request)
                + list(SOURCES)
            )
        )

        start_date = None
        end_date = None
        for text in (original_request, final_text, " ".join(message.get("content", "") for message in messages if message.get("role") == "user")):
            try:
                extracted = extract_date_range(text)
            except ValueError:
                extracted = None
            if extracted is not None:
                start_date, end_date = extracted
                break

        if resolved_symbol and start_date and end_date:
            self._deterministically_load_market_sources(
                agent,
                messages,
                iteration,
                resolved_symbol,
                start_date,
                end_date,
                candidate_sources,
                loaded_sources,
            )
            loaded_sources, unavailable_sources, resolved_symbol_from_context = self._extract_market_source_context_from_messages(messages)
            if not resolved_symbol and resolved_symbol_from_context:
                resolved_symbol = resolved_symbol_from_context
        if not loaded_sources and resolved_symbol:
            loaded_sources = candidate_sources
            logger.warning(
                "market_sources_recovered mode=deterministic symbol=%s sources=%s reason=source_list_known_without_tool_results",
                resolved_symbol,
                loaded_sources,
            )

        if not loaded_sources:
            if unavailable_sources:
                unavailable_summary = "; ".join(
                    f"{item['source']}: {item['reason']}"
                    for item in unavailable_sources
                    if item.get("source")
                )
                logger.warning(
                    "market_all_sources_unavailable symbol=%s details=%s",
                    resolved_symbol,
                    unavailable_summary,
                )
                return [
                    CallbackEvent(
                        CallbackEventType.ERROR,
                        {
                            "agent": agent.name,
                            "message": (
                                "Market data is not available for the requested range/source set. "
                                f"Unavailable sources: {unavailable_summary}."
                            ),
                            "recoverable": True,
                            "user_action": (
                                "Try another date range, ticker, or data source and run the workflow again."
                            ),
                            "unavailable_sources": unavailable_sources,
                        },
                        self.handler.session_id,
                    )
                ]
            self._log_continuation_decision(
                agent.name,
                iteration,
                "completed",
                "market_to_quality_skipped_missing_sources",
            )
            return None

        quality_agent = get_agent("DataQualityAgent")
        if quality_agent is None:
            return None

        transfer_request = (
            f"Check data quality for {resolved_symbol or 'the instrument'} "
            f"from sources: {', '.join(loaded_sources)}. "
            f"Unavailable sources: {json.dumps(unavailable_sources)}. "
            f"Original request: {original_request}"
        )

        logger.info(
            "market_auto_continue mode=deterministic to_agent=%s symbol=%s sources=%s reason=market_agent_completed_with_loaded_data",
            quality_agent.name,
            resolved_symbol,
            loaded_sources,
        )
        self._log_continuation_decision(
            agent.name,
            iteration,
            "delegating",
            f"market_to_quality symbol={resolved_symbol or 'unknown'} sources={','.join(loaded_sources)}",
        )

        quality_sources: list[dict[str, Any]] = []
        historical_payloads: dict[str, dict[str, Any]] = {}
        for message in messages:
            if message.get("role") != "user" or "Tool result:" not in message.get("content", ""):
                continue
            try:
                payload = json.loads(message["content"].replace("Tool result: ", "", 1))
            except json.JSONDecodeError:
                continue
            if (
                isinstance(payload, dict)
                and payload.get("source")
                and isinstance(payload.get("dates"), list)
                and isinstance(payload.get("prices"), list)
            ):
                source_key = str(payload.get("source", "")).strip().casefold()
                if source_key and source_key not in historical_payloads:
                    historical_payloads[source_key] = payload

        quality_tool = get_tool("check_data_quality")
        if quality_tool is not None:
            for source in loaded_sources:
                source_key = str(source).strip().casefold()
                if not source_key:
                    continue
                tool_args: dict[str, Any]
                payload = historical_payloads.get(source_key)
                if payload is not None:
                    tool_args = {"data": payload}
                else:
                    tool_args = {"source": source_key, "symbol": resolved_symbol or "UNKNOWN"}
                    if start_date:
                        tool_args["start_date"] = start_date
                    if end_date:
                        tool_args["end_date"] = end_date
                try:
                    quality_result = quality_tool.invoke(tool_args)
                except Exception:
                    continue
                if isinstance(quality_result, dict) and quality_result.get("source"):
                    quality_sources.append(quality_result)

        data_quality_report = self._build_data_quality_report(
            quality_sources or [
                {"source": source, "symbol": resolved_symbol or "UNKNOWN", "note": "fallback_context_only"}
                for source in loaded_sources
            ],
            unavailable_sources=unavailable_sources,
        )

        continuation_events = [
            CallbackEvent(
                CallbackEventType.AGENT_COMPLETED,
                {
                    "agent": agent.name,
                    "result": {
                        "delegated_to": quality_agent.name,
                        "loaded_sources": loaded_sources,
                        "unavailable_sources": unavailable_sources,
                        "data_quality_report": data_quality_report,
                    },
                },
                self.handler.session_id,
            ),
            CallbackEvent(
                CallbackEventType.DELEGATED,
                {
                    "from_agent": agent.name,
                    "to_agent": quality_agent.name,
                    "request": transfer_request,
                    "routing_mode": "deterministic",
                    "routing_reason": "bypassed user prompt: MarketDataAgent completed with loaded data, auto-delegating to DataQualityAgent",
                },
                self.handler.session_id,
            ),
        ]

        target_result = self._run_agent(
            quality_agent,
            [{"role": "user", "content": transfer_request}],
            visited=visited.copy(),
        )

        return continuation_events + (target_result if self._is_event_list(target_result) else [])

    def _continue_quality_to_reporting(
        self,
        response: str,
        agent: Agent,
        messages: list[dict[str, str]],
        iteration: int,
        visited: set[str],
    ) -> list[CallbackEvent] | None:
        """Auto-delegate from DataQualityAgent to ReportingAgent after quality checks.

        Ensures every discovered market source is represented in the quality report,
        even when the model only checked the first source it noticed.
        """
        quality_sources: list[dict[str, Any]] = []
        unavailable_sources = self._extract_unavailable_market_sources_from_messages(messages)
        historical_payloads: dict[str, dict[str, Any]] = {}
        for message in messages:
            if message.get("role") == "user" and "Tool result:" in message.get("content", ""):
                try:
                    tool_result = json.loads(message["content"].replace("Tool result: ", "", 1))
                    if not isinstance(tool_result, dict):
                        continue
                    if tool_result.get("source") and "completeness_pct" in tool_result:
                        quality_sources.append(tool_result)
                    if (
                        tool_result.get("source")
                        and isinstance(tool_result.get("dates"), list)
                        and isinstance(tool_result.get("prices"), list)
                    ):
                        source_key = str(tool_result.get("source", "")).strip().casefold()
                        if source_key and source_key not in historical_payloads:
                            historical_payloads[source_key] = tool_result
                except (json.JSONDecodeError, KeyError):
                    continue

        original_request = self._extract_original_request_from_messages(messages)
        symbol_guess = self._extract_symbol_candidate_from_text(response)
        if not symbol_guess:
            symbol_guess = self._extract_symbol_candidate_from_text(original_request)

        candidate_sources = list(
            dict.fromkeys(
                list(historical_payloads.keys())
                + self._extract_available_sources_from_messages(messages)
                + self._extract_sources_from_text(response)
                + self._extract_sources_from_text(original_request)
                + list(SOURCES)
            )
        )

        known_quality_sources = {
            str(item.get("source", "")).strip().casefold()
            for item in quality_sources
            if item.get("source")
        }
        missing_sources = [source for source in candidate_sources if source not in known_quality_sources]

        if missing_sources:
            start_date = None
            end_date = None
            try:
                extracted = extract_date_range(original_request)
            except ValueError:
                extracted = None
            if extracted is not None:
                start_date, end_date = extracted

            quality_tool = get_tool("check_data_quality")
            if quality_tool is not None:
                for source in missing_sources:
                    source_key = str(source).strip().casefold()
                    if not source_key:
                        continue
                    tool_args: dict[str, Any]
                    payload = historical_payloads.get(source_key)
                    if payload is not None:
                        tool_args = {"data": payload}
                    else:
                        tool_args = {
                            "source": source_key,
                            "symbol": symbol_guess or "UNKNOWN",
                        }
                        if start_date:
                            tool_args["start_date"] = start_date
                        if end_date:
                            tool_args["end_date"] = end_date
                    try:
                        quality_result = quality_tool.invoke(tool_args)
                    except Exception:
                        continue
                    if isinstance(quality_result, dict) and quality_result.get("source"):
                        quality_sources.append(quality_result)
                        known_quality_sources.add(source_key)
                        self.handler.add_trace_record(
                            "tool_call",
                            {
                                "tool": "check_data_quality",
                                "description": get_tool_description("check_data_quality"),
                                "arguments": tool_args,
                            },
                            agent=agent.name,
                            iteration=iteration,
                        )
                        self.handler.add_trace_record(
                            "tool_result",
                            {
                                "tool": "check_data_quality",
                                "description": get_tool_description("check_data_quality"),
                                "result": quality_result,
                            },
                            agent=agent.name,
                            iteration=iteration,
                        )

        if not quality_sources:
            quality_sources = [
                {
                    "source": source,
                    "symbol": symbol_guess or "UNKNOWN",
                    "note": "fallback_context_only",
                }
                for source in candidate_sources
            ]
            logger.warning(
                "quality_metrics_recovered mode=deterministic symbol=%s reason=final_answer_without_tool_results",
                symbol_guess,
            )

        reporting_agent = get_agent("ReportingAgent")
        if reporting_agent is None:
            return None

        # Build transfer request with quality context
        user_texts = [
            msg.get("content", "")
            for msg in messages
            if msg.get("role") == "user"
        ]
        all_user_text = " ".join(user_texts).strip()
        original_match = re.search(r"original_request=(.+)", all_user_text)
        original_request = original_match.group(1).strip() if original_match else all_user_text

        transfer_request = (
            f"Present quality report and ask the user to select a source. "
            f"Quality data: {json.dumps(quality_sources)}. "
            f"Unavailable sources: {json.dumps(unavailable_sources)}. "
            f"Original request: {original_request}"
        )

        logger.info(
            "quality_auto_continue mode=deterministic to_agent=%s sources=%d reason=quality_agent_completed_with_metrics",
            reporting_agent.name,
            len(quality_sources),
        )
        self._log_continuation_decision(
            agent.name,
            iteration,
            "delegating",
            f"quality_to_reporting sources={len(quality_sources)}",
        )

        quality_report = self._build_data_quality_report(
            quality_sources,
            unavailable_sources=unavailable_sources,
        )

        continuation_events = [
            CallbackEvent(
                CallbackEventType.AGENT_COMPLETED,
                {
                    "agent": agent.name,
                    "result": {
                        "delegated_to": reporting_agent.name,
                        "data_quality_report": quality_report,
                    },
                },
                self.handler.session_id,
            ),
            CallbackEvent(
                CallbackEventType.DELEGATED,
                {
                    "from_agent": agent.name,
                    "to_agent": reporting_agent.name,
                    "request": transfer_request,
                    "routing_mode": "deterministic",
                    "routing_reason": "bypassed user prompt: DataQualityAgent completed with quality metrics, auto-delegating to ReportingAgent",
                },
                self.handler.session_id,
            ),
        ]
        prompt, options = self._build_reporting_selection_prompt(quality_sources)
        awaiting_event = CallbackEvent(
            CallbackEventType.AWAITING_USER_INPUT,
            {
                "prompt": prompt,
                "agent": reporting_agent.name,
                "options": options,
                "context": {
                    "quality_rows": quality_sources,
                    "checkpoint": "source_selection",
                    "data_quality_report": quality_report,
                },
            },
            self.handler.session_id,
        )
        # Keep runtime pause state in sync when emitting the event directly.
        self.handler.current_agent = reporting_agent.name
        self.handler.waiting_for_input = True
        self.handler.paused_state = {
            "agent": reporting_agent.name,
            "messages": messages.copy(),
            "iteration": iteration + 1,
            "checkpoint": "source_selection",
        }
        logger.info(
            "quality_report_pause mode=deterministic agent=%s sources=%d reason=source_selection_prompt_emitted",
            reporting_agent.name,
            len(quality_sources),
        )

        return continuation_events + [awaiting_event]

    def _continue_reporting_to_gapfilling(
        self,
        response: str,
        agent: Agent,
        messages: list[dict[str, str]],
        iteration: int,
        visited: set[str],
    ) -> list[CallbackEvent] | None:
        """Auto-delegate from ReportingAgent to GapFillingAgent after user selects a source.

        Checks if the user has provided a source selection in the conversation
        (either via the paused state user_response or in the final answer).
        """
        if any(
            _REPORTING_FINAL_SUMMARY_NOTE in msg.get("content", "")
            for msg in messages
            if msg.get("role") == "user"
        ):
            self._log_continuation_decision(
                agent.name,
                iteration,
                "completed",
                "reporting_final_summary_mode",
            )
            return None

        # Only continue automatically after an explicit source-selection marker
        # that is injected by process_user_response checkpoint handling.
        selected_source = None
        resolved_symbol = None
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if not content.startswith("[SOURCE_SELECTION]"):
                continue
            selected_source = self._extract_explicit_source_selection(content)
            if selected_source:
                break

        if not selected_source:
            self._log_continuation_decision(
                agent.name,
                iteration,
                "pausing",
                "reporting_requires_user_source_selection",
            )
            quality_rows = self._extract_quality_rows_from_messages(messages)
            prompt, options = self._build_reporting_selection_prompt(quality_rows)
            self.handler.request_human_input(
                prompt,
                options=options,
                context={"quality_rows": quality_rows, "checkpoint": "source_selection"},
            )
            self.handler.paused_state = {
                "agent": agent.name,
                "messages": messages.copy(),
                "iteration": iteration + 1,
            }
            logger.info(
                "reporting_forced_pause mode=deterministic iteration=%d options=%s reason=missing_explicit_source_selection",
                iteration,
                options,
            )
            return self._drain()

        # Extract symbol from tool results in messages
        for message in messages:
            if message.get("role") == "user" and "Tool result:" in message.get("content", ""):
                try:
                    tool_result = json.loads(message["content"].replace("Tool result: ", "", 1))
                    if isinstance(tool_result, dict) and tool_result.get("symbol"):
                        resolved_symbol = tool_result["symbol"]
                        break
                except (json.JSONDecodeError, KeyError):
                    continue

        gap_agent = get_agent("GapFillingAgent")
        if gap_agent is None:
            return None

        user_texts = [
            msg.get("content", "")
            for msg in messages
            if msg.get("role") == "user"
        ]
        all_user_text = " ".join(user_texts).strip()
        original_match = re.search(r"original_request=(.+)", all_user_text)
        original_request = original_match.group(1).strip() if original_match else all_user_text

        transfer_request = (
            f"Apply gap-filling to {resolved_symbol or 'the instrument'} "
            f"using source {selected_source}. "
            f"Original request: {original_request}"
        )

        logger.info(
            "reporting_auto_continue mode=deterministic to_agent=%s source=%s symbol=%s reason=reporting_agent_completed_with_source_selection",
            gap_agent.name,
            selected_source,
            resolved_symbol,
        )
        self._log_continuation_decision(
            agent.name,
            iteration,
            "delegating",
            f"reporting_to_gapfilling source={selected_source}",
        )

        continuation_events = [
            CallbackEvent(
                CallbackEventType.AGENT_COMPLETED,
                {"agent": agent.name, "result": {"delegated_to": gap_agent.name}},
                self.handler.session_id,
            ),
            CallbackEvent(
                CallbackEventType.DELEGATED,
                {
                    "from_agent": agent.name,
                    "to_agent": gap_agent.name,
                    "request": transfer_request,
                    "routing_mode": "deterministic",
                    "routing_reason": "bypassed user prompt: ReportingAgent completed with source selection, auto-delegating to GapFillingAgent",
                },
                self.handler.session_id,
            ),
        ]

        gap_method_options = self._extract_gap_method_options_from_messages(messages)
        gap_method_prompt = self._build_gap_method_prompt(gap_method_options)
        awaiting_event = CallbackEvent(
            CallbackEventType.AWAITING_USER_INPUT,
            {
                "prompt": gap_method_prompt,
                "agent": gap_agent.name,
                "options": gap_method_options,
                "context": {
                    "checkpoint": "gap_method_selection",
                    "selected_source": selected_source,
                    "symbol": resolved_symbol,
                },
            },
            self.handler.session_id,
        )
        self.handler.current_agent = gap_agent.name
        self.handler.waiting_for_input = True
        self.handler.paused_state = {
            "agent": gap_agent.name,
            "messages": messages.copy() + [{"role": "user", "content": f"[SOURCE_SELECTION] {selected_source}"}],
            "iteration": iteration + 1,
            "checkpoint": "gap_method_selection",
        }
        logger.info(
            "gapfilling_method_pause mode=deterministic agent=%s options=%s reason=reporting_selected_source_requires_method_selection",
            gap_agent.name,
            gap_method_options,
        )
        return continuation_events + [awaiting_event]

    def _continue_gapfilling_to_construction(
        self,
        response: str,
        agent: Agent,
        messages: list[dict[str, str]],
        iteration: int,
        visited: set[str],
    ) -> list[CallbackEvent] | None:
        """Auto-delegate from GapFillingAgent to TimeSeriesConstructionAgent after gap-filling.

        Checks if a gap-filling method was applied by looking for apply_gap_filling
        tool results in the conversation.
        """
        # Look for gap-filling results in tool results
        filled_data = None
        final_text = response.split("Final Answer:", 1)[1].strip() if "Final Answer:" in response else response
        for message in messages:
            if message.get("role") == "user" and "Tool result:" in message.get("content", ""):
                try:
                    tool_result = json.loads(message["content"].replace("Tool result: ", "", 1))
                    if isinstance(tool_result, dict) and tool_result.get("method") and tool_result.get("prices"):
                        filled_data = tool_result
                        break
                except (json.JSONDecodeError, KeyError):
                    continue

        if not filled_data:
            selected_method = None
            last_user_message = next(
                (msg.get("content", "") for msg in reversed(messages) if msg.get("role") == "user"),
                "",
            )
            selected_method = self._extract_explicit_gap_method(last_user_message)

            if selected_method:
                recovered_prices = self._recover_gapfilling_prices_from_context(
                    messages,
                    final_text,
                )
                if recovered_prices is not None:
                    apply_tool = get_tool("apply_gap_filling")
                    if apply_tool is None:
                        return None
                    apply_args = {
                        "prices": recovered_prices,
                        "method": selected_method,
                    }
                    try:
                        filled_result = apply_tool.invoke(apply_args)
                    except Exception as error:
                        self.handler.on_tool_error(error)
                        logger.exception("gapfilling_recovery_apply_failed")
                        return [self._user_error("apply_gap_filling", str(error))]
                    if isinstance(filled_result, dict) and filled_result.get("prices"):
                        filled_data = filled_result
                        self.handler.add_trace_record(
                            "tool_call",
                            {
                                "tool": "apply_gap_filling",
                                "description": get_tool_description("apply_gap_filling"),
                                "arguments": apply_args,
                            },
                            agent=agent.name,
                            iteration=iteration,
                        )
                        self.handler.add_trace_record(
                            "tool_result",
                            {
                                "tool": "apply_gap_filling",
                                "description": get_tool_description("apply_gap_filling"),
                                "result": filled_result,
                            },
                            agent=agent.name,
                            iteration=iteration,
                        )
                        logger.info(
                            "gapfilling_recovered_apply mode=deterministic symbol=%s method=%s",
                            filled_data.get("symbol"),
                            selected_method,
                        )

                if filled_data is None:
                    self._log_continuation_decision(
                        agent.name,
                        iteration,
                        "completed",
                        f"gapfilling_explicit_method_detected_without_filled_data method={selected_method}",
                    )
                    return None

            if filled_data is None:
                self._log_continuation_decision(
                    agent.name,
                    iteration,
                    "pausing",
                    "gapfilling_requires_method_or_filled_data",
                )
                options = self._extract_gap_method_options_from_messages(messages)
                prompt = self._build_gap_method_prompt(options)
                self.handler.request_human_input(
                    prompt,
                    options=options,
                    context={"checkpoint": "gap_method_selection"},
                )
                self.handler.paused_state = {
                    "agent": agent.name,
                    "messages": messages.copy(),
                    "iteration": iteration + 1,
                    "checkpoint": "gap_method_selection",
                }
                logger.info(
                    "gapfilling_forced_pause mode=deterministic iteration=%d options=%s reason=missing_explicit_method_selection",
                    iteration,
                    options,
                )
                return self._drain()

        construction_agent = get_agent("TimeSeriesConstructionAgent")
        if construction_agent is None:
            return None

        user_texts = [
            msg.get("content", "")
            for msg in messages
            if msg.get("role") == "user"
        ]
        all_user_text = " ".join(user_texts).strip()
        original_match = re.search(r"original_request=(.+)", all_user_text)
        original_request = original_match.group(1).strip() if original_match else all_user_text

        transfer_request = (
            f"Build and persist the final time series for {filled_data.get('symbol', 'the instrument')} "
            f"using {filled_data.get('method', 'the selected')} gap-filling method. "
            f"Filled data: {json.dumps(filled_data)}. "
            f"Original request: {original_request}"
        )

        logger.info(
            "gapfilling_auto_continue mode=deterministic to_agent=%s symbol=%s method=%s reason=gapfilling_agent_completed_with_filled_data",
            construction_agent.name,
            filled_data.get("symbol"),
            filled_data.get("method"),
        )
        self._log_continuation_decision(
            agent.name,
            iteration,
            "delegating",
            f"gapfilling_to_construction symbol={filled_data.get('symbol')} method={filled_data.get('method')}",
        )

        continuation_events = [
            CallbackEvent(
                CallbackEventType.AGENT_COMPLETED,
                {"agent": agent.name, "result": {"delegated_to": construction_agent.name}},
                self.handler.session_id,
            ),
            CallbackEvent(
                CallbackEventType.DELEGATED,
                {
                    "from_agent": agent.name,
                    "to_agent": construction_agent.name,
                    "request": transfer_request,
                    "routing_mode": "deterministic",
                    "routing_reason": "bypassed user prompt: GapFillingAgent completed with filled data, auto-delegating to TimeSeriesConstructionAgent",
                },
                self.handler.session_id,
            ),
        ]

        target_result = self._run_agent(
            construction_agent,
            [{"role": "user", "content": transfer_request}],
            visited=visited.copy(),
        )

        return continuation_events + (target_result if self._is_event_list(target_result) else [])

    @staticmethod
    def _extract_prices_payload_from_messages(messages: list[dict[str, str]]) -> dict[str, Any] | None:
        """Extract a price series payload (dates/prices/symbol) from conversation context."""
        for message in messages:
            if message.get("role") != "user" or "Tool result:" not in message.get("content", ""):
                continue
            try:
                parsed = json.loads(message["content"].replace("Tool result: ", "", 1))
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            if parsed.get("dates") and parsed.get("prices") and parsed.get("symbol"):
                return parsed
        return None

    def _recover_gapfilling_prices_from_context(
        self,
        messages: list[dict[str, str]],
        final_text: str,
    ) -> dict[str, Any] | None:
        """Recover missing prices for deterministic gap-filling continuation.

        Priority:
        1) Existing tool result payload in current context.
        2) Deterministic historical_prices retrieval from symbol/source/date range.
        """
        direct_payload = self._extract_prices_payload_from_messages(messages)
        if direct_payload is not None:
            return direct_payload

        user_texts = [
            msg.get("content", "")
            for msg in messages
            if msg.get("role") == "user"
        ]
        all_user_text = " ".join(user_texts).strip()
        original_request = self._extract_original_request_from_messages(messages)
        symbol = (
            self._extract_symbol_candidate_from_text(all_user_text)
            or self._extract_symbol_candidate_from_text(final_text)
            or self._extract_symbol_candidate_from_text(original_request)
        )

        selected_source = self._extract_explicit_source_selection(all_user_text)
        if not selected_source:
            selected_source = self._extract_selected_source_marker_from_messages(messages)
        if not selected_source:
            selected_source = self._extract_explicit_source_selection(final_text)
        if not selected_source:
            sources = self._extract_sources_from_text(all_user_text)
            selected_source = sources[0] if len(sources) == 1 else None

        date_range = None
        date_candidates = [original_request, *user_texts, all_user_text, final_text]
        for candidate in date_candidates:
            if not candidate:
                continue
            try:
                parsed_range = extract_date_range(candidate)
            except ValueError:
                parsed_range = None
            if parsed_range is not None:
                date_range = parsed_range
                break

        if date_range is None:
            for candidate in date_candidates:
                if not candidate:
                    continue
                match = re.search(
                    r"(\d{4}-\d{2}-\d{2}).*?(\d{4}-\d{2}-\d{2})",
                    candidate,
                    flags=re.DOTALL,
                )
                if not match:
                    continue
                try:
                    date_range = normalize_date_range(match.group(1), match.group(2))
                except ValueError:
                    date_range = None
                if date_range is not None:
                    break

        if not symbol or not selected_source or date_range is None:
            logger.info(
                "gapfilling_price_recovery_skipped symbol=%s source=%s has_dates=%s",
                symbol,
                selected_source,
                date_range is not None,
            )
            return None

        start_date, end_date = date_range
        historical_tool = get_tool("historical_prices")
        if historical_tool is None:
            return None

        call_args = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "source": selected_source,
        }
        try:
            result = historical_tool.invoke(call_args)
        except Exception as error:
            self.handler.on_tool_error(error)
            logger.exception("gapfilling_price_recovery_failed")
            return None

        if isinstance(result, dict) and result.get("dates") and result.get("prices"):
            self.handler.add_trace_record(
                "tool_call",
                {
                    "tool": "historical_prices",
                    "description": get_tool_description("historical_prices"),
                    "arguments": call_args,
                },
                agent="GapFillingAgent",
            )
            self.handler.add_trace_record(
                "tool_result",
                {
                    "tool": "historical_prices",
                    "description": get_tool_description("historical_prices"),
                    "result": result,
                },
                agent="GapFillingAgent",
            )
            logger.info(
                "gapfilling_price_recovery_completed symbol=%s source=%s start=%s end=%s",
                symbol,
                selected_source,
                start_date,
                end_date,
            )
            return result
        return None

    @staticmethod
    def _extract_filled_data_from_messages(messages: list[dict[str, str]]) -> dict[str, Any] | None:
        """Extract filled series payload from tool results or transfer context."""
        for message in messages:
            if message.get("role") == "user" and "Tool result:" in message.get("content", ""):
                try:
                    tool_result = json.loads(message["content"].replace("Tool result: ", "", 1))
                except json.JSONDecodeError:
                    continue
                if isinstance(tool_result, dict) and tool_result.get("method") and tool_result.get("prices"):
                    return tool_result

        for message in messages:
            if message.get("role") != "user":
                continue
            text = message.get("content", "")
            match = re.search(
                r"Filled data:\s*(\{.*?\})(?:\s*\.\s*Original request:|$)",
                text,
                flags=re.DOTALL,
            )
            if not match:
                continue
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and parsed.get("prices"):
                return parsed
        return None

    @staticmethod
    def _extract_artifact_paths_from_messages(messages: list[dict[str, str]]) -> tuple[str | None, str | None]:
        """Extract generated CSV/PNG artifact paths from construction-stage tool results."""
        csv_path: str | None = None
        chart_path: str | None = None
        for message in messages:
            if message.get("role") != "user" or "Tool result:" not in message.get("content", ""):
                continue
            raw = message["content"].replace("Tool result: ", "", 1)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue

            candidate: str | None = None
            if isinstance(parsed, str):
                candidate = parsed
            elif isinstance(parsed, dict):
                for key in ("path", "output", "file"):
                    value = parsed.get(key)
                    if isinstance(value, str) and value:
                        candidate = value
                        break

            if not candidate:
                continue
            lowered = candidate.casefold()
            if lowered.endswith(".csv"):
                csv_path = candidate
            elif lowered.endswith(".png"):
                chart_path = candidate
        return csv_path, chart_path

    def _continue_construction_to_reporting(
        self,
        response: str,
        agent: Agent,
        messages: list[dict[str, str]],
        iteration: int,
        visited: set[str],
    ) -> list[CallbackEvent] | None:
        """Auto-delegate from TimeSeriesConstructionAgent to ReportingAgent for final summary.

        Ensures both final artifacts exist. If local-model output skipped one of the
        tool calls, this method deterministically produces missing artifacts before
        delegating to ReportingAgent for a final summary.
        """
        reporting_agent = get_agent("ReportingAgent")
        if reporting_agent is None:
            return None

        filled_data = self._extract_filled_data_from_messages(messages)
        csv_path, chart_path = self._extract_artifact_paths_from_messages(messages)

        if filled_data is not None:
            if not csv_path:
                build_tool = get_tool("build_timeseries")
                if build_tool is None:
                    return None
                build_args = {
                    "series": filled_data,
                    "filename": "final_timeseries.csv",
                    "run_id": self.handler.session_id,
                }
                try:
                    csv_result = build_tool.invoke(build_args)
                except Exception as error:
                    self.handler.on_tool_error(error)
                    logger.exception("construction_artifact_build_failed")
                    return [self._user_error("build_timeseries", str(error))]
                csv_path = str(csv_result)
                self.handler.add_trace_record(
                    "tool_call",
                    {
                        "tool": "build_timeseries",
                        "description": get_tool_description("build_timeseries"),
                        "arguments": build_args,
                    },
                    agent=agent.name,
                    iteration=iteration,
                )
                self.handler.add_trace_record(
                    "tool_result",
                    {
                        "tool": "build_timeseries",
                        "description": get_tool_description("build_timeseries"),
                        "result": csv_result,
                    },
                    agent=agent.name,
                    iteration=iteration,
                )
            if not chart_path:
                visualize_tool = get_tool("visualize_timeseries")
                if visualize_tool is None:
                    return None
                title_symbol = str(filled_data.get("symbol") or "Instrument")
                visual_args = {
                    "prices": filled_data,
                    "title": f"{title_symbol} continuous time series",
                    "run_id": self.handler.session_id,
                }
                try:
                    chart_result = visualize_tool.invoke(visual_args)
                except Exception as error:
                    self.handler.on_tool_error(error)
                    logger.exception("construction_artifact_visualization_failed")
                    return [self._user_error("visualize_timeseries", str(error))]
                chart_path = str(chart_result)
                self.handler.add_trace_record(
                    "tool_call",
                    {
                        "tool": "visualize_timeseries",
                        "description": get_tool_description("visualize_timeseries"),
                        "arguments": visual_args,
                    },
                    agent=agent.name,
                    iteration=iteration,
                )
                self.handler.add_trace_record(
                    "tool_result",
                    {
                        "tool": "visualize_timeseries",
                        "description": get_tool_description("visualize_timeseries"),
                        "result": chart_result,
                    },
                    agent=agent.name,
                    iteration=iteration,
                )

        if not csv_path or not chart_path:
            self._log_continuation_decision(
                agent.name,
                iteration,
                "completed",
                "construction_to_reporting_skipped_missing_artifacts",
            )
            return None

        original_request = self._extract_original_request_from_messages(messages)
        symbol = str((filled_data or {}).get("symbol") or "the instrument")
        method = str((filled_data or {}).get("method") or "selected")
        transfer_request = (
            f"{_REPORTING_FINAL_SUMMARY_NOTE} "
            f"Provide the final workflow summary for {symbol}. "
            f"Gap filling method: {method}. "
            f"CSV artifact: {csv_path}. "
            f"Chart artifact: {chart_path}. "
            f"Original request: {original_request}"
        )

        logger.info(
            "construction_auto_continue mode=deterministic to_agent=%s symbol=%s csv=%s chart=%s reason=construction_completed_with_artifacts",
            reporting_agent.name,
            symbol,
            csv_path,
            chart_path,
        )
        self._log_continuation_decision(
            agent.name,
            iteration,
            "delegating",
            f"construction_to_reporting symbol={symbol}",
        )

        continuation_events = [
            CallbackEvent(
                CallbackEventType.AGENT_COMPLETED,
                {
                    "agent": agent.name,
                    "result": {
                        "delegated_to": reporting_agent.name,
                        "timeseries_csv": csv_path,
                        "timeseries_chart": chart_path,
                    },
                },
                self.handler.session_id,
            ),
            CallbackEvent(
                CallbackEventType.DELEGATED,
                {
                    "from_agent": agent.name,
                    "to_agent": reporting_agent.name,
                    "request": transfer_request,
                    "routing_mode": "deterministic",
                    "routing_reason": "bypassed user prompt: TimeSeriesConstructionAgent completed with artifacts, auto-delegating to ReportingAgent for final summary",
                },
                self.handler.session_id,
            ),
        ]

        revisit_allowed = visited.copy()
        revisit_allowed.discard(reporting_agent.name)
        target_result = self._run_agent(
            reporting_agent,
            [{"role": "user", "content": transfer_request}],
            visited=revisit_allowed,
        )

        return continuation_events + (target_result if self._is_event_list(target_result) else [])

    def _execute(
        self,
        call: dict[str, Any],
        agent: Agent,
        messages: list[dict[str, str]],
        prompt: str,
        iteration: int,
        visited: set[str],
    ) -> Any:
        """Execute a single tool call from the agent.

        Args:
            call: Dict with 'name' and 'arguments' keys.
            agent: The current agent definition.
            messages: The conversation messages.
            prompt: The system prompt.
            iteration: Current iteration number.
            visited: Set of visited agent names.

        Returns:
            Tool result, list of CallbackEvents, or None if paused.
        """
        name, args = call["name"], call["arguments"]
        args = self._normalize_tool_args(name, args)
        tool_description = get_tool_description(name)
        if name == "get_instrument_details":
            query_value = str(args.get("query", "")).strip()
            if not query_value:
                user_context = " ".join(
                    message.get("content", "")
                    for message in messages
                    if message.get("role") == "user"
                )
                inferred_query = self._extract_instrument_query(user_context)
                if inferred_query:
                    args["query"] = inferred_query
                    logger.info(
                        "tool_arg_recovered agent=%s tool=%s source=user_context query=%s",
                        agent.name,
                        name,
                        inferred_query,
                    )
        if name == "generate_report" and "data" not in args:
            quality_rows = self._extract_quality_rows_from_messages(messages)
            if quality_rows:
                args["data"] = quality_rows
                logger.info(
                    "tool_arg_recovered agent=%s tool=%s source=quality_context rows=%d",
                    agent.name,
                    name,
                    len(quality_rows),
                )
            else:
                report_payload = {
                    key: value
                    for key, value in args.items()
                    if key not in {"filename", "run_id", "data"}
                }
                if report_payload:
                    args["data"] = report_payload
                    logger.warning(
                        "tool_arg_recovered agent=%s tool=%s source=raw_arguments keys=%s reason=missing_quality_context",
                        agent.name,
                        name,
                        sorted(report_payload.keys()),
                    )
        if name == "historical_prices" and "source" not in args:
            # Recover missing source from conversation context
            all_user_text = " ".join(
                msg.get("content", "")
                for msg in messages
                if msg.get("role") == "user"
            )
            for known_source in ("yahoo", "bloomberg", "reuters"):
                if known_source in all_user_text.casefold():
                    args["source"] = known_source
                    logger.info(
                        "tool_arg_recovered agent=%s tool=%s source=user_context recovered_source=%s",
                        agent.name,
                        name,
                        known_source,
                    )
                    break
            if "source" not in args:
                # Last resort: default to yahoo
                args["source"] = "yahoo"
                logger.warning(
                    "tool_arg_recovered agent=%s tool=%s source=default recovered_source=yahoo reason=no_source_in_context",
                    agent.name,
                    name,
                )
        logger.info(
            "tool_started agent=%s tool=%s iteration=%d", agent.name, name, iteration,
        )
        self._debug_flow_event(
            component="tool_dispatch",
            stage="start",
            agent=agent.name,
            iteration=iteration,
            detail=name,
        )
        logger.info(
            "workflow_tool agent=%s iteration=%d tool=%s phase=start description=%s",
            agent.name,
            iteration,
            name,
            tool_description or "",
        )
        logger.debug(
            "tool_arguments agent=%s tool=%s keys=%s",
            agent.name,
            name,
            sorted(args.keys()),
        )
        self.handler.add_trace_record(
            "tool_call",
            {
                "tool": name,
                "description": tool_description,
                "arguments": args,
            },
            agent=agent.name,
            iteration=iteration,
        )

        if name == "request_human_input":
            if agent.name == "Orchestrator":
                combined_user_context = " ".join(
                    msg.get("content", "")
                    for msg in messages
                    if msg.get("role") == "user"
                )
                auto_delegate_events = self._try_direct_delegate_from_request(combined_user_context)
                if auto_delegate_events is not None:
                    logger.info(
                        "orchestrator_reask_bypassed mode=deterministic reason=context_already_contains_instrument_and_dates"
                    )
                    return auto_delegate_events
            if agent.name == "ReferenceDataAgent":
                has_resolved_instrument = any(
                    '"found": true' in message.get("content", "").lower()
                    and '"symbol"' in message.get("content", "").lower()
                    for message in messages
                    if message.get("role") == "user" and "Tool result:" in message.get("content", "")
                )
                market_agent = get_agent("MarketDataAgent")
                if has_resolved_instrument and market_agent is not None:
                    # Extract the resolved symbol from the tool result
                    resolved_symbol = None
                    for message in messages:
                        if message.get("role") == "user" and "Tool result:" in message.get("content", ""):
                            try:
                                tool_result = json.loads(message["content"].replace("Tool result: ", "", 1))
                                if tool_result.get("found") and tool_result.get("symbol"):
                                    resolved_symbol = tool_result["symbol"]
                                    break
                            except (json.JSONDecodeError, KeyError):
                                continue
                    # Build a clean transfer request: use original_request if available,
                    # otherwise reconstruct from the enriched request fields
                    all_user_text = " ".join(
                        message.get("content", "")
                        for message in messages
                        if message.get("role") == "user"
                    ).strip()
                    original_match = re.search(r"original_request=(.+)", all_user_text)
                    if original_match:
                        original_request = original_match.group(1).strip()
                    else:
                        original_request = all_user_text
                    if resolved_symbol:
                        transfer_request = (
                            f"Retrieve historical prices for {resolved_symbol}. "
                            f"Original request: {original_request}"
                        )
                    else:
                        transfer_request = original_request
                    logger.info(
                        "reference_delegate_guard mode=deterministic to_agent=%s symbol=%s",
                        market_agent.name,
                        resolved_symbol,
                    )
                    guard_events = [
                        CallbackEvent(
                            CallbackEventType.AGENT_COMPLETED,
                            {"agent": agent.name, "result": {"delegated_to": market_agent.name}},
                            self.handler.session_id,
                        ),
                        CallbackEvent(
                            CallbackEventType.DELEGATED,
                            {
                                "from_agent": agent.name,
                                "to_agent": market_agent.name,
                                "request": transfer_request,
                                "routing_mode": "deterministic",
                                "routing_reason": "bypassed ReferenceDataAgent re-ask: instrument already resolved",
                            },
                            self.handler.session_id,
                        ),
                    ]
                    target_result = self._run_agent(
                        market_agent,
                        [{"role": "user", "content": transfer_request}],
                        visited=visited.copy(),
                    )
                    return guard_events + (target_result if self._is_event_list(target_result) else [])
            if (
                agent.name == "MarketDataAgent"
                and self._looks_like_market_source_selection_prompt(args)
            ):
                logger.info(
                    "market_source_selection_prompt_bypassed mode=deterministic reason=retrieving_all_sources"
                )
                return {
                    "selection": "all_sources",
                    "routing_mode": "deterministic",
                    "routing_reason": (
                        "bypassed MarketDataAgent source-selection prompt: "
                        "retrieving all available sources"
                    ),
                }
            if agent.name == "GapFillingAgent":
                # Prevent re-asking loop when the user already selected a valid method.
                last_user_message = next(
                    (msg.get("content", "") for msg in reversed(messages) if msg.get("role") == "user"),
                    "",
                )
                selected_method = self._extract_explicit_gap_method(last_user_message)
                if selected_method:
                    recovered_prices = self._recover_gapfilling_prices_from_context(
                        messages,
                        final_text="",
                    )
                    if recovered_prices is not None:
                        apply_tool = get_tool("apply_gap_filling")
                        if apply_tool is not None:
                            apply_args = {
                                "prices": recovered_prices,
                                "method": selected_method,
                            }
                            try:
                                filled_result = apply_tool.invoke(apply_args)
                            except Exception as error:
                                self.handler.on_tool_error(error)
                                logger.exception("gapfilling_loop_bypass_apply_failed")
                                return [self._user_error("apply_gap_filling", str(error))]
                            self.handler.add_trace_record(
                                "tool_call",
                                {
                                    "tool": "apply_gap_filling",
                                    "description": get_tool_description("apply_gap_filling"),
                                    "arguments": apply_args,
                                },
                                agent=agent.name,
                                iteration=iteration,
                            )
                            self.handler.add_trace_record(
                                "tool_result",
                                {
                                    "tool": "apply_gap_filling",
                                    "description": get_tool_description("apply_gap_filling"),
                                    "result": filled_result,
                                },
                                agent=agent.name,
                                iteration=iteration,
                            )
                            logger.info(
                                "gapfilling_request_input_bypassed mode=deterministic method=%s reason=user_already_selected_method",
                                selected_method,
                            )
                            return filled_result
        if name in {"build_timeseries", "visualize_timeseries"}:
            recovered_filled_data = self._extract_filled_data_from_messages(messages)
            if recovered_filled_data is not None:
                payload_key = "series" if name == "build_timeseries" else "prices"
                payload_value = args.get(payload_key)
                if not isinstance(payload_value, dict) or (
                    isinstance(recovered_filled_data.get("prices"), list)
                    and len(recovered_filled_data.get("prices", [])) >= len(payload_value.get("prices", []))
                ):
                    merged_payload = dict(recovered_filled_data)
                    if isinstance(payload_value, dict):
                        for key, value in payload_value.items():
                            if key not in {
                                "dates",
                                "prices",
                                "filled_dates",
                                "filled_prices",
                                "original_dates",
                                "original_prices",
                                "method",
                                "gap_filling_method",
                            }:
                                merged_payload.setdefault(key, value)
                    args[payload_key] = merged_payload
                    logger.info(
                        "tool_arg_recovered agent=%s tool=%s source=filled_context symbol=%s method=%s",
                        agent.name,
                        name,
                        merged_payload.get("symbol"),
                        merged_payload.get("method"),
                    )

        if name == "delegate_to_agent":
            target_name = str(args.get("agent_name", "")).strip()
            target = get_agent(target_name) if target_name else None
            delegated_request = str(args.get("request", "")).strip()

            # Recovery path: local models may emit malformed delegation payloads
            # (empty target or empty request). Prefer deterministic continuation
            # to the next workflow stage instead of re-entering an LLM loop.
            if target is None or not delegated_request:
                continuation = self._continue_after_agent_completion(
                    response="",
                    agent=agent,
                    messages=messages,
                    iteration=iteration,
                    visited=visited,
                )
                if continuation is not None:
                    logger.warning(
                        "agent_delegation_recovered mode=deterministic from_agent=%s target=%s has_request=%s reason=malformed_delegate_payload",
                        agent.name,
                        target_name,
                        bool(delegated_request),
                    )
                    return continuation

            if target is None:
                logger.error(
                    "agent_delegation_failed agent=%s target=%s",
                    agent.name, target_name,
                )
                self.handler.add_trace_record(
                    "tool_result",
                    {
                        "tool": name,
                        "description": tool_description,
                        "result": {"error": "Unknown target agent."},
                    },
                    agent=agent.name,
                    iteration=iteration,
                )
                return {"error": "Unknown target agent."}

            if target.name == agent.name:
                logger.warning(
                    "agent_self_delegation_recovered mode=deterministic agent=%s target=%s reason=self_delegate_tool_call",
                    agent.name,
                    target.name,
                )
                if agent.name == "TimeSeriesConstructionAgent":
                    # Construction can receive stale self-delegation actions when
                    # upstream deterministic recovery skips intermediate LLM turns.
                    # Treat as no-op and continue iterating within the same agent.
                    return {
                        "status": "skipped_self_delegation",
                        "agent": agent.name,
                    }
                continuation = self._continue_after_agent_completion(
                    response="",
                    agent=agent,
                    messages=messages,
                    iteration=iteration,
                    visited=visited,
                )
                if continuation is not None:
                    return continuation
                return {"error": f"Self-delegation is not allowed for {agent.name}."}

            if (
                agent.name == "GapFillingAgent"
                and target.name == "TimeSeriesConstructionAgent"
                and "Filled data:" not in delegated_request
            ):
                filled_data = self._extract_filled_data_from_messages(messages)
                if filled_data is not None:
                    original_request = self._extract_original_request_from_messages(messages)
                    base_request = delegated_request or (
                        "Build and persist the final continuous series using the selected gap-filling method."
                    )
                    delegated_request = (
                        f"{base_request} "
                        f"Filled data: {json.dumps(filled_data)}. "
                        f"Original request: {original_request}"
                    )
                    logger.info(
                        "gapfilling_delegate_enriched mode=deterministic target=%s symbol=%s method=%s",
                        target.name,
                        filled_data.get("symbol"),
                        filled_data.get("method"),
                    )

            if (
                agent.name == "TimeSeriesConstructionAgent"
                and target.name == "ReportingAgent"
                and _REPORTING_FINAL_SUMMARY_NOTE not in delegated_request
            ):
                delegated_request = f"{_REPORTING_FINAL_SUMMARY_NOTE} {delegated_request}".strip()

            if agent.name == "DataQualityAgent" and target.name == "ReportingAgent":
                quality_rows = self._extract_quality_rows_from_messages(messages)
                original_request = self._extract_original_request_from_messages(messages)
                candidate_sources = list(
                    dict.fromkeys(
                        self._extract_available_sources_from_messages(messages)
                        + self._extract_sources_from_text(original_request)
                        + list(SOURCES)
                    )
                )

                known_quality_sources = {
                    str(item.get("source", "")).strip().casefold()
                    for item in quality_rows
                    if item.get("source")
                }
                missing_sources = [source for source in candidate_sources if source not in known_quality_sources]
                if missing_sources:
                    historical_payloads: dict[str, dict[str, Any]] = {}
                    for message in messages:
                        if message.get("role") != "user" or "Tool result:" not in message.get("content", ""):
                            continue
                        try:
                            parsed = json.loads(message["content"].replace("Tool result: ", "", 1))
                        except json.JSONDecodeError:
                            continue
                        if (
                            isinstance(parsed, dict)
                            and parsed.get("source")
                            and isinstance(parsed.get("dates"), list)
                            and isinstance(parsed.get("prices"), list)
                        ):
                            source_key = str(parsed.get("source", "")).strip().casefold()
                            if source_key and source_key not in historical_payloads:
                                historical_payloads[source_key] = parsed

                    original_request = self._extract_original_request_from_messages(messages)
                    symbol_guess = self._extract_symbol_candidate_from_text(original_request)
                    if not symbol_guess:
                        symbol_guess = self._extract_symbol_candidate_from_text(" ".join(
                            msg.get("content", "") for msg in messages if msg.get("role") == "user"
                        ))
                    start_date = None
                    end_date = None
                    try:
                        extracted = extract_date_range(original_request)
                    except ValueError:
                        extracted = None
                    if extracted is not None:
                        start_date, end_date = extracted

                    quality_tool = get_tool("check_data_quality")
                    if quality_tool is not None:
                        for source in missing_sources:
                            source_key = str(source).strip().casefold()
                            if not source_key:
                                continue
                            tool_args: dict[str, Any]
                            payload = historical_payloads.get(source_key)
                            if payload is not None:
                                tool_args = {"data": payload}
                            else:
                                tool_args = {"source": source_key, "symbol": symbol_guess or "UNKNOWN"}
                                if start_date:
                                    tool_args["start_date"] = start_date
                                if end_date:
                                    tool_args["end_date"] = end_date
                            try:
                                quality_result = quality_tool.invoke(tool_args)
                            except Exception:
                                continue
                            if isinstance(quality_result, dict) and quality_result.get("source"):
                                quality_rows.append(quality_result)

                quality_report = self._build_data_quality_report(
                    quality_rows or [
                        {"source": source, "symbol": self._extract_symbol_candidate_from_text(self._extract_original_request_from_messages(messages)) or "UNKNOWN", "note": "fallback_context_only"}
                        for source in candidate_sources
                    ],
                    unavailable_sources=self._extract_unavailable_market_sources_from_messages(messages),
                )

                logger.info(
                    "quality_auto_continue mode=deterministic to_agent=%s sources=%d reason=dataquality_delegated_to_reporting",
                    target.name,
                    len(quality_report.get("rows", []) or []),
                )
                self._log_continuation_decision(
                    agent.name,
                    iteration,
                    "delegating",
                    f"quality_to_reporting sources={len(quality_report.get('rows', []) or [])}",
                )
                continuation_events = [
                    CallbackEvent(
                        CallbackEventType.AGENT_COMPLETED,
                        {
                            "agent": agent.name,
                            "result": {
                                "delegated_to": target.name,
                                "data_quality_report": quality_report,
                            },
                        },
                        self.handler.session_id,
                    ),
                    CallbackEvent(
                        CallbackEventType.DELEGATED,
                        {
                            "from_agent": agent.name,
                            "to_agent": target.name,
                            "request": delegated_request,
                            "routing_mode": "deterministic",
                            "routing_reason": "bypassed user prompt: DataQualityAgent completed with quality metrics, auto-delegating to ReportingAgent",
                        },
                        self.handler.session_id,
                    ),
                ]

                prompt, options = self._build_reporting_selection_prompt(quality_report.get("rows", []) or [])
                awaiting_event = CallbackEvent(
                    CallbackEventType.AWAITING_USER_INPUT,
                    {
                        "prompt": prompt,
                        "agent": target.name,
                        "options": options,
                        "context": {
                            "quality_rows": quality_report.get("rows", []) or [],
                            "checkpoint": "source_selection",
                            "data_quality_report": quality_report,
                        },
                    },
                    self.handler.session_id,
                )
                # Keep runtime pause state in sync when emitting the event directly.
                self.handler.current_agent = target.name
                self.handler.waiting_for_input = True
                self.handler.paused_state = {
                    "agent": target.name,
                    "messages": messages.copy(),
                    "iteration": iteration + 1,
                    "checkpoint": "source_selection",
                }
                logger.info(
                    "quality_report_pause mode=deterministic agent=%s sources=%d reason=dataquality_direct_reporting_pause",
                    target.name,
                    len(quality_report.get("rows", []) or []),
                )
                return continuation_events + [awaiting_event]

            logger.info(
                "agent_delegated from_agent=%s to_agent=%s",
                agent.name, target.name,
            )
            self._debug_flow_event(
                component="processor",
                stage="delegate",
                agent=agent.name,
                iteration=iteration,
                detail=f"to={target.name}",
            )
            completion_result: dict[str, Any] = {"delegated_to": target.name}
            if agent.name == "DataQualityAgent":
                quality_rows = self._extract_quality_rows_from_messages(messages)
                original_request = self._extract_original_request_from_messages(messages)
                candidate_sources = list(
                    dict.fromkeys(
                        self._extract_available_sources_from_messages(messages)
                        + self._extract_sources_from_text(original_request)
                        + list(SOURCES)
                    )
                )

                known_quality_sources = {
                    str(item.get("source", "")).strip().casefold()
                    for item in quality_rows
                    if item.get("source")
                }
                missing_sources = [source for source in candidate_sources if source not in known_quality_sources]
                if missing_sources:
                    historical_payloads: dict[str, dict[str, Any]] = {}
                    for message in messages:
                        if message.get("role") != "user" or "Tool result:" not in message.get("content", ""):
                            continue
                        try:
                            parsed = json.loads(message["content"].replace("Tool result: ", "", 1))
                        except json.JSONDecodeError:
                            continue
                        if (
                            isinstance(parsed, dict)
                            and parsed.get("source")
                            and isinstance(parsed.get("dates"), list)
                            and isinstance(parsed.get("prices"), list)
                        ):
                            source_key = str(parsed.get("source", "")).strip().casefold()
                            if source_key and source_key not in historical_payloads:
                                historical_payloads[source_key] = parsed

                    original_request = self._extract_original_request_from_messages(messages)
                    symbol_guess = self._extract_symbol_candidate_from_text(original_request)
                    if not symbol_guess:
                        symbol_guess = self._extract_symbol_candidate_from_text(" ".join(
                            msg.get("content", "") for msg in messages if msg.get("role") == "user"
                        ))
                    start_date = None
                    end_date = None
                    try:
                        extracted = extract_date_range(original_request)
                    except ValueError:
                        extracted = None
                    if extracted is not None:
                        start_date, end_date = extracted

                    quality_tool = get_tool("check_data_quality")
                    if quality_tool is not None:
                        for source in missing_sources:
                            source_key = str(source).strip().casefold()
                            if not source_key:
                                continue
                            tool_args: dict[str, Any]
                            payload = historical_payloads.get(source_key)
                            if payload is not None:
                                tool_args = {"data": payload}
                            else:
                                tool_args = {"source": source_key, "symbol": symbol_guess or "UNKNOWN"}
                                if start_date:
                                    tool_args["start_date"] = start_date
                                if end_date:
                                    tool_args["end_date"] = end_date
                            try:
                                quality_result = quality_tool.invoke(tool_args)
                            except Exception:
                                continue
                            if isinstance(quality_result, dict) and quality_result.get("source"):
                                quality_rows.append(quality_result)
                                known_quality_sources.add(source_key)
                                self.handler.add_trace_record(
                                    "tool_call",
                                    {
                                        "tool": "check_data_quality",
                                        "description": get_tool_description("check_data_quality"),
                                        "arguments": tool_args,
                                    },
                                    agent=agent.name,
                                    iteration=iteration,
                                )
                                self.handler.add_trace_record(
                                    "tool_result",
                                    {
                                        "tool": "check_data_quality",
                                        "description": get_tool_description("check_data_quality"),
                                        "result": quality_result,
                                    },
                                    agent=agent.name,
                                    iteration=iteration,
                                )

                if quality_rows:
                    completion_result["data_quality_report"] = self._build_data_quality_report(
                        quality_rows
                    )

            # Build delegation events directly (not via handler queue)
            delegation_events = [
                CallbackEvent(
                    CallbackEventType.AGENT_COMPLETED,
                    {"agent": agent.name, "result": completion_result},
                    self.handler.session_id,
                ),
                CallbackEvent(
                    CallbackEventType.DELEGATED,
                    {
                        "from_agent": agent.name,
                        "to_agent": target.name,
                        "request": delegated_request,
                        "routing_mode": "llm",
                        "routing_reason": "delegated by agent reasoning",
                    },
                    self.handler.session_id,
                ),
            ]

            if not delegated_request:
                self.handler.add_trace_record(
                    "tool_result",
                    {
                        "tool": name,
                        "description": tool_description,
                        "result": {"error": "Delegation request is empty."},
                    },
                    agent=agent.name,
                    iteration=iteration,
                )
                return {"error": "Delegation request is empty."}

            # Run target agent
            target_visited = visited.copy()
            if agent.name == "TimeSeriesConstructionAgent" and target.name == "ReportingAgent":
                target_visited.discard("ReportingAgent")
            target_result = self._run_agent(
                target,
                [{"role": "user", "content": delegated_request}],
                visited=target_visited,
            )
            # Combine delegation events with target result
            if self._is_event_list(target_result):
                return delegation_events + target_result
            return delegation_events + [CallbackEvent(
                CallbackEventType.ERROR,
                {"message": f"Target {target.name} returned unexpected result"},
                self.handler.session_id,
            )]

        tool = get_tool(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}

        if name == "check_data_quality" and agent.name == "DataQualityAgent":
            has_data_dict = isinstance(args.get("data"), dict)
            has_scalar_payload = (
                isinstance(args.get("prices"), list)
                and isinstance(args.get("source"), str)
                and isinstance(args.get("symbol"), str)
            )
            if not has_data_dict and not has_scalar_payload:
                requested_source = str(args.get("source", "")).strip().casefold()
                historical_payloads: dict[str, dict[str, Any]] = {}
                for message in messages:
                    if message.get("role") != "user" or "Tool result:" not in message.get("content", ""):
                        continue
                    try:
                        payload = json.loads(message["content"].replace("Tool result: ", "", 1))
                    except json.JSONDecodeError:
                        continue
                    if (
                        isinstance(payload, dict)
                        and payload.get("source")
                        and isinstance(payload.get("prices"), list)
                        and payload.get("symbol")
                    ):
                        source_key = str(payload.get("source", "")).strip().casefold()
                        if source_key and source_key not in historical_payloads:
                            historical_payloads[source_key] = payload

                recovered_payload = historical_payloads.get(requested_source) if requested_source else None
                if recovered_payload is None and historical_payloads:
                    recovered_payload = next(iter(historical_payloads.values()))
                if recovered_payload is not None:
                    args["data"] = recovered_payload
                    logger.warning(
                        "quality_tool_args_recovered mode=deterministic agent=%s source=%s symbol=%s reason=malformed_check_data_quality_payload",
                        agent.name,
                        recovered_payload.get("source"),
                        recovered_payload.get("symbol"),
                    )

        try:
            result = tool.invoke(args)
            logger.info(
                "tool_completed agent=%s tool=%s result_type=%s",
                agent.name, name, type(result).__name__,
            )
            logger.info(
                "workflow_tool agent=%s iteration=%d tool=%s phase=completed description=%s",
                agent.name,
                iteration,
                name,
                tool_description or "",
            )
            self.handler.add_trace_record(
                "tool_result",
                {
                    "tool": name,
                    "description": tool_description,
                    "result": result,
                },
                agent=agent.name,
                iteration=iteration,
            )
            failure = self._tool_failure(name, result)
            if failure:
                return [failure]
            # Strip large payloads from results that have a data_ref to avoid
            # sending full time series data to the LLM (token cost optimization).
            return self._strip_payload_for_llm(result)
        except Exception as error:
            if name == "check_data_quality" and agent.name == "DataQualityAgent":
                continuation = self._continue_quality_to_reporting(
                    response="",
                    agent=agent,
                    messages=messages,
                    iteration=iteration,
                    visited=visited,
                )
                if continuation is not None:
                    logger.warning(
                        "quality_tool_failure_recovered mode=deterministic agent=%s iteration=%d error=%s",
                        agent.name,
                        iteration,
                        str(error),
                    )
                    return continuation
            if name == "historical_prices" and agent.name == "MarketDataAgent":
                source = str(args.get("source", "")).strip().casefold() or "unknown"
                symbol = str(args.get("symbol", "")).strip() or "unknown"
                start_date = str(args.get("start_date", "")).strip()
                end_date = str(args.get("end_date", "")).strip()
                non_fatal_result = {
                    "market_data_available": False,
                    "non_fatal": True,
                    "source": source,
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "error": str(error),
                }
                logger.warning(
                    "market_data_unavailable_nonfatal symbol=%s source=%s start=%s end=%s error=%s",
                    symbol,
                    source,
                    start_date,
                    end_date,
                    error,
                )
                self.handler.add_trace_record(
                    "tool_result",
                    {
                        "tool": name,
                        "description": tool_description,
                        "result": non_fatal_result,
                    },
                    agent=agent.name,
                    iteration=iteration,
                )
                return non_fatal_result
            self.handler.on_tool_error(error)
            logger.exception("tool_failed agent=%s tool=%s", agent.name, name)
            self.handler.add_trace_record(
                "tool_error",
                {
                    "tool": name,
                    "description": tool_description,
                    "error": str(error),
                },
                agent=agent.name,
                iteration=iteration,
            )
            return [self._user_error(name, str(error))]

    @staticmethod
    def _strip_payload_for_llm(result: Any) -> Any:
        """Strip large payloads from tool results before sending to LLM.

        When a tool result contains a ``data_ref`` (meaning the full data is
        stored in the DataStore), the following large arrays are removed from
        the payload sent to the LLM to prevent token bloat:

        - ``dates``, ``prices`` (from historical_prices)
        - ``filled_dates``, ``filled_prices``, ``original_dates``, ``original_prices`` (from apply_gap_filling)

        The LLM still receives the ``data_ref`` which it can pass to downstream
        tools to load the full data on demand.

        Args:
            result: The raw tool result (dict or other).

        Returns:
            The result with large arrays stripped if ``data_ref`` is present.
        """
        if isinstance(result, dict) and result.get("data_ref"):
            stripped = dict(result)
            stripped.pop("dates", None)
            stripped.pop("prices", None)
            stripped.pop("filled_dates", None)
            stripped.pop("filled_prices", None)
            stripped.pop("original_dates", None)
            stripped.pop("original_prices", None)
            return stripped
        return result

    @staticmethod
    def _normalize_tool_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Accept common LLM aliases while keeping tool schemas explicit."""
        normalized = dict(args)
        if tool_name == "request_human_input":
            if "prompt" not in normalized and isinstance(normalized.get("message"), str):
                normalized["prompt"] = normalized["message"]
            if "options" not in normalized and isinstance(normalized.get("choices"), list):
                normalized["options"] = normalized["choices"]
        if tool_name == "get_instrument_details":
            if "query" in normalized and not isinstance(normalized["query"], str):
                normalized.pop("query", None)
            if "symbol" in normalized and not isinstance(normalized["symbol"], str):
                normalized.pop("symbol", None)
            if "query" not in normalized and "symbol" in normalized:
                normalized["query"] = normalized["symbol"]
            normalized.setdefault("identifier", "auto")
        if tool_name == "delegate_to_agent":
            if "agent_name" not in normalized and "agent" in normalized:
                normalized["agent_name"] = normalized["agent"]
            if "agent_name" not in normalized and "delegate_to" in normalized:
                normalized["agent_name"] = normalized["delegate_to"]
            if "request" not in normalized and isinstance(normalized.get("input"), str):
                normalized["request"] = normalized["input"]
            if "request" not in normalized:
                hints = {
                    key: value
                    for key, value in normalized.items()
                    if key in {
                        "instrument_symbol",
                        "symbol",
                        "source",
                        "start_date",
                        "end_date",
                        "method",
                    }
                }
                if hints:
                    normalized["request"] = json.dumps(hints)
        if tool_name == "historical_prices":
            if "symbol" not in normalized and "instrument_symbol" in normalized:
                normalized["symbol"] = normalized["instrument_symbol"]
            if "symbol" not in normalized and "ticker" in normalized:
                normalized["symbol"] = normalized["ticker"]
            if "source" not in normalized and "data_source" in normalized:
                normalized["source"] = normalized["data_source"]
            if "source" not in normalized and isinstance(normalized.get("sources"), list):
                source_list = [str(item).strip().casefold() for item in normalized["sources"] if str(item).strip()]
                if source_list:
                    normalized["source"] = source_list[0]
                    normalized.setdefault("source_candidates", source_list)
            # Fallback: when source is missing, try to extract from any available context
            # in the args (e.g. if passed as part of another field).
            if "source" not in normalized:
                for candidate_key in ("data", "context", "request"):
                    candidate = normalized.get(candidate_key)
                    if isinstance(candidate, str):
                        for known_source in ("yahoo", "bloomberg", "reuters"):
                            if known_source in candidate.casefold():
                                normalized["source"] = known_source
                                break
            if "start_date" not in normalized:
                if "from_date" in normalized:
                    normalized["start_date"] = normalized["from_date"]
                elif "start" in normalized:
                    normalized["start_date"] = normalized["start"]
            if "end_date" not in normalized:
                if "to_date" in normalized:
                    normalized["end_date"] = normalized["to_date"]
                elif "end" in normalized:
                    normalized["end_date"] = normalized["end"]
            if "date_range" in normalized and (
                "start_date" not in normalized or "end_date" not in normalized
            ):
                try:
                    extracted = extract_date_range(str(normalized["date_range"]))
                except ValueError:
                    extracted = None
                if extracted is not None:
                    normalized["start_date"], normalized["end_date"] = extracted
            if "start_date" in normalized and "end_date" in normalized:
                try:
                    start, end = normalize_date_range(
                        str(normalized["start_date"]),
                        str(normalized["end_date"]),
                    )
                    normalized["start_date"] = start
                    normalized["end_date"] = end
                except ValueError:
                    pass
        if tool_name == "check_data_quality":
            # Support passing the full historical_prices result dict as a single 'data' argument
            if "data" not in normalized and "prices" in normalized and isinstance(normalized["prices"], dict):
                # LLM passed the historical_prices dict as the 'prices' argument
                hp_dict = normalized.pop("prices")
                normalized["data"] = hp_dict
            if "symbol" not in normalized and isinstance(normalized.get("instrument_symbol"), str):
                normalized["symbol"] = normalized["instrument_symbol"]
            if isinstance(normalized.get("source"), list):
                source_list = [str(item).strip().casefold() for item in normalized["source"] if str(item).strip()]
                if source_list:
                    normalized["source"] = source_list[0]
                    normalized.setdefault("source_candidates", source_list)
                else:
                    normalized.pop("source", None)
            if "source" not in normalized and isinstance(normalized.get("sources"), list):
                source_list = [str(item).strip().casefold() for item in normalized["sources"] if str(item).strip()]
                if source_list:
                    normalized["source"] = source_list[0]
                    normalized.setdefault("source_candidates", source_list)
            if "data" not in normalized and "source" in normalized and "prices" not in normalized:
                # LLM passed source but not prices/symbol - likely a malformed call
                pass
            # Support data_ref: if provided, downstream tool will load from DataStore
            if "data_ref" in normalized and not isinstance(normalized["data_ref"], str):
                normalized.pop("data_ref", None)
        if tool_name == "build_timeseries":
            if "series" not in normalized and isinstance(normalized.get("prices"), list) and isinstance(normalized.get("dates"), list):
                normalized["series"] = {
                    "symbol": normalized.get("symbol"),
                    "source": normalized.get("source"),
                    "method": normalized.get("method") or normalized.get("gap_filling_method"),
                    "dates": normalized.get("dates"),
                    "prices": normalized.get("prices"),
                }
            if "series" in normalized and isinstance(normalized["series"], dict):
                series_payload = dict(normalized["series"])
                if "method" not in series_payload and "gap_filling_method" in series_payload:
                    series_payload["method"] = series_payload.get("gap_filling_method")
                normalized["series"] = series_payload
        if tool_name in {"extract_requested_date_range", "normalize_requested_dates"}:
            if "request" not in normalized and "text" in normalized:
                normalized["request"] = normalized["text"]
            if "start_date" not in normalized:
                if "start" in normalized:
                    normalized["start_date"] = normalized["start"]
                elif "from_date" in normalized:
                    normalized["start_date"] = normalized["from_date"]
            if "end_date" not in normalized:
                if "end" in normalized:
                    normalized["end_date"] = normalized["end"]
                elif "to_date" in normalized:
                    normalized["end_date"] = normalized["to_date"]
            if "request" not in normalized and "start_date" in normalized and "end_date" in normalized:
                normalized["request"] = (
                    f"from {normalized['start_date']} to {normalized['end_date']}"
                )
        return normalized

    @staticmethod
    def _is_event_list(value: Any) -> bool:
        """Check if a value is a list of CallbackEvents."""
        return isinstance(value, list) and all(
            isinstance(item, CallbackEvent) for item in value
        )

    @staticmethod
    def _tool_failure(tool_name: str, result: Any) -> CallbackEvent | None:
        """Check if a tool result indicates a failure condition."""
        if not isinstance(result, dict):
            return None
        if result.get("found") is False:
            return TimeSeriesConstructionProcessor._user_error(
                tool_name,
                unavailable_message(
                    "the requested instrument",
                    result.get("message", "No matching instrument was found."),
                ),
            )
        if tool_name == "historical_prices" and not result.get("dates"):
            return TimeSeriesConstructionProcessor._user_error(
                tool_name,
                unavailable_message(
                    "historical data",
                    "No observations exist for the requested ticker, source, or date range.",
                ),
            )
        if result.get("error"):
            return TimeSeriesConstructionProcessor._user_error(
                tool_name, str(result["error"])
            )
        return None

    @staticmethod
    def _user_error(operation: str, message: str) -> CallbackEvent:
        """Create a user-facing error event."""
        return CallbackEvent(
            CallbackEventType.ERROR,
            {
                "operation": operation,
                "message": message,
                "recoverable": True,
                "user_action": "Try another ticker, source, or supported date range.",
            },
        )

    @staticmethod
    def _looks_like_market_source_selection_prompt(args: dict[str, Any]) -> bool:
        """Detect market-data prompts that ask the user to pick one source."""
        prompt = str(args.get("prompt", "")).casefold()
        options = args.get("options") or []
        if not isinstance(options, list):
            return False
        normalized_options = {str(option).strip().casefold() for option in options if str(option).strip()}
        source_like = {"yahoo", "bloomberg", "reuters"}
        asks_to_select = any(phrase in prompt for phrase in ("select", "choose", "pick"))
        return asks_to_select and bool(normalized_options.intersection(source_like))

    @staticmethod
    def _looks_like_market_source_selection_final(response: str) -> bool:
        """Detect final answers that incorrectly ask user to choose one source."""
        content = response.casefold()
        requests_choice = any(
            phrase in content
            for phrase in (
                "please select",
                "select one",
                "choose one",
                "pick one",
            )
        )
        mentions_sources = sum(source in content for source in ("yahoo", "bloomberg", "reuters")) >= 2
        return requests_choice and mentions_sources

    @staticmethod
    def _looks_like_placeholder_final(response: str) -> bool:
        """Detect final answers that describe intent instead of returning actual data.

        Catches patterns like "Please wait while I retrieve..." or "Let me look up..."
        where the LLM describes what it will do rather than having called the tool.
        """
        content = response.casefold()
        placeholder_phrases = (
            "please wait",
            "let me look up",
            "let me retrieve",
            "let me check",
            "let me search",
            "let me find",
            "i will look up",
            "i will retrieve",
            "i will check",
            "i will search",
            "i will find",
            "i will now",
            "retrieving",
            "looking up",
            "searching for",
            "fetching",
        )
        return any(phrase in content for phrase in placeholder_phrases)

    @staticmethod
    def _parse_calls(text: str) -> list[dict[str, Any]]:
        """Parse ReACT Action/Action Input blocks from LLM output.

        Uses a multi-strategy approach to handle different model output styles:
        1. Strict single-line: ``Action: <name> Action Input: <json>``
        2. Multi-line: ``Action: <name>`` then ``Action Input: <json>`` on next line(s)
        3. Code-fenced JSON: ``{"name": "...", "arguments": {...}}``
        4. Loose JSON: any top-level JSON object with ``name`` and ``arguments`` keys
        """
        calls: list[dict[str, Any]] = []
        cleaned_text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)

        # Strategy 1: Strict single-line ReACT format
        action_matches = list(
            re.finditer(r"Action:\s*([A-Za-z_]\w*)\s+Action Input:\s*", cleaned_text)
        )
        for index, match in enumerate(action_matches):
            name = match.group(1)
            input_start = match.end()
            input_end = (
                action_matches[index + 1].start()
                if index + 1 < len(action_matches)
                else len(cleaned_text)
            )
            raw_input = cleaned_text[input_start:input_end].strip()
            raw_input = re.sub(
                r"^```(?:json|python)?\s*|\s*```$",
                "",
                raw_input,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
            parsed = TimeSeriesConstructionProcessor._parse_tool_input(raw_input)
            if parsed is None:
                logger.warning("Could not parse tool call for %s", name)
                continue
            calls.append({"name": name, "arguments": parsed})

        if calls:
            return calls

        # Strategy 2: Multi-line ReACT format (Action on one line, Action Input on next)
        multi_line_matches = list(
            re.finditer(
                r"Action:\s*([A-Za-z_]\w*)\s*\n\s*Action Input:\s*",
                cleaned_text,
                re.MULTILINE,
            )
        )
        for index, match in enumerate(multi_line_matches):
            name = match.group(1)
            input_start = match.end()
            input_end = (
                multi_line_matches[index + 1].start()
                if index + 1 < len(multi_line_matches)
                else len(cleaned_text)
            )
            raw_input = cleaned_text[input_start:input_end].strip()
            raw_input = re.sub(
                r"^```(?:json|python)?\s*|\s*```$",
                "",
                raw_input,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
            parsed = TimeSeriesConstructionProcessor._parse_tool_input(raw_input)
            if parsed is None:
                logger.warning("Could not parse multi-line tool call for %s", name)
                continue
            calls.append({"name": name, "arguments": parsed})

        if calls:
            return calls

        # Strategy 3: Code-fenced JSON tool call pattern
        code_fence_matches = list(
            re.finditer(
                r"```(?:json)?\s*\n?\s*\{\s*\"name\"\s*:\s*\"([A-Za-z_]\w*)\"",
                cleaned_text,
            )
        )
        if code_fence_matches:
            for match in code_fence_matches:
                name = match.group(1)
                # Find the closing fence
                remaining = cleaned_text[match.start():]
                close_fence = remaining.find("```", 6)
                json_block = remaining[:close_fence] if close_fence > 0 else remaining
                json_block = re.sub(
                    r"^```(?:json)?\s*|\s*```$",
                    "",
                    json_block,
                    flags=re.IGNORECASE | re.DOTALL,
                ).strip()
                parsed = TimeSeriesConstructionProcessor._parse_tool_input(json_block)
                if isinstance(parsed, dict) and parsed.get("name") and parsed.get("arguments"):
                    calls.append({"name": str(parsed["name"]), "arguments": parsed["arguments"]})

        if calls:
            return calls

        # Strategy 4: Loose JSON - find any JSON object with name/arguments keys
        json_decoder = json.JSONDecoder()
        search_start = 0
        while search_start < len(cleaned_text):
            try:
                parsed, end_pos = json_decoder.raw_decode(cleaned_text, search_start)
                if isinstance(parsed, dict) and parsed.get("name") and isinstance(parsed.get("arguments"), dict):
                    calls.append({"name": str(parsed["name"]), "arguments": parsed["arguments"]})
                elif isinstance(parsed, dict) and parsed.get("action"):
                    action_name = str(parsed.get("action", "")).strip()
                    if action_name:
                        action_input = parsed.get("input", {})
                        if not isinstance(action_input, dict):
                            action_input = {"value": action_input}
                        calls.append({"name": action_name, "arguments": action_input})
                search_start = end_pos + 1
            except json.JSONDecodeError:
                search_start += 1

        return calls

    @staticmethod
    def _parse_tool_input(raw_input: str) -> dict[str, Any] | None:
        """Parse JSON or Python-style dict output from local models."""
        decoder = json.JSONDecoder()
        try:
            parsed, _ = decoder.raw_decode(raw_input)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(raw_input)
            except (SyntaxError, ValueError):
                return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _prompt(agent: Agent) -> str:
        """Build the system prompt for an agent."""
        return agent_system_prompt(agent)

    def _drain(self) -> list[CallbackEvent]:
        """Drain all pending events from the handler."""
        events = []
        while self.handler.has_events():
            event = self.handler.poll()
            if event:
                events.append(event)
        return events

    def reset(self) -> None:
        """Reset the processor and handler state."""
        logger.info("workflow_reset session_id=%s", self.handler.session_id)
        self.handler.reset()
        self.pending_events.clear()