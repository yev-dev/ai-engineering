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
    extract_date_range,
    get_instrument_details,
    get_tool,
    get_tool_description,
    normalize_date_range,
)
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
    "query parameter. Do not describe what you will do — call the tool now and return the actual result."
)


class TimeSeriesConstructionProcessor:
    """Main processor that orchestrates the ReACT agent workflow.

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
        all_user_text = " ".join(
            msg.get("content", "")
            for msg in messages
            if msg.get("role") == "user"
        ).strip()
        original_match = re.search(r"original_request=(.+)", all_user_text)
        return original_match.group(1).strip() if original_match else all_user_text

    @staticmethod
    def _extract_symbol_candidate_from_text(text: str) -> str | None:
        """Extract a likely ticker symbol token from free-form text."""
        if not text:
            return None
        # Prefer explicit symbol/ticker patterns before generic uppercase tokens.
        explicit = re.search(
            r"\b(?:symbol|ticker)\s*(?:is|=|:)?\s*([A-Za-z]{1,6})\b",
            text,
            re.IGNORECASE,
        )
        if explicit:
            return explicit.group(1).upper()
        generic = re.search(r"\b([A-Z]{1,6})\b", text)
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
    def _log_continuation_decision(
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

    @staticmethod
    def _extract_quality_rows_from_messages(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Extract quality rows from prior tool results or transfer context."""
        rows: list[dict[str, Any]] = []

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
    def _build_data_quality_report(quality_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Build a JSON-serializable data quality report.

        The rows payload is list-of-dicts so it can be serialized directly to a
        DataFrame by downstream consumers.
        """
        normalized_rows: list[dict[str, Any]] = []
        completeness_candidates: list[tuple[str, float]] = []
        total_missing = 0
        symbol = None

        for item in quality_rows:
            row = {
                "source": item.get("source"),
                "symbol": item.get("symbol"),
                "total_values": item.get("total_values"),
                "missing_count": item.get("missing_count", item.get("nan_count")),
                "completeness_pct": item.get("completeness_pct"),
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

        return {
            "report_type": "data_quality_summary",
            "rows": normalized_rows,
            "summary": {
                "symbol": symbol,
                "source_count": len(normalized_rows),
                "sources": [row.get("source") for row in normalized_rows if row.get("source")],
                "total_missing_count": total_missing,
                "average_completeness_pct": average_completeness,
                "best_source_by_completeness": best_source,
                "worst_source_by_completeness": worst_source,
            },
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
        messages = state["messages"] + [{"role": "user", "content": user_input}]
        return self._run_agent(
            get_agent(state["agent"]),
            messages,
            state.get("iteration", 0),
        )

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
        prompt = self._prompt(agent)
        loop_detector = LoopDetector(max_repeats=3)
        log_workflow_progress(agent.name, start, "started")
        logger.info("agent_started agent=%s iteration_start=%d", agent.name, start)

        for iteration in range(start, 10):
            log_workflow_progress(agent.name, iteration, "llm_call")
            logger.info("agent_iteration agent=%s iteration=%d", agent.name, iteration)

            # Log message size to track growing conversation history
            log_message_size(agent.name, iteration, messages)

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

            # Check for Final Answer
            if "Final Answer:" in response:
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
            if calls:
                logger.info(
                    "agent_actions_detail agent=%s iteration=%d tools=%s",
                    agent.name,
                    iteration,
                    ",".join(call["name"] for call in calls),
                )

            if not calls:
                recovery = self._recover_orchestrator_delegation(
                    response, agent, messages, iteration, visited,
                )
                if recovery is not None:
                    return recovery
                messages.append(
                    {
                        "role": "user",
                        "content": "Use the required Action and Action Input JSON format.",
                    }
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
        # Look for loaded data sources in tool results
        loaded_sources = []
        resolved_symbol = None
        for message in messages:
            if message.get("role") == "user" and "Tool result:" in message.get("content", ""):
                try:
                    tool_result = json.loads(message["content"].replace("Tool result: ", "", 1))
                    if isinstance(tool_result, dict) and tool_result.get("dates") and tool_result.get("source"):
                        loaded_sources.append(tool_result["source"])
                        if not resolved_symbol and tool_result.get("symbol"):
                            resolved_symbol = tool_result["symbol"]
                except (json.JSONDecodeError, KeyError):
                    continue

        original_request = self._extract_original_request_from_messages(messages)
        final_text = response.split("Final Answer:", 1)[1].strip() if "Final Answer:" in response else response

        if not resolved_symbol:
            resolved_symbol = self._extract_symbol_candidate_from_text(final_text)
        if not resolved_symbol:
            resolved_symbol = self._extract_symbol_candidate_from_text(original_request)

        if not loaded_sources:
            loaded_sources = self._extract_sources_from_text(final_text)
        if not loaded_sources:
            loaded_sources = self._extract_sources_from_text(original_request)
        if not loaded_sources and resolved_symbol:
            # Deterministic fallback: proceed with known universe when the model
            # summarized success but omitted explicit tool calls.
            loaded_sources = ["yahoo", "bloomberg", "reuters"]
            logger.warning(
                "market_sources_recovered mode=deterministic symbol=%s sources=%s reason=final_answer_without_tool_results",
                resolved_symbol,
                loaded_sources,
            )

        if not loaded_sources:
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

        continuation_events = [
            CallbackEvent(
                CallbackEventType.AGENT_COMPLETED,
                {"agent": agent.name, "result": {"delegated_to": quality_agent.name}},
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

        Checks if quality metrics were computed for at least one source
        by looking for check_data_quality tool results in the conversation.
        """
        # Look for quality check results in tool results
        quality_sources = []
        for message in messages:
            if message.get("role") == "user" and "Tool result:" in message.get("content", ""):
                try:
                    tool_result = json.loads(message["content"].replace("Tool result: ", "", 1))
                    if isinstance(tool_result, dict) and "completeness_pct" in tool_result and tool_result.get("source"):
                        quality_sources.append(tool_result)
                except (json.JSONDecodeError, KeyError):
                    continue

        if not quality_sources:
            # Fallback: continue with instruction to produce comparison + source selection,
            # even when the model omitted explicit check_data_quality tool calls.
            original_request = self._extract_original_request_from_messages(messages)
            symbol_guess = self._extract_symbol_candidate_from_text(response)
            if not symbol_guess:
                symbol_guess = self._extract_symbol_candidate_from_text(original_request)
            quality_sources = [
                {
                    "source": "yahoo",
                    "symbol": symbol_guess or "UNKNOWN",
                    "note": "fallback_context_only",
                },
                {
                    "source": "bloomberg",
                    "symbol": symbol_guess or "UNKNOWN",
                    "note": "fallback_context_only",
                },
                {
                    "source": "reuters",
                    "symbol": symbol_guess or "UNKNOWN",
                    "note": "fallback_context_only",
                },
            ]
            logger.warning(
                "quality_metrics_recovered mode=deterministic symbol=%s reason=final_answer_without_tool_results",
                symbol_guess,
            )

        reporting_agent = get_agent("ReportingAgent")
        if reporting_agent is None:
            return None

        # Build transfer request with quality context
        all_user_text = " ".join(
            msg.get("content", "")
            for msg in messages
            if msg.get("role") == "user"
        ).strip()
        original_match = re.search(r"original_request=(.+)", all_user_text)
        original_request = original_match.group(1).strip() if original_match else all_user_text

        transfer_request = (
            f"Present quality report and ask the user to select a source. "
            f"Quality data: {json.dumps(quality_sources)}. "
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

        quality_report = self._build_data_quality_report(quality_sources)

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

        target_result = self._run_agent(
            reporting_agent,
            [{"role": "user", "content": transfer_request}],
            visited=visited.copy(),
        )

        return continuation_events + (target_result if self._is_event_list(target_result) else [])

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
        # Check if the final answer or explicit user response contains a source selection
        final_text = response.split("Final Answer:", 1)[1].strip() if "Final Answer:" in response else response

        # Look for a selected source in the conversation
        selected_source = None
        resolved_symbol = None

        # If not found in paused state, only inspect the latest user message
        # (explicit response after prompt). Do not scan all history because
        # transfer/context messages may contain all source names.
        if not selected_source:
            last_user_message = next(
                (msg.get("content", "") for msg in reversed(messages) if msg.get("role") == "user"),
                "",
            )
            selected_source = self._extract_explicit_source_selection(last_user_message)

        # Also check final answer if the model explicitly states one selected source.
        if not selected_source:
            selected_source = self._extract_explicit_source_selection(final_text)

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

        all_user_text = " ".join(
            msg.get("content", "")
            for msg in messages
            if msg.get("role") == "user"
        ).strip()
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

        target_result = self._run_agent(
            gap_agent,
            [{"role": "user", "content": transfer_request}],
            visited=visited.copy(),
        )

        return continuation_events + (target_result if self._is_event_list(target_result) else [])

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
            if not selected_method:
                selected_method = self._extract_explicit_gap_method(final_text)

            if selected_method:
                self._log_continuation_decision(
                    agent.name,
                    iteration,
                    "completed",
                    f"gapfilling_explicit_method_detected_without_filled_data method={selected_method}",
                )
                return None

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

        all_user_text = " ".join(
            msg.get("content", "")
            for msg in messages
            if msg.get("role") == "user"
        ).strip()
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
        logger.info(
            "tool_started agent=%s tool=%s iteration=%d", agent.name, name, iteration,
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
            self.handler.request_human_input(
                args.get("prompt", "Please choose an option."),
                args.get("options"),
                args.get("context"),
            )
            self.handler.paused_state = {
                "agent": agent.name,
                "messages": messages.copy(),
                "iteration": iteration + 1,
            }
            logger.info("agent_paused agent=%s iteration=%d", agent.name, iteration)
            return None

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
            logger.info(
                "agent_delegated from_agent=%s to_agent=%s",
                agent.name, target.name,
            )
            completion_result: dict[str, Any] = {"delegated_to": target.name}
            if agent.name == "DataQualityAgent":
                quality_rows = self._extract_quality_rows_from_messages(messages)
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
            target_result = self._run_agent(
                target,
                [{"role": "user", "content": delegated_request}],
                visited=visited.copy(),
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
            return result
        except Exception as error:
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
    def _normalize_tool_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Accept common LLM aliases while keeping tool schemas explicit."""
        normalized = dict(args)
        if tool_name == "get_instrument_details":
            if "query" in normalized and not isinstance(normalized["query"], str):
                normalized.pop("query", None)
            if "symbol" in normalized and not isinstance(normalized["symbol"], str):
                normalized.pop("symbol", None)
            if "query" not in normalized and "symbol" in normalized:
                normalized["query"] = normalized["symbol"]
            normalized.setdefault("identifier", "auto")
        if tool_name == "historical_prices":
            if "symbol" not in normalized and "ticker" in normalized:
                normalized["symbol"] = normalized["ticker"]
            if "source" not in normalized and "data_source" in normalized:
                normalized["source"] = normalized["data_source"]
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
            if "data" not in normalized and "source" in normalized and "prices" not in normalized:
                # LLM passed source but not prices/symbol - likely a malformed call
                pass
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
        """Parse ReACT Action/Action Input blocks from LLM output."""
        calls: list[dict[str, Any]] = []
        action_matches = list(
            re.finditer(r"Action:\s*([A-Za-z_]\w*)\s+Action Input:\s*", text)
        )
        for index, match in enumerate(action_matches):
            name = match.group(1)
            input_start = match.end()
            input_end = (
                action_matches[index + 1].start()
                if index + 1 < len(action_matches)
                else len(text)
            )
            raw_input = text[input_start:input_end].strip()
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