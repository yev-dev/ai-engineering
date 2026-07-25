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
    get_tool,
    normalize_date_range,
)
from financial_time_series_construction.prompts import (
    agent_system_prompt,
    request_prompt,
    unavailable_message,
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
        logger.info("agent_started agent=%s iteration_start=%d", agent.name, start)

        for iteration in range(start, 10):
            logger.info("agent_iteration agent=%s iteration=%d", agent.name, iteration)

            # Get LLM response
            try:
                response = self.factory.chat(
                    LLMRequest(
                        system_prompt=prompt,
                        messages=messages,
                        callbacks=[self.handler],
                    )
                )
            except Exception as error:
                self.handler.on_llm_error(error)
                return self._drain()

            # Record trace
            self.handler.add_to_trace(
                f"[{agent.name}:{iteration}] {response}"
            )
            messages.append({"role": "assistant", "content": response})

            # Check for Final Answer
            if "Final Answer:" in response:
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
                    return continuation
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

            # Execute each tool call
            for call in calls:
                result = self._execute(
                    call, agent, messages, prompt, iteration, visited,
                )
                if result is None:
                    return self._drain()
                if self._is_event_list(result):
                    return result
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
        """Recover when an LLM states delegation instead of calling the tool."""
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
        """Auto-delegate from ReferenceDataAgent to MarketDataAgent after instrument resolution.

        When ReferenceDataAgent completes with a Final Answer that includes a resolved
        instrument symbol, this method automatically delegates to MarketDataAgent to
        continue the workflow without returning to the user for input.

        Args:
            response: The LLM response containing the Final Answer.
            agent: The current agent definition.
            messages: The conversation messages.
            iteration: Current iteration number.
            visited: Set of visited agent names.

        Returns:
            List of CallbackEvents if continuation was triggered, None otherwise.
        """
        if agent.name != "ReferenceDataAgent":
            return None

        # Check if the final answer contains a resolved instrument symbol
        final_text = response.split("Final Answer:", 1)[1].strip() if "Final Answer:" in response else response

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

        if not resolved_symbol:
            return None

        market_agent = get_agent("MarketDataAgent")
        if market_agent is None:
            return None

        # Build the transfer request preserving the original user request context
        all_user_text = " ".join(
            msg.get("content", "")
            for msg in messages
            if msg.get("role") == "user"
        ).strip()
        original_match = re.search(r"original_request=(.+)", all_user_text)
        if original_match:
            original_request = original_match.group(1).strip()
        else:
            original_request = all_user_text

        transfer_request = (
            f"Retrieve historical prices for {resolved_symbol}. "
            f"Original request: {original_request}"
        )

        logger.info(
            "reference_auto_continue mode=deterministic to_agent=%s symbol=%s reason=reference_agent_completed_with_resolved_instrument",
            market_agent.name,
            resolved_symbol,
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
        logger.debug(
            "tool_arguments agent=%s tool=%s keys=%s",
            agent.name,
            name,
            sorted(args.keys()),
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
            target_name = str(args.get("agent_name", ""))
            target = get_agent(target_name)
            if target is None:
                logger.error(
                    "agent_delegation_failed agent=%s target=%s",
                    agent.name, target_name,
                )
                return {"error": "Unknown target agent."}
            logger.info(
                "agent_delegated from_agent=%s to_agent=%s",
                agent.name, target.name,
            )
            # Build delegation events directly (not via handler queue)
            delegation_events = [
                CallbackEvent(
                    CallbackEventType.AGENT_COMPLETED,
                    {"agent": agent.name, "result": {"delegated_to": target.name}},
                    self.handler.session_id,
                ),
                CallbackEvent(
                    CallbackEventType.DELEGATED,
                    {
                        "from_agent": agent.name,
                        "to_agent": target.name,
                        "request": str(args.get("request", "")),
                        "routing_mode": "llm",
                        "routing_reason": "delegated by agent reasoning",
                    },
                    self.handler.session_id,
                ),
            ]

            delegated_request = str(args.get("request", "")).strip()
            if not delegated_request:
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
            failure = self._tool_failure(name, result)
            if failure:
                return [failure]
            return result
        except Exception as error:
            self.handler.on_tool_error(error)
            logger.exception("tool_failed agent=%s tool=%s", agent.name, name)
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