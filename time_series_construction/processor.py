"""LLM-driven ReAct processor with callback-based pause and resume."""
from __future__ import annotations

import json
import logging
import re
import ast
from typing import Any

try:
    from .agents_definition import Agent, CallbackEvent, CallbackEventType, get_agent
    from .handler import TimeSeriesConstructionCallbackHandler
    from .models import LLMRequest, ModelRequestFactory
    from .tools import get_tool
    from .prompts import agent_system_prompt, request_prompt, unavailable_message
except ImportError:
    from agents_definition import Agent, CallbackEvent, CallbackEventType, get_agent
    from handler import TimeSeriesConstructionCallbackHandler
    from models import LLMRequest, ModelRequestFactory
    from tools import get_tool
    from prompts import agent_system_prompt, request_prompt, unavailable_message

logger = logging.getLogger(__name__)


class TimeSeriesConstructionProcessor:
    def __init__(self, factory: ModelRequestFactory | None = None,
                 handler: TimeSeriesConstructionCallbackHandler | None = None) -> None:
        self.factory = factory or ModelRequestFactory.from_environment()
        self.handler = handler or TimeSeriesConstructionCallbackHandler()
        self.pending_events: list[CallbackEvent] = []
        logger.info("processor_initialized session_id=%s", self.handler.session_id)

    def process_user_request(self, user_input: str) -> list[CallbackEvent]:
        user_input = user_input.strip()
        logger.info("request_received characters=%d", len(user_input))
        if self._is_environment_command(user_input):
            logger.warning("request_rejected reason=environment_command")
            return [CallbackEvent(
                CallbackEventType.AWAITING_USER_INPUT,
                {
                    "agent": "System",
                    "prompt": "The conda environment is configured outside the application. Enter a financial request, for example: 'Build AAPL from 2023-01-01 to 2023-12-31'.",
                    "options": [],
                },
            )]
        if not user_input or not self._looks_like_financial_request(user_input):
            logger.info("request_requires_clarification")
            return [CallbackEvent(
                CallbackEventType.AWAITING_USER_INPUT,
                {
                    "agent": "Orchestrator",
                    "prompt": request_prompt(),
                    "options": [],
                },
            )]
        self.handler.emit(CallbackEvent(CallbackEventType.USER_REQUEST, {"request": user_input}))
        logger.info("workflow_started agent=Orchestrator")
        return self._run_agent(get_agent("Orchestrator"), [{"role": "user", "content": user_input}])

    @staticmethod
    def _is_environment_command(user_input: str) -> bool:
        command = user_input.casefold().strip()
        return command.startswith(("conda ", "source ", "export ", "pip ", "python ", "cd "))

    @staticmethod
    def _looks_like_financial_request(user_input: str) -> bool:
        text = user_input.casefold()
        finance_terms = ("ticker", "symbol", "stock", "share", "price", "series", "timeseries", "time series", "security", "asset", "market", "data")
        return any(term in text for term in finance_terms) or any(char.isdigit() for char in text)

    def process_user_response(self, user_input: str) -> list[CallbackEvent]:
        logger.info("user_response_received characters=%d", len(user_input))
        state = self.handler.handle_user_response(user_input)
        if state is None:
            return self._drain()
        messages = state["messages"] + [{"role": "user", "content": user_input}]
        return self._run_agent(get_agent(state["agent"]), messages, state.get("iteration", 0))

    def _run_agent(self, agent: Agent | None, messages: list[dict[str, str]], start: int = 0,
                   visited: set[str] | None = None) -> list[CallbackEvent]:
        if agent is None:
            return [CallbackEvent(CallbackEventType.ERROR, {"message": "Orchestrator is not registered."})]
        visited = visited or set()
        if agent.name in visited:
            return [CallbackEvent(CallbackEventType.ERROR, {"message": f"Agent cycle detected at {agent.name}."})]
        visited.add(agent.name)
        self.handler.current_agent = agent.name
        prompt = self._prompt(agent)
        logger.info("agent_started agent=%s iteration_start=%d", agent.name, start)
        for iteration in range(start, 10):
            logger.info("agent_iteration agent=%s iteration=%d", agent.name, iteration)
            try:
                response = self.factory.chat(LLMRequest(system_prompt=prompt, messages=messages, callbacks=[self.handler]))
            except Exception as error:
                self.handler.on_llm_error(error)
                return self._drain()
            messages.append({"role": "assistant", "content": response})
            if "Final Answer:" in response:
                recovery = self._recover_orchestrator_delegation(
                    response, agent, messages, iteration, visited,
                )
                if recovery is not None:
                    return recovery
                logger.info("agent_completed agent=%s iteration=%d", agent.name, iteration)
                self.handler.emit(CallbackEvent(CallbackEventType.AGENT_COMPLETED, {
                    "agent": agent.name, "result": {"final_answer": response.split("Final Answer:", 1)[1].strip()}}))
                return self._drain()
            calls = self._parse_calls(response)
            logger.info("agent_actions agent=%s iteration=%d count=%d", agent.name, iteration, len(calls))
            if not calls:
                recovery = self._recover_orchestrator_delegation(
                    response, agent, messages, iteration, visited,
                )
                if recovery is not None:
                    return recovery
                messages.append({"role": "user", "content": "Use the required Action and Action Input JSON format."})
                continue
            for call in calls:
                result = self._execute(call, agent, messages, prompt, iteration, visited)
                if result is None:
                    return self._drain()
                if self._is_event_list(result):
                    return result
                messages.append({"role": "user", "content": f"Tool result: {json.dumps(result, default=str)}"})
        self.handler.emit(CallbackEvent(CallbackEventType.ERROR, {"message": f"{agent.name} reached its iteration limit."}))
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
        logger.warning("orchestrator_delegation_recovered target=%s iteration=%d", target.name, iteration)
        original_request = next(
            (message["content"] for message in messages if message.get("role") == "user"),
            "",
        )
        return self._run_agent(
            target,
            [{"role": "user", "content": original_request}],
            visited=visited.copy(),
        )

    def _execute(self, call: dict[str, Any], agent: Agent, messages: list[dict[str, str]],
                 prompt: str, iteration: int, visited: set[str]) -> Any:
        name, args = call["name"], call["arguments"]
        args = self._normalize_tool_args(name, args)
        logger.info("tool_started agent=%s tool=%s iteration=%d", agent.name, name, iteration)
        if name == "request_human_input":
            self.handler.request_human_input(args.get("prompt", "Please choose an option."), args.get("options"), args.get("context"))
            self.handler.paused_state = {"agent": agent.name, "messages": messages.copy(), "iteration": iteration + 1}
            logger.info("agent_paused agent=%s iteration=%d", agent.name, iteration)
            return None
        if name == "delegate_to_agent":
            target_name = str(args.get("agent_name", ""))
            target = get_agent(target_name)
            if target is None:
                logger.error("agent_delegation_failed agent=%s target=%s", agent.name, target_name)
                return {"error": "Unknown target agent."}
            logger.info("agent_delegated from_agent=%s to_agent=%s", agent.name, target.name)
            self.handler.emit(CallbackEvent(
                CallbackEventType.AGENT_COMPLETED,
                {"agent": agent.name, "result": {"delegated_to": target.name}},
                self.handler.session_id,
            ))
            delegated_request = str(args.get("request", "")).strip()
            if not delegated_request:
                return {"error": "Delegation request is empty."}
            return self._run_agent(
                target,
                [{"role": "user", "content": delegated_request}],
                visited=visited.copy(),
            )
        tool = get_tool(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            result = tool.invoke(args)
            logger.info("tool_completed agent=%s tool=%s result_type=%s", agent.name, name, type(result).__name__)
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
            if "query" not in normalized and "symbol" in normalized:
                normalized["query"] = normalized["symbol"]
            normalized.setdefault("identifier", "auto")
        return normalized

    @staticmethod
    def _is_event_list(value: Any) -> bool:
        return isinstance(value, list) and all(isinstance(item, CallbackEvent) for item in value)

    @staticmethod
    def _tool_failure(tool_name: str, result: Any) -> CallbackEvent | None:
        if not isinstance(result, dict):
            return None
        if result.get("found") is False:
            return TimeSeriesConstructionProcessor._user_error(
                tool_name, unavailable_message("the requested instrument", result.get("message", "No matching instrument was found.")),
            )
        if tool_name == "historical_prices" and not result.get("dates"):
            return TimeSeriesConstructionProcessor._user_error(
                tool_name, unavailable_message("historical data", "No observations exist for the requested ticker, source, or date range."),
            )
        if result.get("error"):
            return TimeSeriesConstructionProcessor._user_error(tool_name, str(result["error"]))
        return None

    @staticmethod
    def _user_error(operation: str, message: str) -> CallbackEvent:
        return CallbackEvent(CallbackEventType.ERROR, {
            "operation": operation,
            "message": message,
            "recoverable": True,
            "user_action": "Try another ticker, source, or supported date range.",
        })

    @staticmethod
    def _parse_calls(text: str) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        action_matches = list(re.finditer(r"Action:\s*([A-Za-z_]\w*)\s+Action Input:\s*", text))
        for index, match in enumerate(action_matches):
            name = match.group(1)
            input_start = match.end()
            input_end = action_matches[index + 1].start() if index + 1 < len(action_matches) else len(text)
            raw_input = text[input_start:input_end].strip()
            raw_input = re.sub(r"^```(?:json|python)?\s*|\s*```$", "", raw_input, flags=re.IGNORECASE | re.DOTALL).strip()
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
        return agent_system_prompt(agent)

    def _drain(self) -> list[CallbackEvent]:
        events = []
        while self.handler.has_events():
            event = self.handler.poll()
            if event:
                events.append(event)
        return events

    def reset(self) -> None:
        logger.info("workflow_reset session_id=%s", self.handler.session_id)
        self.handler.reset()
        self.pending_events.clear()
