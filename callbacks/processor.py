"""
ReAct agent loop for the travel booking system.

Tool delegation and human-in-the-loop events are managed through
``TravelBookingCallbackHandler``, leveraging ``BaseCallbackHandler`` callbacks.

Key simplifications over the previous version:
  - Tool execution is handled by the processor's ``_handle_tool_call`` method,
    which delegates special tools (``delegate_to_agent``, ``request_human_input``)
    through the callback handler.
  - The ReAct loop is a single method on the processor.
  - Comprehensive INFO-level logging across all agent, tool, and LLM operations.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from agents import (
    Agent,
    CallbackEvent,
    CallbackEventType,
    get_agent,
)
from tools import get_tool
from handler import TravelBookingCallbackHandler
from models import LLMRequest, ModelRequestFactory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_MODEL = (os.getenv("LLM_MODEL") or "ollama/gemma4:e4b").strip()

# ---------------------------------------------------------------------------
# LLM call via ModelRequestFactory
# ---------------------------------------------------------------------------

_factory = ModelRequestFactory()
_factory.set_defaults(model=OLLAMA_MODEL)


def _call_llm(
    system_prompt: str,
    messages: list[dict[str, str]],
    handler: TravelBookingCallbackHandler,
) -> str:
    """Call the LLM via litellm through ``ModelRequestFactory``.

    The callback handler is attached so litellm fires
    ``on_llm_error`` / ``on_tool_error`` automatically.
    """
    agent_name = handler.current_agent or "unknown"
    logger.info(
        "LLM_CALL agent=%s | messages=%d | system_len=%d",
        agent_name,
        len(messages),
        len(system_prompt),
    )
    for i, m in enumerate(messages):
        logger.debug("LLM_CALL_MSG[%d] agent=%s | role=%s | content_len=%d",
                     i, agent_name, m.get("role"), len(m.get("content", "")))

    request = LLMRequest(
        model=OLLAMA_MODEL,
        system_prompt=system_prompt,
        messages=messages.copy(),
        temperature=0.1,
        max_tokens=2048,
        callbacks=[handler],
    )
    response = _factory.create(request)()

    logger.info(
        "LLM_RESPONSE agent=%s | response_len=%d | preview=%s",
        agent_name,
        len(response),
        response[:200] + "..." if len(response) > 200 else response,
    )
    return response


# ---------------------------------------------------------------------------
# Tool call parsing (supports ReAct Action/Action Input format)
# ---------------------------------------------------------------------------

def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tool calls from LLM output.

    Supports the ReAct format produced by LLM prompts:

        Action: <tool_name>
        Action Input: <JSON arguments for the tool>
    """
    calls: list[dict[str, Any]] = []

    for m in re.finditer(
        r'Action:\s*(\w+)\s*\nAction Input:\s*(\{.*?\})',
        text,
        re.DOTALL,
    ):
        try:
            calls.append({"name": m.group(1), "arguments": json.loads(m.group(2))})
            logger.debug("PARSED_TOOL name=%s | args=%s", m.group(1), m.group(2))
        except json.JSONDecodeError as e:
            logger.warning("TOOL_PARSE_SKIP name=%s | json_error=%s", m.group(1), e)
            calls.append({"name": m.group(1), "arguments": {}})

    return calls


# ---------------------------------------------------------------------------
# ReAct system prompt builder
# ---------------------------------------------------------------------------

def _build_react_prompt(agent: Agent) -> str:
    """Build the ReAct system prompt for *agent* with its tools listed."""
    tools_text_lines: list[str] = []
    for t_name in agent.tools:
        if t_name == "request_human_input":
            tools_text_lines.append(
                "  - request_human_input: Request input from the human user. "
                "Use this when you need the user to make a decision or provide information."
            )
            continue
        tool = get_tool(t_name)
        if tool:
            tools_text_lines.append(f"  - {tool.name}: {tool.description}")

    tools_text = "\n".join(tools_text_lines) or "No tools available."

    return agent.system_prompt + f"""

You are a ReAct (Reasoning + Acting) agent. Follow this format exactly:

    Thought: <your reasoning about what to do next>
    Action: <tool_name>
    Action Input: <JSON arguments for the tool>

When you receive the tool result, continue with:
    Thought: <your reasoning about the result>
    Action: <next tool or final answer>

When you are done, use:
    Thought: I have completed the task.
    Final Answer: <summary of what was done>

Available tools:
{tools_text}

IMPORTANT: Always use the Action/Action Input format to call tools.
NEVER make up results. Always use tools to get real data.
When you need information from the user, use the `request_human_input` tool."""


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class TravelBookingProcessor:
    """Manages agent lifecycle and user interaction.

    Uses ``ModelRequestFactory`` internally for all LLM calls.
    Tool delegation and human-in-the-loop events are routed through
    ``TravelBookingCallbackHandler``.
    """

    def __init__(self) -> None:
        self.handler = TravelBookingCallbackHandler()
        self.pending_events: list[CallbackEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_user_request(self, user_input: str) -> list[CallbackEvent]:
        """Run the orchestrator agent with a new user request."""
        logger.info("PROCESS_REQUEST input=%s", user_input)
        orchestrator = get_agent("Orchestrator")
        if not orchestrator:
            logger.error("PROCESS_REQUEST_FAILED agent=Orchestrator not found")
            return [
                CallbackEvent(
                    CallbackEventType.ERROR,
                    {"message": "Orchestrator not found"},
                )
            ]

        events = self._run_agent(
            orchestrator,
            [{"role": "user", "content": user_input}],
        )
        self.pending_events.extend(events)
        return events

    def process_user_response(self, user_input: str) -> list[CallbackEvent]:
        """Resume the paused agent with the user's response."""
        logger.info("PROCESS_RESPONSE input=%s", user_input)
        state = self.handler.handle_user_response(user_input)
        if state is None:
            logger.info("PROCESS_RESPONSE_CANCELLED")
            return list(self.handler.event_queue)

        agent = get_agent(state.get("agent", "Orchestrator")) or get_agent(
            "Orchestrator"
        )
        logger.info(
            "PROCESS_RESPONSE_RESUME agent=%s | iteration=%s",
            agent.name,
            state.get("iteration"),
        )

        events = self._run_agent(
            agent,
            state.get("messages", []),
            start_iteration=state.get("iteration", 0),
        )
        self.pending_events.extend(events)
        return events

    def get_pending_events(self) -> list[CallbackEvent]:
        """Return and clear any pending events."""
        events = list(self.pending_events)
        self.pending_events.clear()
        return events

    def reset(self) -> None:
        """Reset handler and pending events for a fresh conversation."""
        logger.info("PROCESSOR_RESET")
        self.handler.reset()
        self.pending_events.clear()

    # ------------------------------------------------------------------
    # Internal: ReAct loop
    # ------------------------------------------------------------------

    def _run_agent(
        self,
        agent: Agent,
        messages: list[dict[str, str]],
        start_iteration: int = 0,
        max_iterations: int = 10,
        visited_agents: set[str] | None = None,
    ) -> list[CallbackEvent]:
        """Run the ReAct loop for *agent*.

        Args:
            agent: The agent to run.
            messages: Conversation history so far.
            start_iteration: Which iteration to start from (0 = fresh).
            max_iterations: Maximum total iterations.
            visited_agents: Set of agent names already visited in this
                delegation chain. Prevents infinite delegation loops.

        Returns:
            Events emitted during this loop segment. The loop pauses when
            ``request_human_input`` is called; the caller must resume by
            calling this method again with updated messages.
        """
        if visited_agents is None:
            visited_agents = set()

        # Cycle detection
        if agent.name in visited_agents:
            logger.warning(
                "CYCLE_DETECTED agent=%s | visited=%s",
                agent.name, visited_agents,
            )
            return [
                CallbackEvent(
                    CallbackEventType.AGENT_COMPLETED,
                    {
                        "agent": agent.name,
                        "result": {
                            "final_answer": (
                                f"I already handled this request under "
                                f"{agent.name}. Please provide a new request."
                            )
                        },
                    },
                )
            ]

        visited_agents.add(agent.name)
        self.handler.current_agent = agent.name
        system_prompt = _build_react_prompt(agent)
        events: list[CallbackEvent] = []

        logger.info(
            "REACT_START agent=%s | iterations=[%d..%d] | visited=%s",
            agent.name,
            start_iteration,
            max_iterations - 1,
            visited_agents,
        )

        for iteration in range(start_iteration, max_iterations):
            logger.debug(
                "REACT_ITER agent=%s | iteration=%d/%d",
                agent.name,
                iteration,
                max_iterations - 1,
            )
            response = _call_llm(system_prompt, messages, self.handler)
            messages.append({"role": "assistant", "content": response})

            # Check for final answer
            if "Final Answer:" in response:
                logger.info(
                    "REACT_FINAL agent=%s | iteration=%d",
                    agent.name,
                    iteration,
                )
                final_text = response.split("Final Answer:", 1)[-1].strip()
                events.append(
                    CallbackEvent(
                        CallbackEventType.AGENT_COMPLETED,
                        {
                            "agent": agent.name,
                            "result": {"final_answer": final_text},
                        },
                    )
                )
                return events

            # Parse and execute tool calls
            tool_calls = _parse_tool_calls(response)
            if not tool_calls:
                logger.warning(
                    "REACT_NO_TOOL agent=%s | iteration=%d | response_preview=%s",
                    agent.name,
                    iteration,
                    response[:300] + "..." if len(response) > 300 else response,
                )
                if iteration < max_iterations - 1:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Please use one of the available tools to "
                                "proceed. Use the format:\n"
                                "Action: <tool_name>\n"
                                "Action Input: <JSON>"
                            ),
                        }
                    )
                    continue
                logger.warning(
                    "REACT_MAX_ITER agent=%s | iteration=%d",
                    agent.name,
                    iteration,
                )
                return events

            logger.info(
                "REACT_TOOLS agent=%s | iteration=%d | tool_calls=%s",
                agent.name,
                iteration,
                [tc["name"] for tc in tool_calls],
            )

            for tc in tool_calls:
                result = self._handle_tool_call(
                    tc, agent.name, messages, system_prompt,
                    iteration, visited_agents,
                )
                # If the handler returned events (delegation or pause),
                # return them immediately.
                if isinstance(result, list):
                    logger.info(
                        "REACT_PAUSE agent=%s | iteration=%d | reason=%s",
                        agent.name,
                        iteration,
                        tc["name"],
                    )
                    return result
                messages.append(
                    {
                        "role": "user",
                        "content": f"Tool '{tc['name']}' result: {result}",
                    }
                )

        return events

    # ------------------------------------------------------------------
    # Internal: Tool execution
    # ------------------------------------------------------------------

    def _handle_tool_call(
        self,
        tc: dict[str, Any],
        agent_name: str,
        messages: list[dict[str, str]],
        system_prompt: str,
        iteration: int,
        visited_agents: set[str],
    ) -> str | list[CallbackEvent]:
        """Execute a single tool call.

        Returns:
            - A result string for standard tools.
            - A list of ``CallbackEvent`` for delegation or pause, which
              signals the caller to return immediately.
        """
        tool_name = tc["name"]
        args = tc.get("arguments", {})
        logger.info(
            "TOOL_CALL agent=%s | tool=%s | args=%s",
            agent_name,
            tool_name,
            args,
        )

        # --- Delegation ---------------------------------------------------
        if tool_name == "delegate_to_agent":
            agent_target = args.get("agent_name", "")
            request = args.get("request", "")

            if not agent_target:
                logger.warning("TOOL_DELEGATE_MISSING_AGENT agent=%s", agent_name)
                return json.dumps({
                    "status": "error",
                    "message": "Missing 'agent_name' in delegate_to_agent arguments.",
                })

            delegate = get_agent(agent_target)
            if not delegate:
                logger.warning(
                    "TOOL_DELEGATE_UNKNOWN agent=%s | target=%s",
                    agent_name,
                    agent_target,
                )
                return json.dumps({
                    "status": "error",
                    "message": (
                        f"Unknown agent: {agent_target}. "
                        f"Must be one of: CarBookingAgent, AirTicketAgent, "
                        f"HotelReservationAgent."
                    ),
                })

            logger.info(
                "TOOL_DELEGATE agent=%s | target=%s | request=%s",
                agent_name,
                agent_target,
                request,
            )

            # Emit completion for the delegating agent
            self.handler.emit(
                CallbackEvent(
                    CallbackEventType.AGENT_COMPLETED,
                    {
                        "agent": agent_name,
                        "result": {
                            "final_answer": (
                                f"Delegating to {agent_target}..."
                            )
                        },
                    },
                )
            )

            return self._run_agent(
                delegate,
                [{"role": "user", "content": request}],
                visited_agents=visited_agents,
            )

        # --- Human input --------------------------------------------------
        if tool_name == "request_human_input":
            logger.info(
                "TOOL_HUMAN_INPUT agent=%s | prompt=%s | options=%s",
                agent_name,
                args.get("prompt"),
                args.get("options"),
            )
            self.handler.request_human_input(
                prompt=args.get("prompt", "Please provide input:"),
                options=args.get("options"),
                context={**args.get("context", {}), "agent": agent_name},
            )
            self.handler.paused_state = {
                "agent": agent_name,
                "messages": messages.copy(),
                "system_prompt": system_prompt,
                "iteration": iteration + 1,
            }
            logger.info(
                "TOOL_HUMAN_INPUT_PAUSED agent=%s | iteration=%d",
                agent_name,
                iteration,
            )
            return [
                CallbackEvent(
                    CallbackEventType.AWAITING_USER_INPUT,
                    {"agent": agent_name},
                )
            ]

        # --- Standard LangChain tool --------------------------------------
        tool = get_tool(tool_name)
        if not tool:
            logger.warning("TOOL_UNKNOWN agent=%s | tool=%s", agent_name, tool_name)
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            result = tool.invoke(args)
            logger.info(
                "TOOL_OK agent=%s | tool=%s | result_preview=%s",
                agent_name,
                tool_name,
                str(result)[:200] + "..." if len(str(result)) > 200 else str(result),
            )
            return json.dumps(result) if not isinstance(result, str) else result
        except Exception as e:
            logger.error(
                "TOOL_ERROR agent=%s | tool=%s | error=%s",
                agent_name,
                tool_name,
                e,
            )
            self.handler.on_tool_error(e)
            return json.dumps({"error": f"Tool execution failed: {e}"})