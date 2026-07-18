"""
ReAct agent loop and tool execution for the travel booking system.

Uses **litellm** (via ``ModelRequestFactory``) instead of raw httpx for LLM
interactions.  All tools are proper LangChain ``StructuredTool`` instances
(no manual JSON tool‑call parsing).

Key design:
  - ``ModelRequestFactory`` returns a litellm‑backed chat client.
  - ``TravelBookingCallbackHandler`` is registered with litellm via the
    ``LLMRequest.callbacks`` field so errors are captured automatically.
  - The ReAct loop is unchanged in structure — only the LLM call and
    tool execution paths were swapped.
  - Delegation uses a dedicated ``delegate_to_agent`` tool, separate from
    ``request_human_input``, to prevent the infinite loop that occurred
    when both delegation and user-input pausing were conflated.
  - A visited‑agents set prevents re-entering the same agent recursively.
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
    get_tool,
)
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
    request = LLMRequest(
        model=OLLAMA_MODEL,
        system_prompt=system_prompt,
        messages=messages.copy(),
        temperature=0.1,
        max_tokens=2048,
        callbacks=[handler],
    )
    client = _factory.create(request)
    return client()


# ---------------------------------------------------------------------------
# Tool call parsing (supports JSON, XML, and ReAct Action: formats)
# ---------------------------------------------------------------------------

def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tool calls from LLM output.

    Supports three formats:
      - JSON: ``{"name": "...", "arguments": {...}}``
      - XML:  ``<function_call>name</function_call>`` + JSON body
      - ReAct: ``Action: name\\nAction Input: {...}``
    """
    calls: list[dict[str, Any]] = []

    # JSON: {"name": "...", "arguments": {...}}
    for m in re.finditer(
        r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}',
        text, re.DOTALL,
    ):
        try:
            calls.append({"name": m.group(1), "arguments": json.loads(m.group(2))})
        except json.JSONDecodeError:
            pass

    # XML: <function_call>name</function_call> + JSON body
    for m in re.finditer(r'<function_call>\s*(\w+)\s*</function_call>', text):
        rest = text[m.end():]
        am = re.search(r'\{[^}]+\}', rest)
        try:
            calls.append({
                "name": m.group(1),
                "arguments": json.loads(am.group(0)) if am else {},
            })
        except (json.JSONDecodeError, AttributeError):
            calls.append({"name": m.group(1), "arguments": {}})

    # ReAct: Action: name \n Action Input: {...}
    for m in re.finditer(
        r'Action:\s*(\w+)\s*\nAction Input:\s*(\{.*?\})',
        text, re.DOTALL,
    ):
        try:
            calls.append({"name": m.group(1), "arguments": json.loads(m.group(2))})
        except json.JSONDecodeError:
            calls.append({"name": m.group(1), "arguments": {}})

    return calls


# ---------------------------------------------------------------------------
# Tool execution using LangChain StructuredTool
# ---------------------------------------------------------------------------

def _execute_tool(
    tool_call: dict[str, Any],
    agent_name: str,
    handler: TravelBookingCallbackHandler,
) -> str:
    """Execute a LangChain tool by name, returning a JSON result string.

    ``request_human_input`` is special‑cased — it triggers the callback
    handler instead of running a function.  ``delegate_to_agent`` is also
    special‑cased to return a delegation result that the ReAct loop
    interprets to switch to a specialist agent.
    """
    tool_name = tool_call["name"]

    # Special case: delegate_to_agent — emit delegation result
    if tool_name == "delegate_to_agent":
        args = tool_call.get("arguments", {})
        agent_target = args.get("agent_name", "")
        request = args.get("request", "")
        if not agent_target:
            return json.dumps({
                "status": "error",
                "message": "Missing 'agent_name' in delegate_to_agent arguments.",
            })
        if not get_agent(agent_target):
            return json.dumps({
                "status": "error",
                "message": f"Unknown agent: {agent_target}. "
                           f"Must be one of: CarBookingAgent, AirTicketAgent, HotelReservationAgent.",
            })
        return json.dumps({
            "status": "delegating",
            "delegate_to": agent_target,
            "original_request": request,
        })

    # Special case: request_human_input is handled by the callback system
    if tool_name == "request_human_input":
        args = tool_call.get("arguments", {})
        ctx = args.get("context", {})
        ctx["agent"] = agent_name

        # Specialist agent needs actual user input: pause the loop
        handler.request_human_input(
            prompt=args.get("prompt", "Please provide input:"),
            options=args.get("options"),
            context=ctx,
        )
        return json.dumps({"status": "awaiting_human_input"})

    # Standard LangChain tool
    tool = get_tool(tool_name)
    if not tool:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        result = tool.invoke(tool_call.get("arguments", {}))
        return json.dumps(result) if not isinstance(result, str) else result
    except Exception as e:
        logger.error("Tool '%s' failed: %s", tool_name, e)
        # Notify the handler so litellm callbacks fire
        handler.on_tool_error(e)
        return json.dumps({"error": f"Tool execution failed: {e}"})


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
# Single ReAct loop (handles both initial runs and resumes)
# ---------------------------------------------------------------------------

def _react_loop(
    agent: Agent,
    handler: TravelBookingCallbackHandler,
    messages: list[dict[str, str]],
    system_prompt: str,
    start_iteration: int = 0,
    max_iterations: int = 10,
    visited_agents: set[str] | None = None,
) -> list[CallbackEvent]:
    """Run the ReAct loop from the given state.

    Args:
        agent: The agent to run.
        handler: Callback handler for human-in-the-loop events.
        messages: Conversation history so far.
        system_prompt: The full ReAct system prompt.
        start_iteration: Which iteration to start from (0 = fresh, >0 = resume).
        max_iterations: Maximum total iterations.
        visited_agents: Set of agent names already visited in this delegation
            chain. Used to prevent infinite delegation loops.

    Returns:
        Events emitted during this loop segment. The loop pauses when
        ``request_human_input`` is called; the caller must resume by calling
        this function again with updated messages.
    """
    if visited_agents is None:
        visited_agents = set()

    # Cycle detection: if we have already run this agent in this chain,
    # force-complete to prevent infinite loops.
    if agent.name in visited_agents:
        logger.warning(
            "Cycle detected: agent '%s' was already visited. "
            "Forcing completion to prevent infinite loop.",
            agent.name,
        )
        return [
            CallbackEvent(
                CallbackEventType.AGENT_COMPLETED,
                {
                    "agent": agent.name,
                    "result": {
                        "final_answer": (
                            f"I already handled this request under {agent.name}. "
                            "Please provide a new request or clarify."
                        ),
                    },
                },
            ),
        ]

    visited_agents.add(agent.name)
    handler.current_agent = agent.name
    events: list[CallbackEvent] = []

    for iteration in range(start_iteration, max_iterations):
        response = _call_llm(system_prompt, messages, handler)
        messages.append({"role": "assistant", "content": response})

        # Check for final answer
        if "Final Answer:" in response:
            events.append(CallbackEvent(
                CallbackEventType.AGENT_COMPLETED,
                {"agent": agent.name, "result": {"final_answer": response}},
            ))
            return events

        # Parse and execute tool calls
        tool_calls = _parse_tool_calls(response)
        if not tool_calls:
            if iteration < max_iterations - 1:
                messages.append({
                    "role": "user",
                    "content": "Please use one of the available tools to proceed. "
                               "Use the format:\nAction: <tool_name>\nAction Input: <JSON>",
                })
                continue
            logger.warning("Agent '%s' hit max iterations without completing.", agent.name)
            return events

        for tc in tool_calls:
            result = _execute_tool(tc, agent.name, handler)
            messages.append({"role": "user", "content": f"Tool '{tc['name']}' result: {result}"})

            # Handle delegation: immediately start the specialist agent
            try:
                result_data = json.loads(result)
                if result_data.get("status") == "delegating":
                    delegate_to = result_data.get("delegate_to")
                    original_request = result_data.get("original_request", "")
                    delegate = get_agent(delegate_to) if delegate_to else None
                    if delegate:
                        logger.info("Delegating to '%s' with request: %s",
                                    delegate_to, original_request)
                        delegate_prompt = _build_react_prompt(delegate)
                        delegate_messages = [{"role": "user", "content": original_request}]
                        # Pass visited_agents through so cycles are detected
                        return _react_loop(
                            delegate, handler, delegate_messages,
                            delegate_prompt, start_iteration=0,
                            max_iterations=max_iterations,
                            visited_agents=visited_agents,
                        )
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

            # Pause if waiting for human input
            try:
                if json.loads(result).get("status") == "awaiting_human_input":
                    handler.paused_state = {
                        "agent": agent.name,
                        "messages": messages,
                        "system_prompt": system_prompt,
                        "iteration": iteration + 1,
                        "context": tc.get("arguments", {}).get("context", {}),
                    }
                    return events
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

    return events


# ---------------------------------------------------------------------------
# Processor (thin facade)
# ---------------------------------------------------------------------------

class TravelBookingProcessor:
    """Manages agent lifecycle and user interaction.

    Uses ``ModelRequestFactory`` internally for all LLM calls.
    """

    def __init__(self) -> None:
        self.handler = TravelBookingCallbackHandler()
        self.pending_events: list[CallbackEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_user_request(self, user_input: str) -> list[CallbackEvent]:
        """Run the orchestrator agent with a new user request."""
        orchestrator = get_agent("Orchestrator")
        if not orchestrator:
            return [CallbackEvent(CallbackEventType.ERROR, {"message": "Orchestrator not found"})]

        system_prompt = _build_react_prompt(orchestrator)
        messages = [{"role": "user", "content": user_input}]

        events = _react_loop(orchestrator, self.handler, messages, system_prompt)
        self.pending_events.extend(events)
        return events

    def process_user_response(self, user_input: str) -> list[CallbackEvent]:
        """Resume the paused agent with the user's response."""
        state = self.handler.handle_user_response(user_input)
        if state is None:
            return list(self.handler.event_queue)

        agent = get_agent(state.get("agent", "Orchestrator"))
        if not agent:
            agent = get_agent("Orchestrator")

        events = _react_loop(
            agent,
            self.handler,
            state.get("messages", []),
            state.get("system_prompt", _build_react_prompt(agent)),
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
        self.handler.reset()
        self.pending_events.clear()