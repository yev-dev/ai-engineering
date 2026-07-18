"""
ReAct agent loop and tool execution for the travel booking system.

Uses Ollama API directly (via httpx) to drive ReAct agents. The processor
manages agent lifecycle, tool execution, and human-in-the-loop pauses.

Key design:
  - Single _react_loop() function handles both initial runs and resumes
  - Tool execution is a simple dispatch table
  - The processor class is a thin facade over the loop + handler
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx

from agents import (
    Agent,
    CallbackEvent,
    CallbackEventType,
    get_agent,
    get_tool,
)
from handler import TravelBookingCallbackHandler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE = (os.getenv("OLLAMA_ENDPOINT") or "http://localhost:11434").rstrip("/v1")
OLLAMA_MODEL = (os.getenv("OLLAMA_MODEL") or "gemma4:e4b").strip()
CHAT_URL = f"{OLLAMA_BASE}/api/chat"

# ---------------------------------------------------------------------------
# LLM call — direct Ollama API (no litellm dependency)
# ---------------------------------------------------------------------------

def _call_llm(
    system_prompt: str,
    messages: list[dict[str, str]],
) -> str:
    """Call Ollama's chat API directly via httpx."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 2048},
    }
    try:
        resp = httpx.post(CHAT_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return f"Error: LLM call failed - {e}"


# ---------------------------------------------------------------------------
# Tool call parsing (supports JSON, XML, and ReAct Action: formats)
# ---------------------------------------------------------------------------

def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls = []

    # JSON: {"name": "...", "arguments": {...}}
    for m in re.finditer(r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}', text, re.DOTALL):
        try:
            calls.append({"name": m.group(1), "arguments": json.loads(m.group(2))})
        except json.JSONDecodeError:
            pass

    # XML: <function_call>name</function_call> + JSON
    for m in re.finditer(r'<function_call>\s*(\w+)\s*</function_call>', text):
        rest = text[m.end():]
        am = re.search(r'\{[^}]+\}', rest)
        try:
            calls.append({"name": m.group(1), "arguments": json.loads(am.group(0)) if am else {}})
        except (json.JSONDecodeError, AttributeError):
            calls.append({"name": m.group(1), "arguments": {}})

    # ReAct: Action: name \n Action Input: {...}
    for m in re.finditer(r'Action:\s*(\w+)\s*\nAction Input:\s*(\{.*?\})', text, re.DOTALL):
        try:
            calls.append({"name": m.group(1), "arguments": json.loads(m.group(2))})
        except json.JSONDecodeError:
            calls.append({"name": m.group(1), "arguments": {}})

    return calls


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _execute_tool(
    tool_call: dict[str, Any],
    agent_name: str,
    handler: TravelBookingCallbackHandler,
) -> str:
    """Execute a tool. Returns JSON result string.

    Special-cases request_human_input to trigger the callback handler
    instead of running a function.
    """
    tool = get_tool(tool_call["name"])
    if not tool:
        return json.dumps({"error": f"Unknown tool: {tool_call['name']}"})

    args = tool_call.get("arguments", {})

    if tool.name == "request_human_input":
        ctx = args.get("context", {})
        ctx["agent"] = agent_name
        delegate_to = ctx.get("delegate_to")

        if delegate_to:
            # Delegation: immediately start the specialist agent.
            return json.dumps({
                "status": "delegating",
                "delegate_to": delegate_to,
                "original_request": ctx.get("original_request", ""),
                "agent": agent_name,
            })

        # Orchestrator called request_human_input without delegate_to.
        # Instead of pausing for user input, tell it to delegate properly.
        if agent_name == "Orchestrator":
            return json.dumps({
                "status": "must_delegate",
                "message": "You must delegate to a specialist agent. "
                           "Use context with delegate_to set to one of: "
                           "CarBookingAgent, AirTicketAgent, HotelReservationAgent.",
            })

        # Specialist agent needs actual user input: pause the loop.
        handler.request_human_input(
            prompt=args.get("prompt", "Please provide input:"),
            options=args.get("options"),
            context=ctx,
        )
        return json.dumps({"status": "awaiting_human_input"})

    if tool.fn:
        try:
            return tool.fn(**args)
        except Exception as e:
            logger.error("Tool '%s' failed: %s", tool.name, e)
            return json.dumps({"error": f"Tool execution failed: {e}"})

    return json.dumps({"error": f"Tool '{tool.name}' has no implementation"})


# ---------------------------------------------------------------------------
# ReAct system prompt builder
# ---------------------------------------------------------------------------

def _build_react_prompt(agent: Agent) -> str:
    tools_text = "\n".join(
        f"  - {t.name}: {t.description}"
        for t_name in agent.tools
        if (t := get_tool(t_name))
    ) or "No tools available."

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
) -> list[CallbackEvent]:
    """Run the ReAct loop from the given state.

    Args:
        agent: The agent to run.
        handler: Callback handler for human-in-the-loop events.
        messages: Conversation history so far.
        system_prompt: The full ReAct system prompt.
        start_iteration: Which iteration to start from (0 = fresh, >0 = resume).
        max_iterations: Maximum total iterations.

    Returns:
        Events emitted during this loop segment. The loop pauses when
        request_human_input is called — the caller must resume by calling
        this function again with updated messages.
    """
    handler.current_agent = agent.name
    events: list[CallbackEvent] = []

    for iteration in range(start_iteration, max_iterations):
        response = _call_llm(system_prompt, messages)
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
                               "Use the format:\nAction: <tool_name>\nAction Input: <JSON>"
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
                        logger.info("Delegating to '%s' with request: %s", delegate_to, original_request)
                        delegate_prompt = _build_react_prompt(delegate)
                        delegate_messages = [{"role": "user", "content": original_request}]
                        # Run the specialist agent in the same loop
                        return _react_loop(
                            delegate, handler, delegate_messages,
                            delegate_prompt, start_iteration=0, max_iterations=max_iterations,
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
    """Manages agent lifecycle and user interaction."""

    def __init__(self) -> None:
        self.handler = TravelBookingCallbackHandler()
        self.pending_events: list[CallbackEvent] = []

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

        # Check for delegation from orchestrator context
        if state.get("context", {}).get("delegate_to"):
            delegate = get_agent(state["context"]["delegate_to"])
            if delegate:
                agent = delegate
                # Start fresh messages for the delegated agent
                state["messages"] = [{"role": "user", "content": state["context"].get("original_request", user_input)}]
                state["system_prompt"] = _build_react_prompt(agent)
                state["iteration"] = 0

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
        events = list(self.pending_events)
        self.pending_events.clear()
        return events

    def reset(self) -> None:
        self.handler.reset()
        self.pending_events.clear()