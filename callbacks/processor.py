"""
LangChain/litellm processor for the travel booking agent system using callbacks.

Implements a ReAct (Reasoning + Acting) loop for each agent using
litellm with Ollama (gemma4:e4b model). The processor:

1. Takes a user request and runs it through the Orchestrator agent
2. The Orchestrator delegates to specialist agents via callbacks
3. Each specialist agent runs its own ReAct loop with registered tools
4. Human-in-the-loop callbacks are emitted for user input
5. The processor manages the callback event loop and agent lifecycle

Uses LangChain's callback system for human-in-the-loop interaction.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from litellm import completion

from agents import (
    AGENT_REGISTRY,
    TOOL_REGISTRY,
    CallbackEvent,
    CallbackEventType,
    Agent,
    Tool,
    get_agent,
    get_tool,
)
from handler import TravelBookingCallbackHandler, process_callback_event

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

# litellm configuration
os.environ["LITELLM_LOG"] = "WARNING"


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

def _call_llm(
    system_prompt: str,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> str:
    """Call the LLM via litellm with Ollama backend.

    Args:
        system_prompt: The system prompt for the agent.
        messages: The conversation history.
        tools: Optional list of tool definitions in OpenAI format.
        temperature: LLM temperature.
        max_tokens: Maximum tokens in response.

    Returns:
        The LLM response content as a string.
    """
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    kwargs: dict[str, Any] = {
        "model": f"ollama/{OLLAMA_MODEL}",
        "messages": full_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_base": OLLAMA_ENDPOINT,
    }

    if tools:
        kwargs["tools"] = tools

    try:
        response = completion(**kwargs)
        content = response.choices[0].message.content or ""
        return content
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return f"Error: LLM call failed - {e}"


# ---------------------------------------------------------------------------
# Tool format conversion (OpenAI tool format for litellm)
# ---------------------------------------------------------------------------

def _tool_to_openai_format(tool: Tool) -> dict[str, Any]:
    """Convert a Tool to OpenAI-compatible tool format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _get_agent_tools(agent: Agent) -> list[dict[str, Any]]:
    """Get the OpenAI-formatted tool list for an agent."""
    tools = []
    for tool_name in agent.tools:
        tool = get_tool(tool_name)
        if tool:
            tools.append(_tool_to_openai_format(tool))
    return tools


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------

def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tool calls from LLM response text.

    Supports multiple formats:
    1. JSON function call format: {"name": "...", "arguments": {...}}
    2. XML-style format: <function_call>name</function_call> with args
    3. ReAct format: Action: tool_name\\nAction Input: JSON

    Returns a list of dicts with 'name' and 'arguments' keys.
    """
    calls = []

    # Try to find JSON tool calls in the text
    json_pattern = r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        name = match.group(1)
        try:
            arguments = json.loads(match.group(2))
            calls.append({"name": name, "arguments": arguments})
        except json.JSONDecodeError:
            pass

    # Try to find function_call blocks
    func_pattern = r'<function_call>\s*(\w+)\s*</function_call>'
    for match in re.finditer(func_pattern, text):
        name = match.group(1)
        args_match = re.search(r'\{\s*"[^"]+"\s*:.*?\}', text[match.end():], re.DOTALL)
        if args_match:
            try:
                arguments = json.loads(args_match.group(0))
                calls.append({"name": name, "arguments": arguments})
            except json.JSONDecodeError:
                calls.append({"name": name, "arguments": {}})
        else:
            calls.append({"name": name, "arguments": {}})

    # Try to find Action: tool_name with Action Input: JSON (ReAct format)
    action_pattern = r'Action:\s*(\w+)\s*\nAction Input:\s*(\{.*?\})'
    for match in re.finditer(action_pattern, text, re.DOTALL):
        name = match.group(1)
        try:
            arguments = json.loads(match.group(2))
            calls.append({"name": name, "arguments": arguments})
        except json.JSONDecodeError:
            calls.append({"name": name, "arguments": {}})

    return calls


def _execute_tool(
    tool_call: dict[str, Any],
    agent_name: str,
    session_id: str,
    callback_handler: TravelBookingCallbackHandler,
) -> str:
    """Execute a tool call and return the result.

    Special handling for 'request_human_input' which triggers
    the callback handler's human-in-the-loop mechanism.
    """
    tool_name = tool_call["name"]
    arguments = tool_call.get("arguments", {})

    tool = get_tool(tool_name)
    if not tool:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    # Special handling for request_human_input
    if tool_name == "request_human_input":
        prompt = arguments.get("prompt", "Please provide input:")
        options = arguments.get("options", None)
        context = arguments.get("context", {})

        # Add agent context
        context["agent"] = agent_name

        # Trigger the callback handler's human-in-the-loop
        callback_handler.request_human_input(
            prompt=prompt,
            options=options,
            context=context,
        )

        return json.dumps({
            "status": "awaiting_human_input",
            "prompt": prompt,
            "options": options,
            "message": "Waiting for user input via callback handler.",
        })

    # Execute the tool function
    if tool.fn:
        try:
            result = tool.fn(**arguments)
            return result
        except Exception as e:
            logger.error("Tool '%s' execution failed: %s", tool_name, e)
            return json.dumps({"error": f"Tool execution failed: {e}"})
    else:
        return json.dumps({"error": f"Tool '{tool_name}' has no implementation"})


# ---------------------------------------------------------------------------
# ReAct agent loop
# ---------------------------------------------------------------------------

def _build_react_prompt(agent: Agent) -> str:
    """Build the full system prompt for the ReAct loop."""
    tool_descriptions = []
    for tool_name in agent.tools:
        tool = get_tool(tool_name)
        if tool:
            tool_descriptions.append(
                f"  - {tool.name}: {tool.description}"
            )

    tools_section = "\n".join(tool_descriptions) if tool_descriptions else "No tools available."

    react_instructions = f"""
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
{tools_section}

IMPORTANT: Always use the Action/Action Input format to call tools.
NEVER make up results. Always use tools to get real data.
When you need information from the user, use the `request_human_input` tool.
"""

    return agent.system_prompt + "\n\n" + react_instructions


def run_agent(
    agent: Agent,
    user_input: str,
    callback_handler: TravelBookingCallbackHandler,
    max_iterations: int = 10,
) -> list[CallbackEvent]:
    """Run a ReAct agent loop.

    Args:
        agent: The agent to run.
        user_input: The user's input/request.
        callback_handler: The callback handler for human-in-the-loop events.
        max_iterations: Maximum ReAct iterations.

    Returns:
        List of events emitted during the agent's execution.
    """
    logger.info("Running agent '%s' with input: %s", agent.name, user_input[:100])

    # Set the current agent on the callback handler
    callback_handler.current_agent = agent.name

    system_prompt = _build_react_prompt(agent)
    messages: list[dict[str, str]] = [
        {"role": "user", "content": user_input},
    ]

    all_emitted_events: list[CallbackEvent] = []
    tools_openai = _get_agent_tools(agent)

    for iteration in range(max_iterations):
        logger.debug("Agent '%s' iteration %d", agent.name, iteration)

        # Call the LLM
        response = _call_llm(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools_openai if tools_openai else None,
        )

        messages.append({"role": "assistant", "content": response})
        logger.debug("Agent '%s' response: %s", agent.name, response[:200])

        # Check for Final Answer
        if "Final Answer:" in response:
            logger.info("Agent '%s' completed with final answer.", agent.name)
            # Emit agent_completed event via callback
            all_emitted_events.append(
                CallbackEvent(
                    type=CallbackEventType.AGENT_COMPLETED,
                    payload={
                        "agent": agent.name,
                        "result": {"final_answer": response},
                    },
                    session_id=callback_handler.session_id,
                )
            )
            break

        # Parse tool calls
        tool_calls = _parse_tool_calls(response)

        if not tool_calls:
            # No tool call found, check if we should continue
            if iteration < max_iterations - 1:
                # Ask the LLM to continue with a tool call
                messages.append({
                    "role": "user",
                    "content": "Please use one of the available tools to proceed. "
                               "Use the format:\nAction: <tool_name>\nAction Input: <JSON>"
                })
                continue
            else:
                logger.warning("Agent '%s' reached max iterations without completing.", agent.name)
                break

        # Execute each tool call
        for tool_call in tool_calls:
            result = _execute_tool(
                tool_call,
                agent.name,
                callback_handler.session_id,
                callback_handler,
            )
            messages.append({
                "role": "user",
                "content": f"Tool '{tool_call['name']}' result: {result}",
            })

            # Check if the tool triggered a human-in-the-loop callback
            try:
                result_data = json.loads(result)
                if result_data.get("status") == "awaiting_human_input":
                    # The callback handler is now waiting for user input
                    # We need to pause the agent loop and wait
                    logger.info("Agent '%s' is waiting for human input.", agent.name)
                    # Store the current state so we can resume later
                    callback_handler.pending_context = {
                        "agent": agent.name,
                        "messages": messages,
                        "iteration": iteration,
                        "system_prompt": system_prompt,
                    }
                    return all_emitted_events
            except (json.JSONDecodeError, TypeError):
                pass

    return all_emitted_events


def resume_agent(
    agent: Agent,
    user_input: str,
    callback_handler: TravelBookingCallbackHandler,
    max_iterations: int = 10,
) -> list[CallbackEvent]:
    """Resume a ReAct agent loop after a human-in-the-loop pause.

    Args:
        agent: The agent to resume.
        user_input: The user's response to the human-in-the-loop prompt.
        callback_handler: The callback handler with saved state.
        max_iterations: Maximum remaining ReAct iterations.

    Returns:
        List of events emitted during the agent's execution.
    """
    logger.info("Resuming agent '%s' with user response: %s", agent.name, user_input[:100])

    # Restore saved state from the callback handler
    saved_state = callback_handler.pending_context or {}
    messages: list[dict[str, str]] = saved_state.get("messages", [
        {"role": "user", "content": user_input},
    ])
    start_iteration = saved_state.get("iteration", 0)
    system_prompt = saved_state.get("system_prompt", _build_react_prompt(agent))

    # Add the user's response to the conversation
    messages.append({
        "role": "user",
        "content": f"User response: {user_input}",
    })

    all_emitted_events: list[CallbackEvent] = []
    tools_openai = _get_agent_tools(agent)

    for iteration in range(start_iteration + 1, max_iterations):
        logger.debug("Agent '%s' iteration %d (resumed)", agent.name, iteration)

        # Call the LLM
        response = _call_llm(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools_openai if tools_openai else None,
        )

        messages.append({"role": "assistant", "content": response})
        logger.debug("Agent '%s' response: %s", agent.name, response[:200])

        # Check for Final Answer
        if "Final Answer:" in response:
            logger.info("Agent '%s' completed with final answer.", agent.name)
            all_emitted_events.append(
                CallbackEvent(
                    type=CallbackEventType.AGENT_COMPLETED,
                    payload={
                        "agent": agent.name,
                        "result": {"final_answer": response},
                    },
                    session_id=callback_handler.session_id,
                )
            )
            break

        # Parse tool calls
        tool_calls = _parse_tool_calls(response)

        if not tool_calls:
            if iteration < max_iterations - 1:
                messages.append({
                    "role": "user",
                    "content": "Please use one of the available tools to proceed. "
                               "Use the format:\nAction: <tool_name>\nAction Input: <JSON>"
                })
                continue
            else:
                logger.warning("Agent '%s' reached max iterations without completing.", agent.name)
                break

        # Execute each tool call
        for tool_call in tool_calls:
            result = _execute_tool(
                tool_call,
                agent.name,
                callback_handler.session_id,
                callback_handler,
            )
            messages.append({
                "role": "user",
                "content": f"Tool '{tool_call['name']}' result: {result}",
            })

            # Check if the tool triggered another human-in-the-loop callback
            try:
                result_data = json.loads(result)
                if result_data.get("status") == "awaiting_human_input":
                    logger.info("Agent '%s' is waiting for human input again.", agent.name)
                    callback_handler.pending_context = {
                        "agent": agent.name,
                        "messages": messages,
                        "iteration": iteration,
                        "system_prompt": system_prompt,
                    }
                    return all_emitted_events
            except (json.JSONDecodeError, TypeError):
                pass

    return all_emitted_events


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

class TravelBookingProcessor:
    """Main processor for the travel booking system.

    Manages the callback event loop, agent lifecycle, and user interaction
    using LangChain's callback system.
    """

    def __init__(self) -> None:
        self.callback_handler = TravelBookingCallbackHandler()
        self.active_agent: str | None = None
        self.pending_events: list[CallbackEvent] = []

    def process_user_request(self, user_input: str) -> list[CallbackEvent]:
        """Process a user request through the orchestrator agent.

        Args:
            user_input: The user's natural language request.

        Returns:
            List of events emitted during processing.
        """
        logger.info("Processing user request: %s", user_input[:100])

        # Store in conversation history
        self.callback_handler.conversation_history.append(
            {"role": "user", "content": user_input}
        )

        # Run the orchestrator agent
        orchestrator = get_agent("Orchestrator")
        if not orchestrator:
            error_event = CallbackEvent(
                type=CallbackEventType.ERROR,
                payload={"message": "Orchestrator agent not found"},
                session_id=self.callback_handler.session_id,
            )
            return [error_event]

        events = run_agent(
            agent=orchestrator,
            user_input=user_input,
            callback_handler=self.callback_handler,
        )

        self.pending_events.extend(events)
        return events

    def process_user_response(self, user_input: str) -> list[CallbackEvent]:
        """Process a user response to a human-in-the-loop prompt.

        Args:
            user_input: The user's response text.

        Returns:
            List of events emitted during processing.
        """
        logger.info("Processing user response: %s", user_input[:100])

        # Handle the user response through the callback handler
        context = self.callback_handler.handle_user_response(user_input)
        if context is None:
            # User cancelled or there was an issue
            return list(self.callback_handler.event_queue)

        # Determine which agent should handle the response
        agent_name = context.get("agent", "Orchestrator")
        agent_context = context.get("context", {})

        # Check if there's a delegate_to in the context (from orchestrator)
        delegate_to = agent_context.get("delegate_to")
        if delegate_to:
            agent_name = delegate_to

        agent = get_agent(agent_name)
        if not agent:
            agent = get_agent("Orchestrator")

        if agent:
            # Resume the agent with the user's response
            events = resume_agent(
                agent=agent,
                user_input=user_input,
                callback_handler=self.callback_handler,
            )
            self.pending_events.extend(events)
            return events

        return []

    def get_pending_events(self) -> list[CallbackEvent]:
        """Get and clear pending events."""
        events = list(self.pending_events)
        self.pending_events.clear()
        return events

    def reset(self) -> None:
        """Reset the processor state."""
        self.callback_handler.reset()
        self.active_agent = None
        self.pending_events.clear()