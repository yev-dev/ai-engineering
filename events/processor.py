"""
LangChain/litellm processor for the travel booking agent system.

Implements a ReAct (Reasoning + Acting) loop for each agent using
litellm with Ollama (gemma4:e4b model). The processor:

1. Takes a user request and runs it through the Orchestrator agent
2. The Orchestrator delegates to specialist agents via events
3. Each specialist agent runs its own ReAct loop with registered tools
4. Human-in-the-loop events are emitted for user input
5. The processor manages the event loop and agent lifecycle
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
    Event,
    EventType,
    Agent,
    Tool,
    get_agent,
    get_tool,
)
from handler import process_event, process_events

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

    Supports two formats:
    1. JSON function call format: {"name": "...", "arguments": {...}}
    2. XML-style format: <function_call>name</function_call> with args

    Returns a list of dicts with 'name' and 'arguments' keys.
    """
    calls = []

    # Try to find JSON tool calls in the text
    # Pattern: {"name": "tool_name", "arguments": {...}} or similar
    json_pattern = r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        name = match.group(1)
        try:
            arguments = json.loads(match.group(2))
            calls.append({"name": name, "arguments": arguments})
        except json.JSONDecodeError:
            pass

    # Try to find function_call blocks
    # Pattern: <function_call>tool_name</function_call> with args in JSON
    func_pattern = r'<function_call>\s*(\w+)\s*</function_call>'
    for match in re.finditer(func_pattern, text):
        name = match.group(1)
        # Look for JSON arguments after the function call
        args_match = re.search(r'\{\s*"[^"]+"\s*:.*?\}', text[match.end():], re.DOTALL)
        if args_match:
            try:
                arguments = json.loads(args_match.group(0))
                calls.append({"name": name, "arguments": arguments})
            except json.JSONDecodeError:
                calls.append({"name": name, "arguments": {}})
        else:
            calls.append({"name": name, "arguments": {}})

    # Try to find Action: tool_name with Action Input: JSON
    # This is the standard ReAct format
    action_pattern = r'Action:\s*(\w+)\s*\nAction Input:\s*(\{.*?\})'
    for match in re.finditer(action_pattern, text, re.DOTALL):
        name = match.group(1)
        try:
            arguments = json.loads(match.group(2))
            calls.append({"name": name, "arguments": arguments})
        except json.JSONDecodeError:
            calls.append({"name": name, "arguments": {}})

    return calls


def _execute_tool(tool_call: dict[str, Any], agent_name: str, session_id: str) -> str:
    """Execute a tool call and return the result.

    Special handling for 'emit_event' which generates events
    rather than calling a function.
    """
    tool_name = tool_call["name"]
    arguments = tool_call.get("arguments", {})

    tool = get_tool(tool_name)
    if not tool:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    # Special handling for emit_event
    if tool_name == "emit_event":
        event_type_str = arguments.get("event_type", "")
        payload = arguments.get("payload", {})

        try:
            event_type = EventType(event_type_str)
        except ValueError:
            return json.dumps({"error": f"Unknown event type: {event_type_str}"})

        # Add agent context to payload
        payload["agent"] = agent_name

        event = Event(
            type=event_type,
            payload=payload,
            session_id=session_id,
        )

        # Process the event through the handler
        emitted = process_event(event)

        return json.dumps({
            "status": "event_emitted",
            "event_type": event_type.value,
            "emitted_events": [e.type.value for e in (emitted or [])],
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

def _build_react_prompt(agent: Agent, user_input: str, history: list[dict[str, str]]) -> str:
    """Build the full messages list for the ReAct loop."""
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
"""

    return agent.system_prompt + "\n\n" + react_instructions


def run_agent(
    agent: Agent,
    user_input: str,
    session_id: str = "default",
    max_iterations: int = 10,
) -> list[Event]:
    """Run a ReAct agent loop.

    Args:
        agent: The agent to run.
        user_input: The user's input/request.
        session_id: Session identifier.
        max_iterations: Maximum ReAct iterations.

    Returns:
        List of events emitted during the agent's execution.
    """
    logger.info("Running agent '%s' with input: %s", agent.name, user_input[:100])

    system_prompt = _build_react_prompt(agent, user_input, [])
    messages: list[dict[str, str]] = [
        {"role": "user", "content": user_input},
    ]

    all_emitted_events: list[Event] = []
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
            # Emit agent_completed event
            all_emitted_events.append(
                Event(
                    type=EventType.AGENT_COMPLETED,
                    payload={
                        "agent": agent.name,
                        "result": {"final_answer": response},
                    },
                    session_id=session_id,
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
            result = _execute_tool(tool_call, agent.name, session_id)
            messages.append({
                "role": "user",
                "content": f"Tool '{tool_call['name']}' result: {result}",
            })

            # Check if the tool emitted events (from emit_event)
            try:
                result_data = json.loads(result)
                if result_data.get("status") == "event_emitted":
                    # The event was already processed by the handler
                    pass
            except (json.JSONDecodeError, TypeError):
                pass

    return all_emitted_events


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

class TravelBookingProcessor:
    """Main processor for the travel booking system.

    Manages the event loop, agent lifecycle, and user interaction.
    """

    def __init__(self) -> None:
        self.session_id: str = "default"
        self.active_agent: str | None = None
        self.pending_events: list[Event] = []
        self.conversation_history: list[dict[str, str]] = []

    def process_user_request(self, user_input: str) -> list[Event]:
        """Process a user request through the orchestrator agent.

        Args:
            user_input: The user's natural language request.

        Returns:
            List of events emitted during processing.
        """
        logger.info("Processing user request: %s", user_input[:100])

        # Store in conversation history
        self.conversation_history.append({"role": "user", "content": user_input})

        # Run the orchestrator agent
        orchestrator = get_agent("Orchestrator")
        if not orchestrator:
            error_event = Event(
                type=EventType.ERROR,
                payload={"message": "Orchestrator agent not found"},
                session_id=self.session_id,
            )
            process_event(error_event)
            return [error_event]

        events = run_agent(
            agent=orchestrator,
            user_input=user_input,
            session_id=self.session_id,
        )

        self.pending_events.extend(events)
        return events

    def process_user_response(self, user_input: str, context: dict[str, Any] | None = None) -> list[Event]:
        """Process a user response to a human-in-the-loop prompt.

        Args:
            user_input: The user's response text.
            context: Optional context from the prompt event.

        Returns:
            List of events emitted during processing.
        """
        logger.info("Processing user response: %s", user_input[:100])

        self.conversation_history.append({"role": "user", "content": user_input})

        # Determine which agent should handle the response based on context
        agent_name = "Orchestrator"
        if context:
            agent_name = context.get("agent", "Orchestrator")

        # If there's a pending event type, handle it
        if context and context.get("event_type") == EventType.SELECT_CAR.value:
            # User is responding to a car selection prompt
            # The response should be the car type they want to confirm
            car_id = context.get("car_id", "")
            car_type = user_input.strip()

            # Create a car_type_confirmed event
            confirm_event = Event(
                type=EventType.CAR_TYPE_CONFIRMED,
                payload={
                    "car_id": car_id,
                    "car_type": car_type,
                },
                session_id=self.session_id,
            )

            # Process the confirmation
            emitted = process_event(confirm_event)
            self.pending_events.extend(emitted or [])

            # Now run the CarBookingAgent with the confirmation
            car_agent = get_agent("CarBookingAgent")
            if car_agent:
                agent_events = run_agent(
                    agent=car_agent,
                    user_input=f"The user confirmed car type '{car_type}' for car {car_id}. "
                               f"Please finalise the booking.",
                    session_id=self.session_id,
                )
                self.pending_events.extend(agent_events)
                return agent_events

        # For general responses, run the orchestrator or the active agent
        active_agent = get_agent(agent_name) or get_agent("Orchestrator")
        if active_agent:
            events = run_agent(
                agent=active_agent,
                user_input=user_input,
                session_id=self.session_id,
            )
            self.pending_events.extend(events)
            return events

        return []

    def get_pending_events(self) -> list[Event]:
        """Get and clear pending events."""
        events = list(self.pending_events)
        self.pending_events.clear()
        return events

    def reset(self) -> None:
        """Reset the processor state."""
        self.session_id = "default"
        self.active_agent = None
        self.pending_events.clear()
        self.conversation_history.clear()