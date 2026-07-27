"""Reusable prompt builders for adaptive time-series agents."""
from __future__ import annotations

import os

from financial_time_series_construction.agents_definition import Agent

# Detect model family from environment to adjust prompt style.
# Larger/creative models need more explicit formatting constraints;
# smaller models benefit from concise instructions.
_LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("LLM_OLLAMA_MODEL", "")).casefold()
_IS_LARGE_MODEL = any(
    tag in _LLM_MODEL
    for tag in (
        "deepseek",
        "qwen3",
        "qwen3.5",
        "qwen2.5:7",
        "qwen2.5:14",
        "qwen2.5:32",
        "qwen2.5:72",
        "llama3",
        "gpt-4",
        "gemma",
    )
)
_IS_SMALL_MODEL = any(
    tag in _LLM_MODEL
    for tag in ("1.5b", "1b", "3b", "tiny", "small")
)

REACT_PROTOCOL = """Use this protocol structure:
Thought: <brief decision rationale>
Action: <one tool name>
Action Input: <valid JSON object>

After a tool result, continue the protocol. When complete, use:
Final Answer: <concise user-facing result>

Never invent tool results. If a tool reports an error, explain it to the user
and stop or ask for the missing information. Do not expose hidden chain-of-thought.
Select actions adaptively from the current request context; do not force a rigid step order
when enough information is already available.
"""

# For larger/creative models, add explicit formatting constraints to prevent
# narrative output that doesn't follow the ReAct format.
_REACT_PROTOCOL_STRICT = """You MUST follow this exact protocol structure with no extra text:

Thought: <brief decision rationale>
Action: <one tool name>
Action Input: <valid JSON object>

After a tool result, continue the protocol. When complete, use:
Final Answer: <concise user-facing result>

CRITICAL RULES:
- Every response MUST contain either an Action/Action Input block OR a Final Answer.
- Do NOT describe what you will do - just do it.
- Use exactly one Action per response.
- Do NOT use markdown code fences around Action Input JSON.
- Do NOT output XML-like tags (for example </think> or <analysis>).
- Do NOT add conversational text before or after the protocol.
- If a tool just returned successfully, do not call the same tool again with equivalent inputs.
- Use only tool names from Available tools (exact spelling).
- Never invent tool results. If a tool reports an error, explain it to the user
  and stop or ask for the missing information.
"""

DELEGATION_EXAMPLE = """For most initial requests, the Orchestrator should delegate using:
Action: delegate_to_agent
Action Input: {"agent_name": "ReferenceDataAgent", "request": "<original user request>"}
When context is already complete for a later specialist, delegation may skip intermediate
clarification steps.
"""


def agent_system_prompt(agent: Agent) -> str:
    """Build the full system prompt for an agent including goal, tools, and guardrails.

    Selects a stricter ReAct protocol for larger/creative models that tend to
    produce narrative output instead of structured tool calls.

    Args:
        agent: The agent definition.

    Returns:
        A complete system prompt string.
    """
    tools = ", ".join(agent.tools) or "none"
    goal = agent.goal or agent.description
    guardrails = "\n".join(f"- {rule}" for rule in agent.guardrails) or "- Use only the registered tools."

    # Select protocol variant based on model size
    if _IS_LARGE_MODEL and not _IS_SMALL_MODEL:
        protocol = _REACT_PROTOCOL_STRICT
    else:
        protocol = REACT_PROTOCOL

    return (
        f"{agent.system_prompt}\n\nGoal:\n{goal}\n\n"
        f"Available tools: {tools}\nGuardrails:\n{guardrails}\n\n"
        f"{DELEGATION_EXAMPLE if agent.name == 'Orchestrator' else ''}{protocol}"
    )


def unavailable_message(resource: str, detail: str) -> str:
    """Generate a user-facing message when a resource is unavailable.

    Args:
        resource: The name of the unavailable resource.
        detail: Details about the unavailability.

    Returns:
        A formatted message string.
    """
    return f"I cannot complete this request because {resource} is unavailable. {detail}"


def request_prompt() -> str:
    """Generate the prompt asking the user for a financial time series request.

    Returns:
        The prompt string.
    """
    return (
        "What financial time series should I construct? Provide a ticker or security name and "
        "a start/end date. You can use numeric or word formats, for example: "
        "'AAPL from 2023-01-01 to 2023-12-31', 'AAPL between January 2023 and December 2023', "
        "or 'AAPL from Q1 2023 to Q4 2023'."
    )