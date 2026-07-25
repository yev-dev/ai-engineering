"""Reusable prompt builders for adaptive time-series agents."""
from __future__ import annotations

from financial_time_series_construction.agents_definition import Agent

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

DELEGATION_EXAMPLE = """For most initial requests, the Orchestrator should delegate using:
Action: delegate_to_agent
Action Input: {"agent_name": "ReferenceDataAgent", "request": "<original user request>"}
When context is already complete for a later specialist, delegation may skip intermediate
clarification steps.
"""


def agent_system_prompt(agent: Agent) -> str:
    """Build the full system prompt for an agent including goal, tools, and guardrails.

    Args:
        agent: The agent definition.

    Returns:
        A complete system prompt string.
    """
    tools = ", ".join(agent.tools) or "none"
    goal = agent.goal or agent.description
    guardrails = "\n".join(f"- {rule}" for rule in agent.guardrails) or "- Use only the registered tools."
    return (
        f"{agent.system_prompt}\n\nGoal:\n{goal}\n\n"
        f"Available tools: {tools}\nGuardrails:\n{guardrails}\n\n"
        f"{DELEGATION_EXAMPLE if agent.name == 'Orchestrator' else ''}{REACT_PROTOCOL}"
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