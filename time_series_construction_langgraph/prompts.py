"""Reusable, deterministic prompt builders for the time-series agents."""
from __future__ import annotations

try:
    from .agents_definition import Agent
except ImportError:
    from agents_definition import Agent


REACT_PROTOCOL = """Use exactly this protocol:
Thought: <brief decision rationale>
Action: <one tool name>
Action Input: <valid JSON object>

After a tool result, continue the protocol. When complete, use:
Final Answer: <concise user-facing result>

Never invent tool results. If a tool reports an error, explain it to the user
and stop or ask for the missing information. Do not expose hidden chain-of-thought.
"""

DELEGATION_EXAMPLE = """For the initial request, the Orchestrator must call this exact tool:
Action: delegate_to_agent
Action Input: {"agent_name": "ReferenceDataAgent", "request": "<original user request>"}
The Orchestrator must never return a Final Answer before delegation.
"""


def agent_system_prompt(agent: Agent) -> str:
    tools = ", ".join(agent.tools) or "none"
    goal = agent.goal or agent.description
    guardrails = "\n".join(f"- {rule}" for rule in agent.guardrails) or "- Use only the registered tools."
    return (
        f"{agent.system_prompt}\n\nGoal:\n{goal}\n\n"
        f"Available tools: {tools}\nGuardrails:\n{guardrails}\n\n"
        f"{DELEGATION_EXAMPLE if agent.name == 'Orchestrator' else ''}{REACT_PROTOCOL}"
    )


def unavailable_message(resource: str, detail: str) -> str:
    return f"I cannot complete this request because {resource} is unavailable. {detail}"


def request_prompt() -> str:
    return "What financial time series should I construct? Provide a ticker or security name and a start/end date."