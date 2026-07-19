from __future__ import annotations

from ..common.models import ReActEvent


class LangChainReActEngine:
    """Small ReACT tracer with langchain-like node/tool labels."""

    def __init__(self) -> None:
        self.events: list[ReActEvent] = []

    def node(self, thought: str, tool: str, observation: str) -> None:
        self.events.append(
            ReActEvent(
                agent="LangGraphNode",
                thought=thought,
                action=f"Tool: {tool}",
                observation=observation,
            )
        )
