from __future__ import annotations

from ..common.models import ReActEvent


class AutogenReActEngine:
    """Small ReACT tracer with autogen-like role naming."""

    def __init__(self) -> None:
        self.events: list[ReActEvent] = []

    def step(self, agent: str, thought: str, action: str, observation: str) -> None:
        self.events.append(
            ReActEvent(agent=agent, thought=thought, action=action, observation=observation)
        )
