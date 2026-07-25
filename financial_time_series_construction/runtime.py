"""Runtime abstraction for pluggable agentic frameworks.

This module introduces an Adapter + Factory design so the CLI can run against
multiple agentic frameworks in the future (autogen, crawl, etc.) without
rewriting orchestration UX code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from financial_time_series_construction.handler import TimeSeriesConstructionHandler
from financial_time_series_construction.models import ModelRequestFactory
from financial_time_series_construction.processor import TimeSeriesConstructionProcessor


class AgenticRuntime(ABC):
    """Framework-agnostic runtime interface used by the CLI."""

    @property
    @abstractmethod
    def session_id(self) -> str:
        """Return runtime session id."""

    @property
    @abstractmethod
    def waiting_for_input(self) -> bool:
        """Return whether runtime is paused for human input."""

    @property
    @abstractmethod
    def current_agent(self) -> str | None:
        """Return current active/paused agent name."""

    @abstractmethod
    def process_user_request(self, user_input: str) -> list[Any]:
        """Process initial request and return callback events."""

    @abstractmethod
    def process_user_response(self, user_input: str) -> list[Any]:
        """Resume paused workflow with user input and return callback events."""

    @abstractmethod
    def get_trace(self) -> str:
        """Return trace text."""

    @abstractmethod
    def get_trace_records(self) -> list[dict[str, Any]]:
        """Return structured trace records."""

    @abstractmethod
    def trace_line_count(self) -> int:
        """Return number of trace lines."""


class AutogenProcessorRuntime(AgenticRuntime):
    """Adapter from current processor/handler implementation to AgenticRuntime."""

    def __init__(
        self,
        session_id: str,
        factory: ModelRequestFactory | None = None,
    ) -> None:
        self.handler = TimeSeriesConstructionHandler(session_id=session_id)
        self.processor = TimeSeriesConstructionProcessor(factory=factory, handler=self.handler)

    @property
    def session_id(self) -> str:
        return self.handler.session_id

    @property
    def waiting_for_input(self) -> bool:
        return self.handler.waiting_for_input

    @property
    def current_agent(self) -> str | None:
        return self.handler.current_agent

    def process_user_request(self, user_input: str) -> list[Any]:
        return self.processor.process_user_request(user_input)

    def process_user_response(self, user_input: str) -> list[Any]:
        return self.processor.process_user_response(user_input)

    def get_trace(self) -> str:
        return self.handler.get_trace()

    def get_trace_records(self) -> list[dict[str, Any]]:
        return self.handler.get_trace_records()

    def trace_line_count(self) -> int:
        return len(self.handler.react_trace)


def build_runtime(
    framework: str,
    session_id: str,
    factory: ModelRequestFactory | None = None,
) -> AgenticRuntime:
    """Build a runtime by framework name.

    Supported now:
    - autogen: current processor/handler stack.

    Reserved for future adapters:
    - crawl
    """
    key = (framework or "autogen").strip().casefold()
    if key == "autogen":
        return AutogenProcessorRuntime(session_id=session_id, factory=factory)

    if key == "crawl":
        raise NotImplementedError(
            "Framework 'crawl' is not wired yet. Add a CrawlRuntime adapter in runtime.py "
            "that implements AgenticRuntime."
        )

    raise NotImplementedError(
        f"Unknown framework '{framework}'. Supported: autogen, crawl"
    )
