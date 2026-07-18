"""
Callback handler for the travel booking system.

Extends ``langchain_core.callbacks.base.BaseCallbackHandler`` so it can be
registered with **litellm** via the ``callbacks`` parameter.  litellm
natively supports LangChain callback handlers.

The handler is the bridge between the ReAct agent loop (processor.py)
and the user interface (cli.py) — it:
  - Maintains an event queue for agent → UI communication
  - Manages human-in-the-loop state (pause/resume)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentFinish

from agents import CallbackEvent, CallbackEventType

logger = logging.getLogger(__name__)


class TravelBookingCallbackHandler(BaseCallbackHandler):
    """LangChain-compatible callback handler for human-in-the-loop events.

    litellm accepts LangChain callback handlers natively — pass an instance
    via ``LLMRequest(callbacks=[handler])`` to receive LLM and tool events.
    """

    def __init__(self) -> None:
        super().__init__()
        self.event_queue: deque[CallbackEvent] = deque()
        self.waiting_for_input: bool = False
        self.paused_state: dict[str, Any] | None = None
        self.current_agent: str | None = None
        self.session_id: str = "default"

    # -------------------------------------------------------------------
    # Event queue
    # -------------------------------------------------------------------

    def emit(self, event: CallbackEvent) -> None:
        """Push an event onto the queue."""
        self.event_queue.append(event)

    def poll(self) -> CallbackEvent | None:
        """Pop the oldest event, or return ``None`` if the queue is empty."""
        return self.event_queue.popleft() if self.event_queue else None

    def has_events(self) -> bool:
        return len(self.event_queue) > 0

    def clear_events(self) -> None:
        self.event_queue.clear()

    # -------------------------------------------------------------------
    # Human-in-the-loop
    # -------------------------------------------------------------------

    def request_human_input(
        self,
        prompt: str,
        options: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Emit an ``AWAITING_USER_INPUT`` event and flag the loop to pause."""
        payload: dict[str, Any] = {
            "prompt": prompt,
            "agent": self.current_agent or "System",
        }
        if options:
            payload["options"] = options
        if context:
            payload["context"] = context

        self.emit(CallbackEvent(CallbackEventType.AWAITING_USER_INPUT, payload))
        self.waiting_for_input = True

    def handle_user_response(self, user_input: str) -> dict[str, Any] | None:
        """Process user response and return saved state for agent resume.

        Returns ``None`` if the user cancelled.
        """
        self.waiting_for_input = False
        state = self.paused_state
        self.paused_state = None

        if user_input.lower() in ("cancel", "quit", "exit"):
            self.emit(CallbackEvent(
                CallbackEventType.BOOKING_FAILED,
                {"reason": "User cancelled the operation."},
            ))
            return None

        return state

    # -------------------------------------------------------------------
    # LangChain / litellm callback overrides
    # -------------------------------------------------------------------

    def on_llm_error(self, error: Exception | KeyboardInterrupt, **kwargs: Any) -> None:
        logger.error("LLM error: %s", error)
        self.emit(CallbackEvent(CallbackEventType.ERROR, {"message": f"LLM error: {error}"}))

    def on_tool_error(self, error: Exception | KeyboardInterrupt, **kwargs: Any) -> None:
        logger.error("Tool error: %s", error)
        self.emit(CallbackEvent(CallbackEventType.ERROR, {"message": f"Tool error: {error}"}))

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        self.emit(CallbackEvent(
            CallbackEventType.AGENT_COMPLETED,
            {
                "agent": self.current_agent or "Unknown",
                "result": {"final_answer": finish.return_values.get("output", "")},
            },
        ))

    # -------------------------------------------------------------------
    # Session
    # -------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all state for a fresh conversation."""
        self.event_queue.clear()
        self.waiting_for_input = False
        self.paused_state = None
        self.current_agent = None