"""
LangChain callback handler for the travel booking agent system.

Implements a custom BaseCallbackHandler that:
  - Intercepts human-in-the-loop events from ReAct agents
  - Manages the callback event queue
  - Provides prompt templates for user selection
  - Handles car type confirmation flow

Uses litellm + Ollama (gemma4:e4b) for LLM-driven processing.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable
from collections import deque

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult

from agents import (
    AGENT_REGISTRY,
    TOOL_REGISTRY,
    CallbackEvent,
    CallbackEventType,
    Agent,
    Tool,
    get_agent,
    get_tool,
    CAR_TYPE_SELECTION_PROMPT,
    FLIGHT_SELECTION_PROMPT,
    HOTEL_SELECTION_PROMPT,
    GENERAL_INPUT_PROMPT,
    CONFIRMATION_PROMPT,
    format_car_options,
    format_flight_options,
    format_hotel_options,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom LangChain Callback Handler
# ---------------------------------------------------------------------------

class TravelBookingCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler for travel booking human-in-the-loop events.

    This handler intercepts agent actions and manages the event queue
    for user interaction. It is the bridge between the ReAct agent loop
    and the CLI/user interface.
    """

    def __init__(self) -> None:
        """Initialize the callback handler."""
        super().__init__()
        self.event_queue: deque[CallbackEvent] = deque()
        self.waiting_for_input: bool = False
        self.pending_context: dict[str, Any] | None = None
        self.current_agent: str | None = None
        self.session_id: str = "default"
        self.conversation_history: list[dict[str, str]] = []

    # -----------------------------------------------------------------------
    # Event queue management
    # -----------------------------------------------------------------------

    def emit_event(self, event: CallbackEvent) -> None:
        """Emit a callback event to the queue."""
        logger.debug("Event emitted: %s", event)
        self.event_queue.append(event)

    def get_next_event(self) -> CallbackEvent | None:
        """Get the next event from the queue (non-blocking)."""
        if self.event_queue:
            return self.event_queue.popleft()
        return None

    def has_events(self) -> bool:
        """Check if there are pending events."""
        return len(self.event_queue) > 0

    def clear_events(self) -> None:
        """Clear all pending events."""
        self.event_queue.clear()

    # -----------------------------------------------------------------------
    # Human-in-the-loop handling
    # -----------------------------------------------------------------------

    def request_human_input(
        self,
        prompt: str,
        options: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Request input from the human user.

        This emits an AWAITING_USER_INPUT event that the CLI layer
        will pick up to prompt the user.

        Args:
            prompt: The question or message to show the user.
            options: Optional list of selectable options.
            context: Additional context data to pass through.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "agent": self.current_agent or "System",
            "session_id": self.session_id,
        }
        if options:
            payload["options"] = options
        if context:
            payload["context"] = context

        self.emit_event(
            CallbackEvent(
                type=CallbackEventType.AWAITING_USER_INPUT,
                payload=payload,
                session_id=self.session_id,
            )
        )
        self.waiting_for_input = True
        self.pending_context = {
            "prompt": prompt,
            "options": options,
            "context": context or {},
            "agent": self.current_agent or "System",
        }

    def handle_user_response(self, user_input: str) -> dict[str, Any] | None:
        """Handle a user response to a human-in-the-loop prompt.

        Args:
            user_input: The user's response text.

        Returns:
            Context dict if there's more to process, or None if done.
        """
        self.waiting_for_input = False
        context = self.pending_context
        self.pending_context = None

        # Store in conversation history
        self.conversation_history.append({"role": "user", "content": user_input})

        # If the user cancelled
        if user_input.lower() in ("cancel", "quit", "exit"):
            self.emit_event(
                CallbackEvent(
                    type=CallbackEventType.BOOKING_FAILED,
                    payload={"reason": "User cancelled the operation."},
                    session_id=self.session_id,
                )
            )
            return None

        # Return the context so the processor can continue the agent loop
        return context

    # -----------------------------------------------------------------------
    # LangChain callback overrides
    # -----------------------------------------------------------------------

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        """Called when the LLM starts running."""
        logger.debug("LLM start: %s", prompts[0][:100] if prompts else "")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when the LLM finishes running."""
        logger.debug("LLM end")

    def on_llm_error(
        self, error: Exception | KeyboardInterrupt, **kwargs: Any
    ) -> None:
        """Called when the LLM encounters an error."""
        logger.error("LLM error: %s", error)
        self.emit_event(
            CallbackEvent(
                type=CallbackEventType.ERROR,
                payload={"message": f"LLM error: {error}"},
                session_id=self.session_id,
            )
        )

    def on_agent_action(
        self, action: AgentAction, **kwargs: Any
    ) -> None:
        """Called when an agent takes an action."""
        logger.debug("Agent action: %s", action)

    def on_agent_finish(
        self, finish: AgentFinish, **kwargs: Any
    ) -> None:
        """Called when an agent finishes."""
        logger.debug("Agent finish: %s", finish)
        self.emit_event(
            CallbackEvent(
                type=CallbackEventType.AGENT_COMPLETED,
                payload={
                    "agent": self.current_agent or "Unknown",
                    "result": {"final_answer": finish.return_values.get("output", "")},
                },
                session_id=self.session_id,
            )
        )

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Called when a tool starts running."""
        logger.debug("Tool start: %s - %s", serialized.get("name", ""), input_str[:100])

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Called when a tool finishes running."""
        logger.debug("Tool end: %s", output[:100])

    def on_tool_error(
        self, error: Exception | KeyboardInterrupt, **kwargs: Any
    ) -> None:
        """Called when a tool encounters an error."""
        logger.error("Tool error: %s", error)
        self.emit_event(
            CallbackEvent(
                type=CallbackEventType.ERROR,
                payload={"message": f"Tool error: {error}"},
                session_id=self.session_id,
            )
        )

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Called when the agent outputs text."""
        logger.debug("Agent text: %s", text[:200])

    # -----------------------------------------------------------------------
    # Session management
    # -----------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the handler state."""
        self.event_queue.clear()
        self.waiting_for_input = False
        self.pending_context = None
        self.current_agent = None
        self.conversation_history.clear()


# ---------------------------------------------------------------------------
# Event processing helpers
# ---------------------------------------------------------------------------

def process_callback_event(event: CallbackEvent) -> list[CallbackEvent]:
    """Process a single callback event.

    This is a simple dispatcher that routes events to the appropriate
    handling logic. For most events, it just logs and returns the event
    for the CLI to handle.

    Args:
        event: The callback event to process.

    Returns:
        List of emitted events (may be empty).
    """
    logger.debug("Processing callback event: %s", event.type)

    if event.type == CallbackEventType.AWAITING_USER_INPUT:
        # The CLI will handle this - just pass it through
        return [event]

    elif event.type == CallbackEventType.SELECT_CAR:
        # The CLI will handle this - just pass it through
        return [event]

    elif event.type == CallbackEventType.CAR_TYPE_CONFIRMED:
        car_id = event.payload.get("car_id", "unknown")
        car_type = event.payload.get("car_type", "")
        logger.info("Car type confirmed: car_id=%s, car_type=%s", car_id, car_type)
        return [event]

    elif event.type == CallbackEventType.AGENT_COMPLETED:
        agent_name = event.payload.get("agent", "Unknown")
        logger.info("Agent '%s' completed.", agent_name)
        return [event]

    elif event.type == CallbackEventType.BOOKING_CONFIRMED:
        details = event.payload.get("details", {})
        logger.info("Booking confirmed: %s", details)
        return [event]

    elif event.type == CallbackEventType.BOOKING_FAILED:
        reason = event.payload.get("reason", "Unknown error")
        logger.error("Booking failed: %s", reason)
        return [event]

    elif event.type == CallbackEventType.ERROR:
        message = event.payload.get("message", "Unknown error")
        logger.error("Error: %s", message)
        return [event]

    return []


def process_callback_events(events: list[CallbackEvent]) -> list[CallbackEvent]:
    """Process a batch of callback events.

    Args:
        events: List of events to process.

    Returns:
        All events emitted during processing.
    """
    all_emitted: list[CallbackEvent] = []
    for event in events:
        emitted = process_callback_event(event)
        if emitted:
            all_emitted.extend(emitted)
    return all_emitted