"""Callback event bus and human-in-the-loop pause/resume state."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from collections import deque
from typing import Any

from .agents_definition import CallbackEvent, CallbackEventType

logger = logging.getLogger(__name__)


class TimeSeriesConstructionHandler:
    """Callback handler for time series construction workflow.

    Manages event queue, human-in-the-loop pause/resume, and error handling.
    Designed to be extensible for future handler implementations.
    Follows the callback architecture pattern similar to BaseCallbackHandler
    but adapted for autogen v0.14.x compatibility.
    """

    def __init__(self, session_id: str = "default") -> None:
        self.event_queue: deque[CallbackEvent] = deque()
        self.waiting_for_input = False
        self.paused_state: dict[str, Any] | None = None
        self.current_agent: str | None = None
        self.session_id = session_id
        self.react_trace: list[str] = []
        self.trace_records: list[dict[str, Any]] = []
        logger.info("handler_initialized session_id=%s", session_id)

    def add_trace_record(
        self,
        entry_type: str,
        payload: dict[str, Any],
        agent: str | None = None,
        iteration: int | None = None,
    ) -> None:
        """Append a structured trace record for persistence and debugging."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": entry_type,
            "agent": agent or self.current_agent,
            "iteration": iteration,
            "payload": payload,
        }
        self.trace_records.append(record)
        # Mirror structured records into plain-text trace so react_trace.txt
        # includes every recorded reasoning/decision step.
        self.react_trace.append(json.dumps(record, default=str))
        logger.debug(
            "trace_record_added session_id=%s type=%s agent=%s",
            self.session_id,
            entry_type,
            record.get("agent"),
        )

    def emit(self, event: CallbackEvent) -> None:
        """Emit a callback event to the queue."""
        self.event_queue.append(event)
        logger.info("event=%s agent=%s", event.type.value, event.payload.get("agent"))

    def poll(self) -> CallbackEvent | None:
        """Poll the next event from the queue."""
        event = self.event_queue.popleft() if self.event_queue else None
        if event:
            logger.debug("callback_polled session_id=%s event=%s", self.session_id, event.type.value)
        return event

    def has_events(self) -> bool:
        """Check if there are pending events."""
        return bool(self.event_queue)

    def request_human_input(
        self,
        prompt: str,
        options: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Pause the workflow and request human input.

        Args:
            prompt: The question or prompt to display to the user.
            options: Optional list of valid choices.
            context: Optional additional context data.
        """
        logger.info(
            "handler_pause session_id=%s agent=%s options=%d",
            self.session_id,
            self.current_agent or "System",
            len(options or []),
        )
        payload: dict[str, Any] = {
            "prompt": prompt,
            "agent": self.current_agent or "System",
        }
        if options is not None:
            payload["options"] = options
        if context:
            payload["context"] = context
        self.emit(CallbackEvent(CallbackEventType.AWAITING_USER_INPUT, payload, self.session_id))
        self.add_trace_record(
            "awaiting_user_input",
            payload,
            agent=self.current_agent or "System",
        )
        self.waiting_for_input = True

    def handle_user_response(self, user_input: str) -> dict[str, Any] | None:
        """Resume the workflow after receiving user input.

        Args:
            user_input: The user's response text.

        Returns:
            The paused state dict with user_response added, or None if cancelled.
        """
        logger.info(
            "handler_resume session_id=%s agent=%s input_length=%d has_state=%s",
            self.session_id,
            self.current_agent or "System",
            len(user_input),
            self.paused_state is not None,
        )
        self.waiting_for_input = False
        state = self.paused_state
        self.paused_state = None
        if user_input.strip().lower() in {"cancel", "exit", "quit"}:
            logger.warning("handler_cancelled session_id=%s", self.session_id)
            self.emit(
                CallbackEvent(
                    CallbackEventType.ERROR,
                    {"message": "Operation cancelled by user."},
                    self.session_id,
                )
            )
            return None
        if state is not None:
            state["user_response"] = user_input
        self.add_trace_record(
            "user_response",
            {"content": user_input},
            agent=(state or {}).get("agent", self.current_agent),
            iteration=(state or {}).get("iteration"),
        )
        return state

    def add_to_trace(self, text: str) -> None:
        """Add a ReACT trace entry for later export."""
        self.react_trace.append(text)
        logger.debug("trace_added session_id=%s length=%d", self.session_id, len(text))

    def get_trace_records(self) -> list[dict[str, Any]]:
        """Return structured trace records."""
        return list(self.trace_records)

    def get_trace(self) -> str:
        """Get the full ReACT trace as a single string."""
        return "\n".join(self.react_trace)

    def on_llm_error(self, error: Exception | KeyboardInterrupt) -> None:
        """Called when an LLM encounters an error."""
        logger.error("handler_llm_error session_id=%s error=%s", self.session_id, error)
        self.emit(
            CallbackEvent(
                CallbackEventType.ERROR,
                {"message": f"LLM error: {error}", "recoverable": True},
                self.session_id,
            )
        )

    def on_tool_error(self, error: Exception | KeyboardInterrupt) -> None:
        """Called when a tool encounters an error."""
        logger.error("handler_tool_error session_id=%s error=%s", self.session_id, error)
        self.emit(
            CallbackEvent(
                CallbackEventType.ERROR,
                {"message": f"Tool error: {error}", "recoverable": True},
                self.session_id,
            )
        )

    def on_agent_finish(self, agent_name: str, result: Any = None) -> None:
        """Called when an agent finishes its task."""
        logger.info("handler_agent_finish session_id=%s agent=%s", self.session_id, agent_name)
        self.emit(
            CallbackEvent(
                CallbackEventType.AGENT_COMPLETED,
                {"agent": agent_name, "result": str(result) if result else ""},
                self.session_id,
            )
        )

    def reset(self) -> None:
        """Reset the handler state for a new workflow."""
        logger.info("handler_reset session_id=%s", self.session_id)
        self.event_queue.clear()
        self.waiting_for_input = False
        self.paused_state = None
        self.current_agent = None
        self.react_trace.clear()
        self.trace_records.clear()


class CallbackProcessor:
    """Processes a list of callback handlers for extensible event handling.

    Allows multiple handlers to be registered and called for each event type.
    """

    def __init__(self, handlers: list[TimeSeriesConstructionHandler] | None = None) -> None:
        self.handlers: list[TimeSeriesConstructionHandler] = handlers or []

    def add_handler(self, handler: TimeSeriesConstructionHandler) -> None:
        """Register a new handler."""
        self.handlers.append(handler)
        logger.debug("callback_processor_handler_added total=%d", len(self.handlers))

    def on_event(self, event: CallbackEvent) -> None:
        """Dispatch an event to all registered handlers."""
        for handler in self.handlers:
            handler.emit(event)