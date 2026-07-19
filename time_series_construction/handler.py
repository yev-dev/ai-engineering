"""Callback event bus and human-in-the-loop pause/resume state."""
from __future__ import annotations

import logging
from collections import deque
from typing import Any

from langchain_core.agents import AgentFinish
from langchain_core.callbacks import BaseCallbackHandler

try:
    from .agents_definition import CallbackEvent, CallbackEventType
except ImportError:
    from agents_definition import CallbackEvent, CallbackEventType

logger = logging.getLogger(__name__)


class TimeSeriesConstructionCallbackHandler(BaseCallbackHandler):
    def __init__(self, session_id: str = "default") -> None:
        self.event_queue: deque[CallbackEvent] = deque()
        self.waiting_for_input = False
        self.paused_state: dict[str, Any] | None = None
        self.current_agent: str | None = None
        self.session_id = session_id
        logger.info("handler_initialized session_id=%s", session_id)

    def emit(self, event: CallbackEvent) -> None:
        self.event_queue.append(event)
        logger.info("event=%s agent=%s", event.type.value, event.payload.get("agent"))

    def poll(self) -> CallbackEvent | None:
        event = self.event_queue.popleft() if self.event_queue else None
        if event:
            logger.debug("callback_polled session_id=%s event=%s", self.session_id, event.type.value)
        return event

    def has_events(self) -> bool:
        return bool(self.event_queue)

    def request_human_input(self, prompt: str, options: list[str] | None = None,
                            context: dict[str, Any] | None = None) -> None:
        logger.info(
            "handler_pause session_id=%s agent=%s options=%d",
            self.session_id,
            self.current_agent or "System",
            len(options or []),
        )
        payload: dict[str, Any] = {"prompt": prompt, "agent": self.current_agent or "System"}
        if options is not None:
            payload["options"] = options
        if context:
            payload["context"] = context
        self.emit(CallbackEvent(CallbackEventType.AWAITING_USER_INPUT, payload, self.session_id))
        self.waiting_for_input = True

    def handle_user_response(self, user_input: str) -> dict[str, Any] | None:
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
            self.emit(CallbackEvent(CallbackEventType.ERROR, {"message": "Operation cancelled."}, self.session_id))
            return None
        if state is not None:
            state["user_response"] = user_input
        return state

    def on_llm_error(self, error: Exception | KeyboardInterrupt, **kwargs: Any) -> None:
        logger.error("callback_llm_error session_id=%s error=%s", self.session_id, error)
        self.emit(CallbackEvent(CallbackEventType.ERROR, {"message": f"LLM error: {error}"}, self.session_id))

    def on_tool_error(self, error: Exception | KeyboardInterrupt, **kwargs: Any) -> None:
        logger.error("callback_tool_error session_id=%s error=%s", self.session_id, error)
        self.emit(CallbackEvent(CallbackEventType.ERROR, {"message": f"Tool error: {error}"}, self.session_id))

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        logger.info("callback_agent_finish session_id=%s agent=%s", self.session_id, self.current_agent)
        self.emit(CallbackEvent(CallbackEventType.AGENT_COMPLETED, {
            "agent": self.current_agent or "Unknown", "result": finish.return_values,
        }, self.session_id))

    def reset(self) -> None:
        logger.info("handler_reset session_id=%s", self.session_id)
        self.event_queue.clear()
        self.waiting_for_input = False
        self.paused_state = None
        self.current_agent = None
