"""Interactive CLI for the time series construction LangGraph workflow."""
from __future__ import annotations

import json
import logging

try:
    from .agents_definition import CallbackEvent, CallbackEventType
    from .graph import TimeSeriesConstructionGraph
except ImportError:
    from agents_definition import CallbackEvent, CallbackEventType
    from graph import TimeSeriesConstructionGraph
try:
    from .logging_config import configure_logging
except ImportError:
    from logging_config import configure_logging


def display_event(event: CallbackEvent) -> str:
    payload = event.payload
    if event.type == CallbackEventType.AWAITING_USER_INPUT:
        options = payload.get("options", [])
        suffix = "\n" + "\n".join(f"  {i}. {option}" for i, option in enumerate(options, 1)) if options else ""
        return f"[{payload.get('agent', 'System')}] {payload.get('prompt', '')}{suffix}"
    if event.type == CallbackEventType.AGENT_COMPLETED:
        result = payload.get("result", {})
        return f"[{payload.get('agent', 'System')}] {result.get('final_answer', result)}"
    if event.type == CallbackEventType.ERROR:
        action = payload.get("user_action")
        return f"[Error] {payload.get('message', 'Unknown error')}" + (f"\n[Next step] {action}" if action else "")
    return f"[{event.type.value}] {json.dumps(payload, default=str)}"


class TimeSeriesCLI:
    def __init__(self) -> None:
        self.graph = TimeSeriesConstructionGraph()
        self.waiting = False

    def run(self) -> None:
        print("Time Series Construction (LangGraph) | Human-in-the-loop")
        print("Environment: ai_engineering (activate this before launching the CLI)")
        print("Describe a ticker and date range, or type 'quit'.")
        while True:
            try:
                value = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not value:
                continue
            if value.casefold() in {"quit", "exit"}:
                logging.getLogger(__name__).info("cli_exit_requested")
                return
            logging.getLogger(__name__).info("cli_input_received characters=%d waiting=%s", len(value), self.waiting)
            events = self.graph.process_user_response(value) if self.waiting else self.graph.process_user_request(value)
            self.waiting = False
            for event in events:
                print(display_event(event))
                self.waiting = self.waiting or event.type == CallbackEventType.AWAITING_USER_INPUT


if __name__ == "__main__":
    configure_logging()
    TimeSeriesCLI().run()