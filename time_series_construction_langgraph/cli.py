"""Interactive CLI for the time series construction LangGraph workflow."""
from __future__ import annotations

import logging
import re

try:
    from .graph import TimeSeriesConstructionGraph
    from .logging_config import configure_logging
except ImportError:
    from graph import TimeSeriesConstructionGraph
    from logging_config import configure_logging

# Shell/system command patterns to skip
_SYSTEM_COMMAND_RE = re.compile(
    r"^(conda|source|activate|deactivate|export|set|unset|alias|unalias|"
    r"cd\b|pushd|popd|ls\b|dir|echo|printf|clear|exit|quit)\s",
    re.IGNORECASE,
)


def _is_system_command(value: str) -> bool:
    if not value.strip():
        return True
    if _SYSTEM_COMMAND_RE.match(value):
        return True
    tokens = value.strip().split()
    if len(tokens) == 1:
        lone = tokens[0].casefold()
        if lone in {"ai_engineering", "base", "activate", "deactivate"}:
            return True
    return False


def display_event(event: dict) -> str:
    t = event.get("type", "")
    if t == "await":
        options = event.get("options", [])
        suffix = "\n" + "\n".join(f"  {i}. {option}" for i, option in enumerate(options, 1)) if options else ""
        return f"[{event.get('agent', 'System')}] {event.get('prompt', '')}{suffix}"
    if t == "final":
        return f"[{event.get('agent', 'System')}] {event.get('answer', '')}"
    if t == "intermediate":
        return f"[{event.get('agent', 'System')}] {event.get('message', '')}"
    if t == "error":
        return f"[Error] {event.get('message', 'Unknown error')}"
    if t == "user_request":
        return f"[user_request] {event.get('request', '')}"
    return f"[{t}] {event}"


class TimeSeriesCLI:
    def __init__(self) -> None:
        self.graph = TimeSeriesConstructionGraph()
        self.waiting = False

    def run(self) -> None:
        print("Time Series Construction (LangGraph) | Human-in-the-loop")
        print("Describe a ticker and date range, or type 'quit'.")
        while True:
            try:
                value = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not value:
                continue

            if _is_system_command(value):
                logging.getLogger(__name__).info("skipped system command: %s", value)
                continue

            if value.casefold() in {"quit", "exit"}:
                return

            events = self.graph.process_user_response(value) if self.waiting else self.graph.process_user_request(value)
            self.waiting = False
            for event in events:
                print(display_event(event))
                self.waiting = self.waiting or event.get("type") == "await"


if __name__ == "__main__":
    configure_logging()
    TimeSeriesCLI().run()