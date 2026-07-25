"""Command-line interface for the time series construction autogen workflow."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from financial_time_series_construction.agents_definition import CallbackEventType
from financial_time_series_construction.handler import TimeSeriesConstructionHandler
from financial_time_series_construction.processor import TimeSeriesConstructionProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Rich table for summary output
try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def _save_artifacts(
    events: list[Any],
    handler: TimeSeriesConstructionHandler,
    run_id: str,
) -> None:
    """Save workflow artifacts: trace, events log."""
    output_root = Path(os.getenv(
        "TIME_SERIES_OUTPUT_DIR",
        Path.home() / "time_series_construction",
    ))
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save ReACT trace
    trace_path = run_dir / "react_trace.txt"
    trace_path.write_text(handler.get_trace())
    logger.info("Trace saved to %s", trace_path)

    # Save events log
    events_path = run_dir / "events.json"
    events_data = [
        {
            "type": e.type.value,
            "payload": {k: str(v) for k, v in e.payload.items()},
            "session_id": e.session_id,
        }
        for e in events
    ]
    events_path.write_text(json.dumps(events_data, indent=2))
    logger.info("Events saved to %s", events_path)


def _print_events(events: list[Any]) -> None:
    """Print events to console in a readable format."""
    for event in events:
        event_type = event.type.value
        agent = event.payload.get("agent", "System")
        if event_type == CallbackEventType.AWAITING_USER_INPUT.value:
            prompt = event.payload.get("prompt", "")
            options = event.payload.get("options")
            print(f"\n[PAUSE] Agent: {agent}")
            print(f"Prompt: {prompt}")
            if options:
                print(f"Options: {', '.join(options)}")
        elif event_type == CallbackEventType.AGENT_COMPLETED.value:
            result = event.payload.get("result", {})
            if isinstance(result, dict) and "final_answer" in result:
                print(f"\n[COMPLETE] Agent: {agent}")
                print(f"Result: {result['final_answer']}")
            else:
                print(f"\n[COMPLETE] Agent: {agent}")
        elif event_type == CallbackEventType.ERROR.value:
            print(f"\n[ERROR] Agent: {agent}")
            print(f"Message: {event.payload.get('message', 'Unknown error')}")
        elif event_type == CallbackEventType.DELEGATED.value:
            print(f"\n[DELEGATE] {event.payload.get('from_agent')} → {event.payload.get('to_agent')}")
            routing_reason = event.payload.get("routing_reason")
            routing_mode = event.payload.get("routing_mode")
            if routing_reason and routing_mode == "deterministic":
                print(f"[ROUTE] {routing_reason}")
        elif event_type == CallbackEventType.USER_REQUEST.value:
            print(f"\n[REQUEST] {event.payload.get('request', '')}")
        else:
            print(f"\n[{event_type.upper()}] Agent: {agent}")


def _print_summary(handler: TimeSeriesConstructionHandler) -> None:
    """Print a rich summary table of the workflow execution."""
    if RICH_AVAILABLE:
        console = Console()
        table = Table(title="Time Series Construction - Workflow Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Session ID", handler.session_id)
        table.add_row("Events Processed", str(len(handler.react_trace)))
        table.add_row("Trace Lines", str(len(handler.react_trace)))
        console.print(table)
    else:
        print("\n" + "=" * 60)
        print("WORKFLOW SUMMARY")
        print("=" * 60)
        print(f"Session ID: {handler.session_id}")
        print(f"Events Processed: {len(handler.react_trace)}")
        print(f"Trace Lines: {len(handler.react_trace)}")
        print("=" * 60)


def _is_shell_command(user_input: str) -> bool:
    """Check if input looks like a shell/environment command."""
    command = user_input.casefold().strip()
    return command.startswith(("conda ", "source ", "export ", "pip ", "python ", "cd "))


def main() -> None:
    """Main entry point for the time series construction workflow."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Time Series Construction - Autogen ReACT Workflow",
    )
    parser.add_argument(
        "--request",
        "-r",
        type=str,
        help="Initial request (e.g., 'Build AAPL from 2023-01-01 to 2023-12-31')",
    )
    parser.add_argument(
        "--session-id",
        "-s",
        type=str,
        default=f"tsc_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}",
        help="Session identifier for this run",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    session_id = args.session_id
    handler = TimeSeriesConstructionHandler(session_id=session_id)
    processor = TimeSeriesConstructionProcessor(handler=handler)

    print(f"\n{'='*60}")
    print("TIME SERIES CONSTRUCTION WORKFLOW")
    print(f"Session: {session_id}")
    print(f"{'='*60}\n")

    if args.request:
        events = processor.process_user_request(args.request)
        _print_events(events)

        # Handle human-in-the-loop pauses interactively
        while handler.waiting_for_input:
            try:
                user_input = input("\nYour response (or 'exit' to quit): ").strip()
                if user_input.lower() in ("exit", "quit"):
                    print("Workflow cancelled by user.")
                    break
                events = processor.process_user_response(user_input)
                _print_events(events)
            except (KeyboardInterrupt, EOFError):
                print("\nWorkflow interrupted.")
                break
    else:
        # Interactive mode
        print("Enter a financial time series request.")
        print("Example: 'Create time series for Apple (AAPL) time series for from date 1 January 2023 to 1 January 2024'")
        print("Type 'exit' to quit.")
        print("(Shell commands like 'conda activate' are not supported here — ")
        print(" activate your environment in your terminal before starting this app.)\n")

        try:
            user_input = input("> ").strip()
            while user_input.lower() not in ("exit", "quit"):
                # Catch shell/conda commands at the CLI level before they reach the processor.
                # This avoids misleading [PAUSE] events and provides a clear message.
                if _is_shell_command(user_input):
                    print("\n[SYSTEM] Shell/conda commands are not supported inside this application.")
                    print("Please activate your conda environment in your terminal before starting this app.")
                    print("Enter a financial request, for example: 'Build AAPL from 2023-01-01 to 2023-12-31'\n")
                    user_input = input("> ").strip()
                    continue
                events = processor.process_user_request(user_input)
                _print_events(events)

                while handler.waiting_for_input:
                    user_input = input("\nYour response: ").strip()
                    if user_input.lower() in ("exit", "quit"):
                        print("Workflow cancelled.")
                        break
                    events = processor.process_user_response(user_input)
                    _print_events(events)

                if user_input.lower() not in ("exit", "quit"):
                    user_input = input("\n> ").strip()
                else:
                    break
        except (KeyboardInterrupt, EOFError):
            print("\nWorkflow interrupted.")

    # Save artifacts
    run_id = session_id
    _save_artifacts(
        events if 'events' in dir() else [],
        handler,
        run_id,
    )
    _print_summary(handler)
    print(f"\nArtifacts saved to: {Path.home() / 'time_series_construction' / run_id}")


if __name__ == "__main__":
    main()