"""Command-line interface for the time series construction autogen workflow."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from financial_time_series_construction.agents_definition import CallbackEventType
from financial_time_series_construction.database import (
    DataStore,
    get_datastore,
    init_datastore,
    close_datastore,
)
from financial_time_series_construction.models import ModelRequestFactory
from financial_time_series_construction.prompt_library import (
    format_prompt_menu,
    resolve_prompt_selection,
)
from financial_time_series_construction.runtime import AgenticRuntime, build_runtime
from financial_time_series_construction.workflow_report import (
    build_workflow_report,
    format_workflow_report,
)

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


def _json_safe(value: Any) -> Any:
    """Convert values to JSON-serializable structure without flattening dicts/lists."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _print_data_quality_report(report: dict[str, Any]) -> None:
    """Render a structured data quality report to the console."""
    rows = report.get("rows", []) or []
    summary = report.get("summary", {}) or {}
    if not rows:
        return

    print("\n[DATA QUALITY] Summary")
    if RICH_AVAILABLE:
        console = Console()
        table = Table(title="Data Quality Report")
        table.add_column("Source", style="cyan")
        table.add_column("Symbol", style="green")
        table.add_column("Completeness %", justify="right")
        table.add_column("Available", justify="right")
        table.add_column("Missing", justify="right")
        table.add_column("Min Date")
        table.add_column("Max Date")
        table.add_column("Duplicates", justify="right")
        table.add_column("Issues")
        for row in rows:
            issues = row.get("issues") or []
            table.add_row(
                str(row.get("source", "")),
                str(row.get("symbol", "")),
                str(row.get("completeness_pct", "n/a")),
                str(row.get("available_record_count", "n/a")),
                str(row.get("missing_count", "n/a")),
                str(row.get("min_date") or "n/a"),
                str(row.get("max_date") or "n/a"),
                str(row.get("duplicate_count", "n/a")),
                ", ".join(str(item) for item in issues) if issues else "none",
            )
        console.print(table)
    else:
        print("Source | Symbol | Completeness % | Available | Missing | Min Date | Max Date | Duplicates | Issues")
        for row in rows:
            issues = row.get("issues") or []
            print(
                f"{row.get('source', '')} | {row.get('symbol', '')} | "
                f"{row.get('completeness_pct', 'n/a')} | {row.get('available_record_count', 'n/a')} | "
                f"{row.get('missing_count', 'n/a')} | {row.get('min_date') or 'n/a'} | "
                f"{row.get('max_date') or 'n/a'} | "
                f"{row.get('duplicate_count', 'n/a')} | "
                f"{', '.join(str(item) for item in issues) if issues else 'none'}"
            )

    best_source = summary.get("best_source_by_completeness")
    avg_completeness = summary.get("average_completeness_pct")
    total_missing = summary.get("total_missing_count")
    total_available = summary.get("total_available_records")
    min_date = summary.get("min_date")
    max_date = summary.get("max_date")
    print(
        "Summary: "
        f"sources={summary.get('source_count', len(rows))}, "
        f"best_source={best_source or 'n/a'}, "
        f"available_records={total_available if total_available is not None else 'n/a'}, "
        f"date_range={min_date or 'n/a'}..{max_date or 'n/a'}, "
        f"avg_completeness={avg_completeness if avg_completeness is not None else 'n/a'}, "
        f"total_missing={total_missing if total_missing is not None else 'n/a'}"
    )


def _save_data_quality_summary_csv(events: list[Any], run_dir: Path) -> Path | None:
    """Persist latest data quality summary rows as CSV if present in events."""
    latest_report: dict[str, Any] | None = None
    for event in events:
        if event.type.value != CallbackEventType.AGENT_COMPLETED.value:
            continue
        result = event.payload.get("result", {})
        if isinstance(result, dict) and isinstance(result.get("data_quality_report"), dict):
            latest_report = result["data_quality_report"]

    if not latest_report:
        return None

    rows = latest_report.get("rows", []) or []
    summary = latest_report.get("summary", {}) or {}
    if not rows:
        return None

    flattened_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        issues = item.get("issues")
        if isinstance(issues, list):
            item["issues"] = ";".join(str(value) for value in issues)
        item["summary_total_available_records"] = summary.get("total_available_records")
        item["summary_total_missing_count"] = summary.get("total_missing_count")
        item["summary_min_date"] = summary.get("min_date")
        item["summary_max_date"] = summary.get("max_date")
        item["summary_best_source_by_completeness"] = summary.get("best_source_by_completeness")
        item["summary_average_completeness_pct"] = summary.get("average_completeness_pct")
        flattened_rows.append(item)

    output_path = run_dir / "data_quality_summary.csv"
    pd.DataFrame(flattened_rows).to_csv(output_path, index=False)
    return output_path


def _save_artifacts(
    events: list[Any],
    runtime: AgenticRuntime,
    run_id: str,
) -> None:
    """Save workflow artifacts: trace, events log, and DataStore records."""
    output_root = Path(os.getenv(
        "TIME_SERIES_OUTPUT_DIR",
        Path.home() / "time_series_construction",
    ))
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize DataStore for artifact recording (singleton)
    try:
        datastore = get_datastore()
    except RuntimeError:
        try:
            datastore = init_datastore(
                output_root / "time_series_database" / "database" / "datastore.db"
            )
        except Exception as error:
            logger.warning("Could not initialize DataStore: %s", error)
            datastore = None

    # Save ReACT trace
    trace_path = run_dir / "react_trace.txt"
    trace_path.write_text(runtime.get_trace())
    logger.info("Trace saved to %s", trace_path)
    if datastore is not None:
        try:
            datastore.put_artifact(run_id, "report", str(trace_path))
        except Exception as error:
            logger.debug("artifact_store_failed path=%s error=%s", trace_path, error)

    trace_json_path = run_dir / "react_trace.json"
    trace_json_path.write_text(json.dumps(_json_safe(runtime.get_trace_records()), indent=2))
    logger.info("Structured trace saved to %s", trace_json_path)
    if datastore is not None:
        try:
            datastore.put_artifact(run_id, "report", str(trace_json_path))
        except Exception as error:
            logger.debug("artifact_store_failed path=%s error=%s", trace_json_path, error)

    # Save events log
    events_path = run_dir / "events.json"
    events_data = [
        {
            "type": e.type.value,
            "payload": _json_safe(e.payload),
            "session_id": e.session_id,
        }
        for e in events
    ]
    events_path.write_text(json.dumps(events_data, indent=2))
    logger.info("Events saved to %s", events_path)
    if datastore is not None:
        try:
            datastore.put_artifact(run_id, "report", str(events_path))
        except Exception as error:
            logger.debug("artifact_store_failed path=%s error=%s", events_path, error)

    # Save workflow report and optional validation results.
    validation_rules: dict[str, Any] | None = None
    rules_path = os.getenv("TIME_SERIES_VALIDATION_RULES")
    if rules_path:
        try:
            validation_rules = json.loads(Path(rules_path).read_text())
            logger.info("Loaded validation rules from %s", rules_path)
        except Exception as error:
            logger.warning("Could not load validation rules from %s: %s", rules_path, error)

    workflow_report = build_workflow_report(events, validation_rules=validation_rules)

    report_json_path = run_dir / "workflow_report.json"
    report_json_path.write_text(json.dumps(workflow_report, indent=2))
    logger.info("Workflow report saved to %s", report_json_path)
    if datastore is not None:
        try:
            datastore.put_artifact(run_id, "report", str(report_json_path))
        except Exception as error:
            logger.debug("artifact_store_failed path=%s error=%s", report_json_path, error)

    report_text_path = run_dir / "workflow_report.txt"
    report_text_path.write_text(format_workflow_report(workflow_report))
    logger.info("Workflow report text saved to %s", report_text_path)
    if datastore is not None:
        try:
            datastore.put_artifact(run_id, "report", str(report_text_path))
        except Exception as error:
            logger.debug("artifact_store_failed path=%s error=%s", report_text_path, error)

    quality_csv_path = _save_data_quality_summary_csv(events, run_dir)
    if quality_csv_path is not None:
        logger.info("Data quality summary CSV saved to %s", quality_csv_path)
        if datastore is not None:
            try:
                datastore.put_artifact(run_id, "csv", str(quality_csv_path))
            except Exception as error:
                logger.debug("artifact_store_failed path=%s error=%s", quality_csv_path, error)


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
            if (
                agent == "DataQualityAgent"
                and isinstance(result, dict)
                and result.get("data_quality_report")
            ):
                print(f"\n[COMPLETE] Agent: {agent}")
                _print_data_quality_report(result["data_quality_report"])
            elif isinstance(result, dict) and "final_answer" in result:
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


def _print_summary(runtime: AgenticRuntime) -> None:
    """Print a rich summary table of the workflow execution."""
    if RICH_AVAILABLE:
        console = Console()
        table = Table(title="Time Series Construction - Workflow Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Session ID", runtime.session_id)
        table.add_row("Events Processed", str(runtime.trace_line_count()))
        table.add_row("Trace Lines", str(runtime.trace_line_count()))
        console.print(table)
    else:
        print("\n" + "=" * 60)
        print("WORKFLOW SUMMARY")
        print("=" * 60)
        print(f"Session ID: {runtime.session_id}")
        print(f"Events Processed: {runtime.trace_line_count()}")
        print(f"Trace Lines: {runtime.trace_line_count()}")
        print("=" * 60)


def _is_shell_command(user_input: str) -> bool:
    """Check if input looks like a shell/environment command."""
    command = user_input.casefold().strip()
    return command.startswith(("conda ", "source ", "export ", "pip ", "python ", "cd "))


def main() -> None:
    """Main entry point for the time series construction workflow."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Time Series Construction - ReACT Workflow",
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
    parser.add_argument(
        "--debug-flow",
        action="store_true",
        help="Enable one-line processor/tool flow diagnostics.",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "github", "deepseek"],
        help="LLM provider override for this run.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model override for this run. Can be short model name or full provider/model identifier.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature override for this run.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum output tokens override for this run.",
    )
    parser.add_argument(
        "--framework",
        choices=["langgraph", "autogen"],
        default=os.getenv("AGENTIC_FRAMEWORK", "autogen"),
        help="Agentic framework runtime to use.",
    )
    parser.add_argument(
        "--list-model-config",
        action="store_true",
        help="Print resolved provider/model configuration and exit.",
    )

    args = parser.parse_args()

    # Apply runtime model overrides via environment so the factory resolves
    # configuration in one place (including .env defaults).
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    if args.temperature is not None:
        os.environ["LLM_TEMPERATURE"] = str(args.temperature)
    if args.max_tokens is not None:
        os.environ["LLM_MAX_TOKENS"] = str(args.max_tokens)
    if args.debug_flow:
        os.environ["TSC_DEBUG_FLOW"] = "1"

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = ModelRequestFactory.describe_environment()
    if args.list_model_config:
        print(json.dumps(config, indent=2))
        return

    session_id = args.session_id
    factory = ModelRequestFactory.from_environment()
    runtime = build_runtime(args.framework, session_id=session_id, factory=factory)

    print(f"\n{'='*60}")
    print("TIME SERIES CONSTRUCTION WORKFLOW")
    print(f"Session: {session_id}")
    print(f"Framework: {args.framework}")
    print(f"Provider: {config['provider']}")
    print(f"Model: {config['model']}")
    if os.getenv("TSC_DEBUG_FLOW"):
        print("Debug flow: enabled (TSC_DEBUG_FLOW=1)")
    print(f"{'='*60}\n")

    if args.request:
        all_events: list[Any] = []
        # Show the user request immediately so the user knows something
        # happened before the first LLM call (which takes 10-30s on deepseek).
        # The processor bypasses the Orchestrator LLM call when the request
        # contains an instrument + date range ("AAPL between Jan 2023 and Jan 2024"),
        # but ReferenceDataAgent still makes an LLM call right after. This
        # immediate feedback prevents the "waiting in silence" problem.
        print(f"\n[REQUEST] {args.request}")
        print("[SYSTEM] Processing... (first LLM call may take 10-30s with deepseek-v2:16b)")
        events = runtime.process_user_request(args.request)
        all_events.extend(events)
        _print_events(events)

        # Handle human-in-the-loop pauses interactively with prompt library
        while runtime.waiting_for_input:
            try:
                # Determine the prompt category based on the paused agent
                agent_name = runtime.current_agent or ""
                if agent_name == "ReportingAgent":
                    prompt_category = "source_selection"
                elif agent_name == "GapFillingAgent":
                    prompt_category = "gap_filling"
                elif agent_name == "Orchestrator":
                    prompt_category = "clarification"
                else:
                    prompt_category = "general"

                # Show prompt library menu
                menu = format_prompt_menu(prompt_category)
                if menu:
                    print("\n--- Quick Options ---")
                    print(menu)
                else:
                    print("\nYour response (or 'exit' to quit):")

                user_input = input("> ").strip()
                if user_input.lower() in ("exit", "quit"):
                    print("Workflow cancelled by user.")
                    break

                # Resolve prompt library selection if applicable
                resolved = resolve_prompt_selection(prompt_category, user_input)
                if resolved:
                    user_input = resolved

                events = runtime.process_user_response(user_input)
                all_events.extend(events)
                _print_events(events)
            except (KeyboardInterrupt, EOFError):
                print("\nWorkflow interrupted.")
                break
    else:
        all_events = []
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
                events = runtime.process_user_request(user_input)
                all_events.extend(events)
                _print_events(events)

                while runtime.waiting_for_input:
                    # Determine the prompt category based on the paused agent
                    agent_name = runtime.current_agent or ""
                    if agent_name == "ReportingAgent":
                        prompt_category = "source_selection"
                    elif agent_name == "GapFillingAgent":
                        prompt_category = "gap_filling"
                    elif agent_name == "Orchestrator":
                        prompt_category = "clarification"
                    else:
                        prompt_category = "general"

                    # Show prompt library menu
                    menu = format_prompt_menu(prompt_category)
                    if menu:
                        print("\n--- Quick Options ---")
                        print(menu)
                    else:
                        print("\nYour response (or 'exit' to quit):")

                    user_input = input("> ").strip()
                    if user_input.lower() in ("exit", "quit"):
                        print("Workflow cancelled.")
                        break

                    # Resolve prompt library selection if applicable
                    resolved = resolve_prompt_selection(prompt_category, user_input)
                    if resolved:
                        user_input = resolved

                    events = runtime.process_user_response(user_input)
                    all_events.extend(events)
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
        all_events if 'all_events' in dir() else [],
        runtime,
        run_id,
    )
    _print_summary(runtime)
    logger.info(f"\nArtifacts saved to: {Path.home() / 'time_series_construction' / run_id}")


if __name__ == "__main__":
    main()