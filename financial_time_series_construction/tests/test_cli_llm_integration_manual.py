"""Manual real-LLM CLI integration test for full interactive workflow.

This test is opt-in and intended for local verification with Ollama.
It runs the real CLI entrypoint with LLM_MODEL=ollama/qwen2.5:1.5b and
checks that the workflow pauses for both expected human checkpoints:
1) source selection after ReportingAgent quality comparison,
2) gap-filling method selection after GapFillingAgent recommendations.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.getenv("RUN_LLM_INTEGRATION") != "1",
    reason="Set RUN_LLM_INTEGRATION=1 to run manual real-LLM integration test.",
)
def test_cli_real_llm_two_human_checkpoints(tmp_path: Path) -> None:
    """Run real CLI with qwen and verify two explicit user checkpoints.

    Command under test:
    LLM_MODEL="ollama/qwen2.5:1.5b" python -m financial_time_series_construction.cli
      --request "AAPL from 2023-01-01 to 2024-01-01"
    """
    env = os.environ.copy()
    env["LLM_MODEL"] = "ollama/qwen2.5:1.5b"
    env["TIME_SERIES_OUTPUT_DIR"] = str(tmp_path / "tsc_artifacts")
    env["TIME_SERIES_VALIDATION_RULES"] = str(
        Path(__file__).resolve().parents[1] / "validation_rules.example.json"
    )

    # Provide deterministic responses for the two expected prompts.
    # First line: selected source.
    # Second line: selected gap-filling method via quick-option number.
    user_input = "yahoo\n1\n"

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "financial_time_series_construction.cli",
            "--request",
            "AAPL from 2023-01-01 to 2024-01-01",
            "--session-id",
            "llm_manual_verify",
        ],
        input=user_input,
        text=True,
        capture_output=True,
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
        timeout=240,
        check=False,
    )

    output = (process.stdout or "") + "\n" + (process.stderr or "")

    assert process.returncode == 0, (
        f"CLI exited with code {process.returncode}. Output:\n{output}"
    )

    run_dir = Path(env["TIME_SERIES_OUTPUT_DIR"]) / "llm_manual_verify"
    report_path = run_dir / "workflow_report.json"
    assert report_path.exists(), f"Expected workflow report at {report_path}. Output:\n{output}"

    report = json.loads(report_path.read_text())
    summary = report.get("summary", {})
    checks = report.get("validation", {}).get("checks", {})

    # Rule-driven validation of required HITL checkpoints.
    assert checks.get("required_pause:ReportingAgent") is True
    assert checks.get("required_pause:GapFillingAgent") is True

    # Ensure delegation path reached reporting and gap-filling phases.
    edges = [tuple(item) for item in summary.get("delegation_edges_unique", [])]
    assert ("DataQualityAgent", "ReportingAgent") in edges
    assert ("ReportingAgent", "GapFillingAgent") in edges
    assert ("GapFillingAgent", "TimeSeriesConstructionAgent") in edges
    assert ("TimeSeriesConstructionAgent", "ReportingAgent") in edges

    completed = summary.get("completed_agents_unique", [])
    assert "TimeSeriesConstructionAgent" in completed
    assert "ReportingAgent" in completed

    # Availability metadata should always be present for validation/reporting.
    assert "unavailable_market_source_count" in summary
    assert "unavailable_market_sources" in summary

    # Keep no-error guarantee for this verification run.
    assert summary.get("error_count", 1) == 0
