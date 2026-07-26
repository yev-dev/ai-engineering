"""CLI rendering tests for data quality summary output."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from financial_time_series_construction.agents_definition import CallbackEvent, CallbackEventType


def test_print_events_renders_data_quality_summary(capsys) -> None:
    from financial_time_series_construction.cli import _print_events

    report = {
        "report_type": "data_quality_summary",
        "rows": [
            {
                "source": "yahoo",
                "symbol": "AAPL",
                "total_values": 3,
                "available_record_count": 2,
                "missing_count": 1,
                "completeness_pct": 66.67,
                "min_date": "2023-01-03",
                "max_date": "2023-01-05",
                "duplicate_count": 0,
                "issues": ["missing_or_nan_values"],
                "note": None,
            }
        ],
        "summary": {
            "symbol": "AAPL",
            "source_count": 1,
            "sources": ["yahoo"],
            "total_available_records": 2,
            "total_missing_count": 1,
            "min_date": "2023-01-03",
            "max_date": "2023-01-05",
            "average_completeness_pct": 66.67,
            "best_source_by_completeness": "yahoo",
            "worst_source_by_completeness": "yahoo",
        },
    }
    event = CallbackEvent(
        CallbackEventType.AGENT_COMPLETED,
        {
            "agent": "DataQualityAgent",
            "result": {
                "delegated_to": "ReportingAgent",
                "data_quality_report": report,
            },
        },
        "test_session",
    )

    _print_events([event])
    output = capsys.readouterr().out
    assert "[DATA QUALITY] Summary" in output
    assert "yahoo" in output
    assert "66.67" in output
    assert "best_source=yahoo" in output
    assert "available_records=2" in output
    assert "date_range=2023-01-03..2023-01-05" in output


def test_save_artifacts_persists_data_quality_summary_csv(tmp_path: Path, monkeypatch) -> None:
    from financial_time_series_construction.cli import _save_artifacts

    monkeypatch.setenv("TIME_SERIES_OUTPUT_DIR", str(tmp_path))
    runtime = SimpleNamespace(
        session_id="test_session",
        get_trace=lambda: "trace",
        get_trace_records=lambda: [],
    )

    report = {
        "report_type": "data_quality_summary",
        "rows": [
            {
                "source": "yahoo",
                "symbol": "AAPL",
                "total_values": 3,
                "available_record_count": 2,
                "missing_count": 1,
                "completeness_pct": 66.67,
                "min_date": "2023-01-03",
                "max_date": "2023-01-05",
                "issues": ["missing_or_nan_values"],
            }
        ],
        "summary": {
            "symbol": "AAPL",
            "source_count": 1,
            "sources": ["yahoo"],
            "total_available_records": 2,
            "total_missing_count": 1,
            "min_date": "2023-01-03",
            "max_date": "2023-01-05",
            "average_completeness_pct": 66.67,
            "best_source_by_completeness": "yahoo",
            "worst_source_by_completeness": "yahoo",
        },
    }
    events = [
        CallbackEvent(
            CallbackEventType.AGENT_COMPLETED,
            {
                "agent": "DataQualityAgent",
                "result": {"data_quality_report": report},
            },
            "test_session",
        )
    ]

    run_id = "run_quality_csv"
    _save_artifacts(events, runtime, run_id)

    run_dir = tmp_path / run_id
    csv_path = run_dir / "data_quality_summary.csv"
    assert csv_path.exists()

    content = csv_path.read_text()
    assert "available_record_count" in content
    assert "summary_min_date" in content

    workflow_report_path = run_dir / "workflow_report.json"
    assert workflow_report_path.exists()
    parsed = json.loads(workflow_report_path.read_text())
    assert isinstance(parsed, dict)
