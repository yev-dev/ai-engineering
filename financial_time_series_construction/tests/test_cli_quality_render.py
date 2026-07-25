"""CLI rendering tests for data quality summary output."""
from __future__ import annotations

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
                "missing_count": 1,
                "completeness_pct": 66.67,
                "duplicate_count": 0,
                "issues": ["missing_or_nan_values"],
                "note": None,
            }
        ],
        "summary": {
            "symbol": "AAPL",
            "source_count": 1,
            "sources": ["yahoo"],
            "total_missing_count": 1,
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
