"""Tests for workflow reporting and rule-driven validation."""
from __future__ import annotations

from financial_time_series_construction.workflow_report import build_workflow_report


def test_workflow_report_builds_summary_without_hardcoded_path() -> None:
    events = [
        {
            "type": "delegated",
            "payload": {
                "from_agent": "Orchestrator",
                "to_agent": "ReferenceDataAgent",
                "routing_mode": "llm",
            },
        },
        {
            "type": "agent_completed",
            "payload": {"agent": "ReferenceDataAgent"},
        },
        {
            "type": "awaiting_user_input",
            "payload": {"agent": "ReportingAgent", "prompt": "Choose source"},
        },
    ]

    report = build_workflow_report(events)

    assert report["summary"]["event_count"] == 3
    assert report["summary"]["completed_agents_unique"] == ["ReferenceDataAgent"]
    assert "ReportingAgent" in report["summary"]["paused_agents_unique"]
    assert report["routing"]["llm_delegations"] == 1
    assert report["validation"]["rules_applied"] is False


def test_workflow_report_rule_validation_is_runtime_configurable() -> None:
    events = [
        {
            "type": "delegated",
            "payload": {
                "from_agent": "MarketDataAgent",
                "to_agent": "DataQualityAgent",
                "routing_mode": "llm",
            },
        },
        {
            "type": "agent_completed",
            "payload": {"agent": "DataQualityAgent"},
        },
        {
            "type": "awaiting_user_input",
            "payload": {"agent": "ReportingAgent", "prompt": "Choose source"},
        },
    ]

    rules = {
        "require_no_errors": True,
        "required_pauses": ["ReportingAgent"],
        "required_completed_agents": ["DataQualityAgent"],
        "required_delegations": [{"from": "MarketDataAgent", "to": "DataQualityAgent"}],
        "min_llm_delegations": 1,
    }

    report = build_workflow_report(events, validation_rules=rules)

    assert report["validation"]["rules_applied"] is True
    assert report["validation"]["passed"] is True
    checks = report["validation"]["checks"]
    assert checks["require_no_errors"] is True
    assert checks["required_pause:ReportingAgent"] is True
    assert checks["required_completed:DataQualityAgent"] is True
    assert checks["required_delegation:MarketDataAgent->DataQualityAgent"] is True
    assert checks["min_llm_delegations"] is True


def test_workflow_report_tracks_unavailable_market_sources_and_validates() -> None:
    events = [
        {
            "type": "agent_completed",
            "payload": {
                "agent": "MarketDataAgent",
                "result": {
                    "delegated_to": "DataQualityAgent",
                    "loaded_sources": ["yahoo", "reuters"],
                    "unavailable_sources": [
                        {"source": "bloomberg", "reason": "No historical data for requested date range."}
                    ],
                },
            },
        },
        {
            "type": "delegated",
            "payload": {
                "from_agent": "MarketDataAgent",
                "to_agent": "DataQualityAgent",
                "routing_mode": "deterministic",
            },
        },
    ]

    rules = {
        "min_available_market_sources": 2,
        "max_unavailable_market_sources": 1,
        "required_unavailable_market_sources_logged": True,
    }

    report = build_workflow_report(events, validation_rules=rules)

    summary = report["summary"]
    assert summary["unavailable_market_source_count"] == 1
    assert summary["available_market_source_count"] == 2
    assert summary["unavailable_market_sources"][0]["source"] == "bloomberg"

    checks = report["validation"]["checks"]
    assert checks["min_available_market_sources"] is True
    assert checks["max_unavailable_market_sources"] is True
    assert checks["required_unavailable_market_sources_logged"] is True
