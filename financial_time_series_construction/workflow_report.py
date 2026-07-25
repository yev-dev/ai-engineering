"""Workflow reporting and validation utilities.

Generates a structured report from callback events and supports optional,
rule-driven validation loaded at runtime. This keeps validation adaptable
without hardcoding one fixed workflow path.
"""
from __future__ import annotations

from collections import Counter
from typing import Any


def _event_to_dict(event: Any) -> dict[str, Any]:
    """Normalize event objects/dicts into a common dict format."""
    if isinstance(event, dict):
        event_type = str(event.get("type", ""))
        payload = event.get("payload", {}) or {}
        session_id = event.get("session_id")
        return {"type": event_type, "payload": payload, "session_id": session_id}

    event_type = getattr(getattr(event, "type", None), "value", "")
    payload = getattr(event, "payload", {}) or {}
    session_id = getattr(event, "session_id", None)
    return {"type": str(event_type), "payload": payload, "session_id": session_id}


def _normalize_required_delegation(item: Any) -> tuple[str, str] | None:
    """Normalize required delegation spec from dict or 2-item list/tuple."""
    if isinstance(item, dict):
        from_agent = str(item.get("from", "")).strip()
        to_agent = str(item.get("to", "")).strip()
        return (from_agent, to_agent) if from_agent and to_agent else None
    if isinstance(item, (list, tuple)) and len(item) == 2:
        from_agent = str(item[0]).strip()
        to_agent = str(item[1]).strip()
        return (from_agent, to_agent) if from_agent and to_agent else None
    return None


def build_workflow_report(
    events: list[Any],
    validation_rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a workflow progress report and optional validation results.

    Args:
        events: Callback event objects or event dictionaries.
        validation_rules: Optional runtime rules dict.

    Supported validation_rules keys:
        - require_no_errors: bool
        - required_pauses: list[str]
        - required_completed_agents: list[str]
        - required_delegations: list[{'from': str, 'to': str}] or list[[from, to]]
        - min_llm_delegations: int
    """
    records = [_event_to_dict(event) for event in events]

    completed_agents: list[str] = []
    paused_agents: list[str] = []
    errors: list[dict[str, Any]] = []
    delegations: list[dict[str, Any]] = []

    for record in records:
        event_type = record.get("type", "")
        payload = record.get("payload", {})

        if event_type == "agent_completed":
            agent_name = str(payload.get("agent", "")).strip()
            if agent_name:
                completed_agents.append(agent_name)
        elif event_type == "awaiting_user_input":
            agent_name = str(payload.get("agent", "")).strip()
            if agent_name:
                paused_agents.append(agent_name)
        elif event_type == "error":
            errors.append(payload)
        elif event_type == "delegated":
            delegations.append(
                {
                    "from_agent": str(payload.get("from_agent", "")).strip(),
                    "to_agent": str(payload.get("to_agent", "")).strip(),
                    "routing_mode": str(payload.get("routing_mode", "unknown")).strip() or "unknown",
                    "routing_reason": str(payload.get("routing_reason", "")).strip(),
                }
            )

    completed_unique = list(dict.fromkeys(completed_agents))
    paused_unique = list(dict.fromkeys(paused_agents))
    delegation_edges = [
        (item["from_agent"], item["to_agent"])
        for item in delegations
        if item["from_agent"] and item["to_agent"]
    ]
    delegation_edge_unique = list(dict.fromkeys(delegation_edges))

    routing_modes = Counter(item["routing_mode"] for item in delegations)
    total_delegations = len(delegations)
    llm_delegations = routing_modes.get("llm", 0)
    deterministic_delegations = routing_modes.get("deterministic", 0)

    llm_ratio = round(llm_delegations / total_delegations, 3) if total_delegations else 0.0
    deterministic_ratio = (
        round(deterministic_delegations / total_delegations, 3) if total_delegations else 0.0
    )

    report: dict[str, Any] = {
        "summary": {
            "event_count": len(records),
            "completed_agents_order": completed_agents,
            "completed_agents_unique": completed_unique,
            "paused_agents_order": paused_agents,
            "paused_agents_unique": paused_unique,
            "delegation_count": total_delegations,
            "delegation_edges_order": delegation_edges,
            "delegation_edges_unique": delegation_edge_unique,
            "error_count": len(errors),
        },
        "routing": {
            "by_mode": dict(routing_modes),
            "llm_delegations": llm_delegations,
            "deterministic_delegations": deterministic_delegations,
            "llm_ratio": llm_ratio,
            "deterministic_ratio": deterministic_ratio,
        },
        "delegations": delegations,
        "errors": errors,
        "validation": {
            "rules_applied": bool(validation_rules),
            "checks": {},
            "passed": True,
        },
        "warnings": [],
    }

    if total_delegations == 0:
        report["warnings"].append("No delegation events observed.")
    if llm_delegations == 0 and total_delegations > 0:
        report["warnings"].append(
            "No LLM-driven delegations observed; routing may be over-constrained."
        )
    if len(errors) > 0:
        report["warnings"].append("Workflow emitted one or more error events.")

    if validation_rules:
        checks: dict[str, Any] = {}

        require_no_errors = bool(validation_rules.get("require_no_errors", False))
        if require_no_errors:
            checks["require_no_errors"] = len(errors) == 0

        required_pauses = [
            str(item).strip()
            for item in (validation_rules.get("required_pauses") or [])
            if str(item).strip()
        ]
        for agent_name in required_pauses:
            checks[f"required_pause:{agent_name}"] = agent_name in paused_unique

        required_completed = [
            str(item).strip()
            for item in (validation_rules.get("required_completed_agents") or [])
            if str(item).strip()
        ]
        for agent_name in required_completed:
            checks[f"required_completed:{agent_name}"] = agent_name in completed_unique

        for item in (validation_rules.get("required_delegations") or []):
            normalized = _normalize_required_delegation(item)
            if normalized is None:
                continue
            from_agent, to_agent = normalized
            checks[f"required_delegation:{from_agent}->{to_agent}"] = (
                (from_agent, to_agent) in delegation_edge_unique
            )

        if "min_llm_delegations" in validation_rules:
            min_llm = int(validation_rules.get("min_llm_delegations", 0))
            checks["min_llm_delegations"] = llm_delegations >= min_llm

        report["validation"]["checks"] = checks
        report["validation"]["passed"] = all(checks.values()) if checks else True

    return report


def format_workflow_report(report: dict[str, Any]) -> str:
    """Format workflow report into human-readable text."""
    summary = report.get("summary", {})
    routing = report.get("routing", {})
    validation = report.get("validation", {})

    lines = [
        "WORKFLOW REPORT",
        "==============",
        f"Events: {summary.get('event_count', 0)}",
        f"Completed agents (order): {summary.get('completed_agents_order', [])}",
        f"Paused agents (order): {summary.get('paused_agents_order', [])}",
        f"Delegations: {summary.get('delegation_count', 0)}",
        f"Delegation edges: {summary.get('delegation_edges_order', [])}",
        f"Errors: {summary.get('error_count', 0)}",
        "",
        "Routing",
        "-------",
        f"By mode: {routing.get('by_mode', {})}",
        f"LLM ratio: {routing.get('llm_ratio', 0.0)}",
        f"Deterministic ratio: {routing.get('deterministic_ratio', 0.0)}",
        "",
        "Validation",
        "----------",
        f"Rules applied: {validation.get('rules_applied', False)}",
        f"Passed: {validation.get('passed', True)}",
    ]

    checks = validation.get("checks", {}) or {}
    if checks:
        lines.append("Checks:")
        for name, passed in checks.items():
            mark = "PASS" if passed else "FAIL"
            lines.append(f"- {mark}: {name}")

    warnings = report.get("warnings", []) or []
    if warnings:
        lines.append("")
        lines.append("Warnings")
        lines.append("--------")
        for item in warnings:
            lines.append(f"- {item}")

    return "\n".join(lines) + "\n"
