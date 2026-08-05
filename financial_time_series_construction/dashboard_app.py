"""Backend engine for the Financial Time Series Construction Streamlit dashboard.

This module provides all business logic used by ``dashboard.py`` (the
presentation layer).  It is deliberately free of ``streamlit`` imports so the
backend can be tested and reused independently.

Responsibilities
----------------
* Provider / model management - list available providers, list models per
  provider, and resolve defaults from the ``models.py`` environment config.
* Workflow session management - create a runtime bound to the user-selected
  provider/model and drive the same ``AgenticRuntime`` path as ``cli.py``.
* Agent definitions (from ``agents_definition.py``) and prompt library access
  (from ``prompt_library.py``).
* Human-in-the-loop pause/resume (mirrors the CLI's interactive loop).
* Agent output extraction - parse structured outputs (data-quality tables,
  timeseries artifacts) from callback events for the dashboard results area.
* Event / log capture and formatting for the dashboard progress UI.

``main()`` is a thin launcher that starts the Streamlit app (``dashboard.py``)
so the dashboard can be launched via ``python -m ...dashboard_app`` or from a
VS Code debug configuration.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from financial_time_series_construction.agents_definition import (
    AGENT_REGISTRY,
    CallbackEvent,
    CallbackEventType,
    get_agent,
)
from financial_time_series_construction.models import (
    SUPPORTED_PROVIDERS,
    ModelRequestFactory,
)
from financial_time_series_construction.prompt_library import (
    PROMPT_REGISTRY,
    get_prompts,
)
from financial_time_series_construction.database import (
    DataStore,
    get_datastore,
    init_datastore,
)
from financial_time_series_construction.runtime import AgenticRuntime, build_runtime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

PROVIDER_LABELS: dict[str, str] = {
    "ollama": "Ollama (local)",
    "github": "GitHub Models",
    "deepseek": "DeepSeek",
}

DEEPSEEK_KNOWN_MODELS: list[str] = ["deepseek-chat", "deepseek-reasoner"]

# Loggers captured in the progression tab.  Filters out Streamlit noise.
_CAPTURE_PREFIXES: tuple[str, ...] = (
    "financial_time_series_construction",
    "litellm",
    "autogen",
)

# Most recently created session (used by DataStore queries that need a run id).
_active_session: "WorkflowSession | None" = None

# Emoji icons for each event type, used by the dashboard events table.
EVENT_TYPE_ICONS: dict[str, str] = {
    CallbackEventType.USER_REQUEST.value: "📥",
    CallbackEventType.AWAITING_USER_INPUT.value: "⏸️",
    CallbackEventType.DATA_SOURCE_SELECTED.value: "📊",
    CallbackEventType.GAP_METHOD_RECOMMENDED.value: "💡",
    CallbackEventType.GAP_METHOD_APPLIED.value: "🧩",
    CallbackEventType.TIMESERIES_GENERATED.value: "📈",
    CallbackEventType.TIMESERIES_DOWNLOADED.value: "⬇️",
    CallbackEventType.AGENT_COMPLETED.value: "✅",
    CallbackEventType.DELEGATED.value: "🔀",
    CallbackEventType.ERROR.value: "❌",
    CallbackEventType.WORKFLOW_COMPLETED.value: "🎉",
}

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Convert values to a JSON-serializable structure without flattening."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _event_to_dict(event: CallbackEvent) -> dict[str, Any]:
    """Convert a ``CallbackEvent`` to a JSON-serialisable dict."""
    return {
        "type": event.type.value,
        "payload": _json_safe(event.payload),
        "session_id": event.session_id,
    }


def _utc_now() -> str:
    """Return the current UTC timestamp as a formatted string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _epoch_now() -> float:
    """Return the current epoch seconds."""
    return time.time()


# ---------------------------------------------------------------------------
# Model listing (self-contained, no fin_ai dependency)
# ---------------------------------------------------------------------------


def _normalize_provider_name(provider: str) -> str:
    """Normalise a provider value to a supported key."""
    value = (provider or "").strip().casefold()
    aliases = {
        "gh": "github",
        "github_copilot": "github",
        "copilot": "github",
        "deepseekai": "deepseek",
    }
    value = aliases.get(value, value)
    if value not in SUPPORTED_PROVIDERS:
        logger.warning("Unknown provider '%s'; falling back to ollama.", provider)
        return "ollama"
    return value


def _list_ollama_models(api_base: str | None = None) -> list[str]:
    """List models from a local Ollama instance via ``/api/tags``."""
    endpoint = (
        api_base
        or os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    ).rstrip("/")
    url = f"{endpoint}/api/tags"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "financial-tsc/1.0"})
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except Exception as exc:
        logger.warning("Failed to list Ollama models from %s: %s", url, exc)
        return []

    models = payload.get("models", [])
    return [
        str(model.get("name", "")).strip()
        for model in models
        if model.get("name")
    ]


def _list_github_models(api_key: str | None = None) -> list[str]:
    """List models from the GitHub Models catalog."""
    token = api_key or os.getenv("GITHUB_TOKEN", "")
    url = "https://models.github.ai/catalog/models"
    headers: dict[str, str] = {"User-Agent": "financial-tsc/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except Exception as exc:
        logger.warning("Failed to list GitHub models: %s", exc)
        return []

    if not isinstance(payload, list):
        return []

    return sorted(
        {
            str(item.get("id", "")).strip()
            for item in payload
            if isinstance(item, dict) and item.get("id")
        }
    )


def _list_deepseek_models(api_key: str | None = None) -> list[str]:
    """List DeepSeek models, falling back to known model identifiers."""
    token = api_key or os.getenv("DEEPSEEK_API_KEY", "")
    base_url = (
        os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    ).rstrip("/")
    url = f"{base_url}/models"
    headers: dict[str, str] = {"User-Agent": "financial-tsc/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
        data = payload.get("data") if isinstance(payload, dict) else payload
        if isinstance(data, list):
            model_ids = sorted(
                {
                    str(item.get("id", "")).strip()
                    for item in data
                    if isinstance(item, dict) and item.get("id")
                }
            )
            if model_ids:
                return model_ids
    except Exception as exc:
        logger.debug("Could not fetch DeepSeek model list from %s: %s", url, exc)

    return list(DEEPSEEK_KNOWN_MODELS)


def list_providers() -> list[str]:
    """Return the supported provider names."""
    return list(SUPPORTED_PROVIDERS)


def provider_labels() -> dict[str, str]:
    """Return human-readable labels for each supported provider."""
    return dict(PROVIDER_LABELS)


def list_available_models(
    provider: str,
    api_key: str | None = None,
    api_base: str | None = None,
) -> list[str]:
    """List available models for a provider.

    Args:
        provider: Provider name (``ollama``, ``github``, ``deepseek``).
        api_key: Optional API key / token for authenticated providers.
        api_base: Optional endpoint override (used by Ollama / DeepSeek).

    Returns:
        List of model identifiers, or an empty list when listing fails.
    """
    provider = _normalize_provider_name(provider)
    if provider == "ollama":
        return _list_ollama_models(api_base=api_base)
    if provider == "github":
        return _list_github_models(api_key=api_key)
    if provider == "deepseek":
        return _list_deepseek_models(api_key=api_key)
    return []


def get_default_model_name(provider: str) -> str:
    """Return the default raw model name for a provider.

    The default comes from ``LLM_MODEL`` (with provider prefix stripped) or the
    per-provider ``LLM_<PROVIDER>_MODEL`` environment defaults.  The resolved
    configuration is sourced from ``ModelRequestFactory.describe_environment()``
    so ``.env`` files (defaults + local overrides) are honoured.
    """
    provider = _normalize_provider_name(provider)
    config = ModelRequestFactory.describe_environment()
    full_model = str(config.get("model", ""))
    if full_model:
        prefix = f"{provider}/"
        if full_model.startswith(prefix):
            return full_model[len(prefix):]
        if "/" in full_model:
            # Model was configured for a different provider; use per-provider default.
            pass
        else:
            return full_model

    defaults = {
        "ollama": "qwen2.5:1.5b",
        "github": "gpt-4.1",
        "deepseek": "deepseek-chat",
    }
    return defaults.get(provider, "qwen2.5:1.5b")


def get_default_provider() -> str:
    """Return the default provider from the environment configuration.

    The default is resolved from ``LLM_PROVIDER`` via
    ``ModelRequestFactory.describe_environment()`` so ``.env`` files and local
    overrides are honoured.  Falls back to ``ollama`` when the configured
    provider is unknown or unsupported.
    """
    config = ModelRequestFactory.describe_environment()
    env_provider = str(config.get("provider", "ollama"))
    return env_provider if env_provider in SUPPORTED_PROVIDERS else "ollama"


def get_provider_env_defaults(provider: str) -> dict[str, str]:
    """Return environment-derived default credentials / endpoints for a provider.

    Keeps environment variable knowledge in the backend so the UI never needs
    to reference ``os.getenv`` directly.

    Args:
        provider: Provider name (``ollama``, ``github``, ``deepseek``).

    Returns:
        Dict with ``api_base`` and/or ``api_key`` keys populated from the
        environment when the provider requires them.
    """
    provider = _normalize_provider_name(provider)
    defaults: dict[str, str] = {}
    if provider == "ollama":
        defaults["api_base"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    elif provider == "github":
        defaults["api_key"] = os.getenv("GITHUB_TOKEN", "")
    elif provider == "deepseek":
        defaults["api_key"] = os.getenv("DEEPSEEK_API_KEY", "")
        defaults["api_base"] = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    return defaults


def build_factory(
    provider: str,
    model: str,
    api_key: str | None = None,
    api_base: str | None = None,
) -> ModelRequestFactory:
    """Build a ``ModelRequestFactory`` from dashboard selections.

    Provider-specific kwargs (API keys / base URLs) are seeded from the
    environment configuration and overridden by UI-provided values.
    """
    config = ModelRequestFactory.describe_environment()
    provider_kwargs = dict(config.get("provider_kwargs") or {})

    if api_key:
        provider_kwargs["api_key"] = api_key
    if api_base:
        provider_kwargs["api_base"] = api_base

    full_model = model if "/" in model else f"{provider}/{model}"
    return ModelRequestFactory(
        model=full_model,
        provider=provider,
        temperature=float(config.get("temperature", 0.1)),
        max_tokens=int(config.get("max_tokens", 2048)),
        provider_kwargs=provider_kwargs,
    )


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------


def get_agent_names() -> list[str]:
    """Return all registered agent names."""
    return list(AGENT_REGISTRY.keys())


def get_agent_definition(name: str) -> dict[str, Any] | None:
    """Return a JSON-friendly agent definition for the dashboard.

    Args:
        name: Agent name (case-insensitive fuzzy match supported).

    Returns:
        Dict with name, description, system_prompt, tools, goal, guardrails.
    """
    agent = get_agent(name)
    if agent is None:
        return None
    return {
        "name": agent.name,
        "description": agent.description,
        "system_prompt": agent.system_prompt,
        "tools": list(agent.tools),
        "goal": agent.goal,
        "guardrails": list(agent.guardrails),
    }


# ---------------------------------------------------------------------------
# Prompt library access
# ---------------------------------------------------------------------------


def get_prompt_options() -> list[dict[str, str]]:
    """Flatten ``PROMPT_REGISTRY`` into selectable options.

    Each option carries ``key`` (``category:label``), ``category``, ``label``,
    ``description`` and the template ``response`` text.
    """
    options: list[dict[str, str]] = []
    for category, templates in PROMPT_REGISTRY.items():
        for template in templates:
            options.append(
                {
                    "key": f"{category}:{template.label}",
                    "category": category,
                    "label": template.label,
                    "description": template.description,
                    "response": template.response,
                }
            )
    return options


def resolve_prompt_text(key: str) -> str:
    """Resolve a prompt option key to its template response text."""
    if not key:
        return ""
    try:
        category, label = key.split(":", 1)
    except ValueError:
        return ""
    for template in PROMPT_REGISTRY.get(category, []):
        if template.label == label:
            return template.response
    return ""


def get_prompts_for_category(category: str) -> list[dict[str, str]]:
    """Return prompt templates for a category (pause context)."""
    return [
        {
            "label": template.label,
            "description": template.description,
            "response": template.response,
            "category": template.category,
        }
        for template in get_prompts(category)
    ]


def get_pause_category(agent_name: str | None) -> str:
    """Map a paused agent to its most relevant prompt-library category."""
    mapping = {
        "ReportingAgent": "source_selection",
        "GapFillingAgent": "gap_filling",
        "Orchestrator": "clarification",
    }
    if not agent_name:
        return "general"
    return mapping.get(agent_name, "general")


# ---------------------------------------------------------------------------
# Log capture
# ---------------------------------------------------------------------------


class SessionLogCapture(logging.Handler):
    """Capture workflow log records into a thread-safe session sink.

    Only records emitted by the financial_time_series_construction, litellm and
    autogen loggers are captured; Streamlit internals are excluded.
    """

    def __init__(self, session: "WorkflowSession") -> None:
        super().__init__(level=logging.INFO)
        self._session = session
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            name = record.name or ""
            if not any(name.startswith(prefix) for prefix in _CAPTURE_PREFIXES):
                return
            message = self.format(record)
            session = self._session
            with session._lock:
                session._log_lines.append(message)
                if len(session._log_lines) > 2000:
                    del session._log_lines[:200]
        except Exception:
            pass


def attach_log_capture(session: "WorkflowSession") -> None:
    """Attach a log capture handler to the root logger for a session."""
    if session._log_handler is not None:
        return
    handler = SessionLogCapture(session)
    session._log_handler = handler
    logging.getLogger().addHandler(handler)


def detach_log_capture(session: "WorkflowSession") -> None:
    """Detach the session log capture handler."""
    handler = session._log_handler
    if handler is not None:
        logging.getLogger().removeHandler(handler)
        session._log_handler = None


# ---------------------------------------------------------------------------
# Agent output extraction
# ---------------------------------------------------------------------------

# Event types / payload paths that carry structured agent outputs for the
# dashboard results area.
_DATA_QUALITY_EVENT_TYPES = {
    CallbackEventType.AWAITING_USER_INPUT.value,
    CallbackEventType.AGENT_COMPLETED.value,
}

_GAP_FILLING_EVENT_TYPES = {
    CallbackEventType.AWAITING_USER_INPUT.value,
    CallbackEventType.AGENT_COMPLETED.value,
}

_TIMESERIES_EVENT_TYPES = {
    CallbackEventType.AGENT_COMPLETED.value,
    CallbackEventType.TIMESERIES_GENERATED.value,
}

_DATA_QUALITY_SOURCE_AGENTS = {"DataQualityAgent", "ReportingAgent"}
_GAP_FILLING_SOURCE_AGENTS = {"GapFillingAgent"}
_TIMESERIES_SOURCE_AGENTS = {"TimeSeriesConstructionAgent", "ReportingAgent"}


def _extract_data_quality_output(
    event_type: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract a data-quality report from an event payload.

    Looks for ``data_quality_report`` in:
    * ``payload.context`` (pause checkpoint context)
    * ``payload.result`` (agent completion result)

    Returns a dict shaped for the dashboard UI, or ``None`` if the event does
    not carry a data-quality report.
    """
    report: dict[str, Any] | None = None

    context = payload.get("context")
    if isinstance(context, dict):
        candidate = context.get("data_quality_report")
        if isinstance(candidate, dict) and candidate.get("rows"):
            report = candidate

    if report is None:
        result = payload.get("result")
        if isinstance(result, dict):
            candidate = result.get("data_quality_report")
            if isinstance(candidate, dict) and candidate.get("rows"):
                report = candidate

    if report is None:
        return None

    # Normalise keys for the dashboard.
    summary = report.get("summary") or {}
    return {
        "agent": payload.get("agent", "DataQualityAgent"),
        "event_type": event_type,
        "report_type": report.get("report_type", "data_quality_summary"),
        "rows": list(report.get("rows") or []),
        "summary": {
            "symbol": summary.get("symbol"),
            "source_count": summary.get("source_count"),
            "sources": list(summary.get("sources") or []),
            "unavailable_source_count": summary.get("unavailable_source_count"),
            "unavailable_sources": list(summary.get("unavailable_sources") or []),
            "total_available_records": summary.get("total_available_records"),
            "total_missing_count": summary.get("total_missing_count"),
            "min_date": summary.get("min_date"),
            "max_date": summary.get("max_date"),
            "min_value": summary.get("min_value"),
            "max_value": summary.get("max_value"),
            "average_completeness_pct": summary.get("average_completeness_pct"),
            "best_source_by_completeness": summary.get("best_source_by_completeness"),
            "worst_source_by_completeness": summary.get("worst_source_by_completeness"),
        },
        "unavailable_sources": list(report.get("unavailable_sources") or []),
    }


def _extract_gap_filling_output(
    event_type: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract a gap-filling summary from an event payload.

    Looks for ``gap_filling_report`` in:
    * ``payload.result`` (agent completion result)
    * ``payload.context`` (pause checkpoint context)

    Returns a dict shaped for the dashboard UI, or ``None`` if the event does
    not carry a gap-filling report.
    """
    report: dict[str, Any] | None = None

    result = payload.get("result")
    if isinstance(result, dict):
        candidate = result.get("gap_filling_report")
        if isinstance(candidate, dict) and candidate.get("method"):
            report = candidate

    if report is None:
        context = payload.get("context")
        if isinstance(context, dict):
            candidate = context.get("gap_filling_report")
            if isinstance(candidate, dict) and candidate.get("method"):
                report = candidate

    if report is None:
        return None

    return {
        "agent": payload.get("agent", "GapFillingAgent"),
        "event_type": event_type,
        "symbol": report.get("symbol"),
        "source": report.get("source"),
        "method": report.get("method"),
        "data_ref": report.get("data_ref"),
    }


def _extract_timeseries_output(
    event_type: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract timeseries artifact info from an event payload.

    Looks for ``timeseries_csv`` / ``timeseries_chart`` in:
    * ``payload.result`` (agent completion result)
    * ``payload`` directly (e.g. TIMESERIES_GENERATED events)

    Returns a dict shaped for the dashboard UI, or ``None``.
    """
    csv_path: str | None = None
    chart_path: str | None = None
    agent: str | None = payload.get("agent")

    result = payload.get("result")
    if isinstance(result, dict):
        csv_value = result.get("timeseries_csv") or result.get("csv_path")
        chart_value = result.get("timeseries_chart") or result.get("chart_path")
        if isinstance(csv_value, str) and csv_value:
            csv_path = csv_value
        if isinstance(chart_value, str) and chart_value:
            chart_path = chart_value

    if csv_path is None:
        direct_csv = payload.get("timeseries_csv") or payload.get("csv_path")
        if isinstance(direct_csv, str) and direct_csv:
            csv_path = direct_csv
    if chart_path is None:
        direct_chart = payload.get("timeseries_chart") or payload.get("chart_path")
        if isinstance(direct_chart, str) and direct_chart:
            chart_path = direct_chart

    if csv_path is None and chart_path is None:
        return None

    symbol: str | None = None
    method: str | None = None
    if isinstance(result, dict):
        symbol = result.get("timeseries_symbol") or result.get("symbol")
        method = result.get("gap_filling_method") or result.get("method")

    return {
        "agent": agent or "TimeSeriesConstructionAgent",
        "event_type": event_type,
        "csv_path": csv_path,
        "chart_path": chart_path,
        "symbol": symbol,
        "method": method,
    }


def extract_agent_outputs(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract structured agent outputs from a list of event dicts.

    Scans callback events for payloads produced by specific agents and
    normalises them into a list of output records consumed by the dashboard
    results area.

    Args:
        events: List of event dicts (from ``_event_to_dict``).

    Returns:
        List of output records. Each record has a ``kind`` field
        (``data_quality``, ``gap_filling`` or ``timeseries``) plus
        agent-specific fields.
    """
    outputs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for event in events:
        event_type = str(event.get("type", ""))
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            continue

        agent_name = str(payload.get("agent", "") or "")

        # ── Data quality outputs ──────────────────────────────────────────
        if (
            event_type in _DATA_QUALITY_EVENT_TYPES
            and agent_name in _DATA_QUALITY_SOURCE_AGENTS
        ):
            quality_output = _extract_data_quality_output(event_type, payload)
            if quality_output is not None:
                dedup_key = ("data_quality", str(quality_output.get("symbol", "")))
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    quality_output["kind"] = "data_quality"
                    outputs.append(quality_output)

        # ── Gap filling outputs ───────────────────────────────────────────
        if (
            event_type in _GAP_FILLING_EVENT_TYPES
            and agent_name in _GAP_FILLING_SOURCE_AGENTS
        ):
            gap_output = _extract_gap_filling_output(event_type, payload)
            if gap_output is not None:
                dedup_key = ("gap_filling", str(gap_output.get("symbol", "")))
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    gap_output["kind"] = "gap_filling"
                    outputs.append(gap_output)

        # ── Timeseries artifact outputs ───────────────────────────────────
        if (
            event_type in _TIMESERIES_EVENT_TYPES
            and agent_name in _TIMESERIES_SOURCE_AGENTS
        ):
            ts_output = _extract_timeseries_output(event_type, payload)
            if ts_output is not None:
                dedup_key = ("timeseries", str(ts_output.get("csv_path", "")))
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    ts_output["kind"] = "timeseries"
                    outputs.append(ts_output)

    return outputs


def get_agent_outputs(session: "WorkflowSession") -> list[dict[str, Any]]:
    """Return the structured agent outputs for a session (thread-safe)."""
    with session._lock:
        return list(session.agent_outputs)


def get_agent_output_json(session: "WorkflowSession") -> str:
    """Return the agent outputs as a JSON string (thread-safe)."""
    with session._lock:
        return json.dumps(session.agent_outputs, indent=2, default=str)


def has_agent_outputs(session: "WorkflowSession") -> bool:
    """Return whether the session has any structured agent outputs."""
    with session._lock:
        return bool(session.agent_outputs)


# ---------------------------------------------------------------------------
# Workflow session
# ---------------------------------------------------------------------------


@dataclass
class WorkflowSession:
    """State container for a dashboard workflow session."""

    session_id: str
    provider: str
    model: str
    runtime: AgenticRuntime | None = None
    status: str = "idle"  # idle | running | paused | completed | cancelled | error
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    current_agent: str | None = None
    pause_prompt: str | None = None
    pause_options: list[str] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None
    started_epoch: float = 0.0
    run_count: int = 0
    agent_outputs: list[dict[str, Any]] = field(default_factory=list)
    _thread: threading.Thread | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _log_lines: list[str] = field(default_factory=list)
    _log_handler: SessionLogCapture | None = None

    def reset_run(self) -> None:
        """Reset transient run state but keep the runtime for pause/resume."""
        with self._lock:
            self.status = "idle"
            self.error = None
            self.current_agent = None
            self.pause_prompt = None
            self.pause_options = []
            self.started_at = None
            self.completed_at = None
            self.started_epoch = 0.0
            self._thread = None

    def copy_log_lines(self) -> list[str]:
        """Return a thread-safe copy of captured log lines."""
        with self._lock:
            return list(self._log_lines)


def _extract_pause_info(event_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract the latest pause prompt and options from event dicts."""
    for event in reversed(event_dicts):
        if event.get("type") != CallbackEventType.AWAITING_USER_INPUT.value:
            continue
        payload = event.get("payload", {})
        return {
            "prompt": payload.get("prompt", ""),
            "options": list(payload.get("options") or []),
        }
    return {"prompt": "", "options": []}


def _run_worker(
    session: WorkflowSession,
    user_input: str,
    is_response: bool,
) -> None:
    """Background worker driving the agentic runtime.

    Runs in a daemon thread so Streamlit remains responsive while the workflow
    executes (LLM calls can take 10-30 seconds each).
    """
    try:
        runtime = session.runtime
        if runtime is None:
            raise RuntimeError("Session runtime is not initialised.")

        if is_response:
            events = runtime.process_user_response(user_input)
        else:
            events = runtime.process_user_request(user_input)

        event_dicts = [_event_to_dict(event) for event in events]
        log_lines = format_events_to_log_lines(events)

        with session._lock:
            session.events.extend(event_dicts)
            session._log_lines.extend(log_lines)

            # Extract structured agent outputs from this batch of events and
            # append them to the session-level output list.  Stages supersede
            # earlier stages so older results are replaced rather than
            # accumulated (data_quality → gap_filling → timeseries).
            batch_outputs = extract_agent_outputs(event_dicts)
            if batch_outputs:
                stage_order = {
                    "data_quality": 0,
                    "gap_filling": 1,
                    "timeseries": 2,
                }
                batch_stage = min(
                    stage_order.get(output.get("kind", ""), -1)
                    for output in batch_outputs
                )
                if batch_stage >= 0:
                    session.agent_outputs = [
                        output
                        for output in session.agent_outputs
                        if stage_order.get(output.get("kind", ""), -1) < batch_stage
                    ]
                session.agent_outputs.extend(batch_outputs)

            if runtime.waiting_for_input:
                session.status = "paused"
                session.current_agent = runtime.current_agent
                pause = _extract_pause_info(event_dicts)
                session.pause_prompt = pause["prompt"]
                session.pause_options = pause["options"]
            else:
                cancelled = any(
                    event.get("type") == CallbackEventType.ERROR.value
                    and "cancelled"
                    in str(
                        event.get("payload", {}).get("message", "")
                    ).casefold()
                    for event in event_dicts
                )
                session.status = "cancelled" if cancelled else "completed"
                session.current_agent = None
                session.pause_prompt = None
                session.pause_options = []

            session.completed_at = _utc_now()
    except Exception as exc:
        logger.exception(
            "dashboard_run_failed session_id=%s", session.session_id
        )
        with session._lock:
            session.status = "error"
            session.error = f"{type(exc).__name__}: {exc}"
            session.completed_at = _utc_now()
    finally:
        detach_log_capture(session)


def create_session(
    provider: str,
    model: str,
    api_key: str | None = None,
    api_base: str | None = None,
) -> WorkflowSession:
    """Create a new workflow session bound to the selected provider/model."""
    global _active_session
    provider = _normalize_provider_name(provider)
    session_id = (
        f"tsc_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{int(_epoch_now() * 1000) % 10000}"
    )
    factory = build_factory(provider, model, api_key=api_key, api_base=api_base)
    runtime = build_runtime("autogen", session_id=session_id, factory=factory)
    logger.info(
        "dashboard_session_created session_id=%s provider=%s model=%s",
        session_id,
        provider,
        model,
    )
    session = WorkflowSession(
        session_id=session_id,
        provider=provider,
        model=model,
        runtime=runtime,
    )
    _active_session = session
    return session


def start_run(session: WorkflowSession, user_input: str) -> None:
    """Start a new workflow run with the given user request.

    The request is processed in a background thread; the UI polls the session
    via :func:`is_running` / :func:`get_session_status`.
    """
    if session.runtime is None:
        raise RuntimeError("Session runtime is not initialised.")
    if is_running(session):
        raise RuntimeError("A workflow run is already in progress.")

    session.reset_run()
    with session._lock:
        session.run_count += 1
        session.started_at = _utc_now()
        session.started_epoch = _epoch_now()
        session.status = "running"
        session._log_lines.append(
            f"═══ Run #{session.run_count} | {session.provider}/{session.model} | "
            f"{session.started_at} ═══"
        )

    attach_log_capture(session)
    thread = threading.Thread(
        target=_run_worker,
        args=(session, user_input, False),
        daemon=True,
        name=f"tsc-dashboard-{session.session_id}",
    )
    session._thread = thread
    thread.start()


def submit_response(session: WorkflowSession, user_input: str) -> None:
    """Submit a response to a paused workflow (resume execution).

    Accepts free-form text or quick-option selections.  ``exit`` / ``quit`` /
    ``cancel`` cancels the paused workflow (mirrors CLI behaviour).
    """
    if session.runtime is None:
        raise RuntimeError("Session runtime is not initialised.")
    if is_running(session):
        raise RuntimeError("A workflow run is already in progress.")
    if session.status != "paused":
        raise RuntimeError("Session is not paused; cannot submit a response.")

    with session._lock:
        session.status = "running"
        session.started_at = _utc_now()
        session.started_epoch = _epoch_now()
        session._log_lines.append(
            f"── response | {session.started_at} ──"
        )

    attach_log_capture(session)
    thread = threading.Thread(
        target=_run_worker,
        args=(session, user_input, True),
        daemon=True,
        name=f"tsc-dashboard-{session.session_id}-resume",
    )
    session._thread = thread
    thread.start()


def is_running(session: WorkflowSession) -> bool:
    """Return whether a workflow run is currently executing."""
    thread = session._thread
    return thread is not None and thread.is_alive()


def get_session_status(session: WorkflowSession) -> dict[str, Any]:
    """Return a thread-safe status snapshot for the UI."""
    with session._lock:
        running = is_running(session)
        return {
            "running": running,
            "status": "running" if running else session.status,
            "provider": session.provider,
            "model": session.model,
            "session_id": session.session_id,
            "error": session.error,
            "current_agent": session.current_agent,
            "pause_prompt": session.pause_prompt,
            "pause_options": list(session.pause_options),
            "started_at": session.started_at,
            "completed_at": session.completed_at,
            "run_count": session.run_count,
            "events_count": len(session.events),
            "log_lines_count": len(session._log_lines),
            "agent_outputs_count": len(session.agent_outputs),
        }


def session_bound_to(
    session: WorkflowSession,
    provider: str,
    model: str,
) -> bool:
    """Return whether a session is bound to the given provider/model pair.

    Thread-safe alternative to reading ``session.provider`` / ``session.model``
    directly from the UI layer.
    """
    with session._lock:
        return session.provider == provider and session.model == model


def stop_session(session: WorkflowSession) -> None:
    """Clean up a session (detach log capture, clear runtime)."""
    detach_log_capture(session)
    with session._lock:
        session.runtime = None
        session.status = "idle"
        session.events = []
        session._log_lines = []
        session.agent_outputs = []
        session._thread = None


# ---------------------------------------------------------------------------
# DataStore communication
# ---------------------------------------------------------------------------


def _get_datastore() -> DataStore:
    """Return the global DataStore singleton, initialising it lazily.

    Uses the same ``OUTPUT_ROOT`` / ``time_series_database/database/``
    layout as ``tools.py`` so the dashboard reads and writes the same
    SQLite database as the CLI / agentic workflow.
    """
    try:
        return get_datastore()
    except RuntimeError:
        from financial_time_series_construction.tools import OUTPUT_ROOT

        db_path = OUTPUT_ROOT / "time_series_database" / "database" / "datastore.db"
        init_datastore(db_path)
        return get_datastore()


def get_populated_timeseries(
    run_id: str | None = None,
    symbol: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Return populated time series data (before and after gap-filling).

    Delegates to the ``get_populated_timeseries`` tool, which sources both
    the ``raw_timeseries`` (populated before) and ``filled_timeseries``
    (populated after) tables from the DataStore.

    Args:
        run_id: The run/session identifier.  Defaults to the most recent
            session's run id when available.
        symbol: Optional ticker symbol filter.
        source: Optional data source name filter.

    Returns:
        Dict with ``run_id``, ``symbols``, ``sources``, ``before`` and
        ``after`` series lists.

    Raises:
        ValueError: If no run_id can be resolved.
    """
    from financial_time_series_construction.tools import (
        get_populated_timeseries as _tool_get_populated_timeseries,
        set_run_id,
    )

    resolved_run_id = run_id
    if resolved_run_id is None:
        active_session = _get_current_session()
        if active_session is not None and active_session.session_id:
            resolved_run_id = active_session.session_id

    if not resolved_run_id:
        raise ValueError(
            "No run_id available. Pass an explicit run_id or start a "
            "workflow session to query the DataStore."
        )

    try:
        set_run_id(resolved_run_id)
    except Exception:
        pass

    return _tool_get_populated_timeseries(
        run_id=resolved_run_id,
        symbol=symbol,
        source=source,
    )


def find_run_by_id(run_id: str) -> dict[str, Any]:
    """Find a run record in the DataStore by its ``run_id``.

    Args:
        run_id: The run/session identifier to look up.

    Returns:
        Dict with ``found`` boolean plus run metadata (``run_id``,
        ``start_date``, ``end_date``, ``created_at``, ``updated_at``,
        ``timeseries_count``, ``filled_count``, ``artifact_count``).
    """
    store = _get_datastore()
    try:
        return {"found": True, **store.get_run(run_id)}
    except KeyError:
        return {"found": False, "run_id": run_id}


def list_runs_with_stats() -> list[dict[str, Any]]:
    """List all runs in the DataStore with summary statistics."""
    store = _get_datastore()
    return store.list_runs_with_stats()


def _get_current_session() -> "WorkflowSession | None":
    """Return the most recently created session (module-level registry)."""
    return _active_session


# ---------------------------------------------------------------------------
# Event / log formatting
# ---------------------------------------------------------------------------


def format_event_to_log_line(event: CallbackEvent) -> str:
    """Format a callback event as a single human-readable log line."""
    event_type = event.type
    payload = event.payload
    agent = payload.get("agent", "System")

    if event_type == CallbackEventType.AWAITING_USER_INPUT:
        prompt = payload.get("prompt", "")
        options = payload.get("options") or []
        line = f"[PAUSE] Agent: {agent} | {prompt}"
        if options:
            line += f" | Options: {', '.join(str(o) for o in options)}"
        return line

    if event_type == CallbackEventType.AGENT_COMPLETED:
        result = payload.get("result", "")
        if isinstance(result, dict) and "final_answer" in result:
            return f"[COMPLETE] Agent: {agent} | {result['final_answer']}"
        return f"[COMPLETE] Agent: {agent}"

    if event_type == CallbackEventType.ERROR:
        return f"[ERROR] Agent: {agent} | {payload.get('message', 'Unknown error')}"

    if event_type == CallbackEventType.DELEGATED:
        return (
            f"[DELEGATE] {payload.get('from_agent')} → "
            f"{payload.get('to_agent')}"
        )

    if event_type == CallbackEventType.USER_REQUEST:
        return f"[REQUEST] {payload.get('request', '')}"

    return f"[{event_type.value.upper()}] Agent: {agent}"


def format_events_to_log_lines(events: list[CallbackEvent]) -> list[str]:
    """Format a list of callback events into log lines."""
    return [format_event_to_log_line(event) for event in events]


def get_log_text(session: WorkflowSession) -> str:
    """Return the full captured log text for the progression tab."""
    with session._lock:
        return "\n".join(session._log_lines)


def get_events_json(session: WorkflowSession) -> str:
    """Return the event history as a JSON string for the events tab."""
    with session._lock:
        return json.dumps(session.events, indent=2, default=str)


def get_events(session: WorkflowSession) -> list[dict[str, Any]]:
    """Return the event history as a list (thread-safe)."""
    with session._lock:
        return list(session.events)


def get_trace_text(session: WorkflowSession) -> str:
    """Return the runtime ReACT trace text."""
    if session.runtime is None:
        return ""
    return session.runtime.get_trace()


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the Streamlit dashboard (``dashboard.py``).

    Used by the VS Code launch configuration
    (``financial_time_series_construction.dashboard_app``) and by
    ``python -m financial_time_series_construction.dashboard_app``.
    """
    import sys

    from streamlit.web import cli as stcli

    dashboard_path = Path(__file__).resolve().parent / "dashboard.py"
    sys.argv = [
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port",
        os.getenv("STREAMLIT_PORT", "8501"),
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()