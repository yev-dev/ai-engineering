"""Streamlit dashboard for the Financial Time Series Construction workflow.

This is the presentation layer.  All business logic lives in
``dashboard_app.py``; this module only wires UI components to backend calls.

Layout
------
Left panel  : model provider selection -> populates the available model list.
Main panel  : optional prompt-library selection -> request text box -> agent
              selection with Run button -> progress bar -> log tabs.
Outputs     : Agent output placeholder area where each agent's structured
              results are rendered as they become available (data-quality
              tables, timeseries artifact summaries + downloads).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import streamlit as st

from financial_time_series_construction.dashboard_app import (
    WorkflowSession,
    create_session,
    get_agent_definition,
    get_agent_names,
    get_agent_outputs,
    get_default_model_name,
    get_default_provider,
    get_events,
    get_events_json,
    get_log_text,
    get_pause_category,
    get_prompt_options,
    get_prompts_for_category,
    get_provider_env_defaults,
    get_session_status,
    get_trace_text,
    list_available_models,
    list_providers,
    provider_labels,
    resolve_prompt_text,
    session_bound_to,
    start_run,
    stop_session,
    submit_response,
)

st.set_page_config(
    page_title="Financial Time Series Construction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Financial Time Series Construction")
st.caption(
    "Run the DeFi time-series construction ReACT workflow via "
    "provider-backed LLM agents — an alternative to the CLI."
)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

_SESSION_KEY = "tsc_session"
_SELECTED_PROMPT_KEY = "selected_prompt"
_REQUEST_KEY = "request_text"
_SELECTED_AGENT_KEY = "selected_agent"
_APPLIED_PROMPT_KEY = "applied_prompt_key"
_POLLING_KEY = "tsc_polling"
_PAUSE_QUICK_KEY = "pause_quick_options"
_PAUSE_RESPONSE_KEY = "pause_response_text"

# Only the "clarification" templates are full user requests suitable for the
# initial request box. Source-selection / gap-filling prompts are checkpoint
# responses used during human-in-the-loop pauses, not standalone requests, and
# the "custom" placeholder is likewise only meaningful at a checkpoint.
_PROMPT_OPTIONS: list[dict[str, str]] = [
    option
    for option in get_prompt_options()
    if option["category"] == "clarification" and option["label"] != "custom"
]
_PROMPT_KEYS: list[str] = [""] + [option["key"] for option in _PROMPT_OPTIONS]
_PROMPT_LOOKUP: dict[str, dict[str, str]] = {
    option["key"]: option for option in _PROMPT_OPTIONS
}


def _get_session() -> WorkflowSession | None:
    """Return the active workflow session from Streamlit state."""
    return st.session_state.get(_SESSION_KEY)


def _set_session(session: WorkflowSession | None) -> None:
    """Store the active workflow session in Streamlit state."""
    st.session_state[_SESSION_KEY] = session


def _reset_session_state() -> None:
    """Stop the active session and clear all dashboard state."""
    session = _get_session()
    if session is not None:
        stop_session(session)
    keys_to_remove = [
        key
        for key in list(st.session_state.keys())
        if key in {
            _SESSION_KEY,
            _SELECTED_PROMPT_KEY,
            _REQUEST_KEY,
            _SELECTED_AGENT_KEY,
            _APPLIED_PROMPT_KEY,
            _POLLING_KEY,
            _PAUSE_QUICK_KEY,
            _PAUSE_RESPONSE_KEY,
        }
        or key.startswith(f"{_PAUSE_QUICK_KEY}:")
        or key.startswith(f"{_PAUSE_RESPONSE_KEY}:")
    ]
    for key in keys_to_remove:
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Agent output rendering helpers
# ---------------------------------------------------------------------------


def _render_data_quality_output(output: dict[str, Any]) -> None:
    """Render a data-quality report output from DataQualityAgent."""
    summary = output.get("summary") or {}
    rows = output.get("rows") or []
    agent = output.get("agent", "DataQualityAgent")

    st.markdown(f"#### 📊 Data Quality Report — `{agent}`")

    # Summary metrics row.
    symbol = summary.get("symbol") or "—"
    source_count = summary.get("source_count") or len(rows)
    avg_completeness = summary.get("average_completeness_pct")
    best_source = summary.get("best_source_by_completeness")
    worst_source = summary.get("worst_source_by_completeness")
    total_available = summary.get("total_available_records")
    total_missing = summary.get("total_missing_count")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Symbol", symbol)
    with col2:
        st.metric("Sources", source_count)
    with col3:
        st.metric(
            "Avg Completeness",
            f"{avg_completeness}%" if avg_completeness is not None else "—",
        )
    with col4:
        st.metric(
            "Best Source",
            best_source or "—",
            help=f"Worst: {worst_source or '—'}",
        )

    if total_available is not None or total_missing is not None:
        col5, col6 = st.columns(2)
        with col5:
            st.metric("Available Records", total_available if total_available is not None else "—")
        with col6:
            st.metric("Missing Records", total_missing if total_missing is not None else "—")

    # Render the quality table.
    if rows:
        st.markdown("**Source Quality Comparison**")
        table_data = []
        for row in rows:
            table_data.append(
                {
                    "Source": row.get("source", "—"),
                    "Symbol": row.get("symbol", "—"),
                    "Total Values": row.get("total_values", "—"),
                    "Available": row.get("available_record_count", "—"),
                    "Missing": row.get("missing_count", row.get("nan_count", "—")),
                    "Completeness %": row.get("completeness_pct", "—"),
                    "Min Value": row.get("min_value", "—"),
                    "Max Value": row.get("max_value", "—"),
                    "Min Date": row.get("min_date", "—"),
                    "Max Date": row.get("max_date", "—"),
                    "Duplicates": row.get("duplicate_count", "—"),
                    "Issues": ", ".join(str(i) for i in (row.get("issues") or [])) or "none",
                }
            )
        st.dataframe(table_data, use_container_width=True, hide_index=True)
    else:
        st.info("No quality rows available in the report.")

    # Unavailable sources.
    unavailable = output.get("unavailable_sources") or []
    if unavailable:
        st.markdown("**Unavailable Sources**")
        for item in unavailable:
            source = item.get("source", "unknown")
            reason = item.get("reason", "unavailable")
            st.markdown(f"- `{source}` — {reason}")


def _render_gap_filling_output(output: dict[str, Any]) -> None:
    """Render a gap-filling summary output from GapFillingAgent."""
    agent = output.get("agent", "GapFillingAgent")
    symbol = output.get("symbol") or "—"
    source = output.get("source") or "—"
    method = output.get("method") or "—"
    data_ref = output.get("data_ref")

    st.markdown(f"#### 🧩 Gap Filling Applied — `{agent}`")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Symbol", symbol)
    with col2:
        st.metric("Source", source)
    with col3:
        st.metric("Method", method)

    if data_ref:
        st.caption(f"Data reference: `{data_ref}`")


def _render_timeseries_output(output: dict[str, Any]) -> None:
    """Render a timeseries artifact output from TimeSeriesConstructionAgent."""
    agent = output.get("agent", "TimeSeriesConstructionAgent")
    csv_path = output.get("csv_path")
    chart_path = output.get("chart_path")
    symbol = output.get("symbol")
    method = output.get("method")

    st.markdown(f"#### 📈 Time Series Artifacts — `{agent}`")

    if symbol or method:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Symbol", symbol or "—")
        with col2:
            st.metric("Gap Filling Method", method or "—")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Summary**")
        st.markdown(
            f"- **CSV artifact:** `{csv_path or '—'}`\n"
            f"- **Chart artifact:** `{chart_path or '—'}`"
        )

    # CSV download button.
    if csv_path:
        csv_file = Path(csv_path)
        if csv_file.exists():
            with col2:
                st.markdown("**Download**")
                try:
                    csv_bytes = csv_file.read_bytes()
                    st.download_button(
                        label="⬇️ Download Time Series CSV",
                        data=csv_bytes,
                        file_name=csv_file.name,
                        mime="text/csv",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.warning(f"Could not read CSV file: {exc}")
        else:
            with col2:
                st.caption(f"CSV file not found at `{csv_path}`")

    # Chart image if available.
    if chart_path:
        chart_file = Path(chart_path)
        if chart_file.exists():
            st.image(str(chart_file), caption="Time Series Chart", use_container_width=True)


def _render_agent_outputs(session: WorkflowSession | None) -> None:
    """Render the agent output placeholder area.

    This is the central results area where each agent's structured output is
    presented as it becomes available during the workflow run.
    """
    st.subheader("Agent Outputs")

    if session is None:
        st.info(
            "No agent outputs yet. Start a workflow run to see structured "
            "results from each agent here."
        )
        return

    outputs = get_agent_outputs(session)

    if not outputs:
        status = get_session_status(session)
        if status.get("running"):
            st.info("⏳ Waiting for agent outputs…")
        else:
            st.info(
                "No agent outputs yet. When DataQualityAgent finishes, its "
                "quality table will appear here. When TimeSeriesConstructionAgent "
                "finishes, its summary and downloadable CSV will appear here."
            )
        return

    # Render each output in order.
    for index, output in enumerate(outputs):
        kind = output.get("kind", "")

        if index > 0:
            st.divider()

        if kind == "data_quality":
            _render_data_quality_output(output)
        elif kind == "gap_filling":
            _render_gap_filling_output(output)
        elif kind == "timeseries":
            _render_timeseries_output(output)
        else:
            st.json(output)


# ---------------------------------------------------------------------------
# Sidebar: provider / model selection
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Model Provider")

    providers = list_providers()
    labels = provider_labels()

    provider_keys = [labels.get(p, p) for p in providers]
    label_to_key = {labels.get(p, p): p for p in providers}

    # Default provider from environment (LLM_PROVIDER in .env).
    default_provider = get_default_provider()
    default_provider_label = labels.get(default_provider, default_provider)

    provider_label = st.selectbox(
        "Provider",
        provider_keys,
        index=provider_keys.index(default_provider_label)
        if default_provider_label in provider_keys
        else 0,
    )
    provider = label_to_key[provider_label]

    # Provider-specific credentials / endpoints.
    env_defaults = get_provider_env_defaults(provider)
    api_key: str | None = None
    api_base: str | None = None

    if provider == "ollama":
        api_base = st.text_input(
            "Ollama API Base URL",
            value=env_defaults.get("api_base", "http://localhost:11434"),
        ) or None
    elif provider == "github":
        api_key = st.text_input(
            "GitHub Token",
            value=env_defaults.get("api_key", ""),
            type="password",
        ) or None
    elif provider == "deepseek":
        api_key = st.text_input(
            "DeepSeek API Key",
            value=env_defaults.get("api_key", ""),
            type="password",
        ) or None
        api_base = st.text_input(
            "DeepSeek API Base URL",
            value=env_defaults.get("api_base", "https://api.deepseek.com"),
        ) or None

    # Model listing derived from provider selection.
    with st.spinner("Fetching available models..."):
        models = list_available_models(
            provider,
            api_key=api_key,
            api_base=api_base,
        )

    default_model = get_default_model_name(provider)
    all_models: list[str] = list(dict.fromkeys(models + [default_model]))
    model_index = 0
    if default_model in all_models:
        model_index = all_models.index(default_model)

    if all_models:
        selected_model = st.selectbox(
            "Model",
            all_models,
            index=model_index,
        )
        if not models:
            st.caption("Model list unavailable — using defaults.")
    else:
        selected_model = st.text_input(
            "Model",
            value=default_model,
            help="No models were discovered. Enter a model name manually.",
        )

    st.divider()
    st.caption("Session persistence is active. Run results are captured in the main panel.")

    if st.button("Reset Session", use_container_width=True):
        _reset_session_state()
        st.rerun()

    with st.expander("About"):
        st.markdown(
            """
            This dashboard mirrors the financial time-series construction CLI.
            It uses the same `AgenticRuntime` (autogen ReACT workflow) and the
            same prompt library.
            """
        )


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

main_col_left, main_col_right = st.columns([3, 2], gap="large")

# ── Left column: prompt, request, agent select, run ───────────────────────
with main_col_left:
    st.header("Workflow Request")

    # Dropdown: optional PROMPT_REGISTRY selection.
    prompt_caption = "Optional: select a template from the prompt library."
    prompt_key = st.selectbox(
        "Prompt Template (optional)",
        _PROMPT_KEYS,
        key=_SELECTED_PROMPT_KEY,
        format_func=lambda key: (
            "— Select a prompt template —"
            if not key
            else f"{_PROMPT_LOOKUP[key]['category']} / {_PROMPT_LOOKUP[key]['label']}"
            + f" — {_PROMPT_LOOKUP[key]['description']}"
        ),
        help=prompt_caption,
    )

    # Populate the request box BEFORE the text area is instantiated. Streamlit
    # forbids modifying a widget's session-state value after the widget has been
    # created, so the template text must be set here (before st.text_area below).
    # The applied-prompt marker prevents overwriting the user's own edits.
    if prompt_key and st.session_state.get(_APPLIED_PROMPT_KEY) != prompt_key:
        template_text = resolve_prompt_text(prompt_key)
        if template_text:
            st.session_state[_REQUEST_KEY] = template_text
            st.session_state[_APPLIED_PROMPT_KEY] = prompt_key

    # Text box: either free text or populated from dropdown selection.
    request_text = st.text_area(
        "User Request",
        height=140,
        key=_REQUEST_KEY,
        placeholder=(
            "e.g. Build AAPL from January 2023 to December 2023\n"
            "or select a template above to populate this box."
        ),
    )

    # Clear the applied marker when the template selection is reset.
    if not prompt_key and st.session_state.get(_APPLIED_PROMPT_KEY):
        st.session_state[_APPLIED_PROMPT_KEY] = None

    st.divider()

    # Agent selection.
    agent_names = get_agent_names()
    col_agent, col_run, col_info = st.columns([2, 1, 1])
    with col_agent:
        selected_agent = st.selectbox("Agent", agent_names, key=_SELECTED_AGENT_KEY)
    with col_run:
        st.markdown("&nbsp;")
        run_clicked = st.button(
            "Run",
            type="primary",
            use_container_width=True,
            disabled=bool(
                (current_session := _get_session())
                and get_session_status(current_session).get("running")
            ),
        )
    with col_info:
        st.markdown("&nbsp;")
        current = _get_session()
        session_label = (
            get_session_status(current).get("session_id") if current else "—"
        )
        st.caption(f"Session: {session_label}")

    # Expandable agent definition.
    agent_def = get_agent_definition(selected_agent)
    if agent_def:
        with st.expander(f"Agent Definition — {selected_agent}", expanded=False):
            st.markdown(f"**Description:** {agent_def['description']}")
            if agent_def.get("goal"):
                st.markdown(f"**Goal:** {agent_def['goal']}")
            if agent_def.get("tools"):
                st.markdown("**Tools:**")
                st.code(", ".join(agent_def["tools"]))
            if agent_def.get("guardrails"):
                st.markdown("**Guardrails:**")
                for guardrail in agent_def["guardrails"]:
                    st.markdown(f"- {guardrail}")
            st.markdown("**System Prompt:**")
            st.code(agent_def["system_prompt"])

    # Progress bar for execution.
    st.subheader("Execution Progress")
    progress_bar = st.progress(0.0, text="Idle")

    session = _get_session()
    status_snapshot: dict[str, Any] | None = None
    if session is not None:
        status_snapshot = get_session_status(session)

    if status_snapshot is not None and status_snapshot.get("running"):
        progress_bar.progress(0.5, text=f"Running… ({status_snapshot.get('status', '')})")
    elif status_snapshot is not None and status_snapshot.get("status") == "completed":
        progress_bar.progress(1.0, text="Completed")
    elif status_snapshot is not None and status_snapshot.get("status") == "paused":
        progress_bar.progress(0.75, text="Paused — awaiting user input")
    elif status_snapshot is not None and status_snapshot.get("status") == "error":
        progress_bar.progress(0.0, text="Error")
    elif status_snapshot is not None and status_snapshot.get("status") == "cancelled":
        progress_bar.progress(0.0, text="Cancelled")

    # Handle Run button click.
    if run_clicked:
        if not request_text.strip():
            st.error("Please enter a user request or select a prompt template.")
        else:
            try:
                if session is None or not session_bound_to(session, provider, selected_model):
                    if session is not None:
                        stop_session(session)
                    session = create_session(
                        provider,
                        selected_model,
                        api_key=api_key,
                        api_base=api_base,
                    )
                    _set_session(session)
                st.session_state[_POLLING_KEY] = True
                start_run(session, request_text.strip())
                st.toast(f"Run started for {selected_agent} / {provider_label} / {selected_model}")
            except Exception as exc:
                st.error(str(exc))

# ── Right column: pause / human-in-the-loop response ──────────────────────
with main_col_right:
    st.header("Human-in-the-Loop")

    col_htl_header, col_htl_reset = st.columns([3, 1])
    with col_htl_header:
        st.caption("Respond to workflow checkpoints as they arise.")
    with col_htl_reset:
        if st.button(
            "🔄 Reset",
            use_container_width=True,
            help="Reset the HITL session and clear all state.",
        ):
            _reset_session_state()
            st.rerun()

    if session is None:
        st.info("No active session yet. Enter a request in the left panel and press Run.")
    else:
        status = get_session_status(session)

        if status.get("status") == "running":
            st.info(f"⏳ Workflow is running — session `{status.get('session_id')}`")
            st.caption("LLM calls can take 10–30 seconds per step. The log tab updates automatically.")
            st.button("Refresh", use_container_width=True)

        elif status.get("status") == "paused":
            st.warning("Workflow is paused and awaits your input.")

            pause_prompt = status.get("pause_prompt") or "The workflow needs your input to continue."
            current_agent = status.get("current_agent") or "Unknown"
            st.markdown(f"**Paused at:** `{current_agent}`")
            st.markdown(pause_prompt)

            pause_category = get_pause_category(current_agent)
            quick_options = get_prompts_for_category(pause_category)

            # Pause-scoped widget keys: each checkpoint gets a unique key so
            # Streamlit creates fresh inputs per pause instead of carrying over
            # stale values from the previous checkpoint.
            pause_scope = (
                f"{status.get('run_count', 0)}:{current_agent}:{pause_prompt[:48]}"
            )

            quick_selection = st.selectbox(
                "Quick Options",
                [""] + [opt["label"] for opt in quick_options],
                key=f"{_PAUSE_QUICK_KEY}:{pause_scope}",
                format_func=lambda label: (
                    "— Select a quick option —" if not label else label
                ),
            )

            resume_text = st.text_area(
                "Your Response",
                height=100,
                placeholder="Type your response…",
                key=f"{_PAUSE_RESPONSE_KEY}:{pause_scope}",
            )

            col_submit, col_cancel = st.columns(2)
            with col_submit:
                submit_clicked = st.button("Submit", type="primary", use_container_width=True)
            with col_cancel:
                cancel_clicked = st.button("Cancel", use_container_width=True)

            if submit_clicked:
                response = resume_text.strip()
                if quick_selection:
                    for opt in quick_options:
                        if opt["label"] == quick_selection:
                            response = opt["response"]
                            break
                if not response:
                    st.error("Please enter a response or select a quick option.")
                else:
                    try:
                        st.session_state[_POLLING_KEY] = True
                        submit_response(session, response)
                        st.toast("Response submitted")
                    except Exception as exc:
                        st.error(str(exc))

            if cancel_clicked:
                try:
                    st.session_state[_POLLING_KEY] = True
                    submit_response(session, "exit")
                    st.toast("Workflow cancelled.")
                except Exception as exc:
                    st.error(str(exc))

        elif status.get("status") == "completed":
            st.success("Workflow completed.")
            session_id = status.get("session_id", "")
            events_count = status.get("events_count", 0)
            st.caption(f"Session `{session_id}` processed {events_count} events.")

        elif status.get("status") == "error":
            st.error(f"Workflow failed: {status.get('error', 'Unknown error')}")

        elif status.get("status") == "cancelled":
            st.warning("Workflow was cancelled.")

        else:
            st.caption("Session is idle.")


# ---------------------------------------------------------------------------
# Agent Outputs placeholder area
# ---------------------------------------------------------------------------

st.divider()
_render_agent_outputs(session)


# ---------------------------------------------------------------------------
# Bottom: tabs — Log, Events, Trace
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Run Output")

tab_log, tab_events, tab_trace = st.tabs(
    ["Log Progression", "Structured Events", "ReACT Trace"]
)

session = _get_session()

with tab_log:
    col_refresh, col_clear = st.columns([1, 4])
    with col_refresh:
        refresh_log = st.button("🔄 Refresh")
    with col_clear:
        st.caption("Captures workflow events, LLM/tool activity and runtime output.")

    log_text = get_log_text(session) if session is not None else ""
    st.code(log_text if log_text else "No log output yet.", language="text", height=420)

    if refresh_log:
        st.rerun()

with tab_events:
    st.caption("Structured callback events emitted by the workflow.")
    if session is not None and get_events(session):
        st.json(get_events_json(session))
    else:
        st.info("No events captured yet — start a run to populate this tab.")

with tab_trace:
    st.caption("ReACT reasoning trace from the agentic runtime.")
    trace_text = get_trace_text(session) if session is not None else ""
    st.code(trace_text if trace_text else "No trace yet.", language="text", height=420)


# Auto-refresh while running. When a run finishes after the main panel has
# already rendered, re-render once so the final state (paused / completed /
# error / cancelled) is always displayed instead of a stale "running" panel.
if session is not None:
    status = get_session_status(session)
    if status.get("running"):
        st.caption(
            f"⏳ run in progress — session `{status.get('session_id')}` — "
            f"started {status.get('started_at') or '…'}"
        )
        time.sleep(1)
        st.rerun()
    elif st.session_state.get(_POLLING_KEY):
        st.session_state[_POLLING_KEY] = False
        st.rerun()