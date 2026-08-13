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

import datetime
import json
import sys
import time
import uuid
import re
from pathlib import Path
from typing import Any



import streamlit as st
import traceback

from market_data_ai.dashboard_app import (
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
    get_populated_timeseries,
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
    parse_asset_and_dates,
    stop_session,
    submit_response,
)

from market_data_ai.database import get_datastore
from market_data_ai.dashboard_app import (
    register_file_in_datastore,
    list_registered_files,
    delete_registered_file,
)

# ``streamlit run`` executes this file as a top-level script (``__main__``) with
# no package context, so relative imports are unsupported here.  Add the
# repository root to the import path so the sibling ``market_data_ai`` modules
# load via absolute imports regardless of how the dashboard is launched
# (``python -m market_data_ai.dashboard_app`` or
# ``streamlit run market_data_ai/dashboard.py``).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
    
# Session state keys used for Streamlit widget persistence
_SESSION_KEY = "workflow_session"
_REQUEST_KEY = "workflow_request_text"
_ASSET_KEY = "workflow_asset_text"
_FILES_KEY = "workflow_selected_files"
_SELECTED_AGENT_KEY = "workflow_selected_agent"
_SELECTED_PROMPT_KEY = "workflow_selected_prompt"
_APPLIED_PROMPT_KEY = "workflow_applied_prompt"
_POLLING_KEY = "workflow_polling"
_PAUSE_QUICK_KEY = "workflow_pause_quick"
_PAUSE_RESPONSE_KEY = "workflow_pause_response"
_FILE_INFO_KEY = "workflow_file_info"

# Prompt registry derived helpers (include empty key for free-text)
try:
    _PROMPT_OPTIONS = get_prompt_options()
except Exception:
    _PROMPT_OPTIONS = []

_PROMPT_KEYS = [""] + [opt.get("key") for opt in _PROMPT_OPTIONS]
_PROMPT_LOOKUP: dict[str, dict[str, Any]] = {opt.get("key"): opt for opt in _PROMPT_OPTIONS}

def _data_folder() -> Path:
    """Return the canonical data folder used for uploaded files."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


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
            _ASSET_KEY,
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


def _maybe_rerun() -> None:
    """Safely trigger a Streamlit rerun when available.

    Some Streamlit versions expose `experimental_rerun`, others expose
    `rerun`, and very old/new builds may lack either. Prefer the
    experimental API, fall back to `rerun` if present, otherwise no-op.
    """
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        elif hasattr(st, "rerun"):
            st.rerun()
    except Exception:
        # Best-effort: if rerun can't be called, let Streamlit continue —
        # widget interactions will trigger a normal rerun on the next event.
        pass


def _data_folder() -> Path:
    """Return the canonical data folder used for uploaded files."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _register_file_in_datastore(filename: str, description: str | None, path: str) -> str:
    """Delegate to backend `register_file_in_datastore` implementation."""
    return register_file_in_datastore(filename, description, path)


def _list_registered_files() -> list[dict[str, Any]]:
    """Delegate to backend `list_registered_files` implementation."""
    return list_registered_files()


def _delete_registered_file(file_id: str) -> bool:
    """Delegate to backend `delete_registered_file` implementation."""
    return delete_registered_file(file_id)


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
        st.dataframe(table_data, width="stretch", hide_index=True)
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
                        width="stretch",
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
            st.image(str(chart_file), caption="Time Series Chart", width="stretch")


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




def _render_series_card(
    *,
    symbol: str | None,
    source: str | None,
    dates: list[str] | None,
    prices: list[float | str] | None,
    method: str | None = None,
    label: str,
) -> None:
    """Render a single before/after time series from the DataStore."""
    import pandas as pd  # shipped with Streamlit

    if not dates or not prices or len(dates) != len(prices):
        st.markdown(f"**{label}**")
        st.caption(f"Symbol: `{symbol or '—'}` · Source: `{source or '—'}` — no series data.")
        return

    df = pd.DataFrame({"Date": dates, "Price": prices})
    with st.container(border=True):
        st.markdown(f"**{label}** — `{symbol or '—'}` / `{source or '—'}`")
        if method:
            st.caption(f"Gap-filling method: `{method}` · Observations: {len(df)}")
        else:
            st.caption(f"Observations: {len(df)}")

        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            st.line_chart(df.set_index("Date"), width="stretch")
        with col_table:
            st.dataframe(df, width="stretch", hide_index=True, height=220)


def _render_db_timeseries_tab(session: WorkflowSession | None) -> None:
    """Render the before/after time series persisted in the DataStore.

    Pulls the ``raw_timeseries`` (before gap-filling) and
    ``filled_timeseries`` (after gap-filling) tables for the active run and
    presents them side-by-side in a comparison.
    """
    st.caption(
        "Compare the original series (before gap-filling) with the gap-filled "
        "series (after) persisted to the database for the active run."
    )

    if session is None:
        st.info("No active session yet — start a run to populate the database.")
        return

    try:
        data = get_populated_timeseries(run_id=session.session_id)
    except Exception as exc:
        st.info(f"No populated time series in the database for this run yet. ({exc})")
        return

    before = data.get("before") or []
    after = data.get("after") or []

    if not before and not after:
        st.info("The database has no before/after series for this run yet.")
        return

    tab_before, tab_after = st.tabs(["Before (Original)", "After (Gap-Filled)"])

    with tab_before:
        if not before:
            st.info("No original (raw) series recorded for this run.")
        for entry in before:
            _render_series_card(
                symbol=entry.get("symbol"),
                source=entry.get("source"),
                dates=entry.get("dates"),
                prices=entry.get("prices"),
                label="Original Series",
            )

    with tab_after:
        if not after:
            st.info("No gap-filled series recorded for this run.")
        for entry in after:
            _render_series_card(
                symbol=entry.get("symbol"),
                source=entry.get("source"),
                dates=entry.get("filled_dates") or entry.get("dates"),
                prices=entry.get("filled_prices") or entry.get("prices"),
                method=entry.get("method"),
                label="Gap-Filled Series",
            )


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

    if st.button("Reset Session", width="stretch"):
        _reset_session_state()
        st.rerun()

    st.divider()
    with st.expander("File Maintenance", expanded=False):
        st.markdown("Manage uploaded data files used for time-series construction.")
        upload_file = st.file_uploader(
            "Upload Data File",
            type=["csv", "parquet", "json", "txt"],
            accept_multiple_files=False,
        )
        upload_desc = st.text_input("File Description", help="Optional description for the file.")
        if st.button("Upload and Register File", width="stretch") and upload_file is not None:
            try:
                data_dir = _data_folder()
                save_path = data_dir / upload_file.name
                with open(save_path, "wb") as fh:
                    fh.write(upload_file.getbuffer())
                _register_file_in_datastore(upload_file.name, upload_desc or None, str(save_path))
                st.success(f"Uploaded and registered {upload_file.name}")
            except Exception as exc:
                st.error(f"File upload failed: {exc}")

        st.markdown("**Registered Files**")
        files = _list_registered_files()
        if not files:
            st.info("No uploaded files yet.")
        else:
            # Prefer an editable tabular view when Streamlit's `data_editor` is available
            try:
                import pandas as pd

                df = pd.DataFrame(
                    [
                        {
                            "Filename": f.get("filename"),
                            "Description": f.get("description") or "",
                            "Path": f.get("path"),
                            "Selected": False,
                        }
                        for f in files
                    ]
                )

                if hasattr(st, "data_editor"):
                    edited = st.data_editor(df, num_rows="fixed", width="stretch", height=300)
                    # Map back selected rows to file ids
                    selected_ids = []
                    for idx, row in edited.iterrows():
                        if bool(row.get("Selected")):
                            # match by filename + path (should be unique enough)
                            for f in files:
                                if f.get("filename") == row.get("Filename") and f.get("path") == row.get("Path"):
                                    selected_ids.append(f.get("file_id"))
                                    break

                    # Info selector for viewing details of a single file
                    info_choice = st.selectbox(
                        "View file info",
                        [""] + [f"{f.get('filename')} — {f.get('description') or 'no description'}" for f in files],
                        key="file_info_selector",
                    )
                    if info_choice:
                        # find selected file id
                        sel_label = info_choice
                        sel_file = None
                        for f in files:
                            label = f"{f.get('filename')} — {f.get('description') or 'no description'}"
                            if label == sel_label:
                                sel_file = f
                                break
                        if sel_file:
                            st.session_state[_FILE_INFO_KEY] = sel_file.get("file_id")
                            _maybe_rerun()

                    st.caption(f"Total files: {len(files)} — {len(selected_ids)} selected.")
                    if st.button("🗑️ Delete Selected", width="stretch", disabled=not selected_ids):
                        for fid in selected_ids:
                            try:
                                _delete_registered_file(fid)
                            except Exception as exc:
                                st.error(str(exc))
                        if selected_ids:
                            st.toast(f"Deleted {len(selected_ids)} file(s)")
                            _maybe_rerun()
                else:
                    raise AttributeError("data_editor not available")
            except Exception:
                # Fallback to the simple per-row layout
                selected_ids: list[str] = []
                for f in files:
                    fid = f.get("file_id")
                    col_name, col_select, col_info = st.columns([6, 1, 1])
                    with col_name:
                        st.markdown(f"**{f.get('filename')}**")
                        st.caption(f.get('path') or "—")
                    with col_select:
                        is_selected = st.checkbox("", key=f"file_select:{fid}")
                    with col_info:
                        if st.button("ℹ️", key=f"file_info_btn:{fid}"):
                            st.session_state[_FILE_INFO_KEY] = fid
                            _maybe_rerun()
                    if is_selected:
                        selected_ids.append(fid)

                st.caption(f"Total files: {len(files)} — {len(selected_ids)} selected.")
                if st.button("🗑️ Delete Selected", width="stretch", disabled=not selected_ids):
                    for fid in selected_ids:
                        try:
                            _delete_registered_file(fid)
                        except Exception as exc:
                            st.error(str(exc))
                    if selected_ids:
                        st.toast(f"Deleted {len(selected_ids)} file(s)")
                        _maybe_rerun()

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

# File info modal (renders when a file info button was clicked)
if st.session_state.get(_FILE_INFO_KEY):
    fid = st.session_state.get(_FILE_INFO_KEY)
    try:
        # Prefer DataStore metadata when available
        try:
            ds = get_datastore()
            meta = ds.get_file(fid)
        except Exception:
            # Fallback to registry
            found = None
            for f in _list_registered_files():
                if f.get("file_id") == fid:
                    found = f
                    break
            meta = found or {}

        if hasattr(st, "modal"):
            with st.modal("File Information"):
                st.header(meta.get("filename") or "File")
                st.write("**Description:**", meta.get("description") or "—")
                st.write("**Location:**", meta.get("path") or "—")
                st.write("**Registered at:**", meta.get("created_at") or "—")
                # Optional start/end dates if available in metadata
                st.write("**Start Date:**", meta.get("start_date") or "N/A")
                st.write("**End Date:**", meta.get("end_date") or "N/A")
                if st.button("Close"):
                    st.session_state[_FILE_INFO_KEY] = None
                    st.experimental_rerun()
        else:
            # Fallback for Streamlit versions without `st.modal`
            with st.expander("File Information", expanded=True):
                st.header(meta.get("filename") or "File")
                st.write("**Description:**", meta.get("description") or "—")
                st.write("**Location:**", meta.get("path") or "—")
                st.write("**Registered at:**", meta.get("created_at") or "—")
                st.write("**Start Date:**", meta.get("start_date") or "N/A")
                st.write("**End Date:**", meta.get("end_date") or "N/A")
                if st.button("Close Info"):
                    st.session_state[_FILE_INFO_KEY] = None
                    st.experimental_rerun()
    except Exception as exc:
        st.error(f"Could not load file info: {exc}")

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
            "Custom (free text)"
            if not key
            else f"{_PROMPT_LOOKUP[key]['category']} / {_PROMPT_LOOKUP[key]['label']}"
            + f" — {_PROMPT_LOOKUP[key]['description']}"
        ),
        help=prompt_caption,
    )

    # Optional asset / ticker used to fill each template's {asset} placeholder.
    # Always rendered so the widget stays stable across reruns; it is only used
    # when the selected template actually contains an {asset} placeholder.
    asset_text = st.text_input(
        "Asset / Ticker (optional)",
        key=_ASSET_KEY,
        placeholder="e.g. AAPL, MSFT, GOOGL, NVDA…",
        help="When the selected prompt template contains an {asset} placeholder, "
        "this value is filled into it.",
    )

    # Optional date range selectors. Hidden unless the user opts in so that
    # the dashboard behaves the same when no dates are supplied.
    specify_dates = st.checkbox("Specify start/end dates (optional)", value=False)
    start_date_str: str | None = None
    end_date_str: str | None = None
    if specify_dates:
        start_date = st.date_input("Start Date", key="start_date_picker")
        end_date = st.date_input("End Date", key="end_date_picker")
        try:
            start_date_str = start_date.isoformat()
            end_date_str = end_date.isoformat()
        except Exception:
            start_date_str = str(start_date)
            end_date_str = str(end_date)

    # Populate the request box BEFORE the text area is instantiated. Streamlit
    # forbids modifying a widget's session-state value after the widget has been
    # created, so the template text must be set here (before st.text_area below).
    # The applied-prompt marker is keyed to the selected prompt *and* asset so it
    # does not overwrite the user's own edits yet still re-renders when the asset
    # (or the selected template) changes.
    # Only auto-fill the request box from a selected template when the
    # request box is currently empty. This preserves a user's custom text
    # and makes the "Custom (free text)" option the safe default.
    current_request = st.session_state.get(_REQUEST_KEY, "") or ""
    if (
        prompt_key
        and not current_request.strip()
        and st.session_state.get(_APPLIED_PROMPT_KEY) != (
            prompt_key,
            asset_text,
            start_date_str,
            end_date_str,
        )
    ):
        template_text = resolve_prompt_text(
            prompt_key, asset=asset_text, start_date=start_date_str, end_date=end_date_str
        )
        if template_text:
            st.session_state[_REQUEST_KEY] = template_text
            st.session_state[_APPLIED_PROMPT_KEY] = (prompt_key, asset_text, start_date_str, end_date_str)

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

    # Data source selection from registered files
    st.markdown("---")
    st.markdown("**Data Sources**")
    registered_files = _list_registered_files()
    file_options_labels: list[str] = []
    file_label_to_id: dict[str, str] = {}
    for f in registered_files:
        label = f"{f.get('filename')} — {f.get('description') or 'no description'}"
        file_options_labels.append(label)
        file_label_to_id[label] = f.get("file_id")

    selected_labels = st.multiselect(
        "Select data files to include (optional)",
        file_options_labels,
        key=_FILES_KEY,
    )
    selected_file_ids = [file_label_to_id[l] for l in selected_labels]
    st.caption(f"Selected files: {len(selected_file_ids)}")

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
            width="stretch",
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

    # Execution progress UI removed to reduce visual noise; status is
    # available in the Human-in-the-Loop panel and Run Output tabs.

    session = _get_session()
    status_snapshot: dict[str, Any] | None = None
    if session is not None:
        status_snapshot = get_session_status(session)

    # Note: progress bar updates removed.

    # Handle Run button click.
    if run_clicked:
        # Debug: record the attempted run inputs immediately so we can
        # observe cases where the UI shows text but the run doesn't start.
        try:
            session_for_debug = _get_session()
            dbg = {
                "request_text": (request_text or "")[:1000],
                "prompt_key": prompt_key,
                "asset_text": asset_text,
                "specify_dates": specify_dates,
                "start_date": start_date_str,
                "end_date": end_date_str,
                "selected_file_ids": selected_file_ids,
            }
            if session_for_debug is not None:
                with session_for_debug._lock:
                    session_for_debug._log_lines.append(f"[RUN_CLICKED] {json.dumps(dbg)}")
            else:
                st.info(f"[RUN_CLICKED] {json.dumps(dbg)}")
        except Exception:
            # Non-fatal; continue to normal validation.
            pass

        # If the user didn't fill the Asset or date pickers but provided a
        # natural-language request, ask the backend to parse ticker and
        # dates. The backend uses a robust parser (dateutil) and returns
        # ISO-formatted dates when possible.
        try:
            parsed_asset, parsed_start, parsed_end = parse_asset_and_dates(request_text or "")
            parsed_dbg = {"parsed_asset": parsed_asset, "parsed_start": parsed_start, "parsed_end": parsed_end}
            try:
                if session is not None:
                    with session._lock:
                        session._log_lines.append(f"[PARSED_INPUT] {json.dumps(parsed_dbg)}")
            except Exception:
                pass
        except Exception:
            parsed_asset = parsed_start = parsed_end = None

        # Determine whether parsed values would be used (only when the
        # corresponding explicit widgets are empty).
        use_parsed_asset = (not asset_text) and bool(parsed_asset)
        use_parsed_dates = (not (start_date_str and end_date_str)) and (parsed_start or parsed_end)

        asset_for_run = asset_text or (parsed_asset if use_parsed_asset else None)
        start_date_for_run = start_date_str or (parsed_start if use_parsed_dates else None)
        end_date_for_run = end_date_str or (parsed_end if use_parsed_dates else None)

        # If parsed values are available and the widgets are empty, auto-fill
        # the corresponding widget session state and request a rerun so the
        # UI shows the populated Asset / Date fields. A pending auto-run flag
        # is set so the run starts automatically on the next render.
        if use_parsed_asset or use_parsed_dates:
            try:
                # Fill asset field if empty
                if use_parsed_asset:
                    st.session_state[_ASSET_KEY] = parsed_asset
                # Fill date pickers (ensure the specify_dates checkbox is set)
                if use_parsed_dates:
                    # The checkbox widget has no explicit `key`, so Streamlit
                    # uses the checkbox label as the session-state key. Set
                    # that key so the checkbox renders checked on rerun.
                    st.session_state["Specify start/end dates (optional)"] = True
                    from datetime import date as _dt_date
                    try:
                        if parsed_start:
                            st.session_state["start_date_picker"] = datetime.fromisoformat(parsed_start).date()
                        if parsed_end:
                            st.session_state["end_date_picker"] = datetime.fromisoformat(parsed_end).date()
                    except Exception:
                        # If iso parse fails, ignore — the widget stays unset.
                        pass
                # Mark pending auto-run and rerun so the widgets render with
                # the new session state before we start the workflow.
                st.session_state["_auto_run_pending"] = True
                try:
                    _maybe_rerun()
                except Exception:
                    # Best-effort: fall back to experimental rerun
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass
            except Exception:
                # If auto-fill fails, continue to validation below.
                pass
        # Validate inputs. If the user selected a prompt template (non-empty
        # `prompt_key`), require any placeholders the template needs: an
        # `asset` when indicated, and start/end dates when the template
        # contains `{start_date}` / `{end_date}` placeholders. Free-text
        # requests (the empty prompt key) remain unrestricted.
        if not request_text.strip():
            st.error("Please enter a user request or select a prompt template.")
        else:
            # Determine template requirements when a template is selected.
            if prompt_key:
                lookup = _PROMPT_LOOKUP.get(prompt_key, {})
                needs_asset = lookup.get("needs_asset") == "1"
                response_text = lookup.get("response", "")
                needs_dates = "{start_date}" in response_text or "{end_date}" in response_text

                if needs_asset and not (asset_text and asset_text.strip()):
                    st.error("Selected template requires an Asset/Ticker — please enter it in the Asset / Ticker field.")
                    st.stop()

                if needs_dates and not (specify_dates and start_date_str and end_date_str):
                    st.error(
                        "Selected template requires a start and end date — check 'Specify start/end dates' and pick both dates."
                    )
                    st.stop()

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
                    # Resolve template placeholders into a final request
                    final_request_text = request_text.strip()
                    # Prefer resolving from the template registry so any
                    # template logic is applied consistently.
                    try:
                        template_resolved = resolve_prompt_text(
                            prompt_key, asset=asset_for_run, start_date=start_date_for_run, end_date=end_date_for_run
                        )
                        if template_resolved:
                            final_request_text = template_resolved
                    except Exception:
                        pass

                    # Replace any literal placeholders in free-text with
                    # available values so the Orchestrator receives concrete
                    # inputs rather than '{asset}' placeholders.
                    if asset_for_run:
                        final_request_text = final_request_text.replace("{asset}", asset_for_run)
                    if start_date_for_run:
                        final_request_text = final_request_text.replace("{start_date}", start_date_for_run)
                    if end_date_for_run:
                        final_request_text = final_request_text.replace("{end_date}", end_date_for_run)

                    # If placeholders remain unfilled, abort so the Orchestrator
                    # doesn't loop asking for missing values.
                    if re.search(r"\{\s*asset\s*\}|\{\s*start_date\s*\}|\{\s*end_date\s*\}", final_request_text):
                        st.error("Request contains unresolved placeholders (asset/start_date/end_date). Please fill them in or remove the placeholders.")
                        st.stop()

                    st.session_state[_POLLING_KEY] = True
                    start_run(
                        session,
                        final_request_text,
                        selected_file_ids=selected_file_ids,
                        asset=asset_for_run or None,
                        start_date=start_date_for_run,
                        end_date=end_date_for_run,
                    )
                    try:
                        st.toast(f"Run started for {selected_agent} / {provider_label} / {selected_model}")
                    except Exception:
                        # If toast fails for any reason, don't block the run.
                        pass
                except Exception as exc:
                    # Improve error visibility by capturing a traceback into the
                    # session log and showing a concise error to the user.
                    tb = traceback.format_exc()
                    try:
                        if session is not None:
                            with session._lock:
                                session._log_lines.append(f"[ERROR_RUN_START] {tb}")
                    except Exception:
                        pass
                    st.error(f"Failed to start run: {exc}")

    # If an auto-fill requested a rerun and flagged an auto-run, start the
    # workflow now using the (now-populated) widget values.
    if st.session_state.pop("_auto_run_pending", False):
        try:
            # Recompute session and widget-derived values
            asset_text = st.session_state.get(_ASSET_KEY) or asset_text
            # Read the checkbox value using the label key (no explicit key used)
            specify_dates = st.session_state.get("Specify start/end dates (optional)", specify_dates)
            start_date_val = st.session_state.get("start_date_picker")
            end_date_val = st.session_state.get("end_date_picker")
            try:
                start_date_str = start_date_val.isoformat() if start_date_val else None
                end_date_str = end_date_val.isoformat() if end_date_val else None
            except Exception:
                start_date_str = str(start_date_val) if start_date_val else None
                end_date_str = str(end_date_val) if end_date_val else None

            asset_for_run = asset_text or None
            start_date_for_run = start_date_str or None
            end_date_for_run = end_date_str or None

            # Ensure any template placeholders in the request are resolved
            final_request_text = request_text.strip()
            if prompt_key and ("{asset}" in final_request_text or "{start_date}" in final_request_text or "{end_date}" in final_request_text):
                final_request_text = resolve_prompt_text(prompt_key, asset=asset_for_run, start_date=start_date_for_run, end_date=end_date_for_run)

            if not final_request_text.strip():
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
                    # Replace any leftover placeholders in free-text with
                    # available values so the Orchestrator gets concrete inputs.
                    if asset_for_run:
                        final_request_text = final_request_text.replace("{asset}", asset_for_run)
                    if start_date_for_run:
                        final_request_text = final_request_text.replace("{start_date}", start_date_for_run)
                    if end_date_for_run:
                        final_request_text = final_request_text.replace("{end_date}", end_date_for_run)
                    if re.search(r"\{\s*asset\s*\}|\{\s*start_date\s*\}|\{\s*end_date\s*\}", final_request_text):
                        st.error("Request contains unresolved placeholders (asset/start_date/end_date). Please fill them in or remove the placeholders.")
                        st.stop()

                    start_run(
                        session,
                        final_request_text,
                        selected_file_ids=selected_file_ids,
                        asset=asset_for_run or None,
                        start_date=start_date_for_run,
                        end_date=end_date_for_run,
                    )
                    try:
                        st.toast(f"Run started for {selected_agent} / {provider_label} / {selected_model}")
                    except Exception:
                        pass
                except Exception as exc:
                    tb = traceback.format_exc()
                    try:
                        if session is not None:
                            with session._lock:
                                session._log_lines.append(f"[ERROR_RUN_START] {tb}")
                    except Exception:
                        pass
                    st.error(f"Failed to start run: {exc}")
        except Exception:
            # If auto-run startup fails for any reason, surface error
            st.error("Auto-run failed to start; please try again.")

# ── Right column: pause / human-in-the-loop response ──────────────────────
with main_col_right:
    st.header("Human-in-the-Loop")

    col_htl_header, col_htl_reset = st.columns([3, 1])
    with col_htl_header:
        st.caption("Respond to workflow checkpoints as they arise.")
    with col_htl_reset:
        if st.button(
            "🔄 Reset",
            width="stretch",
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
            st.button("Refresh", width="stretch")

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
                submit_clicked = st.button("Submit", type="primary", width="stretch")
            with col_cancel:
                cancel_clicked = st.button("Cancel", width="stretch")

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
# Bottom: Run Output tabs (Log, Events, Trace, Time Series)
# ---------------------------------------------------------------------------

# Run Output tabs are declared declaratively so that a new tab is a one-line
# registration plus a tiny render function, instead of hand-editing the
# ``st.tabs`` block.  Each entry is: (key, title, render_fn).


def _render_log_tab(session: WorkflowSession | None) -> None:
    """Render the workflow log progression tab."""
    col_refresh, col_clear = st.columns([1, 4])
    with col_refresh:
        refresh_log = st.button("🔄 Refresh")
    with col_clear:
        st.caption("Captures workflow events, LLM/tool activity and runtime output.")

    log_text = get_log_text(session) if session is not None else ""
    # Surface any input metadata stored in-memory and in the DataStore so
    # debugging can see what asset/file/date context was provided for the run.
    if session is not None:
        try:
            # Persisted metadata from DataStore when available. The
            # in-memory `session.input_metadata` display was removed to
            # reduce noise; persisted metadata remains accessible here.
            try:
                ds = get_datastore()
                persisted = ds.get_run_input_metadata(session.session_id)
                if persisted:
                    st.markdown("**Run Input Metadata (persisted):**")
                    st.json(persisted)
            except Exception:
                # DataStore may be unavailable in some environments.
                pass
        except Exception:
            pass
    # Log text is not shown here to reduce noise; use the Events or Trace
    # tabs for detailed runtime output when needed.
    if not log_text:
        st.info("No log output yet.")

    if refresh_log:
        st.rerun()


def _render_events_tab(session: WorkflowSession | None) -> None:
    """Render the structured callback events tab."""
    st.caption("Structured callback events emitted by the workflow.")
    if session is not None and get_events(session):
        st.json(get_events_json(session))
    else:
        st.info("No events captured yet — start a run to populate this tab.")


def _render_trace_tab(session: WorkflowSession | None) -> None:
    """Render the ReACT reasoning trace tab."""
    st.caption("ReACT reasoning trace from the agentic runtime.")
    trace_text = get_trace_text(session) if session is not None else ""
    st.code(trace_text if trace_text else "No trace yet.", language="text", height=420)


def _render_timeseries_tab(session: WorkflowSession | None) -> None:
    """Render the before/after time-series comparison tab."""
    _render_db_timeseries_tab(session)


# Registry of Run Output tabs.  Add a new tab by appending a
# ``(key, title, renderer)`` entry here and defining the renderer function.
RUN_OUTPUT_TABS: list[tuple[str, str, Any]] = [
    ("log", "Log Progression", _render_log_tab),
    ("events", "Structured Events", _render_events_tab),
    ("trace", "ReACT Trace", _render_trace_tab),
    ("timeseries", "Time Series (Before/After)", _render_timeseries_tab),
]


def _render_run_output() -> None:
    """Render the Run Output tab strip from the ``RUN_OUTPUT_TABS`` registry.

    Streamlit requires a stable number of ``st.tabs`` per rerun, so the tabs
    are built and populated by iterating the registry.  Adding a new tab is a
    purely data-driven change (register it in ``RUN_OUTPUT_TABS``).
    """
    st.divider()
    st.subheader("Run Output")

    if not RUN_OUTPUT_TABS:
        st.info("No Run Output tabs registered.")
        return

    session = _get_session()
    tab_handles = st.tabs([title for _, title, _ in RUN_OUTPUT_TABS])

    for tab_handle, (key, title, renderer) in zip(tab_handles, RUN_OUTPUT_TABS):
        with tab_handle:
            renderer(session)


_render_run_output()

# The auto-refresh block below needs the active session at module scope.
session = _get_session()


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