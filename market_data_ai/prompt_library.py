"""Prompt library for human-in-the-loop checkpoints.

Provides pre-defined prompt templates that users can select from at each
human-in-the-loop checkpoint, reducing cognitive load and standardising inputs.
Users can either pick a template or type a free-form response.
"""
from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent


@dataclass
class PromptTemplate:
    """A pre-defined prompt template for a human-in-the-loop checkpoint."""

    label: str
    description: str
    response: str
    category: str = "general"
    asset_placeholder: str | None = None

    def render(self, asset: str | None = None, start_date: str | None = None, end_date: str | None = None) -> str:
        """Render the template, substituting its ``{asset}`` placeholder.

        Args:
            asset: Value to substitute for the ``{asset}`` placeholder.  When
                ``None`` or blank, the raw template is returned unchanged.

        Returns:
            The rendered prompt text.
        """
        # Backwards-compatible: support optional date placeholders
        # The template may include `{asset}`, `{start_date}` and `{end_date}`.
        # If the template declares an asset placeholder but no asset is
        # provided, return the raw response unchanged to avoid accidental
        # overwrites of a user's custom text.
        sd = start_date or ""
        ed = end_date or ""
        try:
            if self.asset_placeholder:
                if asset and asset.strip():
                    return self.response.format(asset=asset.strip(), start_date=sd, end_date=ed)
                # asset expected but not provided: return unmodified
                return self.response
            # No asset placeholder: still allow date substitution if present
            return self.response.format(start_date=sd, end_date=ed)
        except Exception:
            # If formatting fails for any reason, return the raw template.
            return self.response


# ── Source Selection Prompts ──────────────────────────────────────────────
#
# Declared as a ``dict[str, str]`` (label → descriptive text, via ``dedent``)
# for consistency with ``CLARIFICATION_PROMPTS``.  These remain single-token
# quick options: the *response* is the canonical source label the workflow
# expects, while the dict value is the human-readable description.

SOURCE_SELECTION_PROMPTS: dict[str, str] = {
    "yahoo": dedent(
        """
        Select Yahoo as the data source (has some missing values, good for
        testing gap-filling).
        """
    ).strip(),
    "bloomberg": dedent(
        """
        Select Bloomberg as the data source (typically complete data).
        """
    ).strip(),
    "reuters": dedent(
        """
        Select Reuters as the data source (alternative market data).
        """
    ).strip(),
    "all_sources": dedent(
        """
        Use all available sources and let the system decide the best
        combination.
        """
    ).strip(),
}


def _build_source_selection_templates() -> list[PromptTemplate]:
    """Build ``PromptTemplate`` objects from the source-selection dict."""
    return [
        PromptTemplate(
            label=label,
            description=description,
            response=label,  # canonical source token expected by the workflow
            category="source_selection",
        )
        for label, description in SOURCE_SELECTION_PROMPTS.items()
    ]


# Built once at import time and shared by the registry.
SOURCE_SELECTION_TEMPLATES: list[PromptTemplate] = _build_source_selection_templates()


# ── Gap-Filling Method Prompts ────────────────────────────────────────────
#
# Declared as a ``dict[str, str]`` (label → descriptive text, via ``dedent``)
# for consistency with ``CLARIFICATION_PROMPTS``.  Like the source-selection
# options, these are single-token quick options whose *response* is the
# canonical method name the workflow expects.

GAP_FILLING_PROMPTS: dict[str, str] = {
    "linear_interpolation": dedent(
        """
        Fill gaps by linear interpolation between known values (smooth, good
        for gradual trends).
        """
    ).strip(),
    "forward_fill": dedent(
        """
        Fill gaps by carrying forward the last known value (good for sticky
        prices).
        """
    ).strip(),
    "backward_fill": dedent(
        """
        Fill gaps by using the next known value (good for mean-reverting
        series).
        """
    ).strip(),
    "none": dedent(
        """
        Do not fill gaps — keep missing values as-is.
        """
    ).strip(),
}


def _build_gap_filling_templates() -> list[PromptTemplate]:
    """Build ``PromptTemplate`` objects from the gap-filling dict."""
    return [
        PromptTemplate(
            label=label,
            description=description,
            response=label,  # canonical gap-filling method token
            category="gap_filling",
        )
        for label, description in GAP_FILLING_PROMPTS.items()
    ]


# Built once at import time and shared by the registry.
GAP_FILLING_TEMPLATES: list[PromptTemplate] = _build_gap_filling_templates()


# ── Orchestrator Clarification Prompts ────────────────────────────────────
#
# Aligned with ``src/fin_ai/agents/prompts_library.py::RESEARCH_ANALYSIS``: a
# plain dict mapping a human-readable label to a ``dedent``-ed prompt template
# that carries an ``{asset}`` placeholder.  The dashboard lets the user supply
# an asset and renders the template via :meth:`PromptTemplate.render`.

CLARIFICATION_PROMPTS: dict[str, str] = {
    "Build Time Series": dedent(
        """
        Note: this workflow includes a human-in-the-loop. The agent should
        pause at checkpoints to request your confirmation or input before
        taking irreversible actions or proceeding with significant steps.

        Build a continuous {asset} time series from {start_date} to {end_date} using the available tools.

        Workflow:
        - Resolve {asset} in the reference-data catalog (symbol / ticker / name).
        - Load historical prices from every available data source covering {start_date} to {end_date}.
        - Compute data-quality metrics per source and report them.
        - If the series has gaps, recommend and apply a gap-filling method.
        - Build the final continuous time series and visualize it.

        Return the resulting series and any artifacts you produced.
        """
    ).strip(),
    "Build Time Series with Full Report": dedent(
        """
        Note: this workflow includes a human-in-the-loop. The agent should
        pause at checkpoints to request your confirmation or input before
        taking irreversible actions or proceeding with significant steps.

        Build a continuous {asset} time series from {start_date} to {end_date} and produce a full data-quality report.

        Workflow:
        - Resolve {asset} in the reference-data catalog.
        - Load historical prices from every available data source covering {start_date} to {end_date}.
        - Compute and compare data-quality metrics across all sources.
        - Recommend and apply a gap-filling method where needed.
        - Build the final series, visualize it, and generate a workflow report.

        Return the final series plus any CSV / chart / report artifacts.
        """
    ).strip(),
    "Compare Sources and Build Time Series": dedent(
        """
        Note: this workflow includes a human-in-the-loop. The agent should
        pause at checkpoints to request your confirmation or input before
        taking irreversible actions or proceeding with significant steps.

        Compare all available data sources for {asset} from {start_date} to {end_date} and build the best series.

        Workflow:
        - Load {asset} prices from every available source covering {start_date} to {end_date}.
        - Compute data-quality metrics and rank the sources.
        - Apply gap-filling to the chosen series where needed.
        - Build and visualize the final continuous time series.

        Justify the source selection and return the final series + artifacts.
        """
    ).strip(),
    "Research and Build Time Series": dedent(
        """
        Note: this workflow includes a human-in-the-loop. The agent should
        pause at checkpoints to request your confirmation or input before
        taking irreversible actions or proceeding with significant steps.

        Research {asset} and build a continuous time series for analysis covering {start_date} to {end_date}.

        Workflow:
        - Resolve {asset} in the reference-data catalog.
        - Load historical prices and assess data quality per source for {start_date} to {end_date}.
        - Apply gap-filling where needed.
        - Build the final series and visualize it.

        Return the final series and a short note on data quality.
        """
    ).strip(),
}

# Short descriptions shown in the dashboard's template dropdown.
_CLARIFICATION_DESCRIPTIONS: dict[str, str] = {
    "Build Time Series": "Build a continuous series for {asset}",
    "Build Time Series with Full Report": (
        "Build {asset} series plus a full data-quality report"
    ),
    "Compare Sources and Build Time Series": (
        "Compare sources for {asset} and build the best series"
    ),
    "Research and Build Time Series": (
        "Research {asset} and build a continuous time series"
    ),
}


def _build_clarification_templates() -> list[PromptTemplate]:
    """Build ``PromptTemplate`` objects from the clarification dict.

    Each entry becomes a template carrying an ``{asset}`` placeholder so the
    dashboard can fill in the asset via :meth:`PromptTemplate.render`.
    """
    return [
        PromptTemplate(
            label=label,
            description=_CLARIFICATION_DESCRIPTIONS.get(label, label),
            response=template,
            category="clarification",
            asset_placeholder="{asset}",
        )
        for label, template in CLARIFICATION_PROMPTS.items()
    ]


# Built once at import time and shared by the registry.
CLARIFICATION_TEMPLATES: list[PromptTemplate] = _build_clarification_templates()

# ── Category Index ────────────────────────────────────────────────────────

PROMPT_REGISTRY: dict[str, list[PromptTemplate]] = {
    "source_selection": SOURCE_SELECTION_TEMPLATES,
    "gap_filling": GAP_FILLING_TEMPLATES,
    "clarification": CLARIFICATION_TEMPLATES,
}


def get_prompts(category: str) -> list[PromptTemplate]:
    """Get all prompt templates for a given category.

    Args:
        category: The prompt category name (e.g. 'source_selection', 'gap_filling').

    Returns:
        List of PromptTemplate objects for that category.
    """
    return PROMPT_REGISTRY.get(category, [])


def format_prompt_menu(category: str, context: str | None = None) -> str:
    """Format a human-readable menu of prompt options for a given category.

    Args:
        category: The prompt category name.
        context: Optional context string to prepend (e.g. the agent's question).

    Returns:
        A formatted string showing available options.
    """
    prompts = get_prompts(category)
    if not prompts:
        return ""

    lines: list[str] = []
    if context:
        lines.append(context)
        lines.append("")

    lines.append("Available quick options:")
    for i, prompt in enumerate(prompts, 1):
        lines.append(f"  [{i}] {prompt.label}: {prompt.description}")
    lines.append("")
    lines.append("Or type your own response (or 'exit' to quit):")
    return "\n".join(lines)


def resolve_prompt_selection(category: str, user_input: str) -> str | None:
    """Resolve a user's selection to the actual prompt response.

    Supports:
    - Numeric index: '1' -> first prompt's response
    - Label match: 'yahoo' -> matching prompt's response
    - Free-form: anything else returned as-is

    Args:
        category: The prompt category to look up.
        user_input: The user's input (index, label, or free-form text).

    Returns:
        The resolved response string, or None if the input is invalid.
    """
    prompts = get_prompts(category)
    if not prompts:
        return user_input

    stripped = user_input.strip()

    # Try numeric index
    try:
        index = int(stripped) - 1
        if 0 <= index < len(prompts):
            return prompts[index].response
    except ValueError:
        pass

    # Try label match (case-insensitive)
    for prompt in prompts:
        if prompt.label.casefold() == stripped.casefold():
            return prompt.response

    # Return as-is (free-form)
    return stripped