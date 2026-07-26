"""Prompt library for human-in-the-loop checkpoints.

Provides pre-defined prompt templates that users can select from at each
human-in-the-loop checkpoint, reducing cognitive load and standardising inputs.
Users can either pick a template or type a free-form response.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromptTemplate:
    """A pre-defined prompt template for a human-in-the-loop checkpoint."""
    label: str
    description: str
    response: str
    category: str = "general"


# ── Source Selection Prompts ──────────────────────────────────────────────

SOURCE_SELECTION_PROMPTS: list[PromptTemplate] = [
    PromptTemplate(
        label="yahoo",
        description="Select Yahoo as the data source (has some missing values, good for testing gap-filling)",
        response="yahoo",
        category="source_selection",
    ),
    PromptTemplate(
        label="bloomberg",
        description="Select Bloomberg as the data source (typically complete data)",
        response="bloomberg",
        category="source_selection",
    ),
    PromptTemplate(
        label="reuters",
        description="Select Reuters as the data source (alternative market data)",
        response="reuters",
        category="source_selection",
    ),
    PromptTemplate(
        label="all_sources",
        description="Use all available sources and let the system decide the best combination",
        response="all_sources",
        category="source_selection",
    ),
]

# ── Gap-Filling Method Prompts ────────────────────────────────────────────

GAP_FILLING_PROMPTS: list[PromptTemplate] = [
    PromptTemplate(
        label="linear_interpolation",
        description="Fill gaps by linear interpolation between known values (smooth, good for gradual trends)",
        response="linear_interpolation",
        category="gap_filling",
    ),
    PromptTemplate(
        label="forward_fill",
        description="Fill gaps by carrying forward the last known value (good for sticky prices)",
        response="forward_fill",
        category="gap_filling",
    ),
    PromptTemplate(
        label="backward_fill",
        description="Fill gaps by using the next known value (good for mean-reverting series)",
        response="backward_fill",
        category="gap_filling",
    ),
    PromptTemplate(
        label="none",
        description="Do not fill gaps — keep missing values as-is",
        response="none",
        category="gap_filling",
    ),
]

# ── Orchestrator Clarification Prompts ────────────────────────────────────

CLARIFICATION_PROMPTS: list[PromptTemplate] = [
    PromptTemplate(
        label="aapl_2023",
        description="Build AAPL time series from January 2023 to December 2023",
        response="Build AAPL from January 2023 to December 2023",
        category="clarification",
    ),
    PromptTemplate(
        label="msft_2024",
        description="Build MSFT time series from 2024-01-01 to 2024-12-31",
        response="Build MSFT from 2024-01-01 to 2024-12-31",
        category="clarification",
    ),
    PromptTemplate(
        label="googl_q1_2023",
        description="Build GOOGL time series for Q1 2023",
        response="Build GOOGL from Q1 2023 to Q1 2023",
        category="clarification",
    ),
    PromptTemplate(
        label="custom",
        description="Type your own request",
        response="__custom__",
        category="clarification",
    ),
]

# ── Category Index ────────────────────────────────────────────────────────

PROMPT_REGISTRY: dict[str, list[PromptTemplate]] = {
    "source_selection": SOURCE_SELECTION_PROMPTS,
    "gap_filling": GAP_FILLING_PROMPTS,
    "clarification": CLARIFICATION_PROMPTS,
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