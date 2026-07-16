"""
Interactive selection helpers for the time-series pipeline.

Provides rich-prompt-based selection for source and gap-fill method,
with numbered choices, validation, and re-prompt on invalid input.
"""

from __future__ import annotations

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_CHOICES = ["bloomberg", "reuters", "yahoo"]
GAP_METHOD_CHOICES = ["linear", "ffill_then_bfill", "rolling_median"]

SOURCE_LABELS: dict[str, str] = {
    "bloomberg": "Bloomberg Terminal (professional data)",
    "reuters": "Reuters Eikon (professional data)",
    "yahoo": "Yahoo Finance (free, public data)",
}

GAP_METHOD_LABELS: dict[str, str] = {
    "linear": "Linear interpolation — best for small gaps (<5%)",
    "ffill_then_bfill": "Forward-fill then back-fill — moderate gaps (5-20%)",
    "rolling_median": "Rolling median — large gaps (>20%) or high volatility",
}


def _build_choice_table(
    title: str,
    choices: list[str],
    labels: dict[str, str],
) -> Table:
    """Build a rich Table showing numbered choices with descriptions."""
    table = Table(
        title=title,
        title_style="bold cyan",
        border_style="cyan",
        show_header=True,
        header_style="bold white",
    )
    table.add_column("#", style="bold yellow", width=4)
    table.add_column("Choice", style="bold green", width=20)
    table.add_column("Description", style="white")
    for idx, choice in enumerate(choices, start=1):
        desc = labels.get(choice, "")
        table.add_row(str(idx), choice, desc)
    return table


def select_source(
    console: Console | None = None,
    *,
    default: str | None = None,
) -> str:
    """Interactively select a data source from the available choices.

    Parameters
    ----------
    console : Console or None
        Rich console for output.
    default : str or None
        Default source if user presses Enter without input.

    Returns
    -------
    str
        One of ``"bloomberg"``, ``"reuters"``, ``"yahoo"``.
    """
    console = console or Console()

    console.print()
    console.print(
        Panel(
            Text("ReACT Agent: Source Selection — Human Governance Required", style="bold cyan"),
            border_style="cyan",
        )
    )
    console.print(
        Text(
            "Thought: require human governance for source choice.\n"
            "Action: prompt user with ranked options.\n"
            "Observation: selected source captured.",
            style="italic white",
        )
    )
    console.print()

    table = _build_choice_table("Available Data Sources", SOURCE_CHOICES, SOURCE_LABELS)
    console.print(table)

    while True:
        raw = Prompt.ask(
            "[bold yellow]Enter source name or number[/bold yellow]",
            default=default,
            console=console,
        )
        raw = raw.strip().lower()

        # Check by number
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(SOURCE_CHOICES):
                return SOURCE_CHOICES[idx - 1]

        # Check by name
        if raw in SOURCE_CHOICES:
            return raw

        console.print(
            f"[red]Invalid choice:[/red] {raw!r}. "
            f"Enter a number (1-{len(SOURCE_CHOICES)}) or one of: {', '.join(SOURCE_CHOICES)}"
        )


def select_gap_method(
    console: Console | None = None,
    *,
    default: str | None = None,
) -> str:
    """Interactively select a gap-fill method from the available choices.

    Parameters
    ----------
    console : Console or None
        Rich console for output.
    default : str or None
        Default method if user presses Enter without input.

    Returns
    -------
    str
        One of ``"linear"``, ``"ffill_then_bfill"``, ``"rolling_median"``.
    """
    console = console or Console()

    console.print()
    console.print(
        Panel(
            Text("ReACT Agent: Gap-Fill Method Selection — Human Governance Required", style="bold cyan"),
            border_style="cyan",
        )
    )
    console.print(
        Text(
            "Thought: determine missing-data strategy suitability.\n"
            "Action: present gap analysis recommendations.\n"
            "Observation: method selected by human.",
            style="italic white",
        )
    )
    console.print()

    table = _build_choice_table("Available Gap-Fill Methods", GAP_METHOD_CHOICES, GAP_METHOD_LABELS)
    console.print(table)

    while True:
        raw = Prompt.ask(
            "[bold yellow]Enter gap-fill method name or number[/bold yellow]",
            default=default,
            console=console,
        )
        raw = raw.strip().lower()

        # Check by number
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(GAP_METHOD_CHOICES):
                return GAP_METHOD_CHOICES[idx - 1]

        # Check by name
        if raw in GAP_METHOD_CHOICES:
            return raw

        console.print(
            f"[red]Invalid choice:[/red] {raw!r}. "
            f"Enter a number (1-{len(GAP_METHOD_CHOICES)}) or one of: {', '.join(GAP_METHOD_CHOICES)}"
        )