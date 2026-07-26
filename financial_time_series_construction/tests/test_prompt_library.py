"""Tests for the prompt library module."""
from __future__ import annotations

import pytest

from financial_time_series_construction.prompt_library import (
    format_prompt_menu,
    get_prompts,
    resolve_prompt_selection,
)


class TestGetPrompts:
    """Verify prompt retrieval by category."""

    def test_source_selection_prompts_exist(self) -> None:
        prompts = get_prompts("source_selection")
        assert len(prompts) >= 3
        labels = {p.label for p in prompts}
        assert "yahoo" in labels
        assert "bloomberg" in labels
        assert "reuters" in labels

    def test_gap_filling_prompts_exist(self) -> None:
        prompts = get_prompts("gap_filling")
        assert len(prompts) >= 3
        labels = {p.label for p in prompts}
        assert "linear_interpolation" in labels
        assert "forward_fill" in labels
        assert "backward_fill" in labels

    def test_clarification_prompts_exist(self) -> None:
        prompts = get_prompts("clarification")
        assert len(prompts) >= 2

    def test_unknown_category_returns_empty(self) -> None:
        prompts = get_prompts("nonexistent")
        assert prompts == []


class TestFormatPromptMenu:
    """Verify menu formatting."""

    def test_menu_contains_options(self) -> None:
        menu = format_prompt_menu("source_selection")
        assert "Available quick options:" in menu
        assert "[1]" in menu
        assert "yahoo" in menu

    def test_menu_with_context(self) -> None:
        menu = format_prompt_menu("source_selection", context="Choose a source below:")
        assert "Choose a source below:" in menu

    def test_unknown_category_returns_empty(self) -> None:
        menu = format_prompt_menu("nonexistent")
        assert menu == ""


class TestResolvePromptSelection:
    """Verify prompt resolution by index, label, and free-form text."""

    def test_resolve_by_numeric_index(self) -> None:
        result = resolve_prompt_selection("source_selection", "1")
        assert result == "yahoo"

    def test_resolve_by_label(self) -> None:
        result = resolve_prompt_selection("source_selection", "bloomberg")
        assert result == "bloomberg"

    def test_resolve_by_label_case_insensitive(self) -> None:
        result = resolve_prompt_selection("source_selection", "BLOOMBERG")
        assert result == "bloomberg"

    def test_resolve_gap_filling_by_index(self) -> None:
        result = resolve_prompt_selection("gap_filling", "2")
        assert result == "forward_fill"

    def test_resolve_gap_filling_by_label(self) -> None:
        result = resolve_prompt_selection("gap_filling", "linear_interpolation")
        assert result == "linear_interpolation"

    def test_free_form_returns_as_is(self) -> None:
        result = resolve_prompt_selection("source_selection", "some random text")
        assert result == "some random text"

    def test_unknown_category_returns_input(self) -> None:
        result = resolve_prompt_selection("nonexistent", "my_input")
        assert result == "my_input"