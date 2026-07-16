from __future__ import annotations

from typing import Callable

from ..common.tools.toolbox import CommonToolbox, get_langgraph_tool_registry


def build_langgraph_tool_registry(toolbox: CommonToolbox):
    """Build the langgraph tool registry.

    If `langchain_core.tools.tool` is available, wrap the registered callables
    with that decorator so they appear as LangChain-compatible tools while
    preserving the original callable signature used by the orchestrator.
    """
    base = get_langgraph_tool_registry(toolbox)

    try:
        # prefer LangChain's tool decorator when available
        from langchain_core.tools import tool as tools  # type: ignore

        wrapped: dict[str, tuple[str, Callable]] = {}
        for name, (desc, func) in base.items():
            # create a wrapper that calls the original function but is decorated
            # with LangChain's `tool` so external LangChain integrations can
            # discover the tool metadata if needed.
            @tools(name=name, description=desc)
            def _wrapper(context: dict, args: dict, _func=func):
                return _func(context, args)

            wrapped[name] = (desc, _wrapper)

        return wrapped
    except Exception:
        # fallback to the existing registry when langchain_core isn't present
        return base


__all__ = ["build_langgraph_tool_registry", "CommonToolbox"]
