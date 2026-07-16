from __future__ import annotations

from ..common.tools.toolbox import CommonToolbox, get_autogen_tool_registry


def build_autogen_tool_registry(toolbox: CommonToolbox):
    return get_autogen_tool_registry(toolbox)


__all__ = ["build_autogen_tool_registry", "CommonToolbox"]
