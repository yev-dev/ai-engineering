from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


@dataclass(frozen=True)
class ToolMeta:
    name: str
    description: str
    framework: Literal["autogen", "langgraph"]


def autogen_tool(name: str, description: str):
    def decorator(func: Callable):
        setattr(func, "_autogen_tool_meta", ToolMeta(name=name, description=description, framework="autogen"))
        return func

    return decorator


def langgraph_tool(name: str, description: str):
    def decorator(func: Callable):
        setattr(func, "_langgraph_tool_meta", ToolMeta(name=name, description=description, framework="langgraph"))
        return func

    return decorator


def get_decorated_tools(obj: object, framework: Literal["autogen", "langgraph"]) -> dict[str, tuple[str, Callable]]:
    out: dict[str, tuple[str, Callable]] = {}
    meta_attr = "_autogen_tool_meta" if framework == "autogen" else "_langgraph_tool_meta"
    for attr_name in dir(obj):
        attr = getattr(obj, attr_name)
        meta = getattr(attr, meta_attr, None)
        if meta is None:
            continue
        out[meta.name] = (meta.description, attr)
    return out
