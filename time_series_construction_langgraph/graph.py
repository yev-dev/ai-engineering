"""Minimal LangGraph workflow — single agent with all tools.

The LLM drives everything through ReAct format (Thought / Action /
Action Input / Final Answer).  No delegation, no multi-agent switching,
no agent registry lookups.  The system prompt tells the LLM how to
behave, and it calls any tool it needs.
"""
from __future__ import annotations

import json
import logging
import re
import ast
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command
from typing_extensions import Annotated, TypedDict

try:
    from .models import LLMRequest, ModelRequestFactory
    from .prompts import SYSTEM_PROMPT, request_prompt
    from .tools import get_tool, TOOL_REGISTRY
except ImportError:
    from models import LLMRequest, ModelRequestFactory
    from prompts import SYSTEM_PROMPT, request_prompt
    from tools import get_tool, TOOL_REGISTRY

logger = logging.getLogger(__name__)

_factory = ModelRequestFactory()


# ── State schema ─────────────────────────────────────────────────────


class GraphState(TypedDict):
    messages: Annotated[list, "The conversation transcript"]
    error: str | None


# ── LLM node ─────────────────────────────────────────────────────────


def _call_llm_node(state: GraphState) -> dict:
    logger.info("─" * 40)

    request = LLMRequest(
        system_prompt=SYSTEM_PROMPT,
        messages=state["messages"],
        tools=None,  # Tools are described in the system prompt text
    )
    try:
        response = _factory.chat(request)
    except Exception as exc:
        logger.exception("llm_call_failed")
        return {"error": str(exc)}

    new_messages = list(state["messages"])
    new_messages.append({"role": "assistant", "content": response})
    return {"messages": new_messages}


# ── Tools node ────────────────────────────────────────────────────────


def _call_tools_node(state: GraphState) -> dict:
    content = state["messages"][-1]["content"] if state["messages"] else ""

    matches = list(re.finditer(r"Action:\s*(\w+)\s*\n?\s*Action Input:\s*", content))
    if not matches:
        return {
            "messages": state["messages"]
            + [{"role": "user", "content": "Error: use Action: <name>\\nAction Input: <JSON>"}]
        }

    results = []
    interrupt = None

    for i, m in enumerate(matches):
        name = m.group(1)
        raw = content[m.end(): (matches[i + 1].start() if i + 1 < len(matches) else len(content))].strip()

        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            try:
                args = ast.literal_eval(raw)
            except Exception:
                args = {}

        logger.info("tool call: %s  args=%s", name, json.dumps(args, default=str)[:150])

        if name == "request_human_input":
            interrupt = {
                "prompt": args.get("prompt", "Please choose:"),
                "options": args.get("options", []),
            }
            continue

        tool = get_tool(name)
        if tool is None:
            results.append({"error": f"Unknown tool: {name}"})
        else:
            try:
                results.append(tool.invoke(args))
            except Exception as e:
                results.append({"error": str(e)})

    if interrupt:
        return {"interrupt": interrupt}

    new_messages = list(state["messages"])
    for r in results:
        new_messages.append({"role": "user", "content": f"Tool result: {json.dumps(r, default=str)}"})
    return {"messages": new_messages}


def _route(state: GraphState) -> Literal["tools", "__end__"]:
    content = state["messages"][-1]["content"] if state["messages"] else ""
    if "Final Answer:" in content:
        return "__end__"
    if "Action:" in content:
        return "tools"
    return "__end__"


# ── Graph construction ────────────────────────────────────────────────


def _build_graph() -> StateGraph:
    builder = StateGraph(GraphState)
    builder.add_node("llm", _call_llm_node)
    builder.add_node("tools", _call_tools_node)
    builder.set_entry_point("llm")
    builder.add_conditional_edges("llm", _route, {"tools": "tools", "__end__": END})
    builder.add_edge("tools", "llm")
    return builder


# ── Public API ────────────────────────────────────────────────────────


class TimeSeriesConstructionGraph:
    """Single-agent LangGraph workflow — the LLM drives everything."""

    def __init__(self) -> None:
        self._graph = _build_graph().compile(checkpointer=MemorySaver())
        self._thread: dict[str, Any] = {"configurable": {"thread_id": "default"}}

    def process_user_request(self, user_input: str) -> list[dict]:
        """Process a user request and return display-friendly events."""
        user_input = user_input.strip()
        if not user_input:
            return [{"type": "await", "agent": "System", "prompt": request_prompt(), "options": []}]

        events = [{"type": "user_request", "request": user_input}]

        try:
            for chunk in self._graph.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                self._thread,
            ):
                for node, output in (chunk.items() if isinstance(chunk, dict) else [("?", chunk)]):
                    self._process(output, events)
        except Exception as exc:
            logger.exception("graph_failed")
            events.append({"type": "error", "message": str(exc)})

        return events

    def process_user_response(self, user_input: str) -> list[dict]:
        """Resume after an interrupt."""
        if user_input.strip().lower() in {"cancel", "exit", "quit"}:
            return [{"type": "error", "message": "Cancelled."}]

        events = []
        try:
            for chunk in self._graph.invoke(Command(resume=user_input), self._thread):
                for node, output in (chunk.items() if isinstance(chunk, dict) else [("?", chunk)]):
                    self._process(output, events)
        except Exception as exc:
            logger.exception("resume_failed")
            events.append({"type": "error", "message": str(exc)})
        return events

    def _process(self, output: Any, events: list[dict]) -> None:
        if not isinstance(output, dict):
            return

        # Interrupt
        if output.get("interrupt"):
            events.append({
                "type": "await",
                "agent": "System",
                "prompt": output["interrupt"]["prompt"],
                "options": output["interrupt"].get("options", []),
            })
            return

        # Error
        if output.get("error"):
            events.append({"type": "error", "message": output["error"]})
            return

        # Messages
        msgs = output.get("messages", [])
        if not msgs:
            return

        last = msgs[-1]
        content = last.get("content", "")

        if "Final Answer:" in content:
            answer = content.split("Final Answer:", 1)[1].strip()
            events.append({"type": "final", "agent": "System", "answer": answer})
        elif "Action:" in content:
            m = re.search(r"Action:\s*(\w+)", content)
            name = m.group(1) if m else "tool"
            events.append({"type": "intermediate", "agent": "System", "message": f"Calling {name}…"})
        elif content:
            preview = content[:120].replace("\n", " ")
            events.append({
                "type": "intermediate",
                "agent": "System",
                "message": preview + ("…" if len(content) > 120 else ""),
            })

    def reset(self) -> None:
        logger.info("graph_reset")
        self._thread = {"configurable": {"thread_id": "default"}}