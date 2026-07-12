#!/usr/bin/env python3
"""
Approach 3: Heuristic / rule-based supervisor.

A small deterministic "supervisor" layer wraps the LLM and enforces
ordering. After each tool call, the supervisor checks the result and
only allows the next appropriate tool to be called. This guarantees
correct sequence but loses the LLM's ability to adapt.

The supervisor is implemented as a custom LangGraph StateGraph with
conditional edges that route based on state.status — the LLM is only
used within each node to handle human-in-the-loop interactions.

Usage:
    from benchmarks_heuristic_supervisor import build_agent
"""

from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from benchmarks_common import (
    BenchmarkState,
    configure_state,
    list_ollama_models_tool,
    ask_user_to_select_models_tool,
    confirm_selection_tool,
    run_benchmarks_tool,
    report_results_tool,
)


def build_agent(
    iterations: int = 30,
    warmup_iterations: int = 3,
    max_tokens: int = 128,
    agent_model: str = "llama3.2",
) -> Any:
    """Build a heuristic / rule-based supervisor agent.

    A deterministic StateGraph with hard-coded conditional edges routes
    between nodes. Each node calls one tool and updates state.status.
    The supervisor (conditional edges) checks the status and decides
    which node to route to next. The LLM is only used within nodes that
    need human interaction (select/confirm).

    This guarantees correct sequence but is rigid — the LLM cannot
    adapt the workflow.

    Args:
        iterations, warmup_iterations, max_tokens: benchmark config.
        agent_model: Ollama model (only used for human-interaction nodes).

    Returns:
        A compiled LangGraph StateGraph ready for streaming.
    """
    configure_state(iterations, warmup_iterations, max_tokens)

    # We use a simple dict-based state for the graph.
    # The tools use the shared _state from benchmarks_common internally.

    def node_list_models(state: dict) -> dict:
        result = list_ollama_models_tool.invoke({})
        print(result)
        return {"status": "models_listed" if "Found" in result else "no_models"}

    def node_select_models(state: dict) -> dict:
        result = ask_user_to_select_models_tool.invoke({})
        print(result)
        if "cancelled" in result.lower():
            return {"status": "cancelled"}
        return {"status": "models_selected"}

    def node_confirm(state: dict) -> dict:
        result = confirm_selection_tool.invoke({})
        print(result)
        if "cancelled" in result.lower():
            return {"status": "cancelled"}
        return {"status": "confirmed"}

    def node_run_benchmarks(state: dict) -> dict:
        result = run_benchmarks_tool.invoke({})
        print(result)
        return {"status": "completed"}

    def node_report(state: dict) -> dict:
        result = report_results_tool.invoke({})
        print(result)
        return {"status": "done"}

    # Build the deterministic graph
    workflow = StateGraph(dict)

    workflow.add_node("list_models", node_list_models)
    workflow.add_node("select_models", node_select_models)
    workflow.add_node("confirm", node_confirm)
    workflow.add_node("run_benchmarks", node_run_benchmarks)
    workflow.add_node("report", node_report)

    workflow.set_entry_point("list_models")

    workflow.add_conditional_edges(
        "list_models",
        lambda s: "select_models" if s.get("status") == "models_listed" else END,
    )
    workflow.add_conditional_edges(
        "select_models",
        lambda s: "confirm" if s.get("status") == "models_selected" else END,
    )
    workflow.add_conditional_edges(
        "confirm",
        lambda s: "run_benchmarks" if s.get("status") == "confirmed" else END,
    )
    workflow.add_edge("run_benchmarks", "report")
    workflow.add_edge("report", END)

    agent = workflow.compile(checkpointer=MemorySaver())
    return agent