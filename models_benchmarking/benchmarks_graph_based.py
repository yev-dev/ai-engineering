#!/usr/bin/env python3
"""
Approach 4: Graph-based with conditional edges.

A manually defined StateGraph with add_conditional_edges that route
based on state.status. This is the original approach — fully
deterministic with no LLM required for routing. Each node is a
function that calls one tool and returns the next status.

This is the most rigid approach but guarantees correct sequencing
and is easy to reason about.

Usage:
    from benchmarks_graph_based import build_agent
"""

from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from benchmarks_common import (
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
    """Build a graph-based agent with hard-coded conditional edges.

    A manually defined StateGraph with add_conditional_edges routes
    between nodes based on state.status. Each node is a simple function
    that calls the corresponding tool and returns a status update.
    No LLM is used for routing — the flow is fully deterministic.

    This is the most predictable approach but cannot adapt to
    unexpected situations.

    Args:
        iterations, warmup_iterations, max_tokens: benchmark config.
        agent_model: unused in this approach (no LLM for routing).

    Returns:
        A compiled LangGraph StateGraph ready for streaming.
    """
    configure_state(iterations, warmup_iterations, max_tokens)

    def node_list_models(state: dict) -> dict:
        result = list_ollama_models_tool.invoke({})
        print(result)
        return {"status": "models_listed" if "Found" in result else "no_models", "output": result}

    def node_select_models(state: dict) -> dict:
        result = ask_user_to_select_models_tool.invoke({})
        print(result)
        if "cancelled" in result.lower():
            return {"status": "cancelled", "output": result}
        return {"status": "models_selected", "output": result}

    def node_confirm(state: dict) -> dict:
        result = confirm_selection_tool.invoke({})
        print(result)
        if "cancelled" in result.lower():
            return {"status": "cancelled", "output": result}
        return {"status": "confirmed", "output": result}

    def node_run_benchmarks(state: dict) -> dict:
        result = run_benchmarks_tool.invoke({})
        print(result)
        return {"status": "completed", "output": result}

    def node_report(state: dict) -> dict:
        result = report_results_tool.invoke({})
        print(result)
        return {"status": "done", "output": result}

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