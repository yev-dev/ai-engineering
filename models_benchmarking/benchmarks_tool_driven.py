#!/usr/bin/env python3
"""
Approach 2: Tool-driven ReAct agent.

Each tool has a detailed docstring that specifies when it should and
shouldn't be called. The LLM relies on tool descriptions to decide
sequencing, with minimal system prompt guidance. The tool metadata
(description + parameter schema) is the primary driver.

Usage:
    from benchmarks_tool_driven import build_agent
"""

from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
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
    """Build a tool-driven ReAct agent.

    The system prompt is minimal — it only defines the agent's role.
    The LLM relies primarily on each tool's docstring (description) to
    understand when to call it. Each tool's docstring explicitly states
    preconditions (e.g., "Call this after listing available models").

    This demonstrates how tool metadata alone can guide an LLM through
    a multi-step workflow without detailed step-by-step instructions.

    Args:
        iterations, warmup_iterations, max_tokens: benchmark config.
        agent_model: Ollama model for the driving LLM.

    Returns:
        A compiled LangGraph StateGraph ready for streaming.
    """
    configure_state(iterations, warmup_iterations, max_tokens)

    llm = ChatOllama(model=agent_model, temperature=0)

    tools = [
        list_ollama_models_tool,
        ask_user_to_select_models_tool,
        confirm_selection_tool,
        run_benchmarks_tool,
        report_results_tool,
    ]

    # Minimal system prompt — the tool descriptions do the heavy lifting.
    system_prompt = SystemMessage(
        content=(
            "You are a benchmark automation agent. Your goal is to run inference "
            "speed benchmarks on locally available Ollama models. "
            "Use the available tools to complete this task. "
            "Read each tool's description carefully to understand when to call it. "
            "If the user cancels at any point, stop and report that the benchmark "
            "was cancelled."
        )
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        checkpointer=MemorySaver(),
    )

    return agent