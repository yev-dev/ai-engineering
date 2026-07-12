#!/usr/bin/env python3
"""
Approach 1: Prompt-driven ReAct agent.

A single ReAct agent with a system prompt describing the desired workflow.
The LLM decides tool call order based on the prompt + tool outputs.
This is the most flexible approach but depends on prompt engineering
to keep the LLM on track.

Usage:
    from benchmarks_prompt_driven import build_agent
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
    """Build a prompt-driven ReAct agent.

    The system prompt explicitly lists the 5 workflow steps. The LLM uses
    its own judgment to follow them. There are no hard-coded edges —
    the LLM decides the sequence based on the prompt + tool outputs.

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

    system_prompt = SystemMessage(
        content=(
            "You are a benchmark automation agent. Your goal is to run inference "
            "speed benchmarks on locally available Ollama models.\n\n"

            "Follow this general workflow, but use your judgment:\n"
            "1. First, call `list_ollama_models_tool` to discover available models.\n"
            "2. Then call `ask_user_to_select_models_tool` to let the user pick models.\n"
            "3. Call `confirm_selection_tool` to have the user confirm their choice.\n"
            "4. Call `run_benchmarks_tool` to execute the benchmarks.\n"
            "5. Call `report_results_tool` to display and save the final report.\n\n"

            "After each tool call, examine the result to decide the next step. "
            "If the user cancels at any point, stop and report that the benchmark "
            "was cancelled. Respond to the user in a helpful, concise manner."
        )
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        checkpointer=MemorySaver(),
    )

    return agent