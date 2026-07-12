#!/usr/bin/env python3
"""
Approach 5: Structured output + state machine.

The LLM emits a structured JSON output with "next_step" and "reasoning",
which a small deterministic controller interprets to decide the next
action. This combines LLM flexibility (the LLM reasons about what to do
next) with guaranteed termination (the controller enforces the workflow).

The controller maintains a state machine that validates transitions and
prevents invalid tool calls.

Usage:
    from benchmarks_structured_output import build_agent
"""

import json
from typing import Any, Dict, List, Optional
import re

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
try:
    # Newer langchain_core may re-export a pydantic v1 shim
    from langchain_core.pydantic_v1 import BaseModel, Field
except Exception:
    # Fall back to installing/importing pydantic directly
    from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import logging

# Module logger
logger = logging.getLogger("models_benchmarking.structured_output")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

from benchmarks_common import (
    configure_state,
    list_ollama_models_tool,
    ask_user_to_select_models_tool,
    confirm_selection_tool,
    run_benchmarks_tool,
    report_results_tool,
)


# ==================== Structured Output Schema ====================


class NextStep(BaseModel):
    """The LLM's structured decision about what to do next."""

    reasoning: str = Field(description="Brief reasoning for the next step")
    next_step: str = Field(
        description=(
            "One of: 'list', 'select', 'confirm', 'run', 'report', 'done'. "
            "'list'  → list_ollama_models_tool\n"
            "'select' → ask_user_to_select_models_tool\n"
            "'confirm' → confirm_selection_tool\n"
            "'run'   → run_benchmarks_tool\n"
            "'report' → report_results_tool\n"
            "'done'  → terminate the workflow"
        )
    )


# ==================== State Machine ====================
# Valid transitions between steps. The controller enforces these.


VALID_TRANSITIONS: Dict[str, List[str]] = {
    "init":   ["list"],
    "list":   ["select"],
    "select": ["confirm"],
    "confirm": ["run", "select"],  # can go back to select if cancelled
    "run":    ["report"],
    "report": ["done"],
    "done":   [],
    "cancelled": [],
}

COMPLETED_STEPS: List[str] = ["done", "cancelled"]

TOOL_MAP = {
    "list": list_ollama_models_tool,
    "select": ask_user_to_select_models_tool,
    "confirm": confirm_selection_tool,
    "run": run_benchmarks_tool,
    "report": report_results_tool,
}


def build_agent(
    iterations: int = 30,
    warmup_iterations: int = 3,
    max_tokens: int = 128,
    agent_model: str = "llama3.2",
) -> Any:
    """Build a structured output + state machine agent.

    A custom LangGraph with two nodes:
      1. "decide" node: calls the LLM with the conversation history and
         asks it to emit a structured NextStep (JSON with reasoning +
         next_step). The LLM decides what to do next based on the
         current state.
      2. "execute" node: a deterministic controller validates the
         transition against VALID_TRANSITIONS and either executes the
         tool or rejects the step and asks the LLM to reconsider.

    This combines LLM flexibility with guaranteed termination via
    the state machine.

    Args:
        iterations, warmup_iterations, max_tokens: benchmark config.
        agent_model: Ollama model for the driving LLM.

    Returns:
        A compiled LangGraph StateGraph ready for streaming.
    """
    configure_state(iterations, warmup_iterations, max_tokens)

    llm = ChatOllama(model=agent_model, temperature=0)
    parser = PydanticOutputParser(pydantic_object=NextStep)
    logger.info("Built structured-output parser for NextStep schema")

    system_prompt = SystemMessage(
        content=(
            "You are a benchmark automation agent. Your goal is to run inference "
            "speed benchmarks on locally available Ollama models.\n\n"

            "You have 5 tools available:\n"
            "  list    → list_ollama_models_tool — discover available models\n"
            "  select  → ask_user_to_select_models_tool — let the user pick models\n"
            "  confirm → confirm_selection_tool — confirm the user's choice\n"
            "  run     → run_benchmarks_tool — execute the benchmarks\n"
            "  report  → report_results_tool — display and save results\n\n"

            "Your job is to output a JSON object with:\n"
            '  - "reasoning": brief explanation of your decision\n'
            '  - "next_step": one of: list, select, confirm, run, report, done\n\n'

            "Start by listing models. After each tool executes, examine the "
            "tool output to decide the next step. If the user cancels, set "
            'next_step to "done".'
        )
    )

    # Custom state for this graph
    class AgentState(dict):
        messages: List = []
        current_step: str = "init"
        step_history: List[str] = []

    def decide_node(state: dict) -> dict:
        """LLM decides the next step by emitting structured JSON."""
        # Build the message list: system prompt + conversation history
        logger.debug("decide_node invoked; current_step=%s", state.get("current_step"))
        messages = [system_prompt]

        # Add current status context
        status_msg = f"Current workflow step: {state.get('current_step', 'init')}. "
        step_history = state.get("step_history", [])
        if step_history:
            status_msg += f"Steps completed so far: {', '.join(step_history)}. "
        status_msg += "What should the next step be?"
        messages.append(HumanMessage(content=status_msg))

        # Add any tool output from the last execution
        last_output = state.get("last_output")
        if last_output:
            messages.append(HumanMessage(content=f"Last tool output: {last_output}"))

        # Add the format instruction
        messages.append(HumanMessage(
            content=f"{parser.get_format_instructions()}\n\nOutput ONLY valid JSON."
        ))

        # Get LLM response
        response = llm.invoke(messages)
        logger.debug("LLM response received: %s", getattr(response, "content", str(response)))

        # Robust extraction of textual content from the LLM response object.
        def _response_to_text(resp: Any) -> str:
            if resp is None:
                return ""
            if isinstance(resp, str):
                return resp
            # Try common attributes
            raw = getattr(resp, "content", None) or getattr(resp, "text", None) or resp

            # If list of message parts, join their content
            if isinstance(raw, list):
                parts: List[str] = []
                for it in raw:
                    if isinstance(it, str):
                        parts.append(it)
                    elif isinstance(it, dict):
                        parts.append(it.get("content") or it.get("text") or json.dumps(it))
                    else:
                        parts.append(getattr(it, "content", None) or str(it))
                return "\n".join([p for p in parts if p])

            if isinstance(raw, dict):
                # Common LLM result shapes
                if "content" in raw and isinstance(raw["content"], str):
                    return raw["content"]
                if "choices" in raw and isinstance(raw["choices"], list) and raw["choices"]:
                    first = raw["choices"][0]
                    if isinstance(first, dict):
                        if "message" in first and isinstance(first["message"], dict):
                            return first["message"].get("content") or json.dumps(first)
                        return first.get("text") or json.dumps(first)
                # Last resort: stringify
                return json.dumps(raw)

            return str(raw)

        text = _response_to_text(response)

        # Parse the structured output using the PydanticOutputParser, with
        # several fallback strategies for unstructured or noisy LLM output.
        next_step = "done"
        reasoning = ""
        parse_error = None
        try:
            parsed = parser.parse(text)
            next_step = parsed.next_step
            reasoning = parsed.reasoning
            logger.info("LLM decided next_step=%s; reasoning=%s", next_step, reasoning)
            print(f"\n🧠 LLM reasoning: {reasoning}")
            print(f"   Decided next step: {next_step}")
        except Exception as e:
            parse_error = e
            logger.warning("PydanticOutputParser.parse() failed: %s", e)
            # Try direct JSON parsing of the extracted text
            try:
                data = json.loads(text)
                # Look for next_step/reasoning at top-level or nested under common keys
                def _find_keys(d: Any, key: str):
                    if not isinstance(d, dict):
                        return None
                    if key in d:
                        return d[key]
                    # Common wrapper keys
                    for k in ("properties", "output", "result", "response", "data"):
                        if k in d and isinstance(d[k], dict) and key in d[k]:
                            return d[k][key]
                    # Search one level deeper
                    for v in d.values():
                        if isinstance(v, dict) and key in v:
                            return v[key]
                    return None

                next_step = _find_keys(data, "next_step") or _find_keys(data, "nextStep") or "done"
                reasoning = _find_keys(data, "reasoning") or _find_keys(data, "reason") or ""
                logger.info("Parsed JSON fallback next_step=%s", next_step)
                print(f"\n🧠 LLM reasoning (json fallback): {reasoning}")
                print(f"   Decided next step: {next_step}")
            except Exception:
                # Try to extract the first JSON object in the text blob
                try:
                    m = re.search(r"(\{(?:.|\n)*\})", text)
                    if m:
                        data = json.loads(m.group(1))
                        next_step = data.get("next_step", "done")
                        reasoning = data.get("reasoning", "")
                        logger.info("Extracted JSON substring next_step=%s", next_step)
                        print(f"\n🧠 LLM reasoning (json-substring): {reasoning}")
                        print(f"   Decided next step: {next_step}")
                    else:
                        logger.exception("Failed to parse LLM response as JSON; marking next_step as 'done'", exc_info=True)
                        next_step = "done"
                except Exception:
                    logger.exception("Failed to extract JSON from LLM response; marking next_step as 'done'", exc_info=True)
                    next_step = "done"

        return {"next_step": next_step}

    def execute_node(state: dict) -> dict:
        """Deterministic controller: validate and execute the chosen step."""
        next_step = state.get("next_step", "done")
        current_step = state.get("current_step", "init")
        step_history = state.get("step_history", [])
        logger.debug("execute_node invoked; current_step=%s next_step=%s", current_step, next_step)

        # Check if this is a valid transition
        allowed = VALID_TRANSITIONS.get(current_step, [])
        if next_step not in allowed and next_step not in COMPLETED_STEPS:
            # Invalid transition — ask to reconsider
            return {
                "last_output": (
                    f"Invalid transition: '{current_step}' → '{next_step}'. "
                    f"Allowed: {allowed}. Please choose a valid next step."
                ),
                "next_step": None,
            }

        if next_step == "done" or next_step == "cancelled":
            logger.info("Workflow complete (state=%s)", next_step)
            print("✅ Workflow complete.")
            return {"next_step": "done"}

        # Execute the tool
        tool_func = TOOL_MAP.get(next_step)
        if tool_func:
            logger.info("Executing tool for step '%s'", next_step)
            result = tool_func.invoke({})
            logger.debug("Tool '%s' result: %s", next_step, result)
            print(result)

            # Check if cancelled
            if "cancelled" in result.lower():
                step_history = step_history + [next_step]
                return {
                    "current_step": "cancelled",
                    "next_step": None,
                    "step_history": step_history,
                    "last_output": result,
                }

            step_history = step_history + [next_step]
            return {
                "current_step": next_step,
                "next_step": None,
                "step_history": step_history,
                "last_output": result,
            }

        return {"next_step": "done"}

    def should_continue(state: dict) -> str:
        """Decide whether to loop or end."""
        if state.get("next_step") == "done":
            return "end"
        if state.get("current_step") in COMPLETED_STEPS:
            return "end"
        # If next_step is not None and not 'done', we go to execute
        if state.get("next_step") and state["next_step"] != "done":
            return "execute"
        # After execute, if next_step is None, go back to decide
        return "decide"

    # Build the graph
    workflow = StateGraph(AgentState)

    workflow.add_node("decide", decide_node)
    workflow.add_node("execute", execute_node)

    workflow.set_entry_point("decide")

    workflow.add_conditional_edges(
        "decide",
        should_continue,
        {
            "execute": "execute",
            "end": END,
            "decide": "decide",
        },
    )

    workflow.add_conditional_edges(
        "execute",
        should_continue,
        {
            "decide": "decide",
            "end": END,
        },
    )

    agent = workflow.compile(checkpointer=MemorySaver())
    return agent