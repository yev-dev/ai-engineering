"""LangGraph state graph for the time series construction workflow.

Replaces the hand-rolled ReAct processor with a LangGraph StateGraph that
provides built-in tool calling, interrupt/resume for human-in-the-loop,
and checkpoint-based persistence.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from typing_extensions import Annotated, TypedDict

try:
    from .agents_definition import Agent, CallbackEvent, CallbackEventType, get_agent
    from .models import LLMRequest, ModelRequestFactory
    from .prompts import agent_system_prompt, request_prompt, unavailable_message
    from .tools import get_tool, TOOL_REGISTRY
except ImportError:
    from agents_definition import Agent, CallbackEvent, CallbackEventType, get_agent
    from models import LLMRequest, ModelRequestFactory
    from prompts import agent_system_prompt, request_prompt, unavailable_message
    from tools import get_tool, TOOL_REGISTRY

logger = logging.getLogger(__name__)

# ── State schema ─────────────────────────────────────────────────────


class GraphState(TypedDict):
    """Shared state across all graph nodes."""
    messages: Annotated[list, "The conversation transcript"]
    current_agent: str | None
    instrument: dict | None
    prices: dict | None
    quality_reports: list | None
    user_decision: Any | None
    artifacts: dict | None
    error: str | None


# ── LLM calling node ────────────────────────────────────────────────

_factory = ModelRequestFactory()


def _call_llm_node(state: GraphState, config: dict | None = None) -> dict:
    """Generic LLM node — every specialist agent uses this.

    The agent name is read from ``config["configurable"]["agent_name"]``
    or from state["current_agent"], falling back to Orchestrator.
    """
    # Try to get agent name from config, then from state, then default
    agent_name = "Orchestrator"
    if config and "configurable" in config:
        agent_name = config["configurable"].get("agent_name", state.get("current_agent", "Orchestrator"))
    elif state.get("current_agent"):
        agent_name = state["current_agent"]
    
    agent = get_agent(agent_name)
    if agent is None:
        return {"error": f"Unknown agent: {agent_name}"}

    prompt = agent_system_prompt(agent)
    tool_list = [get_tool(name) for name in agent.tools if get_tool(name) is not None]

    request = LLMRequest(
        system_prompt=prompt,
        messages=state["messages"],
        tools=tool_list if tool_list else None,
    )
    try:
        response = _factory.chat(request)
    except Exception as exc:
        logger.exception("llm_call_failed agent=%s", agent_name)
        return {"error": str(exc)}

    new_messages = list(state["messages"])
    new_messages.append({"role": "assistant", "content": response})
    return {"messages": new_messages, "current_agent": agent_name}


# ── Tool node (text-parsing version for ReAct LLMs) ───────────────────


def _call_tools_node(state: GraphState, config: dict | None = None) -> dict:
    """Parse ReAct tool calls from LLM output and execute them."""
    import re
    import ast
    import json
    from typing import cast
    
    last_msg = state["messages"][-1] if state["messages"] else {}
    content = last_msg.get("content", "")
    
    # Parse Action: / Action Input: pairs
    action_matches = list(re.finditer(r"Action:\s*([A-Za-z_]\w*)\s+Action Input:\s*", content))
    
    if not action_matches:
        return {"error": "No tool call found in LLM response"}
    
    all_results: list[Any] = []
    interrupt_payload: dict[str, Any] | None = None
    
    for index, match in enumerate(action_matches):
        name = match.group(1)
        input_start = match.end()
        input_end = action_matches[index + 1].start() if index + 1 < len(action_matches) else len(content)
        raw_input = content[input_start:input_end].strip()
        
        # Parse JSON or Python dict
        try:
            parsed = json.loads(raw_input)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(raw_input)
            except (SyntaxError, ValueError):
                parsed = {}
        
        # Handle request_human_input specially (sets up interrupt)
        if name == "request_human_input":
            prompt = parsed.get("prompt", "Please choose:")
            options = parsed.get("options")
            interrupt_payload = {"prompt": prompt, "options": options or [], "agent": state.get("current_agent", "System")}
            continue
        
        # Execute the tool
        tool = get_tool(name)
        if tool is None:
            all_results.append({"error": f"Unknown tool: {name}"})
        else:
            try:
                result = tool.invoke(parsed)
                all_results.append(result)
            except Exception as e:
                all_results.append({"error": str(e)})
    
    # If we have an interrupt, return it directly
    if interrupt_payload:
        return {"interrupt": interrupt_payload, "current_agent": state.get("current_agent")}
    
    # Otherwise, add tool results to messages for the next LLM turn
    new_messages = list(state["messages"])
    for result in all_results:
        new_messages.append({"role": "user", "content": f"Tool result: {json.dumps(result, default=str)}"})
    
    return {"messages": new_messages, "current_agent": state.get("current_agent")}


def _route_after_llm(
    state: GraphState,
) -> Literal["tools", "__end__"]:
    """Conditional edge: route to tools if the LLM called one, else check for final answer."""
    last_msg = state["messages"][-1] if state["messages"] else {}
    content = last_msg.get("content", "")

    # If the LLM produced a Final Answer, we're done
    if "Final Answer:" in content:
        return "__end__"

    # Check for ReAct tool calls
    if "Action:" in content:
        return "tools"
    
    return "__end__"


# ── Graph construction ──────────────────────────────────────────────

def _build_graph() -> StateGraph:
    builder = StateGraph(GraphState)

    # Nodes
    builder.add_node("call_llm", _call_llm_node)
    builder.add_node("tools", _call_tools_node)

    # Entry point
    builder.set_entry_point("call_llm")

    # Edges
    builder.add_conditional_edges(
        "call_llm",
        _route_after_llm,
        {"tools": "tools", "__end__": END},
    )
    builder.add_edge("tools", "call_llm")

    return builder


# ── Public API ──────────────────────────────────────────────────────


class TimeSeriesConstructionGraph:
    """LangGraph-based time series construction workflow.

    Usage::

        graph = TimeSeriesConstructionGraph()
        for event in graph.run("create a time series for AAPL 2023-2024"):
            print(event)
    """

    def __init__(self) -> None:
        self._checkpointer = MemorySaver()
        self._graph = _build_graph().compile(checkpointer=self._checkpointer)
        self._thread_id = "default"
        self._thread: dict[str, Any] = {"configurable": {"thread_id": self._thread_id, "agent_name": "Orchestrator"}}
        self._pending_interrupt: dict[str, Any] | None = None

    def process_user_request(self, user_input: str) -> list[CallbackEvent]:
        """Start or resume a workflow from a user request.

        Returns a list of ``CallbackEvent`` objects compatible with the
        existing ``cli.py`` display layer.
        """
        user_input = user_input.strip()
        logger.info("graph_request_received characters=%d", len(user_input))

        if not user_input:
            return [CallbackEvent(
                CallbackEventType.AWAITING_USER_INPUT,
                {"agent": "Orchestrator", "prompt": request_prompt(), "options": []},
            )]

        events: list[CallbackEvent] = []
        events.append(CallbackEvent(CallbackEventType.USER_REQUEST, {"request": user_input}))

        try:
            for chunk in self._graph.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                self._thread,
            ):
                self._process_chunk(chunk, events)
        except Exception as exc:
            logger.exception("graph_execution_failed")
            events.append(CallbackEvent(
                CallbackEventType.ERROR,
                {"message": f"Graph execution error: {exc}", "recoverable": True},
            ))

        return events

    def process_user_response(self, user_input: str) -> list[CallbackEvent]:
        """Resume after a human-in-the-loop interrupt."""
        logger.info("graph_resume_received characters=%d", len(user_input))

        if user_input.strip().lower() in {"cancel", "exit", "quit"}:
            return [CallbackEvent(
                CallbackEventType.ERROR,
                {"message": "Operation cancelled.", "recoverable": False},
            )]

        events: list[CallbackEvent] = []
        try:
            for chunk in self._graph.invoke(
                Command(resume=user_input),
                self._thread,
            ):
                self._process_chunk(chunk, events)
        except Exception as exc:
            logger.exception("graph_resume_failed")
            events.append(CallbackEvent(
                CallbackEventType.ERROR,
                {"message": f"Graph resume error: {exc}", "recoverable": True},
            ))

        return events

    def _process_chunk(self, chunk: Any, events: list[CallbackEvent]) -> None:
        """Convert a LangGraph stream chunk into CallbackEvents."""
        if isinstance(chunk, dict):
            # Check for interrupt (from LangGraph's interrupt mechanism)
            if interrupt := chunk.get("__interrupt__"):
                self._pending_interrupt = interrupt.value
                events.append(CallbackEvent(
                    CallbackEventType.AWAITING_USER_INPUT,
                    {
                        "agent": interrupt.value.get("agent", "System"),
                        "prompt": interrupt.value.get("prompt", ""),
                        "options": interrupt.value.get("options", []),
                    },
                ))
                return

            # Check for node output
            for node_name, output in chunk.items():
                if node_name == "tools" and output:
                    # Check for interrupt from tools node
                    if interrupt := output.get("interrupt"):
                        self._pending_interrupt = interrupt
                        events.append(CallbackEvent(
                            CallbackEventType.AWAITING_USER_INPUT,
                            {
                                "agent": interrupt.get("agent", "System"),
                                "prompt": interrupt.get("prompt", ""),
                                "options": interrupt.get("options", []),
                            },
                        ))
                        return
                
                if node_name == "call_llm" and output:
                    if output.get("error"):
                        events.append(CallbackEvent(
                            CallbackEventType.ERROR,
                            {"message": output["error"], "recoverable": True},
                        ))
                    elif output.get("messages"):
                        last_msg = output["messages"][-1]
                        content = last_msg.get("content", "")
                        if "Final Answer:" in content:
                            answer = content.split("Final Answer:", 1)[1].strip()
                            agent = output.get("current_agent", "Unknown")
                            events.append(CallbackEvent(
                                CallbackEventType.AGENT_COMPLETED,
                                {"agent": agent, "result": {"final_answer": answer}},
                            ))

    def reset(self) -> None:
        """Reset the graph state for a new session."""
        self._thread = {"configurable": {"thread_id": "default", "agent_name": "Orchestrator"}}
        self._pending_interrupt = None
        logger.info("graph_reset")