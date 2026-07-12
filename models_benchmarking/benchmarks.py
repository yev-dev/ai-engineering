 #!/usr/bin/env python3
"""
Ollama Model Speed Benchmark - Core benchmark logic.

Provides LangChain tools and a ReAct agent that an LLM drives
to run inference benchmarks against locally available Ollama models.
The LLM decides which tool to call next based on conversation context.

Usage (programmatic):
    from benchmarks import build_benchmark_agent, BenchmarkState
"""

import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import ollama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt


# ==================== Benchmark Data Structures ====================


@dataclass
class BenchmarkResult:
    """Stores benchmark metrics for a single model and prints formatted report."""

    model_name: str = ""
    iterations: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add_latency(self, ms: float) -> None:
        """Record a single latency measurement."""
        self.iterations += 1
        self.latencies_ms.append(ms)

    @property
    def mean_ms(self) -> float:
        """Mean latency in milliseconds."""
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def median_ms(self) -> float:
        """Median latency in milliseconds."""
        return float(np.median(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        """95th percentile latency in milliseconds."""
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0

    @property
    def p99_ms(self) -> float:
        """99th percentile latency in milliseconds."""
        return float(np.percentile(self.latencies_ms, 99)) if self.latencies_ms else 0.0

    @property
    def throughput_tokens_sec(self) -> Optional[float]:
        """Calculate tokens/second based on mean latency and completion tokens."""
        if not self.latencies_ms or self.mean_ms == 0 or not self.completion_tokens:
            return None
        return (self.completion_tokens / self.mean_ms) * 1000

    def print_report(self) -> None:
        """Prints formatted benchmark report to console."""
        print(f"\n{'=' * 60}")
        print(f"📊 BENCHMARK REPORT: {self.model_name}")
        print(f"{'=' * 60}")
        print(f"  Iterations:        {self.iterations}")
        print(f"  ✅ Mean latency:     {self.mean_ms:>8.2f} ms")
        print(f"  🎯 Median latency:   {self.median_ms:>8.2f} ms")
        print(f"  🟢 P95 latency:      {self.p95_ms:>8.2f} ms")
        print(f"  🔴 P99 latency:      {self.p99_ms:>8.2f} ms")
        if self.throughput_tokens_sec:
            print(f"  ⚡ Throughput:       ~{self.throughput_tokens_sec:.1f} tokens/sec")
        if self.latencies_ms:
            latency_std = float(np.std(self.latencies_ms))
            print(f"  📉 Std deviation:    {latency_std:>8.2f} ms")
            outlier_count = sum(1 for l in self.latencies_ms if l > self.mean_ms * 3)
            if outlier_count > 0:
                print(f"  ⚠️ Outliers (>3x mean): {outlier_count} measurements")
        print(f"{'=' * 60}\n")


# ==================== Shared State ====================
# A mutable container shared across all tools. The tools read from and write to
# this object so the agent can coordinate workflow steps. The LLM decides which
# tool to invoke and when — the state simply records results for downstream tools.


class BenchmarkState:
    """Mutable container shared across tools for benchmark workflow state."""

    def __init__(
        self,
        iterations: int = 30,
        warmup_iterations: int = 3,
        max_tokens: int = 128,
    ):
        self.available_models: List[Dict[str, Any]] = []
        self.selected_models: List[str] = []
        self.results: List[BenchmarkResult] = []
        self.iterations = iterations
        self.warmup_iterations = warmup_iterations
        self.max_tokens = max_tokens
        self.status: str = "init"


# Global singleton that tools access via closure. Set by build_benchmark_agent().
_state: BenchmarkState = BenchmarkState()


# ==================== LangChain Tools ====================
# Each tool encapsulates one phase of the benchmark workflow.
# The LLM decides which tool to invoke next based on the conversation
# history and the guidance in the system prompt. There is no hard-coded
# routing between tools — the LLM reads tool outputs and chooses what to do.


@tool
def list_ollama_models_tool() -> str:
    """List all locally available Ollama models via the Ollama API.
    Call this first to discover which models are available for benchmarking."""
    print("\n🔍 Scanning for locally available Ollama models...")
    try:
        models_response = ollama.list()
        models = models_response.get("models", [])
        if not models:
            return "❌ No Ollama models found locally. Pull a model first with: ollama pull <model-name>"

        model_list = []
        for m in models:
            name = m.get("name", m.get("model", "unknown"))
            if name.endswith(":latest"):
                name = name[:-7]
            model_list.append({
                "name": name,
                "size": m.get("size", 0),
                "modified_at": m.get("modified_at", ""),
            })

        _state.available_models = model_list

        lines = [f"\n📋 Found {len(model_list)} locally available model(s):"]
        for i, m in enumerate(model_list, 1):
            size_gb = m["size"] / (1024 ** 3) if m["size"] else 0
            lines.append(f"  {i:2d}. {m['name']:30s} ({size_gb:.1f} GB)")

        return "\n".join(lines)

    except Exception as e:
        return f"❌ Failed to list Ollama models: {e}. Make sure Ollama is running (ollama serve)"


@tool
def ask_user_to_select_models_tool() -> str:
    """Ask the user to pick which models to benchmark.
    Call this after listing available models. Presents the list and captures
    the user's choice via comma-separated numbers, 'all', or 'q' to quit."""
    models = _state.available_models
    if not models:
        return "⚠️ No models available. Call list_ollama_models_tool first."

    model_names = [m["name"] for m in models]

    print("\n📝 Select models to benchmark:")
    print("   Enter comma-separated numbers (e.g., '1,3'), 'all', or 'q' to quit")

    selection = interrupt("model_selection")

    selected: List[str] = []
    if isinstance(selection, str):
        sel = selection.strip().lower()
        if sel in ("q", "quit"):
            return "❌ Benchmark cancelled by user."
        elif sel == "all":
            selected = model_names[:]
        else:
            parts = [p.strip() for p in sel.split(",")]
            for part in parts:
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(model_names):
                        selected.append(model_names[idx])

    if not selected:
        return "⚠️ No valid models selected."

    _state.selected_models = selected
    return f"✅ Selected {len(selected)} model(s): {', '.join(selected)}"


@tool
def confirm_selection_tool() -> str:
    """Ask the user to confirm the model selection before running benchmarks.
    Call this after models have been selected."""
    selected = _state.selected_models
    if not selected:
        return "⚠️ No models selected. Call ask_user_to_select_models_tool first."

    print(f"\n📝 Run benchmarks for {len(selected)} model(s): {', '.join(selected)}? (y/n)")

    confirmation = interrupt("confirmation")

    if isinstance(confirmation, str) and confirmation.strip().lower() in ("y", "yes"):
        return "✅ Proceeding with benchmarks."
    else:
        return "❌ Benchmark cancelled by user."


@tool
def run_benchmarks_tool() -> str:
    """Run inference benchmarks for all selected models.
    Call this after the user has confirmed. Benchmarks each model
    with warmup iterations followed by timed inference runs."""
    selected = _state.selected_models
    if not selected:
        return "⚠️ No models selected. Call confirm_selection_tool first."

    iterations = _state.iterations
    warmup = _state.warmup_iterations
    max_tokens = _state.max_tokens

    test_prompt = (
        "Write a brief summary of the benefits of quantitative finance "
        "in 2-3 sentences."
    )

    results: List[BenchmarkResult] = []
    report_lines = []

    for model_name in selected:
        print(f"\n{'#' * 60}")
        print(f"# Benchmarking model: {model_name}")
        print(f"{'#' * 60}")

        result = BenchmarkResult(model_name=model_name)

        try:
            llm = ChatOllama(
                model=model_name,
                temperature=0,
                num_predict=max_tokens,
            )

            if warmup > 0:
                print(f"  🔧 Warming up ({warmup} passes)...")
                for _ in range(warmup):
                    llm.invoke([HumanMessage(content="Say 'warmup'")])

            print(f"  🔥 Running {iterations} inference iterations...")
            for i in range(iterations):
                t0 = time.perf_counter()
                response = llm.invoke([HumanMessage(content=test_prompt)])
                t1 = time.perf_counter()

                latency_ms = (t1 - t0) * 1000
                result.add_latency(latency_ms)

                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = response.usage_metadata
                    result.prompt_tokens += usage.get("input_tokens", 0) or 0
                    result.completion_tokens += usage.get("output_tokens", 0) or 0

                if (i + 1) % 10 == 0:
                    print(f"    Progress: {i + 1}/{iterations} iterations")

            result.print_report()
            results.append(result)
            report_lines.append(
                f"{model_name}: mean={result.mean_ms:.2f}ms, "
                f"p95={result.p95_ms:.2f}ms, p99={result.p99_ms:.2f}ms"
                f"{', throughput=' + str(round(result.throughput_tokens_sec, 1)) + 'tok/s' if result.throughput_tokens_sec else ''}"
            )

        except Exception as e:
            err_msg = f"❌ Benchmark failed for {model_name}: {e}"
            print(err_msg)
            report_lines.append(err_msg)

    _state.results = results
    return "✅ Benchmarking complete.\n" + "\n".join(report_lines)


@tool
def report_results_tool() -> str:
    """Generate a final comparison report and save results to a JSON file.
    Call this after benchmarks have been run."""
    results = _state.results
    if not results:
        return "⚠️ No benchmark results. Call run_benchmarks_tool first."

    output: Dict[str, Any] = {}
    for r in results:
        output[r.model_name] = {
            "iterations": r.iterations,
            "mean_latency_ms": round(r.mean_ms, 2),
            "median_latency_ms": round(r.median_ms, 2),
            "p95_latency_ms": round(r.p95_ms, 2),
            "p99_latency_ms": round(r.p99_ms, 2),
            "throughput_tokens_sec": (
                round(r.throughput_tokens_sec, 2) if r.throughput_tokens_sec else None
            ),
        }

    output_path = "ollama_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    lines = [f"\n{'=' * 60}", "📊 FINAL COMPARISON REPORT", f"{'=' * 60}"]
    lines.append(f"\n  {'Model':30s} {'Mean (ms)':>10s} {'P95 (ms)':>10s} {'P99 (ms)':>10s} {'Tokens/s':>10s}")
    lines.append(f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    for r in results:
        tps = f"{r.throughput_tokens_sec:.1f}" if r.throughput_tokens_sec else "N/A"
        lines.append(f"  {r.model_name:30s} {r.mean_ms:>10.2f} {r.p95_ms:>10.2f} {r.p99_ms:>10.2f} {tps:>10s}")

    lines.append(f"\n💾 Results saved to {output_path}")
    return "\n".join(lines)


# ==================== Build LLM-Driven Agent ====================


def build_benchmark_agent(
    iterations: int = 30,
    warmup_iterations: int = 3,
    max_tokens: int = 128,
    agent_model: str = "llama3.2",
) -> Any:
    """Build a ReAct agent where an LLM drives the benchmark workflow.

    What is a ReAct agent?
    ----------------------
    ReAct (Reasoning + Acting) is a pattern where an LLM alternates between
    two phases in a loop:
      1. **Reasoning**: the LLM examines the conversation history and decides
         what to do next — either respond directly with a final answer or
         invoke a tool.
      2. **Acting**: if the LLM chose to call a tool, the tool executes and
         its result is returned to the LLM. The LLM then reasons again,
         incorporating the new information.

    This loop continues until the LLM responds without requesting any tool
    call — at which point the graph terminates. The LLM itself decides the
    sequence of tool calls based on the system prompt guidance and the
    outputs it observes from each tool. There are NO hard-coded conditional
    edges routing between steps.

    How the conversation history drives the agent
    ---------------------------------------------
    The agent's "brain" is the full conversation history — a list of messages
    that grows with each step. This history is what the LLM sees as input
    when it reasons. It consists of:

    1. **SystemMessage (the system prompt):**
       - Set via the `state_modifier` parameter of `create_react_agent`.
       - Injected at the start of every LLM call, before any other messages.
       - Defines the agent's role, goals, and the general workflow steps.
       - The LLM treats this as persistent instructions that override any
         user or tool messages if there's a conflict.
       - In this benchmark, the system prompt tells the agent it is a
         "benchmark automation agent" and lists the 5 recommended steps.

    2. **HumanMessage (user messages):**
       - The initial invocation from cli.py sends an empty messages list
         `{"messages": []}`, so the LLM only sees the system prompt at first.
       - When a tool calls `interrupt()`, the graph pauses and cli.py
         captures user input via `input("> ")`. That input is injected back
         as a HumanMessage via `Command(resume=user_input)`.
       - The LLM sees these user responses as part of the conversation and
         can react to them (e.g., if the user types "q" for quit, the LLM
         should stop).

    3. **AIMessage (LLM responses):**
       - Every time the LLM responds (either with a final answer or a tool
         call request), that response is recorded as an AIMessage.
       - If the LLM requests a tool call, the AIMessage contains
         `tool_calls` — structured arguments specifying which tool to invoke
         and with what parameters.

    4. **ToolMessage (tool outputs):**
       - After a tool executes, its return value is wrapped in a ToolMessage
         and appended to the history.
       - The LLM reads this on the next reasoning step to decide what to do
         next.

    The agent loop in detail:
       a) LLM receives: [SystemMessage, HumanMessage?, AIMessage*, ToolMessage*]
       b) LLM reasons and emits an AIMessage.
       c) If AIMessage has tool_calls → tools node executes them →
          ToolMessages appended → go back to (a).
       d) If AIMessage has no tool_calls → graph terminates (final answer).

    Conversation History strategies:
       - **Append-only (this project):** Keep all messages. Simple but can
         exceed the LLM's context window for very long workflows.
       - **Windowing / Sliding window:** Keep only the last N messages.
         Drops older context to stay within token limits.
       - **Summarization:** Periodically summarize older messages into a
         single condensed message. Preserves context while reducing tokens.
       - **Structured output (Approach 5):** Instead of appending raw tool
         outputs, the LLM emits structured JSON that a controller interprets.
         The controller maintains its own compact state (current_step,
         step_history) rather than relying on the full message list.
       - **External database / persistent storage:** Store the full message
         history in a database (SQLite, PostgreSQL, Redis) while keeping
         only recent messages in the LLM's context. The LLM queries the
         database via a retrieval tool when it needs older context.
       - **Vector store / RAG:** Embed messages into a vector database.
         The LLM performs similarity search to retrieve relevant past
         messages on demand. Keeps the context window small while
         preserving access to the full history.
       - **External memory agent:** A separate module manages a knowledge
         graph or key-value store that the main agent reads/writes via
         tools. The main agent only keeps a summary in its context window.

    How databases extend context:
       The conversation history is stored in-memory as part of the LangGraph
       state. To extend beyond the LLM's context window limit:
       1. **Replace the checkpointer** — Use SqliteSaver, PostgresSaver, or
          RedisSaver instead of MemorySaver. This persists the full graph
          state (including messages) to a database, surviving restarts.
          However, the full message list is still sent to the LLM.
       2. **Add a retrieval tool** — A tool that queries an external
          database for relevant history. The LLM calls it on demand,
          keeping the conversation window small.
       3. **Periodic summarization + external storage** — After N steps,
          compress messages into a summary. Store raw messages in a
          database. Replace the raw messages in context with the summary.

    Conversation History vs. Checkpoints:
       These are two separate mechanisms that are often confused:
       - **Conversation history** is the LLM's input context — the list of
         messages it reads to reason. It is *part of* the graph state.
       - **Checkpoints** are snapshots of the entire graph state (including
         messages) stored by the checkpointer. They enable pause/resume.
       The conversation history is *inside* the checkpoint, not the other
       way around. You can have conversation history without checkpoints
       (if you don't need interrupt/resume), but checkpoints always contain
       the full conversation history.

    In this implementation:
    - The LLM that drives decisions is powered by ChatOllama using the
      model specified via ``agent_model`` (default: 'llama3.2',
      temperature=0 for deterministic output).
    - Tools available to the LLM are: list_ollama_models_tool,
      ask_user_to_select_models_tool, confirm_selection_tool,
      run_benchmarks_tool, report_results_tool.
    - The system prompt describes the intended 5-step workflow, but the LLM
      uses its own judgment to decide the order (and can adapt if e.g. a
      tool returns an error or the user cancels).
    - A MemorySaver checkpointer is attached so that interrupt()-based
      human-in-the-loop tools can pause and resume execution.
    - The initial message list is empty (`{"messages": []}`), so the first
      LLM call only contains the system prompt. The agent must proactively
      call `list_ollama_models_tool` to start the workflow.

    Args:
        iterations: Number of inference iterations per model.
        warmup_iterations: Number of warmup iterations.
        max_tokens: Maximum tokens to generate per inference.
        agent_model: Ollama model name to use for the driving LLM
            (default: 'llama3.2'). Any model available locally via
            ``ollama list`` can be used, e.g. 'llama3.2', 'mistral',
            'phi3', 'gemma2', etc.

    Returns:
        A compiled LangGraph StateGraph (the ReAct agent) ready for streaming.
    """
    # Store config in the shared state.
    _state.iterations = iterations
    _state.warmup_iterations = warmup_iterations
    _state.max_tokens = max_tokens

    # The LLM that drives the workflow — it decides which tool to call next.
    # The model is configurable via agent_model; any locally available
    # Ollama model can be used. Lower temperatures favour determinism.
    llm = ChatOllama(
        model=agent_model,
        temperature=0,
    )

    # Tools the agent can invoke. The LLM chooses which to call and when.
    tools = [
        list_ollama_models_tool,
        ask_user_to_select_models_tool,
        confirm_selection_tool,
        run_benchmarks_tool,
        report_results_tool,
    ]

    # The system prompt guides the LLM on the intended workflow, but the LLM
    # ultimately decides the sequence of tool calls based on tool outputs.
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

    # ================================================================
    # create_react_agent — what it is and how it works
    # ================================================================
    # create_react_agent() is a convenience factory from
    # langgraph.prebuilt that builds a ReAct (Reasoning + Acting) agent.
    # It constructs a StateGraph internally with:
    #
    #   1. An "agent" node that calls the LLM with the conversation history
    #      (system prompt + messages). The LLM reasons and either responds
    #      directly or emits a tool call request.
    #   2. A "tools" node that executes whatever tool the LLM requested
    #      and returns the result as a ToolMessage.
    #   3. A loop edge: after the tools node, control goes back to the
    #      agent node so the LLM can examine the tool output and decide
    #      whether to call another tool or respond with a final answer.
    #   4. A conditional edge from the agent node: if the LLM responds
    #      without requesting a tool call, the graph terminates.
    #
    # The agent loop is: LLM thinks → calls tool → observes result →
    # LLM thinks again → ... → LLM returns final answer → END.
    #
    # Because we pass checkpointer=MemorySaver(), the graph state is
    # persisted at each step. This is required by interrupt() calls in
    # the selection/confirmation tools — when a tool calls interrupt(),
    # the graph pauses and yields control to the caller (cli.py), which
    # can later resume by sending Command(resume=...).
    # ================================================================
    # Alternatives for LLM-driven workflows
    # ================================================================
    # There are several approaches to make an LLM drive a multi-step
    # workflow like this benchmark. Each has trade-offs:
    #
    # 1. **Prompt-driven (this approach):**
    #    A single ReAct agent with a system prompt describing the desired
    #    workflow. The LLM decides tool call order based on the prompt +
    #    tool outputs. Pros: flexible, can handle unexpected situations.
    #    Cons: may skip steps or loop if the prompt isn't precise enough.
    #    Mitigation: prompt engineering + tool-level guards (each tool
    #    checks preconditions and returns clear error messages).
    #
    # 2. **Tool-driven (tool metadata & constraints):**
    #    Each tool has a detailed docstring that specifies when it should
    #    and shouldn't be called. The LLM relies on tool descriptions to
    #    decide sequencing. Pros: the LLM sees exactly what each tool
    #    does. Cons: still depends on LLM reasoning, no enforcement.
    #
    # 3. **Heuristic / rule-based supervisor:**
    #    A small deterministic "supervisor" layer wraps the LLM and
    #    enforces ordering. For example: after list_ollama_models_tool
    #    returns successfully, the supervisor only allows the next call
    #    to be ask_user_to_select_models_tool. Pros: guarantees correct
    #    sequence. Cons: loses flexibility; the LLM can't adapt.
    #
    # 4. **Graph-based with conditional edges (the old approach):**
    #    A manually defined StateGraph with add_conditional_edges that
    #    route based on state.status. Pros: fully deterministic, no LLM
    #    required for routing. Cons: rigid, requires manual maintenance.
    #
    # 5. **Structured output + state machine:**
    #    The LLM emits a structured output (e.g., JSON with "next_step"),
    #    which a small deterministic controller interprets to decide
    #    the next action. Pros: combines LLM flexibility with guaranteed
    #    termination. Cons: more complex to implement.
    # ================================================================
    # Termination & loop prevention
    # ================================================================
    # To ensure the agent terminates without infinite looping:
    #
    # - **Recursion limit:** LangGraph's StateGraph has a default
    #   recursion_limit (usually 25 steps). If the LLM keeps calling
    #   tools beyond this limit, LangGraph raises an exception and
    #   aborts. This is a safety net — increase it if workflows are
    #   legitimately long, but keep it reasonable.
    #
    # - **Tool-level preconditions:** each tool checks if the required
    #   preceding step was completed (e.g., run_benchmarks_tool checks
    #   _state.selected_models). If not, it returns a clear error
    #   message telling the LLM what to do instead. This prevents the
    #   LLM from calling tools out of order and getting stuck.
    #
    # - **Cancellation handling:** if the user cancels (via 'q' or 'n'),
    #   the tool returns a clear cancellation message. The system prompt
    #   tells the LLM to stop and report if cancelled.
    #
    # - **Prompt guidance:** the system prompt explicitly lists the 5
    #   steps in order and tells the LLM to "use your judgment". This
    #   reduces the chance of the LLM inventing new steps or looping.
    #
    # - **LLM temperature=0:** deterministic output reduces random
    #   deviations that could cause unexpected tool call sequences.
    # ================================================================
    from langgraph.checkpoint.memory import MemorySaver

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt,
        checkpointer=MemorySaver(),
    )

    return agent


def reset_state() -> None:
    """Reset the shared benchmark state to defaults. Useful for testing."""
    global _state
    _state = BenchmarkState()