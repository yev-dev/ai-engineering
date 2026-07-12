#!/usr/bin/env python3
"""
Ollama Model Speed Benchmark - CLI entry point.

Parses command-line arguments and runs the benchmark workflow
using one of 5 implementation approaches selected via --method.

The agent's LLM (configurable via --agent-model) decides which tool
to call next — there are NO hard-coded workflow edges (except for
'graph-based' and 'heuristic-supervisor' which use deterministic graphs).

Usage:
    python cli.py
    python cli.py --method prompt-driven --agent-model llama3.2
    python cli.py --method tool-driven --iterations 20 --warmup 2 --max-tokens 64
    python cli.py --method heuristic-supervisor
    python cli.py --method graph-based
    python cli.py --method structured-output
"""

import sys
import argparse
import subprocess
import shutil

from langgraph.types import Command
import benchmarks_common as bc


# Map method names to their build_agent functions
METHODS = {
    "prompt-driven": "benchmarks_prompt_driven",
    "tool-driven": "benchmarks_tool_driven",
    "heuristic-supervisor": "benchmarks_heuristic_supervisor",
    "graph-based": "benchmarks_graph_based",
    "structured-output": "benchmarks_structured_output",
}

METHOD_DESCRIPTIONS = {
    "prompt-driven": (
        "A single ReAct agent with a system prompt describing the desired workflow. "
        "The LLM decides tool call order based on the prompt + tool outputs. "
        "Flexible but depends on prompt engineering."
    ),
    "tool-driven": (
        "Minimal system prompt — the LLM relies on each tool's detailed docstring "
        "to understand when to call it. Demonstrates tool metadata as the primary driver."
    ),
    "heuristic-supervisor": (
        "A deterministic StateGraph with conditional edges routes between nodes. "
        "Guarantees correct sequence but is rigid — the LLM cannot adapt."
    ),
    "graph-based": (
        "Manually defined StateGraph with add_conditional_edges routing based on "
        "state.status. Fully deterministic, no LLM required for routing."
    ),
    "structured-output": (
        "The LLM emits structured JSON with 'next_step', which a deterministic "
        "controller validaexptes against a state machine. Combines LLM flexibility "
        "with guaranteed termination."
    ),
}


def main() -> int:
    """Entry point: parse args, build the agent, and run with human-in-the-loop."""
    parser = argparse.ArgumentParser(
        description="Ollama Model Speed Benchmark - LLM-driven agent"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="prompt-driven",
        choices=list(METHODS.keys()),
        help="Which implementation approach to use (default: 'prompt-driven'). "
             "See descriptions below for details.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Number of inference iterations per model (default: 30)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per inference (default: 128)",
    )
    parser.add_argument(
        "--agent-model",
        type=str,
        default="llama3.2",
        help="Ollama model to use for the driving LLM (default: 'llama3.2'). "
             "Any locally available model works, e.g. 'mistral', 'phi3', 'gemma2'.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Automatically answer 'yes' to confirmation prompts (non-interactive runs).",
    )
    parser.add_argument(
        "--auto-select",
        type=str,
        default="",
        help=(
            "Automatically respond to model selection prompts. "
            "Provide comma-separated indices like '1,3' or 'all'."
        ),
    )
    parser.add_argument(
        "--results-json",
        type=str,
        default="",
        help="Optional path to save the structured benchmark JSON results.",
    )
    args = parser.parse_args()

    # Ensure requested Ollama model is available locally; if not, auto-fallback
    def _get_local_ollama_models() -> list[str]:
        """Return a list of local Ollama model names, or empty list if none or ollama missing."""
        if shutil.which("ollama") is None:
            return []
        try:
            proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
        except Exception:
            return []

        out = proc.stdout.strip()
        if not out:
            return []

        lines = [ln for ln in out.splitlines() if ln.strip()]
        # Skip header line if present (common output has header "NAME  TYPE  SIZE")
        if len(lines) <= 1:
            # Single-line output may just be model name
            return [lines[0].split()[0]] if lines else []

        models = []
        for ln in lines[1:]:
            parts = ln.split()
            if parts:
                models.append(parts[0])
        return models

    available_models = _get_local_ollama_models()
    if args.agent_model not in available_models:
        if available_models:
            fallback = available_models[0]
            print(f"⚠️  Requested model '{args.agent_model}' not found locally. Falling back to '{fallback}'.")
            print("   To use a different model, run: ollama list  (then pass --agent-model <name>)")
            args.agent_model = fallback
        else:
            # No local models found or ollama not installed — show actionable message
            if shutil.which("ollama") is None:
                print("\n❌ Ollama CLI not found. Install Ollama and pull a model, for example:")
                print("   https://ollama.com/docs/install")
            else:
                print("\n❌ No local Ollama models found. Install or pull a model, for example:")
                print("   ollama pull llama3.2")
                print("   ollama list")
            return 1

    # Print method description
    print(f"\n{'=' * 60}")
    print(f"🚀 Ollama Model Speed Benchmark Agent")
    print(f"{'=' * 60}")
    print(f"  Method:      {args.method}")
    print(f"  Description: {METHOD_DESCRIPTIONS[args.method]}")
    print(f"  Iterations:  {args.iterations}")
    print(f"  Warmup:      {args.warmup}")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Agent model: {args.agent_model}")
    print(f"{'=' * 60}")

    # Dynamically import the selected method's build_agent
    module_name = METHODS[args.method]
    try:
        mod = __import__(module_name, fromlist=["build_agent"])
        build_agent = getattr(mod, "build_agent")
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        print(f"   Make sure {module_name}.py exists in the same directory.")
        return 1

    # Build the agent
    agent = build_agent(
        iterations=args.iterations,
        warmup_iterations=args.warmup,
        max_tokens=args.max_tokens,
        agent_model=args.agent_model,
    )

    # If requested, tell the benchmark tools where to save JSON results.
    if args.results_json:
        try:
            bc.set_results_json_path(args.results_json)
        except Exception:
            print(f"⚠️ Failed to set results JSON path to {args.results_json}")

    thread_config = {"configurable": {"thread_id": "ollama-benchmark-1"}}

    # Run the agent. The ReAct loop will:
    #   1. LLM thinks and decides which tool to call
    #   2. If the tool calls interrupt(), the graph pauses and yields control here
    #   3. We capture user input and resume with Command(resume=...)
    #   4. The LLM reads the tool output and decides the next step
    # This repeats until the LLM decides the task is complete.
    try:
        # Initial agent invocation — the LLM receives the system prompt and
        # decides the first tool to call (should be list_ollama_models_tool).
        for event in agent.stream(
            {"messages": []}, thread_config, stream_mode="updates"
        ):
            for node_name, node_data in event.items():
                # Check if the agent is waiting for human input (interrupt).
                # create_react_agent yields '__interrupt__' events when a tool
                # calls interrupt(). We capture the last message to determine
                # which interrupt is active.
                if node_name == "__interrupt__":
                    # The agent paused on an interrupt(). Decide how to obtain
                    # user input: interactive `input()` when a TTY, or
                    # automatic responses when running non-interactively or
                    # when `--yes` / `--auto-select` flags are supplied.
                    hint = ""
                    try:
                        hint = str(node_data)
                    except Exception:
                        hint = ""

                    def _auto_response_for(hint_str: str) -> str:
                        hs = hint_str.lower()
                        # Model selection prompts expect comma-separated indices or 'all'
                        if "comma-separated" in hs or "select models" in hs or "enter" in hs and "comma" in hs:
                            return args.auto_select or "all"
                        # Confirmation prompts expect y/n
                        if "(y/n)" in hs or "run benchmarks" in hs or "confirm" in hs:
                            return "y" if args.yes or not sys.stdin.isatty() else "n"
                        # Default fallback
                        return args.auto_select or ("y" if args.yes or not sys.stdin.isatty() else input("> ").strip())

                    if not sys.stdin.isatty() or args.yes or args.auto_select:
                        user_input = _auto_response_for(hint)
                        print(f"> {user_input}  (auto)")
                    else:
                        user_input = input("> ").strip()

                    # Resume the agent with the user's response.
                    for _ in agent.stream(
                        Command(resume=user_input),
                        thread_config,
                        stream_mode="updates",
                    ):
                        pass  # Let the agent continue its loop
    except Exception as e:
        print(f"\n❌ Error during benchmark execution: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())