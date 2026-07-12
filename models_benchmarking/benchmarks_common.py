#!/usr/bin/env python3
"""
Shared data structures and tools used by all benchmark workflow implementations.

This module contains the common building blocks:
  - BenchmarkResult dataclass
  - BenchmarkState mutable container
  - 5 LangChain @tool functions for each workflow phase
  - The Ollama API interaction logic

Each approach file (benchmarks_prompt_driven.py, benchmarks_tool_driven.py, etc.)
imports from this module and wires the tools together using its own strategy.
"""

import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import os

import numpy as np
import ollama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.types import interrupt
import logging

# Module logger
logger = logging.getLogger("models_benchmarking")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


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


# Global singleton that tools access via closure.
_state: BenchmarkState = BenchmarkState()

# Optional path to save structured JSON results automatically.
_results_json_path: Optional[str] = None


def set_results_json_path(path: str) -> None:
    """Configure an optional path to save structured benchmark JSON results."""
    global _results_json_path
    _results_json_path = path


# ==================== LangChain Tools ====================


@tool
def list_ollama_models_tool() -> str:
    """List all locally available Ollama models via the Ollama API.
    Call this first to discover which models are available for benchmarking."""
    logger.info("Scanning for locally available Ollama models...")
    try:
        logger.debug("Calling ollama.list()")
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

        logger.info(f"Found {len(model_list)} local model(s): {[m['name'] for m in model_list]}")

        lines = [f"\n📋 Found {len(model_list)} locally available model(s):"]
        for i, m in enumerate(model_list, 1):
            size_gb = m["size"] / (1024 ** 3) if m["size"] else 0
            lines.append(f"  {i:2d}. {m['name']:30s} ({size_gb:.1f} GB)")

        return "\n".join(lines)

    except Exception as e:
        logger.exception("Failed to list Ollama models")
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

    logger.info(f"Prompting user to select from {len(models)} model(s): {[m['name'] for m in models]}")
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
        logger.info("No valid models selected by user")
        return "⚠️ No valid models selected."

    _state.selected_models = selected
    logger.info(f"User selected models: {selected}")
    return f"✅ Selected {len(selected)} model(s): {', '.join(selected)}"


@tool
def confirm_selection_tool() -> str:
    """Ask the user to confirm the model selection before running benchmarks.
    Call this after models have been selected."""
    selected = _state.selected_models
    if not selected:
        return "⚠️ No models selected. Call ask_user_to_select_models_tool first."

    logger.info(f"Asking user to confirm running benchmarks for: {selected}")
    print(f"\n📝 Run benchmarks for {len(selected)} model(s): {', '.join(selected)}? (y/n)")

    confirmation = interrupt("confirmation")

    if isinstance(confirmation, str) and confirmation.strip().lower() in ("y", "yes"):
        logger.info("User confirmed benchmarks")
        return "✅ Proceeding with benchmarks."
    else:
        logger.info("User cancelled benchmarks")
        return "❌ Benchmark cancelled by user."


@tool
def run_benchmarks_tool() -> str:
    """Run inference benchmarks for all selected models.
    Call this after the user has confirmed. Benchmarks each model
    with warmup iterations followed by timed inference runs."""
    selected = _state.selected_models
    if not selected:
        logger.warning("run_benchmarks_tool invoked with no selected models")
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
        logger.info(f"Starting benchmark for model: {model_name}")
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
                logger.info(f"Warming up model {model_name} for {warmup} passes")
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
                    logger.info(f"{model_name} progress: {i + 1}/{iterations} iterations")
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
            logger.exception("Benchmark failed for model %s", model_name)
            print(err_msg)
            report_lines.append(err_msg)

    _state.results = results
    # Also expose a machine-readable JSON summary so callers (agents/tests)
    # can consume structured `BenchmarkResult` data programmatically.
    summary: Dict[str, Any] = {}
    for r in results:
        summary[r.model_name] = {
            "iterations": r.iterations,
            "mean_ms": round(r.mean_ms, 2),
            "median_ms": round(r.median_ms, 2),
            "p95_ms": round(r.p95_ms, 2),
            "p99_ms": round(r.p99_ms, 2),
            "throughput_tokens_sec": round(r.throughput_tokens_sec, 2) if r.throughput_tokens_sec else None,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
        }

    json_block = "\n__BENCHMARK_RESULTS_JSON__:\n" + json.dumps(summary, indent=2)
    # Save to disk if requested
    if _results_json_path:
        try:
            os.makedirs(os.path.dirname(_results_json_path), exist_ok=True)
            with open(_results_json_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info("Saved structured benchmark JSON to %s", _results_json_path)
        except Exception:
            logger.exception("Failed to save structured benchmark JSON to %s", _results_json_path)

    return "✅ Benchmarking complete.\n" + "\n".join(report_lines) + json_block


@tool
def get_benchmark_results_tool() -> Dict[str, Any]:
    """Return the structured benchmark results as a JSON-serializable dict.

    Useful for programmatic consumption by agents or tests instead of parsing
    the printed output.
    """
    if not _state.results:
        return {}

    summary: Dict[str, Any] = {}
    for r in _state.results:
        summary[r.model_name] = {
            "iterations": r.iterations,
            "mean_ms": round(r.mean_ms, 2),
            "median_ms": round(r.median_ms, 2),
            "p95_ms": round(r.p95_ms, 2),
            "p99_ms": round(r.p99_ms, 2),
            "throughput_tokens_sec": round(r.throughput_tokens_sec, 2) if r.throughput_tokens_sec else None,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
        }
    return summary


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

    logger.info("Saved benchmark results to %s", output_path)

    lines = [f"\n{'=' * 60}", "📊 FINAL COMPARISON REPORT", f"{'=' * 60}"]
    lines.append(f"\n  {'Model':30s} {'Mean (ms)':>10s} {'P95 (ms)':>10s} {'P99 (ms)':>10s} {'Tokens/s':>10s}")
    lines.append(f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    for r in results:
        tps = f"{r.throughput_tokens_sec:.1f}" if r.throughput_tokens_sec else "N/A"
        lines.append(f"  {r.model_name:30s} {r.mean_ms:>10.2f} {r.p95_ms:>10.2f} {r.p99_ms:>10.2f} {tps:>10s}")

    lines.append(f"\n💾 Results saved to {output_path}")
    return "\n".join(lines)


# ==================== Helpers ====================


def configure_state(
    iterations: int = 30,
    warmup_iterations: int = 3,
    max_tokens: int = 128,
) -> None:
    """Configure the shared benchmark state with the given parameters."""
    _state.iterations = iterations
    _state.warmup_iterations = warmup_iterations
    _state.max_tokens = max_tokens


def reset_state() -> None:
    """Reset the shared benchmark state to defaults. Useful for testing."""
    global _state
    _state = BenchmarkState()
