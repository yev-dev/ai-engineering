"""Enhanced debug logging for workflow performance analysis.

Provides timing, message size tracking, and loop detection utilities
to help diagnose slow agent execution and identify problematic patterns.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# ── Timing ────────────────────────────────────────────────────────────────


@contextmanager
def timer(agent: str, iteration: int, label: str = ""):
    """Context manager that logs elapsed time for a block of code.

    Usage:
        with timer("Orchestrator", 0, "LLM call"):
            response = llm.chat(...)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if elapsed > 5.0:
            logger.warning(
                "timer_slow agent=%s iteration=%d label=%s elapsed=%.2fs",
                agent, iteration, label, elapsed,
            )
        elif elapsed > 1.0:
            logger.info(
                "timer_ok agent=%s iteration=%d label=%s elapsed=%.2fs",
                agent, iteration, label, elapsed,
            )
        else:
            logger.debug(
                "timer_fast agent=%s iteration=%d label=%s elapsed=%.2fs",
                agent, iteration, label, elapsed,
            )


# ── Message Size Tracking ────────────────────────────────────────────────


def log_message_size(agent: str, iteration: int, messages: list[dict[str, str]]) -> None:
    """Log the total character count and token estimate of the message list.

    Token estimate is approximate (chars / 4). Helps detect growing
    conversation history that slows down LLM inference.
    """
    total_chars = sum(len(msg.get("content", "")) for msg in messages)
    total_msgs = len(messages)
    estimated_tokens = total_chars // 4

    logger.info(
        "message_size agent=%s iteration=%d messages=%d chars=%d estimated_tokens=%d",
        agent, iteration, total_msgs, total_chars, estimated_tokens,
    )

    # Warn if conversation history is getting large
    if estimated_tokens > 2000:
        logger.warning(
            "message_size_large agent=%s iteration=%d estimated_tokens=%d "
            "— consider summarising or truncating history",
            agent, iteration, estimated_tokens,
        )


# ── Loop Detection ────────────────────────────────────────────────────────


class LoopDetector:
    """Detects repeated tool calls or response patterns across iterations.

    Helps identify when the LLM is stuck in a loop (e.g. calling the same
    tool with the same arguments repeatedly).
    """

    def __init__(self, max_repeats: int = 3) -> None:
        self._history: list[dict[str, Any]] = []
        self._max_repeats = max_repeats

    def record(self, agent: str, iteration: int, tool_name: str, tool_args: dict[str, Any]) -> None:
        """Record a tool call for loop detection."""
        entry = {
            "agent": agent,
            "iteration": iteration,
            "tool": tool_name,
            "args": tool_args,
        }
        self._history.append(entry)

        # Check for repeated identical tool calls
        recent = [
            h for h in self._history
            if h["agent"] == agent and h["tool"] == tool_name
        ]
        if len(recent) >= self._max_repeats:
            # Check if arguments are identical
            last_args = [str(h.get("args", {})) for h in recent[-self._max_repeats:]]
            if len(set(last_args)) == 1:
                logger.warning(
                    "loop_detected agent=%s tool=%s repeats=%d iteration=%d "
                    "— same tool called with identical arguments %d times",
                    agent, tool_name, len(recent), iteration, self._max_repeats,
                )

    def record_response(self, agent: str, iteration: int, response: str) -> None:
        """Record an LLM response for pattern detection."""
        # Check if the response is identical to the previous one
        if self._history:
            last = self._history[-1]
            if last.get("agent") == agent and last.get("response") == response:
                logger.warning(
                    "response_loop_detected agent=%s iteration=%d "
                    "— identical response as previous iteration",
                    agent, iteration,
                )
        self._history.append({
            "agent": agent,
            "iteration": iteration,
            "response": response,
        })

    def reset(self) -> None:
        """Clear history for a new agent run."""
        self._history.clear()


# ── Workflow Progress Logger ──────────────────────────────────────────────


def log_workflow_progress(
    agent: str,
    iteration: int,
    status: str,
    detail: str | None = None,
) -> None:
    """Log a structured workflow progress message.

    Args:
        agent: The current agent name.
        iteration: The current iteration number.
        status: One of 'started', 'llm_call', 'tool_call', 'tool_result',
                'final_answer', 'delegating', 'pausing', 'error', 'completed'.
        detail: Optional additional context.
    """
    msg = f"progress agent={agent} iteration={iteration} status={status}"
    if detail:
        msg += f" detail={detail}"
    logger.info(msg)


# ── Model Performance Analysis ────────────────────────────────────────────


MODEL_BENCHMARKS: dict[str, dict[str, Any]] = {
    "ollama/deepseek-v2:16b": {
        "params": "16B",
        "speed": "slow",
        "quality": "high",
        "notes": (
            "16B parameter model. Requires significant RAM/VRAM (16GB+). "
            "On CPU-only systems, expect 10-30s per response. "
            "On GPU (MPS/CUDA), expect 3-8s per response. "
            "Good for complex reasoning but overkill for simple routing decisions."
        ),
        "recommendation": (
            "Use for MarketDataAgent and GapFillingAgent where reasoning matters. "
            "For Orchestrator and ReferenceDataAgent, consider a smaller model "
            "(e.g. ollama/llama3.2:1b or ollama/qwen2.5:1.5b) to reduce latency."
        ),
    },
    "ollama/llama3.2:1b": {
        "params": "1B",
        "speed": "fast",
        "quality": "low",
        "notes": (
            "1B parameter model. Very fast (1-3s per response on CPU). "
            "Limited reasoning capability — may produce malformed ReAct output. "
            "Suitable only for simple routing decisions."
        ),
        "recommendation": (
            "Use only for Orchestrator routing. Not suitable for data analysis."
        ),
    },
    "ollama/qwen2.5:1.5b": {
        "params": "1.5B",
        "speed": "fast",
        "quality": "medium",
        "notes": (
            "1.5B parameter model. Fast (2-4s per response on CPU). "
            "Better ReAct compliance than llama3.2:1b. "
            "Good balance for routing agents."
        ),
        "recommendation": (
            "Good choice for Orchestrator and ReferenceDataAgent. "
            "Can handle basic tool calling reliably."
        ),
    },
    "ollama/gemma4:e4b": {
        "params": "~4B (estimated)",
        "speed": "medium",
        "quality": "medium-high",
        "notes": (
            "~4B parameter model. Moderate speed (3-6s per response on CPU). "
            "Good ReAct compliance. Balanced for most agents."
        ),
        "recommendation": (
            "Good all-rounder for all agents. Recommended if you have 8GB+ RAM."
        ),
    },
    "ollama/llama3.2:3b": {
        "params": "3B",
        "speed": "medium",
        "quality": "medium",
        "notes": (
            "3B parameter model. Moderate speed (3-5s per response on CPU). "
            "Decent ReAct compliance. Good balance of speed and quality."
        ),
        "recommendation": (
            "Good all-rounder. Recommended as minimum for DataQualityAgent "
            "and GapFillingAgent."
        ),
    },
}


def get_model_advice(model_name: str) -> str:
    """Get performance advice for a specific model.

    Args:
        model_name: The full model name (e.g. 'ollama/deepseek-v2:16b').

    Returns:
        A formatted string with performance analysis and recommendations.
    """
    info = MODEL_BENCHMARKS.get(model_name)
    if info is None:
        # Try to match by partial name
        for key, value in MODEL_BENCHMARKS.items():
            if model_name in key or key in model_name:
                info = value
                break

    if info is None:
        return (
            f"Model '{model_name}' is not in the benchmark database. "
            f"General advice: larger models (>7B) are slow on CPU but produce "
            f"better ReAct output. Smaller models (<3B) are fast but may "
            f"produce malformed tool calls. For this workflow, a 3B-7B model "
            f"is recommended for the best balance of speed and reliability."
        )

    lines = [
        f"Model: {model_name}",
        f"Parameters: {info['params']}",
        f"Speed: {info['speed']}",
        f"Quality: {info['quality']}",
        f"Notes: {info['notes']}",
        f"Recommendation: {info['recommendation']}",
    ]
    return "\n".join(lines)