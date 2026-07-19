from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .llm_clients import LiteLLMClient


@dataclass
class OrchestrationDecision:
    thought: str
    tool: str
    args: dict[str, Any]


class LLMOrchestrator:
    """LLM-driven tool orchestration using JSON next-action decisions."""

    def __init__(self, llm_client: LiteLLMClient, framework: str, max_steps: int = 20) -> None:
        self.llm_client = llm_client
        self.framework = framework
        self.max_steps = max_steps

    def next_action(
        self,
        context: dict[str, Any],
        history: list[dict[str, Any]],
        tool_registry: dict[str, tuple[str, Any]],
    ) -> OrchestrationDecision:
        tools_desc = [
            {"name": tool_name, "description": description}
            for tool_name, (description, _func) in tool_registry.items()
        ]

        compact_context = {
            "request": context.get("request"),
            "selected_source": context.get("selected_source"),
            "gap_method": context.get("gap_method"),
            "has_source_results": context.get("source_results") is not None,
            "has_quality": context.get("quality") is not None,
            "has_gap_suggestions": context.get("gap_suggestions") is not None,
            "has_continuous": context.get("continuous") is not None,
            "has_artifact_paths": context.get("artifact_paths") is not None,
            "done": context.get("done", False),
            "user_selected_source": context.get("user_selected_source"),
            "user_selected_gap_method": context.get("user_selected_gap_method"),
        }

        system_prompt = (
            "You are an LLM workflow orchestrator for a ReACT data pipeline. "
            "Choose exactly one next tool call. Respond strictly as JSON with keys: "
            "thought (string), tool (string), args (object)."
        )
        user_prompt = json.dumps(
            {
                "framework": self.framework,
                "available_tools": tools_desc,
                "context": compact_context,
                "history_tail": history[-8:],
                "requirements": [
                    "Workflow must end by calling finish.",
                    "Before finish, ensure export_artifacts succeeded.",
                    "If user_selected_source exists, use it in select_source args.",
                    "If user_selected_gap_method exists, use it in select_gap_method args.",
                ],
            },
            default=str,
        )

        attempts = 3
        last_raw = None
        for attempt in range(1, attempts + 1):
            try:
                # Debug: surface prompt sizes so callers can confirm user prompt is passed
                try:
                    model_info = getattr(self.llm_client, "model", None)
                    api_base_info = getattr(self.llm_client, "api_base", None)
                    print(
                        f"[LLM DEBUG] model={model_info!r} api_base={api_base_info!r} system_prompt_len={len(system_prompt)} user_prompt_len={len(user_prompt)}"
                    )
                    # print a trimmed user prompt for quick inspection
                    print(f"[LLM DEBUG] user_prompt_preview={user_prompt[:400]!r}")
                except Exception:
                    pass

                raw = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
                last_raw = raw
                payload = json.loads(raw)
                tool = str(payload["tool"])
                thought = str(payload.get("thought", "Proceed with next tool."))
                args = payload.get("args", {})
                if not isinstance(args, dict):
                    args = {}
                if tool not in tool_registry:
                    # Invalid tool name — let the orchestrator decide to retry.
                    raise ValueError(f"Invalid tool name from LLM: {tool}")
                return OrchestrationDecision(thought=thought, tool=tool, args=args)
            except Exception as exc:  # noqa: BLE001
                # On parse/validation failure, retry with a stricter instruction
                # asking the model to return only valid JSON. After the final
                # attempt, surface a clear error so caller can handle it.
                if attempt >= attempts:
                    model_info = getattr(self.llm_client, "model", None)
                    api_base_info = getattr(self.llm_client, "api_base", None)
                    last_output = repr(last_raw) if last_raw is not None else repr(exc)
                    raise RuntimeError(
                        "LLM orchestrator failed to produce a valid JSON decision after multiple attempts. "
                        f"Last raw output: {last_output}. Error: {exc}. "
                        f"LLM model={model_info!r}, api_base={api_base_info!r}. "
                        "Check LLM connectivity, provider/model string (e.g. 'ollama/gemma4' or 'huggingface/starcoder'), "
                        "and api_base/api_key configuration."
                    )

                recovery_system = (
                    "You are an LLM workflow orchestrator for a ReACT data pipeline. "
                    "You MUST respond strictly as JSON with keys: thought (string), tool (string), args (object)."
                )
                recovery_user = json.dumps(
                    {
                        "note": "If you cannot decide, return a minimal safe action.",
                        "available_tools": [t["name"] for t in tools_desc],
                        "example": {"thought": "deciding", "tool": "fetch_market_data", "args": {}},
                    },
                    default=str,
                )
                # replace user_prompt with a recovery instruction for the next attempt
                user_prompt = recovery_user
                system_prompt = recovery_system
