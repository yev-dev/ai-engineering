"""Small LiteLLM factory shared by the processor and future adapters."""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Any

import litellm

logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "ollama/gemma4:26b"))
    temperature: float = 0.1
    max_tokens: int = 2048
    system_prompt: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    tools: list[Any] | None = None
    callbacks: list[Any] | None = None

    def kwargs(self) -> dict[str, Any]:
        messages = ([{"role": "system", "content": self.system_prompt}] if self.system_prompt else [])
        messages.extend(self.messages)
        result: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.tools:
            result["tools"] = self.tools
        if self.callbacks:
            result["callbacks"] = self.callbacks
        return result


class ModelRequestFactory:
    """Provider-neutral wrapper; LiteLLM selects Ollama or GitHub/OpenAI models."""

    def chat(self, request: LLMRequest) -> str:
        logger.info(
            "llm_request model=%s messages=%d tools=%d callbacks=%d",
            request.model,
            len(request.messages),
            len(request.tools or []),
            len(request.callbacks or []),
        )
        try:
            response = litellm.completion(**request.kwargs())
            content = response.choices[0].message.content or ""
            logger.info("llm_response model=%s characters=%d", request.model, len(content))
            return content
        except Exception:
            logger.exception("llm_request_failed model=%s", request.model)
            raise

    @classmethod
    def from_environment(cls) -> "ModelRequestFactory":
        return cls()