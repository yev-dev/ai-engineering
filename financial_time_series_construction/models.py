"""LiteLLM factory for Ollama and GitHub Copilot clients via factory pattern."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import litellm

logger = logging.getLogger(__name__)

LLM_MODEL = os.getenv("LLM_MODEL", "ollama/gemma4:e4b")


@dataclass
class LLMRequest:
    """Data class representing a request to an LLM."""
    model: str = field(default_factory=lambda: LLM_MODEL)
    temperature: float = 0.1
    max_tokens: int = 2048
    system_prompt: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    callbacks: list[Any] | None = None

    def kwargs(self) -> dict[str, Any]:
        """Build the kwargs dict for litellm.completion."""
        messages = (
            [{"role": "system", "content": self.system_prompt}]
            if self.system_prompt
            else []
        )
        messages.extend(self.messages)
        result: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.callbacks:
            result["callbacks"] = self.callbacks
        return result


class ModelRequestFactory:
    """Provider-neutral wrapper using LiteLLM to select Ollama or GitHub/OpenAI models."""

    def chat(self, request: LLMRequest) -> str:
        """Send a chat request and return the response content.

        Args:
            request: The LLM request configuration.

        Returns:
            The response text from the LLM.

        Raises:
            Exception: If the LLM call fails.
        """
        logger.info(
            "llm_request model=%s messages=%d callbacks=%d",
            request.model,
            len(request.messages),
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
    def from_environment(cls) -> ModelRequestFactory:
        """Create a factory configured from environment variables."""
        return cls()