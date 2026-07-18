"""
Model abstraction layer for the travel booking system.

Provides:
  - LLMRequest — a config object describing what model to use and how.
  - ModelRequestFactory — returns a callable chat client given an LLMRequest.

The factory wraps litellm.completion() so we get proper streaming, retries,
and callback support without raw httpx calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import litellm
from litellm import completion as litellm_completion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported client types & models
# ---------------------------------------------------------------------------

SUPPORTED_CLIENTS = {
    "ollama": "ollama",
    "openai": "openai",
    "anthropic": "anthropic",
    "groq": "groq",
    "together": "together",
    "azure": "azure",
}

# ---------------------------------------------------------------------------
# LLMRequest — configuration object
# ---------------------------------------------------------------------------

@dataclass
class LLMRequest:
    """Describes a request to an LLM model.

    Attributes:
        model:         Full model name, e.g. "ollama/gemma4:e4b" or "gpt-4o".
        client_type:   Short name of the provider ("ollama", "openai", …).
        temperature:   Sampling temperature.
        max_tokens:    Maximum tokens in the response.
        system_prompt: System-level instruction.
        messages:      Conversation history (list of {"role": …, "content": …}).
        callbacks:     LangChain callback handlers to attach.
    """
    model: str = "ollama/gemma4:e4b"
    client_type: str = "ollama"
    temperature: float = 0.1
    max_tokens: int = 2048
    system_prompt: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    callbacks: list[Any] | None = None

    def to_litellm_kwargs(self) -> dict[str, Any]:
        """Convert this request to ``litellm.completion`` keyword arguments."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.system_prompt:
            kwargs["messages"] = [
                {"role": "system", "content": self.system_prompt},
                *self.messages,
            ]
        else:
            kwargs["messages"] = self.messages

        if self.callbacks:
            kwargs["callbacks"] = self.callbacks

        return kwargs


# ---------------------------------------------------------------------------
# ModelRequestFactory
# ---------------------------------------------------------------------------

class ModelRequestFactory:
    """Factory that returns a callable chat client for a given LLMRequest.

    Usage::

        factory = ModelRequestFactory()
        client = factory.create(LLMRequest(model="ollama/gemma4:e4b"))
        response = client()           # uses the stored LLMRequest

    Or call directly::

        response = factory.chat(LLMRequest(...))
    """

    def __init__(self) -> None:
        self._default_request = LLMRequest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        request: LLMRequest | None = None,
    ) -> Callable[[], str]:
        """Return a zero-argument callable that issues a chat completion.

        Every invocation calls ``litellm.completion`` with the parameters
        from *request* (or the factory default).  The return value is the
        response content as a plain string.
        """
        req = request or self._default_request

        def _chat() -> str:
            try:
                kwargs = req.to_litellm_kwargs()
                logger.debug("LLM call: model=%s | messages=%d | temp=%s",
                             req.model, len(req.messages), req.temperature)
                resp = litellm_completion(**kwargs)
                content = resp.choices[0].message.content or ""
                return content
            except Exception as e:
                logger.error("litellm completion failed: %s", e)
                return f"Error: LLM call failed - {e}"

        return _chat

    def chat(self, request: LLMRequest) -> str:
        """Convenience: build a client and call it immediately."""
        return self.create(request)()

    def set_defaults(self, **kwargs: Any) -> None:
        """Override default LLMRequest fields on the factory."""
        for key, val in kwargs.items():
            if hasattr(self._default_request, key):
                setattr(self._default_request, key, val)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_environment(cls) -> ModelRequestFactory:
        """Create a factory configured from environment variables."""
        import os
        factory = cls()
        model = (os.getenv("LLM_MODEL") or "ollama/gemma4:e4b").strip()
        factory.set_defaults(model=model)
        return factory
