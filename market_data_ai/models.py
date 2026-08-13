"""LiteLLM factory for Ollama and GitHub Copilot clients via factory pattern."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import litellm

from .configuration import get_config, get_secret

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("ollama", "github", "deepseek")

#: Mapping of provider name to the environment variable holding its API key.
_PROVIDER_API_KEY_ENV = {
    "github": "GITHUB_TOKEN",
    "deepseek": "DEEPSEEK_API_KEY",
}


def _normalize_provider(provider: str | None) -> str:
    """Normalize provider value to a supported key."""
    value = (provider or "").strip().casefold() or "ollama"
    aliases = {
        "gh": "github",
        "github_copilot": "github",
        "copilot": "github",
        "deepseekai": "deepseek",
    }
    value = aliases.get(value, value)
    if value not in SUPPORTED_PROVIDERS:
        logger.warning("Unsupported LLM provider '%s'; falling back to 'ollama'.", provider)
        return "ollama"
    return value


def _resolve_model(provider: str, explicit_model: str | None) -> str:
    """Resolve final LiteLLM model identifier from provider and model inputs."""
    config = get_config()
    if explicit_model:
        model = explicit_model.strip()
    else:
        provider_config = config.llm.providers.get(provider)
        model = provider_config.model if provider_config else "qwen2.5:1.5b"

    if "/" in model:
        return model
    return f"{provider}/{model}"


def _provider_kwargs(provider: str) -> dict[str, Any]:
    """Build provider-specific LiteLLM kwargs from configuration."""
    config = get_config()
    provider_config = config.llm.providers.get(provider)
    kwargs: dict[str, Any] = {}

    if provider_config is None:
        return kwargs

    if provider == "ollama":
        if provider_config.api_base:
            kwargs["api_base"] = provider_config.api_base
    elif provider == "deepseek":
        api_key = provider_config.api_key or get_secret(_PROVIDER_API_KEY_ENV["deepseek"])
        if api_key:
            kwargs["api_key"] = api_key
        if provider_config.api_base:
            kwargs["api_base"] = provider_config.api_base
    elif provider == "github":
        # LiteLLM supports GitHub models; auth is expected via GITHUB_TOKEN or
        # equivalent configuration.
        api_key = provider_config.api_key or get_secret(_PROVIDER_API_KEY_ENV["github"])
        if api_key:
            kwargs["api_key"] = api_key

    return kwargs


@dataclass
class LLMRequest:
    """Data class representing a request to an LLM."""
    model: str = ""
    temperature: float = 0.1
    max_tokens: int = 512
    timeout: float = 300.0
    system_prompt: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    callbacks: list[Any] | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

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
            "timeout": self.timeout,
        }
        if self.callbacks:
            result["callbacks"] = self.callbacks
        if self.extra_kwargs:
            result.update(self.extra_kwargs)
        return result


class ModelRequestFactory:
    """Provider-neutral wrapper using LiteLLM to select Ollama or GitHub/OpenAI models."""

    def __init__(
        self,
        model: str,
        provider: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        provider_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = min(max_tokens, 512) if max_tokens else 512
        self.provider_kwargs = provider_kwargs or {}

    def chat(self, request: LLMRequest) -> str:
        """Send a chat request and return the response content.

        Args:
            request: The LLM request configuration.

        Returns:
            The response text from the LLM.

        Raises:
            Exception: If the LLM call fails.
        """
        effective_model = request.model or self.model
        effective_temperature = request.temperature if request.temperature is not None else self.temperature
        effective_max_tokens = request.max_tokens if request.max_tokens is not None else self.max_tokens
        merged_kwargs = {**self.provider_kwargs, **request.extra_kwargs}

        effective_request = LLMRequest(
            model=effective_model,
            temperature=effective_temperature,
            max_tokens=effective_max_tokens,
            timeout=request.timeout,
            system_prompt=request.system_prompt,
            messages=request.messages,
            callbacks=request.callbacks,
            extra_kwargs=merged_kwargs,
        )

        logger.info(
            "llm_request provider=%s model=%s messages=%d callbacks=%d",
            self.provider,
            effective_request.model,
            len(effective_request.messages),
            len(effective_request.callbacks or []),
        )
        try:
            response = litellm.completion(**effective_request.kwargs())
            content = response.choices[0].message.content or ""
            logger.info(
                "llm_response provider=%s model=%s characters=%d",
                self.provider,
                effective_request.model,
                len(content),
            )
            return content
        except Exception:
            logger.exception(
                "llm_request_failed provider=%s model=%s",
                self.provider,
                effective_request.model,
            )
            raise

    @staticmethod
    def describe_environment() -> dict[str, Any]:
        """Return resolved model configuration from config.yaml."""
        config = get_config()
        provider = _normalize_provider(config.llm.provider)
        explicit_model = config.llm.model
        model = _resolve_model(provider, explicit_model)
        return {
            "provider": provider,
            "model": model,
            "temperature": config.llm.temperature,
            "max_tokens": config.llm.max_tokens,
            "timeout": config.llm.timeout,
            "provider_kwargs": _provider_kwargs(provider),
            "supported_providers": list(SUPPORTED_PROVIDERS),
        }

    @classmethod
    def from_environment(cls) -> ModelRequestFactory:
        """Create a factory configured from config.yaml."""
        config = cls.describe_environment()
        logger.info(
            "llm_factory provider=%s model=%s temperature=%s max_tokens=%s",
            config["provider"],
            config["model"],
            config["temperature"],
            config["max_tokens"],
        )
        return cls(
            model=str(config["model"]),
            provider=str(config["provider"]),
            temperature=float(config["temperature"]),
            max_tokens=int(config["max_tokens"]),
            provider_kwargs=dict(config["provider_kwargs"]),
        )