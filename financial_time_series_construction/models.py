"""LiteLLM factory for Ollama and GitHub Copilot clients via factory pattern."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("ollama", "github", "deepseek")


def _load_default_env_files() -> None:
    """Load default model configuration from repository .env folder.

    Files are optional. Existing process env vars are not overwritten.
    """
    repo_root = Path(__file__).resolve().parents[1]
    package_root = Path(__file__).resolve().parent

    # Preserve compatibility with existing repository .env file.
    root_env_file = repo_root / ".env"
    if root_env_file.exists() and root_env_file.is_file():
        load_dotenv(root_env_file, override=False)

    # New default profile folder (module-scoped) for model/provider options.
    env_dir = package_root / ".env"
    defaults_file = env_dir / "llm.defaults.env"
    local_override_file = env_dir / "llm.local.env"

    if defaults_file.exists() and defaults_file.is_file():
        load_dotenv(defaults_file, override=False)
    if local_override_file.exists() and local_override_file.is_file():
        load_dotenv(local_override_file, override=False)


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
    if explicit_model:
        model = explicit_model.strip()
    else:
        per_provider_default = {
            "ollama": os.getenv("LLM_OLLAMA_MODEL", "deepseek-v2:16b"),
            "github": os.getenv("LLM_GITHUB_MODEL", "gpt-4.1"),
            "deepseek": os.getenv("LLM_DEEPSEEK_MODEL", "deepseek-chat"),
        }
        model = per_provider_default.get(provider, "deepseek-v2:16b")

    if "/" in model:
        return model
    return f"{provider}/{model}"


def _provider_kwargs(provider: str) -> dict[str, Any]:
    """Build provider-specific LiteLLM kwargs from environment."""
    kwargs: dict[str, Any] = {}

    if provider == "ollama":
        api_base = os.getenv("OLLAMA_API_BASE")
        if api_base:
            kwargs["api_base"] = api_base
    elif provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        api_base = os.getenv("DEEPSEEK_API_BASE")
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["api_base"] = api_base
    elif provider == "github":
        # LiteLLM supports GitHub models; auth is expected via GITHUB_TOKEN or
        # equivalent environment variables configured by the runtime.
        token = os.getenv("GITHUB_TOKEN")
        if token:
            kwargs["api_key"] = token

    return kwargs


@dataclass
class LLMRequest:
    """Data class representing a request to an LLM."""
    model: str = ""
    temperature: float = 0.1
    max_tokens: int = 2048
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
        self.max_tokens = max_tokens
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
        """Return resolved model environment configuration."""
        _load_default_env_files()
        provider = _normalize_provider(os.getenv("LLM_PROVIDER", "ollama"))
        explicit_model = os.getenv("LLM_MODEL")
        model = _resolve_model(provider, explicit_model)
        return {
            "provider": provider,
            "model": model,
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048")),
            "provider_kwargs": _provider_kwargs(provider),
            "supported_providers": list(SUPPORTED_PROVIDERS),
        }

    @classmethod
    def from_environment(cls) -> ModelRequestFactory:
        """Create a factory configured from environment variables."""
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