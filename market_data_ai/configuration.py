"""Centralized YAML configuration for financial time series construction."""
from __future__ import annotations

import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")

#: Mapping of provider name to the environment variable holding its API key.
_PROVIDER_API_KEY_ENV = {
    "github": "GITHUB_TOKEN",
    "deepseek": "DEEPSEEK_API_KEY",
}

#: Ordered list of ``.env`` files to load. Later files override earlier ones.
ENV_FILES = (
    Path(__file__).resolve().with_name(".env"),
    Path(__file__).resolve().with_name(".env.local"),
)

#: Credential keys that must never be logged or echoed.
_SENSITIVE_KEYS = frozenset({"DEEPSEEK_API_KEY", "GITHUB_TOKEN"})


class ConfigurationError(ValueError):
    """Raised when application configuration is missing or invalid."""


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for one LLM provider."""

    model: str
    api_base: str | None = None
    api_key: str | None = None


@dataclass(frozen=True)
class LLMConfig:
    """Shared and provider-specific LLM configuration."""

    provider: str
    model: str | None
    temperature: float
    max_tokens: int
    timeout: float
    providers: Mapping[str, ProviderConfig]


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths used by the application."""

    app_dir: Path
    output_dir: Path
    database_dir: Path
    log_dir: Path
    data_dir: Path
    validation_rules: Path | None


@dataclass(frozen=True)
class RuntimeConfig:
    """Agent runtime behavior configuration."""

    agentic_framework: str
    debug_flow: bool


@dataclass(frozen=True)
class DashboardConfig:
    """Dashboard server configuration."""

    port: int
    provider_request_timeout: float


@dataclass(frozen=True)
class ApplicationConfig:
    """Complete application configuration loaded from YAML."""

    name: str
    llm: LLMConfig
    paths: PathsConfig
    runtime: RuntimeConfig
    dashboard: DashboardConfig
    data_sources: tuple[str, ...]


_runtime_overrides: dict[str, Any] = {}


def _redact(value: str) -> str:
    """Return a redacted representation of a sensitive value."""
    if not value:
        return "<unset>"
    if len(value) <= 4:
        return "****"
    return f"{value[:2]}...{value[-2:]}"


def load_env_files(*, override: bool = False) -> list[Path]:
    """Load environment variables from the package's ``.env`` files.

    Files are loaded in order defined by :data:`ENV_FILES`; later files take
    precedence over earlier ones. Existing process environment variables are
    preserved by default (``override=False``) so shell exports win.

    Args:
        override: If ``True``, values in ``.env`` files overwrite existing
            process environment variables.

    Returns:
        The list of ``.env`` files that were successfully loaded.
    """
    loaded: list[Path] = []
    for env_file in ENV_FILES:
        if not env_file.is_file():
            continue
        try:
            load_dotenv(env_file, override=override, verbose=False)
        except Exception as error:  # pragma: no cover - defensive
            logger.warning("Could not load env file %s: %s", env_file, error)
            continue
        loaded.append(env_file)
        logger.debug("Loaded env file %s", env_file)
    return loaded


def get_secret(name: str) -> str | None:
    """Return a credential value from the environment, or ``None`` if unset.

    Args:
        name: The environment variable name (e.g. ``DEEPSEEK_API_KEY``).

    Returns:
        The value of the environment variable, or ``None`` if not set.
    """
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def describe_secrets() -> dict[str, str]:
    """Return a redacted summary of configured credentials for diagnostics.

    Returns:
        A mapping of credential names to redacted values (never the full key).
    """
    summary: dict[str, str] = {}
    for key in sorted(_SENSITIVE_KEYS):
        summary[key] = _redact(get_secret(key) or "")
    return summary


def _require_mapping(value: Any, key: str) -> dict[str, Any]:
    """Return a mapping value or raise a descriptive configuration error."""
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration key '{key}' must be a mapping.")
    return value


def _merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge runtime overrides into loaded YAML values."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _optional_path(value: Any) -> Path | None:
    """Convert an optional YAML path value to an expanded ``Path``."""
    if value is None:
        return None
    return Path(str(value)).expanduser()


def _build_config(raw: dict[str, Any]) -> ApplicationConfig:
    """Validate a raw YAML mapping and create the typed configuration object."""
    application = _require_mapping(raw.get("application"), "application")
    llm = _require_mapping(raw.get("llm"), "llm")
    provider_values = _require_mapping(llm.get("providers"), "llm.providers")
    paths = _require_mapping(raw.get("paths"), "paths")
    runtime = _require_mapping(raw.get("runtime"), "runtime")
    dashboard = _require_mapping(raw.get("dashboard"), "dashboard")
    data_sources = raw.get("data_sources")
    if not isinstance(data_sources, list) or not all(
        isinstance(source, str) and source for source in data_sources
    ):
        raise ConfigurationError("Configuration key 'data_sources' must be a list of strings.")

    providers: dict[str, ProviderConfig] = {}
    for name, value in provider_values.items():
        provider = _require_mapping(value, f"llm.providers.{name}")
        model = provider.get("model")
        if not isinstance(model, str) or not model:
            raise ConfigurationError(f"Configuration key 'llm.providers.{name}.model' is required.")
        api_key = provider.get("api_key")
        if not api_key:
            # Fall back to the provider's credential from the environment
            # (sourced from .env files) so secrets are never committed to YAML.
            env_name = _PROVIDER_API_KEY_ENV.get(str(name).casefold())
            api_key = get_secret(env_name) if env_name else None
        providers[str(name)] = ProviderConfig(
            model=model,
            api_base=provider.get("api_base"),
            api_key=api_key,
        )

    provider_name = str(llm.get("provider", "")).casefold()
    if provider_name not in providers:
        raise ConfigurationError(
            f"Configured LLM provider '{provider_name}' is not defined in llm.providers."
        )

    return ApplicationConfig(
        name=str(application["name"]),
        llm=LLMConfig(
            provider=provider_name,
            model=llm.get("model"),
            temperature=float(llm["temperature"]),
            max_tokens=int(llm["max_tokens"]),
            timeout=float(llm["timeout"]),
            providers=providers,
        ),
        paths=PathsConfig(
            app_dir=Path(str(paths["app_dir"])).expanduser(),
            output_dir=Path(str(paths["output_dir"])).expanduser(),
            database_dir=Path(str(paths["database_dir"])).expanduser(),
            log_dir=Path(str(paths["log_dir"])).expanduser(),
            data_dir=Path(str(paths["data_dir"])).expanduser(),
            validation_rules=_optional_path(paths.get("validation_rules")),
        ),
        runtime=RuntimeConfig(
            agentic_framework=str(runtime["agentic_framework"]),
            debug_flow=bool(runtime["debug_flow"]),
        ),
        dashboard=DashboardConfig(
            port=int(dashboard["port"]),
            provider_request_timeout=float(dashboard["provider_request_timeout"]),
        ),
        data_sources=tuple(data_sources),
    )


@lru_cache(maxsize=1)
def get_config() -> ApplicationConfig:
    """Load, validate, and cache all application values from ``config.yaml``.

    Returns:
        The immutable application configuration.

    Raises:
        ConfigurationError: If the YAML file is absent or invalid.
    """
    if not CONFIG_PATH.is_file():
        raise ConfigurationError(f"Configuration file not found: {CONFIG_PATH}")
    # Load credentials from .env files (git-ignored) before resolving API keys.
    load_env_files()
    try:
        loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML in {CONFIG_PATH}: {error}") from error
    if not isinstance(loaded, dict):
        raise ConfigurationError("The root of config.yaml must be a mapping.")
    return _build_config(_merge(loaded, _runtime_overrides))


def set_runtime_overrides(**overrides: Any) -> ApplicationConfig:
    """Apply nested in-process overrides, primarily for CLI arguments.

    Args:
        **overrides: Top-level configuration sections to merge.

    Returns:
        The refreshed application configuration.
    """
    global _runtime_overrides
    _runtime_overrides = _merge(_runtime_overrides, overrides)
    get_config.cache_clear()
    return get_config()


def reset_config() -> None:
    """Clear runtime overrides and cached values (mainly for tests)."""
    global _runtime_overrides
    _runtime_overrides = {}
    get_config.cache_clear()