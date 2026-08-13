"""Financial time series construction with multi-agent ReAct orchestration.

Public API
----------
The package root re-exports the core, lightweight public API so external
consumers can import directly from the package::

    from market_data_ai import (
        DataStore,
        get_datastore,
        init_datastore,
        get_config,
        get_agent,
    )

Modules that pull in heavier optional dependencies (``litellm``, ``autogen``,
Streamlit) are exposed lazily as submodule attributes so ``import
market_data_ai`` stays lightweight::

    import market_data_ai
    models = market_data_ai.models      # imports market_data_ai.models on demand
    runtime = market_data_ai.runtime
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

from . import (
    agents_definition,
    configuration,
    data_source_registry,
    database,
    prompt_library,
    prompts,
)
from .agents_definition import (
    Agent,
    CallbackEvent,
    CallbackEventType,
    get_agent,
)
from .configuration import (
    ApplicationConfig,
    ConfigurationError,
    get_config,
    get_secret,
    load_env_files,
    reset_config,
)
from .database import (
    DataStore,
    close_datastore,
    get_datastore,
    init_datastore,
    reset_datastore,
)
from .data_source_registry import (
    DataSourceFile,
    DataSourceRegistry,
)
from .prompt_library import (
    PromptTemplate,
    format_prompt_menu,
    get_prompts,
    resolve_prompt_selection,
)
from .prompts import agent_system_prompt, request_prompt

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Exposed submodules.
    "agents_definition",
    "configuration",
    "data_source_registry",
    "database",
    "prompt_library",
    "prompts",
    # Public symbols.
    "Agent",
    "ApplicationConfig",
    "CallbackEvent",
    "CallbackEventType",
    "ConfigurationError",
    "DataStore",
    "DataSourceFile",
    "DataSourceRegistry",
    "PromptTemplate",
    "agent_system_prompt",
    "close_datastore",
    "format_prompt_menu",
    "get_agent",
    "get_config",
    "get_datastore",
    "get_prompts",
    "get_secret",
    "init_datastore",
    "load_env_files",
    "request_prompt",
    "reset_config",
    "reset_datastore",
    "resolve_prompt_selection",
]

# Submodules that pull in heavy optional dependencies (``litellm``, ``autogen``,
# Streamlit) at import time.  They are reachable from the package root but only
# loaded when explicitly referenced, keeping ``import market_data_ai`` light.
_LAZY_MODULES: tuple[str, ...] = (
    "charts",
    "cli",
    "dashboard",
    "dashboard_app",
    "handler",
    "models",
    "processor",
    "runtime",
    "tool_models",
    "tools",
    "workflow_report",
)


def __getattr__(name: str) -> Any:
    """Lazily expose heavy public submodules as package attributes.

    Implements PEP 562 so ``market_data_ai.<module>`` works without importing
    the module eagerly at package load time.  The result is cached on first
    access to avoid repeated imports.
    """
    if name in _LAZY_MODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Tidy namespace for ``dir(market_data_ai)`` / IDE tab-completion."""
    return sorted(globals().keys())