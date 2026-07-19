from __future__ import annotations

import os
from dataclasses import dataclass
try:
    import litellm

    # Ensure LiteLLM tolerates provider-specific params passed through our
    # completion calls (matches the behavior in the working `request.py`).
    litellm.drop_params = True
except Exception:
    # litellm not installed in this environment; completion() uses lazy import
    # so it's safe to continue without setting this flag.
    pass

from ..common.prompts.templates import react_system_prompt, react_user_prompt


@dataclass
class LiteLLMClient:
    model: str
    temperature: float = 0.2
    max_tokens: int = 120
    api_base: str | None = None
    api_key: str | None = None

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        # Lazy import keeps default pipeline working without litellm installed,
        # unless user explicitly enables LLM planning.
        from litellm import completion

        # Build kwargs depending on model/provider. Some providers (for example
        # HuggingFace inference via LiteLLM) expect a single `prompt` string and
        # return a text `choices[0].text` rather than chat-style messages.
        is_hf = isinstance(self.model, str) and self.model.startswith("huggingface/")

        if is_hf:
            # Concatenate system + user prompts into a single prompt for text
            # completion endpoints. `user_prompt` is a plain string here, not
            # a message dict, so use it directly.
            prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"
            kwargs = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "prompt": prompt,
            }
        else:
            kwargs = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key

        response = completion(**kwargs)

        # Support both chat-style responses and text-style responses.
        content = ""
        try:
            # Chat-style: choices[0].message.content
            content = response.choices[0].message.content
        except Exception:
            try:
                # Text-style: choices[0].text
                content = response.choices[0].text
            except Exception:
                # Last-resort: try to stringify the top-level response
                content = str(response)

        content = content.strip() if content else ""

        if not content:
            raise RuntimeError(
                f"Empty LLM response from model={self.model!r}, api_base={self.api_base!r}. "
                "Check model/provider string, connectivity, and API credentials."
            )

        return content


def build_llm_client(
    client_name: str,
    model: str | None,
    temperature: float,
    max_tokens: int,
    api_base: str | None,
    api_key: str | None,
) -> LiteLLMClient | None:
    if client_name == "none":
        return None

    if client_name == "copilot":
        # Copilot adapter uses OpenAI-compatible settings via LiteLLM.
        final_model = model or os.getenv("COPILOT_MODEL", "gpt-4o-mini")
        final_base = api_base or os.getenv("GITHUB_COPILOT_BASE_URL")
        final_key = api_key or os.getenv("GITHUB_COPILOT_API_KEY")
        return LiteLLMClient(
            model=final_model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=final_base,
            api_key=final_key,
        )

    if client_name == "github":
        # GitHub Models: use direct model strings (no provider prefix).
        final_model = model or os.getenv("GITHUB_MODEL", "openai/gpt-4o")
        final_base = api_base or os.getenv("GITHUB_BASE_URL", "https://models.github.ai/inference")
        final_key = api_key or os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_API_KEY")
        return LiteLLMClient(
            model=final_model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=final_base,
            api_key=final_key,
        )

    if client_name == "ollama":
        # Litellm expects a provider prefix in the model string (e.g.
        # "ollama/gemma4"). If the user passed a bare model name like
        # "gemma4", prefix it so LiteLLM can infer the provider.
        if model:
            final_model = model if "/" in model else f"ollama/{model}"
        else:
            final_model = os.getenv("OLLAMA_MODEL", "ollama/llama3.1")
        final_base = api_base or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # Normalize Ollama endpoint so it doesn't include a trailing /v1 path
        def _normalize_ollama_endpoint(endpoint: str) -> str:
            endpoint = endpoint.rstrip("/")
            if endpoint.endswith("/v1"):
                return endpoint[: -len("/v1")]
            return endpoint
        final_base = _normalize_ollama_endpoint(final_base)
        final_key = api_key or os.getenv("OLLAMA_API_KEY")
        return LiteLLMClient(
            model=final_model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=final_base,
            api_key=final_key,
        )

    raise ValueError(f"Unsupported llm client: {client_name}")


class ReActThoughtPlanner:
    def __init__(self, llm_client: LiteLLMClient | None, framework_name: str) -> None:
        self.llm_client = llm_client
        self.framework_name = framework_name

    def thought(self, stage: str, fallback: str) -> str:
        if self.llm_client is None:
            return fallback
        try:
            system_prompt = react_system_prompt()
            user_prompt = react_user_prompt(
                framework=self.framework_name,
                stage=stage,
                agent_name="pipeline_agent",
                action_hint="prepare next deterministic tool call",
            )
            planned = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
            return planned or fallback
        except Exception:
            return fallback
