#!/usr/bin/env python
"""Simple CLI to send a user prompt directly to the LLM via ModelRequestFactory.

Usage:
    python cli_llm_request.py
    python cli_llm_request.py "create time series for apple between 2023 and 2024"
"""
from __future__ import annotations

import logging
import sys

from models import LLMRequest, ModelRequestFactory
from logging_config import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()

    # Build the prompt: use the first CLI argument, or prompt interactively
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        try:
            user_input = input("Enter your request: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_input:
            print("No input provided.")
            return

    print(f"\n── Query ──────────────────────────────────────")
    print(f"  {user_input}")
    print(f"── Response ───────────────────────────────────")

    factory = ModelRequestFactory()
    request = LLMRequest(
        system_prompt="You are a helpful financial assistant.",
        messages=[{"role": "user", "content": user_input}],
    )

    try:
        result = factory.chat(request)
        print(f"  {result}")
    except Exception as exc:
        logger.exception("llm_request_failed")
        print(f"  Error: {exc}")

    print(f"───────────────────────────────────────────────")


if __name__ == "__main__":
    main()