"""LangChain CLI entrypoint: collect inputs and delegate to processor only."""

from __future__ import annotations

import argparse
import re

from .processor import LangChainProcessor, LangChainProcessorConfig


def _prompt(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val or (default or "")


def _collect_ticker(raw: str | None = None) -> str:
    ticker_re = re.compile(r"^[A-Za-z0-9.]{1,10}$")
    while True:
        raw_val = (
            _prompt("Ticker", raw) if raw else _prompt("Ticker", None)
        ).strip().upper()
        if ticker_re.match(raw_val):
            return raw_val
        print(
            f"Invalid ticker: {raw_val!r}. "
            "Use 1-10 alphanumeric characters (e.g. AAPL, BRK.B)."
        )
        raw = None


def _collect_date(prompt: str, raw: str | None = None) -> str:
    while True:
        raw_val = _prompt(prompt, raw)
        if raw_val:
            return raw_val
        print("Date is required.")
        raw = None


def run() -> None:
    parser = argparse.ArgumentParser(description="LangChain LLM-orchestrated time-series CLI")
    parser.add_argument("--ticker", help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument("--start", help="Start date")
    parser.add_argument("--end", help="End date")
    parser.add_argument("--source", choices=["bloomberg", "reuters", "yahoo"])
    parser.add_argument("--gap-method", choices=["linear", "ffill_then_bfill", "rolling_median"])
    parser.add_argument(
        "--yahoo-mode",
        choices=["live", "stub"],
        default="live",
        help="Yahoo data mode (live = real API, stub = synthetic)",
    )
    parser.add_argument(
        "--llm-client",
        choices=["none", "copilot", "ollama"],
        default="none",
        help="LLM client for orchestration",
    )
    parser.add_argument("--llm-model", help="LLM model name (LiteLLM format)")
    parser.add_argument("--llm-base-url", help="Custom LLM base URL")
    parser.add_argument("--llm-api-key", help="LLM API key")
    parser.add_argument(
        "--llm-temperature", type=float, default=0.2, help="LLM temperature"
    )
    parser.add_argument(
        "--llm-max-tokens", type=int, default=120, help="LLM max tokens per thought"
    )
    parser.add_argument(
        "--export-dir",
        default="./artifacts",
        help="Root directory for audit artifact exports",
    )
    parser.add_argument("--run-id", help="Optional explicit run ID")
    parser.add_argument(
        "--check-services",
        action="store_true",
        help="Run external service health checks before processing",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Require all required args and skip prompts",
    )
    args = parser.parse_args()

    if args.non_interactive:
        if not args.ticker or not args.start or not args.end:
            raise ValueError("--ticker, --start, and --end are required in --non-interactive mode.")
        ticker = args.ticker.upper()
        start = args.start
        end = args.end
    else:
        ticker = _collect_ticker(args.ticker)
        start = _collect_date("Start date", args.start)
        end = _collect_date("End date", args.end)

    processor = LangChainProcessor(
        config=LangChainProcessorConfig(
            ticker=ticker,
            start=start,
            end=end,
            source=args.source,
            gap_method=args.gap_method,
            yahoo_mode=args.yahoo_mode,
            llm_client=args.llm_client,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key,
            llm_temperature=args.llm_temperature,
            llm_max_tokens=args.llm_max_tokens,
            export_dir=args.export_dir,
            run_id=args.run_id,
            check_services=args.check_services,
        )
    )
    processor.execute()


if __name__ == "__main__":
    run()