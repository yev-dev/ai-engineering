from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yfinance as yf
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class StructuredFinancialAnswer(BaseModel):
    ticker: str = Field(description="Ticker symbol")
    question: str = Field(description="User question")
    short_answer: str = Field(description="Concise answer")
    key_points: list[str] = Field(description="Main grounded points")
    risks: list[str] = Field(description="Uncertainties or risk notes")
    evidence_used: list[str] = Field(description="Evidence snippets used")
    confidence: float = Field(description="0.0 to 1.0 confidence")


def normalize_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def estimate_tokens(text: str) -> int:
    # Lightweight proxy for budgeting; exact tokenizer is model-dependent.
    return max(1, int(len(text.split()) * 1.1))


def extract_latest_close(hist) -> float | None:
    if hist is None or hist.empty:
        return None
    close_col = hist.get("Close")
    if close_col is None:
        return None
    clean = close_col.dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1])


@tool("get_stock_snapshot")
def get_stock_snapshot(ticker: str) -> dict[str, Any]:
    """Fetch latest stock snapshot, recent OHLCV, and top headlines from yfinance."""
    t = yf.Ticker(ticker.upper())
    hist = t.history(period="5d", interval="1d")
    latest_close = extract_latest_close(hist)

    recent_rows: list[dict[str, Any]] = []
    if hist is not None and not hist.empty:
        for idx, row in hist.tail(5).iterrows():
            recent_rows.append(
                {
                    "date": str(idx.date()),
                    "open": float(row.get("Open", 0.0)),
                    "high": float(row.get("High", 0.0)),
                    "low": float(row.get("Low", 0.0)),
                    "close": float(row.get("Close", 0.0)),
                    "volume": int(row.get("Volume", 0)),
                }
            )

    info = {}
    try:
        # fast_info is usually lighter than full info.
        info_obj = getattr(t, "fast_info", None)
        if info_obj:
            info = {
                "currency": info_obj.get("currency"),
                "exchange": info_obj.get("exchange"),
                "market_cap": info_obj.get("market_cap"),
                "day_high": info_obj.get("day_high"),
                "day_low": info_obj.get("day_low"),
            }
    except Exception as exc:  # pragma: no cover - network/provider variability
        info = {"error": f"fast_info unavailable: {exc}"}

    headlines: list[str] = []
    try:
        news_items = getattr(t, "news", []) or []
        for item in news_items[:5]:
            title = item.get("title")
            if title:
                headlines.append(str(title))
    except Exception as exc:  # pragma: no cover
        headlines.append(f"news unavailable: {exc}")

    return {
        "ticker": ticker.upper(),
        "latest_close": latest_close,
        "snapshot_time_utc": datetime.now(timezone.utc).isoformat(),
        "info": info,
        "recent_ohlcv": recent_rows,
        "headlines": headlines,
    }


def build_evidence_snippets(snapshot: dict[str, Any]) -> list[str]:
    snippets: list[str] = []

    snippets.append(
        (
            f"Ticker {snapshot.get('ticker')} latest_close={snapshot.get('latest_close')} "
            f"snapshot_time_utc={snapshot.get('snapshot_time_utc')}"
        )
    )

    info = snapshot.get("info", {}) or {}
    snippets.append(
        (
            f"Info currency={info.get('currency')} exchange={info.get('exchange')} "
            f"market_cap={info.get('market_cap')} day_high={info.get('day_high')} "
            f"day_low={info.get('day_low')}"
        )
    )

    for row in snapshot.get("recent_ohlcv", [])[:5]:
        snippets.append(
            (
                f"OHLCV date={row.get('date')} open={row.get('open')} high={row.get('high')} "
                f"low={row.get('low')} close={row.get('close')} volume={row.get('volume')}"
            )
        )

    for title in snapshot.get("headlines", [])[:5]:
        snippets.append(f"Headline: {title}")

    return snippets


def rerank_snippets(question: str, snippets: list[str]) -> list[tuple[float, str]]:
    q_terms = normalize_tokens(question)
    ranked: list[tuple[float, str]] = []

    for snippet in snippets:
        s_terms = normalize_tokens(snippet)
        overlap = len(q_terms & s_terms)
        coverage = overlap / max(1, len(q_terms))
        score = overlap + coverage
        ranked.append((score, snippet))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


def select_minimal_context(
    ranked_snippets: list[tuple[float, str]],
    token_budget: int,
    max_snippets: int,
) -> tuple[list[str], int]:
    selected: list[str] = []
    used_tokens = 0

    for _, snippet in ranked_snippets:
        snippet_tokens = estimate_tokens(snippet)
        if selected and used_tokens + snippet_tokens > token_budget:
            continue

        selected.append(snippet)
        used_tokens += snippet_tokens

        if len(selected) >= max_snippets or used_tokens >= token_budget:
            break

    return selected, used_tokens


def memory_path(base_dir: Path, ticker: str) -> Path:
    mem_dir = base_dir / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    return mem_dir / f"{ticker.upper()}.json"


def load_memory(base_dir: Path, ticker: str, limit: int = 3) -> list[dict[str, Any]]:
    path = memory_path(base_dir, ticker)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data[:limit]
        return []
    except Exception:
        return []


def save_memory(base_dir: Path, ticker: str, entry: dict[str, Any], max_entries: int = 20) -> None:
    path = memory_path(base_dir, ticker)
    entries = load_memory(base_dir, ticker, limit=max_entries)
    entries.insert(0, entry)
    path.write_text(json.dumps(entries[:max_entries], indent=2, ensure_ascii=True), encoding="utf-8")


def grounding_check(answer_text: str, evidence: list[str]) -> dict[str, Any]:
    ans_terms = normalize_tokens(answer_text)
    evidence_terms: set[str] = set()
    for item in evidence:
        evidence_terms |= normalize_tokens(item)

    unsupported = sorted(term for term in ans_terms if term not in evidence_terms)
    supported_ratio = 1.0 - (len(unsupported) / max(1, len(ans_terms)))

    return {
        "supported_ratio": round(supported_ratio, 3),
        "unsupported_terms_sample": unsupported[:15],
        "status": "pass" if supported_ratio >= 0.72 else "review",
    }


def run_workflow(
    ticker: str,
    question: str,
    model_name: str,
    base_url: str,
    token_budget: int,
    max_snippets: int,
) -> dict[str, Any]:
    base_dir = Path(__file__).resolve().parent
    prior_memory = load_memory(base_dir, ticker)

    llm = ChatOllama(model=model_name, base_url=base_url, temperature=0)
    tool_llm = llm.bind_tools([get_stock_snapshot])

    tool_message = tool_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a planner. For stock questions, call get_stock_snapshot with the provided ticker. "
                    "Prefer tool calls over guessing."
                )
            ),
            HumanMessage(content=f"Ticker: {ticker}\nQuestion: {question}"),
        ]
    )

    tool_calls = tool_message.tool_calls or []

    snapshots: list[dict[str, Any]] = []
    if tool_calls:
        for call in tool_calls:
            if call.get("name") == "get_stock_snapshot":
                args = call.get("args", {}) or {}
                tool_ticker = str(args.get("ticker", ticker))
                snapshots.append(get_stock_snapshot.invoke({"ticker": tool_ticker}))

    if not snapshots:
        # Fallback keeps flow deterministic if the model skipped function calls.
        snapshots.append(get_stock_snapshot.invoke({"ticker": ticker}))

    broad_snippets: list[str] = []
    for snapshot in snapshots:
        broad_snippets.extend(build_evidence_snippets(snapshot))

    ranked = rerank_snippets(question, broad_snippets)
    selected_snippets, used_tokens = select_minimal_context(
        ranked_snippets=ranked,
        token_budget=token_budget,
        max_snippets=max_snippets,
    )

    parser = PydanticOutputParser(pydantic_object=StructuredFinancialAnswer)
    format_instructions = parser.get_format_instructions()

    memory_blob = json.dumps(prior_memory, ensure_ascii=True, indent=2)
    context_blob = "\n".join(f"- {item}" for item in selected_snippets)

    synthesis_prompt = (
        "You are a grounded financial analysis assistant.\n"
        "Use only the provided evidence context and memory.\n"
        "If evidence is insufficient, state uncertainty explicitly.\n"
        f"{format_instructions}\n\n"
        f"Ticker: {ticker}\n"
        f"Question: {question}\n\n"
        f"Memory (most recent first):\n{memory_blob}\n\n"
        f"Evidence context:\n{context_blob}"
    )

    raw_response = llm.invoke(
        [
            SystemMessage(content="Return only the requested structured JSON."),
            HumanMessage(content=synthesis_prompt),
        ]
    )

    parsed = parser.parse(raw_response.content)
    grounding = grounding_check(parsed.short_answer + " " + " ".join(parsed.key_points), selected_snippets)

    save_memory(
        base_dir,
        ticker,
        {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "short_answer": parsed.short_answer,
            "key_points": parsed.key_points,
            "grounding": grounding,
        },
    )

    return {
        "result": parsed.model_dump(),
        "diagnostics": {
            "model": model_name,
            "ticker": ticker.upper(),
            "tool_calls": len(tool_calls),
            "broad_snippet_count": len(broad_snippets),
            "selected_snippet_count": len(selected_snippets),
            "estimated_context_tokens": used_tokens,
            "token_budget": token_budget,
            "grounding": grounding,
        },
        "selected_context": selected_snippets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangChain + Ollama financial agent")
    parser.add_argument("--ticker", required=True, help="Stock ticker, e.g. NVDA")
    parser.add_argument("--question", required=True, help="User prompt about the stock")
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--token-budget", type=int, default=700, help="Estimated context token budget")
    parser.add_argument("--max-snippets", type=int, default=8, help="Max evidence snippets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_workflow(
        ticker=args.ticker,
        question=args.question,
        model_name=args.model,
        base_url=args.base_url,
        token_budget=args.token_budget,
        max_snippets=args.max_snippets,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
