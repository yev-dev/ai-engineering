# AutoGen Financial Agent (Ollama + yfinance)

This is a front-to-back agentic workflow using AutoGen with Ollama.

It covers:

- orchestration with planner/synthesizer stages
- tool/function calling with yfinance
- structured JSON outputs
- context window budgeting
- memory persistence
- token optimization (broad retrieval -> rerank -> compact context)
- hallucination mitigation via grounding diagnostics

## What it does

Given a ticker and question, the workflow:

1. uses an AutoGen planner agent to call `get_stock_snapshot`
2. gathers broad evidence (snapshot + headlines + OHLCV)
3. reranks and compacts evidence under token budget
4. asks AutoGen synthesizer for structured JSON answer
5. validates and saves memory

## Run

```bash
python main.py --ticker NVDA --question "What is the latest financial picture?" --model llama3.1
```

## Notebook demo

- Open `autogen_financial_agent.ipynb`
- Run cells top-to-bottom
- Adjust `TICKER`, `QUESTION`, and `MODEL_NAME` in the configuration cell

## Optimization notebook

- Open `autogen_optimization_strategies.ipynb`
- Covers prefix caching, semantic caching, LLMLingua compression, concise instructions,
  constrained outputs, task-based routing, and batch processing
- Reuses vector-store persistence under `ai_tutorials/rag/vector_db`

## Arguments

- `--ticker`: stock ticker
- `--question`: user prompt
- `--model`: Ollama model name
- `--base-url`: Ollama OpenAI-compatible base URL (default `http://localhost:11434/v1`)
- `--token-budget`: estimated context token budget
- `--max-snippets`: max evidence snippets kept for synthesis

## Notes

- Requires `pyautogen` and a running Ollama server.
- Memory is stored under `memory/` in this folder.
