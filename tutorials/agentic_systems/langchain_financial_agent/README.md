# LangChain Financial Agent (Ollama + yfinance)

This is a front-to-back agentic workflow built with LangChain.

It covers:

- orchestration loop
- tool/function calling
- structured outputs
- context window management
- memory
- token optimization (retrieve broadly, rerank, send minimal context)
- hallucination mitigation via grounding checks

## What it does

Given a ticker and a question, the workflow:

1. loads memory from prior runs for that ticker
2. calls yfinance tools through LangChain tool calling
3. retrieves broad evidence (snapshot + headlines + recent OHLCV)
4. reranks snippets by lexical relevance to the question
5. keeps only context that fits a token budget
6. asks the Ollama model for a structured JSON answer
7. runs a grounding check and stores a memory summary

## Run

```bash
python main.py --ticker NVDA --question "What is the latest picture for this stock?" --model llama3.1
```

## Notebook demo

- Open `langchain_financial_agent.ipynb`
- Run cells top-to-bottom
- Adjust `TICKER`, `QUESTION`, and `MODEL_NAME` in the configuration cell

## Optimization notebook

- Open `langchain_optimization_strategies.ipynb`
- Covers prefix caching, semantic caching, LLMLingua compression, concise instructions,
  constrained outputs, task-based routing, and batch processing
- Reuses vector-store persistence under `ai_tutorials/rag/vector_db`

## Arguments

- `--ticker`: stock ticker, e.g. `NVDA`
- `--question`: user question
- `--model`: Ollama model name
- `--base-url`: Ollama URL (default `http://localhost:11434`)
- `--token-budget`: max estimated context tokens
- `--max-snippets`: max selected evidence snippets

## Notes

- Requires an Ollama server running locally.
- Uses `yfinance` for latest market data and news.
- Memory is stored under `memory/` in this folder.
