# Time Series Construction (LangGraph Edition)

An AI-assisted workflow for constructing continuous financial time series from multiple data sources using LangGraph for state management and human-in-the-loop interactions.

## Architecture

The system uses a **LangGraph StateGraph** to orchestrate a multi-agent ReAct workflow.

```
┌─────────────────┐
│   CLI Input     │
└────────┬────────┘
         ▼
┌─────────────────┐     ┌─────────────────┐
│ TimeSeriesCLI   │────►│ StateGraph      │
└────────┬────────┘     │ (graph.py)      │
         │              └────────┬────────┘
         │                         │
         │              ┌─────────┴─────────┐
         │              │                     │
         ▼              ▼                     ▼
  ┌─────────────┐ ┌───────────┐       ┌──────────────┐
  │ Orchestrator│ │ Reference │       │   ToolNode   │
  │   (LLM)     │ │ Data Agent│       │ (tools.py)   │
  └─────────────┘ └───────────┘       └──────────────┘
         │              │                     │
         └──────────────┼───────────────────────┘
                        ▼
              ┌─────────────────┐
              │   Human Pause   │
              │ (.interrupt)    │
              └─────────────────┘
```

The graph consists of:
- **`call_llm` node**: Generic LLM node that loads the agent system prompt and tools from the registry.
- **`tools` node**: `ToolNode` that invokes deterministic tools (instrument lookup, data loading, quality checks, gap filling).
- **Human-in-the-loop**: `.interrupt()` pauses execution for user decisions at source/method selection points.
- **Checkpointing**: `MemorySaver` persists state for resume capability.

## Quick Start

```bash
cd /Users/yevyerm/Dev/Projects/ai/ai-engineering
conda activate ai_engineering  # or your Python environment with langgraph installed
python -m time_series_construction_langgraph.cli
```

The default model is read from `LLM_MODEL` and falls back to `ollama/gemma4:26b`. Generated artifacts are written to `~/time_series_construction` unless `TIME_SERIES_OUTPUT_DIR` is set.

### Logging

```bash
LOG_LEVEL=INFO python -m time_series_construction_langgraph.cli
LOG_LEVEL=DEBUG LOG_FILE=~/time_series_construction/application.log python -m time_series_construction_langgraph.cli
```

## Application

The CLI accepts natural-language requests for financial time series:

```
> Build AAPL from 2023 to 2024
> Create a time series for APL between January and December 2023
> Get Apple Inc. stock prices for last year
```

The workflow proceeds:
1. **Orchestrator** parses the request and delegates to **ReferenceDataAgent**
2. **ReferenceDataAgent** resolves APL → AAPL via fuzzy matching
3. **MarketDataAgent** loads prices from all sources (yahoo, bloomberg, reuters)
4. **DataQualityAgent** compares completeness, emits quality reports
5. **ReportingAgent** asks user to select a data source
6. **GapFillingAgent** recommends methods, applies user-selected method
7. **TimeSeriesConstructionAgent** builds final CSV + chart
8. **ReportingAgent** presents final summary

## Commands

| Environment Variable | Purpose | Default |
|---------------------|---------|---------|
| `LLM_MODEL` | LLM model identifier | `ollama/gemma4:26b` |
| `LLM_ENDPOINT` | Ollama API URL | `http://localhost:11434` |
| `TIME_SERIES_OUTPUT_DIR` | Artifact output directory | `~/time_series_construction` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `LOG_FILE` | Optional log file path | (none) |

## Test Cases

### Running Tests

```bash
conda activate ai_engineering
python -m pytest time_series_construction_langgraph/tests/test_workflow_int.py -v
```

### Test Classes

| Class | Purpose |
|-------|---------|
| `TestInstrumentResolution` | Fuzzy matching: APL → AAPL |
| `TestHistoricalData` | Price loading from all 3 sources |
| `TestDataQuality` | Missing-value detection, gap-filling methods |
| `TestArtifacts` | CSV/PNG artifact generation |
| `TestLangGraphWorkflow` | Graph instantiation and basic event flow |

The `mock_data_dir` fixture creates:
- `instruments.csv` with AAPL, GOOGL, MSFT
- Source CSVs with 2023-2024 prices (yahoo has injected NaN gaps)

## Key Differences from the Vanilla ReAct (`time_series_construction`)

| Aspect | Vanilla ReAct | LangGraph Edition |
|--------|--------------|-------------------|
| Tool invocation | Regex parse `Action:` / `Action Input:` | Native function-calling API |
| Pause/resume | Custom `paused_state` dict | Built-in `.interrupt()` / `.resume()` |
| State persistence | In-memory only | `MemorySaver` checkpoint (pluggable) |
| Cycle detection | Manual `visited` set | Built-in depth tracking |
| Delegation | Recursive `_run_agent` | Graph edges |

## File Structure

```
time_series_construction_langgraph/
├── __init__.py
├── agents_definition.py     # Agent contracts & callback types
├── tools.py                 # Deterministic domain tools
├── models.py                # LLM request wrapper + factory
├── prompts.py               # System prompt builders
├── graph.py                 # LangGraph StateGraph definition
├── cli.py                   # Interactive CLI
├── logging_config.py        # Logging setup
├── ARCHITECTURE.md
├── README.md
├── data/
│   ├── instruments.csv
│   ├── bloomberg_stock_data.csv
│   └── reuters_stock_data.csv
└── tests/
    ├── __init__.py
    └── test_workflow_int.py