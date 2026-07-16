# Time Series Generation Framework Comparison (Autogen vs LangChain)

This project compares two ReACT-style orchestrations for continuous time-series construction when a user provides:

- ticker
- start date
- end date

Both implementations include human-in-the-loop checkpoints and the same business flow:

1. Human provides ticker/date range.
2. Market data agent pulls historical prices from multi-source connectors:
  - Yahoo live connector (`yfinance`) with fallback support
  - Bloomberg placeholder adapter
  - Reuters placeholder adapter
3. Data quality agent compares each source.
4. CLI shows a rich table summary and asks human to choose a source.
5. Gap-filling analysis agent recommends methods.
6. Human chooses method.
7. Time-series generation agent builds a continuous series.
8. Reporting agent prints summary and full ReACT trace.
9. Audit layer exports run artifacts (`csv`, `json`) with a run ID.

## Folder Structure

```text
time_series/
  common/
    agent_specs.py
    agent_wrappers.py
    models.py
    stubs.py
    connectors.py
    quality.py
    gap_fill.py
    reporting.py
    llm_clients.py
    audit.py
    prompts/
      templates.py
    tools/
      external_services.py
      pipeline_tools.py
  autogen_framework/
    cli.py
    processor.py
    registry.py
    react_engine.py
  langchain_framework/
    cli.py
    processor.py
    registry.py
    react_engine.py
  scripts/
    run_autogen.sh
    run_langchain.sh
  requirements.txt
```

## Install

```bash
cd /Users/yevyerm/Dev/Projects/ai/ai-engineering/time_series
python -m pip install -r requirements.txt
```

## Run (Interactive)

```bash
cd /Users/yevyerm/Dev/Projects/ai/ai-engineering/time_series
python -m autogen_framework.cli
python -m langchain_framework.cli
```

## Run (Non-Interactive)

```bash
cd /Users/yevyerm/Dev/Projects/ai/ai-engineering/time_series
python -m autogen_framework.cli --ticker AAPL --start 2024-01-01 --end 2024-06-30 --source bloomberg --gap-method linear --yahoo-mode live --llm-client none --export-dir ./artifacts --non-interactive
python -m langchain_framework.cli --ticker AAPL --start 2024-01-01 --end 2024-06-30 --source reuters --gap-method ffill_then_bfill --yahoo-mode stub --llm-client none --export-dir ./artifacts --non-interactive
```

## CLI Parameters (both frameworks)

- `--ticker`, `--start`, `--end`: request context
- `--source`: `bloomberg|reuters|yahoo`
- `--gap-method`: `linear|ffill_then_bfill|rolling_median`
- `--yahoo-mode`: `live|stub`
- `--llm-client`: `none|copilot|ollama`
- `--llm-model`: model name for LiteLLM
- `--llm-base-url`: custom provider base URL
- `--llm-api-key`: provider API key
- `--llm-temperature`, `--llm-max-tokens`: generation settings
- `--run-id`: optional explicit run ID
- `--export-dir`: root export folder for audit artifacts
- `--check-services`: run external endpoint health checks before processing
- `--non-interactive`: disable prompts and require full args

## Architecture Separation

- `cli.py`: input collection and argument handling only
- `processor.py`: orchestration pipeline (`cli -> processor`)
- `registry.py`: framework-specific agent wrapper registration
- `common/tools/*`: deterministic tools for connectivity and data services
- `common/agent_specs.py`: ReACT text definitions per agent
- `common/agent_wrappers.py`: Python wrappers that execute tools
- `common/prompts/templates.py`: prompt templates for thought/action wiring

This keeps agents independent from tool implementation and makes framework registration explicit and testable.

## LiteLLM Provider Notes

- `--llm-client copilot`: OpenAI-compatible endpoint via LiteLLM (set base URL and API key as needed).
- `--llm-client ollama`: local Ollama via LiteLLM (default base URL `http://localhost:11434`).

The planner only generates ReACT thoughts; all tool steps remain deterministic and auditable.

## Audit Artifacts

Each run creates a folder under `--export-dir/<run_id>/` with:

- `quality_summary.csv`
- `continuous_series.csv`
- `run_report.json` (request, decisions, quality metrics, ReACT trace, artifact paths)

## Notes on ReACT Planning

- This scaffold is ReACT-shaped end-to-end: each stage emits Thought/Action/Observation.
- Tool execution is deterministic and auditable; only Thought generation can be LLM-assisted via LiteLLM.

## Next Upgrade Ideas

- Plug in real connectors (Bloomberg/Reuters/Yahoo APIs).
- Add confidence scoring per source from quality + historical alignment.
- Add model-based gap filling (Kalman, state-space, ARIMA).
- Export final continuous series and report artifacts to disk.
