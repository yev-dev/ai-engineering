# Financial Time Series Construction

AI-assisted command-line workflow for constructing continuous financial time series from multiple local data sources (Yahoo, Bloomberg, Reuters) with a multi-agent architecture and human-in-the-loop checkpoints.

## Architecture Decision

This implementation uses a hierarchical multi-agent design:

1. One supervisor `Orchestrator` agent routes requests.
2. Specialist agents execute domain steps.
3. Human-in-the-loop checkpoints pause execution at decision points.

Why this decision:

1. The workflow is multi-stage and stateful.
2. User choices (source selection and gap-filling method) must be explicit.
3. Agent boundaries keep tool usage reliable and testable while orchestration remains adaptive.

## Implemented Components

- `agents_definition.py`: agent registry, callback event types, guardrails
- `processor.py`: ReAct loop, delegation, tool execution, pause/resume
- `handler.py`: callback queue, HITL state, trace capture, error events
- `tools.py`: domain tools over CSV fixtures in `data/`
- `models.py`: LiteLLM model client factory
- `prompts.py`: reusable ReAct and user clarification prompts
- `cli.py`: interactive command line interface
- `tests/test_workflow_int.py`: integration suite and workflow proofs

## Agent Registry and Responsibilities

Registered agents:

1. `Orchestrator`: validates and routes requests.
2. `ReferenceDataAgent`: resolves ticker/security name from `data/instruments.csv`.
3. `MarketDataAgent`: loads source data from local market snapshots.
4. `DataQualityAgent`: computes missingness and quality metrics.
5. `ReportingAgent`: presents source comparison and requests user selection.
6. `GapFillingAgent`: recommends and applies interpolation strategies.
7. `TimeSeriesConstructionAgent`: persists final output artifacts.

## End-to-End Workflow

The default flow below is a common path, not a hard-coded graph. Agents can adapt routing
based on available context and user intent while preserving guardrails and traceability.

1. User submits request in natural language.
2. `Orchestrator` delegates to `ReferenceDataAgent`.
3. `ReferenceDataAgent` resolves instrument and delegates to `MarketDataAgent`.
4. `MarketDataAgent` loads market data for configured sources.
5. `DataQualityAgent` computes quality report per source.
6. `ReportingAgent` pauses and asks user to select a source.
7. `GapFillingAgent` recommends methods and pauses for method selection.
8. Selected method is applied and delegated to `TimeSeriesConstructionAgent`.
9. Final CSV and charts are generated under `~/time_series_construction`.
10. ReAct trace and event logs are persisted per run.

## Human-in-the-Loop Interaction

The processor emits `AWAITING_USER_INPUT` callback events for:

1. Source selection after quality comparison.
2. Gap-filling method selection.

The CLI resumes processing with user responses through `process_user_response()`.

## Data and Artifacts

Input data folder:

- `data/instruments.csv`
- `data/yahoo_stock_data.csv`
- `data/bloomberg_stock_data.csv`
- `data/reuters_stock_data.csv`

Output folder (default):

- `~/time_series_construction/<run_id>/`

Typical outputs:

- final time series CSV
- quality report CSV
- time series PNG plots
- `react_trace.txt`
- `events.json`

## Running the CLI

From workspace root:

```bash
python -m financial_time_series_construction.cli
```

One-shot request mode:

```bash
python -m financial_time_series_construction.cli \
  --request "Build AAPL from 2023-01-01 to 2023-12-31 and fill missing values"
```

Optional environment variables:

- `LLM_MODEL` default `ollama/llama3.2`
- `TIME_SERIES_OUTPUT_DIR` default `~/time_series_construction`
- `LOG_LEVEL` standard Python logging level

## Test Coverage

Run integration suite:

```bash
pytest financial_time_series_construction/tests/test_workflow_int.py -v
```

The test suite validates:

1. Instrument resolution and fuzzy matching.
2. Multi-source historical data loading.
3. Data quality and gap-filling methods.
4. Artifact generation (CSV/PNG).
5. Full workflow with delegation and pause/resume checkpoints.
6. Cancellation and non-financial request handling.

## Notes

- The implementation is designed to be extensible by adding agents and tools to registries.
- Current data connectors are local CSV-backed fixtures for reproducible development and testing.
