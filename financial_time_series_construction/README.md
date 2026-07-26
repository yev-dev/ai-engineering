# Financial Time Series Construction

AI-assisted command-line workflow for constructing continuous financial time series from multiple local data sources (Yahoo, Bloomberg, Reuters) with a multi-agent architecture and human-in-the-loop checkpoints.

## Revised Design (2026-07)

The workflow remains LLM-driven for reasoning and tool selection, but it now enforces two mandatory human checkpoints when the model responds with narrative final answers instead of explicit tool calls:

1. **Reporting checkpoint (source selection)** is force-paused if no explicit source choice is detected.
2. **Gap-filling checkpoint (method selection)** is force-paused if no explicit method choice is detected.

This preserves flexibility in agent reasoning while preventing silent early termination.

## Framework Portability Design

The CLI is now decoupled from a specific agentic engine using an **Adapter + Factory** runtime pattern:

1. `AgenticRuntime` defines a framework-agnostic interface (`process_user_request`, `process_user_response`, pause state, trace access).
2. `AutogenProcessorRuntime` adapts the current processor/handler implementation to that interface.
3. `build_runtime(framework=...)` selects runtime implementation.

Current support:

- `autogen` (wired)
- `crawl` (reserved; adapter scaffolded but not yet implemented)

This allows adding future runtimes (Crawl, LangGraph, custom orchestrators) without rewriting CLI interaction/artifact/reporting logic.

## Architecture Decision

This implementation uses a hierarchical multi-agent design:

1. One supervisor `Orchestrator` agent routes requests.
2. Specialist agents execute domain steps.
3. Human-in-the-loop checkpoints pause execution at decision points.

Why this decision:

1. The workflow is multi-stage and stateful.
2. User choices (source selection and gap-filling method) must be explicit.
3. Agent boundaries keep tool usage reliable and testable while orchestration remains adaptive.

## Agent Structure and Component Map

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI (cli.py)                                │
│  Entry point: user input → processor → event display → artifacts   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Processor (processor.py)                         │
│  ReAct loop · delegation · tool execution · pause/resume           │
│  Deterministic bypasses:                                            │
│  • Orchestrator bypass (instrument + dates in request)              │
│  • ReferenceDataAgent auto-continuation → MarketDataAgent           │
│  • MarketDataAgent source-selection bypass                          │
│  • ReferenceDataAgent placeholder bypass                            │
│  • Forced HITL pause at ReportingAgent when source not explicit      │
│  • Forced HITL pause at GapFillingAgent when method not explicit     │
└──────┬──────────┬──────────┬──────────┬──────────┬──────────┬───────┘
       │          │          │          │          │          │
       ▼          ▼          ▼          ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│Orchestra-│ │Reference │ │ Market   │ │ Data     │ │Reporting │ │   Gap    │
│  tor     │ │  Data    │ │  Data    │ │ Quality  │ │  Agent   │ │ Filling  │
│          │ │  Agent   │ │  Agent   │ │  Agent   │ │          │ │  Agent   │
├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤
│Validates │ │Resolves  │ │Loads     │ │Computes  │ │Presents  │ │Recommends│
│& routes  │ │ticker/   │ │historical│ │quality   │ │quality   │ │gap-filling│
│requests  │ │security  │ │prices    │ │metrics   │ │report &  │ │methods & │
│          │ │name from │ │from all  │ │per source│ │asks user │ │applies   │
│          │ │catalog   │ │sources   │ │          │ │selection │ │selected  │
├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤
│Tools:    │ │Tools:    │ │Tools:    │ │Tools:    │ │Tools:    │ │Tools:    │
│delegate  │ │get_inst  │ │available │ │check_data│ │generate  │ │recommend │
│request   │ │rument_   │ │_data_    │ │_quality  │ │_report   │ │_gap_     │
│human_inp │ │details   │ │sources   │ │delegate  │ │request_  │ │_methods  │
│extract_  │ │delegate  │ │historical│ │          │ │human_inp │ │apply_gap │
│dates     │ │request_  │ │_prices   │ │          │ │visualize │ │_filling  │
│          │ │human_inp │ │delegate  │ │          │ │          │ │request_  │
│          │ │          │ │          │ │          │ │          │ │human_inp │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
                                                                    │
                                                                    ▼
                                                          ┌──────────────────┐
                                                          │TimeSeriesConst-  │
                                                          │ructionAgent      │
                                                          ├──────────────────┤
                                                          │Persists final    │
                                                          │CSV & chart       │
                                                          │artifacts         │
                                                          ├──────────────────┤
                                                          │Tools:            │
                                                          │build_timeseries  │
                                                          │visualize_        │
                                                          │timeseries        │
                                                          └──────────────────┘
```

### Component Descriptions

| Component | File | Role | Key Methods |
|---|---|---|---|
| **CLI** | `cli.py` | Entry point. Reads user input, delegates to processor, displays events, saves artifacts. | `main()`, `_print_events()`, `_save_artifacts()` |
| **Processor** | `processor.py` | Core ReAct loop. Manages agent execution, tool calls, delegation, pause/resume, and deterministic bypasses. | `process_user_request()`, `process_user_response()`, `_run_agent()`, `_execute()` |
| **Runtime Adapter** | `runtime.py` | Framework abstraction layer and runtime factory for pluggable agentic engines. | `AgenticRuntime`, `AutogenProcessorRuntime`, `build_runtime()` |
| **Handler** | `handler.py` | Event bus and HITL state. Queues callback events, manages pause/resume state, captures ReAct traces. | `emit()`, `poll()`, `request_human_input()`, `handle_user_response()` |
| **Agent Definitions** | `agents_definition.py` | Declarative agent registry. Defines 7 agents with name, description, system prompt, tools, goal, and guardrails. | `get_agent()` |
| **Tools** | `tools.py` | Domain tool implementations. 13 tools for instrument resolution, data loading, quality checks, gap filling, and artifact generation. | `get_instrument_details()`, `historical_prices()`, `check_data_quality()`, `apply_gap_filling()` |
| **Models** | `models.py` | LiteLLM model client factory. Configurable via provider/model env vars and CLI flags. | `chat()`, `from_environment()`, `describe_environment()` |
| **Prompts** | `prompts.py` | ReAct protocol and system prompt builders. Generates agent-specific prompts with goals, tools, and guardrails. | `agent_system_prompt()`, `request_prompt()` |
| **Prompt Library** | `prompt_library.py` | Pre-defined prompt templates for HITL checkpoints. Users can select by index or label, or type free-form. | `format_prompt_menu()`, `resolve_prompt_selection()` |

### Agent-to-Agent Delegation Flow

```
User Request
    │
    ▼
┌──────────────┐     deterministic bypass     ┌──────────────────┐
│ Orchestrator ├─────────────────────────────► │ ReferenceData    │
│ (validates &  │  (if request has ticker +    │ Agent            │
│  routes)      │   dates, skip Orchestrator)  │ (resolves symbol)│
└──────┬───────┘                               └────────┬─────────┘
       │                                                │
       │  (Orchestrator delegates)                      │ auto-continuation
       │                                                │ (if Final Answer
       │                                                │  with resolved
       │                                                │  instrument)
       │                                                ▼
       │                                     ┌──────────────────┐
       │                                     │ MarketData       │
       │                                     │ Agent            │
       │                                     │ (loads all       │
       │                                     │  sources)        │
       │                                     └────────┬─────────┘
       │                                                │
       │                                                ▼
       │                                     ┌──────────────────┐
       │                                     │ DataQuality      │
       │                                     │ Agent            │
       │                                     │ (computes        │
       │                                     │  metrics)        │
       │                                     └────────┬─────────┘
       │                                                │
       │                                                ▼
       │                                     ┌──────────────────┐
       │                                     │ Reporting        │
       │                                     │ Agent            │
       │                                     │ (presents report,│
       │                                     │  pauses for user)│
       │                                     └────────┬─────────┘
       │                                                │
       │                                     User selects source
       │                                                │
       │                                                ▼
       │                                     ┌──────────────────┐
       │                                     │ GapFilling       │
       │                                     │ Agent            │
       │                                     │ (recommends      │
       │                                     │  methods, pauses │
       │                                     │  for selection)  │
       │                                     └────────┬─────────┘
       │                                                │
       │                                     User selects method
       │                                                │
       │                                                ▼
       │                                     ┌──────────────────┐
       │                                     │ TimeSeriesConst- │
       │                                     │ ructionAgent     │
       │                                     │ (persists CSV &  │
       │                                     │  chart)          │
       │                                     └──────────────────┘
       │
       ▼
┌──────────────┐
│ All agents   │
│ are          │
│ reachable    │
│ via:         │
│ • LLM        │
│   delegation │
│ • Determin-  │
│   istic      │
│   bypasses   │
│ • Auto-      │
│   continua-  │
│   tion       │
└──────────────┘
```

### Workflow Executability Analysis

Every agent in the registry is reachable through at least one path:

| Agent | Entry Path | Trigger | Always Executable? |
|---|---|---|---|
| **Orchestrator** | `process_user_request()` → `_run_agent()` | Any user request that passes financial heuristic check | ✅ Yes — always runs unless bypassed |
| **ReferenceDataAgent** | Orchestrator delegates via `delegate_to_agent` or deterministic bypass | Request contains instrument query | ✅ Yes — always runs after Orchestrator |
| **MarketDataAgent** | ReferenceDataAgent delegates via `delegate_to_agent` or auto-continuation | Instrument resolved successfully | ✅ Yes — always runs after ReferenceDataAgent |
| **DataQualityAgent** | MarketDataAgent delegates via `delegate_to_agent` | Market data loaded from at least one source | ✅ Yes — always runs after MarketDataAgent |
| **ReportingAgent** | DataQualityAgent delegates via `delegate_to_agent` | Quality metrics computed | ✅ Yes — always runs after DataQualityAgent |
| **GapFillingAgent** | User selects source → ReportingAgent delegates | User provides source selection | ✅ Yes — runs after user input at ReportingAgent pause |
| **TimeSeriesConstructionAgent** | GapFillingAgent delegates via `delegate_to_agent` | Gap-filling method applied | ✅ Yes — runs after GapFillingAgent |

**Cycle detection:** The `_run_agent()` method maintains a `visited` set of agent names. If an agent is visited twice, an `ERROR` event is emitted and the loop terminates. This prevents infinite delegation cycles.

**Error recovery paths:**
- Instrument not found → `ERROR` event with suggestions, workflow stops
- No historical data → `ERROR` event, workflow stops
- LLM produces malformed output → retry with format reminder (up to 10 iterations)
- User cancels → `ERROR` event with cancellation message, workflow stops
- Agent iteration limit reached → `ERROR` event, workflow stops

## Design Principles

1. **LLM-first behavior**: agents should reason and delegate through tool calls when possible.
2. **Guarded continuity**: deterministic fallbacks handle malformed or narrative-only LLM outputs.
3. **Mandatory user checkpoints**: source and method choices are explicit human decisions.
4. **Configuration-driven validation**: workflow checks are defined by runtime rules, not hardcoded paths.

## Implemented Components

- `agents_definition.py`: agent registry, callback event types, guardrails
- `processor.py`: ReAct loop, delegation, tool execution, pause/resume
- `handler.py`: callback queue, HITL state, trace capture, error events
- `tools.py`: domain tools over CSV fixtures in `data/`
- `models.py`: LiteLLM model client factory
- `prompts.py`: reusable ReAct and user clarification prompts
- `prompt_library.py`: pre-defined prompt templates for HITL checkpoints
- `cli.py`: interactive command line interface with prompt library integration
- `tests/test_workflow_int.py`: integration suite and workflow proofs
- `tests/test_prompt_library.py`: prompt library unit tests
- `tests/test_workflow_report.py`: workflow report and rule-validation unit tests
- `tests/test_cli_llm_integration_manual.py`: opt-in real-LLM CLI integration test

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

If an LLM final answer skips these checkpoints, the processor emits a deterministic `AWAITING_USER_INPUT` pause with generated options.

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
- `react_trace.json`
- `events.json`
- `workflow_report.json`
- `workflow_report.txt`

### Persistent Reasoning Trace

The workflow now persists LLM reasoning and runtime activity in two forms:

- `react_trace.txt`: plain-text chronological assistant trace
- `react_trace.json`: structured trace records for:
  - user requests and responses
  - raw LLM responses
  - tool calls
  - tool results
  - tool errors
  - human-in-the-loop checkpoints

Tool records include the tool description from the registry so runtime logs and trace data remain easy to interpret even as the set of tools grows.

### Workflow Report and Validation

Every CLI run now produces a structured report from callback events:

- `workflow_report.json`: machine-readable summary and validation checks
- `workflow_report.txt`: human-readable summary

Validation is runtime-configurable using:

- `TIME_SERIES_VALIDATION_RULES=/path/to/rules.json`

Example rules file:

- `financial_time_series_construction/validation_rules.example.json`

Supported rule keys:

- `require_no_errors`
- `required_pauses`
- `required_completed_agents`
- `required_delegations`
- `min_llm_delegations`

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

- `LLM_PROVIDER` one of `ollama`, `github`, `deepseek`
- `LLM_MODEL` optional full model override (e.g. `ollama/qwen2.5:1.5b`)
- `LLM_OLLAMA_MODEL` provider default when `LLM_PROVIDER=ollama`
- `LLM_GITHUB_MODEL` provider default when `LLM_PROVIDER=github`
- `LLM_DEEPSEEK_MODEL` provider default when `LLM_PROVIDER=deepseek`
- `LLM_TEMPERATURE` default `0.1`
- `LLM_MAX_TOKENS` default `2048`
- `OLLAMA_API_BASE` optional Ollama endpoint
- `DEEPSEEK_API_BASE` optional DeepSeek API endpoint
- `DEEPSEEK_API_KEY` required for DeepSeek cloud usage
- `GITHUB_TOKEN` required for GitHub model usage
- `TIME_SERIES_OUTPUT_DIR` default `~/time_series_construction`
- `TIME_SERIES_VALIDATION_RULES` optional JSON rule file for report validation
- `LOG_LEVEL` standard Python logging level

### Default LLM Profiles in .env Folder

This module loads defaults from:

- `financial_time_series_construction/.env/llm.defaults.env`
- `financial_time_series_construction/.env/llm.local.env` (optional local override)

Use `financial_time_series_construction/.env/llm.local.env.example` as a template for local/provider-specific overrides.

### CLI Provider/Model Overrides

You can switch provider/model at runtime without editing files:

```bash
python -m financial_time_series_construction.cli \
  --provider ollama \
  --model qwen2.5:1.5b \
  --request "AAPL from 2023-01-01 to 2024-01-01"
```

DeepSeek example:

```bash
python -m financial_time_series_construction.cli \
  --provider deepseek \
  --model deepseek-chat \
  --request "AAPL from 2023-01-01 to 2024-01-01"
```

GitHub model example:

```bash
python -m financial_time_series_construction.cli \
  --provider github \
  --model gpt-4.1 \
  --request "AAPL from 2023-01-01 to 2024-01-01"
```

### CLI Framework Overrides

Use a specific runtime implementation:

```bash
python -m financial_time_series_construction.cli \
  --framework autogen \
  --request "AAPL from 2023-01-01 to 2024-01-01"
```

Print resolved model/provider config and exit:

```bash
python -m financial_time_series_construction.cli --list-model-config
```

## Verification Workflow

Run real CLI with qwen and validation rules:

```bash
LLM_MODEL="ollama/qwen2.5:1.5b" \
TIME_SERIES_VALIDATION_RULES="financial_time_series_construction/validation_rules.example.json" \
python -m financial_time_series_construction.cli \
  --request "AAPL from 2023-01-01 to 2024-01-01"
```

Run opt-in live integration test:

```bash
RUN_LLM_INTEGRATION=1 \
LLM_MODEL="ollama/qwen2.5:1.5b" \
TIME_SERIES_VALIDATION_RULES="financial_time_series_construction/validation_rules.example.json" \
python -m pytest financial_time_series_construction/tests/test_cli_llm_integration_manual.py -q
```

## Prompting Guide: Which Prompts Work Fast and Why

### The Deterministic Bypass Explained

The processor has a **deterministic bypass** that skips the Orchestrator's LLM call entirely when the user's request already contains both an instrument (ticker/name) AND a date range. This bypass runs in **milliseconds** — no LLM call needed.

However, the **next agent** (ReferenceDataAgent) **still makes an LLM call** to resolve the instrument. With `deepseek-v2:16b` on CPU, this takes **10-30 seconds**. The Orchestrator bypass saves one LLM call, but the ReferenceDataAgent call is unavoidable because the LLM must decide which tool to call.

**Total LLM calls for a typical request:**

| Bypass activated? | Orchestrator | ReferenceDataAgent | MarketDataAgent | ... | Total |
|---|---|---|---|---|---|
| ✅ Yes (instrument + dates in request) | 0 calls | 1-2 calls | 3-5 calls | ... | **10-15 calls** |
| ❌ No (vague request) | 1-2 calls | 1-2 calls | 3-5 calls | ... | **11-17 calls** |

### Prompt Templates That Activate the Bypass

The bypass fires when `extract_date_range()` finds a date pattern AND `_extract_instrument_query()` finds a ticker token. Here are the exact formats that work:

#### ✅ Fast Path — Orchestrator bypassed (0 LLM calls for Orchestrator)

| Template | Example | Why it works |
|---|---|---|
| `{ticker} from {date} to {date}` | `AAPL from Jan 2023 to Jan 2024` | `from ... to ...` is the primary pattern |
| `{ticker} between {date} and {date}` | `AAPL between Jan 2023 and Jan 2024` | `between ... and ...` is the secondary pattern |
| `{ticker} start date {date} end date {date}` | `AAPL start date 2023-01-01 end date 2024-01-01` | Explicit start/end date labels |
| `{ticker:from {date} to {date}}` | `Build AAPL time series from 2023-01-01 to 2024-01-01` | Leading verbs are ignored |
| `{ticker} {date} {date}` (ISO only) | `AAPL 2023-01-01 2024-01-01` | Numbers trigger financial heuristic |

#### ❌ Slow Path — Orchestrator runs (1-2 LLM calls)

| Template | Why it fails bypass |
|---|---|
| `create AAPL` | No date range found |
| `AAPL for 2023` | Single year doesn't match `from X to Y` pattern |
| `Apple stock` | No dates at all |
| `I want data for Apple` | Too vague, no ticker or dates |

### Recommended Prompt

**Best prompt for fastest execution:**

```bash
python -m financial_time_series_construction.cli \
  --request "AAPL from 2023-01-01 to 2024-01-01"
```

This:
1. ✅ Bypasses the Orchestrator (no LLM call)
2. ✅ Ticker `AAPL` is exact — no fuzzy matching needed
3. ✅ ISO dates `2023-01-01` are unambiguous
4. ❌ Still waits 10-30s for ReferenceDataAgent's first LLM call on deepseek

### Why You Still Wait After the Bypass

Even with the perfect prompt, you will wait because:

1. **ReferenceDataAgent makes an LLM call** (10-30s on deepseek-v2:16b CPU) to decide: "Do I call `get_instrument_details` or do I ask the user for more info?"
2. **Then MarketDataAgent makes 3-5 LLM calls** to load each source.

The **total** time is dominated by model inference, not the Orchestrator bypass.

### Quickest Path to See Output

```bash
# Fastest possible experience:
# 1. Uses a smaller, faster model
# 2. Uses the optimal prompt format
LLM_MODEL="ollama/qwen2.5:1.5b" python -m financial_time_series_construction.cli \
  --request "AAPL from 2023-01-01 to 2024-01-01"
```

This should show `[REQUEST]`, `[DELEGATE]`, and `[COMPLETE]` events within **2-5 seconds** instead of 30+.

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

## Pipeline Performance Analysis

### Front-to-Back LLM Call Map

Each user request triggers a chain of sequential LLM calls — one per ReAct iteration per agent. The table below shows the typical number of LLM calls and token consumption per agent in a full workflow run.

| Step | Agent | LLM Calls | Tokens per Prompt | Total Tokens | Notes |
|---|---|---|---|---|---|
| 1 | Orchestrator | 0–1 | ~450 | 0–450 | Often bypassed by deterministic routing when request includes instrument + dates |
| 2 | ReferenceDataAgent | 1–2 | ~450 | 450–900 | Resolves instrument; auto-delegates to MarketDataAgent on completion |
| 3 | MarketDataAgent | 3–5 | ~450 | 1,350–2,250 | Lists sources, loads each source sequentially (Yahoo, Bloomberg, Reuters) |
| 4 | DataQualityAgent | 1–2 | ~400 | 400–800 | Computes quality metrics per source |
| 5 | ReportingAgent | 1 | ~400 | 400 | Presents report, pauses for user source selection |
| 6 | GapFillingAgent | 2–3 | ~400 | 800–1,200 | Recommends methods, pauses for user method selection, applies method |
| 7 | TimeSeriesConstructionAgent | 1 | ~350 | 350 | Persists final CSV and chart |
| | **Total** | **10–17** | | **3,750–6,350** | |

**Key observation:** The workflow makes **10–17 sequential LLM calls** per request. On a local Ollama model (e.g., `gemma4:e4b`), each call takes 2–10+ seconds depending on hardware, resulting in **20–170+ seconds total** for a full workflow.

### Where Tokens Are Generated

Tokens are consumed in three places:

1. **System prompts** (~350–500 tokens per agent): Include the ReAct protocol, agent goal, tool list, guardrails, and delegation examples. These are sent on every LLM call for that agent.
2. **User messages** (variable): The accumulated conversation history grows as tool results are appended. Each iteration adds the previous assistant response and the tool result.
3. **LLM responses** (50–500 tokens each): The model generates Thought/Action/Action Input blocks or Final Answers. With `max_tokens=2048`, the model may generate far more than needed.

### Performance Optimisations Already Implemented

| Optimisation | Description | Impact |
|---|---|---|
| **Deterministic Orchestrator bypass** | When the user request already includes instrument + date range, the Orchestrator is skipped entirely and ReferenceDataAgent is called directly. | Saves 1–2 LLM calls (~450–900 tokens) |
| **Orchestrator auto-progress** | When the user provides dates in a follow-up response, the Orchestrator delegates immediately instead of re-asking. | Saves 1 LLM call (~450 tokens) |
| **ReferenceDataAgent auto-continuation** | After ReferenceDataAgent resolves an instrument, the processor automatically delegates to MarketDataAgent without returning to the user. | Saves 1 user interaction + 1 LLM call |
| **MarketDataAgent source-selection bypass** | When MarketDataAgent tries to ask the user to pick one source, the processor injects a system note to load all sources automatically. | Saves 1 user interaction + 1 LLM call |
| **ReferenceDataAgent placeholder bypass** | When ReferenceDataAgent describes intent instead of calling the tool, the processor injects a system note to execute immediately. | Saves 1–2 wasted iterations |
| **Tool argument normalisation** | Common LLM aliases (e.g., `ticker` → `symbol`, `from_date` → `start_date`) are handled without retries. | Prevents retry loops |

### Recommended Further Improvements

#### 1. Reduce `max_tokens` and `max_iterations` (immediate, zero risk)

**Current:** `max_tokens=2048`, `max_iterations=10`

Most LLM responses are <200 tokens, and agents rarely need 10 iterations with the deterministic bypasses already in place.

**Change in `models.py`:**
```python
max_tokens: int = 512  # was 2048
```

**Change in `processor.py`:**
```python
for iteration in range(start, 5):  # was 10
```

**Estimated gain:** 40–60% reduction in LLM response time. The model stops generating sooner and fewer retry iterations occur.

#### 2. Compress system prompts (medium effort, high impact)

The current prompts are verbose. For example, the Orchestrator system prompt includes:
- Full ReAct protocol (18 lines)
- Delegation example (6 lines)
- Goal description
- Guardrails (4 bullet points)
- Tool list

**Estimated gain:** 150–200 fewer tokens per call × 10–17 calls = 1,500–3,400 fewer tokens processed per workflow. This translates to 20–30% faster responses.

**Approach:** Rewrite `prompts.py` to use concise language while preserving the same behavioral constraints. Guardrails add ~50–100 tokens per call — removing them entirely would save ~500–1,700 tokens total, but this is not recommended as they prevent expensive retries.

#### 3. Add a `load_all_sources` tool (medium effort, high impact)

MarketDataAgent currently loads Yahoo, Bloomberg, and Reuters in **3 separate sequential LLM calls**. The LLM decides to load each source one at a time.

**Approach:** Add a `load_all_sources` tool that fetches all sources in a single call using `concurrent.futures` to parallelize the CSV reads internally.

**Estimated gain:** Eliminates 2–3 LLM calls (15–25% reduction in total calls). The data loading itself is fast (local CSV reads), so the bottleneck is purely the LLM deciding to load each source individually.

#### 4. Use a smaller model for routing agents (medium effort, medium impact)

Simple routing decisions (Orchestrator, ReferenceDataAgent) don't need a large model. Data-intensive reasoning (MarketDataAgent, GapFillingAgent) benefits from more capable models.

**Approach:** Add per-agent model selection in `models.py`. Use `ollama/qwen2.5:1.5b` or `ollama/llama3.2:1b` for routing agents, keep the current model for data agents.

**Estimated gain:** 10–20% faster responses for routing agents (2–4 of the 10–17 calls).

#### 5. Parallelise independent agent paths (high effort, high impact)

Some agent paths are independent. For example, DataQualityAgent could run quality checks on all three sources in parallel instead of sequentially.

**Approach:** Use `concurrent.futures.ThreadPoolExecutor` in `_run_agent` or add a batch quality-check tool.

**Estimated gain:** 20–30% reduction in wall-clock time for the data quality stage.

### Summary: Performance Improvement Roadmap

| Priority | Improvement | Effort | Gain | Risk |
|---|---|---|---|---|
| P0 | Reduce `max_tokens` to 512, `max_iterations` to 5 | 5 min | 40–60% | None |
| P1 | Compress system prompts | 30 min | 20–30% | Low |
| P2 | Add `load_all_sources` tool | 1 hr | 15–25% | Low |
| P3 | Smaller model for routing agents | 1 hr | 10–20% | Medium |
| P4 | Parallelise independent agent paths | 2 hr | 20–30% | Medium |

## Model Selection Advice

### Current Model: `ollama/deepseek-v2:16b`

**deepseek-v2:16b** is a 16-billion-parameter model that provides high-quality ReAct output but is **slow on CPU-only systems** (10-30s per response). This is the primary bottleneck in the workflow.

#### Performance Characteristics

| Aspect | Rating | Notes |
|---|---|---|
| Response quality | High | Produces well-structured ReAct output with correct tool calls |
| Speed on CPU | Slow | 10-30s per response — the main source of user wait time |
| Speed on GPU (CUDA/MPS) | Moderate | 3-8s per response on GPU-equipped systems |
| RAM/VRAM requirement | High | Requires 16GB+ RAM for reasonable performance |
| ReAct compliance | Good | Reliably produces `Action`/`Action Input`/`Final Answer` format |

#### Recommendation

**Keep deepseek-v2:16b for data-intensive agents** (MarketDataAgent, GapFillingAgent, DataQualityAgent) where reasoning quality matters. For routing agents (Orchestrator, ReferenceDataAgent), consider switching to a smaller model:

```bash
# Route Orchestrator and ReferenceDataAgent with a fast model:
export LLM_MODEL_ROUTING="ollama/qwen2.5:1.5b"

# Keep deepseek for data agents:
export LLM_MODEL_DATA="ollama/deepseek-v2:16b"
```

#### Alternative Models (Best Speed/Quality Trade-off)

| Model | Params | Speed | Quality | Best For |
|---|---|---|---|---|
| `ollama/llama3.2:3b` | 3B | Fast (3-5s) | Medium | All-rounder — minimum for DataQualityAgent |
| `ollama/gemma4:e4b` | ~4B | Medium (3-6s) | Medium-High | Good all-rounder with 8GB+ RAM |
| `ollama/qwen2.5:1.5b` | 1.5B | Fast (2-4s) | Medium | Orchestrator, ReferenceDataAgent |
| `ollama/deepseek-v2:16b` | 16B | Slow (10-30s) | High | MarketDataAgent, GapFillingAgent (current) |

To diagnose performance issues at runtime, use `--verbose` and look for `timer_slow` log messages indicating which agent calls are taking the longest.

## Debug Logging

### Enhanced Debug Logger (`debug_logger.py`)

A dedicated debugging module provides runtime visibility into the workflow:

#### Timer
Logs elapsed time for each LLM call, categorised as:
- `timer_slow` (>5s) — **WARNING level** — indicates a performance bottleneck
- `timer_ok` (1-5s) — INFO level — normal response time  
- `timer_fast` (<1s) — DEBUG level — fast response

Usage in code:
```python
with timer(agent_name, iteration, "LLM call"):
    response = llm.chat(...)
```

#### Loop Detector
Detects two types of loops:
1. **Tool call loop** — same tool called with identical arguments 3+ times
2. **Response loop** — identical LLM response text across consecutive iterations

Both emit WARNING-level logs, making it easy to spot when the LLM is stuck.

#### Message Size Tracking
Logs conversation history growth per iteration:
```log
message_size agent=Orchestrator iteration=2 messages=5 chars=2400 estimated_tokens=600
```
Warns when estimated tokens exceed 2000, which indicates history may need truncation.

#### Workflow Progress Logger
Structured status logging at each stage:
```log
progress agent=Orchestrator iteration=0 status=started
progress agent=Orchestrator iteration=0 status=llm_call
progress agent=Orchestrator iteration=0 status=completed
```

### Interpreting the Logs

Run with `--verbose` to see all debug output:

```bash
python -m financial_time_series_construction.cli \
  --request "Build AAPL from 2023-01-01 to 2023-12-31" \
  --verbose 2>&1 | grep -E "(timer|loop|message_size|progress)"
```

Typical slow-path output:
```
WARNING timer_slow agent=Orchestrator iteration=0 label="LLM call" elapsed=18.34s
WARNING timer_slow agent=MarketDataAgent iteration=0 label="LLM call" elapsed=22.10s
```

If you see repeated `timer_slow` warnings for the same agent, consider:
1. Reducing `max_tokens` in `models.py` (from 2048 to 512)
2. Using a smaller model for that specific agent
3. Adding a `load_all_sources` tool to reduce MarketDataAgent LLM calls

## Notes

- The implementation is designed to be extensible by adding agents and tools to registries.
- Current data connectors are local CSV-backed fixtures for reproducible development and testing.
