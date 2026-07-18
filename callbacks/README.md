# Travel Booking Agent System — Callbacks + ReAct

An LLM-driven travel booking system built with **ReAct agents** and **human-in-the-loop** callbacks. All workflows are driven by a local LLM (Ollama + Gemma 4:e4b) via litellm — no hardcoded state machines.

## Architecture

The system has four files with clear separation of concerns:

```
┌──────────────────────────────────────────────────┐
│                   cli.py                         │
│  Presentation layer                              │
│  - Prompt templates (CAR_TYPE_SELECTION, etc.)   │
│  - Formatters (format_car_options, etc.)         │
│  - Event display dispatch                        │
│  - User input loop                               │
│  Depends on: agents.py (types), processor.py     │
└──────────────────────┬────────────────────────────┘
                       │ user request / response
                       ▼
┌──────────────────────────────────────────────────┐
│               processor.py                        │
│  Orchestration layer                              │
│  - _react_loop() — single ReAct loop             │
│  - _call_llm() — litellm → Ollama                │
│  - _parse_tool_calls() — 3 formats (JSON, XML,   │
│    Action:)                                       │
│  - _execute_tool() — dispatch table               │
│  - TravelBookingProcessor — thin facade           │
│  Depends on: agents.py, handler.py               │
└──────────────────────┬────────────────────────────┘
                       │ events / pause
                       ▼
┌──────────────────────────────────────────────────┐
│                handler.py                         │
│  Callback layer                                   │
│  - TravelBookingCallbackHandler (extends          │
│    BaseCallbackHandler)                           │
│  - Event queue (emit/poll/has_events)             │
│  - Pause/resume state for HITL                   │
│  - Overrides: on_agent_finish, on_*_error         │
│  Depends on: agents.py (types)                   │
└──────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│                agents.py                          │
│  Data layer                                       │
│  - Core types: Agent, Tool, CallbackEvent,        │
│    CallbackEventType                              │
│  - Tool implementations (stubs)                   │
│  - TOOL_REGISTRY, AGENT_REGISTRY                  │
│  - Lookup helpers: get_agent(), get_tool()        │
│  No dependencies on other modules                 │
└──────────────────────────────────────────────────┘
```

## File Responsibilities

### `agents.py` — Data layer (pure configuration)

- **Types**: `Agent`, `Tool`, `CallbackEvent`, `CallbackEventType`
- **Tool stubs**: 8 functions that return JSON strings (`search_cars`, `book_car`, `select_car_type`, `search_flights`, `book_air_ticket`, `search_hotels`, `book_hotel`, `request_human_input`)
- **Registries**: `TOOL_REGISTRY` (8 tools with JSON Schemas + implementations), `AGENT_REGISTRY` (4 agents with system prompts + tool lists)
- **Lookups**: `get_agent()`, `get_tool()`
- No imports from other project modules

### `handler.py` — Callback layer

- **`TravelBookingCallbackHandler`** extends LangChain's `BaseCallbackHandler`
- **Event queue**: `emit()`, `poll()`, `has_events()`, `clear_events()`
- **Human-in-the-loop**: `request_human_input()` pauses the agent and emits an `AWAITING_USER_INPUT` event; `handle_user_response()` returns saved state or None on cancel
- **Overrides**: `on_agent_finish` (emits `AGENT_COMPLETED`), `on_llm_error`/`on_tool_error` (emit `ERROR`)
- 60 lines of logic — no dead code, no unused overrides

### `processor.py` — Orchestration layer

- **`_call_llm()`**: calls litellm `completion()` with `model="ollama/gemma4:e4b"`
- **`_parse_tool_calls()`**: parses LLM output in 3 formats (JSON `{"name", "arguments"}`, XML `<function_call>`, ReAct `Action:` + `Action Input:`)
- **`_execute_tool()`**: dispatch table — special-cases `request_human_input` to trigger the handler, otherwise calls `tool.fn(**args)`
- **`_react_loop()`**: **single** function handling both initial runs and resumed pauses — no `run_agent`/`resume_agent` duplication
- **`TravelBookingProcessor`**: thin facade — `process_user_request()` starts the orchestrator, `process_user_response()` resumes the paused agent

### `cli.py` — Presentation layer

- **Prompt templates**: `CAR_TYPE_SELECTION_PROMPT`, `FLIGHT_SELECTION_PROMPT`, `HOTEL_SELECTION_PROMPT`, `GENERAL_INPUT_PROMPT`, `CONFIRMATION_PROMPT`
- **Formatters**: `format_car_options()`, `format_flight_options()`, `format_hotel_options()`
- **`_event_display()`**: pure function mapping event → display string — one place for all UI formatting
- **`TravelBookingCLI`**: input loop with `help`, `quit`, `reset` commands, delegates to processor

## Data Flow

### New request

```
User: "Book a car in London"
  │
  ▼  cli → processor.process_user_request("Book a car...")
  │
  ▼  _react_loop(Orchestrator, messages=[user_input])
  │     │
  │     ▼ Iteration 1: LLM decides "This needs CarBookingAgent"
  │     → Action: request_human_input(prompt="Delegating...", context={delegate_to: "CarBookingAgent"})
  │     → handler.request_human_input() → emits AWAITING_USER_INPUT
  │     → pauses (saves state in handler.paused_state)
  │
  ▼  cli displays the prompt and waits
  │
User: "Yes, book an SUV"
  │
  ▼  cli → processor.process_user_response("Yes, book an SUV")
  │
  ▼  handler.handle_user_response() → returns saved state
  ▼  Detects delegate_to → starts fresh CarBookingAgent
  ▼  _react_loop(CarBookingAgent, messages=[user_request], start_iteration=0)
  │     │
  │     ▼ Iteration 1: Action: search_cars(location="London")
  │     → result: {"available_cars": [...]}
  │     │
  │     ▼ Iteration 2: Action: request_human_input(prompt="Select car:", options=[...])
  │     → pauses
  │
  ▼  cli displays options → user selects "1"
  ▼  resume → _react_loop with updated messages
  │     ▼ Action: book_car(car_id="car_1")
  │     ▼ Action: request_human_input(prompt="Confirm car type:")
  │     ▼ Action: select_car_type(car_id="car_1", car_type="Toyota Corolla")
  │     ▼ Final Answer: "Car Toyota Corolla booked in London..."
  │     → emits AGENT_COMPLETED
```

### Resume cycle

1. `handler.request_human_input()` emits `AWAITING_USER_INPUT`, sets `waiting_for_input=True`, saves state in `paused_state`
2. `_react_loop()` returns early — loop pauses
3. CLI displays the prompt, collects user input
4. `process_user_response()` calls `handler.handle_user_response()` which returns the saved state
5. Processor calls `_react_loop()` again with the saved messages + user response appended
6. LLM continues reasoning with the new input

## Changes from Original Design

| Aspect | Before | After |
|--------|--------|-------|
| **ReAct loop** | `run_agent()` + `resume_agent()` (80% duplicated, 200+ lines) | Single `_react_loop()` (60 lines) |
| **Presentation** | Prompt templates + formatters in `agents.py` | Moved to `cli.py` — the UI layer |
| **Event helpers** | Dead `process_callback_event/s` in `handler.py` (70 lines, never called) | Removed |
| **Handler imports** | 10+ unused imports from `agents` | Just `CallbackEvent`, `CallbackEventType` |
| **Handler overrides** | 7 overrides, most just logging | 3 meaningful overrides |
| **State management** | `pending_context` in handler + duplicate tracking | Single `paused_state` dict |
| **CLI event dispatch** | 8 separate `_handle_*` methods | Single `_event_display()` function |
| **File sizes** | agents.py 465, handler.py 325, processor.py 593, cli.py 271 | agents.py 290, handler.py 130, processor.py 220, cli.py 250 |

## Adding a New Agent

```python
# In agents.py — add tool implementations
def _search_trains(origin, destination, date):
    return json.dumps({"available_trains": [...], ...})

# Register the tool
TOOL_REGISTRY["search_trains"] = Tool(
    name="search_trains",
    description="Search for available trains...",
    parameters={"type": "object", "properties": {...}, "required": [...]},
    fn=_search_trains,
)

# Define the agent
AGENT_REGISTRY["TrainBookingAgent"] = Agent(
    name="TrainBookingAgent",
    description="Handles train ticket booking.",
    system_prompt="You are a train booking specialist...",
    tools=["search_trains", "book_train_ticket", "request_human_input"],
)
```

The orchestrator automatically discovers the new agent via the LLM's reasoning about which agent name to delegate to.

## Running

```bash
ollama pull gemma4:e4b
python cli.py
```

| Environment Variable | Default | Purpose |
|---------------------|---------|---------|
| `OLLAMA_ENDPOINT` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `gemma4:e4b` | Model name |
| `LITELLM_LOG` | `WARNING` | litellm verbosity |