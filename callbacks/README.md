# Travel Booking Agent System — LangChain Callbacks + ReAct Agents

An LLM-driven travel booking system built with **ReAct agents**, **LangChain-style callbacks**, and **human-in-the-loop** interaction. All agent workflows are driven by a local LLM (Ollama + Gemma 4:e4b) via litellm — no hardcoded workflows.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                        cli.py                            │
│  Interactive CLI — accepts user text, displays prompts   │
└────────────────────────┬─────────────────────────────────┘
                         │ user input / response
                         ▼
┌──────────────────────────────────────────────────────────┐
│                     processor.py                          │
│  TravelBookingProcessor                                  │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  run_agent() / resume_agent()                       │  │
│  │  - ReAct loop (Thought → Action → Observation)      │  │
│  │  - Tool parsing & execution                         │  │
│  │  - Agent lifecycle management                       │  │
│  └─────────────────────────────────────────────────────┘  │
└────────────────────────┬─────────────────────────────────┘
                         │ events
                         ▼
┌──────────────────────────────────────────────────────────┐
│                      handler.py                           │
│  TravelBookingCallbackHandler                            │
│  - LangChain BaseCallbackHandler subclass                │
│  - Event queue management                               │
│  - Human-in-the-loop state machine                      │
│  - request_human_input() → AWAITING_USER_INPUT event     │
│  - handle_user_response() → resume agent                 │
└──────────────────────────────────────────────────────────┘
```

## Components

### 1. `agents.py` — Agent & Tool Definitions

Defines four **ReAct agents** and their **tools**:

| Agent | Role | Tools |
|-------|------|-------|
| **Orchestrator** | Routes user requests to the correct specialist agent | `request_human_input` |
| **CarBookingAgent** | Handles car rental search & booking | `search_cars`, `book_car`, `select_car_type`, `request_human_input` |
| **AirTicketAgent** | Handles flight search & booking | `search_flights`, `book_air_ticket`, `request_human_input` |
| **HotelReservationAgent** | Handles hotel search & booking | `search_hotels`, `book_hotel`, `request_human_input` |

Key design elements:

- **`Agent` dataclass** — each agent has a `system_prompt`, `name`, `description`, and list of `tools` it can use
- **`Tool` dataclass** — each tool has a `name`, `description`, JSON `parameters` schema, and an optional `fn` implementation
- **`TOOL_REGISTRY` / `AGENT_REGISTRY`** — dict-based registries for dynamic lookup
- **`CallbackEvent` / `CallbackEventType`** — typed events that flow through the callback system (e.g., `AWAITING_USER_INPUT`, `SELECT_CAR`, `BOOKING_CONFIRMED`)

**Prompt templates** (`CAR_TYPE_SELECTION_PROMPT`, `FLIGHT_SELECTION_PROMPT`, etc.) format options for the user when a human-in-the-loop decision is needed.

### 2. `handler.py` — LangChain Callback Handler

A custom `BaseCallbackHandler` subclass called `TravelBookingCallbackHandler`. It:

- **Maintains an event queue** (`deque[CallbackEvent]`) — events emitted by agents are queued here
- **Manages human-in-the-loop state** — tracks whether the system is waiting for user input (`waiting_for_input` flag) and stores the context needed to resume the agent (`pending_context`)
- **Provides `request_human_input()`** — emits an `AWAITING_USER_INPUT` event with the prompt and selectable options, then pauses the agent loop
- **Provides `handle_user_response()`** — accepts the user's text, stores it in conversation history, and returns the context so `processor.py` can resume the agent
- **Overrides LangChain callbacks** — `on_llm_start/end`, `on_agent_action/finish`, `on_tool_start/end`, `on_text`, etc. — to log activity and emit events

The handler is the **bridge** between the agent loop (which is LLM-driven) and the CLI (which is human-driven).

### 3. `processor.py` — Agent Loop & Tool Execution

The core engine that runs the ReAct loop for each agent:

**`run_agent(agent, user_input, callback_handler)`**

1. Builds the ReAct system prompt (agent's system prompt + tool descriptions + ReAct format instructions)
2. Calls the LLM via litellm (`completion()` with Ollama/Gemma 4:e4b)
3. Parses tool calls from the LLM response (supports JSON, XML, and ReAct `Action:` formats)
4. Executes each tool via `_execute_tool()`
5. Special-cases `request_human_input` — instead of executing a function, it triggers the callback handler's human-in-the-loop mechanism, which pauses the ReAct loop
6. Repeats until `Final Answer:` is found or max iterations reached

**`resume_agent(agent, user_input, callback_handler)`**

- Restores saved state (`messages`, `iteration`, `system_prompt`) from the callback handler's `pending_context`
- Appends the user's response to the conversation history
- Continues the ReAct loop from where it paused

**`_call_llm()`**

- Uses litellm's `completion()` with `model="ollama/gemma4:e4b"`, configurable via `OLLAMA_ENDPOINT` and `OLLAMA_MODEL` environment variables
- Passes tools in OpenAI-compatible format to enable function-calling when the model supports it

**`_parse_tool_calls()`**

Parses tool calls from LLM text in three formats:
- `{"name": "tool_name", "arguments": {...}}` — JSON function call
- `<function_call>tool</function_call>` + JSON args — XML style
- `Action: tool\nAction Input: {...}` — classic ReAct format

### 4. `cli.py` — Interactive CLI

The user-facing interface. Key methods:

- **`handle_user_input(user_input)`** — routes input to either `process_user_request()` (new requests) or `process_user_response()` (responses to human-in-the-loop prompts), then processes all emitted events
- **Event handlers** — `_handle_awaiting_input_event()`, `_handle_select_car_event()`, `_handle_agent_completed_event()`, `_handle_booking_confirmed_event()`, etc. — display formatted output to the user
- **Commands**: `help`, `quit`/`exit`, `reset`, or any natural language request

## Data Flow (End-to-End)

### New user request flow

```
User: "Book a car in London next week"
        │
        ▼
cli.py ──► processor.py.process_user_request("Book a car...")
                │
                ▼
        run_agent(Orchestrator, "Book a car...")
                │
                ▼
        LLM (Gemma 4:e4b) decides: "This is a car booking → delegate to CarBookingAgent"
                │
                ▼
        request_human_input(prompt="I'll delegate...", context={delegate_to: "CarBookingAgent"})
                │
                ▼
        AWAITING_USER_INPUT event emitted to callback handler's queue
                │
                ▼
cli.py picks up the event, displays prompt to user
                │
User: "Yes, I want an SUV"
                │
                ▼
cli.py ──► processor.py.process_user_response("Yes, I want an SUV")
                │
                ▼
        resume_agent(CarBookingAgent, "Yes, I want an SUV")
                │
                ▼
        LLM decides: "Search cars in London"
                │
        Action: search_cars
        Action Input: {"location": "London", "pickup_date": "2026-07-25", "dropoff_date": "2026-07-28"}
                │
                ▼
        Tool returns: {"available_cars": [...3 cars...]}
                │
                ▼
        LLM decides: "Present options to user"
                │
        Action: request_human_input(prompt="Select a car:", options=["Toyota Corolla $45/day", ...])
                │
                ▼
        AWAITING_USER_INPUT event → CLI displays options
                │
User: "1"  (selects first car)
                │
                ▼
cli.py ──► processor.py.process_user_response("1")
                │
                ▼
        resume_agent(CarBookingAgent, "1")
                │
                ▼
        LLM decides: "User selected Toyota Corolla → book it"
                │
        Action: book_car(car_id="car_1", ...)
                │
                ▼
        Tool returns: {"status": "pending_confirmation", ...}
                │
                ▼
        LLM decides: "Ask user to confirm car type"
                │
        Action: request_human_input(prompt="Confirm car type:", options=["Toyota Corolla", ...])
                │
                ▼
        AWAITING_USER_INPUT event → CLI displays confirmation
                │
User: "1"  (confirms)
                │
                ▼
        resume_agent → LLM → select_car_type(car_id="car_1", car_type="Toyota Corolla")
                │
                ▼
        Final Answer: "Car booked successfully..." → AGENT_COMPLETED event
```

### How human-in-the-loop works step by step

1. **Agent calls `request_human_input`** via the ReAct `Action: request_human_input` format
2. **`_execute_tool()`** detects this is the special human-input tool and calls `callback_handler.request_human_input(prompt, options, context)`
3. **Callback handler** emits an `AWAITING_USER_INPUT` event to its queue and sets `waiting_for_input = True`, storing the agent's current state (`messages`, `iteration`, `system_prompt`) in `pending_context`
4. **`run_agent()` / `resume_agent()`** returns early — the agent loop pauses
5. **CLI** picks up the `AWAITING_USER_INPUT` event, displays the prompt and options, and waits for user input
6. **User types a response** → CLI calls `processor.py.process_user_response(user_input)`
7. **Processor** calls `callback_handler.handle_user_response(user_input)` which stores the response and returns the saved context
8. **Processor** calls `resume_agent()` with the saved state + user response, which appends "User response: ..." to the messages and continues the ReAct loop

### How the ReAct loop works

The ReAct (Reasoning + Acting) loop is the core pattern:

```
Iteration 1:
  LLM: Thought: The user wants to book a car. I should search for available cars.
       Action: search_cars
       Action Input: {"location": "London", "pickup_date": "2026-07-25", "dropoff_date": "2026-07-28"}

  Processor: Executes search_cars → returns results

Iteration 2:
  LLM: Thought: I found 3 cars. I need the user to select one.
       Action: request_human_input
       Action Input: {"prompt": "Select a car:", "options": [...]}

  Processor: Triggers callback → pauses for user input

  ...user responds...

Iteration 3:
  LLM: Thought: The user selected option 1. I'll book that car.
       Action: book_car
       Action Input: {"car_id": "car_1", ...}

  ...continues until Final Answer...
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OLLAMA_ENDPOINT` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `gemma4:e4b` | Ollama model name |
| `LITELLM_LOG` | `WARNING` | litellm log level |

## Running the System

```bash
# Ensure Ollama is running with the required model
ollama pull gemma4:e4b

# Run the CLI
python cli.py
```

## Design Principles

1. **LLM-driven workflows** — no hardcoded state machines or workflows. The LLM decides which tools to call and when, based on its ReAct reasoning.

2. **Human-in-the-loop via callbacks** — when the agent needs user input, it emits a callback event and pauses. The UI layer (CLI) handles the event, collects user input, and resumes the agent.

3. **Modular agent definitions** — agents, tools, and prompt templates are all defined declaratively in `agents.py`. Adding a new agent (e.g., `TrainBookingAgent`) requires only adding its definition and tools — no changes to the processor or handler.

4. **Event-driven architecture** — the callback handler's event queue decouples the agent loop from the UI. The same processor could drive a web UI or an API server by replacing only the CLI layer.

5. **Flexible tool call parsing** — supports multiple LLM output formats (JSON, XML, ReAct) to work with different models and prompt styles.

## Adding a New Agent

```python
# In agents.py:

CAR_BOOKING_SYSTEM_PROMPT = """You are a train booking specialist agent.
    Search for trains, present options, and book when confirmed."""

TRAIN_BOOKING_AGENT = Agent(
    name="TrainBookingAgent",
    description="Handles train ticket booking.",
    system_prompt=TRAIN_BOOKING_SYSTEM_PROMPT,
    tools=["search_trains", "book_train_ticket", "request_human_input"],
)

AGENT_REGISTRY["TrainBookingAgent"] = TRAIN_BOOKING_AGENT

# Add new tools to TOOL_REGISTRY
TOOL_REGISTRY["search_trains"] = Tool(
    name="search_trains",
    description="Search for available trains between cities on a date.",
    parameters={...},
    fn=_search_trains,
)
```

The orchestrator will automatically discover and delegate to the new agent based on the LLM's reasoning about the user's request.