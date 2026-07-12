# Ollama Model Speed Benchmark

A learning tool that benchmarks inference performance for locally available Ollama models, implemented using **5 different LLM-driven workflow approaches** — all sharing the same underlying tools but wired together with different strategies.

## What It Does

1. **Lists** all Ollama models available on your machine.
2. **Asks you** to pick which models to benchmark (human-in-the-loop).
3. **Confirms** your selection.
4. **Benchmarks** each selected model by running N inference iterations, measuring latency, and computing statistics (mean, median, P95, P99, throughput tokens/sec).
5. **Reports** a final comparison table and saves results to `ollama_benchmark_results.json`.

## The ReAct Pattern

All 5 approaches are built on the **ReAct (Reasoning + Acting)** paradigm, where an LLM alternates between two phases in a loop:

```
┌─────────────────────────────────────────────────────────────┐
│                      ReAct Loop                              │
│                                                              │
│   ┌──────────┐    Tool call?     ┌──────────┐               │
│   │  REASON  │ ────────────────→ │   ACT    │               │
│   │  (LLM)   │ ←──────────────── │ (Tool)   │               │
│   └──────────┘   Tool result     └──────────┘               │
│        │                                                     │
│        │  No tool call → Final answer → END                  │
│        └───────────────────────────────────────────────────→ │
└─────────────────────────────────────────────────────────────┘
```

1. **Reasoning**: The LLM examines the conversation history (system prompt, previous tool outputs, user messages) and decides what to do next — either respond directly with a final answer or invoke a tool.
2. **Acting**: If the LLM chose to call a tool, the tool executes and its result is returned. The LLM then reasons again, incorporating the new information.

This loop continues until the LLM responds without requesting any tool call — at which point the graph terminates. **The LLM itself decides the sequence of tool calls** based on the system prompt guidance and the outputs it observes from each tool.

### Conversation History vs. Checkpoints

The agent has **two separate mechanisms** for maintaining state across steps. They are often confused but serve different purposes:

#### 1. Conversation History (the LLM's context)

The agent's "brain" is the **conversation history** — a list of messages that grows with each step. This is what the LLM sees when it reasons. It is the **input context** for the LLM:

| Message Type | Source | Purpose |
|---|---|---|
| **SystemMessage** | `state_modifier` | Persistent instructions defining the agent's role and goals |
| **HumanMessage** | User input (via `Command(resume=...)`) | User responses to interruptions (model selection, confirmation) |
| **AIMessage** | LLM responses | Contains either a final answer or `tool_calls` |
| **ToolMessage** | Tool execution results | Output from each tool, read by the LLM on the next reasoning step |

**How it grows:** Each time the LLM reasons, its response (AIMessage) is appended. Each time a tool executes, its result (ToolMessage) is appended. Each time the user provides input, a HumanMessage is appended. The list grows monotonically — nothing is ever removed.

**Strategies for managing conversation history:**

| Strategy | Description | Context Window Impact | Persistence | Use Case |
|---|---|---|---|---|
| **Append-only** (this project) | Keep all messages. The list grows monotonically | Can exceed LLM's context window for long workflows | In-memory | Simple workflows with few steps |
| **Windowing / Sliding window** | Keep only the last N messages. Drops older context to stay within token limits | Fixed-size context, older context is lost | In-memory | Long-running agents where only recent context matters |
| **Summarization** | Periodically summarize older messages into a single condensed message. An LLM call compresses N messages → 1 summary | Dramatically reduces tokens while preserving key information | In-memory (summary replaces old messages) | Workflows where historical context is needed but too large to keep raw |
| **Structured output** (Approach 5) | Instead of appending raw tool outputs, the LLM emits structured JSON. A deterministic controller maintains its own compact state (`current_step`, `step_history`) | Minimal — only keeps structured state, not raw messages | Controller state (in-memory) | Workflows with predictable steps where only the latest status matters |
| **External database / persistent storage** | Store the full message history in an external database (SQLite, PostgreSQL, Redis, etc.) while keeping only the most recent messages in the LLM's context. When the LLM needs older context, it queries the database (e.g., via a retrieval tool) | Context window only holds recent messages + retrieved chunks | Database (disk or network) | Long-running or stateful agents that need access to full history but cannot fit it in the LLM's context window |
| **Vector store / RAG** | Embed each message into a vector database. When the LLM needs context, it performs a similarity search to retrieve the most relevant past messages. This is Retrieval-Augmented Generation (RAG) applied to conversation history | Context window only holds the query + top-K retrieved chunks | Vector database (e.g., Chroma, Pinecone, Weaviate) | Agents that need to search through very long histories for relevant information (e.g., customer support, research assistants) |
| **External memory agent** | A separate agent or module manages a knowledge graph or key-value store that the main agent reads/writes via tools. The main agent only keeps a summary in its context window | Minimal — only the current query and a summary of stored knowledge | Specialised store (graph, KV-store, etc.) | Complex reasoning tasks where relationships between past steps matter (e.g., multi-session planning) |

**How databases fit in:**

The conversation history is stored in-memory as part of the LangGraph state. By default, this is ephemeral. To extend context beyond the LLM's limit or across sessions, you can:

1. **Replace the checkpointer** — Instead of `MemorySaver`, use `SqliteSaver`, `PostgresSaver`, or `RedisSaver`. These persist the full graph state (including the message list) to a database. This allows recovering the conversation history after a process restart, but does **not** solve the LLM context window limit — the full message list is still sent to the LLM.

2. **Use a retrieval tool** — Add a tool that queries an external database for relevant history. The LLM can call this tool when it needs context from earlier steps. This keeps the conversation window small and lets the LLM fetch what it needs on demand. Example:
   ```python
   @tool
   def query_history_tool(query: str) -> str:
       \"\"\"Search past conversation history for relevant context.\"\"\"
       # Query a vector store or database
       results = vector_store.similarity_search(query, k=3)
       return "\n".join([r.page_content for r in results])
   ```

3. **Periodic summarization + external storage** — After every N steps, an LLM call compresses the message list into a summary. The raw messages are stored in a database, and the summary replaces them in the context window. If the LLM needs details, it can query the database.

#### 2. Checkpoints (graph execution state)

Checkpoints are **snapshots of the entire graph state** at each step, stored by the checkpointer (`MemorySaver`). They are the **execution state** of the LangGraph state machine:

| Aspect | Conversation History | Checkpoints |
|---|---|---|
| **What it stores** | Messages the LLM sees (SystemMessage, HumanMessage, AIMessage, ToolMessage) | The full graph state: messages + any custom state fields (status, available_models, etc.) |
| **Purpose** | Provides context for the LLM to reason and decide | Enables pause/resume of graph execution |
| **Who uses it** | The LLM (as input) | LangGraph runtime (to restore execution) |
| **Growth** | Grows with each step (monotonic) | One snapshot per step (can be pruned) |
| **Persistence** | In-memory (part of the graph state) | Configurable: MemorySaver (RAM), SqliteSaver (disk), PostgresSaver/RedisSaver (network) |
| **Required for interrupt()?** | No | **Yes** — without checkpoints, `interrupt()` cannot work because LangGraph needs to restore the state when `Command(resume=...)` is called |

**Key insight:** The conversation history is *part of* the checkpoint. When a checkpoint is saved, it includes the full messages list. When a checkpoint is restored, the messages list is restored too. But the reverse is not true — you can have a conversation history without checkpoints (if you don't need pause/resume).

## Further Improvements for Context Management

The current implementation uses a simple append-only strategy with `MemorySaver` checkpoints. Here are concrete improvements that could be made, ordered from simplest to most advanced:

### 1. Upgrade the Checkpointer for Persistence

Replace `MemorySaver` with a persistent backend so the conversation history survives process restarts:

| Checkpointer | Package | Trade-offs |
|---|---|---|
| **SqliteSaver** | `langgraph.checkpoint.sqlite` | Simple file-based persistence. No server needed. Good for single-user scripts. |
| **PostgresSaver** | `langgraph.checkpoint.postgres` | Full SQL database with ACID guarantees. Suitable for multi-user or production deployments. Adds operational overhead. |
| **RedisSaver** | `langgraph.checkpoint.redis` | Fast in-memory store with optional persistence. Good for high-throughput, low-latency scenarios. Requires a Redis server. |

**What changes:** Only the checkpointer instantiation. The graph and tools remain unchanged.

### 2. Add a Retrieval Tool for Long-Term Memory

Add a tool that allows the LLM to query an external database for relevant past context. This keeps the conversation window small while preserving access to the full history.

```python
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Setup: embed and store messages as they arrive
vector_store = Chroma(
    collection_name="benchmark_history",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
)

@tool
def query_history_tool(query: str) -> str:
    """Search past conversation history for relevant context.
    Call this when you need to recall details from earlier steps."""
    results = vector_store.similarity_search(query, k=3)
    return "\n".join([r.page_content for r in results])
```

**What changes:** Add `query_history_tool` to the tools list. Add a LangGraph callback that embeds each new message into the vector store after the tools node executes.

### 3. Implement Windowing / Sliding Window

Instead of appending all messages, keep only the last N messages in the state. This prevents the context window from growing unbounded.

```python
from langgraph.graph import StateGraph, END

MAX_MESSAGES = 20  # Keep only the last 20 messages

def truncate_messages(state: dict) -> dict:
    messages = state.get("messages", [])
    if len(messages) > MAX_MESSAGES:
        # Keep the system prompt + the last N-1 messages
        state["messages"] = [messages[0]] + messages[-(MAX_MESSAGES - 1):]
    return state
```

**What changes:** Add a truncation node that runs after every tool call. The trade-off is that the LLM loses access to older context.

### 4. Implement Summarization

After every N steps, use an LLM call to compress the message history into a single summary message, replacing the raw history. This preserves key information while drastically reducing tokens.

```python
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

SUMMARY_PROMPT = "Summarize the following conversation, preserving key decisions, tool outputs, and user preferences:"

def summarize_messages(state: dict) -> dict:
    messages = state.get("messages", [])
    if len(messages) < 10:  # Don't summarise until we have enough messages
        return state

    summarizer_llm = ChatOllama(model="llama3.2", temperature=0)
    summary = summarizer_llm.invoke([
        SystemMessage(content=SUMMARY_PROMPT),
        HumanMessage(content=str(messages[1:]))  # Skip system prompt
    ])

    # Keep the system prompt + replace everything else with the summary
    state["messages"] = [messages[0], summary]
    return state
```

**What changes:** Add a summarization node that runs periodically (e.g., every 10 steps). Store the raw messages in an external database before summarising, so the LLM can still query details via a retrieval tool (combining this with improvement #2).

### 5. Hybrid: Summarization + External Storage + Retrieval

The most robust approach combines all three techniques:

1. **Summarization**: After every N steps, an LLM call compresses the message list into a summary.
2. **External storage**: The raw messages are stored in a database (SQLite or vector store) before summarisation.
3. **Retrieval**: A `query_history_tool` lets the LLM fetch specific details from the raw history on demand.

This keeps the context window small (just the summary), preserves the full history in persistent storage, and allows the LLM to access details when needed.

```
Before summarisation → messages stored in vector DB
After summarisation  → raw messages replaced with summary in context
On demand            → LLM calls query_history_tool() to retrieve specific details
```

### 6. Structured Long-Term Memory with Knowledge Graphs

For complex, multi-session workflows, use a knowledge graph (e.g., Neo4j) to store relationships between entities, decisions, and outcomes. The LLM reads/writes the graph via tools and only keeps a summary in its context window.

```python
@tool
def store_fact_tool(subject: str, predicate: str, obj: str) -> str:
    """Store a fact in the knowledge graph.
    Example: subject='user', predicate='selected_model', obj='llama3.2'"""
    # graph.query("CREATE (a:Entity {name: $s})-[r:RELATION {type: $p}]->(b:Entity {name: $o})",
    #             s=subject, p=predicate, o=obj)
    return f"Stored: {subject} -[{predicate}]-> {obj}"

@tool
def query_graph_tool(query: str) -> str:
    """Query the knowledge graph for relevant context.
    Example: 'What models did the user select?'"""
    # results = graph.query("MATCH (a)-[r]->(b) WHERE ... RETURN ...")
    # return format_results(results)
    return "Query results..."
```

**What changes:** This is a significant architectural change — requires a graph database and substantially more code. Suitable for long-running agents that operate across multiple sessions.

## The 5 Approaches

Each approach is implemented in its own file and can be selected via `--method`:

### 1. Prompt-Driven (`benchmarks_prompt_driven.py`)
**Strategy**: A single ReAct agent with a detailed system prompt describing the 5-step workflow. The LLM decides the tool call order based on the prompt + tool outputs.
- **Pros**: Flexible, handles unexpected situations, adapts to errors.
- **Cons**: May skip steps or loop if the prompt isn't precise enough.
- **Mitigation**: Prompt engineering + tool-level preconditions (each tool checks prerequisites and returns clear error messages).
- **Default method**.

### 2. Tool-Driven (`benchmarks_tool_driven.py`)
**Strategy**: Minimal system prompt — the LLM relies primarily on each tool's docstring (metadata) to understand when to call it. The tool descriptions explicitly state preconditions.
- **Pros**: Demonstrates how tool metadata alone can guide a workflow.
- **Cons**: Still depends on LLM reasoning; no enforcement of order.
- **Key difference from prompt-driven**: The prompt is shorter; the tool descriptions do the heavy lifting.

### 3. Heuristic / Rule-Based Supervisor (`benchmarks_heuristic_supervisor.py`)
**Strategy**: A deterministic StateGraph with hard-coded conditional edges that route based on `state.status`. Each node calls one tool and updates the status. The supervisor enforces ordering.
- **Pros**: Guarantees correct sequence; easy to reason about.
- **Cons**: Rigid — the LLM cannot adapt the workflow.
- **When to use**: When you need predictable, repeatable behaviour and don't need LLM flexibility.

### 4. Graph-Based with Conditional Edges (`benchmarks_graph_based.py`)
**Strategy**: The original approach — a manually defined `StateGraph` with `add_conditional_edges` routing based on `state.status`. Fully deterministic, no LLM required for routing.
- **Pros**: Most predictable; no LLM calls needed for routing decisions.
- **Cons**: Rigid; requires manual maintenance if the workflow changes.
- **Key difference from heuristic-supervisor**: Even simpler — no supervisor layer, just pure graph edges.

### 5. Structured Output + State Machine (`benchmarks_structured_output.py`)
**Strategy**: The LLM emits structured JSON (`{"next_step": "...", "reasoning": "..."}`), which a deterministic controller validates against a state machine of valid transitions. Invalid transitions are rejected and the LLM is asked to reconsider.
- **Pros**: Combines LLM flexibility (the LLM decides *what* to do) with guaranteed termination (the controller enforces *when* it can be done).
- **Cons**: More complex to implement; adds a parsing/validation layer.
- **Key concept**: The controller maintains a transition matrix that defines which steps are allowed from each state.

```
Decision Matrix:
    init   → [list]
    list   → [select]
    select → [confirm]
    confirm → [run, select]  (can go back if cancelled)
    run    → [report]
    report → [done]
    done   → []
```

## Architecture

```
models_benchmarking/
├── README.md                              ← This file
├── cli.py                                 ← CLI entry point with --method selector
├── benchmarks_common.py                   ← Shared tools, data structures, state
├── benchmarks_prompt_driven.py            ← Approach 1: Prompt-driven ReAct
├── benchmarks_tool_driven.py             ← Approach 2: Tool-driven ReAct
├── benchmarks_heuristic_supervisor.py     ← Approach 3: Heuristic supervisor
├── benchmarks_graph_based.py             ← Approach 4: Graph-based conditional edges
└── benchmarks_structured_output.py        ← Approach 5: Structured output + state machine
```

### Common Components (`benchmarks_common.py`)

All approaches share:
- **`BenchmarkResult`** — dataclass storing latency metrics with computed properties (mean, median, P95, P99, throughput).
- **`BenchmarkState`** — mutable container for workflow state (available models, selected models, results, config).
- **5 LangChain `@tool` functions** — `list_ollama_models_tool`, `ask_user_to_select_models_tool`, `confirm_selection_tool`, `run_benchmarks_tool`, `report_results_tool`. Each checks preconditions and returns error messages if called in the wrong order.
- **`configure_state()`** / **`reset_state()`** — helpers to initialise or reset the global state.

### CLI (`cli.py`)

- `--method` — selects which approach to use (default: `prompt-driven`).
- `--iterations` — inference iterations per model (default: 30).
- `--warmup` — warmup iterations (default: 3).
- `--max-tokens` — max tokens per inference (default: 128).
- `--agent-model` — Ollama model for the driving LLM (default: `llama3.2`).

## Usage

```bash
# Default (prompt-driven ReAct with llama3.2)
python cli.py

# Try a different approach
python cli.py --method tool-driven
python cli.py --method structured-output
python cli.py --method graph-based
python cli.py --method heuristic-supervisor

# Custom configuration
python cli.py --method prompt-driven --agent-model mistral --iterations 20 --warmup 2 --max-tokens 64
```

## Loop Prevention & Termination

All approaches include safeguards against infinite loops:

| Mechanism | Description |
|---|---|
| **Recursion limit** | LangGraph's default 25-step limit aborts if exceeded |
| **Tool-level preconditions** | Each tool checks prerequisites and returns error messages |
| **Cancellation handling** | Tools return clear messages on user cancellation |
| **Prompt guidance** | System prompt explicitly lists the 5 steps |
| **LLM temperature=0** | Deterministic output reduces random deviations |

## Checkpointing

The checkpointer (`MemorySaver`) stores snapshots of the graph state at each step, enabling:
- **Pause**: When a tool calls `interrupt()`, the graph yields control and saves state.
- **Resume**: When `Command(resume=...)` is sent, the checkpoint is restored and execution continues.

**Alternatives to `MemorySaver`:**
- `SqliteSaver` — persists to a local SQLite file (survives restarts).
- `PostgresSaver` — stores in PostgreSQL (distributed deployments).
- `RedisSaver` — uses Redis (fast, shared storage across workers).