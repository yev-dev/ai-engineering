# Agentic Systems Overview

This tutorial explains agentic systems and orchestration in practical terms, with executable Python examples in the notebook.

## Learning goals

By the end, you should be able to explain and prototype:

- Agentic systems vs single-call prompting
- Orchestration and control loops
- Tool and function calling patterns
- Structured outputs for reliable machine-readable decisions
- Common agent architectures
- Context window management strategies
- Memory layers (short-term and long-term)
- Token optimization techniques
- Hallucination risks and mitigation patterns

---

## 1. Agentic Systems and Orchestration

### What Makes a System "Agentic"?

A single LLM call takes a prompt and returns a response. An **agentic system** goes further:

1. It **observes** its environment (user input, tool results, previous state).
2. It **reasons** about what to do next (which tool to call, what to respond).
3. It **acts** by calling tools, retrieving data, or generating output.
4. It **loops** until a termination condition is met.

```python
# Single-call prompting (NOT agentic)
response = llm.invoke("What is AAPL's P/E ratio?")

# Agentic loop (simplified)
state = {"task": "What is AAPL's P/E ratio?", "history": []}
while not state.get("done"):
    action = llm_reason(state)       # "I need to call get_stock_info(AAPL)"
    result = execute_tool(action)     # → {"pe_ratio": 28.3}
    state["history"].append(result)
    state["done"] = check_completion(state)
```

### Orchestration Patterns

Orchestration is the "brain" that decides the order and selection of actions. Three common patterns:

| Pattern | Description | When to Use |
|---------|-------------|-------------|
| **Sequential** | Step A → Step B → Step C in a fixed order | Data pipelines, ETL, report generation |
| **Conditional** | Branch based on intermediate results | QA systems, classification routers |
| **Loop with reasoning** | Observe → Think → Act → Repeat | Complex multi-tool tasks, research agents |

### Concrete Example: Sequential vs. Agentic Orchestration

```python
# === Sequential orchestration (LangChain Chain) ===
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial analyst."),
    ("human", "Fetch {ticker} price, analyze, and summarize."),
])
chain = prompt | llm  # single pass, no tool access

# === Agentic orchestration (ReAct loop) ===
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool

@tool
def get_stock_price(ticker: str) -> float:
    """Fetch the current stock price for a ticker."""
    # In production: yfinance, API call, etc.
    return 150.0

tools = [get_stock_price]
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# The agent will: reason → call get_stock_price → process result → respond
result = executor.invoke({"input": "What is AAPL's current price and is it a buy?"})
```

### Control Loops in Detail

The ReAct (Reasoning + Acting) loop is the most common agentic pattern:

```
1. Thought:   "I need the current stock price to answer this."
2. Action:    get_stock_price(ticker="AAPL")
3. Observation: {"price": 150.0, "change": "+2.3%"}
4. Thought:   "The price is $150, up 2.3%. I can now answer."
5. Final Answer: "AAPL is trading at $150.00, up 2.3% today."
```

```python
# Minimal ReAct loop (no framework)
def react_loop(query: str, tools: dict, max_steps: int = 5):
    messages = [{"role": "user", "content": query}]
    for step in range(max_steps):
        response = llm.invoke(messages)
        content = response.content

        if "FINAL ANSWER:" in content:
            return content.split("FINAL ANSWER:")[-1].strip()

        # Parse tool call from response (e.g. "ACTION: get_price(AAPL)")
        action = parse_action(content)
        if action and action["name"] in tools:
            result = tools[action["name"]](**action["args"])
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "tool", "content": str(result)})

    return "Max steps reached."
```

### AutoGen: Agentic Orchestration

AutoGen uses a **conversation-driven** orchestration model. Agents communicate via messages, and the runtime automatically manages the turn-taking and termination.

```python
import autogen
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent

# Define tools as functions registered with agents
def get_stock_price(ticker: str) -> float:
    """Fetch the current stock price for a ticker."""
    return 150.0

def get_pe_ratio(ticker: str) -> float:
    """Get the trailing P/E ratio."""
    return 28.5

# Create agents with tool registration
assistant = AssistantAgent(
    name="financial_analyst",
    llm_config={
        "config_list": [{"model": "llama3.1", "base_url": "http://localhost:11434"}],
        "temperature": 0,
    },
    system_message="You are a financial analyst. Use tools to answer questions.",
)

user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    code_execution_config=False,
    function_map={
        "get_stock_price": get_stock_price,
        "get_pe_ratio": get_pe_ratio,
    },
)

# AutoGen handles the orchestration loop automatically:
# 1. UserProxy sends the query
# 2. Assistant reasons and requests tool calls
# 3. UserProxy executes the tool and returns results
# 4. Loop continues until Assistant produces a final answer
result = user_proxy.initiate_chat(
    assistant,
    message="What is AAPL's current price and P/E ratio?",
    max_turns=5,
)
```

**Key difference from LangChain:** AutoGen's orchestration is implicit — agents negotiate turns automatically. LangChain's orchestration is explicit — you define the loop structure yourself.

### LangGraph: Agentic Orchestration

LangGraph models orchestration as a **state graph** where nodes are steps and edges define transitions. This gives you full control over the loop structure.

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain.tools import tool

# Define the state that flows through the graph
class AgentState(TypedDict):
    messages: list
    next_step: str

@tool
def get_stock_price(ticker: str) -> float:
    """Fetch the current stock price."""
    return 150.0

llm = ChatOllama(model="llama3.1", temperature=0).bind_tools([get_stock_price])

# Define graph nodes
def call_model(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(response)
    # Decide next step based on whether tool was called
    if response.tool_calls:
        state["next_step"] = "tools"
    else:
        state["next_step"] = "end"
    return state

def execute_tools(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    for tool_call in last_message.tool_calls:
        result = get_stock_price(**tool_call["args"])
        state["messages"].append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    state["next_step"] = "continue"
    return state

def should_continue(state: AgentState) -> Literal["tools", "end", "continue"]:
    return state["next_step"]

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", execute_tools)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "end": END,
})
graph.add_edge("tools", "agent")

app = graph.compile()

# Run the agent
result = app.invoke({
    "messages": [HumanMessage(content="What is AAPL's current price?")],
    "next_step": "continue",
})
```

**Key difference from AutoGen:** LangGraph gives you explicit graph-based control over the orchestration flow. You define exactly when to call the model, when to execute tools, and how to route between them.

---

## 2. Tool and Function Calling

Tools are the interface between an agent and the outside world. Every tool has:

- A **name** (unique identifier)
- A **description** (what it does, when to use it)
- **Parameters** (typed inputs)
- An **implementation** (the actual code)

### Defining Tools

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Option A: Decorator-based
@tool
def get_pe_ratio(ticker: str) -> float:
    """Get the trailing P/E ratio for a stock ticker."""
    # ... API call ...
    return 28.5

# Option B: Pydantic schema (for structured tool definitions)
class FinancialMetricInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. AAPL")
    metric: str = Field(description="Metric name: pe_ratio, market_cap, dividend_yield")

@tool(args_schema=FinancialMetricInput)
def get_financial_metric(ticker: str, metric: str) -> float:
    """Retrieve a specific financial metric for a given ticker."""
    # ... API call ...
    return 28.5
```

### Tool Calling in Practice

Modern LLMs (GPT-4, Claude, Llama 3.1) natively support tool/function calling:

```python
# Native tool calling via the LLM's API
from openai import OpenAI

client = OpenAI()

tools = [{
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get the current stock price for a ticker",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker e.g. AAPL"}
            },
            "required": ["ticker"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's AAPL trading at?"}],
    tools=tools,
)

# The response may contain a tool call
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    # Execute the tool, then feed the result back to the model
```

### With Ollama (local models that support tool calling)

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1", temperature=0)
llm_with_tools = llm.bind_tools([get_stock_price, get_pe_ratio])

response = llm_with_tools.invoke("What is AAPL's P/E ratio?")
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Calling {tool_call['name']} with {tool_call['args']}")
```

### AutoGen: Tool Registration

In AutoGen, tools are registered as a `function_map` on the `UserProxyAgent`. The assistant requests a tool call, and the UserProxy executes it and returns the result automatically.

```python
import autogen
from autogen import AssistantAgent, UserProxyAgent

# Define standalone functions (not decorated)
def get_stock_price(ticker: str) -> dict:
    """Simulate fetching stock price."""
    prices = {"AAPL": 150.0, "NVDA": 880.0, "MSFT": 420.0}
    return {"ticker": ticker, "price": prices.get(ticker, 0)}

def get_financial_metric(ticker: str, metric: str) -> float:
    """Simulate fetching a financial metric."""
    data = {"AAPL": {"pe_ratio": 28.5, "market_cap": 2.8e12},
            "NVDA": {"pe_ratio": 75.0, "market_cap": 2.2e12}}
    return data.get(ticker, {}).get(metric, 0)

# Register tools via function_map
user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    code_execution_config=False,
    function_map={
        "get_stock_price": get_stock_price,
        "get_financial_metric": get_financial_metric,
    },
)

# The assistant needs to know about the tools via its system message
assistant = AssistantAgent(
    name="analyst",
    llm_config={"config_list": [{"model": "llama3.1", "base_url": "http://localhost:11434"}]},
    system_message=(
        "You have access to these tools:\n"
        "  get_stock_price(ticker): returns price dict\n"
        "  get_financial_metric(ticker, metric): returns float\n"
        "To use a tool, respond with:\n"
        '  {"tool": "get_stock_price", "args": {"ticker": "AAPL"}}\n'
        "Then the user will execute it and return the result."
    ),
)

# AutoGen automatically routes tool requests between agents
result = user_proxy.initiate_chat(
    assistant,
    message="What is AAPL's P/E ratio and stock price?",
    max_turns=3,
)
```

**Key difference:** LangChain binds tools to the LLM via `bind_tools()` and the LLM natively outputs tool calls. AutoGen uses a `function_map` dictionary and a string-based protocol in the system message to describe available tools.

### LangGraph: Tool Calling with State

In LangGraph, tools are called as part of a graph node. The graph state carries messages, and each tool call produces a `ToolMessage` that gets fed back into the graph.

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END, add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain.tools import tool
from typing import Annotated

# Define tools
@tool
def get_stock_price(ticker: str) -> dict:
    """Get current stock price."""
    prices = {"AAPL": 150.0, "NVDA": 880.0}
    return {"ticker": ticker, "price": prices.get(ticker, 0)}

@tool
def get_financial_metric(ticker: str, metric: str) -> float:
    """Get a financial metric for a ticker."""
    data = {"AAPL": {"pe_ratio": 28.5}, "NVDA": {"pe_ratio": 75.0}}
    return data.get(ticker, {}).get(metric, 0)

tools_by_name = {"get_stock_price": get_stock_price, "get_financial_metric": get_financial_metric}
llm = ChatOllama(model="llama3.1", temperature=0).bind_tools([get_stock_price, get_financial_metric])

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def call_model(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def call_tools(state: AgentState) -> dict:
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for tc in tool_calls:
        tool_fn = tools_by_name[tc["name"]]
        result = tool_fn.invoke(tc["args"])
        results

Different problems require different architectures. Here are the four most common patterns:

### 4a. ReAct Agent (Reasoning + Acting)

The default pattern for single-agent systems. The agent loops: **Think** → **Act** → **Observe** → **Repeat**.

```
                 ┌─────────────────┐
                 │   User Query    │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │   Thought:      │
                 │ "I need price"  │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │   Action:       │
                 │ get_price(AAPL) │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │   Observation   │
                 │   $150.00       │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │  Final Answer   │
                 └─────────────────┘
```

```python
from langchain.agents import create_react_agent, AgentExecutor

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, max_iterations=5)
```

**Best for:** Single-tool chains, question answering, data retrieval.

### 4b. Plan-and-Execute Agent

The agent first creates a plan, then executes each step. This separates **planning** (expensive, done once) from **execution** (cheap, can retry individual steps).

```python
from langchain.chains.plan_and_execute import (
    PlanAndExecute, load_chat_planner, load_agent_executor
)

planner = load_chat_planner(llm)  # Creates step-by-step plan
executor = load_agent_executor(llm, tools, verbose=True)

agent = PlanAndExecute(planner=planner, executor=executor)
agent.run("Analyze NVDA's Q3 earnings and compare to AMD")
```

**Best for:** Complex multi-step tasks, research, report generation.

### 4c. Supervisor / Router Agent

One agent (the supervisor) decides which specialist sub-agent to call.

```python
class Supervisor:
    """Routes queries to specialist sub-agents."""
    def route(self, query: str) -> str:
        intent = classifier(query)  # e.g. "price", "news", "fundamentals"
        specialist = self.specialists[intent]
        return specialist.run(query)

supervisor = Supervisor()
supervisor.register("price", PriceAgent())
supervisor.register("news", NewsAgent())
supervisor.register("fundamentals", FundamentalsAgent())
```

**Best for:** Systems with multiple clearly-defined domains.

### 4d. Multi-Agent with Conversation

Agents talk to each other to reason, debate, and converge on an answer.

```
User Query
    │
    ▼
┌──────────────┐     ┌──────────────┐
│  Researcher  │◄───►│  Analyst     │
│  (gathers    │     │  (evaluates) │
│   facts)     │     │              │
└──────────────┘     └──────────────┘
    │                      │
    └──────────┬───────────┘
               ▼
       ┌──────────────┐
       │  Synthesizer │
       │  (final      │
       │   answer)    │
       └──────────────┘
```

```python
# Simplified two-agent debate
def debate(topic: str, rounds: int = 3):
    pro_agent = ChatOllama(model="llama3.1", system="You argue FOR the proposition.")
    con_agent = ChatOllama(model="llama3.1", system="You argue AGAINST the proposition.")

    pro_args = ""
    con_args = ""
    for _ in range(rounds):
        pro_args = pro_agent.invoke(f"Refute this: {con_args}" if con_args else topic)
        con_args = con_agent.invoke(f"Refute this: {pro_args}")

    # Synthesizer
    synthesizer = ChatOllama(model="llama3.1",
                             system="Synthesize both sides into a balanced conclusion.")
    return synthesizer.invoke(f"Pro: {pro_args}\n\nCon: {con_args}")
```

**Best for:** Reasoning validation, consensus, multi-perspective analysis.

### Architecture Selection Matrix

| Architecture | Complexity | Tools Needed | When to Choose |
|-------------|-----------|-------------|----------------|
| ReAct | Low | 1–5 | Simple Q&A, single domain |
| Plan-and-Execute | Medium | 2–10 | Multi-step research, reports |
| Supervisor | Medium | 3+ per domain | Multiple distinct domains |
| Multi-Agent | High | Varies | Debate, validation, complex reasoning |

---

## 5. Context Window Management

The LLM's context window is finite (typically 8k–200k tokens). Agents that run for many steps can overflow it. Management strategies:

### Strategy A: Sliding Window

Keep only the last N messages. Discard old history.

```python
def sliding_window(messages: list, max_messages: int = 20) -> list:
    """Keep system prompt + last N messages."""
    system = [m for m in messages if m["role"] == "system"]
    recent = [m for m in messages if m["role"] != "system"][-max_messages:]
    return system + recent
```

**Trade-off**: Simple but loses context of early steps. Good for short-lived tasks.

### Strategy B: Summarization

Periodically summarize the conversation into a compressed form.

```python
def compress_history(messages: list, llm, summary_threshold: int = 15) -> list:
    """Summarize old messages when they exceed a threshold."""
    if len(messages) < summary_threshold:
        return messages

    # Messages to summarize (everything except the last few)
    to_summarize = messages[:-5]
    recent = messages[-5:]

    summary = llm.invoke(
        f"Compress this conversation into a concise summary preserving key facts:\n{to_summarize}"
    )

    return [
        {"role": "system", "content": f"Previous conversation summary: {summary}"},
        *recent,
    ]
```

**Trade-off**: May lose nuance. Best for long-running agents.

### Strategy C: RAG-Based Retrieval

Store conversation history in a vector DB and retrieve relevant chunks.

```python
# On each step, embed and store the latest exchange
vector_store.add_texts(
    texts=[f"User: {user_msg}\nAssistant: {assistant_msg}"],
    metadatas=[{"timestamp": time.time(), "step": step_num}],
)

# Before the next LLM call, retrieve relevant history
relevant_history = vector_store.similarity_search(
    query=current_query, k=5
)
context = "\n".join([doc.page_content for doc in relevant_history])
```

**Trade-off**: More complex setup. Best for agents with long, diverse conversations.

### Strategy D: Token Budgeting

Set a hard token limit and truncate oldest/lowest-priority content first.

```python
def apply_token_budget(messages: list, max_tokens: int = 4000) -> list:
    """Truncate oldest non-system messages when over budget."""
    # Simplified: count tokens roughly by splitting on whitespace
    token_count = sum(len(m["content"].split()) for m in messages)

    if token_count <= max_tokens:
        return messages

    # Remove oldest non-system messages until under budget
    trimmed = [m for m in messages if m["role"] == "system"]
    for m in messages:
        if m["role"] != "system":
            candidate = trimmed + [m]
            if sum(len(x["content"].split()) for x in candidate) <= max_tokens:
                trimmed.append(m)

    return trimmed
```

### Strategy Comparison

| Strategy | Info Loss | Implementation | Latency | Best For |
|----------|-----------|---------------|---------|----------|
| Sliding Window | High | Trivial | None | Short tasks |
| Summarization | Medium | Moderate | Additional LLM call | Long-running agents |
| RAG Retrieval | Low | Complex | Embedding + search | Knowledge-heavy agents |
| Token Budgeting | Medium | Simple | None | Cost-sensitive production |

---

## 6. Memory in Agentic Systems

Agents need memory to maintain coherence across steps and sessions. Three layers:

### 6a. Short-Term Memory (Conversation History)

The recent interaction history within a single session. Typically managed by the context window.

```python
# LangChain's built-in memory
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
memory.chat_memory.add_user_message("What's AAPL's price?")
memory.chat_memory.add_ai_message("AAPL is trading at $150.")

# Retrieve for next call
history = memory.load_memory_variables({})
```

### 6b. Long-Term Memory (Vector Store / Database)

Persistent storage that survives across sessions. Agents retrieve relevant memories when needed.

```python
import chromadb

class LongTermMemory:
    def __init__(self, collection_name: str = "agent_memory"):
        self.client = chromadb.PersistentClient(path="./agent_memory_db")
        self.collection = self.client.get_or_create_collection(collection_name)

    def remember(self, fact: str, metadata: dict = None):
        """Store a fact into long-term memory."""
        self.collection.add(
            documents=[fact],
            metadatas=[metadata or {}],
            ids=[f"mem_{int(time.time())}"],
        )

    def recall(self, query: str, k: int = 5) -> list[str]:
        """Retrieve relevant past facts."""
        results = self.collection.query(query_texts=[query], n_results=k)
        return results["documents"][0] if results["documents"] else []

ltm = LongTermMemory()
ltm.remember("AAPL P/E ratio is 28.5", {"ticker": "AAPL", "type": "fundamental"})
ltm.remember("NVDA revenue grew 120% YoY", {"ticker": "NVDA", "type": "earnings"})

# Later, in a new session:
relevant = ltm.recall("What was NVDA's growth rate?")
# → ["NVDA revenue grew 120% YoY"]
```

### 6c. Episodic vs. Semantic Memory

| Memory Type | What It Stores | Example |
|-------------|---------------|---------|
| **Episodic** | Specific past events and interactions | "On 2024-01-15 the user asked about NVDA" |
| **Semantic** | Extracted knowledge and facts | "NVDA is an AI chip company" |
| **Procedural** | How to perform tasks | "To get a stock price, call get_stock_price()" |

```python
class EpisodicMemory:
    """Remembers specific past events with timestamps."""
    def store_episode(self, query: str, response: str, tools_used: list[str]):
        self.db.add(
            documents=[f"Q: {query}\nA: {response}"],
            metadatas=[{
                "timestamp": time.time(),
                "tools": ",".join(tools_used),
                "type": "episode",
            }],
        )

    def recall_similar(self, query: str, k: int = 3) -> list[dict]:
        """Find similar past episodes."""
        return self.db.query(query_texts=[query], n_results=k)

class SemanticMemory:
    """Stores extracted facts, not full conversations."""
    def store_fact(self, fact: str, confidence: float):
        self.db.add(
            documents=[fact],
            metadatas=[{"confidence": confidence, "type": "fact"}],
        )

    def query(self, topic: str) -> list[str]:
        return self.db.query(query_texts=[topic], n_results=5, where={"type": "fact"})
```

### Memory Selection Guide

| Scenario | Short-Term | Long-Term | Episodic | Semantic |
|----------|-----------|-----------|----------|----------|
| Single-turn Q&A | ✅ | ❌ | ❌ | ❌ |
| Multi-step reasoning | ✅ | ❌ | ❌ | ❌ |
| Cross-session user preferences | ❌ | ✅ | ✅ | ❌ |
| Knowledge accumulation | ❌ | ✅ | ❌ | ✅ |
| Learning from past mistakes | ❌ | ✅ | ✅ | ✅ |

---

## Files

- `agentic_systems.ipynb`: detailed walkthrough with code snippets and mini simulations

## Concrete implementations

- `langchain_financial_agent/`: end-to-end LangChain workflow using Ollama + yfinance
- `autogen_financial_agent/`: end-to-end AutoGen workflow using Ollama + yfinance

Both implementations take a ticker, question, and model name, then run:

1. orchestration
2. tool/function calling
3. structured output generation
4. context/token optimization
5. memory persistence
6. grounding diagnostics for hallucination control

## Framework Selection Guide: LangChain vs AutoGen

### **LangChain** — Use when you need:
- ✅ Sequential, deterministic data pipelines (fetch → process → validate → synthesize)
- ✅ Fine-grained control over each agent step and data transformation
- ✅ RAG systems with retrieval, ranking, and context optimization
- ✅ Single or dual-agent workflows with complex tool chains
- ✅ Strong auditability and regulatory compliance (finance, healthcare, legal)
- ✅ Easy token/cost control and caching strategies
- ✅ Gentle onboarding and simpler mental models

**Best for:** Data pipelines, RAG, sequential reasoning, production systems.

### **AutoGen** — Use when you need:
- ✅ Multiple specialized agents collaborating and coordinating
- ✅ Agents debating/validating each other's outputs to improve reasoning
- ✅ Automatic orchestration without explicit control code
- ✅ Role-based agents (researcher, analyzer, critic, synthesizer)
- ✅ Emergent behavior from agent conversations
- ✅ Complex multi-stage reasoning or consensus-driven decisions

**Best for:** Multi-agent collaboration, research systems, knowledge work, reasoning with multiple perspectives.

### **Quick Decision Matrix**

| Scenario | LangChain | AutoGen |
|----------|-----------|---------|
| Financial data fetch & analysis | ✅✅ Primary | ✅ Alternative |
| RAG document retrieval & ranking | ✅✅ Primary | ❌ Not designed for this |
| Single agent with many tools | ✅✅ Primary | ✅ Works, but overkill |
| Multiple agents collaborating | ✅ Possible | ✅✅ Primary |
| Agent consensus/debate | ⚠️ Manual coordination | ✅✅ Automatic |
| Regulated industries | ✅✅ Better audit trail | ✅ Acceptable |
| Cost control priority | ✅✅ Easier | ✅ Requires monitoring |
| Learning curve | ✅✅ Gentler | ⚠️ Steeper |

### **Our Implementation Patterns**

The **LangChain agent** in this tutorial demonstrates optimal data-driven workflows:
- Efficient tool orchestration with explicit step control
- Token budgeting and context ranking before LLM calls
- RAG pipeline: broad retrieval → rerank → minimal context selection
- Ideal for production financial systems

The **AutoGen agent** shows multi-agent coordination capabilities:
- Planner → Executor → Synthesizer role-based pattern
- Agents automatically negotiate and complete tasks
- Useful for exploring multi-perspective reasoning
- Good for learning AutoGen patterns

### **Recommendation: Start with LangChain**
Most production systems benefit from LangChain's explicit control and clarity. Adopt AutoGen when you need multi-agent collaboration that adds genuine value (reasoning validation, role specialization, consensus building).

---

## Embeddings in Agentic Systems

Embeddings are fundamental to modern agents. They transform text into dense numerical vectors that capture semantic meaning, enabling agents to:
- **Retrieve relevant context** from knowledge bases (RAG)
- **Rank and filter** candidate information based on relevance
- **Cache identical queries** using semantic similarity
- **Detect intent** and route tasks to appropriate tools
- **Measure hallucination** by grounding outputs against retrieved facts

### **What Are Embeddings?**

An embedding is a vector representation of text in continuous space where:
- Semantically similar texts are positioned close together
- Semantically different texts are far apart
- Distance metrics (cosine, Euclidean) measure similarity/relevance

**Example:** Queries "Apple's stock price" and "AAPL ticker value" are dissimilar as words but close in semantic space (~0.9 cosine similarity). This is why embeddings outperform keyword search.

### **How Embeddings Enable Agent Capabilities**

| Agent Task | Embedding Benefit | Example |
|-----------|------------------|---------|
| **Context Retrieval** | Find relevant documents from millions without scanning all | Query: "stock volatility" → retrieve only finance documents |
| **Query Deduplication** | Detect if question was asked before (cache hit) | Cache: "What's Apple's P/E?" + New: "Apple's price-to-earnings?" → 0.95 similarity → reuse answer |
| **Intent Classification** | Route to specialized tools without explicit rules | Embedding distance determines if query = price lookup, news search, or ratio analysis |
| **Ranking Evidence** | Order retrieved facts by relevance to question | Top 5 snippets ranked by semantic distance to user query |
| **Hallucination Detection** | Check if LLM answer is grounded in retrieved facts | Compare answer embedding to evidence embedding similarity |

### **Practical Python Snippet: Using Embeddings**

```python
from langchain.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize embeddings with local Ollama model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Embed some financial queries
queries = [
    "What is Apple's stock price?",
    "AAPL ticker current value",
    "Microsoft earnings report"
]

embedded_queries = embeddings.embed_documents(queries)

# Calculate similarity between first two (should be ~0.9)
similarity = cosine_similarity(
    [embedded_queries[0]], 
    [embedded_queries[1]]
)[0][0]
print(f"Query 1 ↔ Query 2 similarity: {similarity:.3f}")

# Store in vector database for retrieval
# vector_db = Chroma.from_documents(
#     documents=my_documents,
#     embedding=embeddings,
#     persist_directory="./vector_db"
# )

# Retrieve relevant context
# retrieved = vector_db.similarity_search_with_score(
#     query=queries[0], k=5
# )
```

### **Semantic Caching with Embeddings**

Semantic caching avoids re-computing answers for semantically similar questions:

```python
def semantic_cache_check(query: str, cache_store, threshold: float = 0.85):
    """Check if a semantically similar query was answered before."""
    cached_questions = cache_store.get_all_questions()  # returns (question, answer, embedding)
    query_emb = embeddings.embed_query(query)

    for cached_q, cached_answer, cached_emb in cached_questions:
        similarity = cosine_similarity([query_emb], [cached_emb])[0][0]
        if similarity > threshold:
            return cached_answer  # Cache hit
    return None  # Cache miss
```

### **Hallucination Detection with Embeddings**

Grounding check using embedding similarity:

```python
def is_grounded(answer: str, evidence: str, threshold: float = 0.72) -> bool:
    """Check if LLM answer is supported by retrieved evidence."""
    answer_emb = embeddings.embed_query(answer)
    evidence_emb = embeddings.embed_query(evidence)
    similarity = cosine_similarity([answer_emb], [evidence_emb])[0][0]
    return similarity > threshold

# Usage
answer = "Apple's P/E ratio is 28.5"
evidence = "AAPL P/E: 28.3-28.7 range per recent filings"
if is_grounded(answer, evidence):
    print("✅ Answer is grounded in evidence")
else:
    print("⚠️ Answer may hallucinate — refetch context")
```

---

## Suggested flow

1. Read the conceptual sections first (Agentic Systems → Tool Calling → Structured Outputs → Architectures → Context Management → Memory).
2. Run the code cells in the notebook in order.
3. Modify the prompts and policies to observe behavior changes.
4. Extend the examples with your own domain tools.