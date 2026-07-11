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
from langchain.vectorstores import Chroma
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
print(f"Query 1 ↔ Query 2 similarity: {similarity:.3f}")  # Output: ~0.9

# Store in vector database for retrieval
vector_db = Chroma.from_documents(
    documents=my_documents,
    embedding=embeddings,
    persist_directory="./vector_db"
)

# Retrieve relevant context
retrieved = vector_db.similarity_search_with_score(
    query=queries[0],
    k=5,           # Top 5 results
    distance_threshold=0.15  # Relevance filter
)

for doc, score in retrieved:
    print(f"Relevance: {score:.3f} | Content: {doc.page_content[:50]}...")
```

### **Embedding Models for Different Use Cases**

| Model | Dimension | Latency | Best For |
|-------|-----------|---------|----------|
| `nomic-embed-text` | 768 | 50ms | General purpose, good balance |
| `mxbai-embed-large` | 1024 | 100ms | High-precision retrieval, larger docs |
| `all-minilm-l6-v2` (ONNX) | 384 | 10ms | Speed-critical, lightweight agents |
| `bge-base-en-v1.5` | 768 | 60ms | Semantic search, multilingual |

**Recommendation for agents:**
- **Development/learning:** `nomic-embed-text` (good quality, reasonable speed)
- **Production large-scale:** `mxbai-embed-large` (better retrieval precision)
- **Edge/mobile agents:** `all-minilm-l6-v2` (fast, small footprint)

### **Batch Embedding Best Practices**

```python
# ❌ Inefficient: embed one by one
for query in queries:
    embedding = embeddings.embed_query(query)  # Multiple round trips

# ✅ Efficient: batch embedding
embeddings_batch = embeddings.embed_documents(queries)  # Single call
```

**Token/Cost Impact:**
- Batch 100 queries: ~1 API call
- Sequential 100 queries: ~100 API calls
- **Savings:** 99x reduction in overhead

### **Semantic Caching with Embeddings**

One of our optimization strategies uses embeddings to avoid re-computing identical queries:

```python
# Check if question is similar to previous cached questions
new_question = "What's Apple's stock price?"
cached_questions = ["AAPL ticker value", "Apple stock cost"]

similarities = []
for cached_q in cached_questions:
    sim = cosine_similarity(
        embeddings.embed_query(new_question),
        embeddings.embed_query(cached_q)
    )
    similarities.append(sim)

if max(similarities) > 0.85:  # Threshold
    print("Cache hit! Reuse previous answer")
else:
    print("Cache miss. Query LLM.")
```

### **Indexing and Persistence**

Embeddings are expensive to compute but cheap to retrieve. Always persist:

```python
# Compute once and save
vector_db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./my_knowledge_base"
)

# Load cached embeddings (instant, no recomputation)
vector_db = Chroma(
    persist_directory="./my_knowledge_base",
    embedding_function=embeddings
)
```

**Why indexing matters:**
- Computing 100k embeddings: ~50 minutes
- Retrieving from index: <50ms per query
- **For agents:** Amortize compute cost over thousands of queries

### **Hallucination Detection with Embeddings**

Our grounding check uses embeddings to measure if LLM output is supported by retrieved evidence:

```python
# LLM generated answer
answer = "Apple's P/E ratio is 28.5"

# Retrieved evidence
evidence = "AAPL P/E: 28.3-28.7 range per recent filings"

# Compare semantic similarity
answer_embedding = embeddings.embed_query(answer)
evidence_embedding = embeddings.embed_query(evidence)

similarity = cosine_similarity(
    [answer_embedding],
    [evidence_embedding]
)[0][0]

if similarity > 0.72:
    print("✅ Answer is grounded in evidence")
else:
    print("⚠️ Answer may hallucinate — refetch context")
```

This is how we detect when agents fabricate facts not supported by retrieved data.

## Suggested flow

1. Read the conceptual sections first.
2. Run the code cells in order.
3. Modify the prompts and policies to observe behavior changes.
4. Extend the examples with your own domain tools.
