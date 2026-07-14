# RAG Pipeline Study Material

This lesson shows a practical Retrieval-Augmented Generation pipeline built with Python and ChromaDB.

## What is built

- A PDF document loader for the showcase file in `doc/NVDA_report.pdf`
- A text-splitting step with overlap so chunks keep local context
- A persistent ChromaDB vector database stored in `vector_db/`
- An embedding provider that works with local Ollama embeddings
- A GitHub-hosted embedding path via the OpenAI-compatible client when configured
- A prompt-based query flow that retrieves relevant chunks and sends them to a chat model

## Files

- `rag_pipeline.ipynb`: end-to-end notebook with ChromaDB code
- `rag_faiss.ipynb`: FAISS — multiple index types (Flat, IVF, HNSW), retrieval strategies, optimisation benchmarks, RAG with LLM
- `rag_chromadb.ipynb`: ChromaDB — dual-store design, metadata filtering, HNSW tuning, hybrid reranking, RAG with LLM
- `rag_llamaindex.ipynb`: LlamaIndex — unified API over FAISS/ChromaDB/Native backends, QueryFusionRetriever, cross-encoder reranking, RAG with LLM
- `doc/NVDA_report.pdf`: sample document used in the notebooks
- `vector_db/`: persistent local ChromaDB output created by `rag_pipeline.ipynb`
- `vector_db_faiss/`: FAISS indexes created by `rag_faiss.ipynb`
- `vector_db_chromadb/`: ChromaDB collections created by `rag_chromadb.ipynb`
- `vector_db_llamaindex/`: LlamaIndex persisted indexes (ChromaDB, FAISS, Native) created by `rag_llamaindex.ipynb`

## How to use it

1. Open the notebook and run the setup cells.
2. Load the PDF document.
3. Build the ChromaDB index in `vector_db/`.
4. Ask a question and inspect the retrieved context.

## Design notes

The notebook defaults to Ollama so it can run locally. If you want to use GitHub-hosted models, switch the embedding and chat provider configuration in the notebook and provide the required environment variables.

---

## LlamaIndex Integration

[LlamaIndex](https://www.llamaindex.ai/) is a data framework for building RAG applications. It provides a unified abstraction layer over document ingestion, chunking, embedding, indexing, and querying. Unlike working directly with a vector store library, LlamaIndex manages the entire pipeline — from parsing PDFs to constructing retrievers and query engines — with built-in support for dozens of vector stores, embedding models, and LLMs.

### Comparing Vector Storage Options: FAISS, ChromaDB, and LlamaIndex

| Feature | FAISS | ChromaDB | LlamaIndex (as framework) |
|---|---|---|---|
| **Role** | Low‑level vector index library | Purpose‑built vector database | High‑level RAG framework (can use FAISS or ChromaDB underneath) |
| **Persistence** | Writes index to a single `.faiss` file + optional `.pkl` for ID mapping | Persistent directory with SQLite + Parquet per collection | Delegates to the underlying vector store; also provides its own `StorageContext` |
| **Licence** | MIT | Apache 2.0 | MIT |
| **Index types** | Flat (brute‑force), IVF, HNSW, PQ, scalar quantisation | HNSW (default), configurable distance functions | Supports 30+ vector stores; index type depends on the store |
| **Query speed** | Fastest for large‑scale search (native C++ BLAS) | Good for mid‑scale (single‑node, up to millions of vectors) | Depends on underlying store; adds negligible overhead |
| **When to choose** | You need maximum performance, control over index parameters, and don't need a metadata store | You want a zero‑setup, persistent vector DB with built‑in metadata filtering | You want a complete RAG pipeline with ingestion, chunking, retrieval, and query-building out of the box |

**Recommendations by use case:**

- **Experimenting / local learning** → ChromaDB. Minimal config, persistent to disk, plays well with LangChain and LlamaIndex.
- **High‑performance production search** → FAISS directly (or FAISS via LangChain/LlamaIndex). Use HNSW or IVF‑PQ for sub‑50 ms query times on millions of vectors.
- **End‑to‑end RAG application** → LlamaIndex. It ties together document parsing, chunking, embedding, indexing, retrieval, and LLM calling in one consistent API. You swap the vector store underneath when scaling needs change.
- **Multi‑modal / structured data** → LlamaIndex, which natively supports images, tables, PDFs, code, and mixes of unstructured + structured indices.

### Embedding and Indexing with LlamaIndex

Below are Python snippets that show how to build embeddings and an index with LlamaIndex using the same local Ollama embedding model and NVIDIA report that the notebook uses.

```python
from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


# ------------------------------------------------------------
# 1.  Set the global embedding model (like the notebook does)
# ------------------------------------------------------------
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)


# ------------------------------------------------------------
# 2.  Load and chunk the PDF
# ------------------------------------------------------------
RAG_DIR = Path(".")  # adjust to your location
documents = SimpleDirectoryReader(
    input_dir=str(RAG_DIR / "doc"),
    required_exts=[".pdf"],
).load_data()

parser = SentenceSplitter(chunk_size=900, chunk_overlap=150)
nodes = parser.get_nodes_from_documents(documents)
print(f"Built {len(nodes)} nodes from {len(documents)} document(s)")


# ------------------------------------------------------------
# 3.  Build a persistent ChromaDB index
# ------------------------------------------------------------
VECTOR_DB_DIR = RAG_DIR / "vector_db_llamaindex"
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

db = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
chroma_collection = db.get_or_create_collection("nvda_report")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
)

# Optional: the same index can be persisted to disk as JSON for portability
# index.storage_context.persist(persist_dir=str(VECTOR_DB_DIR / "index_json"))


# ------------------------------------------------------------
# 4.  Query the index
# ------------------------------------------------------------
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query(
    "What are the key business highlights in this report?"
)
print(response)


# ------------------------------------------------------------
# 5.  Load the persisted index later (no re-embedding needed)
# ------------------------------------------------------------
from llama_index.core import load_index_from_storage

reloaded_index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir=str(VECTOR_DB_DIR))
)
reloaded_query_engine = reloaded_index.as_query_engine(similarity_top_k=3)
```

#### Using FAISS directly via LlamaIndex

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# Dimension must match your embedding model (nomic-embed-text = 768)
d = 768
faiss_index = faiss.IndexFlatL2(d)  # brute-force L2 search
vector_store = FaissVectorStore(faiss_index=faiss_index)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
)

# Persist both the FAISS index and metadata separately
index.storage_context.persist(persist_dir="./faiss_index_dir")

# Reload:
from llama_index.core import load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore

vector_store = FaissVectorStore.from_persist_dir("./faiss_index_dir")
storage_context = StorageContext.from_defaults(
    vector_store=vector_store, persist_dir="./faiss_index_dir"
)
reloaded_index = load_index_from_storage(storage_context)
```

### How Indexes Are Stored

Vector indexes are stored in dramatically different ways depending on the backend:

| Store | On‑disk format | Contents |
|---|---|---|
| **ChromaDB** | Directory containing `chroma.sqlite3` (metadata, collection config) + a `parquet/` subdirectory (embeddings as Apache Parquet columns) | One Parquet file per collection stores the float32 embeddings; SQLite holds document IDs, metadata, and collection settings |
| **FAISS** | Single `.faiss` flat file (binary, platform‑portable) | Contains the vector index structure (e.g. HNSW graph edges, IVF centroids, or the raw flat vectors) and the vectors themselves. The index is self‑contained. ID mapping is often stored separately in a `.pkl` pickle file |
| **LlamaIndex persistence** | Directory with `docstore.json`, `index_store.json`, `vector_store.json` | JSON files containing serialised document metadata, node relationships, and a reference to the underlying vector store. The actual embedding vectors live inside whichever vector‑store backend you chose (ChromaDB Parquet, FAISS `.faiss`, etc.) |

**Key principle**: embeddings are always stored **separately from the model**. The embedding model (e.g. `nomic-embed-text`) is loaded at runtime to vectorise new queries; the index only holds the resulting vectors. This separation means:
- The model weights are not duplicated per index.
- You can change the model later, but you must then re‑embed all documents.
- Moving an index between machines only requires copying the index files — you install the same embedding model on the target machine.

### How Indexing Works in Detail

Indexing is the process of transforming raw text into a searchable structure of vectors. The pipeline has four stages:

**1. Document ingestion and chunking**

Documents are loaded and split into smaller pieces (nodes/chunks). This is the only stage that touches the original text. Chunk boundaries matter because a query will retrieve whole chunks, not sub‑chunk fragments.

```text
PDF → page-level text → chunked into overlapping segments (e.g. 900 tokens, 150 overlap)
```

**2. Embedding**

Each chunk is passed through an embedding model to produce a dense vector — a fixed‑length list of floats (e.g. 768 for `nomic-embed-text`). The model is a transformer that maps semantically similar sentences to nearby points in the vector space.

```python
# Simplified: what the library does internally
chunk_text = "NVIDIA reported record data-center revenue..."
vector = embedding_model.encode(chunk_text)  # → np.array of shape (768,)
```

**3. Index construction**

The vectors are assembled into a data structure optimised for nearest‑neighbour search. Different index types make different trade-offs:

- **Flat (brute‑force)**: stores all vectors in a simple array. Query computes distance to every vector — O(N) time, 100 % accuracy. Fine for <100 k vectors.
- **IVF (Inverted File Index)**: clusters the vector space into Voronoi cells. At query time, only the nearest cells are searched. Fast (sub‑linear) at the cost of a small accuracy loss.
- **HNSW (Hierarchical Navigable Small World)**: builds a multi‑layer graph. Each layer is a sparser subsample of the vectors. Search starts at the top layer (few nodes, long jumps) and descends to finer layers. O(log N) typical performance. ChromaDB and FAISS both support HNSW.
- **PQ (Product Quantisation)**: compresses vectors into compact codes (e.g. 8 bytes instead of 768 × 4 = 3072 bytes). Used for very large datasets where the index must fit in RAM. Accuracy degrades with compression ratio.

**4. Index persistence**

The constructed index (graph edges, centroids, compressed codes, or plain vectors) is flushed to disk. On reload, the binary structure is memory‑mapped or fully deserialised so that queries can be served without re‑embedding the corpus.

**For a concrete analogy:** think of the raw embedding process as taking a photo of each document chunk — expensive, done once. The index is like the museum catalogue that lets you find the right photo in milliseconds without flipping through every picture.

### Query‑time flow (what happens when you ask a question)

1. The query text is embedded with **the same model** that was used for the documents.
2. The query vector is passed to the index's search method (e.g. HNSW traversal, IVF cell search, or brute‑force scan).
3. The index returns the top‑k nearest neighbour vectors and their associated chunk IDs.
4. The system looks up the original chunk text (stored in the metadata store) using those IDs.
5. The retrieved chunks are assembled into a context prompt and sent to an LLM.

The crucial insight is that **step 2 never re‑runs the embedding model** on the indexed documents — it only computes distances between the query vector and the stored vectors. That is what makes vector search fast.