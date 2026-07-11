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

- `rag_pipeline.ipynb`: end-to-end notebook with code
- `doc/NVDA_report.pdf`: sample document used in the notebook
- `vector_db/`: persistent local ChromaDB output created by the notebook

## How to use it

1. Open the notebook and run the setup cells.
2. Load the PDF document.
3. Build the ChromaDB index in `vector_db/`.
4. Ask a question and inspect the retrieved context.

## Design notes

The notebook defaults to Ollama so it can run locally. If you want to use GitHub-hosted models, switch the embedding and chat provider configuration in the notebook and provide the required environment variables.