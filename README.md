# RAG Training Repository

A simple Retrieval-Augmented Generation (RAG) chatbot built with Python, ChromaDB, and the OpenAI API. The system loads text documents, creates vector embeddings, and uses semantic search to provide contextually relevant answers to user questions.

## How it works

1. **Ingestion** (`ingestion.py`) — Loads `.txt` files from the `docs/` folder, splits them into chunks, generates embeddings via OpenAI and stores them in a local ChromaDB vector store.
2. **Retrieval** (`retrival.py`) — Given a user query, finds the most semantically similar chunks from the database.
3. **Chat** (`main.py`) — Interactive CLI chat loop that retrieves relevant context and passes it to GPT-4o-mini to generate an answer.

## Project structure

```
docs/          # Source documents (.txt)
db/            # ChromaDB vector store (auto-generated)
src/
  ingestion.py # Document loading, chunking & embedding
  retrival.py  # Vector similarity search
  user_chat.py # Chat and message classes
  main.py      # CLI entry point
```

## Setup

1. Create a `.env` file in the project root:

   ```
   OPENAI_API_KEY="sk-..."
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

3. Ingest your documents:

   ```bash
   uv run src/ingestion.py
   ```

4. Start the chatbot:
   ```bash
   uv run src/main.py
   ```

Type `exit`, `quit`, or `konec` to stop the chat.

## Tech stack

- [OpenAI API](https://platform.openai.com/) — embeddings (`text-embedding-3-small`) and chat (`gpt-4o-mini`)
- [ChromaDB](https://www.trychroma.com/) — local vector store
- [LangChain](https://www.langchain.com/) — document loading, splitting and retrieval
