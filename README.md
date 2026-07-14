# INFOBOT — InfoBeans RAG Chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions about InfoBeans Technologies by grounding an LLM's responses in a local FAISS index built from your own documents.

**Status:** working demo, containerized and ready to run. Not yet hardened for multi-user production use — see [docs/01-PROJECT_SUMMARY.md](docs/01-PROJECT_SUMMARY.md) for current limitations.

![INFOBOT chat UI](image.png)

## Overview

Large language models don't know anything about your private documents, and asking them to answer from memory alone invites hallucination. INFOBOT solves that by ingesting local documents (PDF, Word, Excel, CSV, TXT, JSON) from a `data/` folder, splitting and embedding them locally into a FAISS vector index, and — at query time — retrieving the most relevant chunks and passing them as context to a Groq-hosted LLM, which answers using only that retrieved context. The whole flow is served through a Streamlit chat UI with source citations, so every answer stays traceable back to the original document, page, or row.

## How Data Flows: Frontend to Backend (No API Layer)

There is no network API between the frontend and the RAG pipeline in this project. `streamlit_app.py` and `src/search.py` run in the **same Python process** — Streamlit calls `RAGSearch` methods as ordinary in-process function calls, not HTTP requests. There is no client/server split, no JSON serialization boundary, and no localhost port between "frontend" and "backend." The only real network call in the whole flow is the outbound HTTPS request `ChatGroq` makes to Groq's cloud API to run the LLM — that's a call *out* to a third party, not a call between this app's own frontend and backend.

### 1. The function Streamlit calls

[streamlit_app.py:265](streamlit_app.py#L265) calls:

```python
RAGSearch.search_and_summarize_with_sources(query: str, top_k: int = 5, history: list[dict] = None) -> dict
```

defined in [src/search.py:107](src/search.py#L107). (A second, simpler method, `search_and_summarize(query: str, top_k: int = 5) -> str`, exists for the `app.py` CLI entry point and returns only the answer text, with no history support — the Streamlit UI does not use it.)

### 2. Exact input

- **`query`** — the raw user question string typed into `st.chat_input`, stripped of leading/trailing whitespace and validated to be non-empty and ≤ `MAX_QUERY_CHARS` (1000 chars) before the call is made ([streamlit_app.py:255-257](streamlit_app.py#L255-L257)).
- **`top_k`** — an int from 1–10 (default 5), set by a sidebar slider ([streamlit_app.py:107](streamlit_app.py#L107)) and passed straight through as the number of chunks to retrieve from FAISS.
- **`history`** — a capped, stripped list of prior `{"role", "content"}` turns, oldest first, built by `get_recent_history(st.session_state.messages)` ([streamlit_app.py:217](streamlit_app.py#L217)). It takes the last `MAX_HISTORY_EXCHANGES` (5) user/assistant exchanges and drops the `sources`/`usage` fields before passing them along. `None`/empty on the first turn of a conversation.

### 3. Exact output

`search_and_summarize_with_sources` returns a `dict` with four keys ([src/search.py:132-145](src/search.py#L132-L145)):

| Key | Type | Contents |
|---|---|---|
| `answer` | `str` | The LLM-generated answer, or the literal fallback string `"No relevant documents found."` if retrieval returned no usable chunks (in which case the LLM is never called). |
| `sources` | `list[dict]` | One entry per retrieved chunk: the chunk's loader metadata (`text`, `source`, and `page` or `row` depending on file type) plus a `distance` key (`float`, raw FAISS L2 distance — lower is more similar). Empty list when no context was found. |
| `usage` | `dict` | `input_tokens`, `output_tokens`, `total_tokens` (all `int`), pulled from Groq's response metadata. Includes tokens spent condensing the query, if that step ran. All zero when no LLM call was made. |
| `retrieval_query` | `str` | The query actually used for retrieval — equal to `query` when there's no history, or the standalone rewrite produced by query condensation when there is. |

### 4. Call path, end to end

1. User types a question into the `st.chat_input` box and submits it.
2. `main()` calls `handle_user_query(rag, prompt, top_k)` ([streamlit_app.py:310](streamlit_app.py#L310)).
3. `handle_user_query` builds `history` via `get_recent_history(st.session_state.messages)`, then calls `rag.search_and_summarize_with_sources(query, top_k=top_k, history=history)` directly — an in-process method call.
4. Inside `RAGSearch`: if `history` is non-empty, `_condense_query()` first makes a Groq call to rewrite `query` into a standalone question (resolving pronouns/implicit references, e.g. "What does it do?" → "What does the InfoBeans Foundation do?"); this step is skipped on turn 1. `self.vectorstore.query(retrieval_query, top_k=top_k)` then embeds the (possibly rewritten) query with the local `SentenceTransformer` and runs a FAISS `IndexFlatL2` search, returning the top-k chunks with their metadata and distances.
5. The retrieved chunk texts are joined into a context block. If the context is empty, `RAGSearch` returns the fallback dict immediately — no answer-generation LLM call (though a condensation call may already have happened). Otherwise it builds a prompt containing the resolved question, the context, and the conversation history (history is explicitly scoped in the prompt to interpreting the question only, not as a source of facts), and calls `self.llm.invoke([prompt])` on `ChatGroq`. This is the second Groq call for a follow-up turn, or the only one on turn 1 — both are outbound HTTPS requests to Groq's API, the only network calls in this whole flow.
6. `RAGSearch` returns the `answer` / `sources` / `usage` / `retrieval_query` dict back to `handle_user_query`, still as a plain in-memory Python object.
7. `handle_user_query` appends it to `st.session_state.messages` and persists the updated list to `chat_history.json`, then calls `st.rerun()`.
8. On rerun, `render_chat_history()` and `render_sources()` render the answer text and an expandable source-citation list (file name, page/row, a heuristic relevance percentage derived from the FAISS distance) directly from that same dict — no additional fetch is needed, because it was never "fetched" over a network in the first place.

**Latency/cost note:** because condensation and answer generation are separate Groq calls, a follow-up question (any turn after the first) makes two LLM calls instead of one. See [docs/03-TECH_STACK.md](docs/03-TECH_STACK.md) and [docs/05-TROUBLESHOOTING.md § Conversational memory](docs/05-TROUBLESHOOTING.md#conversational-memory).

**Future improvement:** if this project is later split into a genuine frontend/backend architecture (e.g. a separate API service behind Streamlit or another UI), this in-process call to `search_and_summarize_with_sources` would become an HTTP request/response instead, and the `dict` documented above would become the JSON response body. That is not the current design — today it's a direct function call within one process.

## Tech stack (summary)

Full detail, rationale, and swap-in alternatives for each of these live in [docs/03-TECH_STACK.md](docs/03-TECH_STACK.md).

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq API via `langchain-groq` (`llama-3.1-8b-instant`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`), local, CPU |
| Chunking | `langchain-text-splitters` `RecursiveCharacterTextSplitter` (1000 chars / 200 overlap) |
| Vector store | `faiss-cpu` (`IndexFlatL2`) + pickled chunk metadata |
| Document ingestion | PDF, TXT, CSV, XLSX, DOCX, JSON via `langchain-community` loaders |
| Deployment | Docker (multi-stage build) + Docker Compose |

## Quick start (Docker)

```bash
cp .env.example .env
# edit .env and set GROQ_API_KEY
docker compose up --build
```

Then open **http://localhost:8501**. See [docs/04-SETUP.md](docs/04-SETUP.md) for the full setup guide, including first-run index building, the manual (non-Docker) install path, and a verification checklist.

## Documentation

Read in this order:

1. [docs/FRAMEWORK_GUIDE.md](docs/FRAMEWORK_GUIDE.md) — what's reusable framework vs. what's specific to this InfoBeans instance, and how to start a new client project from this base.
2. [docs/01-PROJECT_SUMMARY.md](docs/01-PROJECT_SUMMARY.md) — what this is, the problem it solves, and how it works, in plain language.
3. [docs/02-ARCHITECTURE.md](docs/02-ARCHITECTURE.md) — pipeline and deployment diagrams, component responsibilities, known limitations. See also this README's [How Data Flows: Frontend to Backend (No API Layer)](#how-data-flows-frontend-to-backend-no-api-layer) for the exact frontend↔pipeline call path.
4. [docs/03-TECH_STACK.md](docs/03-TECH_STACK.md) — every technology choice in depth: what's used, why, alternatives, and known failure modes.
5. [docs/04-SETUP.md](docs/04-SETUP.md) — step-by-step setup (Docker primary, manual alternative), environment variables, and a verification checklist.
6. [docs/05-TROUBLESHOOTING.md](docs/05-TROUBLESHOOTING.md) — common failure symptoms and fixes, plus improvement areas identified during code review but not yet implemented.

## License

No license file is currently included in this repository. All rights reserved by default until a license is added.
