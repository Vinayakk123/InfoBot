# Framework Guide — Reusable RAG Base vs. Client Instance

This document explains how this codebase is meant to be reused. It does not describe how INFOBOT (the InfoBeans chatbot) works internally — for that, see [01-PROJECT_SUMMARY.md](01-PROJECT_SUMMARY.md) and [02-ARCHITECTURE.md](02-ARCHITECTURE.md). This document describes the boundary between the **framework** (reusable code) and **this particular client instance** (InfoBeans' branding, documents, and index).

## 1. What makes this a framework

This repository is a reusable retrieval-augmented-generation base: document ingestion, chunking, local embedding, FAISS indexing, and Groq-backed retrieval-QA, wired up behind a Streamlit chat UI. The INFOBOT chatbot in this repo is not a one-off — it is the **reference instance**, proving the base works end-to-end against a real client's documents. Every future client engagement should start by forking this base, swapping in the new client's documents and branding, and re-running ingestion — not by rewriting the pipeline.

## 2. Core vs. client-specific

| Path | Core or client-specific | What changes per client |
|---|---|---|
| `src/data_loader.py` | **Core** — reusable as-is | Nothing. Loaders are format-based (PDF/TXT/CSV/XLSX/DOCX/JSON), not content-aware. `data_dir` is already a function parameter. |
| `src/embedding.py` | **Core** — reusable as-is | Nothing structurally. `chunk_size`/`chunk_overlap`/`embedding_model` defaults are hardcoded (see §3 audit note) but the chunking logic itself is generic. |
| `src/vectorstore.py` | **Core** — reusable as-is | Nothing structurally. Same caveat: `persist_dir`, `embedding_model`, `chunk_size`, `chunk_overlap` are hardcoded defaults, not client logic. |
| `src/search.py` | **Core** — reusable as-is | The two prompt templates (`search_and_summarize`, `search_and_summarize_with_sources`) are domain-generic ("answer using only the provided context") — no InfoBeans references. `persist_dir`/`llm_model`/`embedding_model` defaults and a `data` fallback path are hardcoded (see §3). The `if __name__ == "__main__":` demo block has an InfoBeans-specific example query — harmless, but should be replaced when forking. |
| `src/logger.py` | **Core**, trivially rename | Fully generic logging setup. Only the output filename `infobot.log` (line 20) and the module docstring mention INFOBOT. |
| `data/pdf/`, `data/text_files/` | **Client-specific** | Entirely replaced. These are InfoBeans' actual source documents. |
| `data/vector_store*/` (Chroma artifacts) | **Client-specific / stale** | These appear to be earlier experiment artifacts (Chroma DB), not the vector store the app actually uses. Not part of the reusable base — do not carry these into a new client fork. |
| `faiss_store/` (`faiss.index`, `metadata.pkl`) | **Client-specific — generated output** | This is the InfoBeans index, built from InfoBeans documents via `sentence-transformers`. Delete and rebuild per client (§4). Never copy this directory into a new client's fork. |
| `chat_history.json` | **Client-specific — runtime data** | Contains real InfoBeans conversation history. Reset to empty (or delete) for a new client. |
| `logs/infobot.log` | **Client-specific — runtime data** | Reset per client; filename itself is hardcoded (see §3 audit note). |
| Prompt templates (in `src/search.py`) | **Core** — reusable as-is | Already domain-generic; no per-client edits needed unless a client wants different answer behavior (e.g. citation style, refusal wording). |
| `streamlit_app.py` (UI logic: chat loop, source rendering, history persistence, sidebar controls) | **Core** — reusable as-is | The mechanics (chat input, `st.chat_message`, source expander, top-k slider, history load/save) are generic. |
| `streamlit_app.py` (branding/copy: `APP_TITLE`, page title, sidebar "About" text, chat placeholder, caption, spinner text, error string) | **Client-specific** | All InfoBeans-branded strings — see the full list in §3. These are the first things to change for a new client. |
| `.env` / `.env.example` | **Mixed** | `GROQ_API_KEY` — client/deployment-specific secret, always replaced. `LOG_LEVEL` — reusable as-is (operational knob, not client data). |
| `requirements.txt` / `requirements-dev.txt` / `pyproject.toml` | **Core** — reusable as-is | Fully generic dependency stack (FAISS, LangChain, Groq, Streamlit, sentence-transformers). Only client coupling is the Groq-as-LLM-provider architectural choice, which is a framework decision, not a per-client one. |
| `Dockerfile`, `docker-compose.yml` | **Core**, minor rename | Pipeline/build logic is reusable. Service/image names (`infobot`, `infobot-rag-chatbot:local`) are client-flavored labels, not functional.|
| `docs/01-PROJECT_SUMMARY.md`, `docs/02-ARCHITECTURE.md`, `docs/03-TECH_STACK.md`, `docs/04-SETUP.md` | **Client-specific content, reusable structure** | Written specifically about INFOBOT/InfoBeans. Reuse the structure and explanations of *how the pipeline works*, but rewrite the client-facing framing (name, example queries, screenshots) per engagement. |
| `notebook/*.ipynb` | **Client-specific — dev artifacts** | Exploratory notebooks likely contain InfoBeans document content in cell outputs. Don't carry into a new client fork; recreate empty notebooks if needed. |
| `image.png` (README screenshot) | **Client-specific** | Screenshot of the InfoBeans-branded UI. Replace with a screenshot of the new client's UI, or drop it. |
| `main.py` | **Unused scaffold** | Currently just a placeholder stub, not part of the actual app (the real entrypoints are `app.py` and `streamlit_app.py`). Not client-specific, but not load-bearing either. |

## 3. Starting a new client project

**Step 1 — Fork the base.**
Copy/fork this repository into a new repo for the engagement (e.g. `rag-chatbot-<client>`). Do not fork from the InfoBeans *instance* if you can avoid it — fork from a point in history before InfoBeans-specific documents and generated artifacts were added, or strip them immediately after forking (step 2).

**Step 2 — Strip client-specific data and artifacts.**
Before touching any code, remove or replace:
- `data/pdf/`, `data/text_files/` → replace with the new client's source documents.
- `data/vector_store*/` → delete; these are stale experiment artifacts, not needed.
- `faiss_store/` → delete (`faiss.index`, `metadata.pkl`); it will be regenerated from the new documents.
- `chat_history.json` → delete or reset to an empty file.
- `logs/infobot.log` → delete; a fresh log will be created on next run.
- `notebook/*.ipynb` → delete or clear outputs if you don't need the InfoBeans exploration history.
- `image.png` → replace once the new UI is running, or remove the README reference until then.

**Step 3 — Replace branding and hardcoded client references.**
All found in the Step 1 audit — update these in `streamlit_app.py`:
- `APP_TITLE` (line 21) — page title / browser tab.
- Spinner text (`"Loading INFOBOT engine..."`, line 33).
- Fallback error string (line 64).
- Sidebar "About" subheader and body text (lines 128–132).
- Footer caption (line 294).
- Chat input placeholder (line 308).

And elsewhere:
- `src/logger.py` line 20 — rename `infobot.log` to a generic or client-specific log filename.
- `src/logger.py`, `streamlit_app.py`, `README.md`, `docs/*.md` — module docstrings and prose mentioning "INFOBOT"/"InfoBeans" — rewrite for the new client.
- Demo queries in `app.py`, `src/search.py`, `src/vectorstore.py` (`if __name__ == "__main__":` blocks) — cosmetic only, but replace the InfoBeans example query so it doesn't confuse future readers.
- `docker-compose.yml` / `Dockerfile` — rename the `infobot` service and `infobot-rag-chatbot:local` image tag.
- `pyproject.toml` — rename `rag-chatbot-ib` to match the new client.
- `.env` — set a fresh `GROQ_API_KEY` (never reuse a key across client deployments).

**Step 4 — Re-run ingestion to build a fresh vector store.**
With the new client's documents placed in `data/`, run:
```bash
python app.py
```
This calls `load_all_documents("data")` and `FaissVectorStore("faiss_store")`, which builds a new `faiss_store/faiss.index` + `metadata.pkl` from scratch. Alternatively, deleting `faiss_store/` and starting `streamlit_app.py` will trigger the same auto-build-on-first-run path inside `RAGSearch.__init__` (`src/search.py`). Confirm the index picked up the new documents by checking `logs/*.log` for ingestion counts, then smoke-test a query against the new content before handing off.

**Step 5 — What should NOT need to change.**
The pipeline code should require zero logic changes for a new client:
- `src/data_loader.py`, `src/embedding.py`, `src/vectorstore.py` — format-based, content-agnostic.
- `src/search.py` retrieval/prompt logic — the two prompt templates are already domain-generic.
- `streamlit_app.py` chat mechanics (message loop, source citation rendering, history persistence, top-k slider) — only the copy/branding around it changes, not the logic.
- `requirements.txt` / dependency stack.

**Found during audit — currently hardcoded but should ideally be config-driven (not fixed, flagged for your decision):**
- **No `config.py`/settings module exists.** `chunk_size` (1000), `chunk_overlap` (200), and `embedding_model` (`"all-MiniLM-L6-v2"`) are each independently redeclared as default parameters in *three separate files* (`src/embedding.py`, `src/vectorstore.py`, `src/search.py`), so changing one for a client means editing three places and keeping them in sync by hand.
- `llm_model` (`"llama-3.1-8b-instant"`) is hardcoded in **two** places independently: `src/search.py`'s default parameter and `streamlit_app.py`'s `LLM_MODEL` constant (the latter has a comment noting "This app only ever talks to one Groq model, per project requirements" — a client-specific constraint baked into code).
- `persist_dir` (`"faiss_store"`) and the ingestion source folder (`"data"`, hardcoded inline in `src/search.py`'s auto-build fallback and in `app.py`) are not environment-driven — a client wanting a different folder layout would need a code edit, not a config change.
- `top_k` default (5) is duplicated between `src/vectorstore.py`, `src/search.py` method defaults, and `streamlit_app.py`'s `DEFAULT_TOP_K` constant.
- `temperature` is never set at all (relies on the Groq API's implicit default) — a client wanting more/less deterministic answers currently has no way to tune this without a code edit.
- FAISS index type (`IndexFlatL2`) is hardcoded — fine for small corpora, but a client with a much larger document set might need a different index type, which today means editing `src/vectorstore.py` directly.
- Branding strings in `streamlit_app.py` (§3 above) are plain Python string literals, not sourced from `.env` or a config file — every new client currently requires a code edit (not just a config edit) to rebrand the UI.

Consolidating these into a single `config.py` (or `.env`-driven `Settings` object) would let a new client engagement be configured without touching pipeline code at all — currently it requires touching `src/embedding.py`, `src/vectorstore.py`, `src/search.py`, and `streamlit_app.py`.

## 4. Configuration points (already safe to change via `.env`, no code edit needed)

| Variable | Purpose | Default if unset |
|---|---|---|
| `GROQ_API_KEY` | Groq API credential used by `ChatGroq` in `src/search.py` | none — required |
| `LOG_LEVEL` | Log verbosity for the whole app, read in `src/logger.py` | `INFO` |

This is the entire externalized configuration surface today. Everything else listed in the audit note above (`chunk_size`, `chunk_overlap`, `embedding_model`, `llm_model`, `top_k`, `persist_dir`, `data` folder path, `temperature`, UI branding strings) is currently a Python literal, not a `.env`/config value.

## 5. Versioning note

As this base gets reused across multiple client engagements, keep a `CHANGELOG.md` at the repository root of the *framework* (not each client fork) documenting pipeline/infrastructure improvements as they're made — e.g. "added config.py for centralized settings," "swapped default embedding model," "fixed FAISS rebuild race condition." When a client engagement produces a genuine framework improvement (as opposed to that client's branding/documents), port the change back into the base and note it in the changelog with the originating client engagement for context. This lets future client forks start from a base that already carries forward fixes made for earlier clients, instead of re-discovering the same issues each time.
