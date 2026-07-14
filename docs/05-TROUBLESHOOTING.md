# Troubleshooting

Failure symptoms encountered while building this project, grouped by area, plus improvement areas identified during code review but not yet implemented. Every fix below has already been applied in this codebase — see [03-TECH_STACK.md](03-TECH_STACK.md) and [04-SETUP.md](04-SETUP.md) for the resulting configuration.

## Resolved Issues

### Environment & Python version

**Symptom:** `import torch` (and anything that depends on it, including `from sentence_transformers import SentenceTransformer`) fails with `OSError: [WinError 1114] Error loading c10.dll` or a generic `DLL initialization routine failed`.

**Cause:** The virtual environment was created with Python 3.14. PyTorch, Sentence-Transformers, FAISS, and several LangChain components are validated mainly against Python 3.11 — 3.14 support isn't there yet, and the failure surfaces as a DLL load error rather than a clear version-mismatch message. Sentence-Transformers fails for the same underlying reason, since it imports Torch internally.

**Fix:** Recreate the virtual environment on Python 3.11 (`py -3.11 -m venv .venv`), confirm with `python --version`, then install `torch==2.5.1` and verify with `python -c "import torch; print(torch.__version__)"` (expected: `2.5.1+cpu`). Sentence-Transformers then imports without any further changes. Avoid Python 3.14 for this project until every dependency officially supports it.

**Symptom:** Scripts, the integrated terminal, VS Code, and Jupyter notebooks behave inconsistently — a script works from one terminal but fails from another, or a notebook can't import a package that's clearly installed.

**Cause:** More than one Python installation is present on the machine (the project `.venv` on 3.11, plus a system-level 3.14), and the terminal, VS Code interpreter, or Jupyter kernel weren't all pointed at the same `.venv`.

**Fix:** Always activate `.venv` before running anything, and confirm with `where python` — the first result should resolve inside the project's `.venv\Scripts\python.exe`. In VS Code, explicitly select **Python 3.11 (.venv)** as the interpreter rather than leaving it on the system default. For notebooks, confirm the kernel is set to the same `.venv` before debugging import errors.

### Dependencies

**Symptom:** `ImportError` on `from langchain.text_splitter import RecursiveCharacterTextSplitter`, as shown in older tutorials.

**Cause:** LangChain moved text splitters into a separate package in newer releases.

**Fix:** Use `from langchain_text_splitters import RecursiveCharacterTextSplitter` instead.

**Symptom:** Dependency installation pulls in incompatible or broken package versions.

**Cause:** Copying a `requirements.txt` from an older tutorial without checking it against current package releases.

**Fix:** Pin a verified, working set of packages instead (`langchain`, `langchain-core`, `langchain-community`, `langchain-text-splitters`, `sentence-transformers`, `chromadb`, `faiss-cpu`, `numpy`, `scikit-learn`, `pypdf`, `pymupdf`, `langchain-groq`, `python-dotenv`) rather than blindly reusing an older requirements file.

### LLM / Groq API

**Symptom:** Groq API calls fail with `model_decommissioned`.

**Cause:** The originally configured model, `gemma2-9b-it`, is no longer supported by Groq.

**Fix:** Switch to a currently supported model (`llama-3.1-8b-instant` in this project). Check Groq's model documentation before hardcoding a model name, since availability changes over time.

**Symptom:** API credentials risk being exposed in source control.

**Cause:** The Groq API key was initially hardcoded directly in Python source (`groq_api_key = "gsk_..."`).

**Fix:** Load the key from the environment instead — `load_dotenv()` followed by `groq_api_key = os.getenv("GROQ_API_KEY")` — with the actual key stored only in a local `.env` file that isn't committed.

**Symptom:** The chatbot answers confidently even when the uploaded documents contain no relevant information (e.g. responding "No information found... however, the attention mechanism is..." instead of stopping at "not found").

**Cause:** The prompt didn't constrain the LLM to the retrieved context, so it fell back on its own pretrained knowledge — a hallucination.

**Fix:** Constrain the prompt to answer only from retrieved context and explicitly instruct a fallback response when the answer isn't present. The current wording, in the answer-generation prompt inside `search_and_summarize_with_sources()` ([src/search.py:176-186](../src/search.py#L176-L186)), is: "Answer the current question using only the provided context... If the context does not contain the answer, say so." This was one of the most impactful changes for keeping answers grounded. (Retrieval itself already short-circuits to the literal fallback string `"No relevant documents found."` when no chunks are retrieved at all — see [src/search.py:162-169](../src/search.py#L162-L169) — so the prompt-level instruction only matters when context was retrieved but doesn't contain the answer.)

### Vector store (FAISS)

**Symptom:** Retrieved results can't be traced back to source text, or the index appears to be missing after being deleted.

**Cause:** A raw FAISS index only stores vectors and returns numeric vector IDs on search — it has no concept of the original chunk text, file, or page. That reconstruction lives entirely in `metadata.pkl`, a separate pickled file kept index-aligned with the FAISS vectors.

**Fix:** Treat `faiss.index` and `metadata.pkl` as a single unit — both are required to answer queries, and neither should be deleted independently of the other unless you intend to rebuild the index from scratch.

**Symptom:** Every run re-embeds all documents from scratch, even when nothing changed, making startup slow.

**Cause:** `build_from_documents(docs)` was being called unconditionally on every run instead of reusing a previously built index.

**Fix:** Check whether `faiss.index` and `metadata.pkl` already exist and load them if so, only building from documents when they don't — this is the pattern used in `RAGSearch.__init__` ([src/search.py:19](../src/search.py#L19)).

### Expected behavior (not bugs)

**Symptom:** Console warning: "Unauthenticated requests" when downloading the embedding model.

**Cause:** No Hugging Face token is configured. This is informational, not an error — it only means slower downloads and lower rate limits, not a broken setup.

**Fix:** No action required. Optionally set `HF_TOKEN` in `.env` for faster downloads and higher rate limits.

**Symptom:** "Loading weights..." printed to the console on first run.

**Cause:** Sentence-Transformers is downloading the embedding model for the first time.

**Fix:** No action required — this is expected on a cold cache. Subsequent runs use the cached copy and skip the download.

### Conversational memory

**Symptom:** Follow-up questions that rely on a pronoun or implicit reference to the previous turn (e.g. "What does it do?" after asking about the InfoBeans Foundation) either retrieved irrelevant chunks or produced a hedging answer ("it is not explicitly defined...") even when the correct chunk was available.

**Cause:** `streamlit_app.py` already kept a `st.session_state.messages` chat history for display, but `handle_user_query()` called `RAGSearch.search_and_summarize_with_sources(query, top_k=top_k)` with only the current query string — history never reached retrieval or the LLM. Passing the raw pronoun query straight into FAISS similarity search retrieved the wrong chunks; passing the raw pronoun query into the final answer prompt (even alongside history) made the small 8B model unstable, since it was doing pronoun resolution and grounding-compliance in the same call.

**Fix:** Two changes, both required — history-only-in-the-final-prompt was tried and doesn't work on its own:

1. `RAGSearch._condense_query()` in [src/search.py:52](../src/search.py#L52) makes a small extra LLM call that rewrites a follow-up like "What does it do?" into a standalone question ("What does the InfoBeans Foundation do?") before it hits the vector store. It's skipped entirely on turn 1, when there's no history yet.
2. `search_and_summarize_with_sources()` ([src/search.py:107](../src/search.py#L107)) now accepts a `history` argument and uses the already-resolved query (not the raw one) in the final answer prompt, alongside the history — this fixed answers to 4/4 stable and correct in isolated testing, vs. intermittent hedging before. The prompt makes the grounding constraint explicit: history is only for interpreting the question and "is not a source of facts and must not be used to answer."

On the `streamlit_app.py` side, `get_recent_history()` ([streamlit_app.py:217](../streamlit_app.py#L217)) caps history to `MAX_HISTORY_EXCHANGES = 5` (the last 10 messages, i.e. 5 user/assistant pairs) and strips `sources`/`usage` before it's sent to the LLM. "Clear conversation" needed no changes — it already reset the single `st.session_state.messages` store that history now reuses.

Verified via the real `RAGSearch` code path (`faiss_store` + Groq): asking "What is the InfoBeans Foundation?" followed by "What does it do?" correctly resolved the retrieval query to "What does the InfoBeans Foundation do?" (not the more frequent "InfoBeans Technologies" entity) and returned a grounded answer, repeated across 4 isolated attempts and 2 full end-to-end runs with no hedging after the fix.

## Identified Improvement Areas

Found during code review; not yet implemented.

**Observation:** The FAISS index is only rebuilt when `faiss.index`/`metadata.pkl` are entirely missing ([src/search.py:19](../src/search.py#L19)). There's no mechanism to detect that source documents under `data/` have changed since the last build, so edits or additions to existing documents won't be reflected in the index without manually deleting it first.

**Planned Improvement:** Rebuild the index automatically when documents change, using a content hash or modification timestamp per file to decide whether re-ingestion is needed, rather than requiring a full manual rebuild every time.

**Observation:** Document ingestion happens only via the `data/` folder plus a manual `python app.py` run ([04-SETUP.md](04-SETUP.md)) — there's no way to add documents or trigger re-ingestion from the Streamlit UI itself.

**Planned Improvement:** Add PDF/document upload and an ingestion-progress indicator to the Streamlit UI, so documents can be added without a separate manual step outside the app.

**Observation:** There's currently no systematic way to measure retrieval quality or hallucination rate — verification has been manual and ad hoc.

**Planned Improvement:** Build a small evaluation set with questions both present and absent in the source documents, to measure retrieval accuracy and confirm the fallback (`"No relevant documents found."`) triggers correctly when it should.

**Observation:** Model names, chunk size/overlap, and `top_k` are hardcoded defaults in the Python source ([03-TECH_STACK.md](03-TECH_STACK.md)) rather than externally configurable.

**Planned Improvement:** Move these into a configuration file so they can be tuned per deployment without editing source code.

**Observation:** Some document chunks carry extra whitespace (repeated spaces, blank lines) inherited from source documents. This adds a small amount of unnecessary storage overhead during embedding without contributing any additional information.

**Planned Improvement:** Normalize whitespace during the chunking step before embedding, reducing storage footprint with no change to retrieval accuracy.

## Where to look first

This project now logs to `logs/infobot.log` (rotating, 5 MB × 3 backups — see [src/logger.py](../src/logger.py)), covering ingestion, retrieval queries, and LLM calls including full tracebacks on failure. Set `LOG_LEVEL=DEBUG` in `.env` for more detail. Most of the issues above had to be diagnosed manually before this logging existed — check the logs first before working through this document line by line.
