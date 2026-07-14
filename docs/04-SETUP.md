# Setup Guide

This guide assumes no live support is available — follow it top to bottom. The **Docker path is primary and recommended**; the manual path is for anyone who can't or doesn't want to use Docker.

## 1. Prerequisites

**Docker path (recommended):**
- Docker Engine with Compose v2 (`docker compose version` should work — bundled with Docker Desktop on Windows/Mac, or `docker-compose-plugin` on Linux)

**Manual path (alternative):**
- Python 3.11 specifically. Do **not** use the version implied by `.python-version` (`3.14`) — `requirements.txt` was verified against 3.11, and the Docker image itself builds on `python:3.11-slim`. Using 3.14 is untested and may fail to install some pinned dependencies.
- [`uv`](https://docs.astral.sh/uv/) (optional but recommended for fast, reproducible installs) — or plain `pip`/`venv`.

**Either path:** a [Groq API key](https://console.groq.com/keys) (free tier available).

## 2. PRIMARY path: Docker

From the project root (where `Dockerfile` and `docker-compose.yml` live):

```bash
# 1. Configure secrets — copy the template and fill in your real key
cp .env.example .env
# then edit .env and set GROQ_API_KEY

# 2. Build and start the container
docker compose up --build
```

This builds the image from `Dockerfile` (multi-stage: installs dependencies with `uv` into a venv, then copies just that venv plus `app.py`/`main.py`/`streamlit_app.py`/`src/` into a slim runtime image running as a non-root user) and starts the `infobot` service defined in `docker-compose.yml`, which:

- Reads secrets from `.env` (`env_file: .env`)
- Publishes the app on `http://localhost:8501`
- Mounts `./faiss_store` so the vector index persists across container restarts/rebuilds
- Mounts `./data` so you can add/edit source documents without rebuilding the image
- Mounts `./chat_history.json` so conversation history persists (see step 5 below — this file must exist on the host before first `up`)
- Uses a named volume (`hf-cache`) to cache the downloaded embedding model so it isn't re-fetched from Hugging Face on every container recreation

Once it's running, open **http://localhost:8501**.

To stop it: `docker compose down` (add `-v` only if you intentionally want to also delete the `hf-cache` volume and force a re-download of the embedding model next time).

## 3. ALTERNATIVE path: manual local setup (no Docker)

```bash
# 1. Clone/unzip and enter the project directory
cd "RAG chatbot IB"

# 2. Create and activate a Python 3.11 virtual environment
python3.11 -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install runtime dependencies — requirements.txt is the verified, working set
#    (do not use `uv sync` / uv.lock — they're out of sync with the actual code; see docs/03-TECH_STACK.md)
pip install -r requirements.txt

# 4. Configure secrets
cp .env.example .env
# then edit .env and set GROQ_API_KEY

# 5. Add documents to ingest under data/ (PDF/TXT/CSV/XLSX/DOCX/JSON)
#    Sample PDFs and .txt files are already included under data/pdf and data/text_files.

# 6. Build the FAISS index (creates faiss_store/)
python app.py

# 7. Launch the chatbot
streamlit run streamlit_app.py
```

Then open **http://localhost:8501**.

## 4. Environment variables

Every variable actually read via `os.getenv`/`dotenv` in the code, cross-referenced with [.env.example](../.env.example):

| Variable | Required | Default | Read in | Purpose |
|---|---|---|---|---|
| `GROQ_API_KEY` | **Yes** | none | [src/search.py:25](../src/search.py#L25), checked on startup in [streamlit_app.py:51](../streamlit_app.py#L51) | Groq API key used by `ChatGroq` to generate answers. Get one at [console.groq.com/keys](https://console.groq.com/keys). |
| `LOG_LEVEL` | No | `INFO` | [src/logger.py:35](../src/logger.py#L35) | Log verbosity for the whole app. Set to `DEBUG` for per-file ingestion detail, FAISS/embedding internals, etc. |

There are no other environment variables read anywhere in this codebase — the LLM model name, embedding model name, chunk size/overlap, and FAISS persist directory are all hardcoded defaults in the Python source, not environment-configurable (see [03-TECH_STACK.md](03-TECH_STACK.md) for exact locations).

## 5. First-run: building the vector store

The app needs a populated `faiss_store/` (a `faiss.index` file plus a `metadata.pkl` file) before it can answer questions.

- **Docker path:** the `faiss_store/` volume starts empty on a first-ever run. Currently the container only runs `streamlit_app.py`, which does **not** build the index itself — you must build it once before (or alongside) first use. The straightforward way: run `python app.py` locally against the same `data/` and `faiss_store/` paths that the compose volumes point at (so the container picks up the resulting `faiss_store/` on its next start/restart), or run it inside the container: `docker compose run --rm infobot python app.py`.
- **Manual path:** step 6 above (`python app.py`) does this for you.

`chat_history.json` must exist as a file (even if empty, e.g. `[]`) before the Docker container starts, since `docker-compose.yml` bind-mounts it as a single file rather than a directory — create it with `echo [] > chat_history.json` if it doesn't already exist.

## 6. Verification checklist

You'll know it worked if:

- [ ] `docker compose up --build` (or `streamlit run streamlit_app.py`) prints no Python traceback, and the terminal shows Streamlit's "You can now view your Streamlit app" message.
- [ ] Opening `http://localhost:8501` shows the **INFOBOT — InfoBeans RAG Assistant** page, not an error screen.
- [ ] If `GROQ_API_KEY` is missing, the app shows a clear on-screen error ("GROQ_API_KEY is not set...") rather than a blank page or crash — this confirms error handling is working, not that setup is complete. Fix `.env` and restart.
- [ ] If `faiss_store/` hasn't been built yet, the app shows an on-screen error pointing you to run `python app.py` first.
- [ ] The sidebar's **About INFOBOT** section shows `Vector store: FAISS`, an embedding model name, and a non-zero **Indexed chunks** count — a zero count means ingestion ran but found nothing.
- [ ] Asking a question found in the sample documents (e.g. "What is InfoBeans' location?") returns an answer with an expandable **View sources** section citing a real file and page/row number.
- [ ] `logs/infobot.log` exists and contains lines like `Groq LLM call succeeded in 0.7s (model=llama-3.1-8b-instant)` — confirms the LLM call path is working end-to-end.

## 7. If something fails

See [05-TROUBLESHOOTING.md](05-TROUBLESHOOTING.md) for common failure symptoms and fixes. Check `logs/infobot.log` first — every ingestion step, retrieval query, and LLM call is logged there, including full tracebacks on failure.
