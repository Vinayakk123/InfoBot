"""Streamlit chat frontend for INFOBOT, InfoBeans Technologies' RAG assistant.

This module is UI-only: it renders the chat interface and calls into the
existing RAG pipeline (src/search.py::RAGSearch) for retrieval + generation.
No retrieval or LLM logic is duplicated here.
"""

import json
import logging
import os

import streamlit as st
from dotenv import load_dotenv

from src.search import RAGSearch

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_TITLE = "INFOBOT — InfoBeans RAG Assistant"
MAX_QUERY_CHARS = 1000
DEFAULT_TOP_K = 5

# This app only ever talks to one Groq model, per project requirements.
LLM_MODEL = "llama-3.1-8b-instant"

# Conversation history is persisted here so it survives app restarts.
HISTORY_FILE = "chat_history.json"


@st.cache_resource(show_spinner="Loading INFOBOT engine...")
def load_rag_engine(llm_model: str) -> RAGSearch:
    """Instantiate (and cache) the RAG pipeline for a given LLM model.

    Args:
        llm_model: Groq model name to pass through to RAGSearch.

    Returns:
        A ready-to-query RAGSearch instance.

    Raises:
        RuntimeError: With a user-friendly message if initialization fails
            (missing GROQ_API_KEY, missing/corrupt FAISS index, etc.). The
            underlying exception is logged, not shown to the user. Raising
            here (instead of calling st.error/st.stop() inline) lets the
            cache_resource spinner clean up properly before the caller
            renders the error, avoiding an overlapping-UI glitch.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file and restart the app."
        )
    try:
        return RAGSearch(llm_model=llm_model)
    except Exception as exc:
        logger.exception("Failed to initialize RAG engine")
        if isinstance(exc, ModuleNotFoundError) and "data_loader" in str(exc):
            raise RuntimeError(
                "Vector store index not found or incomplete. Run `python app.py` "
                "from the project root to build faiss_store/, then restart this app."
            ) from exc
        raise RuntimeError(
            "Failed to initialize INFOBOT. Check server logs for details."
        ) from exc


def load_history() -> list[dict]:
    """Load persisted conversation history from disk, if any exists.

    Returns:
        The saved list of chat messages, or an empty list if no history
        file exists yet or it can't be parsed.
    """
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to load chat history, starting fresh")
        return []


def save_history(messages: list[dict]) -> None:
    """Persist the current conversation history to disk.

    Args:
        messages: The full st.session_state.messages list to save.
    """
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
    except OSError:
        logger.exception("Failed to save chat history")


def render_sidebar_controls() -> int:
    """Render the sidebar's top-k slider and clear-conversation button.

    Returns:
        The selected top-k (number of chunks to retrieve per question).
    """
    st.sidebar.header("Settings")
    st.sidebar.caption(f"Model: `{LLM_MODEL}`")
    top_k = st.sidebar.slider(
        "Chunks to retrieve (top-k)",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
        help="Number of document chunks retrieved from FAISS per question.",
    )
    if st.sidebar.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        save_history([])
        st.rerun()
    return top_k


def render_engine_info(rag: RAGSearch) -> None:
    """Render read-only pipeline info in the sidebar.

    Args:
        rag: The loaded RAGSearch instance to introspect.
    """
    st.sidebar.divider()
    st.sidebar.subheader("About INFOBOT")
    st.sidebar.markdown(
        "INFOBOT answers questions from InfoBeans Technologies' indexed "
        "documents using retrieval-augmented generation."
    )
    st.sidebar.markdown(
        f"- **Vector store:** FAISS\n"
        f"- **Embedding model:** `{rag.vectorstore.embedding_model}`\n"
        f"- **Indexed chunks:** {len(rag.vectorstore.metadata)}"
    )


def render_token_usage() -> None:
    """Render cumulative Groq token usage in the sidebar.

    Sums the "usage" data stored on every assistant message in the
    persisted conversation history. There's no way to query Groq for the
    account's actual quota/limit from this codebase, so this is a running
    total of consumption, not a "% remaining" against a real cap.
    """
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for msg in st.session_state.messages:
        usage = msg.get("usage")
        if not usage:
            continue
        for key in totals:
            totals[key] += usage.get(key, 0)

    st.sidebar.divider()
    st.sidebar.subheader("Token usage")
    st.sidebar.markdown(
        f"- **Prompt tokens:** {totals['input_tokens']:,}\n"
        f"- **Completion tokens:** {totals['output_tokens']:,}\n"
        f"- **Total tokens used:** {totals['total_tokens']:,}"
    )
    st.sidebar.caption(
        "Cumulative usage across saved conversation history. Groq doesn't "
        "expose your account's actual quota via the API, so this is not a "
        "\"% remaining\" figure."
    )


def get_relevance_label(distance: float) -> str:
    """Convert a FAISS L2 distance into a simple, human-readable relevance label.

    Args:
        distance: Raw L2 distance returned by FAISS (lower = more similar).

    Returns:
        A percentage-style heuristic relevance label. Not a calibrated
        probability, just a monotonic display cue.
    """
    score = 1 / (1 + distance)
    return f"{score:.0%} relevance (heuristic)"


def render_sources(sources: list[dict]) -> None:
    """Render an expandable list of source chunks under an assistant message.

    Args:
        sources: List of source dicts as returned by
            RAGSearch.search_and_summarize_with_sources (each has "text",
            "distance", and loader metadata like "source"/"page"/"row").
    """
    if not sources:
        return
    with st.expander(f"📄 View sources ({len(sources)})"):
        for i, src in enumerate(sources, start=1):
            filename = os.path.basename(src.get("source", "unknown"))
            location = ""
            if "page" in src and src["page"] is not None:
                location = f", page {int(src['page']) + 1}"
            elif "row" in src and src["row"] is not None:
                location = f", row {src['row']}"
            st.markdown(f"**Source {i} — {filename}{location}**")
            st.caption(get_relevance_label(src.get("distance", 0.0)))
            text = src.get("text", "")
            preview = text[:500] + "..." if len(text) > 500 else text
            st.text(preview)
            if i < len(sources):
                st.divider()


def init_session_state() -> None:
    """Initialize chat history in session state, loading any persisted history."""
    if "messages" not in st.session_state:
        st.session_state.messages = load_history()


def render_chat_history() -> None:
    """Replay all prior chat turns, including any stored source citations."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])


def handle_user_query(rag: RAGSearch, query: str, top_k: int) -> None:
    """Validate, run, and render a single user question end-to-end.

    Args:
        rag: The loaded RAGSearch instance to query.
        query: Raw text the user submitted via st.chat_input.
        top_k: Number of chunks to retrieve for this query.
    """
    query = query.strip()
    if not query:
        st.warning("Please enter a question.")
        return
    if len(query) > MAX_QUERY_CHARS:
        st.warning(f"Question is too long (max {MAX_QUERY_CHARS} characters).")
        return

    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Retrieving relevant documents and generating an answer..."):
        try:
            result = rag.search_and_summarize_with_sources(query, top_k=top_k)
        except Exception:
            logger.exception("RAG query failed")
            st.session_state.messages.append(
                {"role": "assistant", "content": "⚠️ Sorry, I ran into an error answering that question. Please try again."}
            )
            save_history(st.session_state.messages)
            st.rerun()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "usage": result.get("usage"),
        }
    )
    save_history(st.session_state.messages)
    # Rerun so the sidebar (token usage, chunk count) and chat history both
    # reflect this turn immediately, instead of lagging one interaction behind.
    st.rerun()


def main() -> None:
    """Entry point: wire up page config, sidebar, chat history, and input handling."""
    st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
    st.title(APP_TITLE)
    st.caption("InfoBeans Technologies — internal RAG assistant")

    init_session_state()
    top_k = render_sidebar_controls()
    try:
        rag = load_rag_engine(LLM_MODEL)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    render_engine_info(rag)
    render_token_usage()

    render_chat_history()

    prompt = st.chat_input("Ask INFOBOT a question about InfoBeans...")
    if prompt:
        handle_user_query(rag, prompt, top_k)


if __name__ == "__main__":
    main()
