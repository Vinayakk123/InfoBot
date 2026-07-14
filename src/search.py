import os
import time
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq
from src.logger import get_logger

logger = get_logger(__name__)

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant"):
        self.llm_model = llm_model
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.getenv("GROQ_API_KEY")

        self.llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=llm_model)

        # Never log groq_api_key or any ChatGroq config object here (may hold the key indirectly).
        logger.info(f"Groq LLM initialized: {llm_model}")

    def _format_history(self, history: list[dict]) -> str:
        """Render capped chat history as a plain-text transcript.

        Args:
            history: List of {"role", "content"} dicts, oldest first.

        Returns:
            A "User: ...\\nAssistant: ...\\n" transcript, or "(none)" if
            history is empty.
        """
        if not history:
            return "(none)"
        lines = []
        for msg in history:
            speaker = "User" if msg.get("role") == "user" else "Assistant"
            lines.append(f"{speaker}: {msg.get('content', '')}")
        return "\n".join(lines)

    def _condense_query(self, query: str, history: list[dict]) -> tuple[str, dict]:
        """Rewrite a follow-up question into a standalone question for retrieval.

        Pronoun/implicit-reference follow-ups (e.g. "who is he") embed
        poorly on their own, so FAISS similarity search over the raw query
        would retrieve irrelevant chunks. This resolves such references
        against the conversation history before retrieval happens.

        Args:
            query: The user's raw follow-up question.
            history: Capped list of prior {"role", "content"} turns.

        Returns:
            A (standalone_query, usage) tuple. Falls back to the original
            query if the LLM returns a blank rewrite.
        """
        history_text = self._format_history(history)
        prompt = f"""Given the conversation history and a follow-up question, rewrite the follow-up question as a standalone question that includes any context needed to understand it (e.g. resolve pronouns like "he", "it", "that" to what they refer to). Output only the rewritten question, nothing else. If the follow-up question is already standalone, return it unchanged.

Conversation history:
{history_text}

Follow-up question: {query}

Standalone question:"""
        logger.info(f"Calling Groq LLM (model={self.llm_model}) to condense follow-up question...")
        start = time.perf_counter()
        response = self.llm.invoke([prompt])
        elapsed = time.perf_counter() - start
        usage = getattr(response, "usage_metadata", None) or {}
        standalone = response.content.strip()
        if not standalone:
            standalone = query
        logger.info(f"Condensed query in {elapsed:.2f}s: '{query}' -> '{standalone}'")
        return standalone, {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            logger.info("No relevant documents found for query; skipping LLM call.")
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        logger.info(f"Calling Groq LLM (model={self.llm_model}) for summarization...")
        start = time.perf_counter()
        response = self.llm.invoke([prompt])
        elapsed = time.perf_counter() - start
        logger.info(f"Groq LLM call succeeded in {elapsed:.2f}s (model={self.llm_model})")
        return response.content

    def search_and_summarize_with_sources(self, query: str, top_k: int = 5, history: list[dict] = None) -> dict:
        """
        Retrieve relevant chunks for `query`, generate an LLM answer, and
        return both the answer and the raw source chunks used to produce it.

        Unlike `search_and_summarize`, this performs retrieval once and
        exposes the underlying chunk metadata (text, source file, page/row,
        FAISS L2 distance) so a caller such as the Streamlit UI can render
        citations alongside the answer. `search_and_summarize` is left
        unchanged so existing callers (e.g. app.py) keep working as-is.

        When `history` is provided, `query` is first rewritten into a
        standalone question (resolving pronouns/implicit references against
        history) before it's used for retrieval, and the trimmed history is
        also included in the final answer prompt so the LLM can interpret
        the original question. History is only ever used to interpret what
        is being asked, never as a source of facts — the answer must still
        come strictly from the retrieved context.

        Args:
            query: User's natural-language question.
            top_k: Number of chunks to retrieve from the vector store.
            history: Capped list of prior {"role", "content"} turns, oldest
                first. Pass None or [] for a fresh conversation.

        Returns:
            A dict with keys:
              "answer": the LLM-generated answer, or a fallback message if
                no relevant documents were found.
              "sources": one entry per retrieved chunk, each containing
                "text", "distance" (python float), and whatever metadata
                keys the loader attached (e.g. "source", "page", "row").
              "usage": dict with "input_tokens", "output_tokens", and
                "total_tokens" (all 0 if no LLM call was made, e.g. when no
                relevant context was found). Includes tokens spent on query
                condensation, if that step ran.
              "retrieval_query": the query actually used for retrieval
                (equal to `query` when no condensation was needed).
        """
        empty_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        condense_usage = empty_usage
        retrieval_query = query
        if history:
            retrieval_query, condense_usage = self._condense_query(query, history)

        results = self.vectorstore.query(retrieval_query, top_k=top_k)
        sources = []
        texts = []
        for r in results:
            meta = r.get("metadata")
            if not meta:
                continue
            texts.append(meta.get("text", ""))
            sources.append({**meta, "distance": float(r["distance"])})
        context = "\n\n".join(texts)
        if not context:
            logger.info("No relevant documents found for query; skipping LLM call.")
            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "usage": condense_usage,
                "retrieval_query": retrieval_query,
            }
        history_text = self._format_history(history) if history else "(none)"
        # Use retrieval_query (already resolved to a standalone question by
        # _condense_query when history is present) rather than the raw
        # `query`, so this call doesn't have to re-resolve pronouns itself
        # on top of applying the grounding constraint — asking a small model
        # to do both at once measurably destabilized answers in testing.
        prompt = f"""Answer the current question using only the provided context. Conversation history is included solely to help you understand what the current question refers to; it is not a source of facts and must not be used to answer. If the context does not contain the answer, say so.

Conversation history:
{history_text}

Current question: '{retrieval_query}'

Context:
{context}

Answer:"""
        logger.info(f"Calling Groq LLM (model={self.llm_model}) with {len(sources)} retrieved chunk(s)...")
        start = time.perf_counter()
        response = self.llm.invoke([prompt])
        elapsed = time.perf_counter() - start
        usage = getattr(response, "usage_metadata", None) or {}
        logger.info(
            f"Groq LLM call succeeded in {elapsed:.2f}s (model={self.llm_model}, "
            f"total_tokens={usage.get('total_tokens', 0)})"
        )
        total_usage = {
            "input_tokens": condense_usage["input_tokens"] + usage.get("input_tokens", 0),
            "output_tokens": condense_usage["output_tokens"] + usage.get("output_tokens", 0),
            "total_tokens": condense_usage["total_tokens"] + usage.get("total_tokens", 0),
        }
        return {
            "answer": response.content,
            "sources": sources,
            "usage": total_usage,
            "retrieval_query": retrieval_query,
        }

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "what is infobeans location?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    logger.info(f"Summary: {summary}")