import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant"):
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

        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content

    def search_and_summarize_with_sources(self, query: str, top_k: int = 5) -> dict:
        """
        Retrieve relevant chunks for `query`, generate an LLM answer, and
        return both the answer and the raw source chunks used to produce it.

        Unlike `search_and_summarize`, this performs retrieval once and
        exposes the underlying chunk metadata (text, source file, page/row,
        FAISS L2 distance) so a caller such as the Streamlit UI can render
        citations alongside the answer. `search_and_summarize` is left
        unchanged so existing callers (e.g. app.py) keep working as-is.

        Args:
            query: User's natural-language question.
            top_k: Number of chunks to retrieve from the vector store.

        Returns:
            A dict with keys:
              "answer": the LLM-generated answer, or a fallback message if
                no relevant documents were found.
              "sources": one entry per retrieved chunk, each containing
                "text", "distance" (python float), and whatever metadata
                keys the loader attached (e.g. "source", "page", "row").
              "usage": dict with "input_tokens", "output_tokens", and
                "total_tokens" (all 0 if no LLM call was made, e.g. when no
                relevant context was found).
        """
        results = self.vectorstore.query(query, top_k=top_k)
        sources = []
        texts = []
        for r in results:
            meta = r.get("metadata")
            if not meta:
                continue
            texts.append(meta.get("text", ""))
            sources.append({**meta, "distance": float(r["distance"])})
        context = "\n\n".join(texts)
        empty_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if not context:
            return {"answer": "No relevant documents found.", "sources": [], "usage": empty_usage}
        prompt = f"""Answer the following question using only the provided context. If the context does not contain the answer, say so.\n\nQuestion: '{query}'\n\nContext:\n{context}\n\nAnswer:"""
        response = self.llm.invoke([prompt])
        usage = getattr(response, "usage_metadata", None) or {}
        return {
            "answer": response.content,
            "sources": sources,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "what is infobeans location?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)