import time
from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents
from src.logger import get_logger

logger = get_logger(__name__)

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name} (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        start = time.perf_counter()
        embeddings = self.model.encode(texts, show_progress_bar=True)
        elapsed = time.perf_counter() - start
        logger.info(f"Generated {len(texts)} embeddings in {elapsed:.2f}s (shape={embeddings.shape})")
        return embeddings

# Example usage
if __name__ == "__main__":

    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    logger.debug(f"Example embedding: {embeddings[0] if len(embeddings) > 0 else None}")