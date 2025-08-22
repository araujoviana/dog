import logging
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from utils.logging import DOG_LOGGER_NAME

log = logging.getLogger(f"{DOG_LOGGER_NAME}.{__name__}")


class VectorStoreManager:
    """Manages the lifecycle of the document vector store using Faiss."""

    def __init__(self, embedding_model_name: str):
        log.info(f"Loading sentence transformer model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index = None
        self.documents = []
        self.metadata = []

    def _load_and_chunk(self, data_dir: Path, chunk_size: int, overlap: int):
        """Loads text from files and splits them into manageable chunks."""
        text_files = list(data_dir.glob("*.txt"))
        log.info(f"Found {len(text_files)} cleaned text files to process.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )

        for file_path in text_files:
            with file_path.open("r", encoding="utf-8") as f:
                # Skip the metadata header line we added during cleaning
                next(f, None)
                text = f.read()

            chunks = text_splitter.split_text(text)
            self.documents.extend(chunks)
            self.metadata.extend([{"source": file_path.name}] * len(chunks))

    def build_index(self, data_dir: Path, chunk_size: int, chunk_overlap: int):
        """Builds a Faiss IndexFlatL2 from the documents in the data directory."""
        self._load_and_chunk(data_dir, chunk_size, chunk_overlap)

        if not self.documents:
            log.warning("No documents found to build index. Vector store is empty.")
            return

        log.info(f"Embedding {len(self.documents)} document chunks...")
        embeddings = self.embedding_model.encode(
            self.documents, show_progress_bar=True, convert_to_numpy=True
        )

        log.info("Building Faiss IndexFlatL2...")
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings.astype(np.float32))
        log.info(f"Index built successfully with {self.index.ntotal} vectors.")

    def search(self, query: str, k: int) -> list[dict]:
        """Performs a similarity search on the index."""
        if not self.index or self.index.ntotal == 0:
            log.warning("Search attempted on an empty or non-existent index.")
            return []

        query_vector = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_vector.astype(np.float32), k)

        results = [
            {
                "text": self.documents[idx],
                "metadata": self.metadata[idx],
                "score": dist,
            }
            for dist, idx in zip(distances[0], indices[0])
        ]
        return results
