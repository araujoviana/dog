import logging
from sentence_transformers import CrossEncoder
from ml.vector import VectorStoreManager

from utils.logging import DOG_LOGGER_NAME

log = logging.getLogger(f"{DOG_LOGGER_NAME}.{__name__}")


class Retriever:
    """Handles the retrieval and reranking of documents."""

    def __init__(self, vector_store: VectorStoreManager, cross_encoder_name: str):
        self.vector_store = vector_store
        log.info(f"Loading CrossEncoder model: {cross_encoder_name}")
        self.cross_encoder = CrossEncoder(cross_encoder_name)

    def retrieve_and_rerank(
        self, query: str, initial_k: int, final_k: int
    ) -> list[str]:
        """
        Retrieves initial candidates from the vector store and reranks them.
        Returns the text of the top `final_k` results.
        """
        log.info(f"Performing initial search for {initial_k} candidates.")
        candidates = self.vector_store.search(query, k=initial_k)

        if not candidates:
            log.warning("No candidates found from initial search.")
            return []

        log.info("Reranking candidates with CrossEncoder...")
        sentence_pairs = [[query, cand["text"]] for cand in candidates]
        scores = self.cross_encoder.predict(sentence_pairs)

        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = score

        # Sort by the new rerank score in descending order
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        # Return the text of the top `final_k` documents
        top_results_text = [doc["text"] for doc in reranked[:final_k]]
        return top_results_text


def build_rag_prompt(
    query: str, retrieved_chunks: list[str], prompt_template: str
) -> str:
    """
    Constructs the final prompt for the RAG model.

    Args:
        query (str): The user's original question.
        retrieved_chunks (list[str]): The context retrieved from the vector store.
        prompt_template (str): The prompt template string from the config.

    Returns:
        str: The fully formatted prompt ready for the generation model.
    """
    context = "\n\n---\n\n".join(retrieved_chunks)
    return prompt_template.format(context=context, query=query)
