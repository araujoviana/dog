from pathlib import Path
import argparse
import sys
import os
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from ml.llm import TextCleaner
from ml.vector import VectorStoreManager
from ml.retrieval import Retriever, build_rag_prompt
from ml.generation import AnswerGenerator
from processing.audio import AudioProcessor
from processing.text import TEXT_EXTENSIONS
from processing.docs import DocumentProcessor
from utils.logging import setup_logging, DOG_LOGGER_NAME
from utils.config import load_configuration

log = logging.getLogger(f"{DOG_LOGGER_NAME}.{__name__}")


class RAGPipeline:
    """
    Encapsulates the entire RAG pipeline from data processing to answer generation.
    """

    def __init__(self, config_path: str = "config.toml"):
        """
        Loads configuration and initializes all components of the pipeline.
        This is the expensive, one-time setup.
        """
        log.info("Initializing RAG Pipeline...")
        try:
            self.config = load_configuration(Path(config_path))
            self.config.setup_directories()
            setup_logging(log_folder=self.config.paths["log_folder"])

            # Initialize all processors and managers
            self.audio_processor = AudioProcessor(
                model_name=self.config.audio_settings["model"],
                device=self.config.audio_settings["device"],
            )
            self.doc_processor = DocumentProcessor(
                language=self.config.ocr_settings["language"]
            )
            self.text_cleaner = TextCleaner(
                auth_config=self.config.auth,
                cleanup_config=self.config.cleanup_settings,
            )
            self.vector_store = VectorStoreManager(
                embedding_model_name=self.config.embedding_settings["embedding_model"]
            )
            self.retriever = Retriever(
                vector_store=self.vector_store,
                cross_encoder_name=self.config.embedding_settings["cross_encoder"],
            )
            self.answer_generator = AnswerGenerator(
                auth_config=self.config.auth,
                retrieval_config=self.config.retrieval_settings,
            )
            log.info("RAG Pipeline initialized successfully.")
        except Exception as e:
            log.critical(f"Failed to initialize RAG Pipeline: {e}", exc_info=True)
            raise

    def process_data_sources(self):
        """
        Runs the full data ingestion and processing pipeline.
        This includes audio, documents, and text cleaning.
        """
        log.info("Starting data source processing...")
        # 1. Audio Processing
        if not self.config.audio_settings.get("ignore_processing", False):
            self.audio_processor.process_directory(
                input_dir=self.config.paths["audio_folder"],
                preprocessed_dir=self.config.paths["preprocessed_audio_folder"],
                transcription_dir=self.config.paths["transcription_folder"],
            )
        # 2. Document Processing
        self.doc_processor.process_directory(
            input_dir=self.config.paths["docs_folder"],
            output_dir=self.config.paths["cleaned_folder"],
        )
        # 3. Text Cleaning
        transcription_files = list(
            self.config.paths["transcription_folder"].glob("*.txt")
        )
        if transcription_files:
            self.text_cleaner.process_files(
                file_paths=transcription_files,
                output_dir=self.config.paths["cleaned_folder"],
                file_type="audio",
            )
        text_files = [
            f
            for ext in TEXT_EXTENSIONS
            for f in self.config.paths["docs_folder"].glob(ext)
        ]
        if text_files:
            self.text_cleaner.process_files(
                file_paths=text_files,
                output_dir=self.config.paths["cleaned_folder"],
                file_type="text",
            )
        log.info("Data source processing finished.")

    def build_knowledge_base(self):
        """Builds or rebuilds the vector store index from the processed text."""
        log.info("Building knowledge base vector store...")
        self.vector_store.build_index(
            data_dir=self.config.paths["cleaned_folder"],
            chunk_size=self.config.embedding_settings["chunk_size"],
            chunk_overlap=self.config.embedding_settings["overlap"],
        )
        log.info("Knowledge base built successfully.")

    def query(self, query_string: str) -> str:
        """
        Takes a user query, retrieves context, and generates a final answer.

        Args:
            query_string (str): The user's question.

        Returns:
            str: The generated answer, or an error message.
        """
        log.info(f"Received query: '{query_string}'")
        try:
            # 1. Retrieve context
            retrieved_context = self.retriever.retrieve_and_rerank(
                query=query_string,
                initial_k=self.config.embedding_settings["candidates_to_retrieve"],
                final_k=self.config.embedding_settings["final_chunks_to_use"],
            )
            if not retrieved_context:
                log.warning("No relevant context found for the query.")
                return (
                    "Error: Could not find relevant information to answer the question."
                )

            # 2. Build the prompt
            final_prompt = build_rag_prompt(
                query=query_string,
                retrieved_chunks=retrieved_context,
                prompt_template=self.config.retrieval_settings["prompt"],
            )

            # 3. Generate the answer
            final_answer = self.answer_generator.generate(prompt=final_prompt)
            if not final_answer:
                return "Error: The model failed to generate a final answer."

            log.info("Successfully generated answer.")
            return final_answer
        except Exception as e:
            log.error(f"An error occurred during the query process: {e}", exc_info=True)
            return "Error: A critical error occurred while processing the request."


def cli_runner():
    """
    A simple command-line interface to run the RAG pipeline.
    """
    try:
        pipeline = RAGPipeline()

        parser = argparse.ArgumentParser()
        parser.add_argument("--process", action="store_true")
        parser.add_argument("--build", action="store_true")

        args = parser.parse_args()
        if args.process:
            pipeline.process_data_sources()
        if args.build:
            pipeline.build_knowledge_base()

        user_query = input("Query: ")
        answer = pipeline.query(user_query)

        print("\n--- Generated Answer ---\n")
        print(answer)
        print("\n----------------------\n")

    except Exception as e:
        # Fallback for critical init failures
        print(f"Failed to start the application: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli_runner()
