# stdlib imports
import sys
import os
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Local imports
from ml.llm import TextCleaner
from ml.vector import VectorStoreManager
from ml.retrieval import Retriever, build_rag_prompt
from ml.generation import AnswerGenerator
from processing.audio import AudioProcessor
from processing.text import TEXT_EXTENSIONS
from processing.docs import DocumentProcessor
from utils.logging import setup_logging, DOG_LOGGER_NAME
from utils.config import load_configuration

# TODO Add a "ignore audio" flag
# TODO Fix docker stuff

# Gets a logger for this module
log = logging.getLogger(f"{DOG_LOGGER_NAME}.{__name__}")


def main():
    # TODO add frontend
    query = input("Query:")

    try:
        app_config = load_configuration()
        app_config.setup_directories()
    except FileNotFoundError as e:
        log.error(f"Error {e}")
        return 1

    # Sets up logging
    setup_logging(log_folder=app_config.paths["log_folder"])

    log.info("Starting dog 🐕")

    log.info("Starting audio processing pipeline.")

    try:
        if not app_config.audio_settings["ignore_processing"]:
            audio_processor = AudioProcessor(
                model_name=app_config.audio_settings["model"],
                device=app_config.audio_settings["device"],
            )

            audio_processor.process_directory(
                input_dir=app_config.paths["audio_folder"],
                preprocessed_dir=app_config.paths["preprocessed_audio_folder"],
                transcription_dir=app_config.paths["transcription_folder"],
            )
            log.info("Audio processing pipeline finished successfully.")
        else:
            log.warning(
                "Skipping audio processing, you can enable it again in the config.toml file"
            )
    except KeyError as e:
        log.error(
            f"Configuration key error: {e}. Check that the key exists in config.toml."
        )
        sys.exit(1)
    except Exception as e:
        log.error(
            f"An unexpected error occurred during audio processing: {e}", exc_info=True
        )
        sys.exit(1)

    log.info("Audio processing pipeline finished.")

    # Processes document files

    log.info("Starting document processing pipeline.")
    try:
        doc_processor = DocumentProcessor(language=app_config.ocr_settings["language"])

        doc_processor.process_directory(
            input_dir=app_config.paths["docs_folder"],
            output_dir=app_config.paths["cleaned_folder"],
        )
        log.info("Document processing pipeline finished successfully.")
    except KeyError as e:
        log.error(f"Configuration key error in [ocr_settings]: {e}.")
        sys.exit(1)
    except Exception as e:
        log.error(
            f"An unexpected error occurred during document processing: {e}",
            exc_info=True,
        )
        sys.exit(1)

    log.info("Initializing text cleaning pipeline.")
    try:
        text_cleaner = TextCleaner(
            auth_config=app_config.auth, cleanup_config=app_config.cleanup_settings
        )

        # Clean audio transcriptions
        transcription_files = list(
            app_config.paths["transcription_folder"].glob("*.txt")
        )
        if transcription_files:
            text_cleaner.process_files(
                file_paths=transcription_files,
                output_dir=app_config.paths["cleaned_folder"],
                file_type="audio",
            )

        # Clean raw text documents
        text_files = [
            f
            for ext in TEXT_EXTENSIONS
            for f in app_config.paths["docs_folder"].glob(ext)
        ]
        if text_files:
            text_cleaner.process_files(
                file_paths=text_files,
                output_dir=app_config.paths["cleaned_folder"],
                file_type="text",
            )

        log.info("Text cleaning pipeline finished.")
    except Exception as e:
        log.error(
            f"An unexpected error occurred during text cleaning: {e}", exc_info=True
        )
        sys.exit(1)

    # --- RAG stuff starts here ---

    log.info("Initializing RAG vector store and retriever.")
    try:
        # Initialize and build the vector store
        vector_store = VectorStoreManager(
            embedding_model_name=app_config.embedding_settings["embedding_model"]
        )
        vector_store.build_index(
            data_dir=app_config.paths["cleaned_folder"],
            chunk_size=app_config.embedding_settings["chunk_size"],
            chunk_overlap=app_config.embedding_settings["overlap"],
        )

        # Initialize the retriever
        retriever = Retriever(
            vector_store=vector_store,
            cross_encoder_name=app_config.embedding_settings["cross_encoder"],
        )

        # Execute the retrieval and reranking process
        retrieved_context = retriever.retrieve_and_rerank(
            query=query,
            initial_k=app_config.embedding_settings["candidates_to_retrieve"],
            final_k=app_config.embedding_settings["final_chunks_to_use"],
        )

        # Generate final prompt
        if retrieved_context:
            final_prompt = build_rag_prompt(
                query=query,
                retrieved_chunks=retrieved_context,
                prompt_template=app_config.retrieval_settings["prompt"],
            )
            log.info("Successfully generated final RAG prompt.")
            log.info(f"Final prompt -> {final_prompt}")
        else:
            log.warning(
                "Could not retrieve any context for the query. Cannot generate prompt."
            )

    except Exception as e:
        log.error(f"An error occurred during the RAG process: {e}", exc_info=True)
        sys.exit(1)

    # --- Retrieval starts here ---

    if final_prompt:
        log.info("Generating final answer from retrieved context.")
        try:
            # Instantiate and run the final generation component
            answer_generator = AnswerGenerator(
                auth_config=app_config.auth,
                retrieval_config=app_config.retrieval_settings,
            )
            final_answer = answer_generator.generate(prompt=final_prompt)

            if final_answer:
                print("\n--- Generated Answer ---\n")
                print(final_answer)
                print("\n----------------------\n")
            else:
                print("\nError: The model failed to generate a final answer.")
                log.error("Answer generation returned None, indicating an API failure.")

        except Exception as e:
            log.error(
                f"A critical error occurred during the answer generation stage: {e}",
                exc_info=True,
            )
            sys.exit(1)
    else:
        log.warning(
            "Could not retrieve any context for the query. Aborting answer generation."
        )

    log.info("Application finished.")


if __name__ == "__main__":
    main()
