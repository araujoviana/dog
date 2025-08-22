import logging
from datetime import datetime
from pathlib import Path

DOG_LOGGER_NAME = "DOG_RAG"


def setup_logging(log_folder: Path, log_level: int = logging.INFO) -> None:
    """
    Configures a dedicated logger for the RAG pipeline.

    This function sets up a logger that writes to both a timestamped file
    and the console. It configures a named logger to avoid interfering with
    the root logger or third-party library loggers.
    """
    logger = logging.getLogger(DOG_LOGGER_NAME)
    logger.setLevel(log_level)

    # Prevent adding duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    exec_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = log_folder / f"rag_{exec_timestamp}.log"
    file_handler = logging.FileHandler(log_filename, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized. Log file at: {log_filename}")
