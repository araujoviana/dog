import tomllib
from pathlib import Path
from typing import Dict, Any
import logging

# Get a logger for this module
from utils.logging import DOG_LOGGER_NAME

log = logging.getLogger(f"{DOG_LOGGER_NAME}.{__name__}")


class AppConfig:
    """
    A container for application configuration that mirrors the config.toml structure.
    """

    def __init__(self, config_data: Dict[str, Any]):
        # --- Paths ---
        paths_config = config_data.get("paths", {})
        self.paths = {key: Path(value) for key, value in paths_config.items()}

        # --- Settings ---
        self.auth = config_data.get("auth", {})
        self.audio_settings = config_data.get("audio", {})
        self.ocr_settings = config_data.get("ocr_settings", {})
        self.cleanup_settings = config_data.get("clean_up", {})
        self.embedding_settings = config_data.get("embedding", {})
        self.retrieval_settings = config_data.get("retrieval", {})

    def setup_directories(self) -> None:
        """
        Creates all directories necessary for the pipeline.
        Input directories are assumed to exist (!!!).
        """
        output_keys = [
            "preprocessed_audio_folder",
            "output_folder",
            "log_folder",
            "transcription_folder",
            "cleaned_folder",
        ]
        for key in output_keys:
            if key in self.paths:
                self.paths[key].mkdir(parents=True, exist_ok=True)
            else:
                log.error(f"Warning: Output path key '{key}' not found in config.toml.")


def load_configuration(config_path: Path = Path("config.toml")) -> AppConfig:
    """Loads configuration from a TOML file and returns an AppConfig object."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with config_path.open("rb") as f:
        config_data = tomllib.load(f)

    return AppConfig(config_data)
