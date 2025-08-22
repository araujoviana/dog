import logging
import re
import time
from pathlib import Path
from groq import Groq

from utils.logging import DOG_LOGGER_NAME

log = logging.getLogger(f"{DOG_LOGGER_NAME}.{__name__}")

# TODO add options for locally running LLMs


class TextCleaner:
    """Encapsulates text cleaning using an LLM via the Groq API."""

    def __init__(self, auth_config: dict, cleanup_config: dict):
        """
        Initializes the Groq client and configures the cleaner.
        """
        try:
            self.api_key = auth_config["api_key"]
            self.model = cleanup_config["model"]
            self.audio_prompt = cleanup_config["audio_prompt"]
            self.text_prompt = cleanup_config["text_prompt"]
            self.cooldown = cleanup_config["api_call_cooldown"]
            self.chunk_size = cleanup_config["chunk_size"]

            if not self.api_key:
                raise ValueError("Groq API key is not set.")

            self.client = Groq(api_key=self.api_key)
            log.info(f"TextCleaner initialized for model: {self.model}")

        except (KeyError, ValueError) as e:
            log.critical(
                f"Failed to initialize TextCleaner due to configuration error: {e}"
            )
            raise

    def _clean_chunk_with_groq(self, text_chunk: str, prompt_template: str) -> str:
        """Sends a single chunk to the Groq API for cleaning."""
        full_prompt = prompt_template.format(text_chunk=text_chunk)
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": full_prompt}],
                model=self.model,
                # Theres no necessity to let the user
                # configure the settings below in this stage
                temperature=0.2,
                top_p=0.9,
            )
            cleaned_text = completion.choices[0].message.content.strip()
            return cleaned_text if cleaned_text else text_chunk
        except Exception as e:
            log.error(
                f"Groq API error during cleaning: {e}. Falling back to original text."
            )
            return text_chunk

    def _process_file(self, input_path: Path, output_path: Path, prompt_template: str):
        """Reads, chunks, cleans, and writes a single text file."""
        with input_path.open("r", encoding="utf-8") as f:
            text = f.read()

        sentences = re.split(r"(?<=[.!?])\s+", text)  # Ugly but whatever
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 < self.chunk_size:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())

        log.info(f"Processing '{input_path.name}' split into {len(chunks)} chunks.")

        with output_path.open("w", encoding="utf-8") as f_out:
            f_out.write(f"CLEANED: true | SOURCE_FILE: {input_path.name}\n\n")
            for i, chunk in enumerate(chunks):
                log.debug(
                    f"Cleaning chunk {i + 1}/{len(chunks)} from '{input_path.name}'"
                )
                cleaned_chunk = self._clean_chunk_with_groq(chunk, prompt_template)
                f_out.write(cleaned_chunk + " ")
                f_out.flush()
                time.sleep(self.cooldown)

    def process_files(self, file_paths: list[Path], output_dir: Path, file_type: str):
        """
        Cleans a list of text files using the appropriate prompt.

        Args:
            file_paths (list[Path]): List of file paths to process.
            output_dir (Path): Directory to save cleaned files.
            file_type (str): 'audio' or 'text' to select the correct prompt.
        """
        if file_type == "audio":
            prompt = self.audio_prompt
        elif file_type == "text":
            prompt = self.text_prompt
        else:
            log.error(
                f"Invalid file_type '{file_type}' provided. Use 'audio' or 'text'."
            )
            return

        log.info(
            f"Starting cleaning process for {len(file_paths)} files of type '{file_type}'."
        )
        for path in file_paths:
            output_file = output_dir / f"{path.stem}-cleaned.txt"
            if output_file.exists():
                log.info(f"Skipping already cleaned file: {output_file}")
                continue

            try:
                self._process_file(path, output_file, prompt)
            except Exception as e:
                log.error(f"Failed to process file {path}: {e}", exc_info=True)
        log.info(f"Finished cleaning process for type '{file_type}'.")
