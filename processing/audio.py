import logging
from pathlib import Path
import ffmpeg
from faster_whisper import WhisperModel

# Get a logger for this module
from utils.logging import DOG_LOGGER_NAME

log = logging.getLogger(f"{DOG_LOGGER_NAME}.{__name__}")

# Supported audio extensions
AUDIO_EXTENSIONS = ["*.mp3", "*.wav", "*.flac", "*.aac", "*.ogg", "*.m4a"]


class AudioProcessor:
    """Encapsulates audio preprocessing and transcription."""

    def __init__(
        self, model_name: str, device: str = "cpu", compute_type: str = "int8"
    ):
        """
        Initializes the processor and loads the faster-whisper model.
        """
        self.device = device
        self.model_name = model_name
        self.compute_type = compute_type  # New: Store compute_type

        log.info(
            f"Loading Whisper model '{self.model_name}' onto device '{self.device}' "
            f"with compute type '{self.compute_type}'."
        )
        try:
            # Changed: Use WhisperModel for loading
            self.model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
            log.info("faster-whisper model loaded successfully.")
        except Exception as e:
            log.error(f"Failed to load faster-whisper model: {e}")
            raise

    def _preprocess_ffmpeg(self, input_path: Path, output_path: Path) -> bool:
        """
        Normalizes and cleans an audio file using FFmpeg.

        Returns:
            bool: True if processing was successful or skipped, False on error.
        """
        if output_path.exists():
            log.warning(f"Preprocessed file already exists, skipping: {output_path}")
            return True

        log.info(f"Preprocessing audio file: {input_path}")
        try:
            (
                ffmpeg.input(str(input_path))
                .filter("volume", "6dB")
                .filter("highpass", f=80)
                .filter("loudnorm")
                .output(str(output_path), acodec="libopus")
                .run(overwrite_output=True, quiet=True)
            )
            return True
        except ffmpeg.Error as e:
            log.error(f"FFmpeg error on {input_path}: {e.stderr.decode('utf8')}")
            return False

    def _transcribe_file(self, audio_path: Path, transcription_path: Path) -> bool:
        """Transcribes a single audio file using the pre-loaded faster-whisper model."""
        if transcription_path.exists():
            log.info(f"Transcription already exists, skipping: {transcription_path}")
            return True

        log.info(f"Transcribing: {audio_path}")
        try:
            segments, info = self.model.transcribe(
                str(audio_path), language="pt"
            )  # REVIEW remove `info`?

            with transcription_path.open("w", encoding="utf-8") as f:
                for segment in segments:
                    f.write(segment.text)
            return True
        except Exception as e:
            log.error(f"faster-whisper error on {audio_path}: {e}")
            return False

    def process_directory(
        self, input_dir: Path, preprocessed_dir: Path, transcription_dir: Path
    ):
        """
        Processes all audio files in a directory: preprocesses and then transcribes them.
        """
        log.info(f"Scanning for audio files in: {input_dir}")
        source_files = [f for ext in AUDIO_EXTENSIONS for f in input_dir.glob(ext)]

        if not source_files:
            log.warning(f"No audio files found in {input_dir}. Nothing to process.")
            return

        for source_file in source_files:
            preprocessed_file = (
                preprocessed_dir / f"{source_file.stem}-processed{source_file.suffix}"
            )

            # Step 1: Preprocess
            if not self._preprocess_ffmpeg(source_file, preprocessed_file):
                log.error(
                    f"Skipping transcription for {source_file} due to preprocessing error."
                )
                continue

            # Step 2: Transcribe
            transcription_file = transcription_dir / f"{source_file.stem}.txt"
            self._transcribe_file(preprocessed_file, transcription_file)
