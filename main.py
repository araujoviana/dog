import ffmpeg
from faster_whisper import WhisperModel
import logging
from joblib import Parallel, delayed
from pathlib import Path
from datetime import datetime
import tomllib
from logging import basicConfig, info, debug, warning, error, critical, INFO

# REVIEW Learn audio processing
def process_audio(input_audio, output_audio):
    """
    Process audio file for better transcription.

    Args:
        input_audio (str): Path to input .ogg file
        output_audio (str): Path to output .ogg file
    """

    if output_audio.exists():
        warning(f"{output_audio} already exists, skipping processing")
        return


    info(f"Processing audio file: {input_audio}")

    try:
        (
            ffmpeg.input(str(input_audio))
                .filter("volume", "6dB")
                .filter("highpass", f=80)
                .filter("loudnorm")
                .output(str(output_audio), acodec="libopus")
                .run(overwrite_output=True)
        )
    except ffmpeg.Error as e:
        error(f"Error processing {input_audio}: {e.stderr.decode('utf8')}")


def main():
    # Load config
    config_path = Path("input-data/config.toml")
    with config_path.open("rb") as f:
        config = tomllib.load(f)

    # Paths from config
    audio_folder = Path(config["paths"]["audio-folder"])
    output_folder = Path(config["paths"]["output_folder"])
    log_folder = Path(config["paths"]["log_folder"])

    # Ensure folders exist
    output_folder.mkdir(parents=True, exist_ok=True)
    log_folder.mkdir(parents=True, exist_ok=True)

    # Setup logging
    exec_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = log_folder / f"rag_{exec_timestamp}.log"
    basicConfig(
        filename=str(log_filename),
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=INFO
    )
    logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

    info("Starting dog 🐕")

    # Collect audio files
    audio_extensions = ["*.mp3", "*.wav", "*.flac", "*.aac", "*.ogg", "*.m4a"]
    audio_files = [f for ext in audio_extensions for f in audio_folder.glob(ext)]

    # Processing audio function
    def audio_processing_job(audio_file):
        output_file = output_folder / f"{audio_file.stem}-processed{audio_file.suffix}"
        process_audio(audio_file, output_file)

    # Parallel audio processing
    Parallel(n_jobs=-1)(delayed(audio_processing_job)(f) for f in audio_files)

    info("Done audio processing")

    # Folder for transcriptions
    transcription_folder = Path(config["paths"].get("transcription_folder", "transcriptions"))
    transcription_folder.mkdir(parents=True, exist_ok=True)


    # REVIEW tweak model for performance

    def transcribe_file(f, transcription_folder):
        info(f"Transcribing file: {f}")

        model = WhisperModel(config["whisper"]["model"], device="cpu", compute_type="int8")
        segments, winfo = model.transcribe(str(f), beam_size=5, language="pt", vad_filter=True)

        info(f"Detected language '{winfo.language}' with probability {winfo.language_probability}")

        transcription = "".join([segment.text for segment in segments])

        info(f"Done transcribing file: {f}")

        output_file = transcription_folder / f"{f.stem}.txt"
        with output_file.open("w", encoding="utf-8") as out:
            out.write(transcription)

    processed_files = list(output_folder.glob("*.*"))

    for f in processed_files:
        transcribe_file(f, transcription_folder)


    info("Done audio processing")

if __name__ == "__main__":
    main()
