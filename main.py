import ffmpeg
import re
import whisper
import logging
from joblib import Parallel, delayed
from pathlib import Path
from datetime import datetime
import tomllib
from logging import INFO, info, warning, error, critical
from groq import Groq
import time


# REVIEW Learn audio processing
def process_audio(input_audio, output_audio):
    """
    Process audio file for better transcription.

    Args:
        input_audio (str): Path to input .ogg file
        output_audio (str): Path to output .ogg file
    """

    # Skips existing files
    if output_audio.exists():
        warning(f"{output_audio} already exists, skipping processing")
        return

    info(f"Processing audio file: {input_audio}")

    try:
        (
            ffmpeg.input(str(input_audio))
                .filter("volume", "6dB")                # Apply volume boost
                .filter("highpass", f=80)               # Apply high-pass filter (removes low rumble)
                .filter("loudnorm")                     # Apply loudness normalization
                .output(str(output_audio), acodec="libopus")  # Encode output as Opus
                .run(overwrite_output=True)             # Execute and overwrite if file exists
        )
    except ffmpeg.Error as e:
        error(f"Error processing {input_audio}: {e.stderr.decode('utf8')}")


def clean_with_groq(text_chunk, client, llm_model, prompt):
    """
    Cleans a text chunk using the Groq API.
    """
    full_prompt = prompt.format(text_chunk=text_chunk)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": full_prompt,
                }
            ],
            model       = llm_model,
            temperature = 0.2, # Lower temperature (more) deterministic cleaning
            top_p       = 0.9, # Larger set of tokens picked, which makes it more creative in interpreting text
        )
        cleaned = chat_completion.choices[0].message.content.strip()
        return cleaned if cleaned else text_chunk
    except Exception as e:
        error(f"Groq API error: {e}")
        return text_chunk  # Fallback to original text on API error


def process_long_text(file_path, output_path, client, llm_model, chunk_size, prompt, api_call_cooldown):
    """
    Sends text to Groq API in chunks.
    """

    with file_path.open("r", encoding="utf-8") as f:
        text = f.read()

    info(f"Processing {len(text)} characters from {file_path.name}")

    # Split by sentences to preserve context
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        info(f"Processing sentence: {sentence}")

        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) + 1 < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    info(f"Split into {len(chunks)} chunks")

    with output_path.open("w", encoding="utf-8") as outfile:
        # Write metadata header
        outfile.write(f"LIMPO: true | ARQUIVO_ORIGEM: {file_path.name}\n\n")

        for i, chunk in enumerate(chunks):
            info(f"Processing chunk {i+1}/{len(chunks)}")

            try:
                cleaned = clean_with_groq(chunk, client, llm_model, prompt)
                outfile.write(cleaned + " ")
                outfile.flush()  # Write immediately

            except Exception as e:
                critical(f"Error processing chunk {i+1}: {e}")
                outfile.write(chunk + " ")  # Fallback to original

            # API cooldown to avoid 429 response
            info(f"Waiting {api_call_cooldown} seconds for the API...")
            time.sleep(api_call_cooldown)

    info(f"Finished cleaning {file_path.name}")


def main():
    # Load local config
    config_path = Path("config.toml")
    with config_path.open("rb") as f:
        config = tomllib.load(f)

    # Paths from config
    audio_folder = Path(config["paths"]["audio_folder"])
    text_folder = Path(config["paths"]["text_folder"])
    output_folder = Path(config["paths"]["output_folder"])
    log_folder = Path(config["paths"]["log_folder"])

    # Ensure folders exist
    output_folder.mkdir(parents=True, exist_ok=True)
    log_folder.mkdir(parents=True, exist_ok=True)

    # Setup logging
    exec_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = log_folder / f"rag_{exec_timestamp}.log"
    # File handler
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    info("Starting dog 🐕")

    # Collect audio files
    audio_extensions = ["*.mp3", "*.wav", "*.flac", "*.aac", "*.ogg", "*.m4a"]
    audio_files = [f for ext in audio_extensions for f in audio_folder.glob(ext)]

    # Collect text files
    text_extensions = ["*.txt", "*.org", "*.md"]
    text_files = [f for ext in text_extensions for f in text_folder.glob(ext)]

    # Processing audio function
    def audio_processing_job(audio_file):
        output_file = output_folder / f"{audio_file.stem}-processed{audio_file.suffix}"
        process_audio(audio_file, output_file)

    # Parallel audio processing
    if audio_files:
        Parallel(n_jobs=-1)(delayed(audio_processing_job)(f) for f in audio_files)
        info("Done audio processing")

    # Folder for transcriptions
    transcription_folder = Path(config["paths"].get("transcription_folder", "transcriptions"))
    transcription_folder.mkdir(parents=True, exist_ok=True)

    # REVIEW tweak model for performance
    def transcribe_file(f, transcription_folder):
        output_file = transcription_folder / f"{f.stem}.txt"
        if output_file.exists():
            info(f"Transcription already exists for {f}, skipping")
            return

        info(f"Transcribing file: {f}")

        # Load model on CPU
        model = whisper.load_model("base", device="cpu")
        result = model.transcribe(str(f), language="pt")

        with output_file.open("w", encoding="utf-8") as out:
            out.write(result["text"])

    processed_files = list(output_folder.glob("*.*"))

    # Transcribe files sequentially
    if processed_files:
        for f in processed_files:
            transcribe_file(f, transcription_folder)
        info("Done audio transcription")

    # Groq client setup
    info("Setting up Groq client")
    try:
        groq_api_key = config["auth"]["api_key"]
        llm_model = config["llm"]["model"]
        audio_prompt = config["llm"]["audio_prompt"]
        text_prompt = config["llm"]["text_prompt"]
        api_call_cooldown = config["llm"]["api_call_cooldown"]
        if not groq_api_key:
            raise ValueError("Groq API key is not set in config.toml")
    except KeyError as e:
        critical(f"Missing configuration in config.toml: {e}")
        return

    client = Groq(api_key=groq_api_key)
    info(f"Groq client configured to use model: {llm_model}")

    # Ensure cleaned folder exists
    cleaned_folder = Path(config["paths"].get("cleaned_folder", "cleaned_transcriptions"))
    cleaned_folder.mkdir(parents=True, exist_ok=True)

    # Process all transcription files
    transcription_files = list(transcription_folder.glob("*.txt"))
    info(f"Found {len(transcription_files)} transcription files to clean")

    if transcription_files:
        for f in transcription_files:
            output_file = cleaned_folder / f"{f.stem}-cleaned.txt"

            if output_file.exists():
                info(f"Cleaned transcription already exists for {f}, skipping")
                continue

            try:
                process_long_text(f, output_file, client, llm_model, 8000, audio_prompt, api_call_cooldown)
            except Exception as e:
                error(f"Failed to process {f}: {e}")
                continue
        info("Done transcription cleaning")

    # Process all text files
    info(f"Found {len(text_files)} text files to clean")
    if text_files:
        for f in text_files:
            output_file = cleaned_folder / f"{f.stem}-cleaned.txt"

            if output_file.exists():
                info(f"Cleaned text file already exists for {f}, skipping")
                continue

            try:
                process_long_text(f, output_file, client, llm_model, 8000, text_prompt, api_call_cooldown)
            except Exception as e:
                error(f"Failed to process {f}: {e}")
                continue
        info("Done text cleaning")


if __name__ == "__main__":
    main()
