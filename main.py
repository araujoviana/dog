import ffmpeg
from joblib import Parallel, delayed
from pathlib import Path
from datetime import datetime
from logging import basicConfig, info, debug, warning, error, critical, INFO

# REVIEW Learn audio processing
def process_audio(input_audio, output_audio):
    """
    Process audio file for better transcription.

    Args:
        input_audio (str): Path to input .ogg file
        output_audio (str): Path to output .ogg file
    """

    info(f"Processing audio file: {input_audio}")

    try:
        (
            ffmpeg
                .input(str(input_audio))
                .audio.filter(
                    "acompressor", threshold="-18dB", ratio=1.8, attack=10, release=200
                )
                .filter("volume", "12dB")
                .filter("highpass", f=90)
                .filter("equalizer", f=5000, t="q", w=1, g=2)
                .output(str(output_audio), acodec="libopus")  # <- convert Path to str
                .run(overwrite_output=True)
        )
    except ffmpeg.Error as e:
        error(f"Error processing {input_audio}: {e.stderr.decode('utf8')}")


def main():

    # Setup logging
    exec_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"log/rag_{exec_timestamp}.log"
    basicConfig(
        filename=log_filename,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=INFO
    )

    info("Starting dog 🐕")

    audio_folder = Path("input-data") # User data

    output_folder = Path("processed-audio") # Treated audio
    output_folder.mkdir(exist_ok=True)

    audio_extensions = ["*.mp3", "*.wav", "*.flac", "*.aac", "*.ogg", "*.m4a"]
    audio_files = [f for ext in audio_extensions for f in audio_folder.glob(ext)]

    def audio_processing_job(audio_file):
        output_file = output_folder / f"{audio_file.stem}-processed{audio_file.suffix}"
        process_audio(audio_file, output_file)

    Parallel(n_jobs=-1)(delayed(audio_processing_job)(f) for f in audio_files)


    info("Done audio processing")


if __name__ == "__main__":
    main()
