import ffmpeg
from datetime import datetime
import logging


def process_audio(input_audio, output_audio):
    """
    Process audio file for better transcription.

    Args:
        input_audio (str): Path to input .ogg file
        output_audio (str): Path to output .ogg file
    """
    try:
        audio = (
            ffmpeg.input(input_audio)
            .audio.filter(
                "acompressor", threshold="-18dB", ratio=1.8, attack=10, release=200
            )
            .filter("volume", "12dB")
            .filter("highpass", f=90)
            .filter("equalizer", f=5000, t="q", w=1, g=2)
        )

        return ffmpeg.output(audio, output_audio, acodec="libopus")
    except ffmpeg.Error as e:
        print(f"Error processing {input_audio}: {e.stderr.decode('utf8')}")


def main():


    # Replace "data/auto.ogg" and "data/output.ogg" with your actual file paths.
    process_audio("data/auto.ogg", "data/output.ogg")


if __name__ == "__main__":
    main()
