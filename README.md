# Dog - Local RAG

This project is a local RAG (Retrieval-Augmented Generation) pipeline that uses Whisper for audio transcription.

## Usage

There are two main scripts in this project:

*   `main.py`: This script is optimized for CPU usage and uses the `faster-whisper` library for fast transcription on CPUs.
*   `main_gpu.py`: This script is optimized for GPU usage and uses the `faster-whisper` library with CUDA acceleration.

To run the scripts, you first need to install the dependencies using `rye sync`.

Then, you can run the scripts using the following commands:

```bash
# For CPU usage
python main.py

# For GPU usage
python main_gpu.py
```

## Configuration

The configuration for the project is located in the `input-data/config.toml` file. You can copy the `input-data/config.toml.example` file to `input-data/config.toml` and modify it to your needs.

The configuration file allows you to specify the following:

*   Paths to the audio folder, output folder, and log folder.
*   The Whisper model to use for transcription.