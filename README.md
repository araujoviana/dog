# 🐕

`dog` is a local Retrieval-Augmented Generation (RAG) pipeline.

## What it does

`dog` can process local audio and text files to answer questions about their content. It performs the following steps:

1.  **Audio Processing**: Cleans and enhances audio files (`.mp3`, `.wav`, `.flac`, `.aac`, `.ogg`, `.m4a`).
2.  **Audio Transcription**: Transcribes audio files into text using Whisper.
3.  **Text Cleaning**: Cleans the transcribed text and any provided text files (`.txt`, `.org`, `.md`) using the Groq API.
4.  **Indexing**: Creates embeddings of the cleaned text and indexes them using FAISS for efficient retrieval.
5.  **Question Answering**: Takes a user-defined query, retrieves the most relevant text chunks, and uses a large language model to generate a comprehensive answer.

## How to use it

### Dependencies

- **ffmpeg**: This project requires `ffmpeg` to be installed on your system. You can download it from [ffmpeg.org](https://ffmpeg.org/download.html) or install it using your system's package manager (e.g., `sudo apt install ffmpeg` on Debian/Ubuntu, `brew install ffmpeg` on macOS).

### Installation

This project uses `uv` for package management. You can find installation instructions for `uv` [here](https://github.com/astral-sh/uv).

1.  **Create a virtual environment**:
    ```bash
    uv venv
    ```
2.  **Activate the virtual environment**:
    ```bash
    source .venv/bin/activate
    ```
3.  **Install Python dependencies**:
    ```bash
    uv pip install -e .
    ```

### Configuration

1.  Create a `config.toml` file from the `config.toml.example` file.
2.  Add your Groq API key to the `config.toml` file.
3.  Define the input folders for your audio and text files in `config.toml`.
4.  Add your audio/text files to the input folder

### Usage

1.  **Add a question**:
    - Modify the `QUERY` variable in `main.py` to ask your question.
2.  **Run the pipeline**:
    ```bash
    uv run main.py
    ```
