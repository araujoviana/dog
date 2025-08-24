# 🐕

`dog` is a local Retrieval-Augmented Generation (RAG) pipeline.

> [!WARNING]
> This is a prototype and not ready for production use. The Docker setup is currently not functional.

## What it does

`dog` can process local audio, text, PDF, and image files to answer questions about their content. It performs the following steps:

1.  **Audio Processing**: Cleans and enhances audio files (`.mp3`, `.wav`, `.flac`, `.aac`, `.ogg`, `.m4a`).
2.  **Audio Transcription**: Transcribes audio files into text using Whisper.
3.  **Document Processing**: Extracts text from PDF and image files (`.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`) using Tesseract OCR.
4.  **Text Cleaning**: Cleans the transcribed and extracted text, as well as any provided text files (`.txt`, `.org`, `.md`), using the Groq API.
5.  **Indexing**: Creates embeddings of the cleaned text and indexes them using FAISS for efficient retrieval.
6.  **Question Answering**: Takes a user-defined query, retrieves the most relevant text chunks, and uses a large language model to generate a comprehensive answer.

## How to use it

### Dependencies

- **ffmpeg**: This project requires `ffmpeg` to be installed on your system. You can download it from [ffmpeg.org](https://ffmpeg.org/download.html) or install it using your system's package manager (e.g., `sudo apt install ffmpeg` on Debian/Ubuntu, `brew install ffmpeg` on macOS).
- **Tesseract**: This project uses `Tesseract` for OCR. You can find installation instructions [here](https://tesseract-ocr.github.io/tessdoc/Installation.html).

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
3.  Define the input folders for your audio, text, and document files in `config.toml`.
4.  Add your audio/text/document files to the respective input folders.
5.  (Optional) Configure the different settings in `config.toml` to customize the pipeline's behavior. The available settings are:
    - `[audio]`: Settings for the audio processing pipeline.
    - `[ocr_settings]`: Settings for the OCR pipeline.
    - `[clean_up]`: Settings for the text cleaning pipeline.
    - `[embedding]`: Settings for the text embedding and vector store.
    - `[retrieval]`: Settings for the retrieval and answer generation pipeline.

### Usage

1.  **Add a question**:
    - The tool will prompt you to enter a question when you run it.
2.  **Run the pipeline**:
    ```bash
    uv run main.py
    ```