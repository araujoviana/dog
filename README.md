# 🐕

`dog` is a local Retrieval-Augmented Generation (RAG) pipeline.

> [!WARNING]
> This is a prototype and not ready for production use. The Docker setup is currently not functional.

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
    uv sync
    ```

### Configuration

1.  Create a `config.toml` file from the `config.toml.example` file.
2.  Add your Groq API key to the `config.toml` file.
3.  Define the input folders for your audio, and document files in `config.toml`.
4.  (Optional) Configure the different settings in `config.toml` to customize the pipeline's behavior. The available settings are:
    - `[audio]`: Settings for the audio processing pipeline.
    - `[ocr_settings]`: Settings for the OCR pipeline.
    - `[clean_up]`: Settings for the text cleaning pipeline.
    - `[embedding]`: Settings for the text embedding and vector store.
    - `[retrieval]`: Settings for the retrieval and answer generation pipeline.

### Usage

1.  **Run the server**:
    ```bash
    uv run server.py
    ```
2.  **Open the web interface**:
    - Open your browser and go to `http://127.0.0.1:8000`.
3.  **Upload files**:
    - Use the "Data" tab to upload your audio and document files.
4.  **Process data**:
    - Click the "Process data sources" button to process the uploaded files.
5.  **Refresh knowledge base**:
    - Click the "Refresh knowledge base" button to build the knowledge base.
6.  **Ask questions**:
    - Use the "Chat" tab to ask questions about the content of your files.
