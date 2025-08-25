# 🐕

`dog` is a local Retrieval-Augmented Generation (RAG) pipeline designed to answer questions based on a knowledge base of your own documents and audio files. It uses local models for embedding and reranking, and the Groq API for fast inference.

> [!WARNING]
> This is a prototype and not ready for production use. The Docker setup is currently not functional.

## Features

*   **Local-First:** Runs entirely on your machine, ensuring data privacy.
*   **Multiple Data Sources:** Process documents (`.pdf`, `.txt`, `.md`) and audio files (`.mp3`, `.wav`).
*   **OCR Support:** Extracts text from scanned PDFs and images within PDFs.
*   **Web Interface:** Easy-to-use web UI for uploading files, managing the knowledge base, and chatting with the RAG pipeline.
*   **Command-Line Interface:** A CLI for processing data and querying the pipeline.
*   **Configurable:** Customize the pipeline's behavior through a simple `config.toml` file.

## Getting Started

### Prerequisites

*   **Python >=3.11,<3.13**
*   **uv:** This project uses `uv` for package management. You can find installation instructions for `uv` [here](https://github.com/astral-sh/uv).
*   **ffmpeg:** This project requires `ffmpeg` to be installed on your system. You can download it from [ffmpeg.org](https://ffmpeg.org/download.html) or install it using your system's package manager (e.g., `sudo apt install ffmpeg` on Debian/Ubuntu, `brew install ffmpeg` on macOS).
*   **Tesseract:** This project uses `Tesseract` for OCR. You can find installation instructions [here](https://tesseract-ocr.github.io/tessdoc/Installation.html).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/dog.git
    cd dog
    ```

2.  **Create a virtual environment:**
    ```bash
    uv venv
    ```

3.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```

4.  **Install Python dependencies:**
    ```bash
    uv sync
    ```

### Configuration

1.  **Create a `config.toml` file from the example:**
    ```bash
    cp config.toml.example config.toml
    ```

2.  **Add your Groq API key:**
    Open `config.toml` and add your Groq API key to the `api_key` field under the `[auth]` section.

3.  **Define input folders:**
    In `config.toml`, specify the input folders for your audio and document files.

4.  **(Optional) Customize the pipeline:**
    The `config.toml` file allows you to configure various aspects of the RAG pipeline. See the "Configuration Details" section for more information.

## Usage

You can interact with the RAG pipeline through either the web interface or the command-line interface.

### Web Interface (Recommended)

1.  **Run the server:**
    ```bash
    uv run server.py
    ```

2.  **Open the web interface:**
    Open your browser and go to `http://127.0.0.1:8000`.

3.  **Upload files:**
    Use the "Data" tab to upload your audio and document files.

4.  **Process data:**
    Click the "Process data sources" button to process the uploaded files. This will transcribe audio files, extract text from documents, and clean the text.

5.  **Refresh knowledge base:**
    Click the "Refresh knowledge base" button to build the vector store from the processed text.

6.  **Ask questions:**
    Use the "Chat" tab to ask questions about the content of your files.

### Command-Line Interface

The `main.py` script provides a command-line interface for interacting with the RAG pipeline.

1.  **Process data sources:**
    ```bash
    uv run main.py --process
    ```

2.  **Build the knowledge base:**
    ```bash
    uv run main.py --build
    ```

3.  **Query the pipeline:**
    ```bash
    uv run main.py
    ```
    You will be prompted to enter a query.

## Configuration Details

The `config.toml` file allows you to configure the following settings:

*   **`[auth]`**: Contains your Groq API key in the `api_key` field.
*   **`[paths]`**: The input and output folders for the pipeline.
*   **`[audio]`**: Settings for the audio processing pipeline, including the Whisper model to use.
*   **`[ocr_settings]`**: The language to use for OCR.
*   **`[clean_up]`**: Settings for the text cleaning pipeline.
*   **`[embedding]`**: Settings for the text embedding and vector store, including the embedding model and cross-encoder to use.
*   **`[retrieval]`**: Settings for the retrieval and answer generation pipeline, including the prompt template to use.
