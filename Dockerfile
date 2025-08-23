FROM python:3.12-slim

RUN useradd -m dog
WORKDIR /home/dog/app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    tesseract-ocr \
    tesseract-ocr-por \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY --chown=dog:dog pyproject.toml uv.lock* ./

RUN su dog -c "uv sync --frozen --no-install-project"

COPY --chown=dog:dog . .

USER dog

CMD ["uv", "run", "main.py"]
