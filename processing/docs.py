import logging
from pathlib import Path

import fitz
import ocrmypdf
import pytesseract
from PIL import Image

# Logging!
from utils.logging import DOG_LOGGER_NAME

log = logging.getLogger(f"{DOG_LOGGER_NAME}.{__name__}")

# Define supported file extensions
PDF_EXTENSIONS = ["*.pdf"]
IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.tiff", "*.bmp"]
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS + IMAGE_EXTENSIONS


class DocumentProcessor:
    """Encapsulates OCR processing for PDFs and images."""

    def __init__(self, language: str):
        """
        Initializes the document processor.

        Args:
            language (str): The language code for the OCR.
        """
        self.language = language
        log.info(f"DocumentProcessor initialized for language: '{self.language}'")

    def _ocr_and_extract_pdf(self, input_path: Path, output_path: Path) -> bool:
        """Processes a single PDF file using OCRmyPDF and PyMuPDF."""
        temp_ocred_pdf = input_path.with_suffix(".ocred.pdf")
        try:
            log.info(f"Starting OCR for PDF: {input_path}")

            ocrmypdf.ocr(
                input_path,
                temp_ocred_pdf,
                skip_text=True,
                language=self.language,
                progress_bar=False,
            )

            log.info(f"Extracting text from OCR'd PDF: {temp_ocred_pdf}")

            with fitz.open(temp_ocred_pdf) as doc:
                text = "".join(page.get_text() for page in doc)

            with output_path.open("w", encoding="utf-8") as f:
                f.write(text)
            return True

        except Exception as e:
            log.error(f"Failed to process PDF {input_path}: {e}")
            return False
        finally:
            if temp_ocred_pdf.exists():
                temp_ocred_pdf.unlink()  # Cleanup temporary file

    def _ocr_image(self, input_path: Path, output_path: Path) -> bool:
        """Processes a single image file using Tesseract."""
        try:
            log.info(f"Starting OCR for image: {input_path}")

            text = pytesseract.image_to_string(
                Image.open(input_path), lang=self.language
            )
            with output_path.open("w", encoding="utf-8") as f:
                f.write(text)

            return True
        except Exception as e:
            log.error(f"Failed to process image {input_path}: {e}")
            return False

    def process_directory(self, input_dir: Path, output_dir: Path):
        """
        Processes all supported documents in an input directory.
        """
        log.info(f"Scanning for documents and images in: {input_dir}")
        files_to_process = [
            f for ext in SUPPORTED_EXTENSIONS for f in input_dir.glob(ext)
        ]

        if not files_to_process:
            log.warning(f"No supported document files found in {input_dir}.")
            return

        for file in files_to_process:
            output_txt_path = output_dir / f"{file.stem}.txt"
            if output_txt_path.exists():
                log.info(f"Output already exists, skipping: {output_txt_path}")
                continue

            file_suffix = file.suffix.lower()  # As precaution
            if file_suffix == ".pdf":
                self._ocr_and_extract_pdf(file, output_txt_path)
            elif file_suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
                self._ocr_image(file, output_txt_path)
