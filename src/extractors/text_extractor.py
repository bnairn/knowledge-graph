"""Unified text extraction from various file formats."""

import re
from pathlib import Path

from config.logging_config import get_logger

from .pdf_extractor import PDFExtractor
from .document_extractor import DocumentExtractor

logger = get_logger(__name__)


# MIME type to extractor mapping
MIME_TYPE_MAP = {
    # PDF
    "application/pdf": "pdf",
    # Microsoft Office (modern)
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    # Microsoft Office (legacy)
    "application/msword": "docx",  # Will try as docx
    "application/vnd.ms-excel": "xlsx",
    "application/vnd.ms-powerpoint": "pptx",
    # Plain text
    "text/plain": "text",
    "text/markdown": "text",
    "text/csv": "text",
    # Google Workspace (exported as text)
    "application/vnd.google-apps.document": "text",
    "application/vnd.google-apps.spreadsheet": "text",
    "application/vnd.google-apps.presentation": "text",
}


class TextExtractor:
    """Unified text extraction from various file formats."""

    def __init__(self):
        """Initialize extractors."""
        self.pdf_extractor = PDFExtractor()
        self.doc_extractor = DocumentExtractor()

    def extract(self, content: bytes, mime_type: str) -> str:
        """Extract text from content based on MIME type.

        Args:
            content: File content as bytes
            mime_type: MIME type of the content

        Returns:
            Extracted plain text

        Raises:
            ValueError: If MIME type is not supported
        """
        extractor_type = MIME_TYPE_MAP.get(mime_type)

        if extractor_type is None:
            raise ValueError(f"Unsupported MIME type: {mime_type}")

        logger.debug("extracting_text", mime_type=mime_type, extractor=extractor_type)

        if extractor_type == "pdf":
            return self.pdf_extractor.extract(content)
        elif extractor_type == "docx":
            return self.doc_extractor.extract_docx(content)
        elif extractor_type == "xlsx":
            return self.doc_extractor.extract_xlsx(content)
        elif extractor_type == "pptx":
            return self.doc_extractor.extract_pptx(content)
        elif extractor_type == "text":
            return content.decode("utf-8", errors="replace")
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")

    def extract_from_file(self, file_path: Path) -> str:
        """Extract text from a local file.

        Args:
            file_path: Path to the file

        Returns:
            Extracted plain text
        """
        mime_type = self._guess_mime_type(file_path)
        content = file_path.read_bytes()
        return self.extract(content, mime_type)

    def normalize_text(self, text: str) -> str:
        """Normalize extracted text.

        - Remove excessive whitespace
        - Normalize line endings
        - Remove control characters

        Args:
            text: Raw extracted text

        Returns:
            Normalized text
        """
        # Remove control characters except newlines and tabs
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse multiple blank lines into two
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse multiple spaces into one
        text = re.sub(r"[^\S\n]+", " ", text)

        # Strip leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    def is_supported(self, mime_type: str) -> bool:
        """Check if a MIME type is supported.

        Args:
            mime_type: MIME type to check

        Returns:
            True if supported
        """
        return mime_type in MIME_TYPE_MAP

    def _guess_mime_type(self, file_path: Path) -> str:
        """Guess MIME type from file extension.

        Args:
            file_path: Path to file

        Returns:
            Guessed MIME type
        """
        extension_map = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".ppt": "application/vnd.ms-powerpoint",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".csv": "text/csv",
        }

        ext = file_path.suffix.lower()
        mime_type = extension_map.get(ext, "application/octet-stream")
        return mime_type
