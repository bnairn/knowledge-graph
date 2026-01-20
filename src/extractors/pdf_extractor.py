"""PDF text extraction using PyMuPDF."""

from io import BytesIO

import pymupdf

from config.logging_config import get_logger

logger = get_logger(__name__)


class PDFExtractor:
    """Extracts text from PDF documents."""

    def extract(self, content: bytes) -> str:
        """Extract text from PDF bytes.

        Args:
            content: PDF file as bytes

        Returns:
            Extracted text content
        """
        try:
            doc = pymupdf.open(stream=content, filetype="pdf")
            text_parts = []

            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(page_text)

            doc.close()

            result = "\n\n".join(text_parts)
            logger.debug("pdf_extracted", pages=len(text_parts), chars=len(result))
            return result

        except Exception as e:
            logger.error("pdf_extraction_error", error=str(e))
            raise

    def extract_with_layout(self, content: bytes) -> str:
        """Extract text preserving spatial layout.

        Args:
            content: PDF file as bytes

        Returns:
            Text with layout preserved
        """
        try:
            doc = pymupdf.open(stream=content, filetype="pdf")
            text_parts = []

            for page in doc:
                # Use "blocks" mode for better layout preservation
                blocks = page.get_text("blocks")
                page_text = []

                for block in blocks:
                    if block[6] == 0:  # Text block (not image)
                        page_text.append(block[4])

                text_parts.append("\n".join(page_text))

            doc.close()
            return "\n\n---\n\n".join(text_parts)

        except Exception as e:
            logger.error("pdf_layout_extraction_error", error=str(e))
            raise

    def get_metadata(self, content: bytes) -> dict:
        """Extract PDF metadata.

        Args:
            content: PDF file as bytes

        Returns:
            Dictionary of metadata
        """
        try:
            doc = pymupdf.open(stream=content, filetype="pdf")
            metadata = doc.metadata
            page_count = len(doc)
            doc.close()

            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "page_count": page_count,
            }
        except Exception as e:
            logger.error("pdf_metadata_error", error=str(e))
            return {}
