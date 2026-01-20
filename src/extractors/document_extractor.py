"""Microsoft Office document text extraction."""

from io import BytesIO

from docx import Document
import openpyxl
from pptx import Presentation

from config.logging_config import get_logger

logger = get_logger(__name__)


class DocumentExtractor:
    """Extracts text from Microsoft Office documents."""

    def extract_docx(self, content: bytes) -> str:
        """Extract text from DOCX file.

        Args:
            content: DOCX file as bytes

        Returns:
            Extracted text content
        """
        try:
            doc = Document(BytesIO(content))
            text_parts = []

            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)

            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_text:
                        text_parts.append(" | ".join(row_text))

            result = "\n\n".join(text_parts)
            logger.debug("docx_extracted", paragraphs=len(text_parts), chars=len(result))
            return result

        except Exception as e:
            logger.error("docx_extraction_error", error=str(e))
            raise

    def extract_xlsx(self, content: bytes) -> str:
        """Extract text from XLSX file.

        Args:
            content: XLSX file as bytes

        Returns:
            Extracted text content (CSV-like format)
        """
        try:
            workbook = openpyxl.load_workbook(BytesIO(content), read_only=True, data_only=True)
            text_parts = []

            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                sheet_text = [f"## Sheet: {sheet_name}"]

                for row in sheet.iter_rows(values_only=True):
                    # Filter out None values and convert to strings
                    row_values = [str(cell) if cell is not None else "" for cell in row]
                    if any(v.strip() for v in row_values):
                        sheet_text.append(" | ".join(row_values))

                if len(sheet_text) > 1:  # Has content beyond just the header
                    text_parts.append("\n".join(sheet_text))

            workbook.close()

            result = "\n\n".join(text_parts)
            logger.debug("xlsx_extracted", sheets=len(text_parts), chars=len(result))
            return result

        except Exception as e:
            logger.error("xlsx_extraction_error", error=str(e))
            raise

    def extract_pptx(self, content: bytes) -> str:
        """Extract text from PPTX file.

        Args:
            content: PPTX file as bytes

        Returns:
            Extracted text content
        """
        try:
            prs = Presentation(BytesIO(content))
            text_parts = []

            for slide_num, slide in enumerate(prs.slides, 1):
                slide_text = [f"## Slide {slide_num}"]

                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_text.append(shape.text)

                    # Handle tables in slides
                    if shape.has_table:
                        for row in shape.table.rows:
                            row_text = [
                                cell.text.strip()
                                for cell in row.cells
                                if cell.text.strip()
                            ]
                            if row_text:
                                slide_text.append(" | ".join(row_text))

                if len(slide_text) > 1:  # Has content beyond just the header
                    text_parts.append("\n".join(slide_text))

            result = "\n\n".join(text_parts)
            logger.debug("pptx_extracted", slides=len(text_parts), chars=len(result))
            return result

        except Exception as e:
            logger.error("pptx_extraction_error", error=str(e))
            raise
