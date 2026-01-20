"""Entity extraction using Claude API."""

from typing import Literal

from pydantic import BaseModel, Field

from config.logging_config import get_logger

from .claude_client import ClaudeClient
from .prompts import ENTITY_EXTRACTION_SYSTEM, ENTITY_EXTRACTION_PROMPT

logger = get_logger(__name__)


EntityType = Literal[
    "person", "organization", "concept", "topic", "project", "event", "location"
]


class ExtractedEntity(BaseModel):
    """An entity extracted from text."""

    name: str = Field(description="The canonical name of the entity")
    type: EntityType = Field(description="The type of entity")
    aliases: list[str] = Field(
        default_factory=list, description="Alternative names or spellings"
    )
    description: str | None = Field(
        default=None, description="Brief description based on context"
    )
    confidence: float = Field(
        ge=0, le=1, description="Confidence score for the extraction"
    )
    context: str = Field(description="Text snippet where the entity appears")


class EntityExtractionResult(BaseModel):
    """Result of entity extraction from a document."""

    entities: list[ExtractedEntity] = Field(
        default_factory=list, description="List of extracted entities"
    )


class EntityExtractor:
    """Extracts entities from text using Claude."""

    def __init__(self, claude_client: ClaudeClient):
        """Initialize the extractor.

        Args:
            claude_client: Configured Claude client
        """
        self.client = claude_client

    def extract_entities(
        self,
        text: str,
        source_id: str,
        existing_entities: list[str] | None = None,
        max_text_length: int = 100000,
    ) -> EntityExtractionResult:
        """Extract entities from text.

        Args:
            text: Document text to analyze
            source_id: ID of source document/email for logging
            existing_entities: Known entity names for resolution
            max_text_length: Maximum text length to process

        Returns:
            EntityExtractionResult with extracted entities
        """
        # Truncate very long texts
        if len(text) > max_text_length:
            logger.warning(
                "text_truncated",
                source_id=source_id,
                original_length=len(text),
                truncated_to=max_text_length,
            )
            text = text[:max_text_length]

        # Build prompt
        existing_section = ""
        if existing_entities:
            existing_section = (
                "Known entities for reference (match to these if the same entity):\n"
                + "\n".join(f"- {e}" for e in existing_entities[:50])  # Limit to 50
            )

        prompt = ENTITY_EXTRACTION_PROMPT.format(
            text=text,
            existing_entities_section=existing_section,
        )

        try:
            result = self.client.extract_structured(
                prompt=prompt,
                schema=EntityExtractionResult,
                system=ENTITY_EXTRACTION_SYSTEM,
            )

            logger.info(
                "entities_extracted",
                source_id=source_id,
                count=len(result.entities),
            )

            return result

        except Exception as e:
            logger.error(
                "entity_extraction_failed",
                source_id=source_id,
                error=str(e),
            )
            return EntityExtractionResult(entities=[])

    def extract_entities_chunked(
        self,
        text: str,
        source_id: str,
        chunk_size: int = 50000,
        chunk_overlap: int = 1000,
        existing_entities: list[str] | None = None,
    ) -> EntityExtractionResult:
        """Extract entities from long text by processing in chunks.

        Args:
            text: Full document text
            source_id: Source ID for logging
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks to catch entities at boundaries
            existing_entities: Known entities for resolution

        Returns:
            Combined EntityExtractionResult
        """
        if len(text) <= chunk_size:
            return self.extract_entities(text, source_id, existing_entities)

        all_entities: list[ExtractedEntity] = []
        seen_names: set[str] = set()

        # Process in overlapping chunks
        start = 0
        chunk_num = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunk_num += 1

            logger.debug(
                "processing_chunk",
                source_id=source_id,
                chunk=chunk_num,
                start=start,
                end=end,
            )

            # Include previously found entities for resolution
            combined_existing = list(seen_names)
            if existing_entities:
                combined_existing.extend(existing_entities)

            result = self.extract_entities(
                chunk,
                f"{source_id}_chunk{chunk_num}",
                combined_existing,
            )

            # Deduplicate entities
            for entity in result.entities:
                key = (entity.name.lower(), entity.type)
                if entity.name.lower() not in seen_names:
                    all_entities.append(entity)
                    seen_names.add(entity.name.lower())

            # Move to next chunk with overlap
            start = end - chunk_overlap
            if start >= len(text):
                break

        logger.info(
            "chunked_extraction_complete",
            source_id=source_id,
            chunks=chunk_num,
            total_entities=len(all_entities),
        )

        return EntityExtractionResult(entities=all_entities)
