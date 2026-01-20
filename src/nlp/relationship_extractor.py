"""Relationship extraction between entities."""

from typing import Any

from pydantic import BaseModel, Field

from config.logging_config import get_logger

from .claude_client import ClaudeClient
from .entity_extractor import ExtractedEntity
from .prompts import RELATIONSHIP_EXTRACTION_SYSTEM, RELATIONSHIP_EXTRACTION_PROMPT

logger = get_logger(__name__)


class ExtractedRelationship(BaseModel):
    """A relationship extracted between two entities."""

    source_entity: str = Field(description="Name of the source entity")
    target_entity: str = Field(description="Name of the target entity")
    relationship_type: str = Field(
        description="Type of relationship (e.g., WORKS_FOR, KNOWS, WORKS_ON)"
    )
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional properties like dates, roles, etc.",
    )
    confidence: float = Field(
        ge=0, le=1, description="Confidence score for the relationship"
    )
    context: str = Field(description="Text snippet where the relationship is mentioned")


class RelationshipExtractionResult(BaseModel):
    """Result of relationship extraction."""

    relationships: list[ExtractedRelationship] = Field(
        default_factory=list, description="List of extracted relationships"
    )


class RelationshipExtractor:
    """Extracts relationships between entities using Claude."""

    def __init__(self, claude_client: ClaudeClient):
        """Initialize the extractor.

        Args:
            claude_client: Configured Claude client
        """
        self.client = claude_client

    def extract_relationships(
        self,
        text: str,
        entities: list[ExtractedEntity],
        source_id: str,
        max_text_length: int = 100000,
    ) -> RelationshipExtractionResult:
        """Extract relationships between known entities.

        Args:
            text: Document text to analyze
            entities: List of entities found in the text
            source_id: ID of source document/email
            max_text_length: Maximum text length to process

        Returns:
            RelationshipExtractionResult with found relationships
        """
        if not entities or len(entities) < 2:
            logger.debug(
                "skipping_relationship_extraction",
                source_id=source_id,
                reason="insufficient_entities",
                entity_count=len(entities),
            )
            return RelationshipExtractionResult(relationships=[])

        # Truncate very long texts
        if len(text) > max_text_length:
            text = text[:max_text_length]

        # Format entities for the prompt
        entities_text = "\n".join(
            f"- {e.name} ({e.type}): {e.description or 'No description'}"
            for e in entities
        )

        prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
            entities=entities_text,
            text=text,
        )

        try:
            result = self.client.extract_structured(
                prompt=prompt,
                schema=RelationshipExtractionResult,
                system=RELATIONSHIP_EXTRACTION_SYSTEM,
            )

            # Validate that relationships reference known entities
            valid_entity_names = {e.name.lower() for e in entities}
            valid_relationships = []

            for rel in result.relationships:
                source_valid = rel.source_entity.lower() in valid_entity_names
                target_valid = rel.target_entity.lower() in valid_entity_names

                if source_valid and target_valid:
                    valid_relationships.append(rel)
                else:
                    logger.debug(
                        "invalid_relationship_skipped",
                        source=rel.source_entity,
                        target=rel.target_entity,
                        source_valid=source_valid,
                        target_valid=target_valid,
                    )

            logger.info(
                "relationships_extracted",
                source_id=source_id,
                total=len(result.relationships),
                valid=len(valid_relationships),
            )

            return RelationshipExtractionResult(relationships=valid_relationships)

        except Exception as e:
            logger.error(
                "relationship_extraction_failed",
                source_id=source_id,
                error=str(e),
            )
            return RelationshipExtractionResult(relationships=[])

    def extract_relationships_chunked(
        self,
        text: str,
        entities: list[ExtractedEntity],
        source_id: str,
        chunk_size: int = 50000,
        chunk_overlap: int = 1000,
    ) -> RelationshipExtractionResult:
        """Extract relationships from long text by processing in chunks.

        Args:
            text: Full document text
            entities: All entities found in the document
            source_id: Source ID for logging
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks

        Returns:
            Combined RelationshipExtractionResult
        """
        if len(text) <= chunk_size:
            return self.extract_relationships(text, entities, source_id)

        all_relationships: list[ExtractedRelationship] = []
        seen_relationships: set[tuple[str, str, str]] = set()

        start = 0
        chunk_num = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunk_num += 1

            logger.debug(
                "processing_relationship_chunk",
                source_id=source_id,
                chunk=chunk_num,
            )

            result = self.extract_relationships(
                chunk,
                entities,
                f"{source_id}_chunk{chunk_num}",
            )

            # Deduplicate relationships
            for rel in result.relationships:
                key = (
                    rel.source_entity.lower(),
                    rel.target_entity.lower(),
                    rel.relationship_type.upper(),
                )
                if key not in seen_relationships:
                    all_relationships.append(rel)
                    seen_relationships.add(key)

            start = end - chunk_overlap
            if start >= len(text):
                break

        logger.info(
            "chunked_relationship_extraction_complete",
            source_id=source_id,
            chunks=chunk_num,
            total_relationships=len(all_relationships),
        )

        return RelationshipExtractionResult(relationships=all_relationships)
