"""Relationship CRUD operations."""

from typing import Any

from config.logging_config import get_logger

from src.nlp.relationship_extractor import ExtractedRelationship

from .neo4j_client import Neo4jClient

logger = get_logger(__name__)

# Relationship type normalization
RELATIONSHIP_TYPE_MAP = {
    "works_for": "WORKS_FOR",
    "knows": "KNOWS",
    "works_on": "WORKS_ON",
    "attended": "ATTENDED",
    "expert_in": "EXPERT_IN",
    "located_in": "LOCATED_IN",
    "partnered_with": "PARTNERED_WITH",
    "uses": "USES",
    "related_to": "RELATED_TO",
    "owns": "OWNS",
    "subtopic_of": "SUBTOPIC_OF",
    "sent": "SENT",
    "received": "RECEIVED",
}


class RelationshipManager:
    """Manages relationship creation and updates in Neo4j."""

    def __init__(self, neo4j_client: Neo4jClient):
        """Initialize the manager.

        Args:
            neo4j_client: Connected Neo4j client
        """
        self.client = neo4j_client

    def create_or_update_relationship(
        self,
        relationship: ExtractedRelationship,
        source_id: str,
    ) -> bool:
        """Create or update a relationship between entities.

        Args:
            relationship: Extracted relationship to store
            source_id: ID of the source document/email

        Returns:
            True if relationship was created/updated
        """
        rel_type = self._normalize_relationship_type(relationship.relationship_type)

        # Build properties for the relationship
        props = {
            "confidence": relationship.confidence,
            "context": relationship.context[:500],  # Limit context length
        }
        props.update(relationship.properties)

        # Dynamic relationship creation query
        # We need to use APOC or construct query dynamically since relationship
        # types can't be parameterized in standard Cypher
        query = f"""
        MATCH (source)
        WHERE (source:Person OR source:Organization OR source:Concept OR source:Topic
               OR source:Project OR source:Event OR source:Location)
          AND (source.name = $source_name OR $source_name IN source.aliases)
        MATCH (target)
        WHERE (target:Person OR target:Organization OR target:Concept OR target:Topic
               OR target:Project OR target:Event OR target:Location)
          AND (target.name = $target_name OR $target_name IN target.aliases)
        MERGE (source)-[r:{rel_type}]->(target)
        ON CREATE SET
            r.created_at = datetime(),
            r.source_ids = [$source_id]
        ON MATCH SET
            r.source_ids = CASE
                WHEN $source_id IN r.source_ids THEN r.source_ids
                ELSE r.source_ids + $source_id
            END
        SET
            r.confidence = CASE
                WHEN r.confidence IS NULL OR $confidence > r.confidence
                THEN $confidence
                ELSE r.confidence
            END,
            r.context = $context,
            r.updated_at = datetime()
        RETURN r
        """

        params = {
            "source_name": relationship.source_entity,
            "target_name": relationship.target_entity,
            "source_id": source_id,
            "confidence": relationship.confidence,
            "context": relationship.context[:500],
        }

        # Add any additional properties
        for key, value in relationship.properties.items():
            if key not in params:
                params[key] = value

        try:
            results = self.client.execute_read(query, params)
            created = len(results) > 0

            if created:
                logger.debug(
                    "relationship_created",
                    source=relationship.source_entity,
                    target=relationship.target_entity,
                    type=rel_type,
                )

            return created
        except Exception as e:
            logger.error(
                "relationship_creation_failed",
                source=relationship.source_entity,
                target=relationship.target_entity,
                type=rel_type,
                error=str(e),
            )
            return False

    def create_email_relationships(
        self,
        email_id: str,
        from_address: str,
        to_addresses: list[str],
    ) -> None:
        """Create SENT and RECEIVED relationships for an email.

        This links Person nodes (matched by email) to Email nodes.

        Args:
            email_id: Gmail message ID
            from_address: Sender email address
            to_addresses: Recipient email addresses
        """
        # Create SENT relationship
        sent_query = """
        MATCH (p:Person)
        WHERE p.email = $from_address OR $from_address CONTAINS p.name
        MATCH (e:Email {id: $email_id})
        MERGE (p)-[r:SENT]->(e)
        ON CREATE SET r.created_at = datetime()
        """
        self.client.execute_write(
            sent_query,
            {"email_id": email_id, "from_address": from_address},
        )

        # Create RECEIVED relationships
        for to_addr in to_addresses:
            received_query = """
            MATCH (p:Person)
            WHERE p.email = $to_address OR $to_address CONTAINS p.name
            MATCH (e:Email {id: $email_id})
            MERGE (p)-[r:RECEIVED]->(e)
            ON CREATE SET r.created_at = datetime()
            """
            self.client.execute_write(
                received_query,
                {"email_id": email_id, "to_address": to_addr},
            )

    def get_relationships_for_entity(
        self,
        entity_name: str,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """Get all relationships for an entity.

        Args:
            entity_name: Name of the entity
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of relationship data
        """
        if direction == "outgoing":
            pattern = "(e)-[r]->(other)"
        elif direction == "incoming":
            pattern = "(e)<-[r]-(other)"
        else:
            pattern = "(e)-[r]-(other)"

        query = f"""
        MATCH (e)
        WHERE (e:Person OR e:Organization OR e:Concept OR e:Topic
               OR e:Project OR e:Event OR e:Location)
          AND (e.name = $name OR $name IN e.aliases)
        MATCH {pattern}
        RETURN type(r) AS relationship_type,
               other.name AS other_entity,
               labels(other) AS other_labels,
               r.confidence AS confidence,
               r.context AS context
        """

        return self.client.execute_read(query, {"name": entity_name})

    def _normalize_relationship_type(self, rel_type: str) -> str:
        """Normalize relationship type to uppercase format.

        Args:
            rel_type: Raw relationship type

        Returns:
            Normalized relationship type
        """
        normalized = rel_type.lower().replace(" ", "_").replace("-", "_")
        return RELATIONSHIP_TYPE_MAP.get(normalized, rel_type.upper().replace(" ", "_"))
