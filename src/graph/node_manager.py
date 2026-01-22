"""Node CRUD operations for all entity types."""

from datetime import datetime
from typing import Literal

from config.logging_config import get_logger

from src.nlp.entity_extractor import ExtractedEntity

from .neo4j_client import Neo4jClient

logger = get_logger(__name__)

# Map entity types to Neo4j labels
ENTITY_LABEL_MAP = {
    "person": "Person",
    "organization": "Organization",
    "concept": "Concept",
    "topic": "Topic",
    "project": "Project",
    "event": "Event",
    "location": "Location",
}


class NodeManager:
    """Manages node creation and updates in Neo4j."""

    def __init__(self, neo4j_client: Neo4jClient):
        """Initialize the manager.

        Args:
            neo4j_client: Connected Neo4j client
        """
        self.client = neo4j_client

    def create_or_update_entity(
        self,
        entity: ExtractedEntity,
        source_id: str,
    ) -> str:
        """Create or update an entity node.

        Uses MERGE to avoid duplicates. Updates properties if entity exists.

        Args:
            entity: Extracted entity to store
            source_id: ID of the source document/email

        Returns:
            Node ID of the created/updated entity
        """
        label = ENTITY_LABEL_MAP.get(entity.type, "Entity")

        query = f"""
        MERGE (e:{label} {{name: $name}})
        ON CREATE SET
            e.id = randomUUID(),
            e.created_at = datetime(),
            e.first_seen = datetime()
        ON MATCH SET
            e.last_seen = datetime()
        SET
            e.aliases = CASE
                WHEN e.aliases IS NULL THEN $aliases
                ELSE [x IN e.aliases + $aliases WHERE x IS NOT NULL | x]
            END,
            e.description = COALESCE($description, e.description),
            e.confidence = CASE
                WHEN e.confidence IS NULL OR $confidence > e.confidence
                THEN $confidence
                ELSE e.confidence
            END,
            e.updated_at = datetime()
        RETURN e.id AS id
        """

        params = {
            "name": entity.name,
            "aliases": [],
            "description": None,
            "confidence": entity.confidence,
        }

        results = self.client.execute_read(query, params)
        node_id = results[0]["id"] if results else None

        if node_id:
            logger.debug(
                "entity_node_created",
                entity_type=entity.type,
                name=entity.name,
                node_id=node_id,
            )

        return node_id

    def create_document_node(
        self,
        doc_id: str,
        name: str,
        mime_type: str,
        google_drive_id: str,
        content_hash: str,
        modified_time: datetime | None = None,
    ) -> str:
        """Create or update a document source node.

        Args:
            doc_id: Internal document ID
            name: Document name
            mime_type: MIME type
            google_drive_id: Google Drive file ID
            content_hash: Hash of content for change detection
            modified_time: Last modification time

        Returns:
            Node ID
        """
        query = """
        MERGE (d:Document {google_drive_id: $google_drive_id})
        ON CREATE SET
            d.id = $doc_id,
            d.created_at = datetime()
        SET
            d.name = $name,
            d.mime_type = $mime_type,
            d.content_hash = $content_hash,
            d.modified_time = $modified_time,
            d.last_synced = datetime(),
            d.sync_status = 'synced',
            d.updated_at = datetime()
        RETURN d.id AS id
        """

        params = {
            "doc_id": doc_id,
            "name": name,
            "mime_type": mime_type,
            "google_drive_id": google_drive_id,
            "content_hash": content_hash,
            "modified_time": modified_time.isoformat() if modified_time else None,
        }

        results = self.client.execute_read(query, params)
        return results[0]["id"] if results else doc_id

    def create_email_node(
        self,
        email_id: str,
        thread_id: str,
        subject: str,
        from_address: str,
        to_addresses: list[str],
        date: datetime | None = None,
        content_hash: str = "",
    ) -> str:
        """Create or update an email source node.

        Args:
            email_id: Gmail message ID
            thread_id: Gmail thread ID
            subject: Email subject
            from_address: Sender address
            to_addresses: Recipient addresses
            date: Email date
            content_hash: Hash of content

        Returns:
            Node ID
        """
        query = """
        MERGE (e:Email {id: $email_id})
        ON CREATE SET
            e.created_at = datetime()
        SET
            e.thread_id = $thread_id,
            e.subject = $subject,
            e.from_address = $from_address,
            e.to_addresses = $to_addresses,
            e.date = $date,
            e.content_hash = $content_hash,
            e.last_synced = datetime(),
            e.sync_status = 'synced',
            e.updated_at = datetime()
        RETURN e.id AS id
        """

        params = {
            "email_id": email_id,
            "thread_id": thread_id,
            "subject": subject,
            "from_address": from_address,
            "to_addresses": to_addresses,
            "date": date.isoformat() if date else None,
            "content_hash": content_hash,
        }

        results = self.client.execute_read(query, params)
        return results[0]["id"] if results else email_id

    def link_entity_to_source(
        self,
        entity_name: str,
        entity_type: str,
        source_id: str,
        source_type: Literal["document", "email"],
        confidence: float,
        context: str | None = None,
    ) -> None:
        """Create EXTRACTED_FROM relationship between entity and source.

        Args:
            entity_name: Name of the entity
            entity_type: Type of the entity
            source_id: ID of the source document/email
            source_type: Type of source (document or email)
            confidence: Extraction confidence
            context: Optional text context where entity appears
        """
        entity_label = ENTITY_LABEL_MAP.get(entity_type, "Entity")
        source_label = "Document" if source_type == "document" else "Email"
        source_id_field = "google_drive_id" if source_type == "document" else "id"

        query = f"""
        MATCH (e:{entity_label} {{name: $entity_name}})
        MATCH (s:{source_label} {{{source_id_field}: $source_id}})
        MERGE (e)-[r:EXTRACTED_FROM]->(s)
        ON CREATE SET
            r.extraction_date = datetime(),
            r.contexts = [$context]
        ON MATCH SET
            r.contexts = CASE
                WHEN $context IN r.contexts THEN r.contexts
                ELSE r.contexts + $context
            END
        SET
            r.confidence = CASE
                WHEN r.confidence IS NULL OR $confidence > r.confidence
                THEN $confidence
                ELSE r.confidence
            END,
            r.updated_at = datetime()
        """

        params = {
            "entity_name": entity_name,
            "source_id": source_id,
            "context": (context[:500] if context else ""),  # Limit context length
            "confidence": confidence,
        }

        self.client.execute_write(query, params)
        logger.debug(
            "entity_linked_to_source",
            entity=entity_name,
            source_id=source_id,
        )

    def get_entity_by_name(
        self,
        name: str,
        entity_type: str | None = None,
    ) -> dict | None:
        """Find an entity by name.

        Args:
            name: Entity name to search
            entity_type: Optional type filter

        Returns:
            Entity data or None
        """
        if entity_type:
            label = ENTITY_LABEL_MAP.get(entity_type, "Entity")
            query = f"""
            MATCH (e:{label})
            WHERE e.name = $name OR $name IN e.aliases
            RETURN e
            LIMIT 1
            """
        else:
            query = """
            MATCH (e)
            WHERE (e:Person OR e:Organization OR e:Concept OR e:Topic
                   OR e:Project OR e:Event OR e:Location)
              AND (e.name = $name OR $name IN e.aliases)
            RETURN e
            LIMIT 1
            """

        results = self.client.execute_read(query, {"name": name})
        return results[0]["e"] if results else None

    def get_all_entity_names(self) -> list[str]:
        """Get all entity names for deduplication.

        Returns:
            List of entity names
        """
        query = """
        MATCH (e)
        WHERE e:Person OR e:Organization OR e:Concept OR e:Topic
              OR e:Project OR e:Event OR e:Location
        RETURN DISTINCT e.name AS name
        """

        results = self.client.execute_read(query)
        return [r["name"] for r in results]
