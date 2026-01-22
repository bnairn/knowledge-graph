"""Main pipeline coordinator."""

from dataclasses import dataclass, field
from datetime import datetime

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from config.logging_config import get_logger

from src.connectors.google_drive import GoogleDriveConnector, DriveFile
from src.connectors.gmail import GmailConnector, EmailMessage
from src.extractors.text_extractor import TextExtractor
from src.nlp.entity_extractor import EntityExtractor
from src.nlp.relationship_extractor import RelationshipExtractor
from src.graph.node_manager import NodeManager
from src.graph.relationship_manager import RelationshipManager

from config.settings import get_settings
from .sync_manager import SyncManager

logger = get_logger(__name__)


def should_skip_email(from_address: str) -> bool:
    """Check if email should be skipped based on sender.

    Args:
        from_address: Email sender address

    Returns:
        True if email should be skipped
    """
    settings = get_settings()
    skip_senders = settings.get_skip_senders_list()
    from_lower = from_address.lower()
    return any(skip in from_lower for skip in skip_senders)


@dataclass
class ProcessingResult:
    """Result of processing a single item."""

    item_id: str
    item_type: str  # 'document' or 'email'
    success: bool
    entities_extracted: int = 0
    relationships_extracted: int = 0
    error: str | None = None


@dataclass
class SyncReport:
    """Report of a sync operation."""

    started_at: datetime
    completed_at: datetime | None = None
    documents_processed: int = 0
    emails_processed: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    errors: list[str] = field(default_factory=list)


class PipelineOrchestrator:
    """Coordinates the full processing pipeline."""

    def __init__(
        self,
        drive_connector: GoogleDriveConnector | None,
        gmail_connector: GmailConnector | None,
        text_extractor: TextExtractor,
        entity_extractor: EntityExtractor,
        relationship_extractor: RelationshipExtractor,
        node_manager: NodeManager,
        relationship_manager: RelationshipManager,
        sync_manager: SyncManager,
    ):
        """Initialize the orchestrator.

        Args:
            drive_connector: Google Drive connector (optional)
            gmail_connector: Gmail connector (optional)
            text_extractor: Text extraction utility
            entity_extractor: Entity extractor
            relationship_extractor: Relationship extractor
            node_manager: Neo4j node manager
            relationship_manager: Neo4j relationship manager
            sync_manager: Sync state manager
        """
        self.drive = drive_connector
        self.gmail = gmail_connector
        self.text_extractor = text_extractor
        self.entity_extractor = entity_extractor
        self.relationship_extractor = relationship_extractor
        self.node_manager = node_manager
        self.relationship_manager = relationship_manager
        self.sync_manager = sync_manager

    def run_full_sync(self, show_progress: bool = True, limit: int | None = None) -> SyncReport:
        """Run complete sync of all data sources.

        Args:
            show_progress: Show progress bar
            limit: Maximum number of items to process per source

        Returns:
            SyncReport with results
        """
        report = SyncReport(started_at=datetime.now())

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            disable=not show_progress,
        ) as progress:
            # Sync Google Drive
            if self.drive:
                drive_task = progress.add_task("Syncing Google Drive...", total=limit)
                try:
                    for doc in self.drive.list_items():
                        if limit and report.documents_processed >= limit:
                            break
                        result = self.process_document(doc)
                        if result.success:
                            report.documents_processed += 1
                            report.total_entities += result.entities_extracted
                            report.total_relationships += result.relationships_extracted
                        else:
                            report.errors.append(result.error or f"Failed: {doc.name}")
                        progress.update(drive_task, advance=1)

                    self.sync_manager.update_sync_state(
                        "drive",
                        datetime.now(),
                        report.documents_processed,
                    )
                except Exception as e:
                    logger.error("drive_sync_failed", error=str(e))
                    report.errors.append(f"Drive sync error: {e}")
                finally:
                    progress.update(drive_task, completed=True)

            # Sync Gmail
            if self.gmail:
                gmail_task = progress.add_task("Syncing Gmail...", total=limit)
                try:
                    for email in self.gmail.list_items():
                        if limit and report.emails_processed >= limit:
                            break
                        # Skip transactional emails
                        if should_skip_email(email.from_address):
                            logger.debug("skipping_transactional_email", from_addr=email.from_address)
                            progress.update(gmail_task, advance=1)
                            continue
                        result = self.process_email(email)
                        if result.success:
                            report.emails_processed += 1
                            report.total_entities += result.entities_extracted
                            report.total_relationships += result.relationships_extracted
                        else:
                            report.errors.append(result.error or f"Failed: {email.subject}")
                        progress.update(gmail_task, advance=1)

                    self.sync_manager.update_sync_state(
                        "gmail",
                        datetime.now(),
                        report.emails_processed,
                    )
                except Exception as e:
                    logger.error("gmail_sync_failed", error=str(e))
                    report.errors.append(f"Gmail sync error: {e}")
                finally:
                    progress.update(gmail_task, completed=True)

        report.completed_at = datetime.now()
        logger.info(
            "full_sync_complete",
            documents=report.documents_processed,
            emails=report.emails_processed,
            entities=report.total_entities,
            relationships=report.total_relationships,
            errors=len(report.errors),
        )

        return report

    def run_incremental_sync(self, show_progress: bool = True, limit: int | None = None) -> SyncReport:
        """Sync only changed items since last sync.

        Args:
            show_progress: Show progress bar
            limit: Maximum number of items to process per source

        Returns:
            SyncReport with results
        """
        report = SyncReport(started_at=datetime.now())

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            disable=not show_progress,
        ) as progress:
            # Incremental Drive sync
            if self.drive:
                last_drive_sync = self.sync_manager.get_last_sync("drive")
                drive_task = progress.add_task(
                    f"Syncing Drive (since {last_drive_sync or 'never'})...",
                    total=None,
                )
                try:
                    for doc in self.drive.list_items(since=last_drive_sync):
                        # Check if content changed
                        content = self._get_document_content(doc)
                        content_hash = self.sync_manager.compute_hash(content)

                        if self.sync_manager.should_process(doc.id, content_hash):
                            result = self._process_document_with_content(doc, content)
                            if result.success:
                                self.sync_manager.record_processed(doc.id, content_hash)
                                report.documents_processed += 1
                                report.total_entities += result.entities_extracted
                                report.total_relationships += result.relationships_extracted
                            else:
                                report.errors.append(result.error or f"Failed: {doc.name}")
                        progress.update(drive_task, advance=1)

                    self.sync_manager.update_sync_state(
                        "drive",
                        datetime.now(),
                        report.documents_processed,
                    )
                except Exception as e:
                    logger.error("incremental_drive_sync_failed", error=str(e))
                    report.errors.append(f"Drive sync error: {e}")
                finally:
                    progress.update(drive_task, completed=True)

            # Incremental Gmail sync
            if self.gmail:
                last_gmail_sync = self.sync_manager.get_last_sync("gmail")
                gmail_task = progress.add_task(
                    f"Syncing Gmail (since {last_gmail_sync or 'never'})...",
                    total=None,
                )
                try:
                    for email in self.gmail.list_items(since=last_gmail_sync):
                        # Skip transactional emails
                        if should_skip_email(email.from_address):
                            logger.debug("skipping_transactional_email", from_addr=email.from_address)
                            progress.update(gmail_task, advance=1)
                            continue

                        content = email.body_plain or email.snippet
                        content_hash = self.sync_manager.compute_hash(content)

                        if self.sync_manager.should_process(email.id, content_hash):
                            result = self._process_email_with_content(email, content)
                            if result.success:
                                self.sync_manager.record_processed(email.id, content_hash)
                                report.emails_processed += 1
                                report.total_entities += result.entities_extracted
                                report.total_relationships += result.relationships_extracted
                            else:
                                report.errors.append(result.error or f"Failed: {email.subject}")
                        progress.update(gmail_task, advance=1)

                    self.sync_manager.update_sync_state(
                        "gmail",
                        datetime.now(),
                        report.emails_processed,
                    )
                except Exception as e:
                    logger.error("incremental_gmail_sync_failed", error=str(e))
                    report.errors.append(f"Gmail sync error: {e}")
                finally:
                    progress.update(gmail_task, completed=True)

        report.completed_at = datetime.now()
        logger.info(
            "incremental_sync_complete",
            documents=report.documents_processed,
            emails=report.emails_processed,
            entities=report.total_entities,
            relationships=report.total_relationships,
            errors=len(report.errors),
        )

        return report

    def process_document(self, doc: DriveFile) -> ProcessingResult:
        """Process a single document through the pipeline.

        Args:
            doc: Drive file to process

        Returns:
            ProcessingResult
        """
        try:
            content = self._get_document_content(doc)
            return self._process_document_with_content(doc, content)
        except Exception as e:
            logger.error("document_processing_failed", doc_id=doc.id, error=str(e))
            return ProcessingResult(
                item_id=doc.id,
                item_type="document",
                success=False,
                error=str(e),
            )

    def _get_document_content(self, doc: DriveFile) -> str:
        """Get text content from a document.

        Args:
            doc: Drive file

        Returns:
            Extracted text content
        """
        # Download the file
        content_bytes = self.drive.download_file(doc.id, doc.mime_type)

        # Extract text
        if self.text_extractor.is_supported(doc.mime_type):
            return self.text_extractor.extract(content_bytes, doc.mime_type)
        else:
            # Try as plain text
            return content_bytes.decode("utf-8", errors="replace")

    def _process_document_with_content(
        self,
        doc: DriveFile,
        content: str,
    ) -> ProcessingResult:
        """Process document with pre-extracted content.

        Args:
            doc: Drive file
            content: Extracted text content

        Returns:
            ProcessingResult
        """
        try:
            # Normalize text
            content = self.text_extractor.normalize_text(content)

            if not content.strip():
                logger.warning("empty_document", doc_id=doc.id)
                return ProcessingResult(
                    item_id=doc.id,
                    item_type="document",
                    success=True,
                    entities_extracted=0,
                    relationships_extracted=0,
                )

            # Get existing entities for resolution
            existing_entities = self.node_manager.get_all_entity_names()

            # Extract entities
            entity_result = self.entity_extractor.extract_entities_chunked(
                content,
                doc.id,
                existing_entities=existing_entities,
            )

            # Create document node
            content_hash = self.sync_manager.compute_hash(content)
            self.node_manager.create_document_node(
                doc_id=doc.id,
                name=doc.name,
                mime_type=doc.mime_type,
                google_drive_id=doc.id,
                content_hash=content_hash,
                modified_time=doc.modified_time,
            )

            # Store entities and link to document
            for entity in entity_result.entities:
                self.node_manager.create_or_update_entity(entity, doc.id)
                self.node_manager.link_entity_to_source(
                    entity.name,
                    entity.type,
                    doc.id,
                    "document",
                    entity.confidence,
                )

            # Extract relationships
            rel_result = self.relationship_extractor.extract_relationships_chunked(
                content,
                entity_result.entities,
                doc.id,
            )

            # Store relationships
            for rel in rel_result.relationships:
                self.relationship_manager.create_or_update_relationship(rel, doc.id)

            logger.info(
                "document_processed",
                doc_id=doc.id,
                name=doc.name,
                entities=len(entity_result.entities),
                relationships=len(rel_result.relationships),
            )

            return ProcessingResult(
                item_id=doc.id,
                item_type="document",
                success=True,
                entities_extracted=len(entity_result.entities),
                relationships_extracted=len(rel_result.relationships),
            )

        except Exception as e:
            logger.error("document_processing_failed", doc_id=doc.id, error=str(e))
            return ProcessingResult(
                item_id=doc.id,
                item_type="document",
                success=False,
                error=str(e),
            )

    def process_email(self, email: EmailMessage) -> ProcessingResult:
        """Process a single email through the pipeline.

        Args:
            email: Email message to process

        Returns:
            ProcessingResult
        """
        content = email.body_plain or email.snippet
        return self._process_email_with_content(email, content)

    def _process_email_with_content(
        self,
        email: EmailMessage,
        content: str,
    ) -> ProcessingResult:
        """Process email with pre-extracted content.

        Args:
            email: Email message
            content: Email body text

        Returns:
            ProcessingResult
        """
        try:
            # Normalize text
            content = self.text_extractor.normalize_text(content)

            # Build full text including metadata
            full_text = f"""
Subject: {email.subject}
From: {email.from_address}
To: {', '.join(email.to_addresses)}
Date: {email.date}

{content}
"""

            # Get existing entities
            existing_entities = self.node_manager.get_all_entity_names()

            # Extract entities
            entity_result = self.entity_extractor.extract_entities(
                full_text,
                email.id,
                existing_entities=existing_entities,
            )

            # Create email node
            content_hash = self.sync_manager.compute_hash(content)
            self.node_manager.create_email_node(
                email_id=email.id,
                thread_id=email.thread_id,
                subject=email.subject,
                from_address=email.from_address,
                to_addresses=email.to_addresses,
                date=email.date,
                content_hash=content_hash,
            )

            # Store entities and link to email
            for entity in entity_result.entities:
                self.node_manager.create_or_update_entity(entity, email.id)
                self.node_manager.link_entity_to_source(
                    entity.name,
                    entity.type,
                    email.id,
                    "email",
                    entity.confidence,
                )

            # Extract relationships
            rel_result = self.relationship_extractor.extract_relationships(
                full_text,
                entity_result.entities,
                email.id,
            )

            # Store relationships
            for rel in rel_result.relationships:
                self.relationship_manager.create_or_update_relationship(rel, email.id)

            # Create email-specific relationships
            self.relationship_manager.create_email_relationships(
                email.id,
                email.from_address,
                email.to_addresses,
            )

            logger.info(
                "email_processed",
                email_id=email.id,
                subject=email.subject[:50],
                entities=len(entity_result.entities),
                relationships=len(rel_result.relationships),
            )

            return ProcessingResult(
                item_id=email.id,
                item_type="email",
                success=True,
                entities_extracted=len(entity_result.entities),
                relationships_extracted=len(rel_result.relationships),
            )

        except Exception as e:
            logger.error("email_processing_failed", email_id=email.id, error=str(e))
            return ProcessingResult(
                item_id=email.id,
                item_type="email",
                success=False,
                error=str(e),
            )
