#!/usr/bin/env python3
"""Run knowledge graph sync from Google Drive and Gmail."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings
from config.logging_config import configure_logging


def main():
    parser = argparse.ArgumentParser(
        description="Sync knowledge graph from Google Drive and Gmail"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full sync instead of incremental",
    )
    parser.add_argument(
        "--drive-only",
        action="store_true",
        help="Only sync Google Drive",
    )
    parser.add_argument(
        "--gmail-only",
        action="store_true",
        help="Only sync Gmail",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: authenticate and list 5 files, don't process",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    settings = get_settings()
    log_level = "DEBUG" if args.verbose else settings.log_level
    configure_logging(log_level)

    # Validate settings
    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in environment or .env file")
        sys.exit(1)

    if not settings.neo4j_password:
        print("ERROR: NEO4J_PASSWORD not set in environment or .env file")
        sys.exit(1)

    from src.auth import GoogleOAuthManager
    from src.connectors import GoogleDriveConnector, GmailConnector

    # Authenticate
    print("Authenticating with Google...")
    oauth_manager = GoogleOAuthManager(
        credentials_path=settings.google_credentials_file,
        token_path=settings.google_token_file,
    )

    if not oauth_manager.is_authenticated():
        print("Not authenticated. Starting OAuth flow...")
        oauth_manager.authenticate()

    credentials = oauth_manager.get_credentials()
    print("✓ Authenticated successfully\n")

    # Test mode
    if args.test:
        print("Test mode: listing files...\n")

        if not args.gmail_only:
            print("Google Drive files:")
            drive = GoogleDriveConnector(credentials)
            count = 0
            for file in drive.list_items():
                print(f"  - {file.name} ({file.mime_type})")
                count += 1
                if count >= 5:
                    break
            print()

        if not args.drive_only:
            print("Gmail messages:")
            gmail = GmailConnector(credentials)
            count = 0
            for msg in gmail.list_items():
                print(f"  - {msg.subject} (from: {msg.from_address})")
                count += 1
                if count >= 5:
                    break

        print("\n✓ Test completed successfully")
        return

    # Full sync
    from src.extractors import TextExtractor
    from src.nlp import ClaudeClient, EntityExtractor, RelationshipExtractor
    from src.graph import Neo4jClient, NodeManager, RelationshipManager
    from src.pipeline import PipelineOrchestrator, SyncManager

    # Initialize connectors
    drive_connector = None
    gmail_connector = None

    if not args.gmail_only:
        drive_connector = GoogleDriveConnector(credentials)
        print("✓ Connected to Google Drive")

    if not args.drive_only:
        gmail_connector = GmailConnector(credentials)
        print("✓ Connected to Gmail")

    # Initialize extractors
    text_extractor = TextExtractor()

    # Initialize Claude
    claude_client = ClaudeClient(
        api_key=settings.anthropic_api_key,
        model=settings.claude_model,
        max_tokens=settings.claude_max_tokens,
        requests_per_minute=settings.claude_requests_per_minute,
    )
    print("✓ Claude client initialized")

    entity_extractor = EntityExtractor(claude_client)
    relationship_extractor = RelationshipExtractor(claude_client)

    # Initialize Neo4j
    neo4j_client = Neo4jClient(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )

    if not neo4j_client.verify_connectivity():
        print("ERROR: Failed to connect to Neo4j")
        sys.exit(1)

    print("✓ Connected to Neo4j")

    node_manager = NodeManager(neo4j_client)
    relationship_manager = RelationshipManager(neo4j_client)

    # Initialize sync manager
    sync_manager = SyncManager(settings.sync_state_file)

    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        drive_connector=drive_connector,
        gmail_connector=gmail_connector,
        text_extractor=text_extractor,
        entity_extractor=entity_extractor,
        relationship_extractor=relationship_extractor,
        node_manager=node_manager,
        relationship_manager=relationship_manager,
        sync_manager=sync_manager,
    )

    print()

    # Run sync
    if args.full:
        print("Running full sync...")
        report = orchestrator.run_full_sync()
    else:
        print("Running incremental sync...")
        report = orchestrator.run_incremental_sync()

    # Print report
    print("\n" + "=" * 50)
    print("SYNC REPORT")
    print("=" * 50)
    print(f"Documents processed: {report.documents_processed}")
    print(f"Emails processed: {report.emails_processed}")
    print(f"Entities extracted: {report.total_entities}")
    print(f"Relationships extracted: {report.total_relationships}")
    print(f"Errors: {len(report.errors)}")

    if report.completed_at and report.started_at:
        duration = report.completed_at - report.started_at
        print(f"Duration: {duration}")

    if report.errors:
        print("\nErrors:")
        for error in report.errors[:10]:
            print(f"  - {error}")

    # Claude usage
    usage = claude_client.get_usage_stats()
    print(f"\nClaude API usage: {usage['total_tokens']:,} tokens")

    neo4j_client.close()
    print("\n✓ Sync complete")


if __name__ == "__main__":
    main()
