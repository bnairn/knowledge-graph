#!/usr/bin/env python3
"""Initialize Neo4j database schema with constraints and indexes."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings
from config.logging_config import configure_logging
from src.graph import Neo4jClient


SCHEMA_QUERIES = [
    # Unique constraints on entity names
    "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT organization_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE",
    "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT project_name IF NOT EXISTS FOR (p:Project) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT event_name IF NOT EXISTS FOR (e:Event) REQUIRE e.name IS UNIQUE",
    "CREATE CONSTRAINT location_name IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",

    # Unique constraints on source IDs
    "CREATE CONSTRAINT document_drive_id IF NOT EXISTS FOR (d:Document) REQUIRE d.google_drive_id IS UNIQUE",
    "CREATE CONSTRAINT email_id IF NOT EXISTS FOR (e:Email) REQUIRE e.id IS UNIQUE",

    # Performance indexes
    "CREATE INDEX person_confidence IF NOT EXISTS FOR (p:Person) ON (p.confidence)",
    "CREATE INDEX organization_confidence IF NOT EXISTS FOR (o:Organization) ON (o.confidence)",
    "CREATE INDEX document_modified IF NOT EXISTS FOR (d:Document) ON (d.modified_time)",
    "CREATE INDEX document_sync_status IF NOT EXISTS FOR (d:Document) ON (d.sync_status)",
    "CREATE INDEX email_date IF NOT EXISTS FOR (e:Email) ON (e.date)",
    "CREATE INDEX email_thread IF NOT EXISTS FOR (e:Email) ON (e.thread_id)",
    "CREATE INDEX email_sync_status IF NOT EXISTS FOR (e:Email) ON (e.sync_status)",
]


def main():
    """Set up Neo4j schema."""
    settings = get_settings()
    configure_logging(settings.log_level)

    print("Setting up Neo4j schema...")
    print(f"URI: {settings.neo4j_uri}")
    print(f"Database: {settings.neo4j_database}")
    print()

    if not settings.neo4j_password:
        print("ERROR: NEO4J_PASSWORD environment variable not set")
        sys.exit(1)

    try:
        with Neo4jClient(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        ) as client:
            if not client.verify_connectivity():
                print("ERROR: Failed to connect to Neo4j")
                sys.exit(1)

            print("Connected to Neo4j successfully!\n")

            success_count = 0
            error_count = 0

            for query in SCHEMA_QUERIES:
                try:
                    client.execute_write(query)
                    print(f"✓ {query[:70]}...")
                    success_count += 1
                except Exception as e:
                    print(f"✗ {query[:70]}... ({e})")
                    error_count += 1

            print(f"\nSchema setup complete: {success_count} succeeded, {error_count} failed")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
