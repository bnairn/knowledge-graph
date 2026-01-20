"""Command-line interface for the knowledge graph."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import get_settings
from config.logging_config import configure_logging

console = Console()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool) -> None:
    """Personal Knowledge Graph - Build a knowledge graph from your Google Drive and Gmail."""
    settings = get_settings()
    log_level = "DEBUG" if verbose else settings.log_level
    configure_logging(log_level, settings.log_file)


@main.command()
def auth() -> None:
    """Authenticate with Google APIs."""
    from src.auth import GoogleOAuthManager

    settings = get_settings()

    console.print("[bold]Starting Google OAuth authentication...[/bold]")
    console.print(f"Credentials file: {settings.google_credentials_file}")

    try:
        oauth_manager = GoogleOAuthManager(
            credentials_path=settings.google_credentials_file,
            token_path=settings.google_token_file,
        )
        credentials = oauth_manager.authenticate()
        console.print("[green]✓ Authentication successful![/green]")
        console.print(f"Token saved to: {settings.google_token_file}")
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("\n[yellow]To set up Google OAuth:[/yellow]")
        console.print("1. Go to console.cloud.google.com")
        console.print("2. Create a project and enable Drive & Gmail APIs")
        console.print("3. Create OAuth 2.0 credentials (Desktop app)")
        console.print(f"4. Download and save as {settings.google_credentials_file}")
    except Exception as e:
        console.print(f"[red]Authentication failed: {e}[/red]")


@main.command()
def setup_db() -> None:
    """Initialize Neo4j database schema."""
    from src.graph import Neo4jClient

    settings = get_settings()

    console.print("[bold]Setting up Neo4j schema...[/bold]")

    try:
        with Neo4jClient(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        ) as client:
            if not client.verify_connectivity():
                console.print("[red]Failed to connect to Neo4j[/red]")
                return

            # Create constraints and indexes
            schema_queries = [
                # Unique constraints
                "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
                "CREATE CONSTRAINT organization_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE",
                "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
                "CREATE CONSTRAINT project_name IF NOT EXISTS FOR (p:Project) REQUIRE p.name IS UNIQUE",
                "CREATE CONSTRAINT event_name IF NOT EXISTS FOR (e:Event) REQUIRE e.name IS UNIQUE",
                "CREATE CONSTRAINT location_name IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",
                "CREATE CONSTRAINT document_drive_id IF NOT EXISTS FOR (d:Document) REQUIRE d.google_drive_id IS UNIQUE",
                "CREATE CONSTRAINT email_id IF NOT EXISTS FOR (e:Email) REQUIRE e.id IS UNIQUE",
                # Indexes
                "CREATE INDEX document_modified IF NOT EXISTS FOR (d:Document) ON (d.modified_time)",
                "CREATE INDEX email_date IF NOT EXISTS FOR (e:Email) ON (e.date)",
            ]

            for query in schema_queries:
                try:
                    client.execute_write(query)
                    console.print(f"[green]✓[/green] {query[:60]}...")
                except Exception as e:
                    console.print(f"[yellow]⚠[/yellow] {query[:60]}... ({e})")

            console.print("\n[green]✓ Schema setup complete![/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@main.command()
@click.option("--full", is_flag=True, help="Run full sync instead of incremental")
@click.option("--drive-only", is_flag=True, help="Only sync Google Drive")
@click.option("--gmail-only", is_flag=True, help="Only sync Gmail")
@click.option("--limit", "-n", type=int, default=None, help="Limit number of items to process")
def sync(full: bool, drive_only: bool, gmail_only: bool, limit: int | None) -> None:
    """Sync data from Google Drive and Gmail."""
    from src.auth import GoogleOAuthManager
    from src.connectors import GoogleDriveConnector, GmailConnector
    from src.extractors import TextExtractor
    from src.nlp import ClaudeClient, OllamaClient, EntityExtractor, RelationshipExtractor
    from src.graph import Neo4jClient, NodeManager, RelationshipManager
    from src.pipeline import PipelineOrchestrator, SyncManager

    settings = get_settings()

    console.print("[bold]Starting knowledge graph sync...[/bold]\n")

    # Validate required settings based on provider
    if settings.llm_provider == "claude" and not settings.anthropic_api_key:
        console.print("[red]Error: ANTHROPIC_API_KEY not set (required for Claude)[/red]")
        return

    if not settings.neo4j_password:
        console.print("[red]Error: NEO4J_PASSWORD not set[/red]")
        return

    try:
        # Authenticate with Google
        oauth_manager = GoogleOAuthManager(
            credentials_path=settings.google_credentials_file,
            token_path=settings.google_token_file,
        )

        if not oauth_manager.is_authenticated():
            console.print("[yellow]Not authenticated. Running auth flow...[/yellow]")
            oauth_manager.authenticate()

        credentials = oauth_manager.get_credentials()

        # Initialize connectors
        drive_connector = None
        gmail_connector = None

        if not gmail_only:
            drive_connector = GoogleDriveConnector(credentials)
            console.print("[green]✓[/green] Connected to Google Drive")

        if not drive_only:
            gmail_connector = GmailConnector(credentials)
            console.print("[green]✓[/green] Connected to Gmail")

        # Initialize extractors
        text_extractor = TextExtractor()

        # Initialize LLM client based on provider
        if settings.llm_provider == "ollama":
            llm_client = OllamaClient(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                timeout=settings.ollama_timeout,
            )
            if not llm_client.check_connection():
                console.print("[red]Error: Cannot connect to Ollama. Is it running?[/red]")
                console.print("[yellow]Start Ollama with: ollama serve[/yellow]")
                console.print(f"[yellow]Pull model with: ollama pull {settings.ollama_model}[/yellow]")
                return
            console.print(f"[green]✓[/green] Ollama client initialized ({settings.ollama_model})")
        else:
            llm_client = ClaudeClient(
                api_key=settings.anthropic_api_key,
                model=settings.claude_model,
                max_tokens=settings.claude_max_tokens,
                requests_per_minute=settings.claude_requests_per_minute,
            )
            console.print("[green]✓[/green] Claude client initialized")

        entity_extractor = EntityExtractor(llm_client)
        relationship_extractor = RelationshipExtractor(llm_client)

        # Initialize Neo4j
        neo4j_client = Neo4jClient(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        console.print("[green]✓[/green] Connected to Neo4j")

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

        console.print()

        # Run sync
        if full:
            console.print("[bold]Running full sync...[/bold]")
            if limit:
                console.print(f"[dim]Limited to {limit} items[/dim]")
            report = orchestrator.run_full_sync(limit=limit)
        else:
            console.print("[bold]Running incremental sync...[/bold]")
            if limit:
                console.print(f"[dim]Limited to {limit} items[/dim]")
            report = orchestrator.run_incremental_sync(limit=limit)

        # Display report
        console.print()
        table = Table(title="Sync Report")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Documents Processed", str(report.documents_processed))
        table.add_row("Emails Processed", str(report.emails_processed))
        table.add_row("Entities Extracted", str(report.total_entities))
        table.add_row("Relationships Extracted", str(report.total_relationships))
        table.add_row("Errors", str(len(report.errors)))

        if report.completed_at and report.started_at:
            duration = report.completed_at - report.started_at
            table.add_row("Duration", str(duration))

        console.print(table)

        if report.errors:
            console.print("\n[yellow]Errors:[/yellow]")
            for error in report.errors[:10]:
                console.print(f"  - {error}")
            if len(report.errors) > 10:
                console.print(f"  ... and {len(report.errors) - 10} more")

        # Show LLM usage
        usage = llm_client.get_usage_stats()
        if settings.llm_provider == "claude":
            console.print(f"\n[dim]Claude API usage: {usage['total_tokens']:,} tokens[/dim]")

        neo4j_client.close()

    except Exception as e:
        console.print(f"[red]Sync failed: {e}[/red]")
        raise


@main.command()
@click.argument("query")
@click.option("--type", "-t", "entity_type", help="Filter by entity type")
def search(query: str, entity_type: str | None) -> None:
    """Search for entities in the knowledge graph."""
    from src.graph import Neo4jClient
    from src.query import QueryInterface

    settings = get_settings()

    try:
        with Neo4jClient(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        ) as client:
            qi = QueryInterface(client)

            types = [entity_type] if entity_type else None
            results = qi.search_entities(query, types)

            if not results:
                console.print(f"[yellow]No entities found matching '{query}'[/yellow]")
                return

            table = Table(title=f"Search Results: '{query}'")
            table.add_column("Name", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Description")
            table.add_column("Confidence", justify="right")

            for row in results:
                table.add_row(
                    row["name"],
                    row["type"],
                    (row.get("description") or "")[:50],
                    f"{row.get('confidence', 0):.2f}",
                )

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@main.command()
@click.argument("name")
def entity(name: str) -> None:
    """Get details about a specific entity."""
    from src.graph import Neo4jClient
    from src.query import QueryInterface

    settings = get_settings()

    try:
        with Neo4jClient(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        ) as client:
            qi = QueryInterface(client)

            details = qi.get_entity_details(name)

            if not details:
                console.print(f"[yellow]Entity '{name}' not found[/yellow]")
                return

            panel = Panel(
                f"""[bold]{details['name']}[/bold] ({details['type']})

{details.get('description') or 'No description'}

Aliases: {', '.join(details.get('aliases', [])) or 'None'}
Confidence: {details.get('confidence', 'N/A')}
""",
                title="Entity Details",
            )
            console.print(panel)

            if details.get("outgoing_relationships"):
                console.print("\n[bold]Outgoing Relationships:[/bold]")
                for rel in details["outgoing_relationships"][:10]:
                    console.print(f"  --[{rel['type']}]--> {rel['target']}")

            if details.get("incoming_relationships"):
                console.print("\n[bold]Incoming Relationships:[/bold]")
                for rel in details["incoming_relationships"][:10]:
                    console.print(f"  <--[{rel['type']}]-- {rel['source']}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@main.command()
def stats() -> None:
    """Show knowledge graph statistics."""
    from src.graph import Neo4jClient
    from src.query import QueryInterface
    from src.pipeline import SyncManager

    settings = get_settings()

    try:
        with Neo4jClient(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        ) as client:
            qi = QueryInterface(client)
            graph_stats = qi.get_statistics()

            table = Table(title="Knowledge Graph Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Count", justify="right", style="green")

            for key, value in sorted(graph_stats.items()):
                label = key.replace("_", " ").title()
                table.add_row(label, f"{value:,}")

            console.print(table)

        # Sync stats
        sync_manager = SyncManager(settings.sync_state_file)
        sync_stats = sync_manager.get_stats()

        console.print("\n[bold]Sync Status:[/bold]")
        console.print(f"  Drive last sync: {sync_stats['drive_last_sync'] or 'Never'}")
        console.print(f"  Gmail last sync: {sync_stats['gmail_last_sync'] or 'Never'}")
        console.print(f"  Total processed items: {sync_stats['total_processed_items']}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
