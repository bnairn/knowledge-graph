"""Neo4j database connection and transaction management."""

from contextlib import contextmanager
from typing import Any, Generator

from neo4j import GraphDatabase, Driver, Session, Result

from config.logging_config import get_logger

logger = get_logger(__name__)


class Neo4jClient:
    """Neo4j database client with connection management."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ):
        """Initialize the client.

        Args:
            uri: Neo4j connection URI (e.g., bolt://localhost:7687)
            username: Database username
            password: Database password
            database: Database name
        """
        self.uri = uri
        self.database = database
        self.driver: Driver = GraphDatabase.driver(uri, auth=(username, password))
        logger.info("neo4j_connected", uri=uri, database=database)

    def verify_connectivity(self) -> bool:
        """Verify database connectivity.

        Returns:
            True if connected successfully
        """
        try:
            self.driver.verify_connectivity()
            logger.info("neo4j_connectivity_verified")
            return True
        except Exception as e:
            logger.error("neo4j_connectivity_failed", error=str(e))
            return False

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Get a database session context manager.

        Yields:
            Neo4j Session
        """
        session = self.driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()

    def execute_read(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a read query.

        Args:
            query: Cypher query
            parameters: Query parameters

        Returns:
            List of result records as dictionaries
        """
        with self.session() as session:
            result = session.run(query, parameters or {})
            records = [record.data() for record in result]
            logger.debug("neo4j_read", query=query[:100], records=len(records))
            return records

    def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> Result:
        """Execute a write query.

        Args:
            query: Cypher query
            parameters: Query parameters

        Returns:
            Query result
        """
        with self.session() as session:
            result = session.run(query, parameters or {})
            summary = result.consume()
            logger.debug(
                "neo4j_write",
                query=query[:100],
                nodes_created=summary.counters.nodes_created,
                relationships_created=summary.counters.relationships_created,
            )
            return result

    def execute_batch(
        self,
        query: str,
        batch_data: list[dict[str, Any]],
        batch_size: int = 1000,
    ) -> int:
        """Execute query in batches for large datasets.

        Args:
            query: Cypher query with UNWIND $batch AS item
            batch_data: List of data items
            batch_size: Items per batch

        Returns:
            Total items processed
        """
        total_processed = 0

        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i : i + batch_size]
            with self.session() as session:
                session.run(query, {"batch": batch})
            total_processed += len(batch)
            logger.debug(
                "neo4j_batch_processed",
                processed=total_processed,
                total=len(batch_data),
            )

        return total_processed

    def close(self) -> None:
        """Close the driver connection."""
        self.driver.close()
        logger.info("neo4j_disconnected")

    def __enter__(self) -> "Neo4jClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
