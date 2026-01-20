"""Query interface for exploring the knowledge graph."""

from typing import Any

from src.graph.neo4j_client import Neo4jClient

from config.logging_config import get_logger

logger = get_logger(__name__)


class QueryInterface:
    """Interface for querying the knowledge graph."""

    def __init__(self, neo4j_client: Neo4jClient):
        """Initialize the query interface.

        Args:
            neo4j_client: Connected Neo4j client
        """
        self.client = neo4j_client

    def search_entities(
        self,
        query: str,
        entity_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search for entities by name.

        Args:
            query: Search query (name or partial match)
            entity_types: Optional list of entity types to filter
            limit: Maximum results

        Returns:
            List of matching entities
        """
        type_filter = ""
        if entity_types:
            labels = " OR ".join(f"e:{t.capitalize()}" for t in entity_types)
            type_filter = f"AND ({labels})"

        cypher = f"""
        MATCH (e)
        WHERE (e:Person OR e:Organization OR e:Concept OR e:Topic
               OR e:Project OR e:Event OR e:Location)
          AND (toLower(e.name) CONTAINS toLower($query)
               OR ANY(alias IN e.aliases WHERE toLower(alias) CONTAINS toLower($query)))
          {type_filter}
        RETURN e.name AS name,
               labels(e)[0] AS type,
               e.description AS description,
               e.confidence AS confidence
        ORDER BY e.confidence DESC
        LIMIT $limit
        """

        return self.client.execute_read(cypher, {"query": query, "limit": limit})

    def get_entity_details(self, name: str) -> dict[str, Any] | None:
        """Get detailed information about an entity.

        Args:
            name: Entity name

        Returns:
            Entity details or None
        """
        cypher = """
        MATCH (e)
        WHERE (e:Person OR e:Organization OR e:Concept OR e:Topic
               OR e:Project OR e:Event OR e:Location)
          AND (e.name = $name OR $name IN e.aliases)
        OPTIONAL MATCH (e)-[r]->(related)
        OPTIONAL MATCH (e)<-[r2]-(related2)
        RETURN e,
               collect(DISTINCT {type: type(r), target: related.name, labels: labels(related)}) AS outgoing,
               collect(DISTINCT {type: type(r2), source: related2.name, labels: labels(related2)}) AS incoming
        LIMIT 1
        """

        results = self.client.execute_read(cypher, {"name": name})
        if not results:
            return None

        result = results[0]
        entity = result["e"]
        return {
            "name": entity.get("name"),
            "type": list(entity.labels)[0] if hasattr(entity, "labels") else "Unknown",
            "description": entity.get("description"),
            "aliases": entity.get("aliases", []),
            "confidence": entity.get("confidence"),
            "outgoing_relationships": [r for r in result["outgoing"] if r["target"]],
            "incoming_relationships": [r for r in result["incoming"] if r["source"]],
        }

    def get_entity_network(
        self,
        name: str,
        depth: int = 2,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get an entity's network up to N hops.

        Args:
            name: Entity name
            depth: Maximum hops
            limit: Maximum nodes

        Returns:
            Network with nodes and edges
        """
        cypher = f"""
        MATCH (start)
        WHERE (start:Person OR start:Organization OR start:Concept OR start:Topic
               OR start:Project OR start:Event OR start:Location)
          AND (start.name = $name OR $name IN start.aliases)
        CALL apoc.path.subgraphAll(start, {{
            maxLevel: $depth,
            limit: $limit
        }})
        YIELD nodes, relationships
        RETURN nodes, relationships
        """

        # Fallback if APOC not available
        fallback_cypher = f"""
        MATCH (start)
        WHERE (start:Person OR start:Organization OR start:Concept OR start:Topic
               OR start:Project OR start:Event OR start:Location)
          AND (start.name = $name OR $name IN start.aliases)
        MATCH path = (start)-[*1..{depth}]-(connected)
        WITH start, collect(DISTINCT connected) AS nodes,
             collect(DISTINCT relationships(path)) AS rels
        RETURN start, nodes, rels
        LIMIT $limit
        """

        try:
            results = self.client.execute_read(
                cypher,
                {"name": name, "depth": depth, "limit": limit},
            )
        except Exception:
            # APOC not available, use fallback
            results = self.client.execute_read(
                fallback_cypher,
                {"name": name, "depth": depth, "limit": limit},
            )

        if not results:
            return {"nodes": [], "edges": []}

        # Process results into nodes and edges
        nodes = []
        edges = []
        seen_nodes = set()

        for record in results:
            for node in record.get("nodes", []):
                if hasattr(node, "element_id"):
                    node_id = node.element_id
                else:
                    node_id = node.get("name", str(node))

                if node_id not in seen_nodes:
                    seen_nodes.add(node_id)
                    nodes.append({
                        "id": node_id,
                        "name": node.get("name"),
                        "type": list(node.labels)[0] if hasattr(node, "labels") else "Unknown",
                    })

        return {"nodes": nodes, "edges": edges}

    def find_path(
        self,
        entity1: str,
        entity2: str,
        max_hops: int = 5,
    ) -> list[dict[str, Any]]:
        """Find paths between two entities.

        Args:
            entity1: First entity name
            entity2: Second entity name
            max_hops: Maximum path length

        Returns:
            List of paths
        """
        cypher = f"""
        MATCH (e1), (e2)
        WHERE (e1:Person OR e1:Organization OR e1:Concept OR e1:Topic
               OR e1:Project OR e1:Event OR e1:Location)
          AND (e2:Person OR e2:Organization OR e2:Concept OR e2:Topic
               OR e2:Project OR e2:Event OR e2:Location)
          AND (e1.name = $entity1 OR $entity1 IN e1.aliases)
          AND (e2.name = $entity2 OR $entity2 IN e2.aliases)
        MATCH path = shortestPath((e1)-[*1..{max_hops}]-(e2))
        RETURN [node IN nodes(path) | node.name] AS path_nodes,
               [rel IN relationships(path) | type(rel)] AS path_relationships
        LIMIT 5
        """

        return self.client.execute_read(
            cypher,
            {"entity1": entity1, "entity2": entity2},
        )

    def get_entity_sources(self, name: str) -> list[dict[str, Any]]:
        """Get documents/emails where an entity was mentioned.

        Args:
            name: Entity name

        Returns:
            List of sources
        """
        cypher = """
        MATCH (e)-[r:EXTRACTED_FROM]->(source)
        WHERE (e:Person OR e:Organization OR e:Concept OR e:Topic
               OR e:Project OR e:Event OR e:Location)
          AND (e.name = $name OR $name IN e.aliases)
        RETURN CASE
                 WHEN source:Document THEN 'document'
                 WHEN source:Email THEN 'email'
                 ELSE 'unknown'
               END AS source_type,
               source.name AS name,
               source.subject AS subject,
               r.confidence AS confidence,
               r.contexts AS contexts
        ORDER BY r.confidence DESC
        """

        return self.client.execute_read(cypher, {"name": name})

    def get_statistics(self) -> dict[str, int]:
        """Get graph statistics.

        Returns:
            Dictionary with counts
        """
        cypher = """
        MATCH (n)
        WITH labels(n)[0] AS label, count(*) AS count
        RETURN label, count
        ORDER BY count DESC
        """

        results = self.client.execute_read(cypher)

        stats = {"total_nodes": 0}
        for row in results:
            label = row["label"]
            count = row["count"]
            stats[f"{label.lower()}_count"] = count
            stats["total_nodes"] += count

        # Count relationships
        rel_cypher = "MATCH ()-[r]->() RETURN count(r) AS count"
        rel_results = self.client.execute_read(rel_cypher)
        stats["total_relationships"] = rel_results[0]["count"] if rel_results else 0

        return stats

    def execute_cypher(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a custom Cypher query.

        Args:
            query: Cypher query
            params: Query parameters

        Returns:
            Query results
        """
        logger.info("executing_custom_cypher", query=query[:100])
        return self.client.execute_read(query, params or {})
