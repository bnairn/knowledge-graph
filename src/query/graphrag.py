"""GraphRAG query interface using neo4j-graphrag."""

import re

import httpx
from neo4j import Driver

from config.logging_config import get_logger

logger = get_logger(__name__)

# Patterns that indicate aggregate/counting questions
AGGREGATE_PATTERNS = [
    r"\bhow many\b",
    r"\bcount\b",
    r"\btotal\b",
    r"\bnumber of\b",
    r"\bhow much\b",
    r"\ball the\b",
    r"\blist all\b",
    r"\bshow all\b",
    r"\bevery\b",
    r"\bmost common\b",
    r"\btop \d+\b",
    r"\bfrequent\b",
]

# Entity type keywords mapping
ENTITY_KEYWORDS = {
    "Person": ["person", "people", "contact", "contacts", "who", "someone", "anyone", "names"],
    "Organization": ["organization", "organizations", "company", "companies", "org", "orgs"],
    "Concept": ["concept", "concepts", "technology", "technologies", "tech"],
    "Topic": ["topic", "topics", "subject", "subjects", "theme", "themes"],
    "Project": ["project", "projects"],
    "Event": ["event", "events", "meeting", "meetings"],
    "Location": ["location", "locations", "place", "places", "where"],
    "Email": ["email", "emails", "message", "messages"],
}


class OllamaEmbedder:
    """Generate embeddings using Ollama."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
    ):
        """Initialize embedder.

        Args:
            base_url: Ollama API base URL
            model: Embedding model to use
        """
        self.base_url = base_url
        self.model = model
        self.client = httpx.Client(timeout=60.0)
        self.dimensions = 768  # nomic-embed-text dimensions

    def embed(self, text: str) -> list[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        response = self.client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return [self.embed(text) for text in texts]


class GraphRAGQueryEngine:
    """Natural language query engine using GraphRAG."""

    def __init__(
        self,
        driver: Driver,
        ollama_base_url: str = "http://localhost:11434",
        llm_model: str = "llama3.1:8b",
        embedding_model: str = "nomic-embed-text",
    ):
        """Initialize the query engine.

        Args:
            driver: Neo4j driver
            ollama_base_url: Ollama API base URL
            llm_model: LLM model for generation
            embedding_model: Model for embeddings
        """
        self.driver = driver
        self.ollama_base_url = ollama_base_url
        self.llm_model = llm_model
        self.embedder = OllamaEmbedder(ollama_base_url, embedding_model)
        self.client = httpx.Client(timeout=300.0)

    def setup_vector_index(self) -> None:
        """Create vector index in Neo4j if it doesn't exist."""
        with self.driver.session() as session:
            # Check Neo4j version supports vector
            result = session.run("CALL dbms.components()")
            version = result.single()["versions"][0]
            logger.info("neo4j_version", version=version)

            # Create vector index for entities
            try:
                session.run("""
                    CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
                    FOR (n:Entity)
                    ON (n.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 768,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)
                logger.info("vector_index_created", index="entity_embeddings")
            except Exception as e:
                logger.warning("vector_index_exists_or_error", error=str(e))

    def generate_embeddings_for_nodes(self, batch_size: int = 50) -> int:
        """Generate embeddings for all entity nodes.

        Args:
            batch_size: Number of nodes to process at once

        Returns:
            Number of nodes updated
        """
        total_updated = 0

        # Get all entity types
        entity_labels = ["Person", "Organization", "Concept", "Topic", "Project", "Event", "Location"]

        for label in entity_labels:
            with self.driver.session() as session:
                # Get nodes without embeddings
                result = session.run(f"""
                    MATCH (n:{label})
                    WHERE n.embedding IS NULL
                    RETURN n.name as name, elementId(n) as id
                    LIMIT $batch_size
                """, batch_size=batch_size)

                nodes = list(result)
                if not nodes:
                    continue

                logger.info("generating_embeddings", label=label, count=len(nodes))

                for node in nodes:
                    try:
                        # Create text representation for embedding
                        text = f"{label}: {node['name']}"
                        embedding = self.embedder.embed(text)

                        # Store embedding
                        session.run("""
                            MATCH (n)
                            WHERE elementId(n) = $id
                            SET n.embedding = $embedding
                        """, id=node["id"], embedding=embedding)

                        total_updated += 1
                    except Exception as e:
                        logger.error("embedding_failed", name=node["name"], error=str(e))

        logger.info("embeddings_complete", total=total_updated)
        return total_updated

    def search_similar(self, query: str, limit: int = 10) -> list[dict]:
        """Find nodes similar to query using vector search.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of similar nodes with scores
        """
        query_embedding = self.embedder.embed(query)

        with self.driver.session() as session:
            # Use manual cosine similarity since we have multiple labels
            try:
                result = session.run("""
                    MATCH (n)
                    WHERE n.embedding IS NOT NULL AND n.name IS NOT NULL
                    WITH n,
                         reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) |
                                dot + n.embedding[i] * $embedding[i]) AS dotProduct,
                         reduce(norm1 = 0.0, i IN range(0, size(n.embedding)-1) |
                                norm1 + n.embedding[i] * n.embedding[i]) AS norm1,
                         reduce(norm2 = 0.0, i IN range(0, size($embedding)-1) |
                                norm2 + $embedding[i] * $embedding[i]) AS norm2
                    WITH n, dotProduct / (sqrt(norm1) * sqrt(norm2)) AS similarity
                    WHERE similarity > 0.3
                    RETURN labels(n)[0] as type, n.name as name, similarity as score
                    ORDER BY similarity DESC
                    LIMIT $limit
                """, embedding=query_embedding, limit=limit)

                results = [dict(r) for r in result]
                if results:
                    return results
            except Exception as e:
                logger.warning("vector_search_failed", error=str(e))

            # Fallback to text search
            return self._text_search(query, limit)

    def _text_search(self, query: str, limit: int) -> list[dict]:
        """Fallback text search when vector search unavailable."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.name IS NOT NULL
                AND toLower(n.name) CONTAINS toLower($query)
                RETURN labels(n)[0] as type, n.name as name, 1.0 as score
                LIMIT $limit
            """, query=query, limit=limit)
            return [dict(r) for r in result]

    def _is_aggregate_question(self, question: str) -> bool:
        """Check if the question is asking for counts or aggregates."""
        q_lower = question.lower()
        for pattern in AGGREGATE_PATTERNS:
            if re.search(pattern, q_lower):
                return True
        return False

    def _detect_entity_type(self, question: str) -> str | None:
        """Detect which entity type the question is asking about."""
        q_lower = question.lower()
        for entity_type, keywords in ENTITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in q_lower:
                    return entity_type
        return None

    def _answer_aggregate_question(self, question: str) -> str:
        """Answer aggregate/counting questions using direct Cypher queries."""
        q_lower = question.lower()
        entity_type = self._detect_entity_type(question)

        with self.driver.session() as session:
            # Count questions
            if any(re.search(p, q_lower) for p in [r"\bhow many\b", r"\bcount\b", r"\btotal\b", r"\bnumber of\b"]):
                if entity_type:
                    result = session.run(f"""
                        MATCH (n:{entity_type})
                        RETURN count(n) as count
                    """)
                    count = result.single()["count"]
                    return f"There are {count} {entity_type} nodes in the knowledge graph."
                else:
                    # Count all entity types
                    result = session.run("""
                        MATCH (n)
                        WHERE n.name IS NOT NULL
                        WITH labels(n)[0] as type, count(*) as count
                        RETURN type, count
                        ORDER BY count DESC
                    """)
                    counts = [f"- {r['type']}: {r['count']}" for r in result]
                    return "Entity counts in the knowledge graph:\n" + "\n".join(counts)

            # List all questions
            if any(re.search(p, q_lower) for p in [r"\ball the\b", r"\blist all\b", r"\bshow all\b", r"\bevery\b"]):
                if entity_type:
                    result = session.run(f"""
                        MATCH (n:{entity_type})
                        RETURN n.name as name
                        ORDER BY n.name
                        LIMIT 100
                    """)
                    names = [r["name"] for r in result]
                    if len(names) == 100:
                        return f"First 100 {entity_type} nodes (showing partial list):\n" + "\n".join(f"- {n}" for n in names)
                    return f"All {len(names)} {entity_type} nodes:\n" + "\n".join(f"- {n}" for n in names)

            # Most common/frequent questions
            if any(re.search(p, q_lower) for p in [r"\bmost common\b", r"\btop \d+\b", r"\bfrequent\b"]):
                # Extract limit from "top N" if present
                limit_match = re.search(r"\btop (\d+)\b", q_lower)
                limit = int(limit_match.group(1)) if limit_match else 10

                if entity_type == "Topic" or "topic" in q_lower:
                    result = session.run("""
                        MATCH (t:Topic)-[:EXTRACTED_FROM]->(e:Email)
                        RETURN t.name as topic, count(e) as email_count
                        ORDER BY email_count DESC
                        LIMIT $limit
                    """, limit=limit)
                    topics = [f"- {r['topic']}: {r['email_count']} emails" for r in result]
                    if topics:
                        return f"Top {limit} most discussed topics:\n" + "\n".join(topics)
                    return "No topics with email associations found."

                if entity_type == "Person" or "person" in q_lower or "contact" in q_lower:
                    result = session.run("""
                        MATCH (p:Person)-[:EXTRACTED_FROM]->(e:Email)
                        RETURN p.name as person, count(e) as email_count
                        ORDER BY email_count DESC
                        LIMIT $limit
                    """, limit=limit)
                    people = [f"- {r['person']}: {r['email_count']} emails" for r in result]
                    if people:
                        return f"Top {limit} most mentioned people:\n" + "\n".join(people)
                    return "No people with email associations found."

                # Generic: most common entity types
                result = session.run("""
                    MATCH (n)-[:EXTRACTED_FROM]->(e:Email)
                    WHERE n.name IS NOT NULL
                    RETURN labels(n)[0] as type, n.name as name, count(e) as email_count
                    ORDER BY email_count DESC
                    LIMIT $limit
                """, limit=limit)
                items = [f"- {r['name']} ({r['type']}): {r['email_count']} emails" for r in result]
                if items:
                    return f"Top {limit} most mentioned entities:\n" + "\n".join(items)
                return "No entities with email associations found."

        # Fallback - didn't match any specific aggregate pattern
        return None

    def get_node_context(self, name: str, depth: int = 2) -> str:
        """Get graph context around a node.

        Args:
            name: Node name
            depth: How many hops to traverse

        Returns:
            Text description of node's context
        """
        with self.driver.session() as session:
            # Get the node and its relationships
            result = session.run("""
                MATCH (n {name: $name})
                OPTIONAL MATCH (n)-[r]-(connected)
                RETURN labels(n)[0] as type, n.name as name,
                       type(r) as rel_type,
                       labels(connected)[0] as connected_type,
                       connected.name as connected_name
                LIMIT 50
            """, name=name)

            records = list(result)
            if not records:
                return f"No information found about '{name}'"

            # Build context string
            node_type = records[0]["type"]
            context_parts = [f"{name} is a {node_type}."]

            relationships = []
            for r in records:
                if r["connected_name"]:
                    rel = r["rel_type"].replace("_", " ").lower()
                    relationships.append(f"- {rel} {r['connected_name']} ({r['connected_type']})")

            if relationships:
                context_parts.append("Connections:")
                context_parts.extend(relationships[:20])  # Limit context size

            return "\n".join(context_parts)

    def ask(self, question: str) -> str:
        """Answer a natural language question using GraphRAG.

        Args:
            question: Natural language question

        Returns:
            Generated answer
        """
        # Check if this is an aggregate question (counting, listing, etc.)
        if self._is_aggregate_question(question):
            logger.info("detected_aggregate_question", question=question[:50])
            aggregate_answer = self._answer_aggregate_question(question)
            if aggregate_answer:
                return aggregate_answer
            # If aggregate handler didn't match, fall through to semantic search

        # 1. Find relevant nodes via semantic search
        similar_nodes = self.search_similar(question, limit=5)

        if not similar_nodes:
            return "I couldn't find any relevant information in the knowledge graph."

        # 2. Get context from graph for top results
        contexts = []
        for node in similar_nodes[:3]:
            context = self.get_node_context(node["name"])
            contexts.append(context)

        graph_context = "\n\n".join(contexts)

        # 3. Generate answer using LLM
        prompt = f"""Based on the following information from a personal knowledge graph, answer the question.

Knowledge Graph Context:
{graph_context}

Question: {question}

Answer the question based only on the information provided. If the information doesn't contain the answer, say so.
Be concise and direct."""

        response = self.client.post(
            f"{self.ollama_base_url}/api/chat",
            json={
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        response.raise_for_status()

        answer = response.json().get("message", {}).get("content", "")
        logger.info("question_answered", question=question[:50])

        return answer

    def find_connections(self, entity1: str, entity2: str) -> str:
        """Find how two entities are connected.

        Args:
            entity1: First entity name
            entity2: Second entity name

        Returns:
            Description of connections
        """
        with self.driver.session() as session:
            # Find shortest paths
            result = session.run("""
                MATCH (a {name: $entity1}), (b {name: $entity2})
                MATCH path = shortestPath((a)-[*..5]-(b))
                RETURN [n in nodes(path) | n.name] as nodes,
                       [r in relationships(path) | type(r)] as relationships
                LIMIT 5
            """, entity1=entity1, entity2=entity2)

            paths = list(result)
            if not paths:
                return f"No connection found between '{entity1}' and '{entity2}'"

            # Format the paths
            descriptions = []
            for path in paths:
                nodes = path["nodes"]
                rels = path["relationships"]
                path_parts = []
                for i, node in enumerate(nodes):
                    path_parts.append(node)
                    if i < len(rels):
                        path_parts.append(f"--[{rels[i]}]-->")
                descriptions.append(" ".join(path_parts))

            return f"Connections between {entity1} and {entity2}:\n" + "\n".join(descriptions)
