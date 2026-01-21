"""Entity deduplication for knowledge graph."""

import re

from neo4j import Driver

from config.logging_config import get_logger

logger = get_logger(__name__)


class EntityDeduplicator:
    """Deduplicate and merge similar entities in the graph."""

    def __init__(self, driver: Driver):
        self.driver = driver

    def merge_exact_duplicates(self) -> int:
        """Merge nodes with identical names (case-insensitive).

        Returns:
            Number of nodes merged
        """
        merged = 0

        with self.driver.session() as session:
            # Find groups of duplicate Person nodes
            result = session.run("""
                MATCH (p:Person)
                WITH toLower(p.name) as lower_name, collect(p) as nodes
                WHERE size(nodes) > 1
                RETURN lower_name, nodes
            """)

            for record in result:
                nodes = record["nodes"]
                # Keep the first node, merge others into it
                primary = nodes[0]
                for duplicate in nodes[1:]:
                    self._merge_nodes(session, primary, duplicate)
                    merged += 1

            # Same for Organizations
            result = session.run("""
                MATCH (o:Organization)
                WITH toLower(o.name) as lower_name, collect(o) as nodes
                WHERE size(nodes) > 1
                RETURN lower_name, nodes
            """)

            for record in result:
                nodes = record["nodes"]
                primary = nodes[0]
                for duplicate in nodes[1:]:
                    self._merge_nodes(session, primary, duplicate)
                    merged += 1

        logger.info("merged_exact_duplicates", count=merged)
        return merged

    def merge_email_to_person(self) -> int:
        """Merge person nodes that are email addresses into their named counterparts.

        For example, merge 'bryan.nairn@gmail.com' into 'Bryan Nairn'.

        Returns:
            Number of nodes merged
        """
        merged = 0

        with self.driver.session() as session:
            # Find email-like person names
            result = session.run("""
                MATCH (email_person:Person)
                WHERE email_person.name CONTAINS '@'
                RETURN email_person
            """)

            email_persons = [r["email_person"] for r in result]

            for email_person in email_persons:
                email = email_person["name"].lower()
                # Extract name from email (before @)
                local_part = email.split("@")[0]
                # Handle formats like "bryan.nairn" or "bryannairn"
                name_parts = re.split(r"[._]", local_part)

                if len(name_parts) >= 2:
                    # Try to find matching person by first.last name
                    first = name_parts[0]
                    last = name_parts[-1]

                    match_result = session.run("""
                        MATCH (p:Person)
                        WHERE p.name <> $email
                        AND NOT p.name CONTAINS '@'
                        AND toLower(p.name) CONTAINS $first
                        AND toLower(p.name) CONTAINS $last
                        RETURN p
                        LIMIT 1
                    """, email=email_person["name"], first=first, last=last)

                    match = match_result.single()
                    if match:
                        primary = match["p"]
                        # Add email as alias before merging
                        session.run("""
                            MATCH (p:Person)
                            WHERE elementId(p) = $id
                            SET p.email = $email,
                                p.aliases = coalesce(p.aliases, []) + $email
                        """, id=primary.element_id, email=email_person["name"])

                        self._merge_nodes(session, primary, email_person)
                        merged += 1
                        logger.info("merged_email_to_person",
                                    email=email_person["name"],
                                    person=primary["name"])

        logger.info("merged_email_to_person_total", count=merged)
        return merged

    def merge_name_variations(self) -> int:
        """Merge obvious name variations like 'Bryan Nairn (email)' into 'Bryan Nairn'.

        Returns:
            Number of nodes merged
        """
        merged = 0

        with self.driver.session() as session:
            # Find names with parenthetical annotations
            result = session.run("""
                MATCH (annotated:Person)
                WHERE annotated.name CONTAINS '('
                RETURN annotated
            """)

            for record in result:
                annotated = record["annotated"]
                # Extract base name (before parenthesis)
                base_name = annotated["name"].split("(")[0].strip()

                if len(base_name) > 2:
                    # Find matching base person
                    match_result = session.run("""
                        MATCH (p:Person)
                        WHERE toLower(p.name) = toLower($base_name)
                        AND NOT p.name CONTAINS '('
                        RETURN p
                        LIMIT 1
                    """, base_name=base_name)

                    match = match_result.single()
                    if match:
                        primary = match["p"]
                        self._merge_nodes(session, primary, annotated)
                        merged += 1
                        logger.info("merged_name_variation",
                                    variation=annotated["name"],
                                    primary=primary["name"])

            # Also merge short first-name-only nodes into full names
            # But only if there's exactly one match
            result = session.run("""
                MATCH (short:Person)
                WHERE NOT short.name CONTAINS ' '
                AND NOT short.name CONTAINS '@'
                AND NOT short.name CONTAINS '('
                AND size(short.name) <= 15
                RETURN short
            """)

            for record in result:
                short = record["short"]
                short_name = short["name"].lower()

                # Find full names that start with this first name
                match_result = session.run("""
                    MATCH (p:Person)
                    WHERE p.name <> $short_name
                    AND p.name CONTAINS ' '
                    AND toLower(p.name) STARTS WITH $short_lower + ' '
                    RETURN p
                """, short_name=short["name"], short_lower=short_name)

                matches = list(match_result)
                # Only merge if exactly one match (unambiguous)
                if len(matches) == 1:
                    primary = matches[0]["p"]
                    self._merge_nodes(session, primary, short)
                    merged += 1
                    logger.info("merged_short_to_full",
                                short=short["name"],
                                full=primary["name"])

        logger.info("merged_name_variations_total", count=merged)
        return merged

    def _merge_nodes(self, session, primary, duplicate) -> None:
        """Merge duplicate node into primary, transferring all relationships."""
        # Transfer outgoing relationships from duplicate to primary
        session.run("""
            MATCH (dup) WHERE elementId(dup) = $dup_id
            MATCH (primary) WHERE elementId(primary) = $primary_id
            OPTIONAL MATCH (dup)-[r]->(target)
            WHERE target <> primary
            WITH primary, collect(target) as targets
            UNWIND targets as t
            MERGE (primary)-[:EXTRACTED_FROM]->(t)
        """, dup_id=duplicate.element_id, primary_id=primary.element_id)

        # Transfer incoming relationships from duplicate to primary
        session.run("""
            MATCH (dup) WHERE elementId(dup) = $dup_id
            MATCH (primary) WHERE elementId(primary) = $primary_id
            OPTIONAL MATCH (source)-[r]->(dup)
            WHERE source <> primary
            WITH primary, collect(source) as sources
            UNWIND sources as s
            MERGE (s)-[:RELATED_TO]->(primary)
        """, dup_id=duplicate.element_id, primary_id=primary.element_id)

        # Delete the duplicate
        session.run("""
            MATCH (dup)
            WHERE elementId(dup) = $dup_id
            DETACH DELETE dup
        """, dup_id=duplicate.element_id)

    def run_all(self) -> dict:
        """Run all deduplication steps.

        Returns:
            Dict with counts for each step
        """
        results = {
            "exact_duplicates": self.merge_exact_duplicates(),
            "email_to_person": self.merge_email_to_person(),
            "name_variations": self.merge_name_variations(),
        }
        results["total"] = sum(results.values())
        logger.info("deduplication_complete", **results)
        return results

    def get_duplicate_candidates(self, limit: int = 50) -> list[dict]:
        """Get pairs of nodes that might be duplicates for manual review.

        Returns:
            List of potential duplicate pairs with similarity info
        """
        candidates = []

        with self.driver.session() as session:
            # Find similar names using string similarity
            result = session.run("""
                MATCH (p1:Person), (p2:Person)
                WHERE elementId(p1) < elementId(p2)
                AND (
                    // Same name, different case
                    toLower(p1.name) = toLower(p2.name)
                    // One contains the other
                    OR (toLower(p1.name) CONTAINS toLower(p2.name) AND size(p2.name) > 4)
                    OR (toLower(p2.name) CONTAINS toLower(p1.name) AND size(p1.name) > 4)
                )
                RETURN p1.name as name1, p2.name as name2,
                       CASE
                         WHEN toLower(p1.name) = toLower(p2.name) THEN 'exact_match'
                         ELSE 'partial_match'
                       END as match_type
                LIMIT $limit
            """, limit=limit)

            for r in result:
                candidates.append(dict(r))

        return candidates
