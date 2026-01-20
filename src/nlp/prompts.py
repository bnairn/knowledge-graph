"""Prompt templates for entity and relationship extraction."""

ENTITY_EXTRACTION_SYSTEM = """You are an expert at extracting structured information from text.
Your task is to identify and extract all named entities from the provided text.

Entity types to extract:
- PERSON: Individual people (names, roles, titles)
- ORGANIZATION: Companies, institutions, teams, groups
- CONCEPT: Technologies, methodologies, frameworks, abstract ideas
- TOPIC: Subject areas, themes, domains of knowledge
- PROJECT: Named projects, initiatives, products
- EVENT: Meetings, conferences, deadlines, milestones
- LOCATION: Cities, countries, offices, venues

Guidelines:
- Extract the canonical/full name when possible
- Include any aliases or alternative names mentioned
- Provide brief context based on how the entity appears in the text
- Assign a confidence score (0.0-1.0) based on how clearly the entity is identified
- Include the surrounding text context where the entity appears
- Do not infer entities that are not explicitly mentioned
- For ambiguous references, use lower confidence scores"""

ENTITY_EXTRACTION_PROMPT = """Extract all named entities from the following text.

Text:
{text}

{existing_entities_section}

Extract all entities with their type, name, aliases, description, confidence score, and the context where they appear."""

RELATIONSHIP_EXTRACTION_SYSTEM = """You are an expert at understanding relationships between entities.
Given a text and a list of identified entities, your task is to identify all relationships between them.

Common relationship types:
- WORKS_FOR: Person works for an organization
- KNOWS: Person knows another person (colleague, friend, contact)
- WORKS_ON: Person works on a project
- ATTENDED: Person attended an event
- EXPERT_IN: Person has expertise in a concept/technology
- LOCATED_IN: Entity is located in a place
- PARTNERED_WITH: Organization partners with another organization
- USES: Project/Organization uses a concept/technology
- RELATED_TO: General relationship between entities
- OWNS: Organization owns a project
- SUBTOPIC_OF: Topic is a subtopic of another topic

Guidelines:
- Only identify relationships that are explicitly stated or strongly implied
- Include any relevant properties (dates, roles, etc.)
- Assign confidence scores based on how explicitly the relationship is stated
- Include the text context where the relationship is mentioned
- Do not invent relationships that aren't supported by the text"""

RELATIONSHIP_EXTRACTION_PROMPT = """Identify all relationships between the following entities based on the text.

Entities:
{entities}

Text:
{text}

For each relationship found, specify the source entity, target entity, relationship type, any properties (like dates or roles), confidence level, and the context where it appears."""

ENTITY_RESOLUTION_SYSTEM = """You are an expert at entity resolution - determining when two entity mentions refer to the same real-world entity.

Consider:
- Name variations (nicknames, abbreviations, full names)
- Spelling variations
- Titles and honorifics
- Context clues about the entity
- Common aliases

Output whether the new entity matches an existing one, and if so, which one."""

ENTITY_RESOLUTION_PROMPT = """Determine if this new entity matches any existing entities:

New Entity:
Name: {new_name}
Type: {new_type}
Context: {new_context}

Existing Entities:
{existing_entities}

If the new entity matches an existing one, provide the matching entity's name. If it's a new distinct entity, indicate that it should be created as new."""
