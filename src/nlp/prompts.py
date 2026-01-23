"""Prompt templates for entity and relationship extraction."""

ENTITY_EXTRACTION_SYSTEM = """You extract contacts and topics from personal emails.

ONLY extract these entity types:
- PERSON: Real people the email author knows personally or professionally. This includes:
  - Email sender/recipients
  - People being scheduled for meetings
  - People mentioned as colleagues, friends, family
  - People the author needs to contact or follow up with

  DO NOT extract:
  - Celebrities, actors, politicians, athletes mentioned in passing
  - Authors of articles or books being discussed
  - Historical figures
  - Fictional characters
  - Names in email signatures of forwarded messages
  - Names in marketing/promotional content

- ORGANIZATION: Companies, schools, or groups the author works with or interacts with
- TOPIC: Main subjects being discussed (limit to 2-3 per email)

Keep it brief. Only extract entities directly relevant to the email author's network."""

ENTITY_EXTRACTION_PROMPT = """Extract contacts and topics from this email.

Email:
{text}

{existing_entities_section}

Return entities with: name, type, and confidence (0.0-1.0).
Only include people the email author actually knows or interacts with."""

RELATIONSHIP_EXTRACTION_SYSTEM = """You are an expert at understanding relationships between entities.
Given a text and a list of identified entities, your task is to identify all relationships between them.

Common relationship types:
- WORKS_FOR: Person works for an organization
- KNOWS: Person knows another person (colleague, friend, contact)
- PARTNERED_WITH: Organization partners with another organization
- RELATED_TO: General relationship between entities
- INTERESTED_IN: Person is interested in a topic
- DISCUSSES: Person discusses a topic

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
