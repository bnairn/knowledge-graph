# Personal Knowledge Graph

A personal knowledge graph built from Gmail emails, extracting entities and relationships to create a searchable network of people, organizations, topics, and concepts from your communications.

## Overview

This project ingests emails from Gmail, uses LLM-powered entity extraction to identify people, organizations, topics, and other entities, then stores them in a Neo4j graph database. The result is a queryable knowledge graph that reveals connections and patterns in your personal and professional network.

## Tech Stack

- **Database**: Neo4j Community Edition (running in Docker)
- **Language**: Python 3.11+
- **Entity Extraction**: Ollama (local LLM inference)
- **Data Sources**: Gmail API
- **CLI**: Click + Rich for terminal interface
- **Embeddings**: nomic-embed-text via Ollama (for semantic search)

## Architecture

```
Gmail API → Email Fetcher → Text Extraction → Entity Extraction (Ollama) → Neo4j Graph
                                                      ↓
                                              GraphRAG Query Engine
```

## Why Ollama Instead of Claude API

We initially built this project using the Anthropic Claude API for entity extraction. While Claude produced excellent results, the API costs quickly became prohibitive:

- **~$0.003-0.01 per email** for entity extraction
- Processing thousands of emails meant costs in the $30-100+ range just for initial ingestion
- Each re-sync or schema change required re-processing

Switching to **Ollama with llama3.1:8b** gave us:
- **Zero marginal cost** after initial setup
- Comparable extraction quality for our use case
- Full privacy - all data stays local
- Ability to experiment and iterate without watching costs

The tradeoff is speed (Ollama is slower) and slightly lower extraction quality, but for a personal knowledge graph these are acceptable.

## Neo4j Setup

We run Neo4j Community Edition in Docker for easy setup and portability:

```bash
docker compose up -d
```

The `docker-compose.yml` configures:
- Neo4j 5.x Community Edition
- Bolt port 7687 for driver connections
- HTTP port 7474 for Neo4j Browser
- Persistent volume for data

Access Neo4j Browser at http://localhost:7474

## Installation

1. **Clone and install dependencies**:
   ```bash
   git clone <repo-url>
   cd neo4j
   pip install -e .
   ```

2. **Start Neo4j**:
   ```bash
   docker compose up -d
   ```

3. **Install Ollama and models**:
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```

4. **Configure environment**:
   Secrets live in 1Password (item: `Private / Knowledge Graph neo4j - env`) and are
   loaded at runtime via `op run --env-file=.env.op` — `.env.op` holds only `op://`
   references, no plaintext. Requires the 1Password CLI signed in. Run commands as:
   ```bash
   op run --env-file=.env.op -- kg <command>   # e.g. kg auth, kg sync
   ```

5. **Set up Google OAuth**:
   - Create a project in Google Cloud Console
   - Enable Gmail API
   - Download OAuth credentials as `credentials.json`

## Usage

### Sync emails to the knowledge graph

```bash
# Full sync (processes all emails)
kg sync --source gmail

# Limit number of emails
kg sync --source gmail --limit 100
```

### Query the graph

```bash
# Ask natural language questions
kg ask "Who do I email most frequently?"
kg ask "What topics come up in emails with John?"

# Find connections between entities
kg connect "Alice Smith" "Bob Jones"

# Get entity details
kg entity "Alice Smith"
```

### Manage duplicates

```bash
# Preview potential duplicates
kg dedupe --dry-run

# Automatically merge obvious duplicates
kg dedupe
```

### View statistics

```bash
kg stats
```

## Graph Schema

### Node Types
- **Person**: People mentioned in or sending/receiving emails
- **Organization**: Companies, institutions, groups
- **Topic**: Discussion topics and themes
- **Concept**: Technologies, methodologies, ideas
- **Email**: Individual email messages (metadata only)

### Key Relationships
- `(:Person)-[:EXTRACTED_FROM]->(:Email)` - Person mentioned in email
- `(:Topic)-[:EXTRACTED_FROM]->(:Email)` - Topic discussed in email
- `(:Person)-[:SPOUSE_OF]->(:Person)` - Family relationships (manually added)

## Entity Deduplication

The LLM often extracts the same person with different name variations:
- "Bryan Nairn", "Bryan D Nairn", "bryan.nairn@gmail.com"

The `kg dedupe` command automatically merges:
- Exact duplicates (case-insensitive)
- Email addresses to their person names
- Name variations (e.g., "John Smith (work)" → "John Smith")

For manual merges, use Neo4j Browser or the Python API directly.

## Configuration

Key environment variables (`.env`):

```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# LLM Provider
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Google OAuth
GOOGLE_CREDENTIALS_FILE=credentials.json
```

## Visualization

Open Neo4j Browser at http://localhost:7474 and run Cypher queries:

```cypher
// See your email network centered on yourself
MATCH (me:Person {name: "Your Name"})-[:EXTRACTED_FROM]->(e:Email)<-[:EXTRACTED_FROM]-(other:Person)
WITH me, other, count(DISTINCT e) as weight
WHERE weight >= 2
RETURN me, other, weight
ORDER BY weight DESC
LIMIT 30
```

## Project Structure

```
├── config/
│   ├── settings.py          # Pydantic settings
│   └── logging_config.py
├── src/
│   ├── auth/                 # Google OAuth
│   ├── connectors/           # Gmail connector
│   ├── extractors/           # Text extraction
│   ├── nlp/                  # Ollama client, entity extraction
│   ├── graph/                # Neo4j client, node/relationship managers
│   ├── pipeline/             # Sync orchestration
│   └── query/                # GraphRAG query engine
├── scripts/
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

## License

MIT
