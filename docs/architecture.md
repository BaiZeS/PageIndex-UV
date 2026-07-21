# Architecture

PageIndex-UV is a **non-vector, reasoning-based RAG** tool over long documents (PDF / Markdown). It exposes both an interactive CLI and an MCP server, backed by a shared `PageIndexClient` runtime.

## At a glance

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Entry points                                    │
│   ┌──────────────────────┐    ┌───────────────────────────────────────┐  │
│   │ main.py  (CLI REPL)  │    │ server.py  (Starlette HTTP+SSE MCP)    │  │
│   │  /add /list /doc …   │    │  /sse  /messages/  /upload  /api/…     │  │
│   └──────────┬───────────┘    └───────────────────┬───────────────────┘  │
│              │                                    │                       │
│              └──────────────┬─────────────────────┘                       │
│                             ▼                                             │
│                ┌────────────────────────────────┐                         │
│                │      PageIndexClient  (singleton,  in-process)         │ │
│                │  - documents  - uuid ↔ db_id map  - LLM client         │ │
│                └──────────────┬─────────────────────┘                   │
│                             │                                             │
│              ┌──────────────┴──────────────────┐                         │
│              ▼                                 ▼                         │
│   ┌──────────────────────┐        ┌──────────────────────────────┐      │
│   │ pageindex_mutil/     │        │  db.py (SQLite cache)         │      │
│   │  - super_tree.py     │        │  - nodes / pages / docs       │      │
│   │  - closet_index.py   │        │  - closet_tags / doc_keywords │      │
│   │  - agentic/          │        │  - kb_identity                │      │
│   │  - utils / config    │        │  - delete_document (cascade)  │      │
│   └──────────┬───────────┘        └──────────────────────────────┘      │
│              ▼                                                           │
│   ┌─────────────────────────────────────────────────────────────────┐    │
│   │  configure_llm()  →  OpenAI-compatible client  (Qwen / OpenAI)  │    │
│   │  resolved from: explicit arg > OPENAI_API_KEY > DASHSCOPE_API_KEY│   │
│   └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core pipeline (per question)

```
question
   │
   ▼
HyDE: Query Expansion (Planner)
   │  Generate hypothetical answer + query variants
   ▼
L0: Four-Channel Prefilter (SuperTreeIndex)
   │  Channel A: ClosetIndex (semantic tag matching)
   │  Channel B: KeywordIndex (jieba inverted index, includes node titles)
   │  Channel C: ChromaDB vector search (1.5x weight)
   │  Channel D: Entity graph matching (cross-document relations)
   ▼
L1: Super-Tree Document Selection (LLM)
   │  Based on enriched mini-TOC (depth=1 + depth=2 child titles) + KB Identity
   │  6000 tokens budget, max 50 candidates → 3-5 selected
   ▼
L2: Parallel Per-Document Node Recall (AgenticRouter)
   │  asyncio.gather parallel tree reasoning per selected doc
   │  Results ranked by relevance (matched node count)
   ▼
L3: Context Extraction (ranked by relevance)
   │  Entity relationship enrichment + page merging
   │  16K token budget truncation
   ▼
LLM Answer Composition (with evidence)
   │
   ▼
CRAG Verification (verifier.py)
   │  Confidence-based: answer / expand / refuse
   ▼
{ answer, confidence, matched_docs, selected_nodes, pages }
```

## Process model

- **`PageIndexClient` is a single in-process singleton**, owned by:
  - `main.py` — created at startup, dies with the REPL.
  - `server.py` — created in `lifespan`, lives for the uvicorn worker.
- **All queries route through `client.search(query, top_k)`** — same code path whether invoked from CLI or MCP.
- **The LLM client is rebuilt via `configure_llm()`** whenever the user changes the API key / base URL via the web console's model config tab. See [web-console-design-system.md](web-console-design-system.md) for the live-switch UX.

## Key components

### Super-Tree Retrieval v3

The multi-document retrieval uses a four-layer architecture:

1. **L0 Four-Channel Prefilter** (`SuperTreeIndex.prefilter()`):
   - **Channel A (Semantic)**: `ClosetIndex` matches query against `closet_tags` table using jieba tokenization
   - **Channel B (Keyword)**: `KeywordIndex` matches query against `doc_keywords` table (includes node titles, doc names, descriptions)
   - **Channel C (Vector)**: ChromaDB vector search via `SearchBackend.search()` (1.5x weight for semantic understanding)
   - **Channel D (Entity)**: Entity graph matching via `db.search_entities()` + `db.get_entity_documents()`
   - Results merged into candidate set (max 50 docs)

2. **L1 Super-Tree Document Selection** (`SuperTreeIndex.select_documents()`):
   - Builds enriched mini-TOC (depth=1 nodes + depth=2 child titles, max 8 top nodes per doc)
   - Combines with KB Identity (knowledge base summary)
   - Single LLM call selects 3-5 most relevant documents
   - Token budget: 6000 tokens (auto-truncation if exceeded)
   - HyDE integration: planner generates hypothetical answer for query expansion

3. **L2 Parallel Per-Document Node Recall** (`AgenticRouter._act_tree_search()`):
   - `asyncio.gather` parallel tree reasoning per selected document
   - Each doc's recall runs in a thread via `_recall_nodes_for_doc()`
   - Results ranked by relevance score (matched node count / total nodes)

4. **L3 Context Extraction** (ranked by relevance):
   - Entity relationship enrichment from cross-document graph
   - Converts nodes to page ranges
   - Reads page text from SQLite `pages` table
   - 16K token hard limit with priority-based truncation

### Agentic Router (v2 fallback)

When Super-Tree fails, `AgenticRouter.search()` falls back to:
- **Plan**: Query analysis with HyDE (query type classification + query variants)
- **Route**: Parallel strategy execution (Metadata/Semantics/Description)
- **Act**: Parallel node recall + context assembly
- **Verify**: CRAG verification with confidence thresholds (answer/expand/refuse)

### Hybrid Search Backend

`HybridSearchBackend` combines three search channels using Reciprocal Rank Fusion (RRF):
- **Vector search**: ChromaDB with sentence-transformers embeddings
- **Keyword search**: jieba inverted index on doc names, descriptions, and node titles
- **Tag search**: ClosetIndex semantic tag matching

Configurable weights per channel (default: vector 1.5x, keyword 1.0x, tag 1.0x).

### Entity Knowledge Graph

`EntityExtractor` automatically extracts entities (people, projects, organizations, concepts) and relationships from documents during indexing. Stored in SQLite tables (`entities`, `entity_mentions`, `entity_relations`). Used in:
- L0 prefilter Channel D for entity-driven document recall
- L3 context enrichment with cross-document entity relationships
- MCP tools: `search_entities`, `get_entity`, `get_document_entities`, `get_related_documents`

### DocIdMapper

`DocIdMapper` class centralizes UUID ↔ DB ID bidirectional mapping. Replaces scattered `_uuid_to_db` dict usage across client.py, super_tree.py, router.py, and server.py.

## Surface map

| Surface | Path / Tool | Notes |
|---|---|---|
| CLI | `main.py` | `/add /list /doc <n> /clear /help` |
| Web console | `web/index.html` + `web/static/{tokens,base,components}.css` + `web/static/{lib,components,app.js}` | Vue 3 + Element Plus, no build step. See [web-console-design-system.md](web-console-design-system.md) |
| MCP (SSE) | `GET /sse`, `POST /messages/` | Tools: `search`, `list_documents`, `get_document`, `delete_document`. See [mcp-tools.md](mcp-tools.md) |
| REST API | `GET /health`, `POST /upload`, `GET/DELETE /api/documents`, `POST /api/search`, `GET/POST /api/config`, `POST /api/config/test` | Used by the web console; same auth as MCP (`X-API-Key`) |
| Auth | `APIKeyMiddleware` | Public: `/`, `/health`, `/static/*`. Gated: `/api/*`, `/sse`, `/messages/`, `/upload` |
| Documents store | `WORKSPACE/` (PDF/MD) | Path from `.env` (`WORKSPACE=…`) |
| Index cache | `DB_PATH` (SQLite) | Path from `.env`. Holds `nodes`, `pages`, `documents`, `closet_tags`, `doc_keywords`, `kb_identity` tables |

## SQLite schema

```sql
-- Core tables
documents (id, pdf_name, pdf_path, tree_json, doc_description, created_at)
nodes (id, doc_id, node_id, title, summary, start_index, end_index, parent_node_id)
pages (id, doc_id, page_number, content)

-- Super-Tree v3 tables
closet_tags (id, doc_id, tag_text, tag_token, confidence, source)
doc_keywords (id, doc_id, keyword, field)  -- field: "name"|"description"|"node_title"
kb_identity (id, identity_text, doc_count, updated_at)

-- Entity knowledge graph tables
entities (id, entity_type, name, aliases, doc_count, created_at)
entity_mentions (id, entity_id, doc_id, node_id, context_snippet, confidence, created_at)
entity_relations (id, subject_id, predicate, object_id, doc_id, confidence, created_at)
```

## Stability promise

- **HTTP API** (`/api/*`) is the public contract for integrators; additive changes only within a major.
- **MCP tool schemas** are public; adding optional fields is non-breaking.
- **`pageindex_mutil/`** is internal — refactors don't bump the version.
- **Web console** (`web/static/*`) ships with the server; pin via the server release.