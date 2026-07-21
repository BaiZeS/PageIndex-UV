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
L0: Dual-Channel Prefilter (SuperTreeIndex)
   │  Channel A: ClosetIndex (semantic tag matching)
   │  Channel B: KeywordIndex (jieba inverted index)
   ▼
L1: Super-Tree Document Selection (LLM)
   │  Based on mini-TOC + KB Identity
   │  6000 tokens budget, max 50 candidates → 3-5 selected
   ▼
L2: Per-Document Node Recall (AgenticRouter)
   │  Independent tree reasoning per selected doc
   │  Avoids multi-tree mixing in single prompt
   ▼
L3: Context Extraction
   │  Page merging + 16K token budget truncation
   ▼
LLM Answer Composition (with evidence)
   │
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

1. **L0 Dual-Channel Prefilter** (`SuperTreeIndex.prefilter()`):
   - **Channel A (Semantic)**: `ClosetIndex` matches query against `closet_tags` table using jieba tokenization
   - **Channel B (Keyword)**: `KeywordIndex` matches query against `doc_keywords` table
   - Results merged into candidate set (max 50 docs)

2. **L1 Super-Tree Document Selection** (`SuperTreeIndex.select_documents()`):
   - Builds mini-TOC (depth=1 nodes, max 8 per doc)
   - Combines with KB Identity (knowledge base summary)
   - Single LLM call selects 3-5 most relevant documents
   - Token budget: 6000 tokens (auto-truncation if exceeded)

3. **L2 Per-Document Node Recall**:
   - Independent tree reasoning per selected document
   - Uses `get_relevant_nodes()` for each doc's tree structure
   - Max 5 nodes per document

4. **L3 Context Extraction**:
   - Converts nodes to page ranges
   - Reads page text from SQLite `pages` table
   - 16K token hard limit with priority-based truncation

### Agentic Router (v2 fallback)

When Super-Tree fails, `AgenticRouter.search()` falls back to:
- **Plan**: Query analysis with HyDE
- **Route**: Strategy selection (Metadata/Semantics/Description)
- **Act**: Parallel strategy execution
- **Verify**: CRAG verification with confidence thresholds

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
doc_keywords (id, doc_id, keyword, field)
kb_identity (id, identity_text, doc_count, updated_at)
```

## Stability promise

- **HTTP API** (`/api/*`) is the public contract for integrators; additive changes only within a major.
- **MCP tool schemas** are public; adding optional fields is non-breaking.
- **`pageindex_mutil/`** is internal — refactors don't bump the version.
- **Web console** (`web/static/*`) ships with the server; pin via the server release.