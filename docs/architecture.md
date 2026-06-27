# Architecture Overview

> **Status**: frozen at v1.2.0 (commit `3d09544`)
> **Audience**: contributors + integrators who want a single mental model of the system.

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
│   │  - pdf_parser        │        │  - get_all_documents         │      │
│   │  - md_parser         │        │  - delete_document (cascade)  │      │
│   │  - utils / config    │        └──────────────────────────────┘      │
│   └──────────┬───────────┘                                               │
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
Agentic Multi-Strategy Router   ←─── docs/superpowers/plans/2026-05-05-…
   │  (selects tree-search vs TOC vs flat-search strategy)
   ▼
Super-Tree reasoning (per selected doc)  ←─── LLM walks the doc's tree
   │
   ▼
selected_node_ids + page ranges
   │
   ▼
SQLite lookup → page text snippets
   │
   ▼
LLM answer composition (with evidence)
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

## Surface map

| Surface | Path / Tool | Notes |
|---|---|---|
| CLI | `main.py` | `/add /list /doc <n> /clear /help` |
| Web console | `web/index.html` + `web/static/{tokens,base,components}.css` + `web/static/{lib,components,app.js}` | Vue 3 + Element Plus, no build step. See [web-console-design-system.md](web-console-design-system.md) |
| MCP (SSE) | `GET /sse`, `POST /messages/` | Tools: `search`, `list_documents`, `get_document`, `delete_document`. See [mcp-tools.md](mcp-tools.md) |
| REST API | `GET /health`, `POST /upload`, `GET/DELETE /api/documents`, `POST /api/search`, `GET/POST /api/config`, `POST /api/config/test` | Used by the web console; same auth as MCP (`X-API-Key`) |
| Auth | `APIKeyMiddleware` | Public: `/`, `/health`, `/static/*`. Gated: `/api/*`, `/sse`, `/messages/`, `/upload` |
| Documents store | `WORKSPACE/` (PDF/MD) | Path from `.env` (`WORKSPACE=…`) |
| Index cache | `DB_PATH` (SQLite) | Path from `.env`. Holds `nodes`, `pages`, `documents` tables |

## Design doc layout

All per-feature design work lives in `docs/design-docs/PageIndex/<feature>/{spec,tasks}.md`:

```
docs/design-docs/PageIndex/
├── architecture-review-2026-06/review-report.md
├── batch-upload/
├── db-concurrency-hardening/
├── delete-path-integrity/
├── model-config-completion/
├── non-vector-retrieval-optimization/
├── project-refactor/
├── super-tree-retrieval-v3/
└── web-console/
```

The convention is `spec.md` (what & why) + `tasks.md` (how, broken into atomic tasks).

## Stability promise

- **HTTP API** (`/api/*`) is the public contract for integrators; additive changes only within a major.
- **MCP tool schemas** are public; adding optional fields is non-breaking.
- **`pageindex_mutil/`** is internal — refactors don't bump the version.
- **Web console** (`web/static/*`) ships with the server; pin via the server's commit hash.