# MCP Tools Reference

> Server: `pageindex-uv` v1.27.0 (the underlying `pageindex` package's version — see `pyproject.toml`)
> Protocol: MCP `2025-11-25`
> Transport: HTTP + SSE (`GET /sse`, `POST /messages/?session_id=…`)
> Auth: `X-API-Key` header (same as REST API)

The server exposes **4 tools** and **N resources** (one per indexed document, URI scheme `document://<uuid>`).

## Tools

### `search`

Multi-strategy RAG over indexed documents.

**Input**

```json
{
  "query": "<natural language question>",
  "top_k": 3   // optional, default 3
}
```

**Output** — single `TextContent` block with JSON:

```json
{
  "answer": "<LLM-composed answer with evidence>",
  "confidence": "high | medium | low | unknown",
  "matched_docs": [
    { "doc_id": "<uuid>", "doc_name": "<name>", "score": 0.91 }
  ],
  "selected_nodes": [
    { "node_id": "...", "title": "...", "path": "...", "summary": "...",
      "pages": [3, 4] }
  ],
  "pages": [
    { "doc_id": "<uuid>", "page": 7, "text": "..." }
  ]
}
```

The `confidence` field is a **backend string enum** — never multiplied into a fabricated percentage. Map it via `confidenceRatio`/`confidenceLabel` in `web/static/lib/format.js` if you need a progress bar.

### `list_documents`

List every indexed document with metadata.

**Input**: none.

**Output**

```json
{
  "documents": [
    { "id": "<uuid>", "name": "...", "description": "..." }
  ]
}
```

The `id` is the UUID used by `get_document` / `delete_document`. The `name` is the `pdf_name` from the SQLite cache.

### `get_document`

Get detailed metadata + tree structure for one document.

**Input**

```json
{ "doc_id": "<uuid>" }
```

**Output** — JSON string with the full PageIndex tree (nodes with `title`, `summary`, `node_id`, `pages`, `start_index`, `end_index`).

### `delete_document`

Remove a document from memory + DB + on-disk workspace.

**Input**

```json
{ "doc_id": "<uuid>" }
```

**Output**

```json
{ "success": true, "doc_id": "<uuid>" }
```

Same code path as the REST `DELETE /api/documents/{id}` and the web console's delete button (see `docs/design-docs/PageIndex/delete-path-integrity/`). Disk cleanup + orphan migration are guaranteed by `db.delete_document` + `on_document_removed`.

## Resources

One resource per indexed document:

```
document://<uuid>
```

Read returns the full document metadata as text. Use when you want the entire tree without parsing the `list_documents` output.

## Authentication

Every request to `/sse` and `/messages/` must carry the API key:

```http
GET /sse HTTP/1.1
X-API-Key: <API_KEY from .env>
Accept: text/event-stream
```

`APIKeyMiddleware` returns `403 {"error":"Invalid or missing API Key"}` for any mismatch. Public paths (`/`, `/health`, `/static/*`) bypass auth.

## Client config examples

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "pageindex-uv": {
      "url": "http://127.0.0.1:3000/sse",
      "headers": { "X-API-Key": "<your API_KEY>" }
    }
  }
}
```

### Cursor / Cline / generic

Same shape — `url` + `headers.X-API-Key`. The MCP client opens SSE, parses the `endpoint` event for the session URL, then POSTs JSON-RPC `initialize` / `tools/list` / `tools/call` messages.

### Python (official `mcp` SDK)

```python
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def main():
    async with sse_client("http://127.0.0.1:3000/sse",
                          headers={"X-API-Key": "<your API_KEY>"}) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            r = await session.call_tool("search",
                                         {"query": "差旅费", "top_k": 2})
            print(r.content[0].text)

asyncio.run(main())
```

## What you get back when something fails

- **Wrong / missing API key**: HTTP 403 with `{"error":"Invalid or missing API Key"}` — surfaces before SSE opens.
- **No documents indexed yet**: `search` returns a low-confidence answer explaining the search returned no candidates (router says "Super-Tree selection returned no documents"). Empty `matched_docs` / `selected_nodes` / `pages`.
- **Tool name typo**: MCP `tools/call` returns `[TextContent(type="text", text=JSON.stringify({"error":"Unknown tool: <name>"}))]`. Check the `name` field.
- **Network drop / uvicorn restart**: SSE stream closes; client should reconnect with exponential backoff.

## Versioning

The MCP tool schemas are part of the **public contract** for integrators. Adding optional fields is non-breaking. Removing or renaming a tool is a major-version bump (the next would be `v2.0.0`).