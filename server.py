#!/usr/bin/env python3
"""MCP Server for PageIndex-UV RAG retrieval.

Exposes PageIndex RAG capabilities via MCP protocol (SSE transport)
with API Key authentication and file upload endpoint.
"""

import os
import sys
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

# Ensure project root is on path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "PageIndex"))

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.routing import Route, Mount
from starlette.types import ASGIApp
import uvicorn

from mcp.server import Server
from mcp.server.sse import SseServerTransport
import mcp.types as types

from PageIndex.pageindex import PageIndexClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pageindex-mcp")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.getenv("API_KEY", "")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "3000"))
WORKSPACE = os.getenv("WORKSPACE", "/app/data/workspace")
DB_PATH = os.getenv("DB_PATH", "/app/data/index.db")

# ---------------------------------------------------------------------------
# API Key middleware
# ---------------------------------------------------------------------------
class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip auth for health endpoint
        if request.url.path == "/health":
            return await call_next(request)

        # Allow OPTIONS for CORS preflight
        if request.method == "OPTIONS":
            return await call_next(request)

        if API_KEY:
            header_key = request.headers.get("X-API-Key", "")
            if header_key != API_KEY:
                return JSONResponse(
                    {"error": "Invalid or missing API Key"},
                    status_code=403,
                )
        return await call_next(request)


# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------
class CORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, X-API-Key",
                },
            )
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response


# ---------------------------------------------------------------------------
# Global client instance (created at startup)
# ---------------------------------------------------------------------------
client: PageIndexClient | None = None


def get_client() -> PageIndexClient:
    if client is None:
        raise RuntimeError("PageIndexClient not initialized")
    return client


# ---------------------------------------------------------------------------
# MCP Server setup
# ---------------------------------------------------------------------------
mcp_server = Server("pageindex-uv")


@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle MCP tool calls."""
    try:
        if name == "search":
            query = arguments.get("query", "")
            top_k = int(arguments.get("top_k", 3))
            if not query:
                return [types.TextContent(type="text", text=json.dumps({"error": "query is required"}, ensure_ascii=False))]

            result = await get_client().search(query, top_k)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "list_documents":
            db = get_client().db
            if db is None:
                return [types.TextContent(type="text", text=json.dumps({"documents": []}, ensure_ascii=False))]

            c = get_client()
            docs = []
            for doc in db.get_all_documents():
                db_id = str(doc.get("id", ""))
                # Map DB id back to UUID if possible
                doc_id = db_id
                for uuid, mapped_db_id in c._uuid_to_db.items():
                    if str(mapped_db_id) == db_id:
                        doc_id = uuid
                        break
                docs.append({
                    "id": doc_id,
                    "name": doc.get("pdf_name", ""),
                    "description": doc.get("doc_description", ""),
                })
            return [types.TextContent(type="text", text=json.dumps({"documents": docs}, ensure_ascii=False, indent=2))]

        elif name == "get_document":
            doc_id = arguments.get("doc_id", "")
            if not doc_id:
                return [types.TextContent(type="text", text=json.dumps({"error": "doc_id is required"}, ensure_ascii=False))]

            try:
                info = get_client().get_document(doc_id)
                return [types.TextContent(type="text", text=info)]
            except Exception as e:
                return [types.TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]

        elif name == "delete_document":
            doc_id = arguments.get("doc_id", "")
            if not doc_id:
                return [types.TextContent(type="text", text=json.dumps({"error": "doc_id is required"}, ensure_ascii=False))]

            # Remove from memory and DB
            c = get_client()
            if doc_id in c.documents:
                del c.documents[doc_id]

            db_id = c._uuid_to_db.pop(doc_id, None)
            if db_id is not None and c.db is not None:
                try:
                    c.db.delete_document(db_id)
                except Exception as e:
                    logger.warning("Failed to delete document from DB: %s", e)

            if c.closet_index and db_id is not None:
                try:
                    c.closet_index.remove_document(db_id)
                except Exception as e:
                    logger.warning("Failed to delete closet tags: %s", e)

            return [types.TextContent(type="text", text=json.dumps({"success": True, "doc_id": doc_id}, ensure_ascii=False))]

        else:
            return [types.TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False))]

    except Exception as e:
        logger.exception("Tool %s failed", name)
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]


@mcp_server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search",
            description="Search across indexed documents using the Agentic Multi-Strategy Router. Returns an answer with confidence level and source references.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "top_k": {"type": "integer", "description": "Number of top documents to consider", "default": 3},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="list_documents",
            description="List all indexed documents with their metadata.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        types.Tool(
            name="get_document",
            description="Get detailed metadata and structure for a specific document.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "The document ID (UUID)"},
                },
                "required": ["doc_id"],
            },
        ),
        types.Tool(
            name="delete_document",
            description="Delete a document and its index from the system.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "The document ID (UUID) to delete"},
                },
                "required": ["doc_id"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------
@mcp_server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Handle MCP resource reads."""
    prefix = "document://"
    if not uri.startswith(prefix):
        raise ValueError(f"Unknown resource URI: {uri}")

    rest = uri[len(prefix):]
    parts = rest.split("/", 1)
    doc_id = parts[0]
    sub = parts[1] if len(parts) > 1 else ""

    c = get_client()
    if sub == "structure":
        return c.get_document_structure(doc_id)
    else:
        return c.get_document(doc_id)


@mcp_server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    c = get_client()
    db = c.db
    if db is None:
        return []

    resources = []
    for doc in db.get_all_documents():
        doc_id = str(doc.get("id", ""))
        # Map DB id back to UUID if possible
        for uuid, db_id in c._uuid_to_db.items():
            if str(db_id) == doc_id:
                doc_id = uuid
                break
        name = doc.get("pdf_name", "") or "Untitled"
        resources.append(
            types.Resource(
                uri=f"document://{doc_id}",
                name=name,
                mimeType="application/json",
            )
        )
    return resources


# ---------------------------------------------------------------------------
# HTTP endpoints (non-MCP)
# ---------------------------------------------------------------------------
async def health_endpoint(request: Request) -> Response:
    c = get_client()
    doc_count = 0
    if c.db is not None:
        try:
            doc_count = len(c.db.get_all_documents())
        except Exception:
            pass
    return JSONResponse({"status": "ok", "documents": doc_count})


async def upload_endpoint(request: Request) -> Response:
    """Upload a PDF or Markdown file and index it."""
    try:
        import multipart
    except ImportError:
        return JSONResponse(
            {"error": "python-multipart not installed"},
            status_code=500,
        )

    # Parse multipart form data
    try:
        form = await request.form()
    except Exception as e:
        logger.exception("Failed to parse multipart form")
        return JSONResponse({"error": f"Failed to parse form: {e}"}, status_code=400)

    file_field = form.get("file")
    if file_field is None:
        return JSONResponse({"error": "Missing 'file' field"}, status_code=400)

    # Save to temp location (use client's actual workspace)
    upload_dir = Path(get_client().workspace) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / file_field.filename
    content = await file_field.read()
    temp_path.write_bytes(content)

    # Determine mode from filename
    mode = "auto"
    filename = str(file_field.filename).lower()
    if filename.endswith(".pdf"):
        mode = "pdf"
    elif filename.endswith(".md") or filename.endswith(".markdown"):
        mode = "md"

    # Index the document
    try:
        doc_id = get_client().index(str(temp_path), mode=mode)
        return JSONResponse({
            "success": True,
            "doc_id": doc_id,
            "mode": mode,
            "filename": file_field.filename,
        })
    except Exception as e:
        logger.exception("Indexing failed")
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# SSE handlers
# ---------------------------------------------------------------------------
sse_transport = SseServerTransport("/messages/")


async def handle_sse(request: Request) -> Response:
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options(),
        )
    return Response()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    global client
    logger.info("Initializing PageIndexClient...")
    workspace = WORKSPACE
    db_path = DB_PATH
    try:
        Path(workspace).mkdir(parents=True, exist_ok=True)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Fallback to local data dir when running outside Docker
        fallback = Path(__file__).parent / "data"
        workspace = str(fallback / "workspace")
        db_path = str(fallback / "index.db")
        Path(workspace).mkdir(parents=True, exist_ok=True)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info("Using fallback workspace: %s", workspace)

    client = PageIndexClient(workspace=workspace, db_path=db_path)
    logger.info("PageIndexClient ready. Documents: %d", len(client.documents))
    yield
    logger.info("Shutting down...")
    if client is not None:
        client.close()


# ---------------------------------------------------------------------------
# Build Starlette app
# ---------------------------------------------------------------------------
middleware = [
    Middleware(CORSMiddleware),
    Middleware(APIKeyMiddleware),
]

routes = [
    Route("/health", endpoint=health_endpoint, methods=["GET"]),
    Route("/upload", endpoint=upload_endpoint, methods=["POST", "OPTIONS"]),
    Route("/sse", endpoint=handle_sse, methods=["GET"]),
    Mount("/messages/", app=sse_transport.handle_post_message),
]

app = Starlette(
    debug=False,
    routes=routes,
    middleware=middleware,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting PageIndex-UV MCP Server on %s:%d", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
