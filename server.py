#!/usr/bin/env python3
"""MCP Server for PageIndex-UV RAG retrieval.

Exposes PageIndex RAG capabilities via MCP protocol (SSE transport)
with API Key authentication and file upload endpoint.
"""

import os
import sys
import json
import uuid
import asyncio
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse, FileResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp
import uvicorn

import web_config

from mcp.server import Server
from mcp.server.sse import SseServerTransport
import mcp.types as types

import openai
from openai import OpenAI
from pageindex_mutil import PageIndexClient
from pageindex_mutil.utils import configure_llm, get_llm_config, ConfigLoader

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
# Static web console directory (module-relative so it resolves under Docker WORKDIR and local dev alike)
WEB_DIR = Path(__file__).resolve().parent / "web"

# ---------------------------------------------------------------------------
# API Key middleware
# ---------------------------------------------------------------------------
class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip auth for health + static console (page itself is public; API still gated)
        if request.url.path == "/health" \
                or request.url.path == "/" \
                or request.url.path.startswith("/static"):
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
# No-cache middleware for the static console
# ---------------------------------------------------------------------------
class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    """Disable HTTP caching for /static/* so front-end edits to CSS/JS
    are picked up on the next page load without the user having to
    hard-reload or hunt for the right query-string. Only applies to the
    app's own /static/ tree — external CDNs (Vue, Element Plus, fonts)
    keep their normal caching policies."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        return response


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


_startup_workspace = WORKSPACE
_startup_db_path = DB_PATH


def _rebuild_client(*, model=None, retrieve_model=None):
    """Rebuild the global PageIndexClient with new model overrides.

    Credentials live in the shared utils client (configure_llm); only model
    names need a client rebuild because closet_index/router are bound at __init__.
    Construct-first, then close+swap: a construction failure leaves the global
    `client` pointing at the still-live old instance (not a closed one), so the
    server is not bricked until restart.
    """
    global client
    if client is None:
        return
    overrides = {}
    if model:
        overrides["model"] = model
    if retrieve_model:
        overrides["retrieve_model"] = retrieve_model
    if not overrides:
        return
    new_client = PageIndexClient(
        workspace=_startup_workspace, db_path=_startup_db_path, **overrides
    )  # if this raises, the old `client` is untouched
    try:
        client.close()
    except Exception:
        pass
    client = new_client
    logger.info("Rebuilt PageIndexClient with overrides=%s", overrides)


def _safe_remove_upload(pdf_path: str, workspace) -> None:
    """Remove an uploaded file only if it resolves under workspace/uploads/.

    TOCTOU guard: resolve() + parent check before os.remove; FileNotFoundError
    swallowed (idempotent); other OSError logged but non-blocking. Refuses to
    delete files outside the uploads directory to prevent accidental removal
    of unrelated user files (spec §5.4 / R2).
    """
    try:
        upload_dir = (Path(workspace) / "uploads").resolve()
        target = Path(pdf_path).resolve()
        # Guard: only delete files that live inside workspace/uploads/.
        if upload_dir not in target.parents:
            logger.warning(
                "Skipping disk cleanup: %s not under uploads/", pdf_path
            )
            return
        os.remove(target)
    except FileNotFoundError:
        pass  # idempotent — file already gone
    except OSError as e:
        logger.warning("Failed to remove upload file %s: %s", pdf_path, e)


def delete_document_internal(c, db_id: int) -> bool:
    """Shared delete pipeline for MCP `delete_document` and REST
    `DELETE /api/documents/{doc_id}` (v1.1 spec §5.1.1 / AC10.3 / AC10.7).

    Performs the full W2 delete chain: DB cascade (FR1) -> closet index
    invalidation -> super-tree + kb_identity invalidation -> guarded disk
    cleanup (FR4). R7 timing is preserved: pdf_path is captured BEFORE
    `c.db.delete_document` because the row (and its `pdf_path` column)
    is gone after the cascade. Each step is independently try/except-wrapped
    so one failure cannot mask later cleanup; all failures are logged but
    do not raise (W2 behavior the MCP handler has always exposed).

    Returns True on the no-error path; False only when the DB layer is
    unavailable (c.db is None) -- the REST handler maps this to 503.
    """
    if c.db is None:
        return False
    # R7 timing: fetch pdf_path BEFORE cascade delete (the row and its
    # pdf_path column are gone after delete_document).
    try:
        doc = c.db.get_document_by_id(db_id)
        pdf_path = doc.get("pdf_path") if doc else None
    except Exception as e:
        logger.warning("Failed to fetch document for disk cleanup: %s", e)
        pdf_path = None

    # FR1: DB cascade delete (documents + child tables).
    try:
        c.db.delete_document(db_id)
    except Exception as e:
        logger.warning("Failed to delete document from DB: %s", e)

    # FR2: invalidate closet tags (idempotent via cascade even if this fails).
    if getattr(c, "closet_index", None):
        try:
            c.closet_index.remove_document(db_id)
        except Exception as e:
            logger.warning("Failed to delete closet tags: %s", e)

    # FR2: invalidate super-tree keyword index + kb identity.
    if getattr(c, "super_tree_index", None):
        try:
            c.super_tree_index.on_document_removed(db_id)
        except Exception as e:
            logger.warning("Failed to invalidate super-tree index: %s", e)

    # FR4: remove the uploaded file from disk (guarded by _safe_remove_upload).
    if pdf_path:
        _safe_remove_upload(pdf_path, c.workspace)
    return True


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
            if db_id is not None:
                # v1.1 spec §5.1.1: MCP and REST share delete_document_internal
                # so behavior cannot drift between surfaces (AC10.6 / AC10.7).
                delete_document_internal(c, db_id)
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


async def documents_endpoint(request: Request) -> Response:
    c = get_client()
    docs = []
    if c.db is not None:
        try:
            docs = c.db.get_all_documents()
        except Exception:
            logger.exception("documents_endpoint failed")
    safe = [
        {
            "id": d.get("id"),
            "doc_name": d.get("pdf_name"),
            "doc_description": d.get("doc_description"),
            "pdf_path": d.get("pdf_path"),
        }
        for d in docs
    ]
    return JSONResponse({"documents": safe})


async def document_delete_endpoint(request: Request) -> Response:
    """v1.1 spec section 5.1.1 / FR10 -- REST DELETE for a single document.

    Path param `doc_id` is the integer primary key from `documents.id`
    (NOT the v1.0 MCP UUID; see R-X1). Delegates the actual delete work
    to `delete_document_internal` so MCP and REST cannot drift (AC10.7).

    Responses:
      200 {"success": true, "doc_id": <int>}        -- normal delete or
         idempotent no-op (row already gone).
      400 {"error": "doc_id must be integer"}        -- non-integer path param.
      404 {"error": "Document not found"}            -- internal reports False.
      500 {"error": "Delete failed: ..."}            -- unexpected exception.
    """
    raw = request.path_params.get("doc_id")
    try:
        doc_id = int(raw)
    except (TypeError, ValueError):
        return JSONResponse({"error": "doc_id must be integer"}, status_code=400)
    try:
        c = get_client()
        if c.db is None:
            return JSONResponse({"error": "DB unavailable"}, status_code=503)
        ok = delete_document_internal(c, doc_id)
        if not ok:
            return JSONResponse({"error": "Document not found"}, status_code=404)
    except Exception as e:
        logger.exception("document_delete_endpoint failed")
        return JSONResponse({"error": f"Delete failed: {e}"}, status_code=500)
    return JSONResponse({"success": True, "doc_id": doc_id})



async def search_endpoint(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    query = (body or {}).get("query")
    if not query or not str(query).strip():
        return JSONResponse({"error": "Missing 'query'"}, status_code=400)
    top_k = body.get("top_k", 3)
    try:
        result = await get_client().search(str(query).strip(), top_k=top_k)
    except Exception as e:
        logger.exception("search_endpoint failed")
        return JSONResponse({"error": f"Search failed: {e}"}, status_code=500)
    # Enrich matched_docs with doc_name for the UI: search returns doc_id=uuid,
    # which does not align with /api/documents db-integer ids. The client's
    # in-memory `documents` dict is uuid-keyed and holds doc_name, so resolve
    # here. Additive + getattr-guarded so stub clients (e.g. SimpleNamespace
    # without a `documents` attr) don't crash.
    cl = get_client()
    docs_in_mem = getattr(cl, "documents", None) or {}
    for md in result.get("matched_docs", []) or []:
        doc = docs_in_mem.get(md.get("doc_id"))
        if doc:
            md.setdefault("doc_name", doc.get("doc_name", ""))
    return JSONResponse(result)


async def config_get_endpoint(request: Request) -> Response:
    snap = web_config.read_config_snapshot(get_client())
    return JSONResponse(snap)


async def config_post_endpoint(request: Request) -> Response:
    try:
        body = await request.json() or {}
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    persist = body.get("persist", True)
    model = body.get("model")
    retrieve_model = body.get("retrieve_model")
    api_key = body.get("api_key")
    base_url = body.get("base_url")

    persisted = False
    wrote_any = model or retrieve_model or api_key or base_url
    try:
        # 1) Runtime apply (immediate effect) + sync os.environ so the read-back
        #    snapshot (which reads os.getenv via ConfigLoader/get_llm_config) is
        #    consistent. configure_llm rebuilds the shared client but does NOT touch
        #    os.environ; _rebuild_client sets client.model but ConfigLoader().load()
        #    re-reads os.getenv — so without these env syncs the snapshot would be stale.
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url
        if api_key or base_url:
            configure_llm(api_key=api_key, base_url=base_url)
        if model:
            os.environ["MODEL_NAME"] = model
        if retrieve_model:
            os.environ["RETRIEVE_MODEL_NAME"] = retrieve_model
        if model or retrieve_model:
            _rebuild_client(model=model, retrieve_model=retrieve_model)

        # 2) Persist to disk (per-file guarded backup -> targeted write -> verify).
        #    spec §5.4 step 4 + §7.2: on ANY failure in apply or persist, restore the
        #    .bak backups written below so config.yaml/.env return to their original
        #    pre-request state (consistent failure state).
        if persist and wrote_any:
            if model or retrieve_model:
                # config.yaml: back up + line-replace only when present
                if web_config.DEFAULT_CONFIG_YAML.exists():
                    web_config.backup_file(web_config.DEFAULT_CONFIG_YAML)
                web_config.set_config_yaml_model(
                    web_config.DEFAULT_CONFIG_YAML, model=model, retrieve_model=retrieve_model)
                env_fields = {}
                if model:
                    env_fields["MODEL_NAME"] = model
                if retrieve_model:
                    env_fields["RETRIEVE_MODEL_NAME"] = retrieve_model
                if web_config.DEFAULT_ENV.exists():
                    web_config.backup_file(web_config.DEFAULT_ENV)
                web_config.set_env_fields(web_config.DEFAULT_ENV, env_fields)
            cred_fields = {}
            if api_key:
                cred_fields["OPENAI_API_KEY"] = api_key
            if base_url:
                cred_fields["OPENAI_BASE_URL"] = base_url
            if cred_fields:
                if web_config.DEFAULT_ENV.exists():
                    web_config.backup_file(web_config.DEFAULT_ENV)
                web_config.set_env_fields(web_config.DEFAULT_ENV, cred_fields)
            persisted = True
    except Exception as e:
        logger.exception("config apply/persist failed; restoring .bak backups")
        # spec §5.4 step 4: roll back each file whose .bak exists (restore original).
        # Each restore is independently guarded so a restore failure can't mask the
        # original error (it is logged instead).
        for path in (web_config.DEFAULT_CONFIG_YAML, web_config.DEFAULT_ENV):
            try:
                bak = path.with_suffix(path.suffix + ".bak")
                if bak.exists():
                    shutil.copy2(bak, path)
            except Exception as restore_err:
                logger.error("rollback restore failed for %s: %s", path, restore_err)
        return JSONResponse({"error": f"Persist failed: {e}", "applied": True,
                             "persisted": False}, status_code=500)
    snap = web_config.read_config_snapshot(get_client())
    snap.update({"applied": True, "persisted": persisted})
    return JSONResponse(snap)


async def config_test_endpoint(request: Request) -> Response:
    """FR12 ping — model connectivity test (spec §5.1.2).

    Body (all optional, fallback to active config):
      {model?: str, api_key?: str, base_url?: str}

    Performs a minimal openai-compatible chat.completions.create call wrapped
    in `asyncio.to_thread` + `asyncio.wait_for(timeout=10)`. NEVER writes state
    (no configure_llm / env writes / yaml writes) — pure read-only ping.

    Always returns HTTP 200. Errors are surfaced as `ok: false` in the body so
    the frontend can render green/red based on `ok` (per spec §5.1.2 / AC12.4).
    """
    import time as _time

    # Parse body (malformed JSON -> 400)
    try:
        body = await request.json() or {}
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    # A-RESOLVED-7: body overrides; fall back to active values.
    active_key, active_url = get_llm_config()
    api_key = body.get("api_key") or active_key
    base_url = body.get("base_url") or active_url
    # model: body overrides; else current ConfigLoader().load(None).model
    model = body.get("model") or ConfigLoader().load(None).model

    if not api_key or not model:
        return JSONResponse(
            {"ok": False, "error": "missing api_key or model",
             "model": model, "base_url": base_url},
            status_code=200,
        )

    # Build a TEMP openai client — never mutate the shared `_client`/`_async_client`.
    # This keeps the ping read-only and lets the active client keep serving traffic.
    temp_client = OpenAI(api_key=api_key, base_url=base_url)
    start = _time.monotonic()

    def _ping_call():
        # A-RESOLVED-6: max_tokens=8, 10s timeout (via asyncio.wait_for outside).
        return temp_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8,
        )

    try:
        # Run blocking sync openai call in a worker thread so the event loop
        # stays free (NFR3: do not block other endpoints).
        result = await asyncio.wait_for(
            asyncio.to_thread(_ping_call),
            timeout=10.0,
        )
        latency_ms = int((_time.monotonic() - start) * 1000)
        # Extract response snippet when available (best-effort, openai-shaped).
        snippet = ""
        try:
            if result and getattr(result, "choices", None):
                snippet = (result.choices[0].message.content or "")[:200]
        except Exception:
            snippet = ""
        return JSONResponse(
            {"ok": True, "latency_ms": latency_ms, "model": model,
             "base_url": base_url, "response": snippet},
            status_code=200,
        )
    except asyncio.TimeoutError:
        # On timeout the budget was fully consumed; report the 10s budget so the
        # frontend can show "Timeout after 10s" deterministically even if the
        # caller never actually waited (test stubs, etc.).
        latency_ms = int((_time.monotonic() - start) * 1000)
        if latency_ms < 10_000:
            latency_ms = 10_000
        return JSONResponse(
            {"ok": False, "error": "timeout", "latency_ms": latency_ms,
             "model": model, "base_url": base_url},
            status_code=200,
        )
    except openai.AuthenticationError as e:
        latency_ms = int((_time.monotonic() - start) * 1000)
        return JSONResponse(
            {"ok": False, "error": "auth_error", "detail": str(e),
             "latency_ms": latency_ms, "model": model, "base_url": base_url},
            status_code=200,
        )
    except openai.NotFoundError as e:
        latency_ms = int((_time.monotonic() - start) * 1000)
        return JSONResponse(
            {"ok": False, "error": "model_not_found", "detail": str(e),
             "latency_ms": latency_ms, "model": model, "base_url": base_url},
            status_code=200,
        )
    except Exception as e:
        latency_ms = int((_time.monotonic() - start) * 1000)
        logger.exception("config_test_endpoint unexpected error")
        return JSONResponse(
            {"ok": False, "error": type(e).__name__, "detail": str(e),
             "latency_ms": latency_ms, "model": model, "base_url": base_url},
            status_code=200,
        )


# Global semaphore to limit concurrent indexing during batch uploads
_UPLOAD_SEMAPHORE = asyncio.Semaphore(3)


def _determine_mode(filename: str) -> str:
    """Determine indexing mode from filename extension."""
    name_lower = filename.lower()
    if name_lower.endswith(".pdf"):
        return "pdf"
    elif name_lower.endswith(".md") or name_lower.endswith(".markdown"):
        return "md"
    return "auto"


async def _index_one_file(temp_path: Path, filename: str) -> dict:
    """Index a single file with concurrency limiting.

    Returns a result dict with filename, success, and either doc_id or error.
    """
    mode = _determine_mode(filename)
    try:
        async with _UPLOAD_SEMAPHORE:
            doc_id = await asyncio.to_thread(
                get_client().index, str(temp_path), mode=mode
            )
        return {
            "filename": filename,
            "success": True,
            "doc_id": doc_id,
        }
    except Exception as e:
        logger.exception("Indexing failed for %s", filename)
        return {
            "filename": filename,
            "success": False,
            "error": str(e),
        }


async def upload_endpoint(request: Request) -> Response:
    """Upload PDF or Markdown files and index them (batch upload supported)."""
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

    # Collect all file fields (support "file" and "files[]")
    raw_files = form.get("file")
    if raw_files is None:
        raw_files = form.get("files[]")

    if raw_files is None:
        return JSONResponse(
            {"error": "Missing 'file' or 'files[]' field"}, status_code=400
        )

    # Normalize to a list regardless of single or multiple uploads
    if not isinstance(raw_files, list):
        raw_files = [raw_files]

    if len(raw_files) == 0:
        return JSONResponse({"error": "No files provided"}, status_code=400)

    # Prepare upload directory
    upload_dir = Path(get_client().workspace) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save all files to temp locations first
    temp_paths: list[tuple[Path, str]] = []
    for file_field in raw_files:
        # Sanitize to basename only to prevent path traversal
        safe_name = Path(file_field.filename).name
        # Use UUID prefix to prevent filename collisions between concurrent uploads
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        temp_path = upload_dir / unique_name
        content = await file_field.read()
        await asyncio.to_thread(temp_path.write_bytes, content)
        temp_paths.append((temp_path, safe_name))

    # Process indexing concurrently with semaphore limiting
    tasks = [
        _index_one_file(temp_path, filename)
        for temp_path, filename in temp_paths
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Normalize any unexpected exceptions into error results
    normalized_results: list[dict] = []
    for idx, res in enumerate(results):
        filename = temp_paths[idx][1]
        if isinstance(res, Exception):
            logger.exception("Unexpected error indexing %s", filename)
            normalized_results.append({
                "filename": filename,
                "success": False,
                "error": str(res),
            })
        else:
            normalized_results.append(res)

    succeeded = sum(1 for r in normalized_results if r.get("success"))
    failed = len(normalized_results) - succeeded

    return JSONResponse({
        "results": normalized_results,
        "total": len(normalized_results),
        "succeeded": succeeded,
        "failed": failed,
    })


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
    global client, _startup_workspace, _startup_db_path
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

    _startup_workspace = workspace
    _startup_db_path = db_path
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
    Middleware(NoCacheStaticMiddleware),
]

routes = [
    Route("/", endpoint=lambda request: FileResponse(str(WEB_DIR / "index.html")), methods=["GET"]),
    Route("/health", endpoint=health_endpoint, methods=["GET"]),
    Route("/upload", endpoint=upload_endpoint, methods=["POST", "OPTIONS"]),
    Route("/api/documents", endpoint=documents_endpoint, methods=["GET"]),
    Route("/api/documents/{doc_id:int}", endpoint=document_delete_endpoint, methods=["DELETE"]),
    Route("/api/search", endpoint=search_endpoint, methods=["POST"]),
    Route("/api/config", endpoint=config_get_endpoint, methods=["GET"]),
    Route("/api/config", endpoint=config_post_endpoint, methods=["POST"]),
    Route("/api/config/test", endpoint=config_test_endpoint, methods=["POST"]),
    Route("/sse", endpoint=handle_sse, methods=["GET"]),
    Mount("/static", app=StaticFiles(directory=str(WEB_DIR / "static")), name="static"),
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
