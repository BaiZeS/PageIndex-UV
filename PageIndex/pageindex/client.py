import logging
import os
import uuid
import json
import asyncio
import concurrent.futures
from pathlib import Path

import PyPDF2

from .page_index import page_index
from .page_index_md import md_to_tree
from .retrieve import get_document, get_document_structure, get_page_content
from .utils import ConfigLoader, remove_fields, create_clean_structure_for_description, create_node_mapping
from .closet_index import ClosetIndex

# Optional: db.py lives at project root; gracefully degrade if unavailable.
try:
    from db import PageIndexDB
except ImportError:
    PageIndexDB = None  # type: ignore[misc,assignment]

try:
    from .agentic.router import AgenticRouter
except ImportError:
    AgenticRouter = None  # type: ignore[misc,assignment]

META_INDEX = "_meta.json"


def _normalize_retrieve_model(model: str) -> str:
    """Normalize model name for OpenAI-compatible endpoints."""
    if not model:
        return model
    return model


class PageIndexClient:
    """
    A client for indexing and retrieving document content.
    Flow: index() -> get_document() / get_document_structure() / get_page_content()

    For agent-based QA, see examples/agentic_vectorless_rag_demo.py.
    """
    def __init__(self, api_key: str = None, model: str = None, retrieve_model: str = None, workspace: str = None, db_path: str = None):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY") and os.getenv("CHATGPT_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.getenv("CHATGPT_API_KEY")
        self.workspace = Path(workspace).expanduser() if workspace else None
        overrides = {}
        if model:
            overrides["model"] = model
        if retrieve_model:
            overrides["retrieve_model"] = retrieve_model
        opt = ConfigLoader().load(overrides or None)
        self.model = opt.model
        self.retrieve_model = _normalize_retrieve_model(opt.retrieve_model or self.model)
        if self.workspace:
            self.workspace.mkdir(parents=True, exist_ok=True)
        self.documents = {}
        if self.workspace:
            self._load_workspace()

        # Optional persistent layer for agentic retrieval
        self.db = None
        self.closet_index = None
        self.router = None
        self._uuid_to_db: dict[str, int] = {}

        if db_path and PageIndexDB:
            self.db = PageIndexDB(db_path)
            self.closet_index = ClosetIndex(self.db, self.model)
            if AgenticRouter:
                self.router = AgenticRouter(self, self.model)

    def index(self, file_path: str, mode: str = "auto") -> str:
        """Index a document. Returns a document_id."""
        # Persist a canonical absolute path so workspace reloads do not
        # reinterpret caller-relative paths against the workspace directory.
        file_path = os.path.abspath(os.path.expanduser(file_path))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = str(uuid.uuid4())
        ext = os.path.splitext(file_path)[1].lower()

        is_pdf = ext == '.pdf'
        is_md = ext in ['.md', '.markdown']

        if mode == "pdf" or (mode == "auto" and is_pdf):
            logging.info("Indexing PDF: %s", file_path)
            result = page_index(
                doc=file_path,
                model=self.model,
                if_add_node_summary='yes',
                if_add_node_text='yes',
                if_add_node_id='yes',
                if_add_doc_description='yes'
            )
            # Extract per-page text so queries don't need the original PDF
            pages = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(pdf_reader.pages, 1):
                    pages.append({'page': i, 'content': page.extract_text() or ''})

            self.documents[doc_id] = {
                'id': doc_id,
                'type': 'pdf',
                'path': file_path,
                'doc_name': result.get('doc_name', ''),
                'doc_description': result.get('doc_description', ''),
                'page_count': len(pages),
                'structure': result['structure'],
                'pages': pages,
            }

        elif mode == "md" or (mode == "auto" and is_md):
            logging.info("Indexing Markdown: %s", file_path)
            coro = md_to_tree(
                md_path=file_path,
                if_thinning=False,
                if_add_node_summary='yes',
                summary_token_threshold=200,
                model=self.model,
                if_add_doc_description='yes',
                if_add_node_text='yes',
                if_add_node_id='yes'
            )
            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, coro).result()
            except RuntimeError:
                result = asyncio.run(coro)
            self.documents[doc_id] = {
                'id': doc_id,
                'type': 'md',
                'path': file_path,
                'doc_name': result.get('doc_name', ''),
                'doc_description': result.get('doc_description', ''),
                'line_count': result.get('line_count', 0),
                'structure': result['structure'],
            }
        else:
            raise ValueError(f"Unsupported file format for: {file_path}")

        # Persist metadata to db for agentic retrieval
        doc = self.documents[doc_id]
        if self.db:
            try:
                db_doc_id = self.db.insert_document(
                    pdf_name=doc.get('doc_name', ''),
                    pdf_path=file_path,
                    doc_description=doc.get('doc_description', '')
                )
                self._uuid_to_db[doc_id] = db_doc_id
                if self.closet_index and doc.get('structure'):
                    self.closet_index.add_document(
                        db_doc_id,
                        doc.get('doc_name', ''),
                        doc.get('doc_description', ''),
                        doc['structure']
                    )
            except Exception as e:
                logging.warning("Failed to persist to db: %s", e)

        logging.info("Indexing complete. Document ID: %s", doc_id)
        if self.closet_index and doc_id in self._uuid_to_db:
            db_id = self._uuid_to_db[doc_id]
            doc = self.documents.get(doc_id, {})
            try:
                self.closet_index.add_document(
                    db_id,
                    doc.get("doc_name", ""),
                    doc.get("doc_description", ""),
                    doc.get("structure", [])
                )
            except Exception as e:
                logging.warning("Failed to index closet tags: %s", e)
        if self.workspace:
            self._save_doc(doc_id)
        return doc_id

    @staticmethod
    def _make_meta_entry(doc: dict) -> dict:
        """Build a lightweight meta entry from a document dict."""
        entry = {
            'type': doc.get('type', ''),
            'doc_name': doc.get('doc_name', ''),
            'doc_description': doc.get('doc_description', ''),
            'path': doc.get('path', ''),
        }
        if doc.get('type') == 'pdf':
            entry['page_count'] = doc.get('page_count')
        elif doc.get('type') == 'md':
            entry['line_count'] = doc.get('line_count')
        return entry

    @staticmethod
    def _read_json(path) -> dict | None:
        """Read a JSON file, returning None on any error."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logging.warning("Corrupt %s: %s", Path(path).name, e)
            return None

    def _save_doc(self, doc_id: str):
        doc = self.documents[doc_id].copy()
        # Strip text from structure nodes — redundant with pages (PDF only)
        if doc.get('structure') and doc.get('type') == 'pdf':
            doc['structure'] = remove_fields(doc['structure'], fields=['text'])
        path = self.workspace / f"{doc_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        self._save_meta(doc_id, self._make_meta_entry(doc))
        # Drop heavy fields; will lazy-load on demand
        self.documents[doc_id].pop('structure', None)
        self.documents[doc_id].pop('pages', None)

    def _rebuild_meta(self) -> dict:
        """Scan individual doc JSON files and return a meta dict."""
        meta = {}
        for path in self.workspace.glob("*.json"):
            if path.name == META_INDEX:
                continue
            doc = self._read_json(path)
            if doc and isinstance(doc, dict):
                meta[path.stem] = self._make_meta_entry(doc)
        return meta

    def _read_meta(self) -> dict | None:
        """Read and validate _meta.json, returning None on any corruption."""
        meta = self._read_json(self.workspace / META_INDEX)
        if meta is not None and not isinstance(meta, dict):
            logging.warning("%s is not a JSON object, ignoring", META_INDEX)
            return None
        return meta

    def _save_meta(self, doc_id: str, entry: dict):
        meta = self._read_meta() or self._rebuild_meta()
        meta[doc_id] = entry
        meta_path = self.workspace / META_INDEX
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _load_workspace(self):
        meta = self._read_meta()
        if meta is None:
            meta = self._rebuild_meta()
            if meta:
                logging.info("Loaded %d document(s) from workspace (legacy mode).", len(meta))
        for doc_id, entry in meta.items():
            doc = dict(entry, id=doc_id)
            if doc.get('path') and not os.path.isabs(doc['path']):
                doc['path'] = str((self.workspace / doc['path']).resolve())
            self.documents[doc_id] = doc

    def _ensure_doc_loaded(self, doc_id: str):
        """Load full document JSON on demand (structure, pages, etc.)."""
        doc = self.documents.get(doc_id)
        if not doc or doc.get('structure') is not None:
            return
        full = self._read_json(self.workspace / f"{doc_id}.json")
        if not full:
            return
        doc['structure'] = full.get('structure', [])
        if full.get('pages'):
            doc['pages'] = full['pages']

    def get_document(self, doc_id: str) -> str:
        """Return document metadata JSON."""
        return get_document(self.documents, doc_id)

    def get_document_structure(self, doc_id: str) -> str:
        """Return document tree structure JSON (without text fields)."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_document_structure(self.documents, doc_id)

    def get_page_content(self, doc_id: str, pages: str) -> str:
        """Return page content for the given pages string (e.g. '5-7', '3,8', '12')."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_page_content(self.documents, doc_id, pages)

    def close(self):
        """Close database connection and release resources."""
        if self.db is not None:
            try:
                self.db.close()
            except Exception:
                pass
            self.db = None

    # ------------------------------------------------------------------
    # Search (single- and multi-document)
    # ------------------------------------------------------------------

    async def search(self, query: str, top_k: int = 3) -> dict:
        """Search across indexed documents.

        Single-document mode skips the agentic router and performs direct
        tree reasoning.  Multi-document mode runs the full
        Plan -> Route -> Act -> Verify pipeline.
        """
        if len(self.documents) == 1:
            doc_id = list(self.documents.keys())[0]
            return await self._search_single(query, doc_id)

        if self.router:
            return await self.router.search(query, top_k)

        return {
            "query": query,
            "mode": "multi",
            "answer": (
                "Router not available. Initialise PageIndexClient with db_path="
                "to enable multi-document search."
            ),
            "confidence": "unknown",
            "matched_docs": [],
            "selected_nodes": [],
            "pages": [],
        }

    async def _search_single(self, query: str, doc_id: str) -> dict:
        """Direct tree search for a single document (zero router overhead)."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)

        doc = self.documents.get(doc_id)
        if not doc:
            return {
                "query": query,
                "mode": "single",
                "answer": "Document not found.",
                "confidence": "unknown",
                "matched_docs": [],
                "selected_nodes": [],
                "pages": [],
            }

        structure = doc.get("structure", [])
        if not structure:
            return {
                "query": query,
                "mode": "single",
                "answer": "No document structure available.",
                "confidence": "unknown",
                "matched_docs": [],
                "selected_nodes": [],
                "pages": [],
            }

        # Lazy-import main.py helpers (project root must be on sys.path)
        try:
            import main as _main
            get_relevant_nodes = _main.get_relevant_nodes
            pages_from_nodes = _main.pages_from_nodes
            generate_answer = _main.generate_answer
        except ImportError:
            return {
                "query": query,
                "mode": "single",
                "answer": "Search backend not available.",
                "confidence": "unknown",
                "matched_docs": [],
                "selected_nodes": [],
                "pages": [],
            }

        tree_json = json.dumps(structure, ensure_ascii=False)
        node_ids = get_relevant_nodes(query, tree_json)
        if not node_ids:
            return {
                "query": query,
                "mode": "single",
                "answer": "No relevant sections found.",
                "confidence": "low",
                "matched_docs": [{"doc_id": doc_id, "score": 1.0}],
                "selected_nodes": [],
                "pages": [],
            }

        mapping = create_node_mapping(structure)
        selected = [mapping.get(nid) for nid in node_ids if nid in mapping]
        selected = [n for n in selected if n]
        if not selected:
            return {
                "query": query,
                "mode": "single",
                "answer": "No valid sections found.",
                "confidence": "low",
                "matched_docs": [{"doc_id": doc_id, "score": 1.0}],
                "selected_nodes": [],
                "pages": [],
            }

        pages = pages_from_nodes(selected)

        # Assemble context
        ctx_parts = [f"\n=== Document: {doc.get('doc_name', '')} ===\n"]
        if doc.get("type") == "pdf" and doc.get("pages"):
            page_map = {p["page"]: p["content"] for p in doc["pages"]}
            for p in sorted(set(pages)):
                text = page_map.get(p, "")
                if text:
                    ctx_parts.append(f"\n--- Page {p} ---\n{text}")
        elif doc.get("type") == "md":
            for node in selected:
                txt = node.get("text", "")
                if txt:
                    ctx_parts.append(
                        f"\n--- {node.get('title', '')} ---\n{txt}"
                    )

        context = "".join(ctx_parts)
        answer = generate_answer(query, context)

        return {
            "query": query,
            "mode": "single",
            "answer": answer,
            "confidence": "high",
            "matched_docs": [{"doc_id": doc_id, "score": 1.0}],
            "selected_nodes": [
                {"node_id": n.get("node_id"), "title": n.get("title")}
                for n in selected
            ],
            "pages": [{"doc_id": doc_id, "pages": sorted(set(pages))}],
        }
