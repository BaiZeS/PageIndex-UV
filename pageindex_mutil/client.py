import logging
import os
import threading
import uuid
import json
import asyncio
import concurrent.futures
from pathlib import Path

from .page_index import page_index
from .page_index_md import md_to_tree
from .retrieve import get_document, get_document_structure, get_page_content
from .utils import ConfigLoader, remove_fields, create_clean_structure_for_description, create_node_mapping, configure_llm
from .closet_index import ClosetIndex
from .super_tree import SuperTreeIndex
from .corpus_tree import CorpusTreeBuilder
from .page_index_liteparse import is_liteparse_format, liteparse_to_tree

# Import search backends
from .search_backend import SearchBackend
from .keyword_backend import KeywordSearchBackend

try:
    from .chroma_backend import ChromaSearchBackend
    from .hybrid_backend import HybridSearchBackend
except ImportError:
    ChromaSearchBackend = None
    HybridSearchBackend = None
from .entity_extractor import EntityExtractor

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


class DocIdMapper:
    """Centralized UUID ↔ DB ID mapping with bidirectional lookup."""

    def __init__(self):
        self._uuid_to_db: dict[str, int] = {}
        self._db_to_uuid: dict[int, str] = {}

    def register(self, uuid_id: str, db_id: int) -> None:
        self._uuid_to_db[uuid_id] = db_id
        self._db_to_uuid[db_id] = uuid_id

    def unregister(self, uuid_id: str) -> int | None:
        db_id = self._uuid_to_db.pop(uuid_id, None)
        if db_id is not None:
            self._db_to_uuid.pop(db_id, None)
        return db_id

    def to_db(self, uuid_id: str) -> int | None:
        return self._uuid_to_db.get(uuid_id)

    def to_uuid(self, db_id: int) -> str | None:
        return self._db_to_uuid.get(db_id)

    def items(self):
        return self._uuid_to_db.items()


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
    def __init__(self, api_key: str = None, model: str = None, retrieve_model: str = None, 
                 workspace: str = None, db_path: str = None,
                 search_backend: str = "hybrid", vector_db_path: str = "./data/vectors"):
        # Delegate LLM credentials/endpoint to the unified config source in utils.
        # configure_llm() handles OPENAI_API_KEY + CHATGPT_API_KEY alias and rebuilds
        # the shared OpenAI/AsyncOpenAI clients.
        if api_key:
            configure_llm(api_key=api_key)
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
        # Optional persistent layer for agentic retrieval
        self.db = None
        self.closet_index = None
        self.super_tree_index = None
        self.corpus_tree = None
        self.router = None
        self._id_mapper = DocIdMapper()
        self._pending_enrichment: set[int] = set()

        if self.workspace:
            self._load_workspace()

        if db_path and PageIndexDB:
            self.db = PageIndexDB(db_path)
            self.closet_index = ClosetIndex(self.db, self.model, self.retrieve_model)
            self.super_tree_index = SuperTreeIndex(self.db, self.model, self, self.retrieve_model)
            self.corpus_tree = CorpusTreeBuilder(self.db, self.model, self.retrieve_model)
            if AgenticRouter:
                self.router = AgenticRouter(self, self.model, self.retrieve_model)

            # Initialize search backends (ChromaDB hybrid is required)
            self._init_search_backends(search_backend, vector_db_path)

    @property
    def _uuid_to_db(self):
        """Backward-compatible access to the UUID→DB mapping."""
        return self._id_mapper._uuid_to_db

    def _init_search_backends(self, search_backend: str, vector_db_path: str):
        """Initialize search backends based on configuration."""
        try:
            if search_backend == "keyword":
                # Keyword-only mode: jieba + SQLite, no embedding model
                self.chroma_backend = None
                self.search_backend = KeywordSearchBackend(self.db)
                logging.info("Initialized keyword search backend (jieba + tags)")
            elif search_backend in ("hybrid", "chroma"):
                if ChromaSearchBackend is None:
                    logging.warning("chromadb/sentence-transformers not installed; falling back to keyword backend")
                    self.chroma_backend = None
                    self.search_backend = KeywordSearchBackend(self.db)
                else:
                    self.chroma_backend = ChromaSearchBackend(
                        db_path=vector_db_path,
                        embedding_model="local"
                    )
                    if search_backend == "hybrid":
                        self.search_backend = HybridSearchBackend(
                            self.db,
                            self.chroma_backend
                        )
                        logging.info("Initialized hybrid search backend (ChromaDB + keywords)")
                    else:
                        self.search_backend = self.chroma_backend
                        logging.info("Initialized ChromaDB vector search backend")
            else:
                # Default to keyword (fast, no model loading)
                self.chroma_backend = None
                self.search_backend = KeywordSearchBackend(self.db)
                logging.info("Initialized keyword search backend (default)")

            # Initialize entity extractor
            self.entity_extractor = EntityExtractor(self.model, self.retrieve_model)

        except Exception as e:
            logging.warning("Failed to initialize search backends: %s", e)
            logging.warning("Vector search will be unavailable")
            self.search_backend = None
            self.chroma_backend = None
            self.entity_extractor = None

    def _resolve_entity(
        self, entity_type: str, name: str, aliases: list,
        extractor: "EntityExtractor",
    ) -> int:
        """Resolve a new entity against the existing set, merging or creating.

        Incremental pattern (mirrors P1 tag normalization):
        1. Quick check: name/alias overlap → merge immediately.
        2. LLM adjudication via disambiguate_entity → merge if flagged.
        3. No match / LLM failure → create new entity (conservative).
        Returns the entity ID (existing or newly created).
        """
        existing = self.db.get_entities_by_type(entity_type)
        if not existing:
            return self.db.insert_entity(entity_type, name, aliases)

        # --- Quick merge: name or alias overlap ---
        for ent in existing:
            ent_aliases = json.loads(ent.get("aliases", "[]") or "[]")
            if name == ent["name"] or name in ent_aliases:
                if aliases:
                    self.db.merge_entity_aliases(ent["id"], aliases)
                return ent["id"]
            if aliases and ent["name"] in aliases:
                self.db.merge_entity_aliases(ent["id"], aliases)
                return ent["id"]
            if aliases and ent_aliases and set(aliases) & set(ent_aliases):
                self.db.merge_entity_aliases(ent["id"], aliases)
                return ent["id"]

        # --- LLM adjudication ---
        match = extractor.disambiguate_entity(name, aliases, existing)
        if match:
            if aliases:
                self.db.merge_entity_aliases(match["id"], aliases)
            return match["id"]

        # --- No match: create new ---
        return self.db.insert_entity(entity_type, name, aliases)

    def index(self, file_path: str, mode: str = "auto", sync: bool = True) -> str:
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
            import PyPDF2
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
        elif mode == "auto" and is_liteparse_format(file_path):
            logging.info("Indexing via LiteParse: %s", file_path)
            coro = liteparse_to_tree(
                file_path=file_path,
                model=self.model,
                if_thinning=False,
                if_add_node_summary='yes',
                summary_token_threshold=200,
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

            doc_type = ext.lstrip('.')
            self.documents[doc_id] = {
                'id': doc_id,
                'type': doc_type,
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
                self._id_mapper.register(doc_id, db_doc_id)
                
                # Save tree structure to database for persistence
                if doc.get('structure'):
                    from .utils import create_clean_structure_for_description
                    tree_for_reasoning = create_clean_structure_for_description(doc['structure'])
                    self.db.update_document_tree(db_doc_id, json.dumps(tree_for_reasoning, ensure_ascii=False))
                
                if self.closet_index and doc.get('structure'):
                    self.closet_index.add_document(
                        db_doc_id,
                        doc.get('doc_name', ''),
                        doc.get('doc_description', ''),
                        doc['structure']
                    )
                # Index Super-Tree keywords
                if hasattr(self, 'super_tree_index') and self.super_tree_index:
                    self.super_tree_index.on_document_added(db_doc_id)

                # Phase 2: corpus tree, search backend, entity extraction.
                # By default (sync=True) this runs inline.
                if sync:
                    self._enrich_document(db_doc_id, doc)
                else:
                    # Snapshot fields Phase 2 needs BEFORE launching thread.
                    # _save_doc will pop structure/pages from the shared dict,
                    # so without a snapshot the thread would see empty values.
                    phase2_doc = {
                        'doc_name': doc.get('doc_name', ''),
                        'doc_description': doc.get('doc_description', ''),
                        'structure': doc.get('structure'),
                        'pages': doc.get('pages'),
                    }
                    self._pending_enrichment.add(db_doc_id)
                    threading.Thread(
                        target=self._enrich_document,
                        args=(db_doc_id, phase2_doc),
                        daemon=True,
                    ).start()
            except Exception as e:
                logging.warning("Failed to persist to db: %s", e)

        logging.info("Indexing complete. Document ID: %s", doc_id)
        if self.workspace:
            self._save_doc(doc_id)
        return doc_id

    def _extract_entities_for_doc(self, db_doc_id: int, doc: dict) -> None:
        """Extract and store entities + relations for a document.

        Runs in a background thread, parallel with tags/corpus_tree/search_backend.
        """
        if not (self.entity_extractor and doc.get('structure')):
            return
        try:
            result = self.entity_extractor.extract_from_document(
                doc.get('doc_name', ''),
                doc.get('doc_description', ''),
                doc['structure']
            )
            entities, relations, node_contexts = result

            def _find_context(entity_name, contexts):
                name_lower = entity_name.lower()
                for ctx in contexts:
                    if name_lower in ctx.lower():
                        return ctx[:200]
                return None

            entity_ids = {}
            for entity in entities:
                eid = self._resolve_entity(
                    entity.entity_type,
                    entity.name,
                    entity.aliases,
                    self.entity_extractor,
                )
                if eid:
                    entity_ids[entity.name] = eid
                    context_snippet = _find_context(entity.name, node_contexts)
                    self.db.insert_entity_mention(
                        eid, db_doc_id,
                        context_snippet=context_snippet,
                        confidence=entity.confidence
                    )

            for rel in relations:
                subject_id = entity_ids.get(rel.subject)
                object_id = entity_ids.get(rel.object)
                if subject_id and object_id:
                    self.db.insert_entity_relation(
                        subject_id, rel.predicate, object_id,
                        doc_id=db_doc_id, confidence=rel.confidence
                    )

            logging.info("Extracted %d entities and %d relations for %s",
                       len(entities), len(relations), doc.get('doc_name'))
        except Exception as e:
            logging.warning("Entity extraction failed for doc %d: %s", db_doc_id, e)

    def _enrich_document(self, db_doc_id: int, doc: dict) -> None:
        """Phase 2: corpus tree + search backend + entity extraction.

        Runs in a background thread (or inline when sync=True).
        Entity extraction runs in parallel with tags/corpus_tree/search_backend
        (it is independent of those steps).
        Exceptions are caught and logged — never propagate.
        """
        try:
            # Start entity extraction in parallel (independent of tags/corpus_tree).
            entity_future = None
            if self.entity_extractor and doc.get('structure'):
                entity_future = concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(
                    self._extract_entities_for_doc, db_doc_id, doc
                )

            # Corpus tree (P1): incremental attach after closet tags exist.
            if self.corpus_tree:
                try:
                    self.corpus_tree.update_for_document(db_doc_id)
                except Exception as e:
                    logging.warning("Corpus tree incremental update failed: %s", e)

            # Index in vector search backend
            if self.search_backend and doc.get('structure'):
                try:
                    self.search_backend.index_document(
                        db_doc_id,
                        doc['structure'],
                        doc.get('pages')
                    )
                except Exception as e:
                    logging.warning("Failed to index in search backend: %s", e)

            # Wait for entity extraction to complete
            if entity_future is not None:
                try:
                    entity_future.result()
                except Exception as e:
                    logging.warning("Entity extraction thread failed for doc %d: %s", db_doc_id, e)

            logging.info("Background enrichment complete for doc %d", db_doc_id)
        except Exception as e:
            logging.warning("Background enrichment failed for doc %d: %s", db_doc_id, e)
        finally:
            self._pending_enrichment.discard(db_doc_id)

    def _parse_and_insert_doc(self, file_path: str, mode: str) -> tuple[str, int, dict]:
        """Parse a document and insert into DB. Returns (uuid_doc_id, db_doc_id, doc)."""
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
                doc=file_path, model=self.model,
                if_add_node_summary='yes', if_add_node_text='yes',
                if_add_node_id='yes', if_add_doc_description='yes')
            import PyPDF2
            pages = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(pdf_reader.pages, 1):
                    pages.append({'page': i, 'content': page.extract_text() or ''})
            self.documents[doc_id] = {
                'id': doc_id, 'type': 'pdf', 'path': file_path,
                'doc_name': result.get('doc_name', ''),
                'doc_description': result.get('doc_description', ''),
                'page_count': len(pages), 'structure': result['structure'], 'pages': pages,
            }
        elif mode == "md" or (mode == "auto" and is_md):
            logging.info("Indexing Markdown: %s", file_path)
            coro = md_to_tree(
                md_path=file_path, if_thinning=False,
                if_add_node_summary='yes', summary_token_threshold=200,
                model=self.model, if_add_doc_description='yes',
                if_add_node_text='yes', if_add_node_id='yes')
            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, coro).result()
            except RuntimeError:
                result = asyncio.run(coro)
            self.documents[doc_id] = {
                'id': doc_id, 'type': 'md', 'path': file_path,
                'doc_name': result.get('doc_name', ''),
                'doc_description': result.get('doc_description', ''),
                'line_count': result.get('line_count', 0), 'structure': result['structure'],
            }
        elif mode == "auto" and is_liteparse_format(file_path):
            logging.info("Indexing via LiteParse: %s", file_path)
            coro = liteparse_to_tree(
                file_path=file_path, model=self.model, if_thinning=False,
                if_add_node_summary='yes', summary_token_threshold=200,
                if_add_doc_description='yes', if_add_node_text='yes', if_add_node_id='yes')
            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, coro).result()
            except RuntimeError:
                result = asyncio.run(coro)
            doc_type = ext.lstrip('.')
            self.documents[doc_id] = {
                'id': doc_id, 'type': doc_type, 'path': file_path,
                'doc_name': result.get('doc_name', ''),
                'doc_description': result.get('doc_description', ''),
                'line_count': result.get('line_count', 0), 'structure': result['structure'],
            }
        else:
            raise ValueError(f"Unsupported file format for: {file_path}")

        doc = self.documents[doc_id]
        db_doc_id = self.db.insert_document(
            pdf_name=doc.get('doc_name', ''),
            pdf_path=file_path,
            doc_description=doc.get('doc_description', ''))
        self._id_mapper.register(doc_id, db_doc_id)

        if doc.get('structure'):
            from .utils import create_clean_structure_for_description
            tree_for_reasoning = create_clean_structure_for_description(doc['structure'])
            self.db.update_document_tree(db_doc_id, json.dumps(tree_for_reasoning, ensure_ascii=False))

        if self.closet_index and doc.get('structure'):
            self.closet_index.add_document(
                db_doc_id, doc.get('doc_name', ''),
                doc.get('doc_description', ''), doc['structure'])

        return doc_id, db_doc_id, doc

    def index_batch(self, file_paths: list[str], mode: str = "auto") -> list[str]:
        """Batch index multiple documents with batch normalization.

        Phase 1 — Extraction (per-doc, concurrent with semaphore)
        Phase 2 — Batch corpus tree rebuild
        Phase 3 — Batch entity normalization
        Phase 4 — Search backend + super_tree indexing

        Returns list of doc_ids (UUIDs).
        """
        if not self.db:
            raise RuntimeError("Database required for batch mode. Pass db_path to PageIndexClient.")

        import os as _os
        concurrency = int(_os.environ.get("LLM_CONCURRENCY", "2"))

        # -- Phase 1: per-doc extraction (concurrent) --
        phase1_results: list[tuple[str, int, dict]] = []
        llm_semaphore = threading.Semaphore(concurrency)

        def _extract_one(file_path: str) -> tuple[str, int, dict]:
            with llm_semaphore:
                return self._parse_and_insert_doc(file_path, mode)

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_extract_one, fp) for fp in file_paths]
            for fut in concurrent.futures.as_completed(futures):
                phase1_results.append(fut.result())

        # Entity + relation extraction (sequential, under semaphore)
        for _, db_doc_id, doc in phase1_results:
            if not (self.entity_extractor and doc.get('structure')):
                continue
            try:
                result = self.entity_extractor.extract_from_document(
                    doc.get('doc_name', ''), doc.get('doc_description', ''),
                    doc['structure'])
                entities, relations, node_contexts = result

                def _find_context(entity_name, contexts):
                    name_lower = entity_name.lower()
                    for ctx in contexts:
                        if name_lower in ctx.lower():
                            return ctx[:200]
                    return None

                entity_ids = {}
                for entity in entities:
                    eid = self.db.insert_entity(
                        entity.entity_type, entity.name, entity.aliases)
                    if eid:
                        entity_ids[entity.name] = eid
                        context_snippet = _find_context(entity.name, node_contexts)
                        self.db.insert_entity_mention(
                            eid, db_doc_id,
                            context_snippet=context_snippet,
                            confidence=entity.confidence)

                for rel in relations:
                    subject_id = entity_ids.get(rel.subject)
                    object_id = entity_ids.get(rel.object)
                    if subject_id and object_id:
                        self.db.insert_entity_relation(
                            subject_id, rel.predicate, object_id,
                            doc_id=db_doc_id, confidence=rel.confidence)

                logging.info("Extracted %d entities and %d relations for %s",
                             len(entities), len(relations), doc.get('doc_name'))
            except Exception as e:
                logging.warning("Entity extraction failed for doc %d: %s", db_doc_id, e)

        # -- Phase 2: batch corpus tree rebuild --
        try:
            self.rebuild_corpus_tree()
        except Exception as e:
            logging.warning("Batch corpus tree rebuild failed: %s", e)

        # -- Phase 3: batch entity normalization --
        if self.entity_extractor:
            try:
                self.entity_extractor.normalize_entities_batch(self.db)
            except Exception as e:
                logging.warning("Batch entity normalization failed: %s", e)

        # -- Phase 4: search backend + super_tree indexing --
        for doc_id, db_doc_id, doc in phase1_results:
            try:
                if hasattr(self, 'super_tree_index') and self.super_tree_index:
                    self.super_tree_index.on_document_added(db_doc_id)
                if self.search_backend and doc.get('structure'):
                    self.search_backend.index_document(
                        db_doc_id, doc['structure'], doc.get('pages'))
            except Exception as e:
                logging.warning("Phase 4 indexing failed for doc %d: %s", db_doc_id, e)

        doc_ids = [r[0] for r in phase1_results]
        logging.info("Batch indexing complete: %d documents", len(doc_ids))
        return doc_ids

    def rebuild_corpus_tree(self) -> dict:
        """Full (re)build of the corpus tree (P1). Returns the inspectable tree.

        Use for initial construction or periodic structural adjustment; the
        per-document incremental path is wired into index().
        """
        if not self.corpus_tree:
            return {}
        return self.corpus_tree.rebuild()

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
        
        # Also load documents from database if available
        if self.db:
            try:
                db_docs = self.db.get_all_documents()
                for db_doc in db_docs:
                    db_doc_id = db_doc['id']
                    pdf_name = db_doc.get('pdf_name', '')
                    pdf_path = db_doc.get('pdf_path', '')
                    
                    # Try to find matching workspace document by pdf_name or pdf_path
                    found_uuid = None
                    for uuid, doc in self.documents.items():
                        if (doc.get('doc_name') == pdf_name or 
                            doc.get('path') == pdf_path):
                            found_uuid = uuid
                            break
                    
                    if found_uuid:
                        # Document already loaded from workspace - register mapping
                        self._id_mapper.register(found_uuid, db_doc_id)
                        # Update with tree_json if available
                        if db_doc.get('tree_json'):
                            self.documents[found_uuid]['structure'] = json.loads(db_doc['tree_json'])
                    else:
                        # Document exists in DB but not in workspace - create entry
                        doc_id = str(uuid.uuid4())
                        self._id_mapper.register(doc_id, db_doc_id)
                        self.documents[doc_id] = {
                            'id': doc_id,
                            'type': 'pdf',
                            'doc_name': pdf_name,
                            'doc_description': db_doc.get('doc_description', ''),
                            'path': pdf_path,
                        }
                        if db_doc.get('tree_json'):
                            self.documents[doc_id]['structure'] = json.loads(db_doc['tree_json'])
                
                logging.info("Loaded %d document(s) from database, %d mappings registered", 
                           len(db_docs), len(self._uuid_to_db))
            except Exception as e:
                logging.warning("Failed to load from database: %s", e)

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

        # Import shared reasoning helpers
        try:
            from .reasoning import (
                get_relevant_nodes, pages_from_nodes, generate_answer,
                build_context_for_doc,
            )
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

        # Rank nodes by relevance: prefer nodes with longer summaries
        # (more detailed summaries indicate deeper coverage of the topic)
        selected.sort(
            key=lambda n: len(n.get("summary", "")),
            reverse=True
        )

        pages = pages_from_nodes(selected)

        # Assemble context using shared helper
        context = build_context_for_doc(doc, selected, pages)
        answer = generate_answer(query, context)

        return {
            "query": query,
            "mode": "single",
            "answer": answer,
            "confidence": "high",
            "matched_docs": [{"doc_id": doc_id, "score": 1.0}],
            "selected_nodes": [
                {
                    "node_id": n.get("node_id"),
                    "title": n.get("title"),
                    "summary": n.get("summary", ""),
                    "text": n.get("text", ""),
                    "pages": list(range(n.get("start_index") or 0, (n.get("end_index") or 0) + 1)) if n.get("start_index") else [],
                }
                for n in selected
            ],
            "pages": [
                {"doc_id": doc_id, "page": p, "text": (page_map.get(p, "") or "")[:500]}
                for p in sorted(set(pages))
            ],
        }
