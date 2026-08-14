import logging
import math
import os
import threading
import uuid
import json
import asyncio
import concurrent.futures
from collections import Counter
from pathlib import Path

from .page_index import page_index
from .page_index_md import md_to_tree
from .retrieve import get_document, get_document_structure, get_page_content
from .utils import ConfigLoader, remove_fields, create_clean_structure_for_description, create_node_mapping, configure_llm, llm_completion
from .closet_index import ClosetIndex
from .super_tree import SuperTreeIndex, KeywordIndex
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

# P1.2: cap on node-level mention rows per (entity, doc) — deterministic
# attribution can fan out on repetitive docs; keep the table bounded.
_MAX_NODE_MENTIONS_PER_ENTITY = 20

# P2: max nodes to feed to context assembly when LLM degrades to union passthrough
_MAX_DEGRADE_NODES = 20


def _iter_structure_nodes(structure):
    """Yield every node dict in a nested TOC structure, depth-first."""
    for node in structure or []:
        yield node
        children = node.get("nodes")
        if children:
            yield from _iter_structure_nodes(children)


def _extract_structure_text(structure) -> str:
    """Concatenate every node's text from a nested TOC structure (depth-first).

    Used to feed full document body into the doc-level keyword index. Needed
    because MD documents indexed via client.index_batch do NOT populate the
    pages table, so get_document_content() returns empty for them and the
    doc-level BM25 channel would otherwise see only titles/description.
    """
    parts = []
    for node in _iter_structure_nodes(structure):
        text = node.get("text")
        if text:
            parts.append(text)
    return "\n".join(parts)


# P2.6 关键词签名去噪：引用模板垃圾 token（纯数字/日期形/引用词）不得霸占
# top-K——eval 实证存储签名被 08/2015/官网/引用/日期 淹没，查询概念词落选。
_JUNK_KEYWORD_TOKENS = frozenset({"官网", "引用", "日期"})
_JUNK_KEYWORD_CHARS = frozenset("0123456789-/:.年月日")


def _is_junk_keyword_token(tok: str) -> bool:
    """纯数字/日期形/引用模板词 → True（无信息量，不进关键词签名）。"""
    if not any(c.isalnum() for c in tok):
        return True  # 纯标点/符号
    if any(c.isdigit() for c in tok) and all(c in _JUNK_KEYWORD_CHARS for c in tok):
        return True  # 纯数字或日期形：08、2015、2015-08-01、2015年8月1日
    return tok in _JUNK_KEYWORD_TOKENS


def _compute_node_keywords(structure, topk):
    """Per-node salient keywords via within-document TF-IDF (P1.4, no LLM).

    Tokenizes each node's title + (text or summary) with the shared jieba
    tokenizer (min length 2, stopwords dropped — same convention as
    entity→node attribution), then drops junk tokens (pure digits, date-like
    strings, citation-template words) so citation boilerplate cannot drown the
    informative top-K (P2.6 signature hygiene). tf = normalized count within
    the node; idf = smoothed log over the document's N nodes:
    log((N+1)/(df+1)) + 1, so a term in every node gets the minimum weight
    1.0 while a term unique to one node is boosted. Returns
    {node_id: [<= topk tokens]} ordered by score desc (ties broken by token
    for determinism). Nodes without usable text map to []. Complexity is
    O(total tokens) per document.
    """
    node_tokens = []
    for node in _iter_structure_nodes(structure):
        node_id = node.get("node_id")
        if not node_id:
            continue
        text = f"{node.get('title') or ''} " \
               f"{node.get('text') or node.get('summary') or ''}"
        # KeywordIndex._tokenize does not use `self`; reuse the canonical
        # jieba + _STOPWORDS filtering without constructing an index.
        node_tokens.append((
            node_id,
            [t for t in KeywordIndex._tokenize(None, text)
             if not _is_junk_keyword_token(t)],
        ))

    if not node_tokens or topk <= 0:
        return {node_id: [] for node_id, _ in node_tokens}

    n_docs = len(node_tokens)
    df = Counter()
    for _, tokens in node_tokens:
        df.update(set(tokens))

    keywords_by_node = {}
    for node_id, tokens in node_tokens:
        if not tokens:
            keywords_by_node[node_id] = []
            continue
        total = len(tokens)
        counts = Counter(tokens)
        scored = sorted(
            (-(cnt / total) * (math.log((n_docs + 1) / (df[tok] + 1)) + 1.0),
             tok)
            for tok, cnt in counts.items()
        )
        keywords_by_node[node_id] = [tok for _, tok in scored[:topk]]
    return keywords_by_node


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
        # P1.4: per-node salient-keyword count for the node profile signature.
        try:
            self._node_keyword_topk = int(getattr(opt, "node_keyword_topk", 5))
        except (TypeError, ValueError):
            self._node_keyword_topk = 5
        if self.workspace:
            self.workspace.mkdir(parents=True, exist_ok=True)
        self.documents = {}
        # Optional persistent layer for agentic retrieval
        self.db = None
        self.closet_index = None
        self.super_tree_index = None
        self.router = None
        self._id_mapper = DocIdMapper()
        self._pending_enrichment: set[int] = set()

        if self.workspace:
            self._load_workspace()

        if db_path and PageIndexDB:
            self.db = PageIndexDB(db_path)
            self.closet_index = ClosetIndex(self.db, self.model, self.retrieve_model)
            self.super_tree_index = SuperTreeIndex(self.db, self.model, self, self.retrieve_model)
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

    @staticmethod
    def _match_nodes_for_entity(name, aliases, structure,
                                limit=_MAX_NODE_MENTIONS_PER_ENTITY):
        """Deterministic entity→node attribution (no LLM).

        Returns node_ids of TOC nodes whose title/text contains the entity
        name or any alias (case-insensitive substring; sufficient for CJK).
        Nodes without 'text' (PDF) fall back to 'summary'. First `limit`
        matches win, in document order.
        """
        terms = [t.strip().casefold()
                 for t in ([name] + list(aliases or [])) if t and t.strip()]
        if not terms:
            return []
        matched = []
        for node in _iter_structure_nodes(structure):
            node_id = node.get("node_id")
            if not node_id:
                continue
            # Check fields separately so multi-word terms can't match across
            # the title/text boundary.
            fields = [
                (node.get("title") or "").casefold(),
                (node.get("text") or node.get("summary") or "").casefold(),
            ]
            if any(t in f for t in terms for f in fields):
                matched.append(node_id)
                if len(matched) >= limit:
                    break
        return matched

    def _insert_entity_mentions(self, eid, db_doc_id, entity,
                                context_snippet, structure):
        """Insert mention rows attributed to matching TOC nodes.

        One row per (entity, doc, matched node); a single doc-level row
        (node_id NULL) only when no node matches. Attribution uses the
        entity row's accumulated name + aliases (post-resolution).
        """
        row = self.db.get_entity_by_id(eid)
        name = row["name"] if row else entity.name
        try:
            aliases = json.loads((row or {}).get("aliases") or "[]")
        except ValueError:
            aliases = []
        if not aliases:
            aliases = list(entity.aliases or [])
        node_ids = self._match_nodes_for_entity(name, aliases, structure)
        if node_ids:
            for nid in node_ids:
                self.db.insert_entity_mention(
                    eid, db_doc_id, node_id=nid,
                    context_snippet=context_snippet,
                    confidence=entity.confidence,
                )
        else:
            self.db.insert_entity_mention(
                eid, db_doc_id, context_snippet=context_snippet,
                confidence=entity.confidence,
            )

    def _write_node_profiles(self, db_doc_id, structure=None):
        """Build and persist node profiles for a document (P1.2 + P1.4).

        Profiles aggregate entity_mentions (node-level rows) joined with the
        entities table, so names are ALWAYS canonical (post merge/normalize).
        Keywords are per-node TF-IDF top-K salient tokens (pure statistics,
        no LLM). Tags reuse the semantic subset of the doc-level closet_tags
        (source="llm" only; jieba fallback words are excluded). When a structure is
        given, EVERY node gets a profile row (entities may be empty) and the
        'entities'/'keywords' keys are attached onto the live structure node
        dicts. Returns the list of profile dicts written.
        """
        profiles_by_node = {}
        for m in self.db.get_entity_mentions_by_doc(db_doc_id):
            node_id = m.get("node_id")
            if not node_id:
                continue
            entries = profiles_by_node.setdefault(node_id, [])
            entry = {"name": m["entity_name"], "type": m["entity_type"]}
            if entry not in entries:
                entries.append(entry)
        keywords_by_node = (
            _compute_node_keywords(structure, self._node_keyword_topk)
            if structure else {}
        )
        # Union: structure nodes in document order, then any mention-only
        # node ids not present in the structure (defensive).
        node_ids = []
        seen = set()
        if structure:
            for node in _iter_structure_nodes(structure):
                node_id = node.get("node_id")
                if node_id and node_id not in seen:
                    seen.add(node_id)
                    node_ids.append(node_id)
        for node_id in profiles_by_node:
            if node_id not in seen:
                seen.add(node_id)
                node_ids.append(node_id)
        # Node-profile 标签是语义属性：只取 LLM 抽象标签（fallback 原词不进语义漏斗）
        tags = [t["tag_text"] for t in self.db.get_doc_tags(db_doc_id, source="llm")]
        profiles = [
            {"node_id": node_id,
             "entities": profiles_by_node.get(node_id, []),
             "keywords": keywords_by_node.get(node_id, []),
             "tags": tags}
            for node_id in node_ids
        ]
        self.db.upsert_node_profiles(db_doc_id, profiles)
        if structure:
            for node in _iter_structure_nodes(structure):
                node_id = node.get("node_id")
                node["entities"] = profiles_by_node.get(node_id, [])
                node["keywords"] = keywords_by_node.get(node_id, [])
        return profiles

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
                    self.super_tree_index.on_document_added(
                        db_doc_id, content=_extract_structure_text(doc.get('structure')))

                # Phase 2: doc_summary, search backend, entity extraction.
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

        Runs in a background thread, parallel with doc_summary/search_backend.
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

            # Re-index: replace this doc's mentions (one row per entity/doc/node)
            self.db.delete_entity_mentions(db_doc_id)

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
                    self._insert_entity_mentions(
                        eid, db_doc_id, entity,
                        context_snippet, doc.get('structure')
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

    def _generate_doc_summary(self, db_doc_id: int, doc: dict) -> None:
        """Generate the document-level grounded summary (doc_summary, [S3]).

        Input is the cleaned structure (titles + summaries) via the existing
        description pipeline helper. LLM failure / empty response leaves the
        column empty — L1 presentation falls back to doc_description (T9).
        """
        if not doc.get('structure'):
            return
        try:
            structure = create_clean_structure_for_description(doc['structure'])
            prompt = (
                "你是一个文档摘要专家。基于文档结构生成覆盖式接地摘要（≤200字）："
                "涵盖主要章节范围、关键实体与概念、适合回答的问题类型。"
                "仅输出摘要文本。\n\n文档结构：\n"
                f"{json.dumps(structure, ensure_ascii=False)}"
            )
            # NFR4: retrieval-path LLM call sites use retrieve_model or model.
            summary = llm_completion(
                self.retrieve_model or self.model, prompt, thinking_disabled=True
            )
            if summary:
                self.db.update_doc_summary(db_doc_id, summary.strip())
        except Exception as e:
            logging.warning("doc_summary generation failed for doc %d: %s", db_doc_id, e)

    def _enrich_document(self, db_doc_id: int, doc: dict) -> None:
        """Phase 2: doc_summary + search backend + entity extraction.

        Runs in a background thread (or inline when sync=True).
        Entity extraction and doc_summary generation run in parallel with the
        search backend index (they are independent of it).
        Exceptions are caught and logged — never propagate.
        """
        try:
            # Start entity extraction in parallel (independent of search backend).
            entity_future = None
            if self.entity_extractor and doc.get('structure'):
                entity_future = concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(
                    self._extract_entities_for_doc, db_doc_id, doc
                )

            # doc_summary ([S3]): document-level grounded summary, generated in
            # parallel with entity extraction. Empty on failure → L1 falls back
            # to doc_description.
            summary_future = None
            if self.db and doc.get('structure'):
                summary_future = concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(
                    self._generate_doc_summary, db_doc_id, doc
                )

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

            # Wait for entity extraction + doc_summary to complete
            if entity_future is not None:
                try:
                    entity_future.result()
                except Exception as e:
                    logging.warning("Entity extraction thread failed for doc %d: %s", db_doc_id, e)
            if summary_future is not None:
                try:
                    summary_future.result()
                except Exception as e:
                    logging.warning("doc_summary thread failed for doc %d: %s", db_doc_id, e)

            # Node profiles (P1.2): canonical entity signatures per TOC node.
            # Runs after entity extraction so mentions exist; also attaches the
            # 'entities' key onto the structure before _save_doc persists it.
            if self.db and doc.get('structure'):
                try:
                    self._write_node_profiles(db_doc_id, doc['structure'])
                except Exception as e:
                    logging.warning("Node profile build failed for doc %d: %s", db_doc_id, e)

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
        Phase 2 — Batch entity normalization
        Phase 3 — Search backend + super_tree indexing
        Phase 4 — Node profiles (post-normalization → canonical entities)

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

        # Entity + relation extraction (parallel with semaphore)
        def _extract_entities_one(item):
            _, db_doc_id, doc = item
            if not (self.entity_extractor and doc.get('structure')):
                return
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

                # Re-index: replace this doc's mentions (one row per entity/doc/node)
                self.db.delete_entity_mentions(db_doc_id)

                entity_ids = {}
                for entity in entities:
                    eid = self.db.insert_entity(
                        entity.entity_type, entity.name, entity.aliases)
                    if eid:
                        entity_ids[entity.name] = eid
                        context_snippet = _find_context(entity.name, node_contexts)
                        self._insert_entity_mentions(
                            eid, db_doc_id, entity,
                            context_snippet, doc.get('structure'))

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

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(_extract_entities_one, phase1_results))

        # -- Phase 2: batch entity normalization --
        if self.entity_extractor:
            try:
                self.entity_extractor.normalize_entities_batch(self.db)
            except Exception as e:
                logging.warning("Batch entity normalization failed: %s", e)

        # -- Phase 3: search backend + super_tree indexing --
        for doc_id, db_doc_id, doc in phase1_results:
            try:
                if hasattr(self, 'super_tree_index') and self.super_tree_index:
                    self.super_tree_index.on_document_added(
                        db_doc_id, content=_extract_structure_text(doc.get('structure')))
                if self.search_backend and doc.get('structure'):
                    self.search_backend.index_document(
                        db_doc_id, doc['structure'], doc.get('pages'))
            except Exception as e:
                logging.warning("Phase 3 indexing failed for doc %d: %s", db_doc_id, e)

        # -- Phase 4: node profiles (after normalization → canonical names) --
        for _, db_doc_id, doc in phase1_results:
            try:
                if self.db and doc.get('structure'):
                    self._write_node_profiles(db_doc_id, doc['structure'])
            except Exception as e:
                logging.warning("Node profile build failed for doc %d: %s", db_doc_id, e)

        doc_ids = [r[0] for r in phase1_results]
        logging.info("Batch indexing complete: %d documents", len(doc_ids))
        return doc_ids

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

        Unified single chain ([S4]): every query — including a single-document
        corpus — goes through the agentic router. Scale differences are
        internalised by the router's top-k / union cap / grouping parameters;
        a single candidate short-circuits inside the chain at near-zero cost.
        """
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

    def _resolve_node_profiles(self, doc_id: str, mapping: dict) -> dict:
        """Resolve per-node evidence signatures for enhance_and_select ([3.4]).

        Delegates to the shared resolve_node_profiles helper (T6.4): DB
        node_profiles first via the doc's integer id; structure-key fallback;
        else an empty dict (enhance handles missing profiles gracefully, [7.7]).
        """
        from .agentic.enhance import resolve_node_profiles
        db_doc_id = self._id_mapper.to_db(doc_id) if self.db is not None else None
        return resolve_node_profiles(self.db, db_doc_id, mapping)

    async def _search_single(self, query: str, doc_id: str) -> dict:
        """Direct tree search for a single document (zero router overhead).

        [3.4][3.2.1]: the LLM remains the node-selection decision maker via
        enhance_and_select, with keyword/entity signatures injected as
        grounding evidence. Candidates carry each node's text (P2.6 content
        channel: a query token present in the node body admits the node to the
        union even when its stored signature is junk-drowned or missing). No
        len(summary) re-rank and no hardcoded scores: node order preserves the
        LLM's selection order, matched_docs score is the "selection coverage"
        (selected nodes / all candidate nodes), and confidence is "high"
        unless a pool_concern signal survived the optional retry (then
        "medium").
        """
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
                pages_from_nodes, generate_answer, build_context_for_doc,
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

        try:
            from .agentic.enhance import (
                UnifiedNodeEnhancement, resolve_query_entities,
                retry_on_pool_concern,
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

        # unit = 节点：扁平结构全量节点作为候选（[3.2.1]）
        mapping = create_node_mapping(structure)
        candidates = [
            {
                "node_id": nid,
                "title": node.get("title") or "",
                "summary": node.get("summary") or "",
                # 正文内容通道（P2.6）：直接内容接地，存储签名淹没/缺失时保召回
                "text": node.get("text") or "",
            }
            for nid, node in mapping.items()
        ]

        profiles = self._resolve_node_profiles(doc_id, mapping)
        query_entities = (
            resolve_query_entities(self.db, query, limit=5) if self.db else []
        )

        # NFR4: retrieval LLM call site uses retrieve_model with model fallback
        enhancer = UnifiedNodeEnhancement(self.model, retrieve_model=self.retrieve_model)
        result = await enhancer.enhance_and_select(
            query, candidates, profiles, query_entities=query_entities,
        )

        # [3.2.1] pool_concern re-selection (at most once, exclusive branches)
        # via the shared helper: ① deferred pool nonempty → relax union cap and
        # re-select; ② no deferred → force-all full pool (cap widened too, so
        # zero-signal candidates are not bottom-sorted and re-truncated). See
        # retry_on_pool_concern.
        result = await retry_on_pool_concern(
            enhancer, result, query, candidates, profiles,
            query_entities=query_entities,
        )

        selected_ids = result["selected_ids"]
        if not selected_ids:
            return {
                "query": query,
                "mode": "single",
                "answer": "No relevant sections found.",
                "confidence": "low",
                "matched_docs": [],
                "selected_nodes": [],
                "pages": [],
            }

        # Preserve the LLM's selection order — no len(summary) re-rank ([3.4])
        selected = [mapping[nid] for nid in selected_ids if nid in mapping]
        if not selected:
            return {
                "query": query,
                "mode": "single",
                "answer": "No valid sections found.",
                "confidence": "low",
                "matched_docs": [],
                "selected_nodes": [],
                "pages": [],
            }

        pages = pages_from_nodes(selected)

        # Context assembly covers ALL selected nodes ([3.4.1]① multi-span)
        ctx_selected = selected
        if result.get("concern_reason") == "llm_unavailable" and len(ctx_selected) > _MAX_DEGRADE_NODES:
            ctx_selected = ctx_selected[:_MAX_DEGRADE_NODES]
        context = build_context_for_doc(doc, ctx_selected, pages)
        answer = generate_answer(query, context)

        page_map = {p["page"]: p["content"] for p in doc.get("pages", [])}

        # matched_docs score = selection coverage (selected / all candidates),
        # evidence-derived in (0,1]; replaces the legacy hardcoded 1.0.
        score = min(round(len(selected) / max(len(candidates), 1), 4), 1.0)

        # confidence: high = 无 pool_concern（候选池完整，精挑可信）；
        # medium = pool_concern 留存（候选或被截，答案仅供参考）；
        # low = LLM 不可用（降级为 union 放行，[7.7]）
        if result.get("concern_reason") == "llm_unavailable":
            confidence = "low"
        else:
            confidence = "medium" if result["pool_concern"] else "high"

        return {
            "query": query,
            "mode": "single",
            "answer": answer,
            "confidence": confidence,
            "matched_docs": [{"doc_id": doc_id, "score": score}],
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
