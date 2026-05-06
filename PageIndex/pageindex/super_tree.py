import logging
from typing import List, Tuple

try:
    import jieba
except ImportError:
    jieba = None

from .closet_index import _STOPWORDS


class KeywordIndex:
    """jieba-based inverted index for document name and description."""

    def __init__(self, db):
        self.db = db
        if jieba is None:
            logging.warning("jieba not installed; KeywordIndex will be unavailable")

    def _tokenize(self, text: str) -> List[str]:
        if not text or jieba is None:
            return []
        tokens = jieba.lcut(text)
        return [
            t.strip().lower()
            for t in tokens
            if len(t.strip()) > 1 and t.strip().lower() not in _STOPWORDS
        ]

    def add_document(self, doc_id: int, doc_name: str, doc_description: str) -> None:
        records = []
        for token in self._tokenize(doc_name):
            records.append((doc_id, token, "name"))
        for token in self._tokenize(doc_description or ""):
            records.append((doc_id, token, "description"))
        self.db.insert_doc_keywords(doc_id, records)

    def remove_document(self, doc_id: int) -> None:
        self.db.delete_doc_keywords(doc_id)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        tokens = self._tokenize(query)
        if not tokens:
            return []
        return self.db.match_doc_keywords(tokens, top_k)


import json
from .utils import llm_completion


class KBIdentity:
    """Lazy-generated knowledge base identity summary."""

    def __init__(self, db, model: str):
        self.db = db
        self.model = model

    def get_identity(self) -> str:
        cached = self.db.get_kb_identity()
        if cached:
            return cached
        return self._build()

    def invalidate(self) -> None:
        with self.db._connect() as conn:
            conn.execute("DELETE FROM kb_identity WHERE id = 1")

    def _build(self) -> str:
        docs = self.db.get_all_documents()
        if not docs:
            return "知识库中暂无文档。"

        try:
            return self._generate_with_llm(docs)
        except Exception as e:
            logging.warning("KB Identity LLM generation failed: %s", e)
            return self._build_fallback(docs)

    def _generate_with_llm(self, docs) -> str:
        doc_list = []
        for doc in docs:
            top_nodes = self.db.get_top_level_nodes(doc["id"])
            sections = [n.get("title", "") for n in top_nodes if n.get("title")]
            sections_str = "、".join(sections[:5]) if sections else "无章节信息"
            doc_list.append({
                "name": doc.get("pdf_name", ""),
                "sections": sections_str,
            })

        prompt = f"""你是一个知识库管理员。给定以下文档列表，请生成一段简短的摘要（不超过200字），描述知识库的整体内容和主要主题。

文档列表：
{json.dumps(doc_list, ensure_ascii=False, indent=2)}

要求：
1. 说明文档总数
2. 概括主要主题领域
3. 不要列出每个文档的详细内容

直接返回纯文本摘要，不要输出 JSON 或其他格式。"""

        response = llm_completion(self.model, prompt)
        if response:
            self.db.set_kb_identity(response.strip(), len(docs))
            return response.strip()
        raise RuntimeError("LLM returned empty response")

    def _build_fallback(self, docs) -> str:
        names = [doc.get("pdf_name", "") for doc in docs]
        text = f"知识库共 {len(docs)} 个文档：" + "、".join(names[:10])
        if len(names) > 10:
            text += " 等"
        self.db.set_kb_identity(text, len(docs))
        return text


import asyncio
from typing import Set, Dict

from .utils import llm_acompletion, count_tokens, extract_json


class SuperTreeIndex:
    """L0 dual-channel prefilter + L1 Super-Tree document selection."""

    _MAX_TOP_NODES_PER_DOC = 8
    _MAX_CANDIDATE_DOCS = 50
    _MAX_SUPER_TREE_TOKENS = 6000
    _SUMMARY_MAX_LEN = 100

    def __init__(self, db, model: str, client):
        self.db = db
        self.model = model
        self.client = client
        self.keyword_index = KeywordIndex(db)
        self.kb_identity = KBIdentity(db, model)
        self._backfill_existing_docs()

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def on_document_added(self, db_doc_id: int) -> None:
        doc = self.db.get_document_by_id(db_doc_id)
        if doc:
            self.keyword_index.add_document(
                db_doc_id, doc.get("pdf_name", ""), doc.get("doc_description", "")
            )
        self.kb_identity.invalidate()

    def on_document_removed(self, db_doc_id: int) -> None:
        self.keyword_index.remove_document(db_doc_id)
        self.kb_identity.invalidate()

    # ------------------------------------------------------------------
    # L0: Dual-channel prefilter
    # ------------------------------------------------------------------
    def prefilter(self, query: str) -> Dict[int, float]:
        """Return candidate doc_ids with cumulative channel scores."""
        scores: Dict[int, float] = {}

        # Channel A: tag matching (ClosetIndex)
        if hasattr(self.client, "closet_index") and self.client.closet_index:
            try:
                tag_results = self.client.closet_index.search(query, top_k=20)
                for doc_id, score in tag_results:
                    scores[int(doc_id)] = scores.get(int(doc_id), 0.0) + float(score)
            except Exception as e:
                logging.warning("Tag matching failed: %s", e)

        # Channel B: keyword inverted index
        try:
            keyword_results = self.keyword_index.search(query, top_k=20)
            for doc_id, score in keyword_results:
                scores[int(doc_id)] = scores.get(int(doc_id), 0.0) + float(score)
        except Exception as e:
            logging.warning("Keyword search failed: %s", e)

        return scores

    def _truncate_candidates(self, scores: Dict[int, float]) -> list[int]:
        if len(scores) <= self._MAX_CANDIDATE_DOCS:
            return list(scores.keys())

        sorted_docs = sorted(
            scores.keys(), key=lambda d: scores.get(d, 0.0), reverse=True
        )
        return sorted_docs[:self._MAX_CANDIDATE_DOCS]

    # ------------------------------------------------------------------
    # L1: Super-Tree document selection
    # ------------------------------------------------------------------
    async def select_documents(self, query: str, candidate_db_ids: Dict[int, float]) -> list[str]:
        if not candidate_db_ids:
            return []

        truncated = self._truncate_candidates(candidate_db_ids)
        super_tree = self._build_super_tree(truncated)
        kb_identity = self.kb_identity.get_identity()

        # Token budget check
        tree_json = json.dumps(super_tree, ensure_ascii=False)
        total_tokens = count_tokens(tree_json) + count_tokens(kb_identity) + count_tokens(query)
        if total_tokens > self._MAX_SUPER_TREE_TOKENS:
            logging.warning("Super-Tree exceeds token budget (%d), truncating docs", total_tokens)
            while total_tokens > self._MAX_SUPER_TREE_TOKENS and len(super_tree["documents"]) > 5:
                super_tree["documents"].pop()
                tree_json = json.dumps(super_tree, ensure_ascii=False)
                total_tokens = count_tokens(tree_json) + count_tokens(kb_identity) + count_tokens(query)

        prompt = f"""你是一个文档检索专家。给定以下用户问题、知识库概览和候选文档结构，请选出最可能包含答案的 3-5 个文档。

[知识库概览]
{kb_identity}

[用户问题]
{query}

[候选文档结构]
{tree_json}

要求：
1. 基于文档的顶层章节标题和摘要判断相关性
2. 优先选择直接相关文档，次选间接相关文档
3. 如果知识库概览显示没有相关主题，返回空列表

返回JSON格式：
{{
  "thinking": "推理过程...",
  "doc_ids": ["uuid-1", "uuid-2", ...]
}}
直接返回最终JSON结构，不要输出其他内容。"""

        response = await llm_acompletion(self.model, prompt)
        if not response:
            return []

        # Parse JSON response (handles markdown code blocks, etc.)
        data = extract_json(response)
        if not isinstance(data, dict):
            data = {}
        doc_ids = data.get("doc_ids", [])

        # Map back to UUIDs
        uuid_results = []
        db_to_uuid = {v: k for k, v in getattr(self.client, "_uuid_to_db", {}).items()}
        for doc_id in doc_ids:
            if isinstance(doc_id, int):
                uuid = db_to_uuid.get(doc_id)
                if uuid:
                    uuid_results.append(uuid)
            elif isinstance(doc_id, str):
                uuid_results.append(doc_id)
        return uuid_results

    # ------------------------------------------------------------------
    # Super-Tree builder
    # ------------------------------------------------------------------
    def _build_super_tree(self, doc_ids: list[int]) -> Dict:
        documents = []
        db_to_uuid = {v: k for k, v in getattr(self.client, "_uuid_to_db", {}).items()}

        for db_doc_id in doc_ids:
            doc = next((d for d in self.db.get_all_documents() if d["id"] == db_doc_id), None)
            if not doc:
                continue

            uuid = db_to_uuid.get(db_doc_id, str(db_doc_id))
            top_nodes = self.db.get_top_level_nodes(db_doc_id)

            # Sort by child count (more children = more structurally important)
            nodes_with_children = []
            with self.db._connect() as conn:
                for node in top_nodes:
                    children = conn.execute(
                        "SELECT COUNT(*) FROM nodes WHERE parent_node_id = ?",
                        (node.get("node_id"),),
                    ).fetchone()[0]
                    nodes_with_children.append((node, children))

            nodes_with_children.sort(key=lambda x: x[1], reverse=True)
            selected = nodes_with_children[:self._MAX_TOP_NODES_PER_DOC]

            top_nodes_out = []
            for node, _ in selected:
                summary = node.get("summary", "") or ""
                if len(summary) > self._SUMMARY_MAX_LEN:
                    summary = summary[:self._SUMMARY_MAX_LEN] + "..."
                top_nodes_out.append({
                    "title": node.get("title", ""),
                    "summary": summary,
                })

            documents.append({
                "doc_id": uuid,
                "db_id": db_doc_id,
                "doc_name": doc.get("pdf_name", ""),
                "description": doc.get("doc_description", ""),
                "top_nodes": top_nodes_out,
            })

        return {"documents": documents}

    # ------------------------------------------------------------------
    # Backfill existing documents on first init
    # ------------------------------------------------------------------
    def _backfill_existing_docs(self) -> None:
        try:
            with self.db._connect() as conn:
                row = conn.execute("SELECT COUNT(*) FROM doc_keywords").fetchone()
                if row and row[0] > 0:
                    return
        except Exception:
            return

        for doc in self.db.get_all_documents():
            try:
                self.keyword_index.add_document(
                    doc["id"],
                    doc.get("pdf_name", ""),
                    doc.get("doc_description", ""),
                )
            except Exception as e:
                logging.warning("Backfill failed for doc %s: %s", doc.get("pdf_name"), e)
