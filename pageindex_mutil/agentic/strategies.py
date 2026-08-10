import json
import logging
from typing import List, Tuple, Dict

import jieba

from ..utils import llm_completion, extract_json
from ..closet_index import ClosetIndex, _STOPWORDS


class MetadataStrategy:
    def search(self, query: str, docs_info: List[Dict]) -> List[Tuple[str, int]]:
        try:
            tokens = jieba.lcut(query)
        except Exception:
            return []
        keywords = [
            t.strip().lower()
            for t in tokens
            if len(t.strip()) > 1 and t.strip().lower() not in _STOPWORDS
        ]
        if not keywords:
            return []

        scored = []
        for doc in docs_info:
            doc_name = (doc.get("doc_name") or "").lower()
            description = (doc.get("description") or "").lower()
            score = 0
            for kw in keywords:
                if kw in doc_name:
                    score += 2
                if kw in description:
                    score += 1
            if score > 0:
                scored.append((str(doc.get("doc_id", "")), score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [(doc_id, rank + 1) for rank, (doc_id, _) in enumerate(scored)]


class SemanticsStrategy:
    def __init__(self, closet_index: ClosetIndex):
        self.closet_index = closet_index

    def search(self, query: str, docs_info: List[Dict]) -> List[Tuple[str, int]]:
        results = self.closet_index.search(query, top_k=10)
        return [(str(doc_id), rank + 1) for rank, (doc_id, _) in enumerate(results)]


class ContentStrategy:
    """Search based on document content keywords (not just metadata).

    Returns node-level matching information for downstream LLM enhancement.
    """

    def __init__(self, client):
        self.client = client

    def _extract_keyword_context(self, text: str, keyword: str, window: int = 50) -> str:
        """Extract keyword and its surrounding context."""
        idx = text.lower().find(keyword.lower())
        if idx == -1:
            return None
        start = max(0, idx - window)
        end = min(len(text), idx + len(keyword) + window)
        context = text[start:end]
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
        return context

    def search(self, query: str, docs_info: List[Dict]) -> List[Tuple[str, int, List[Dict]]]:
        """Return (doc_id, score, matched_nodes_info).

        matched_nodes_info: [{"node_id": str, "keyword": str, "context": str}]
        """
        try:
            tokens = jieba.lcut(query)
        except Exception:
            return []
        keywords = [
            t.strip().lower()
            for t in tokens
            if len(t.strip()) > 1 and t.strip().lower() not in _STOPWORDS
        ]
        if not keywords:
            return []

        scored = []
        for doc_info in docs_info:
            doc_id = str(doc_info.get("doc_id", ""))
            doc = self.client.documents.get(doc_id) if self.client else None
            if not doc:
                continue

            matched_nodes = []
            seen_nodes = set()

            # Search in structure text (both MD and PDF)
            for node in doc.get("structure", []):
                text = (node.get("text") or "")
                text_lower = text.lower()
                node_id = node.get("node_id")
                if not node_id or node_id in seen_nodes:
                    continue

                for kw in keywords:
                    if kw in text_lower:
                        context = self._extract_keyword_context(text, kw)
                        if context:
                            matched_nodes.append({
                                "node_id": node_id,
                                "keyword": kw,
                                "context": context
                            })
                            seen_nodes.add(node_id)
                            break  # One match per node is enough

            # Also search in pages (PDF)
            if not matched_nodes:
                for page in doc.get("pages", []):
                    page_text = (page.get("content") or "").lower()
                    for kw in keywords:
                        if kw in page_text:
                            context = self._extract_keyword_context(page.get("content", ""), kw)
                            if context:
                                matched_nodes.append({
                                    "node_id": f"page_{page.get('page', 0)}",
                                    "keyword": kw,
                                    "context": context
                                })
                                break

            if matched_nodes:
                scored.append((doc_id, len(matched_nodes), matched_nodes))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


class DescriptionStrategy:
    def __init__(self, model: str, retrieve_model: str = None):
        self.model = model
        self.retrieve_model = retrieve_model
        self._main_get_relevant = None
        # Attempt one-time import of reasoning helper
        try:
            from ..reasoning import get_relevant_documents_for_multidoc
            self._main_get_relevant = get_relevant_documents_for_multidoc
        except Exception:
            pass

    def search(self, query: str, docs_info: List[Dict]) -> List[Tuple[str, int]]:
        if not docs_info:
            return []

        # Primary: reuse main.py helper (spec FR4)
        if self._main_get_relevant is not None:
            try:
                doc_ids = self._main_get_relevant(query, docs_info)
                if isinstance(doc_ids, list):
                    return [(str(doc_id), rank + 1) for rank, doc_id in enumerate(doc_ids)]
            except Exception as e:
                logging.warning("Description strategy (main.py) failed: %s", e)

        # Fallback: built-in implementation
        prompt = f"""你是一个文档相关性判断专家。给定用户问题和文档列表，选出最可能包含答案的文档。

用户问题: {query}

文档列表:
{json.dumps(docs_info, indent=2, ensure_ascii=False)}

请返回JSON格式: {{"doc_ids": ["doc_id_1", "doc_id_2", ...]}}
最多返回5个最相关的文档。直接返回JSON，不要其他内容。
"""
        try:
            response = llm_completion(self.retrieve_model or self.model, prompt, thinking_disabled=False)
            if not response:
                return []
            data = extract_json(response)
            if isinstance(data, dict):
                doc_ids = data.get("doc_ids", [])
            elif isinstance(data, list):
                doc_ids = data
            else:
                return []
            return [(str(doc_id), rank + 1) for rank, doc_id in enumerate(doc_ids)]
        except Exception as e:
            logging.warning("Description strategy failed: %s", e)
            return []
