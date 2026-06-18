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


class DescriptionStrategy:
    def __init__(self, model: str):
        self.model = model
        self._main_get_relevant = None
        # Attempt one-time import of main.py helper to avoid runtime side effects
        try:
            import main
            self._main_get_relevant = main.get_relevant_documents_for_multidoc
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
            response = llm_completion(self.model, prompt)
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
