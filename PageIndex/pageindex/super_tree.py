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
