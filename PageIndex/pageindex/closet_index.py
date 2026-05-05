from dataclasses import dataclass
import logging
from typing import List, Tuple

try:
    import jieba
except ImportError:
    jieba = None

from .utils import llm_completion, extract_json


# Simple stopword set for Chinese/English mixed queries.
_STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "看", "好", "自己", "这", "那", "怎么", "什么", "如何", "吗", "呢",
    "吧", "啊", "哦", "嗯", "没有",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "and", "but", "or", "yet", "so", "if", "because", "although", "though",
    "while", "where", "when", "that", "which", "who", "whom", "whose",
    "what", "how", "why", "this", "these", "those", "it", "its", "they",
    "them", "their", "we", "us", "our", "you", "your", "he", "him", "his",
    "she", "her", "i", "me", "my",
}


@dataclass
class Tag:
    text: str
    confidence: float


class ClosetIndex:
    _MIN_TAG_CONFIDENCE = 0.5
    _MIN_TOKEN_LENGTH = 1

    def __init__(self, db, model: str):
        self.db = db
        self.model = model
        if jieba is None:
            logging.warning("jieba not installed; ClosetIndex search will be unavailable")
        if hasattr(db, "ensure_closet_schema"):
            db.ensure_closet_schema()

    def _extract_tags(
        self, doc_name: str, doc_description: str, node_titles: List[str]
    ) -> List[Tag]:
        titles_text = "\n".join(f"- {t}" for t in node_titles if t)
        prompt = f"""你是一个文档语义标签提取专家。给定以下文档信息，请提取3-5个最能代表此文档核心内容的语义概念标签。

文档名: {doc_name or "未知"}
文档描述: {doc_description or "无"}
文档章节列表:
{titles_text}

要求:
1. 标签应为抽象语义概念，而非文档中的原词
   - 例如：用"容器编排"而非"Kubernetes"，用"微服务治理"而非"Istio"
2. 覆盖不同语义维度（如领域、场景、方法论、架构模式）
3. 每个标签附带置信度(0.0-1.0)，仅返回高于0.5的标签
4. 使用中文输出标签

返回JSON格式: [{{"tag": "概念名", "confidence": 0.95}}, ...]
直接返回最终JSON结构，不要输出其他内容。
"""
        try:
            response = llm_completion(self.model, prompt)
            if not response:
                return []
            data = extract_json(response)
            if not isinstance(data, list):
                return []
            tags = []
            for item in data:
                if isinstance(item, dict):
                    text = item.get("tag", "")
                    conf = float(item.get("confidence", 0))
                    if text and conf >= self._MIN_TAG_CONFIDENCE:
                        tags.append(Tag(text=text.strip(), confidence=conf))
            return tags
        except Exception as e:
            logging.warning(f"Tag extraction failed for {doc_name}: {e}")
            return []

    def _fallback_tags(self, doc_name: str, doc_description: str) -> List[Tag]:
        text = f"{doc_name or ''} {doc_description or ''}"
        text = text.strip()
        if not text:
            return []
        if jieba is None:
            tokens = text.split()
        else:
            tokens = jieba.lcut(text)
        seen = set()
        tags = []
        for t in tokens:
            t = t.strip().lower()
            if len(t) <= self._MIN_TOKEN_LENGTH or t in _STOPWORDS:
                continue
            if t not in seen:
                seen.add(t)
                tags.append(Tag(text=t, confidence=0.3))
        return tags

    def _tokenize_tag(self, tag_text: str) -> str:
        if jieba is None:
            return tag_text
        tokens = jieba.lcut(tag_text)
        filtered = [t.strip() for t in tokens if len(t.strip()) > 1]
        return " ".join(filtered) if filtered else tag_text

    def add_document(
        self, doc_id, doc_name: str, doc_description: str, nodes: List[dict]
    ) -> None:
        node_titles = [n.get("title", "") for n in nodes if n.get("title")]
        tags = self._extract_tags(doc_name, doc_description, node_titles)
        if not tags:
            tags = self._fallback_tags(doc_name, doc_description)
        if not tags:
            return

        records = []
        for tag in tags:
            tag_token = self._tokenize_tag(tag.text)
            records.append((doc_id, tag.text, tag_token, tag.confidence, "llm"))

        self.db.insert_closet_tags(doc_id, records)

    def remove_document(self, doc_id) -> None:
        self.db.delete_closet_tags(doc_id)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if jieba is None:
            logging.warning("jieba not installed; skipping semantics strategy")
            return []
        tokens = jieba.lcut(query)
        filtered = [
            t.strip()
            for t in tokens
            if len(t.strip()) > 1 and t.strip().lower() not in _STOPWORDS
        ]
        if not filtered:
            return []
        return self.db.match_closet_tags(filtered, top_k)

    def rebuild(self) -> None:
        with self.db._connect() as conn:
            conn.execute("DELETE FROM closet_tags")
