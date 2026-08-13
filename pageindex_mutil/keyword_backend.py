"""Keyword-based search backend.

Provides fast document search using jieba tokenization + SQLite inverted
index (doc_keywords) and ClosetIndex tags.  No embedding model required.
"""

import hashlib
import logging
import time
from typing import List, Tuple, Dict

from .search_backend import SearchBackend

try:
    import jieba
except ImportError:
    jieba = None

from .closet_index import _STOPWORDS

logger = logging.getLogger(__name__)


class KeywordSearchBackend(SearchBackend):
    """Search backend using jieba keywords + ClosetIndex tags only.

    Zero model-loading overhead — suitable for CPU-only deployments where
    ChromaDB / SentenceTransformer is too slow.
    """

    is_vector = False

    _search_cache = {}  # hash(query+top_k) -> (result, timestamp)
    _cache_ttl = 300  # 5 minutes
    _cache_max = 128

    def __init__(self, db, keyword_weight: float = 1.0, tag_weight: float = 1.0):
        """
        Args:
            db: PageIndexDB instance
            keyword_weight: weight for doc_keywords channel
            tag_weight: weight for closet_tags channel
        """
        self.db = db
        self.keyword_weight = keyword_weight
        self.tag_weight = tag_weight
        if jieba is None:
            logger.warning("jieba not installed; KeywordSearchBackend will be unavailable")

    def _cache_key(self, query: str, top_k: int) -> str:
        return hashlib.md5(f"{query}:{top_k}".encode()).hexdigest()

    def _cache_get(self, key: str):
        entry = self._search_cache.get(key)
        if entry and (time.monotonic() - entry[1]) < self._cache_ttl:
            return entry[0]
        if entry:
            del self._search_cache[key]
        return None

    def _cache_set(self, key: str, value):
        if len(self._search_cache) >= self._cache_max:
            oldest_key = min(self._search_cache, key=lambda k: self._search_cache[k][1])
            del self._search_cache[oldest_key]
        self._search_cache[key] = (value, time.monotonic())

    def _tokenize(self, query: str) -> List[str]:
        if not query or jieba is None:
            return []
        return [
            t.strip().lower()
            for t in jieba.lcut(query)
            if len(t.strip()) > 1 and t.strip().lower() not in _STOPWORDS
        ]

    def index_document(self, doc_id: int, nodes: List[Dict], pages: List[Dict] = None) -> None:
        """No-op: keyword/tag indexing is handled by ClosetIndex and SuperTreeIndex."""
        pass

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search using keyword + tag channels, fused by weighted sum."""
        # Check cache first
        cache_key = self._cache_key(query, top_k)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores: Dict[int, float] = {}

        # Channel A: doc_keywords
        try:
            for doc_id, score in self.db.match_doc_keywords(tokens, top_k):
                scores[int(doc_id)] = scores.get(int(doc_id), 0.0) + float(score) * self.keyword_weight
        except Exception as e:
            logger.warning("Keyword search failed: %s", e)

        # Channel B: closet_tags
        try:
            for doc_id, score in self.db.match_closet_tags(tokens, top_k):
                scores[int(doc_id)] = scores.get(int(doc_id), 0.0) + float(score) * self.tag_weight
        except Exception as e:
            logger.warning("Tag search failed: %s", e)

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Cache result
        self._cache_set(cache_key, sorted_results)
        return sorted_results

    def remove_document(self, doc_id: int) -> None:
        """No-op: cleanup handled by existing code."""
        pass

    def clear(self) -> None:
        """No-op: cleanup handled by existing code."""
        pass
