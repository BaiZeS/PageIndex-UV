"""Hybrid search backend combining vector and keyword search.

Merges results from ChromaDB vector search and SQLite keyword/tag search
using Reciprocal Rank Fusion (RRF) for improved retrieval accuracy.
"""

import logging
from typing import List, Tuple, Dict, Optional

from .search_backend import SearchBackend

logger = logging.getLogger(__name__)


class HybridSearchBackend(SearchBackend):
    """Hybrid search combining vector similarity and keyword matching."""

    def __init__(self, db, chroma_backend: SearchBackend, rrf_k: int = 60):
        """Initialize hybrid backend.
        
        Args:
            db: PageIndexDB instance for keyword/tag search
            chroma_backend: ChromaSearchBackend instance for vector search
            rrf_k: RRF constant for rank fusion (higher = less steep)
        """
        self.db = db
        self.chroma = chroma_backend
        self.rrf_k = rrf_k
        
        # Import keyword search dependencies
        try:
            import jieba
            self.jieba = jieba
        except ImportError:
            self.jieba = None
            logger.warning("jieba not installed; keyword search will be limited")

    def _keyword_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Perform keyword-based search using jieba and SQLite."""
        if self.jieba is None:
            return []

        # Tokenize query
        from .closet_index import _STOPWORDS
        tokens = [
            t.strip().lower()
            for t in self.jieba.lcut(query)
            if len(t.strip()) > 1 and t.strip().lower() not in _STOPWORDS
        ]
        
        if not tokens:
            return []

        # Search doc_keywords table
        try:
            keyword_results = self.db.match_doc_keywords(tokens, top_k)
            return [(doc_id, float(score)) for doc_id, score in keyword_results]
        except Exception as e:
            logger.warning("Keyword search failed: %s", e)
            return []

    def _tag_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Perform semantic tag search using ClosetIndex."""
        if self.jieba is None:
            return []

        from .closet_index import _STOPWORDS
        tokens = [
            t.strip().lower()
            for t in self.jieba.lcut(query)
            if len(t.strip()) > 1 and t.strip().lower() not in _STOPWORDS
        ]
        
        if not tokens:
            return []

        try:
            tag_results = self.db.match_closet_tags(tokens, top_k)
            return [(doc_id, float(score)) for doc_id, score in tag_results]
        except Exception as e:
            logger.warning("Tag search failed: %s", e)
            return []

    def _rrf_fusion(
        self,
        result_sets: List[List[Tuple[int, float]]],
        weights: List[float] = None
    ) -> List[Tuple[int, float]]:
        """Fuse multiple result sets using Reciprocal Rank Fusion.
        
        Args:
            result_sets: List of result lists, each with (doc_id, score)
            weights: Optional weights for each result set
            
        Returns:
            Fused and sorted results
        """
        if weights is None:
            weights = [1.0] * len(result_sets)
        
        # Convert scores to ranks
        ranked_sets = []
        for results in result_sets:
            # Sort by score descending, assign ranks
            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
            ranked = [(doc_id, rank + 1) for rank, (doc_id, _) in enumerate(sorted_results)]
            ranked_sets.append(ranked)
        
        # Apply RRF
        scores: Dict[int, float] = {}
        for ranked_results, weight in zip(ranked_sets, weights):
            for doc_id, rank in ranked_results:
                rrf_score = weight * (1.0 / (self.rrf_k + rank))
                scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score
        
        # Sort by fused score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def index_document(self, doc_id: int, nodes: List[Dict], pages: List[Dict] = None) -> None:
        """Index a document for both vector and keyword search."""
        # Index in ChromaDB
        self.chroma.index_document(doc_id, nodes, pages)
        
        # Keyword indexing is handled by existing ClosetIndex/SuperTreeIndex
        # This method is primarily for ChromaDB indexing

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search using both vector similarity and keyword matching.
        
        Combines results using RRF for improved accuracy.
        """
        if not query or not query.strip():
            return []

        # Run all search channels in parallel conceptually
        # (they're fast enough to run sequentially)
        vector_results = self.chroma.search(query, top_k * 2)
        keyword_results = self._keyword_search(query, top_k * 2)
        tag_results = self._tag_search(query, top_k * 2)

        # Combine using RRF with weights
        # Vector search gets higher weight for semantic understanding
        # Keyword/tag search gets weight for exact matching
        result_sets = [vector_results, keyword_results, tag_results]
        weights = [1.5, 1.0, 1.0]  # Vector gets 1.5x weight
        
        fused_results = self._rrf_fusion(result_sets, weights)
        
        # Return top_k results
        return fused_results[:top_k]

    def remove_document(self, doc_id: int) -> None:
        """Remove a document from both vector and keyword indices."""
        self.chroma.remove_document(doc_id)
        # Keyword/tag cleanup is handled by existing code

    def clear(self) -> None:
        """Clear all indexed data."""
        self.chroma.clear()
