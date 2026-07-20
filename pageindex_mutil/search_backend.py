"""Search backend abstraction layer.

Provides a unified interface for different search backends:
- SQLiteBackend: Current default (jieba inverted index + ClosetIndex tags)
- ChromaBackend: ChromaDB vector search (required)
- HybridBackend: Combined vector + keyword search (default)
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional


class SearchBackend(ABC):
    """Abstract base class for search backends."""

    @abstractmethod
    def index_document(self, doc_id: int, nodes: List[Dict], pages: List[Dict] = None) -> None:
        """Index a document's content for searching.
        
        Args:
            doc_id: Database document ID
            nodes: List of node dicts with node_id, title, summary, text
            pages: Optional list of page dicts with page_number, content
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search for documents matching the query.
        
        Args:
            query: Search query text
            top_k: Maximum number of results
            
        Returns:
            List of (doc_id, score) tuples sorted by relevance
        """
        pass

    @abstractmethod
    def remove_document(self, doc_id: int) -> None:
        """Remove a document from the index.
        
        Args:
            doc_id: Database document ID to remove
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all indexed data."""
        pass
