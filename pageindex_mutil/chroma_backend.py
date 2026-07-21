"""ChromaDB vector search backend.

Provides vector-based document search using ChromaDB for semantic retrieval.
Uses ONNX embeddings by default for zero-dependency local operation.
"""

import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .search_backend import SearchBackend

logger = logging.getLogger(__name__)


class ChromaSearchBackend(SearchBackend):
    """ChromaDB-based vector search backend."""

    def __init__(self, db_path: str = "./data/vectors", embedding_model: str = "local"):
        """Initialize ChromaDB backend.
        
        Args:
            db_path: Path to store ChromaDB data
            embedding_model: Embedding model type ("local" for ONNX, "ollama", "openai")
        """
        if chromadb is None:
            raise ImportError(
                "chromadb is required for vector search. "
                "Install with: pip install chromadb"
            )
        
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="pageindex_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding model
        self.embedding_model = embedding_model
        self._embedder = None
        
        logger.info("ChromaDB backend initialized at %s", self.db_path)

    def _get_embedder(self):
        """Lazy-load embedding model."""
        if self._embedder is None:
            if self.embedding_model == "local":
                if SentenceTransformer is None:
                    raise ImportError(
                        "sentence-transformers is required for local embeddings. "
                        "Install with: pip install sentence-transformers"
                    )
                # Use a lightweight multilingual model
                self._embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            elif self.embedding_model == "ollama":
                # TODO: Implement Ollama embedding support
                raise NotImplementedError("Ollama embedding not yet implemented")
            elif self.embedding_model == "openai":
                # TODO: Implement OpenAI embedding support
                raise NotImplementedError("OpenAI embedding not yet implemented")
            else:
                raise ValueError(f"Unknown embedding model: {self.embedding_model}")
        return self._embedder

    def _embed_text(self, text: str) -> List[float]:
        """Convert text to embedding vector."""
        embedder = self._get_embedder()
        embedding = embedder.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Convert batch of texts to embedding vectors."""
        embedder = self._get_embedder()
        embeddings = embedder.encode(texts, normalize_embeddings=True, batch_size=32)
        return embeddings.tolist()

    def index_document(self, doc_id: int, nodes: List[Dict], pages: List[Dict] = None) -> None:
        """Index a document's content for vector search.
        
        Creates one vector per node, combining title + summary + text.
        """
        if not nodes:
            return

        # Prepare texts for embedding
        ids = []
        texts = []
        metadatas = []
        
        for node in nodes:
            node_id = node.get("node_id", "")
            title = node.get("title", "")
            summary = node.get("summary", "")
            text = node.get("text", "")
            
            # Combine title + summary + text for embedding
            combined_text = f"{title}\n{summary}\n{text}".strip()
            if not combined_text:
                continue
            
            # Create unique ID
            chroma_id = f"doc_{doc_id}_node_{node_id}"
            ids.append(chroma_id)
            texts.append(combined_text)
            metadatas.append({
                "doc_id": doc_id,
                "node_id": node_id,
                "title": title,
                "has_text": bool(text)
            })
        
        if not ids:
            return

        # Generate embeddings
        embeddings = self._embed_batch(texts)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        logger.info("Indexed %d nodes for document %d", len(ids), doc_id)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search for documents using vector similarity.
        
        Returns:
            List of (doc_id, score) tuples sorted by similarity
        """
        if not query or not query.strip():
            return []

        # Embed query
        query_embedding = self._embed_text(query)
        
        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,  # Get more to account for multiple nodes per doc
            include=["metadatas", "distances"]
        )
        
        if not results or not results["metadatas"]:
            return []

        # Aggregate scores by document using weighted max + avg strategy
        doc_similarities: Dict[int, List[float]] = {}
        for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
            doc_id = metadata["doc_id"]
            similarity = 1.0 - distance
            if doc_id not in doc_similarities:
                doc_similarities[doc_id] = []
            doc_similarities[doc_id].append(similarity)

        doc_scores: Dict[int, float] = {}
        for doc_id, sims in doc_similarities.items():
            sims_sorted = sorted(sims, reverse=True)
            max_sim = sims_sorted[0]
            # Weighted combination: 60% max similarity + 40% average of top-3
            top_k_sims = sims_sorted[:3]
            avg_top_k = sum(top_k_sims) / len(top_k_sims)
            doc_scores[doc_id] = max_sim * 0.6 + avg_top_k * 0.4

        # Sort by score and return top_k
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def remove_document(self, doc_id: int) -> None:
        """Remove all nodes for a document from the index."""
        try:
            # Get all IDs for this document
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"]
            )
            
            if results and results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info("Removed %d nodes for document %d", len(results["ids"]), doc_id)
        except Exception as e:
            logger.warning("Failed to remove document %d: %s", doc_id, e)

    def clear(self) -> None:
        """Clear all indexed data."""
        try:
            self.client.delete_collection("pageindex_documents")
            self.collection = self.client.get_or_create_collection(
                name="pageindex_documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Cleared ChromaDB collection")
        except Exception as e:
            logger.warning("Failed to clear ChromaDB: %s", e)
