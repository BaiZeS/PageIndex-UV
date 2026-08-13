"""Tests for search backends (Keyword, ChromaDB, and Hybrid)."""

import pytest
import tempfile
import shutil
from pathlib import Path

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    _HAS_VECTOR_DEPS = True
except ImportError:
    _HAS_VECTOR_DEPS = False


# ---------------------------------------------------------------------------
# KeywordSearchBackend tests (always available)
# ---------------------------------------------------------------------------

class TestKeywordSearchBackend:
    """Tests for KeywordSearchBackend."""

    def test_import(self):
        from pageindex_mutil.keyword_backend import KeywordSearchBackend
        assert KeywordSearchBackend is not None

    def test_search(self):
        from pageindex_mutil.keyword_backend import KeywordSearchBackend

        class MockDB:
            def match_doc_keywords(self, tokens, top_k):
                return [(1, 2.0), (2, 1.0)]
            def match_closet_tags(self, tokens, top_k):
                return [(1, 1.5), (3, 1.0)]

        backend = KeywordSearchBackend(db=MockDB())
        results = backend.search("Python programming", top_k=5)
        assert len(results) > 0
        # Doc 1 should rank highest (appears in both channels)
        assert results[0][0] == 1

    def test_empty_query(self):
        from pageindex_mutil.keyword_backend import KeywordSearchBackend

        class MockDB:
            pass

        backend = KeywordSearchBackend(db=MockDB())
        assert backend.search("") == []
        assert backend.search("   ") == []

    def test_index_document_noop(self):
        """index_document is a no-op for keyword backend."""
        from pageindex_mutil.keyword_backend import KeywordSearchBackend

        class MockDB:
            pass

        backend = KeywordSearchBackend(db=MockDB())
        # Should not raise
        backend.index_document(1, [{"node_id": "1", "title": "test"}])


# ---------------------------------------------------------------------------
# ChromaSearchBackend tests (require chromadb + sentence-transformers)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_VECTOR_DEPS, reason="chromadb/sentence-transformers not installed")
class TestChromaSearchBackend:
    """Tests for ChromaSearchBackend."""

    def setup_method(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.test_dir) / "vectors")

    def teardown_method(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_import(self):
        from pageindex_mutil.chroma_backend import ChromaSearchBackend
        assert ChromaSearchBackend is not None

    def test_initialization(self):
        from pageindex_mutil.chroma_backend import ChromaSearchBackend
        backend = ChromaSearchBackend(db_path=self.db_path)
        assert backend.db_path.exists()
        assert backend.collection is not None

    def test_index_document(self):
        from pageindex_mutil.chroma_backend import ChromaSearchBackend
        backend = ChromaSearchBackend(db_path=self.db_path)

        nodes = [
            {"node_id": "0001", "title": "Introduction", "summary": "This is an introduction section.", "text": "Welcome to the document."},
            {"node_id": "0002", "title": "Methods", "summary": "This section describes methods.", "text": "We used the following methods..."},
        ]
        backend.index_document(doc_id=1, nodes=nodes)
        assert backend.collection.count() == 2

    def test_search(self):
        from pageindex_mutil.chroma_backend import ChromaSearchBackend
        backend = ChromaSearchBackend(db_path=self.db_path)

        nodes = [
            {"node_id": "0001", "title": "Python Programming", "summary": "Learn Python basics.", "text": "Python is a programming language."},
            {"node_id": "0002", "title": "Java Programming", "summary": "Learn Java basics.", "text": "Java is a programming language."},
        ]
        backend.index_document(doc_id=1, nodes=nodes)
        results = backend.search("Python", top_k=1)
        assert len(results) == 1
        assert results[0][0] == 1
        assert results[0][1] > 0

    def test_remove_document(self):
        from pageindex_mutil.chroma_backend import ChromaSearchBackend
        backend = ChromaSearchBackend(db_path=self.db_path)

        nodes = [{"node_id": "0001", "title": "Test", "summary": "Test content", "text": "Test text"}]
        backend.index_document(doc_id=1, nodes=nodes)
        assert backend.collection.count() == 1
        backend.remove_document(doc_id=1)
        assert backend.collection.count() == 0


# ---------------------------------------------------------------------------
# HybridSearchBackend tests (require chromadb + sentence-transformers)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_VECTOR_DEPS, reason="chromadb/sentence-transformers not installed")
class TestHybridSearchBackend:
    """Tests for HybridSearchBackend."""

    def setup_method(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.test_dir) / "vectors")

    def teardown_method(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_import(self):
        from pageindex_mutil.hybrid_backend import HybridSearchBackend
        assert HybridSearchBackend is not None

    def test_rrf_fusion(self):
        from pageindex_mutil.hybrid_backend import HybridSearchBackend

        class MockDB:
            def match_doc_keywords(self, tokens, top_k):
                return [(1, 2.0), (2, 1.0)]
            def match_closet_tags(self, tokens, top_k):
                return [(1, 1.5), (3, 1.0)]

        backend = HybridSearchBackend(db=MockDB(), chroma_backend=None)
        result_sets = [
            [(1, 0.9), (2, 0.8)],
            [(1, 0.7), (3, 0.6)],
        ]
        weights = [1.0, 1.0]
        fused = backend._rrf_fusion(result_sets, weights)
        assert len(fused) == 3
        assert fused[0][0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# BM25 keyword channel tests (P2-Fix1)
# ---------------------------------------------------------------------------

class TestBM25KeywordChannel:
    """Tests for BM25 scoring in match_doc_keywords (P2-Fix1)."""

    def setup_method(self):
        import tempfile
        self.tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.tmp.close()
        from db import PageIndexDB
        self.db = PageIndexDB(self.tmp.name)
        self.db.ensure_schema()

    def teardown_method(self):
        import os
        self.db._connect().close()
        os.unlink(self.tmp.name)

    def _add_doc(self, doc_id, name="test"):
        """Insert a document with a specific id. Returns the doc_id."""
        with self.db._connect() as conn:
            conn.execute("INSERT INTO documents (id, pdf_name, pdf_path) VALUES (?, ?, ?)",
                         (doc_id, name, "/tmp/test.pdf"))
        return doc_id

    def _add_keywords(self, doc_id, keywords):
        """Add keywords with tf counts. keywords: {token: tf}"""
        records = [(doc_id, tok, "content", tf) for tok, tf in keywords.items()]
        self.db.insert_doc_keywords(doc_id, records)

    def test_bm25_higher_tf_ranks_higher(self):
        """BM25: document with higher TF for the same token ranks higher."""
        self._add_doc(1, "doc1")
        self._add_doc(2, "doc2")
        self._add_keywords(1, {"python": 5})
        self._add_keywords(2, {"python": 1})

        results = self.db.match_doc_keywords(["python"], top_k=10)
        assert len(results) == 2
        assert results[0][0] == 1  # higher TF ranks first

    def test_bm25_shorter_doc_with_density_ranks_higher(self):
        """BM25: shorter document with higher TF density ranks higher (doc-length normalization)."""
        self._add_doc(1, "doc1")
        self._add_doc(2, "doc2")
        # doc1: 1 python out of 1 → 100% density
        self._add_keywords(1, {"python": 1})
        # doc2: 1 python out of 1000 → 0.1% density
        long_kw = {"python": 1, **{f"word{i}": 1 for i in range(999)}}
        self._add_keywords(2, long_kw)

        results = self.db.match_doc_keywords(["python"], top_k=10)
        assert results[0][0] == 1  # shorter doc with higher density wins

    def test_bm25_score_positive_for_exact_match(self):
        """BM25: score > 0 for exact token match."""
        self._add_doc(1, "doc1")
        self._add_keywords(1, {"machine": 3, "learning": 2})

        results = self.db.match_doc_keywords(["machine", "learning"], top_k=10)
        assert len(results) == 1
        assert results[0][1] > 0

    def test_bm25_single_token_backward_compat(self):
        """BM25: single-token queries still return results (backward compat)."""
        self._add_doc(1, "doc1")
        self._add_keywords(1, {"test": 1})

        results = self.db.match_doc_keywords(["test"], top_k=10)
        assert len(results) == 1
        assert results[0][0] == 1

    def test_bm25_no_match(self):
        """BM25: no match returns empty list."""
        self._add_doc(1, "doc1")
        self._add_keywords(1, {"test": 1})

        results = self.db.match_doc_keywords(["nonexistent"], top_k=10)
        assert results == []
