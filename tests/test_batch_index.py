"""T11: Batch mode index_batch tests.

Phase 1 — Extraction (per-doc, concurrent with semaphore)
Phase 2 — Batch corpus tree rebuild
Phase 3 — Batch entity normalization (LLM merges synonyms once)
Phase 4 — Search backend + super_tree indexing

All LLM calls mocked. No real LLM, no vectors.
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from db import PageIndexDB


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = PageIndexDB(db_path)
    yield db
    db.close()


@pytest.fixture
def client_factory(tmp_path):
    """Create a PageIndexClient with a temp DB."""
    sys.modules["PyPDF2"] = MagicMock()
    from pageindex_mutil.client import PageIndexClient

    def _make():
        db_path = str(tmp_path / "test.db")
        return PageIndexClient(db_path=db_path, search_backend="keyword")
    return _make


def _make_md_file(tmp_path, name="test.md", content=None):
    """Create a temp markdown file."""
    p = tmp_path / name
    p.write_text(content or "# Test\n\nHello world.\n")
    return str(p)


def _mock_md_to_tree(doc_name="test.md", description="A test document"):
    """Return a patched md_to_tree that returns a fixed structure."""
    return patch(
        "pageindex_mutil.client.md_to_tree",
        return_value={
            "doc_name": doc_name,
            "doc_description": description,
            "line_count": 3,
            "structure": [
                {
                    "node_id": "n1",
                    "title": "Test",
                    "text": "Hello world.",
                    "summary": "A test section",
                    "level": 1,
                }
            ],
        },
    )


# ===========================================================================
# (a) Batch mode extracts all docs
# ===========================================================================

class TestBatchExtractsAllDocs:
    """index_batch should parse, DB-insert, and extract entities for every doc."""

    def test_batch_returns_doc_ids_for_all_files(self, client_factory, tmp_path):
        """index_batch returns one doc_id per input file."""
        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            client.corpus_tree.rebuild = MagicMock(return_value={})
            client.search_backend.index_document = MagicMock()
            client.entity_extractor.extract_from_document = MagicMock(
                return_value=([], [], [])
            )

            paths = [
                _make_md_file(tmp_path, f"doc{i}.md", f"# Doc {i}\n\nContent {i}\n")
                for i in range(3)
            ]
            with patch(
                "pageindex_mutil.client.md_to_tree",
                side_effect=[
                    {
                        "doc_name": f"doc{i}.md",
                        "doc_description": f"Document {i}",
                        "line_count": 3,
                        "structure": [
                            {
                                "node_id": f"n{i}",
                                "title": f"Doc {i}",
                                "text": f"Content {i}",
                                "summary": f"Summary {i}",
                                "level": 1,
                            }
                        ],
                    }
                    for i in range(3)
                ],
            ):
                doc_ids = client.index_batch(paths, mode="md")

            assert len(doc_ids) == 3
            for did in doc_ids:
                assert isinstance(did, str)
                assert len(did) > 0
        finally:
            client.close()

    def test_batch_inserts_documents_in_db(self, client_factory, tmp_path):
        """Each document from index_batch is persisted in the DB."""
        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            client.corpus_tree.rebuild = MagicMock(return_value={})
            client.search_backend.index_document = MagicMock()
            client.entity_extractor.extract_from_document = MagicMock(
                return_value=([], [], [])
            )

            paths = [
                _make_md_file(tmp_path, f"doc{i}.md", f"# Doc {i}\n\nContent {i}\n")
                for i in range(2)
            ]
            with patch(
                "pageindex_mutil.client.md_to_tree",
                side_effect=[
                    {
                        "doc_name": f"doc{i}.md",
                        "doc_description": f"Document {i}",
                        "line_count": 3,
                        "structure": [
                            {
                                "node_id": f"n{i}",
                                "title": f"Doc {i}",
                                "text": f"Content {i}",
                                "summary": f"Summary {i}",
                                "level": 1,
                            }
                        ],
                    }
                    for i in range(2)
                ],
            ):
                doc_ids = client.index_batch(paths, mode="md")

            # All docs should be in the DB
            db_docs = client.db.get_all_documents()
            assert len(db_docs) == 2
        finally:
            client.close()

    def test_batch_entity_extraction_called_per_doc(self, client_factory, tmp_path):
        """Entity extraction is called once per document in batch mode."""
        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            client.corpus_tree.rebuild = MagicMock(return_value={})
            client.search_backend.index_document = MagicMock()
            extract_mock = MagicMock(return_value=([], [], []))
            client.entity_extractor.extract_from_document = extract_mock

            paths = [
                _make_md_file(tmp_path, f"doc{i}.md", f"# Doc {i}\n\nContent {i}\n")
                for i in range(3)
            ]
            with patch(
                "pageindex_mutil.client.md_to_tree",
                side_effect=[
                    {
                        "doc_name": f"doc{i}.md",
                        "doc_description": f"Document {i}",
                        "line_count": 3,
                        "structure": [
                            {
                                "node_id": f"n{i}",
                                "title": f"Doc {i}",
                                "text": f"Content {i}",
                                "summary": f"Summary {i}",
                                "level": 1,
                            }
                        ],
                    }
                    for i in range(3)
                ],
            ):
                client.index_batch(paths, mode="md")

            # Entity extraction called 3 times (once per doc)
            assert extract_mock.call_count == 3
        finally:
            client.close()


# ===========================================================================
# (b) Batch normalization runs once (not per-doc)
# ===========================================================================

class TestBatchNormalizationRunsOnce:
    """Batch mode should call corpus_tree.rebuild() once, NOT
    corpus_tree.update_for_document() per doc."""

    def test_rebuild_called_once_not_incremental(self, client_factory, tmp_path):
        """corpus_tree.rebuild() called exactly once; update_for_document NOT called."""
        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            rebuild_mock = MagicMock(return_value={})
            client.corpus_tree.rebuild = rebuild_mock
            update_mock = MagicMock()
            client.corpus_tree.update_for_document = update_mock
            client.search_backend.index_document = MagicMock()
            client.entity_extractor.extract_from_document = MagicMock(
                return_value=([], [], [])
            )

            paths = [
                _make_md_file(tmp_path, f"doc{i}.md", f"# Doc {i}\n\nContent {i}\n")
                for i in range(5)
            ]
            with patch(
                "pageindex_mutil.client.md_to_tree",
                side_effect=[
                    {
                        "doc_name": f"doc{i}.md",
                        "doc_description": f"Document {i}",
                        "line_count": 3,
                        "structure": [
                            {
                                "node_id": f"n{i}",
                                "title": f"Doc {i}",
                                "text": f"Content {i}",
                                "summary": f"Summary {i}",
                                "level": 1,
                            }
                        ],
                    }
                    for i in range(5)
                ],
            ):
                client.index_batch(paths, mode="md")

            # Batch: rebuild once, NOT incremental per-doc
            rebuild_mock.assert_called_once()
            update_mock.assert_not_called()
        finally:
            client.close()

    def test_entity_normalize_batch_called_once(self, client_factory, tmp_path):
        """normalize_entities_batch is called exactly once after extraction."""
        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            client.corpus_tree.rebuild = MagicMock(return_value={})
            client.search_backend.index_document = MagicMock()
            client.entity_extractor.extract_from_document = MagicMock(
                return_value=([], [], [])
            )
            normalize_mock = MagicMock()
            client.entity_extractor.normalize_entities_batch = normalize_mock

            paths = [
                _make_md_file(tmp_path, f"doc{i}.md", f"# Doc {i}\n\nContent {i}\n")
                for i in range(3)
            ]
            with patch(
                "pageindex_mutil.client.md_to_tree",
                side_effect=[
                    {
                        "doc_name": f"doc{i}.md",
                        "doc_description": f"Document {i}",
                        "line_count": 3,
                        "structure": [
                            {
                                "node_id": f"n{i}",
                                "title": f"Doc {i}",
                                "text": f"Content {i}",
                                "summary": f"Summary {i}",
                                "level": 1,
                            }
                        ],
                    }
                    for i in range(3)
                ],
            ):
                client.index_batch(paths, mode="md")

            # Batch normalization: called once, not per-doc
            normalize_mock.assert_called_once()
        finally:
            client.close()


# ===========================================================================
# (c) Entity normalization merges synonyms
# ===========================================================================

class TestEntityNormalizationMergesSynonyms:
    """normalize_entities_batch should merge synonym entities via LLM."""

    def test_normalize_merges_synonym_entities(self, db):
        """Two entities of same type with synonym names → merged into one."""
        from pageindex_mutil.entity_extractor import EntityExtractor

        extractor = EntityExtractor(model="test", retrieve_model="test")

        # Insert two synonym entities
        e1 = db.insert_entity("person", "张三", ["小张"])
        e2 = db.insert_entity("person", "张先生", ["老张"])
        doc1 = db.insert_document("doc1", "/tmp/doc1.pdf")
        doc2 = db.insert_document("doc2", "/tmp/doc2.pdf")
        db.insert_entity_mention(e1, doc1, confidence=0.9)
        db.insert_entity_mention(e2, doc2, confidence=0.8)

        # Mock LLM to merge them
        with patch("pageindex_mutil.entity_extractor.llm_completion") as mock_llm:
            mock_llm.return_value = json.dumps({
                "groups": [
                    {"canonical": "张三", "synonyms": ["张三", "张先生"]}
                ]
            })
            extractor.normalize_entities_batch(db)

        # After normalization: only one entity "张三" should remain
        entities = db.get_entities_by_type("person")
        names = {e["name"] for e in entities}
        assert "张三" in names
        # "张先生" should be merged into "张三"
        zhang_san = [e for e in entities if e["name"] == "张三"][0]
        aliases = json.loads(zhang_san.get("aliases", "[]"))
        assert "张先生" in aliases or "老张" in aliases

    def test_normalize_preserves_distinct_entities(self, db):
        """Entities with different meanings should NOT be merged."""
        from pageindex_mutil.entity_extractor import EntityExtractor

        extractor = EntityExtractor(model="test", retrieve_model="test")

        db.insert_entity("person", "张三")
        db.insert_entity("person", "李四")

        with patch("pageindex_mutil.entity_extractor.llm_completion") as mock_llm:
            mock_llm.return_value = json.dumps({
                "groups": [
                    {"canonical": "张三", "synonyms": ["张三"]},
                    {"canonical": "李四", "synonyms": ["李四"]},
                ]
            })
            extractor.normalize_entities_batch(db)

        entities = db.get_entities_by_type("person")
        names = {e["name"] for e in entities}
        assert "张三" in names
        assert "李四" in names

    def test_normalize_handles_llm_failure_gracefully(self, db):
        """LLM failure → entities unchanged (conservative)."""
        from pageindex_mutil.entity_extractor import EntityExtractor

        extractor = EntityExtractor(model="test", retrieve_model="test")

        e1 = db.insert_entity("person", "张三")
        e2 = db.insert_entity("person", "张先生")

        with patch("pageindex_mutil.entity_extractor.llm_completion") as mock_llm:
            mock_llm.return_value = None  # LLM failure
            extractor.normalize_entities_batch(db)

        # Both entities should still exist
        entities = db.get_entities_by_type("person")
        names = {e["name"] for e in entities}
        assert "张三" in names
        assert "张先生" in names

    def test_normalize_groups_by_type(self, db):
        """normalize_entities_batch processes each entity type separately."""
        from pageindex_mutil.entity_extractor import EntityExtractor

        extractor = EntityExtractor(model="test", retrieve_model="test")

        # Need 2+ entities per type to trigger normalization
        db.insert_entity("person", "张三")
        db.insert_entity("person", "张先生")
        db.insert_entity("concept", "风控")
        db.insert_entity("concept", "风险管理")

        llm_calls = []

        def mock_llm(model, prompt, **kw):
            llm_calls.append(prompt)
            # Return identity mapping (no merges)
            if "张三" in prompt:
                return json.dumps({
                    "groups": [{"canonical": "张三", "synonyms": ["张三", "张先生"]}]
                })
            else:
                return json.dumps({
                    "groups": [{"canonical": "风控", "synonyms": ["风控", "风险管理"]}]
                })

        with patch("pageindex_mutil.entity_extractor.llm_completion", side_effect=mock_llm):
            extractor.normalize_entities_batch(db)

        # Should make 2 LLM calls (one per type)
        assert len(llm_calls) == 2

    def test_normalize_merges_mentions_to_canonical(self, db):
        """After merge, mentions of synonym entity should point to canonical."""
        from pageindex_mutil.entity_extractor import EntityExtractor

        extractor = EntityExtractor(model="test", retrieve_model="test")

        e1 = db.insert_entity("person", "张三")
        e2 = db.insert_entity("person", "张先生")
        doc1 = db.insert_document("doc1", "/tmp/doc1.pdf")
        doc2 = db.insert_document("doc2", "/tmp/doc2.pdf")
        db.insert_entity_mention(e1, doc1, confidence=0.9)
        db.insert_entity_mention(e2, doc2, confidence=0.8)

        with patch("pageindex_mutil.entity_extractor.llm_completion") as mock_llm:
            mock_llm.return_value = json.dumps({
                "groups": [
                    {"canonical": "张三", "synonyms": ["张三", "张先生"]}
                ]
            })
            extractor.normalize_entities_batch(db)

        # Both docs should be mentioned under "张三"
        zhang_san = db.get_entity_by_name("张三")
        assert zhang_san is not None
        docs = db.get_entity_documents(zhang_san["id"])
        doc_ids = {d["id"] for d in docs}
        assert doc1 in doc_ids
        assert doc2 in doc_ids


# ===========================================================================
# (d) Quality: batch normalization sees full tag/entity set
# ===========================================================================

class TestBatchQualityImprovement:
    """Batch normalization is MORE consistent than incremental because
    it sees the complete tag/entity set."""

    def test_batch_sees_all_entities_for_normalization(self, client_factory, tmp_path):
        """Batch mode's normalize_entities_batch receives DB with all entities
        from all docs, enabling better merge decisions."""
        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            client.corpus_tree.rebuild = MagicMock(return_value={})
            client.search_backend.index_document = MagicMock()

            # Simulate: doc1 extracts "张三", doc2 extracts "张先生"
            from pageindex_mutil.entity_extractor import Entity, EntityRelation
            entity_sets = [
                ([Entity(name="张三", entity_type="person", aliases=["小张"], confidence=0.9)], [], []),
                ([Entity(name="张先生", entity_type="person", aliases=["老张"], confidence=0.85)], [], []),
            ]
            call_idx = {"i": 0}

            def mock_extract(*args, **kwargs):
                i = call_idx["i"]
                call_idx["i"] += 1
                return entity_sets[i]

            client.entity_extractor.extract_from_document = mock_extract

            # normalize_entities_batch should see both entities
            captured_db_states = []
            original_normalize = MagicMock()

            def capture_normalize(db):
                entities = db.get_entities_by_type("person")
                captured_db_states.append({e["name"] for e in entities})
                return original_normalize(db)

            client.entity_extractor.normalize_entities_batch = capture_normalize

            paths = [
                _make_md_file(tmp_path, f"doc{i}.md", f"# Doc {i}\n\nContent {i}\n")
                for i in range(2)
            ]
            with patch(
                "pageindex_mutil.client.md_to_tree",
                side_effect=[
                    {
                        "doc_name": f"doc{i}.md",
                        "doc_description": f"Document {i}",
                        "line_count": 3,
                        "structure": [
                            {
                                "node_id": f"n{i}",
                                "title": f"Doc {i}",
                                "text": f"Content {i}",
                                "summary": f"Summary {i}",
                                "level": 1,
                            }
                        ],
                    }
                    for i in range(2)
                ],
            ):
                client.index_batch(paths, mode="md")

            # normalize_entities_batch should see BOTH entities
            assert len(captured_db_states) == 1
            assert "张三" in captured_db_states[0]
            assert "张先生" in captured_db_states[0]
        finally:
            client.close()

    def test_batch_no_incremental_corpus_update(self, client_factory, tmp_path):
        """Batch mode does NOT call corpus_tree.update_for_document (incremental)."""
        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            rebuild_mock = MagicMock(return_value={})
            client.corpus_tree.rebuild = rebuild_mock
            update_mock = MagicMock()
            client.corpus_tree.update_for_document = update_mock
            client.search_backend.index_document = MagicMock()
            client.entity_extractor.extract_from_document = MagicMock(
                return_value=([], [], [])
            )
            client.entity_extractor.normalize_entities_batch = MagicMock()

            paths = [
                _make_md_file(tmp_path, f"doc{i}.md", f"# Doc {i}\n\nContent {i}\n")
                for i in range(3)
            ]
            with patch(
                "pageindex_mutil.client.md_to_tree",
                side_effect=[
                    {
                        "doc_name": f"doc{i}.md",
                        "doc_description": f"Document {i}",
                        "line_count": 3,
                        "structure": [
                            {
                                "node_id": f"n{i}",
                                "title": f"Doc {i}",
                                "text": f"Content {i}",
                                "summary": f"Summary {i}",
                                "level": 1,
                            }
                        ],
                    }
                    for i in range(3)
                ],
            ):
                client.index_batch(paths, mode="md")

            # No incremental updates
            update_mock.assert_not_called()
            # One batch rebuild
            rebuild_mock.assert_called_once()
        finally:
            client.close()

    def test_batch_search_backend_indexed_for_all_docs(self, client_factory, tmp_path):
        """Phase 4: search_backend.index_document called for each doc."""
        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            client.corpus_tree.rebuild = MagicMock(return_value={})
            search_mock = MagicMock()
            client.search_backend.index_document = search_mock
            client.entity_extractor.extract_from_document = MagicMock(
                return_value=([], [], [])
            )
            client.entity_extractor.normalize_entities_batch = MagicMock()

            paths = [
                _make_md_file(tmp_path, f"doc{i}.md", f"# Doc {i}\n\nContent {i}\n")
                for i in range(3)
            ]
            with patch(
                "pageindex_mutil.client.md_to_tree",
                side_effect=[
                    {
                        "doc_name": f"doc{i}.md",
                        "doc_description": f"Document {i}",
                        "line_count": 3,
                        "structure": [
                            {
                                "node_id": f"n{i}",
                                "title": f"Doc {i}",
                                "text": f"Content {i}",
                                "summary": f"Summary {i}",
                                "level": 1,
                            }
                        ],
                    }
                    for i in range(3)
                ],
            ):
                client.index_batch(paths, mode="md")

            # Search backend indexed for each doc
            assert search_mock.call_count == 3
        finally:
            client.close()
