import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from db import PageIndexDB, _TOKENIZE_CACHE


@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = PageIndexDB(path)
    yield db
    db.close()
    os.unlink(path)


class TestDocKeywords:
    def test_insert_and_match(self, tmp_db):
        doc_id = tmp_db.insert_document("test.pdf", "/tmp/test.pdf")
        tmp_db.insert_doc_keywords(doc_id, [
            (doc_id, "前端", "name"),
            (doc_id, "脚本", "name"),
            (doc_id, "开发", "description"),
        ])
        results = tmp_db.match_doc_keywords(["前端", "脚本"], top_k=5)
        assert len(results) == 1
        assert results[0][0] == doc_id

    def test_delete_doc_keywords(self, tmp_db):
        doc_id = tmp_db.insert_document("test.pdf", "/tmp/test.pdf")
        tmp_db.insert_doc_keywords(doc_id, [(doc_id, "test", "name")])
        tmp_db.delete_doc_keywords(doc_id)
        results = tmp_db.match_doc_keywords(["test"], top_k=5)
        assert len(results) == 0


class TestKBIdentity:
    def test_set_and_get(self, tmp_db):
        tmp_db.set_kb_identity("知识库共3个文档", 3)
        identity = tmp_db.get_kb_identity()
        assert identity == "知识库共3个文档"

    def test_get_missing_returns_none(self, tmp_db):
        assert tmp_db.get_kb_identity() is None


class TestClosetTags:
    def test_get_doc_tags(self, tmp_db):
        doc_id = tmp_db.insert_document("test.pdf", "/tmp/test.pdf")
        tmp_db.insert_closet_tags(doc_id, [
            (doc_id, "容器编排", "容器 编排", 0.95, "llm"),
            (doc_id, "微服务", "微服务", 0.8, "llm"),
        ])
        tags = tmp_db.get_doc_tags(doc_id)
        assert tags == [
            {"tag_text": "容器编排", "confidence": 0.95},
            {"tag_text": "微服务", "confidence": 0.8},
        ]

    def test_get_doc_tags_empty(self, tmp_db):
        doc_id = tmp_db.insert_document("test.pdf", "/tmp/test.pdf")
        assert tmp_db.get_doc_tags(doc_id) == []


def _count_rows(db, table, doc_id):
    """Count child-table rows for a given doc_id (helper for cascade tests)."""
    conn = db._connect()
    row = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    return row[0]


class TestDeleteDocumentCascade:
    """W2 FR1/AC1.1 — delete_document cascades to nodes/pages/closet_tags/doc_keywords.

    Proves P0-2: currently PageIndexDB has no delete_document method, so this
    test must fail with AttributeError (RED) until FR1 is implemented.
    """

    def test_delete_document_cascades_children(self, tmp_db):
        doc_id = tmp_db.insert_document("cascade.pdf", "/tmp/cascade.pdf")
        # Populate all 4 child tables that declare ON DELETE CASCADE.
        tmp_db.insert_nodes(doc_id, [(doc_id, "n1", "title", "summary", 0, 10, None)])
        tmp_db.insert_pages(doc_id, [(doc_id, 1, "page one")])
        tmp_db.insert_closet_tags(doc_id, [(doc_id, "tag", "token", 0.9, "manual")])
        tmp_db.insert_doc_keywords(doc_id, [(doc_id, "keyword", "name")])

        # Pre-condition: child rows exist.
        assert _count_rows(tmp_db, "nodes", doc_id) == 1
        assert _count_rows(tmp_db, "pages", doc_id) == 1
        assert _count_rows(tmp_db, "closet_tags", doc_id) == 1
        assert _count_rows(tmp_db, "doc_keywords", doc_id) == 1

        tmp_db.delete_document(doc_id)

        # FR1/AC1.1: documents + all 4 child tables cleared via cascade.
        conn = tmp_db._connect()
        doc_count = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()[0]
        assert doc_count == 0
        assert _count_rows(tmp_db, "nodes", doc_id) == 0
        assert _count_rows(tmp_db, "pages", doc_id) == 0
        assert _count_rows(tmp_db, "closet_tags", doc_id) == 0
        assert _count_rows(tmp_db, "doc_keywords", doc_id) == 0

class TestInsertEntityAliasMerge:
    """C1: insert_entity UPSERT must MERGE aliases, not overwrite them."""

    def test_two_inserts_same_entity_different_aliases_merged(self, tmp_db):
        """Two docs extracting the same entity with different aliases → both aliases kept."""
        eid1 = tmp_db.insert_entity("PERSON", "Alice", ["Ali", "A"])
        eid2 = tmp_db.insert_entity("PERSON", "Alice", ["Ally", "A"])
        assert eid1 == eid2  # same entity
        row = tmp_db._connect().execute(
            "SELECT aliases FROM entities WHERE id = ?", (eid1,)
        ).fetchone()
        aliases = set(__import__("json").loads(row["aliases"]))
        # All four unique aliases must be present (A deduplicated)
        assert aliases == {"Ali", "A", "Ally"}

    def test_insert_with_empty_aliases_preserves_existing(self, tmp_db):
        """Inserting with aliases=[] must not discard existing aliases."""
        tmp_db.insert_entity("ORG", "Acme", ["Acme Corp"])
        tmp_db.insert_entity("ORG", "Acme", [])
        row = tmp_db._connect().execute(
            "SELECT aliases FROM entities WHERE name = 'Acme'"
        ).fetchone()
        aliases = __import__("json").loads(row["aliases"])
        assert "Acme Corp" in aliases


    def test_delete_document_idempotent_nonexistent(self, tmp_db):
        """NFR2/AC1.2 — deleting a non-existent id deletes 0 rows, no error."""
        # 999999 does not exist; DELETE matches 0 rows and returns normally.
        tmp_db.delete_document(999999)
        # Sanity: documents table still empty.
        conn = tmp_db._connect()
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


class TestGetDocumentCount:
    def test_empty_db_returns_zero(self, tmp_db):
        assert tmp_db.get_document_count() == 0

    def test_returns_correct_count(self, tmp_db):
        tmp_db.insert_document("a.pdf", "/tmp/a.pdf")
        tmp_db.insert_document("b.pdf", "/tmp/b.pdf")
        assert tmp_db.get_document_count() == 2

    def test_count_matches_get_all(self, tmp_db):
        tmp_db.insert_document("x.pdf", "/tmp/x.pdf")
        assert tmp_db.get_document_count() == len(tmp_db.get_all_documents())


class TestInsertEntityMentionsBatch:
    def test_batch_insert(self, tmp_db):
        eid = tmp_db.insert_entity("PERSON", "Alice")
        doc1 = tmp_db.insert_document("a.pdf", "/tmp/a.pdf")
        doc2 = tmp_db.insert_document("b.pdf", "/tmp/b.pdf")
        records = [
            (eid, doc1, "Alice in doc1", 0.9),
            (eid, doc2, "Alice in doc2", 0.8),
        ]
        tmp_db.insert_entity_mentions_batch(records)
        docs = tmp_db.get_entity_documents(eid)
        assert len(docs) == 2
        # Verify doc_count was updated
        entity = tmp_db.get_entity_by_name("Alice")
        assert entity["doc_count"] == 2

    def test_batch_empty_noop(self, tmp_db):
        tmp_db.insert_entity_mentions_batch([])  # should not raise


class TestInsertClosetTagsBatch:
    def test_batch_insert_single_doc(self, tmp_db):
        doc_id = tmp_db.insert_document("test.pdf", "/tmp/test.pdf")
        records = [
            (doc_id, "容器编排", "容器 编排", 0.9, "llm"),
            (doc_id, "微服务", "微服务", 0.8, "llm"),
        ]
        tmp_db.insert_closet_tags_batch(records)
        tags = tmp_db.get_doc_tags(doc_id)
        assert len(tags) == 2
        assert tags[0]["tag_text"] == "容器编排"

    def test_batch_insert_multi_doc(self, tmp_db):
        d1 = tmp_db.insert_document("a.pdf", "/tmp/a.pdf")
        d2 = tmp_db.insert_document("b.pdf", "/tmp/b.pdf")
        records = [
            (d1, "tag1", "token1", 0.9, "llm"),
            (d2, "tag2", "token2", 0.8, "llm"),
        ]
        tmp_db.insert_closet_tags_batch(records)
        assert len(tmp_db.get_doc_tags(d1)) == 1
        assert len(tmp_db.get_doc_tags(d2)) == 1

    def test_batch_replaces_existing(self, tmp_db):
        doc_id = tmp_db.insert_document("test.pdf", "/tmp/test.pdf")
        tmp_db.insert_closet_tags(doc_id, [(doc_id, "old", "old", 0.5, "llm")])
        tmp_db.insert_closet_tags_batch([(doc_id, "new", "new", 0.9, "llm")])
        tags = tmp_db.get_doc_tags(doc_id)
        assert len(tags) == 1
        assert tags[0]["tag_text"] == "new"

    def test_batch_empty_noop(self, tmp_db):
        tmp_db.insert_closet_tags_batch([])  # should not raise


class TestTokenizationCache:
    def setup_method(self):
        _TOKENIZE_CACHE.clear()

    def test_cache_hit(self, tmp_db):
        result1 = PageIndexDB._tokenize_query("人工智能技术")
        result2 = PageIndexDB._tokenize_query("人工智能技术")
        assert result1 == result2
        assert "人工智能技术" in _TOKENIZE_CACHE

    def test_cache_different_queries(self, tmp_db):
        result1 = PageIndexDB._tokenize_query("机器学习")
        result2 = PageIndexDB._tokenize_query("深度学习")
        assert "机器学习" in _TOKENIZE_CACHE
        assert "深度学习" in _TOKENIZE_CACHE

    def test_cache_ttl_expiry(self, tmp_db):
        _TOKENIZE_CACHE["test_query"] = (["old_tokens"], time.monotonic() - 600)
        result = PageIndexDB._tokenize_query("test_query")
        assert result != ["old_tokens"]  # should have been recomputed
        assert "test_query" in _TOKENIZE_CACHE

    def test_cache_max_eviction(self, tmp_db):
        # Fill cache to max
        for i in range(512):
            _TOKENIZE_CACHE[f"q{i}"] = (["tok"], time.monotonic())
        # One more should evict oldest
        PageIndexDB._tokenize_query("new_query")
        assert "new_query" in _TOKENIZE_CACHE
