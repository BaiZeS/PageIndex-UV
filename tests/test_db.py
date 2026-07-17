import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from db import PageIndexDB


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

    def test_delete_document_idempotent_nonexistent(self, tmp_db):
        """NFR2/AC1.2 — deleting a non-existent id deletes 0 rows, no error."""
        # 999999 does not exist; DELETE matches 0 rows and returns normally.
        tmp_db.delete_document(999999)
        # Sanity: documents table still empty.
        conn = tmp_db._connect()
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
