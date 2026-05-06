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
