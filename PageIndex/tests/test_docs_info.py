"""Unit tests for _build_docs_info reverse mapping."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


class FakeDB:
    def __init__(self, docs):
        self._docs = docs

    def get_all_documents(self):
        return self._docs

    def get_top_level_nodes(self, doc_id):
        return [{"title": "Section 1"}]


class FakeClient:
    def __init__(self, docs, uuid_to_db):
        self.documents = docs
        self.db = None
        self._uuid_to_db = uuid_to_db


def test_build_docs_info_with_reverse_mapping():
    from PageIndex.pageindex.agentic.router import AgenticRouter

    # DB has integer ids; client.documents has UUID keys
    db_docs = [
        {"id": 1, "pdf_name": "doc1.pdf", "doc_description": "Desc 1"},
        {"id": 2, "pdf_name": "doc2.pdf", "doc_description": "Desc 2"},
    ]
    in_mem = {
        "uuid-1": {"doc_name": "doc1.pdf", "doc_description": "Desc 1"},
    }
    uuid_to_db = {"uuid-1": 1, "uuid-2": 2}

    client = FakeClient(in_mem, uuid_to_db)
    client.db = FakeDB(db_docs)

    router = AgenticRouter(client, "qwen-plus")
    docs_info = router._build_docs_info()

    # Only uuid-1 should appear because it's the only one in both DB and memory
    assert len(docs_info) == 1
    assert docs_info[0]["doc_id"] == "uuid-1"
    assert docs_info[0]["doc_name"] == "doc1.pdf"


def test_build_docs_info_fallback_to_memory():
    from PageIndex.pageindex.agentic.router import AgenticRouter

    client = FakeClient({"uuid-1": {"doc_name": "mem.pdf"}}, {})
    client.db = None

    router = AgenticRouter(client, "qwen-plus")
    docs_info = router._build_docs_info()

    assert len(docs_info) == 1
    assert docs_info[0]["doc_id"] == "uuid-1"
    assert docs_info[0]["doc_name"] == "mem.pdf"
