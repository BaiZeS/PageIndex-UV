"""T12: doc_summary 落库迁移与兜底语义测试。

[S3] 语料树简化替代：索引期 LLM 生成文档级接地摘要 doc_summary，落库到
documents.doc_summary（空值回退 doc_description，回退语义在 L1 读取侧实现——T9）。

验收覆盖：
1. doc_summary 列 ALTER 迁移（幂等）+ update_doc_summary 写入；
2. 不覆盖 doc_description（其消费者 closet_index/entity_extractor/router/KB identity
   均不受影响）；
3. 未生成摘要的旧行 doc_summary 为空（""/None），L1 消费端 `or` 回退自动生效；
4. 索引期 _enrich_document 生成 doc_summary（llm_completion，NFR4 retrieve_model or model）。
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_doc_summary_column_migrated_and_update(tmp_path):
    from db import PageIndexDB

    db = PageIndexDB(str(tmp_path / "t.db"))
    try:
        did = db.insert_document(pdf_name="A", pdf_path="", doc_description="旧描述")
        db.update_doc_summary(did, "新接地摘要")
        assert db.get_document_by_id(did)["doc_summary"] == "新接地摘要"
        # 不覆盖 doc_description（其消费者不受影响）
        assert db.get_document_by_id(did)["doc_description"] == "旧描述"
    finally:
        db.close()


def test_doc_summary_defaults_empty(tmp_path):
    from db import PageIndexDB

    db = PageIndexDB(str(tmp_path / "t.db"))
    try:
        did = db.insert_document(pdf_name="A", pdf_path="", doc_description="d")
        assert db.get_document_by_id(did).get("doc_summary") in ("", None)
    finally:
        db.close()


class TestDocSummaryEnrichment:
    """索引期 _enrich_document 生成 doc_summary（并行线程）并落库。"""

    @pytest.fixture
    def client_factory(self, tmp_path):
        sys.modules["PyPDF2"] = MagicMock()
        from pageindex_mutil.client import PageIndexClient

        def _make(retrieve_model=None):
            db_path = str(tmp_path / "test.db")
            return PageIndexClient(
                db_path=db_path, search_backend="keyword",
                retrieve_model=retrieve_model,
            )
        return _make

    def test_index_generates_doc_summary_using_retrieve_model(
        self, client_factory, tmp_path
    ):
        """LLM 产出摘要 → 落库 doc_summary，且用 retrieve_model（NFR4）。"""
        client = client_factory(retrieve_model="r-model")
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            client.search_backend.index_document = MagicMock()
            client.entity_extractor.extract_from_document = MagicMock(
                return_value=([], [], [])
            )

            calls = []

            def fake_llm(model, prompt, **kw):
                calls.append(model)
                return "覆盖式接地摘要"

            md_path = tmp_path / "doc.md"
            md_path.write_text("# Test\n\ncontent\n", encoding="utf-8")
            with patch(
                "pageindex_mutil.client.md_to_tree",
                return_value={
                    "doc_name": "doc.md",
                    "doc_description": "旧描述",
                    "line_count": 2,
                    "structure": [{
                        "node_id": "n1", "title": "Test",
                        "text": "content", "summary": "s", "level": 1,
                    }],
                },
            ), patch(
                "pageindex_mutil.client.llm_completion", side_effect=fake_llm
            ):
                client.index(str(md_path), mode="md")

            db_doc = client.db.get_document_by_name("doc.md")
            assert db_doc["doc_summary"] == "覆盖式接地摘要"
            # 不覆盖 doc_description
            assert db_doc["doc_description"] == "旧描述"
            assert calls == ["r-model"]
        finally:
            client.close()

    def test_index_llm_empty_leaves_summary_empty(self, client_factory, tmp_path):
        """LLM 失败/空响应 → doc_summary 留空（L1 回退 doc_description）。"""
        client = client_factory(retrieve_model=None)
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            client.search_backend.index_document = MagicMock()
            client.entity_extractor.extract_from_document = MagicMock(
                return_value=([], [], [])
            )

            md_path = tmp_path / "doc.md"
            md_path.write_text("# Test\n\ncontent\n", encoding="utf-8")
            with patch(
                "pageindex_mutil.client.md_to_tree",
                return_value={
                    "doc_name": "doc.md",
                    "doc_description": "旧描述",
                    "line_count": 2,
                    "structure": [{
                        "node_id": "n1", "title": "Test",
                        "text": "content", "summary": "s", "level": 1,
                    }],
                },
            ), patch("pageindex_mutil.client.llm_completion", return_value=""):
                client.index(str(md_path), mode="md")

            db_doc = client.db.get_document_by_name("doc.md")
            assert db_doc.get("doc_summary") in ("", None)
            assert db_doc["doc_description"] == "旧描述"
        finally:
            client.close()

    def test_index_batch_generates_doc_summary_using_retrieve_model(
        self, client_factory, tmp_path
    ):
        """Batch path also generates doc_summary (NFR4 retrieve_model wiring)."""
        client = client_factory(retrieve_model="r-model")
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            client.search_backend.index_document = MagicMock()
            client.entity_extractor.extract_from_document = MagicMock(
                return_value=([], [], [])
            )
            client.entity_extractor.normalize_entities_batch = MagicMock()

            calls = []

            def fake_llm(model, prompt, **kw):
                calls.append(model)
                return "批量接地摘要"

            md_path = tmp_path / "doc.md"
            md_path.write_text("# Test\n\ncontent\n", encoding="utf-8")
            with patch(
                "pageindex_mutil.client.md_to_tree",
                return_value={
                    "doc_name": "doc.md",
                    "doc_description": "旧描述",
                    "line_count": 2,
                    "structure": [{
                        "node_id": "n1", "title": "Test",
                        "text": "content", "summary": "s", "level": 1,
                    }],
                },
            ), patch(
                "pageindex_mutil.client.llm_completion", side_effect=fake_llm
            ):
                client.index_batch([str(md_path)], mode="md")

            db_doc = client.db.get_document_by_name("doc.md")
            assert db_doc["doc_summary"] == "批量接地摘要"
            # 不覆盖 doc_description
            assert db_doc["doc_description"] == "旧描述"
            assert calls == ["r-model"]
        finally:
            client.close()
