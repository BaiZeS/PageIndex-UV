import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from db import PageIndexDB


class TestPageIndexClientSuperTree:
    def test_super_tree_index_initialized_with_db(self):
        """PageIndexClient with db_path should initialize super_tree_index."""
        # We need to mock PyPDF2 since client.py imports it at top level
        sys.modules["PyPDF2"] = MagicMock()

        from pageindex_mutil.client import PageIndexClient

        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            client = PageIndexClient(db_path=db_path)
            assert hasattr(client, "super_tree_index")
            assert client.super_tree_index is not None
            assert client.router is not None
            assert client.router.super_tree_index is client.super_tree_index
        finally:
            client.close()
            os.unlink(db_path)

    def test_super_tree_index_none_without_db(self):
        """PageIndexClient without db_path should not have super_tree_index."""
        sys.modules["PyPDF2"] = MagicMock()

        from pageindex_mutil.client import PageIndexClient

        client = PageIndexClient()
        assert hasattr(client, "super_tree_index")
        assert client.super_tree_index is None
        assert client.router is None

    def test_on_document_added_called_during_index(self):
        """index() should call super_tree_index.on_document_added after DB insert."""
        sys.modules["PyPDF2"] = MagicMock()

        from pageindex_mutil.client import PageIndexClient

        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            client = PageIndexClient(db_path=db_path)

            # Mock super_tree_index.on_document_added
            client.super_tree_index.on_document_added = MagicMock()

            # Create a temp markdown file to index
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
                f.write("# Test Document\n\nThis is a test.\n")
                md_path = f.name

            try:
                # Mock md_to_tree to avoid async complexity and LLM calls
                mock_structure = [
                    {
                        "node_id": "n1",
                        "title": "Test Document",
                        "text": "This is a test.",
                        "summary": "A test doc",
                        "level": 1,
                    }
                ]
                with patch("pageindex_mutil.client.md_to_tree") as mock_md:
                    mock_md.return_value = {
                        "doc_name": "test.md",
                        "doc_description": "A test markdown file",
                        "line_count": 3,
                        "structure": mock_structure,
                    }
                    doc_id = client.index(md_path, mode="md")

                # Verify on_document_added was called
                assert client.super_tree_index.on_document_added.called
                # It should be called with the db_doc_id (which is 1 for first insert)
                call_args = client.super_tree_index.on_document_added.call_args
                assert call_args[0][0] == 1  # First document gets db_id = 1
            finally:
                os.unlink(md_path)
        finally:
            client.close()
            os.unlink(db_path)


@pytest.mark.asyncio
async def test_single_doc_goes_through_unified_chain(tmp_path):
    """单文档不再走 _search_single 分支——统一链（router.search → super_tree）接管。

    [S4] 单链：候选=1 不再短路 _search_single；client.search 一律经 router。
    _search_single 若被调用会记录 doc_id（应保持为空）。
    """
    import types

    sys.modules["PyPDF2"] = MagicMock()

    from pageindex_mutil.client import PageIndexClient
    from pageindex_mutil.agentic.router import AgenticRouter

    client = PageIndexClient(db_path=str(tmp_path / "t.db"), search_backend="keyword")
    try:
        client.documents = {
            "d1": {
                "doc_name": "单文档", "doc_description": "", "type": "md",
                "structure": [{
                    "node_id": "0001", "title": "t", "summary": "s",
                    "text": "浴血内容", "span_kind": "line",
                    "line_num": 1, "end_line": 2, "nodes": [],
                }],
            }
        }

        calls = []

        async def fake_single(self, q, doc_id):
            calls.append(doc_id)
            return {"mode": "single"}

        client._search_single = types.MethodType(fake_single, client)

        router = AgenticRouter(client, "m")
        router.super_tree_index = MagicMock()  # truthy → 单链走 _search_super_tree

        async def fake_unified(q, top_k):
            calls.append(("unified", q))
            return {"mode": "multi"}

        router._search_super_tree = fake_unified
        client.router = router

        await client.search("浴血")

        # 统一链（super_tree）被调用；_search_single 未被调用
        assert calls == [("unified", "浴血")]
    finally:
        client.close()
