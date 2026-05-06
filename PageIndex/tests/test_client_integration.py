import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from db import PageIndexDB


class TestPageIndexClientSuperTree:
    def test_super_tree_index_initialized_with_db(self):
        """PageIndexClient with db_path should initialize super_tree_index."""
        # We need to mock PyPDF2 since client.py imports it at top level
        sys.modules["PyPDF2"] = MagicMock()

        from PageIndex.pageindex.client import PageIndexClient

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

        from PageIndex.pageindex.client import PageIndexClient

        client = PageIndexClient()
        assert hasattr(client, "super_tree_index")
        assert client.super_tree_index is None
        assert client.router is None

    def test_on_document_added_called_during_index(self):
        """index() should call super_tree_index.on_document_added after DB insert."""
        sys.modules["PyPDF2"] = MagicMock()

        from PageIndex.pageindex.client import PageIndexClient

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
                with patch("PageIndex.pageindex.client.md_to_tree") as mock_md:
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
