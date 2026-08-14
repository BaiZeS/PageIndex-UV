"""T10: Two-phase indexing tests.

Phase 1 (synchronous): parse → DB insert → tags → keyword index → return fast.
Phase 2 (background): doc_summary → search backend → entity extraction.

All LLM calls mocked. No real LLM, no vectors.
"""
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

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


def _make_md_file(tmp_path, name="test.md"):
    """Create a temp markdown file."""
    p = tmp_path / name
    p.write_text("# Test\n\nHello world.\n")
    return str(p)


def _mock_md_to_tree():
    """Return a patched md_to_tree that returns a fixed structure."""
    return patch(
        "pageindex_mutil.client.md_to_tree",
        return_value={
            "doc_name": "test.md",
            "doc_description": "A test document",
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


def _phase2_barrier(barrier):
    """Return a side-effect function that signals a barrier after Phase 2 ops."""
    def _side_effect(*args, **kwargs):
        barrier.wait(timeout=5)
    return _side_effect


# ===========================================================================
# (a) Phase 1 returns fast with tags indexed
# ===========================================================================

class TestPhase1FastReturn:
    """Phase 1 should return doc_id without waiting for search backend or entity extraction."""

    def test_phase1_returns_doc_id(self, client_factory, tmp_path):
        """index() returns a valid doc_id immediately."""
        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            client.search_backend.index_document = MagicMock()
            client.entity_extractor = MagicMock()

            md_path = _make_md_file(tmp_path)
            with _mock_md_to_tree():
                doc_id = client.index(md_path, mode="md")

            assert doc_id is not None
            assert isinstance(doc_id, str)
            assert len(doc_id) > 0
        finally:
            client.close()

    def test_phase1_tags_indexed(self, client_factory, tmp_path):
        """Phase 1 completes closet_index.add_document (tag extraction)."""
        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            add_doc_mock = MagicMock()
            client.closet_index.add_document = add_doc_mock
            client.search_backend.index_document = MagicMock()
            client.entity_extractor = MagicMock()

            md_path = _make_md_file(tmp_path)
            with _mock_md_to_tree():
                doc_id = client.index(md_path, mode="md")

            add_doc_mock.assert_called_once()
        finally:
            client.close()

    def test_phase1_super_tree_indexed(self, client_factory, tmp_path):
        """Phase 1 completes super_tree.on_document_added (keyword index)."""
        client = client_factory()
        try:
            on_added_mock = MagicMock()
            client.super_tree_index.on_document_added = on_added_mock
            client.closet_index.add_document = MagicMock()
            client.search_backend.index_document = MagicMock()
            client.entity_extractor = MagicMock()

            md_path = _make_md_file(tmp_path)
            with _mock_md_to_tree():
                doc_id = client.index(md_path, mode="md")

            on_added_mock.assert_called_once()
        finally:
            client.close()

    def test_phase1_pending_enrichment_tracked(self, client_factory, tmp_path):
        """After Phase 1, doc's db_id is in _pending_enrichment set."""
        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            # Make Phase 2 hang so we can inspect the set
            barrier = threading.Event()
            client.search_backend.index_document = MagicMock(side_effect=lambda *a, **k: barrier.wait(timeout=10))
            client.search_backend.index_document = MagicMock()
            client.entity_extractor = MagicMock()

            md_path = _make_md_file(tmp_path)
            with _mock_md_to_tree():
                doc_id = client.index(md_path, mode="md", sync=False)

            # Phase 2 is running in background, doc should be in pending set
            db_doc_id = client._id_mapper.to_db(doc_id)
            assert db_doc_id in client._pending_enrichment

            # Let Phase 2 finish
            barrier.set()
            time.sleep(0.5)
        finally:
            client.close()


# ===========================================================================
# (b) Phase 2 runs in background and completes enrichment
# ===========================================================================

class TestPhase2Background:
    """Phase 2 should run in background and complete enrichment."""

    def test_phase2_search_backend_called(self, client_factory, tmp_path):
        """Phase 2 calls search_backend.index_document."""
        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            search_mock = MagicMock()
            client.search_backend.index_document = search_mock
            client.entity_extractor = MagicMock()

            md_path = _make_md_file(tmp_path)
            with _mock_md_to_tree():
                doc_id = client.index(md_path, mode="md")

            time.sleep(1)
            search_mock.assert_called_once()
        finally:
            client.close()

    def test_phase2_entity_extraction_called(self, client_factory, tmp_path):
        """Phase 2 calls entity_extractor.extract_from_document."""
        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            client.search_backend.index_document = MagicMock()
            extract_mock = MagicMock(return_value=([], [], []))
            client.entity_extractor.extract_from_document = extract_mock

            md_path = _make_md_file(tmp_path)
            with _mock_md_to_tree():
                doc_id = client.index(md_path, mode="md")

            time.sleep(1)
            extract_mock.assert_called_once()
        finally:
            client.close()

    def test_phase2_removes_from_pending(self, client_factory, tmp_path):
        """After Phase 2 completes, doc_id is removed from _pending_enrichment."""
        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            client.search_backend.index_document = MagicMock()
            client.entity_extractor = MagicMock()

            md_path = _make_md_file(tmp_path)
            with _mock_md_to_tree():
                doc_id = client.index(md_path, mode="md")

            db_doc_id = client._id_mapper.to_db(doc_id)

            # Wait for background thread to finish
            for _ in range(50):
                if db_doc_id not in client._pending_enrichment:
                    break
                time.sleep(0.1)

            assert db_doc_id not in client._pending_enrichment
        finally:
            client.close()

    def test_phase2_completes_while_phase1_already_returned(self, client_factory, tmp_path):
        """Phase 1 returns before Phase 2 finishes."""
        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            phase2_started = threading.Event()
            phase2_continue = threading.Event()

            def slow_search(*args, **kwargs):
                phase2_started.set()
                phase2_continue.wait(timeout=10)

            client.search_backend.index_document = slow_search
            client.entity_extractor = MagicMock()

            md_path = _make_md_file(tmp_path)
            with _mock_md_to_tree():
                t0 = time.monotonic()
                doc_id = client.index(md_path, mode="md", sync=False)
                elapsed = time.monotonic() - t0

            # Phase 1 should return quickly (well under 1s since all mocked)
            assert elapsed < 1.0
            # Phase 2 should be blocked
            phase2_started.wait(timeout=5)
            # Cleanup
            phase2_continue.set()
            time.sleep(0.5)
        finally:
            client.close()


# ===========================================================================
# (c) Phase 2 failure doesn't affect searchability
# ===========================================================================

class TestPhase2Failure:
    """Phase 2 failure must not crash or prevent searchability."""

    def test_phase2_entity_extraction_failure_logged(self, client_factory, tmp_path):
        """Phase 2 entity extraction failure is logged, not raised."""
        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            client.search_backend.index_document = MagicMock()
            client.entity_extractor.extract_from_document = MagicMock(
                side_effect=RuntimeError("entity boom")
            )

            md_path = _make_md_file(tmp_path)
            with _mock_md_to_tree():
                doc_id = client.index(md_path, mode="md")

            assert doc_id is not None
            time.sleep(1)
        finally:
            client.close()

    def test_phase2_search_backend_failure_logged(self, client_factory, tmp_path):
        """Phase 2 search backend failure is logged, not raised."""
        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            client.search_backend.index_document = MagicMock(
                side_effect=RuntimeError("search boom")
            )
            client.entity_extractor = MagicMock()

            md_path = _make_md_file(tmp_path)
            with _mock_md_to_tree():
                doc_id = client.index(md_path, mode="md")

            assert doc_id is not None
            time.sleep(1)
        finally:
            client.close()

    def test_pending_set_cleaned_on_failure(self, client_factory, tmp_path):
        """Pending enrichment set is cleaned even if Phase 2 fails."""
        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            client.search_backend.index_document = MagicMock(
                side_effect=RuntimeError("boom")
            )
            client.entity_extractor = MagicMock()

            md_path = _make_md_file(tmp_path)
            with _mock_md_to_tree():
                doc_id = client.index(md_path, mode="md")

            db_doc_id = client._id_mapper.to_db(doc_id)
            # Wait for background thread
            for _ in range(50):
                if db_doc_id not in client._pending_enrichment:
                    break
                time.sleep(0.1)
            assert db_doc_id not in client._pending_enrichment
        finally:
            client.close()


# ===========================================================================
# (d) sync=True preserves current behavior
# ===========================================================================

class TestSyncMode:
    """sync=True should run Phase 2 synchronously (current behavior)."""

    def test_sync_true_runs_entity_extraction_synchronously(self, client_factory, tmp_path):
        """With sync=True, entity extraction runs before index() returns."""
        call_order = []

        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            client.search_backend.index_document = MagicMock()

            def track_entity(*args, **kwargs):
                call_order.append("entity_extraction")
                return ([], [], [])

            client.entity_extractor.extract_from_document = track_entity

            md_path = _make_md_file(tmp_path)
            with _mock_md_to_tree():
                call_order.append("index_start")
                doc_id = client.index(md_path, mode="md", sync=True)
                call_order.append("index_end")

            assert call_order.index("entity_extraction") > call_order.index("index_start")
            assert call_order.index("entity_extraction") < call_order.index("index_end")
        finally:
            client.close()

    def test_sync_true_no_pending_enrichment(self, client_factory, tmp_path):
        """With sync=True, no entries remain in _pending_enrichment after index()."""
        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            client.search_backend.index_document = MagicMock()
            client.entity_extractor = MagicMock()

            md_path = _make_md_file(tmp_path)
            with _mock_md_to_tree():
                doc_id = client.index(md_path, mode="md", sync=True)

            db_doc_id = client._id_mapper.to_db(doc_id)
            assert db_doc_id not in client._pending_enrichment
        finally:
            client.close()

    def test_default_is_sync(self, client_factory, tmp_path):
        """Default (no sync param) should be synchronous (backward-compatible)."""
        call_order = []

        client = client_factory()
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()

            def slow_search(*args, **kwargs):
                time.sleep(0.3)
                call_order.append("search_backend")

            client.search_backend.index_document = slow_search
            client.entity_extractor = MagicMock()

            md_path = _make_md_file(tmp_path)
            with _mock_md_to_tree():
                call_order.append("index_start")
                t0 = time.monotonic()
                doc_id = client.index(md_path, mode="md")
                elapsed = time.monotonic() - t0
                call_order.append("index_end")

            # index() should wait for Phase 2 since default is sync=True
            assert elapsed >= 0.3
            assert call_order.index("search_backend") < call_order.index("index_end")
        finally:
            client.close()
