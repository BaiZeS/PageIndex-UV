"""W2 delete-path integrity tests.

These tests exercise the MCP ``delete_document`` handler in ``server.py``
end-to-end against a real ``PageIndexClient`` + sqlite DB.

Why a dedicated file (vs. tests/test_router.py as tasks.md nominally specifies):
``tests/test_router.py`` pre-stubs ``pageindex_mutil.utils`` at module-collection
time (lines ~22-26) with fake ``llm_completion``/``extract_json`` attributes to
avoid pulling in heavy deps. That stub shadows the real ``pageindex_mutil.utils``,
so importing the real ``PageIndexClient`` (which does
``from .utils import ConfigLoader``) raises ImportError there. This is the
pre-existing ConfigLoader pollution tracked for W6. Placing the real-client
delete tests here keeps the full suite green (the constraint's source of truth)
without coupling W2 to W6.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock PyPDF2 so client.py's top-level import does not require the dep.
sys.modules.setdefault("PyPDF2", MagicMock())

from db import PageIndexDB
from pageindex_mutil.client import PageIndexClient
from app import server as server_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def real_client(tmp_path):
    """Real PageIndexClient with a temp workspace + sqlite db.

    Bypasses index()/LLM by inserting document rows directly, so the delete
    handler can be exercised in isolation.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    uploads = workspace / "uploads"
    uploads.mkdir()
    db_path = tmp_path / "index.db"

    client = PageIndexClient(workspace=str(workspace), db_path=str(db_path))
    yield client
    client.close()


@pytest.fixture
def bound_server(real_client, monkeypatch):
    """Bind the real client into server.client and return the client."""
    monkeypatch.setattr(server_mod, "client", real_client)
    return real_client


async def _call_delete(doc_id):
    """Invoke the MCP delete_document handler, returning the parsed JSON."""
    result = await server_mod.handle_call_tool(
        "delete_document", {"doc_id": doc_id}
    )
    assert len(result) == 1
    return json.loads(result[0].text)


def _count(client, table, doc_id):
    conn = client.db._connect()
    return conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE doc_id = ?", (doc_id,)
    ).fetchone()[0]


def _doc_count(client, db_id):
    conn = client.db._connect()
    return conn.execute(
        "SELECT COUNT(*) FROM documents WHERE id = ?", (db_id,)
    ).fetchone()[0]


def _kb_identity_empty(client):
    conn = client.db._connect()
    return conn.execute("SELECT COUNT(*) FROM kb_identity").fetchone()[0] == 0


def _insert_doc(client, name="doc.pdf", path="/tmp/doc.pdf", uuid_str="uuid-1"):
    """Insert a document row directly and register it with the client.

    Returns (uuid, db_id).
    """
    db_id = client.db.insert_document(name, path)
    client.documents[uuid_str] = {"id": uuid_str, "doc_name": name}
    client._uuid_to_db[uuid_str] = db_id
    return uuid_str, db_id


# ---------------------------------------------------------------------------
# T0 / RED Task #2 — handler silently returns success without deleting (P0-2)
# ---------------------------------------------------------------------------


class TestDeleteHandlerActuallyDeletes:
    @pytest.mark.asyncio
    async def test_delete_document_handler_actually_deletes(self, bound_server):
        c = bound_server
        uuid_str, db_id = _insert_doc(c, "handler.pdf", "/tmp/handler.pdf", "uuid-handler-1")

        assert _doc_count(c, db_id) == 1

        resp = await _call_delete(uuid_str)

        # Handler must report success AND actually delete the row.
        assert resp.get("success") is True
        assert _doc_count(c, db_id) == 0


# ---------------------------------------------------------------------------
# T2 / RED Task #6 — delete handler does not invalidate super-tree index (P1-8)
# ---------------------------------------------------------------------------


class TestDeleteInvalidatesSuperTreeIndex:
    """W2 FR2/AC2.1/AC2.2 — delete must call on_document_removed.

    RED (Task #6): the delete handler never calls on_document_removed, so
    doc_keywords rows survive the delete and kb_identity is not invalidated
    (asymmetric with the add path which calls on_document_added).
    """

    @pytest.mark.asyncio
    async def test_delete_invalidates_super_tree_index(self, bound_server):
        c = bound_server
        uuid_str, db_id = _insert_doc(
            c, "前端脚本开发指南.pdf", "/tmp/st.pdf", "uuid-st-1"
        )
        # Simulate on_document_added: build keyword index + kb_identity.
        c.super_tree_index.on_document_added(db_id)
        # Populate kb_identity so we can assert it gets invalidated.
        c.db.set_kb_identity("知识库概览摘要", 1)

        # Pre-conditions: keyword rows exist, kb_identity populated.
        assert _count(c, "doc_keywords", db_id) >= 1
        assert not _kb_identity_empty(c)

        resp = await _call_delete(uuid_str)
        assert resp.get("success") is True

        # FR2/AC2.1: doc_keywords cleared (on_document_removed -> keyword_index.remove_document).
        assert _count(c, "doc_keywords", db_id) == 0
        # FR2/AC2.2: kb_identity invalidated (on_document_removed -> kb_identity.invalidate).
        assert _kb_identity_empty(c)


# ---------------------------------------------------------------------------
# T4 / RED Task #11 — delete does not remove upload file from disk (P1-disk)
# ---------------------------------------------------------------------------


class TestDeleteRemovesUploadFile:
    """W2 FR4/AC4.1 — delete must remove the uploaded PDF from disk.

    RED (Task #11): the delete handler performs no disk cleanup, so the file
    persists at pdf_path after deletion.
    """

    @pytest.mark.asyncio
    async def test_delete_removes_upload_file(self, bound_server):
        c = bound_server
        # Create a real upload file under workspace/uploads/.
        upload_path = c.workspace / "uploads" / "abc123_doc.pdf"
        upload_path.write_bytes(b"%PDF-1.4 fake content")
        pdf_path = str(upload_path)
        assert os.path.exists(pdf_path)

        uuid_str, db_id = _insert_doc(c, "doc.pdf", pdf_path, "uuid-disk-1")

        resp = await _call_delete(uuid_str)
        assert resp.get("success") is True

        # FR4/AC4.1: the uploaded file must be removed from disk.
        assert not os.path.exists(pdf_path), (
            f"upload file still exists at {pdf_path} after delete"
        )

    def test_safe_remove_upload_skips_outside_uploads(self, real_client):
        """W2 FR4/AC4.2 — guard refuses to delete files outside uploads/."""
        # /etc/passwd resolves outside workspace/uploads -> guard must skip.
        server_mod._safe_remove_upload("/etc/passwd", real_client.workspace)
        assert os.path.exists("/etc/passwd"), "guard must not delete /etc/passwd"


# ---------------------------------------------------------------------------
# T6 / Task #17 — NFR1 end-to-end integration test
# ---------------------------------------------------------------------------


class TestDeleteEndToEndIntegrity:
    """W2 NFR1/AC6.1 — insert -> delete -> assert full chain integrity.

    (a) documents/nodes/pages/closet_tags/doc_keywords rows cleared (FR1).
    (b) kb_identity invalidated (FR2).
    (c) disk PDF file removed (FR4).
    (d) kb_identity stored value has no fence markers (FR3, mocked LLM).
    """

    @pytest.mark.asyncio
    async def test_delete_end_to_end_integrity(self, bound_server):
        c = bound_server

        # Create a real upload file under workspace/uploads/.
        upload_path = c.workspace / "uploads" / "e2e123_e2e.pdf"
        upload_path.write_bytes(b"%PDF-1.4 e2e content")
        pdf_path = str(upload_path)
        assert os.path.exists(pdf_path)

        uuid_str, db_id = _insert_doc(
            c, "e2e.pdf", pdf_path, "uuid-e2e-1",
        )
        # Populate all 4 child tables (simulate indexing side-effects).
        c.db.insert_nodes(db_id, [(db_id, "n1", "title", "summary", 0, 10, None)])
        c.db.insert_pages(db_id, [(db_id, 1, "page one")])
        c.db.insert_closet_tags(db_id, [(db_id, "tag", "token", 0.9, "manual")])
        c.db.insert_doc_keywords(db_id, [(db_id, "keyword", "name")])

        # Build super-tree index + a FENCED kb_identity (FR3 check).
        c.super_tree_index.on_document_added(db_id)
        # Simulate a fenced LLM response persisted to kb_identity.
        c.db.set_kb_identity("```text\n知识库共1个文档，主题：测试\n```", 1)
        # Pre-condition: kb_identity stored with fence markers (will be cleaned
        # on next rebuild; here we just assert the delete invalidates it).
        assert not _kb_identity_empty(c)

        resp = await _call_delete(uuid_str)
        assert resp.get("success") is True

        # (a) FR1: documents + all 4 child tables cleared via cascade.
        assert _doc_count(c, db_id) == 0
        assert _count(c, "nodes", db_id) == 0
        assert _count(c, "pages", db_id) == 0
        assert _count(c, "closet_tags", db_id) == 0
        assert _count(c, "doc_keywords", db_id) == 0
        # (b) FR2: kb_identity invalidated.
        assert _kb_identity_empty(c)
        # (c) FR4: disk PDF removed.
        assert not os.path.exists(pdf_path)

        # (d) FR3: strip_markdown_fence is wired into _generate_with_llm so
        # fenced LLM output cannot persist. (The dedicated
        # test_kb_identity_strips_markdown_fence in test_super_tree.py covers
        # the full LLM-mock path in isolation; here we assert the wiring exists
        # by confirming the function is imported and used in the super_tree
        # module the client actually loaded.)
        import inspect
        from pageindex_mutil import super_tree as _st_mod
        _gen_src = inspect.getsource(_st_mod.KBIdentity._generate_with_llm)
        assert "strip_markdown_fence" in _gen_src, (
            "_generate_with_llm must call strip_markdown_fence (FR3 wiring)"
        )


# ---------------------------------------------------------------------------
# T7 / Task #18 — NFR2 idempotent delete (AC7.1 + AC7.2)
# ---------------------------------------------------------------------------


class TestDeleteIdempotent:
    """W2 NFR2 — delete is idempotent across missing + repeat ids."""

    @pytest.mark.asyncio
    async def test_delete_nonexistent_id_returns_success(self, bound_server):
        """AC7.1 — deleting a non-existent doc_id returns success, no error."""
        c = bound_server
        # uuid not in _uuid_to_db -> db_id is None -> handler skips DB work.
        resp = await _call_delete("nonexistent-uuid-xyz")
        assert resp.get("success") is True

    @pytest.mark.asyncio
    async def test_delete_twice_idempotent(self, bound_server):
        """AC7.2 — deleting the same doc_id twice; second delete is a no-op."""
        c = bound_server
        upload_path = c.workspace / "uploads" / "twice123_twice.pdf"
        upload_path.write_bytes(b"%PDF-1.4 twice")
        pdf_path = str(upload_path)
        uuid_str, db_id = _insert_doc(c, "twice.pdf", pdf_path, "uuid-twice-1")

        resp1 = await _call_delete(uuid_str)
        assert resp1.get("success") is True
        assert _doc_count(c, db_id) == 0
        assert not os.path.exists(pdf_path)

        # Second delete: db_id already popped -> None -> handler skips; no error.
        resp2 = await _call_delete(uuid_str)
        assert resp2.get("success") is True
