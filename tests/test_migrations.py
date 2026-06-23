"""W2 FR5 — orphan cleanup migration tests.

Exercises pageindex_mutil.migrations.cleanup_orphans: removes child-table rows
whose doc_id no longer exists in documents (nodes/pages/closet_tags/doc_keywords),
using batched DELETE (rowid IN (...) LIMIT N) to avoid long locks on large DBs.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from db import PageIndexDB


@pytest.fixture
def tmp_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = PageIndexDB(path)
    db.close()
    yield path
    if os.path.exists(path):
        os.unlink(path)
    # WAL/shm sidecar files
    for suffix in ("-wal", "-shm"):
        side = path + suffix
        if os.path.exists(side):
            os.unlink(side)


def _count_orphans(db_path, table):
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            f"SELECT COUNT(*) FROM {table} "
            "WHERE doc_id NOT IN (SELECT id FROM documents)"
        ).fetchone()[0]
    finally:
        conn.close()


def _insert_orphans(db_path, table, doc_id, count):
    """Insert `count` orphan rows into `table` with the given doc_id."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        for _ in range(count):
            if table == "nodes":
                conn.execute(
                    "INSERT INTO nodes (doc_id, node_id) VALUES (?, ?)",
                    (doc_id, f"orphan-{os.urandom(4).hex()}"),
                )
            elif table == "pages":
                conn.execute(
                    "INSERT INTO pages (doc_id, page_number, content) VALUES (?, ?, ?)",
                    (doc_id, _next_page_number(conn, doc_id), "x"),
                )
            elif table == "closet_tags":
                conn.execute(
                    "INSERT INTO closet_tags (doc_id, tag_text, tag_token, confidence, source) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (doc_id, "t", "tok", 0.5, "manual"),
                )
            elif table == "doc_keywords":
                conn.execute(
                    "INSERT INTO doc_keywords (doc_id, keyword, field) VALUES (?, ?, ?)",
                    (doc_id, f"kw-{os.urandom(4).hex()}", "name"),
                )
        conn.commit()
    finally:
        conn.close()


def _next_page_number(conn, doc_id):
    row = conn.execute(
        "SELECT COALESCE(MAX(page_number), 0) + 1 FROM pages WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    return row[0]


CHILD_TABLES = ["nodes", "pages", "closet_tags", "doc_keywords"]


# ---------------------------------------------------------------------------
# T5 / RED Task #14 — migration script does not exist / orphans not cleaned
# ---------------------------------------------------------------------------


class TestCleanupOrphans:
    def test_cleanup_orphans_removes_orphans(self, tmp_db_path):
        """W2 FR5/AC5.1 — migration removes orphan rows from all 4 child tables."""
        from pageindex_mutil.migrations import cleanup_orphans

        orphan_doc_id = 999999  # not in documents
        for table in CHILD_TABLES:
            _insert_orphans(tmp_db_path, table, orphan_doc_id, 3)
            assert _count_orphans(tmp_db_path, table) == 3

        cleanup_orphans.main(tmp_db_path)

        for table in CHILD_TABLES:
            assert _count_orphans(tmp_db_path, table) == 0, (
                f"orphans remain in {table}"
            )

    def test_cleanup_orphans_idempotent(self, tmp_db_path):
        """W2 FR5/AC5.2 — running twice deletes 0 rows the second time, no error."""
        from pageindex_mutil.migrations import cleanup_orphans

        orphan_doc_id = 888888
        for table in CHILD_TABLES:
            _insert_orphans(tmp_db_path, table, orphan_doc_id, 2)

        cleanup_orphans.main(tmp_db_path)
        # Second run on a now-clean DB: no orphans, no error.
        cleanup_orphans.main(tmp_db_path)
        for table in CHILD_TABLES:
            assert _count_orphans(tmp_db_path, table) == 0

    def test_cleanup_orphans_batched_large(self, tmp_db_path):
        """W2 R1 — batched DELETE terminates for >1000 orphan rows."""
        from pageindex_mutil.migrations import cleanup_orphans

        orphan_doc_id = 777777
        # Insert > batch size (1000) orphans into one table to exercise the loop.
        _insert_orphans(tmp_db_path, "doc_keywords", orphan_doc_id, 1500)
        assert _count_orphans(tmp_db_path, "doc_keywords") == 1500

        cleanup_orphans.main(tmp_db_path)

        assert _count_orphans(tmp_db_path, "doc_keywords") == 0
