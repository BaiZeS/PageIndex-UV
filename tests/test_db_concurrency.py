"""Tests for DB concurrency hardening (W1).

Covers spec §8 acceptance criteria:
- FR1: multi-thread asyncio.to_thread safety (no ProgrammingError / corruption)
- FR2: WAL + synchronous=NORMAL + busy_timeout
- FR3: 3 indexes exist (idx_nodes_doc_id, idx_nodes_parent_node_id, idx_pages_doc_id)
- NFR1: concurrent upload + concurrent search integration
- NFR2: indexes transparent to existing data
- NFR3: public method signatures unchanged
- NFR4: WAL + index disk overhead <= 1.5x db size

Design (spec §4.3, chosen): Plan B — thread-local connection pool (threading.local)
+ WAL + synchronous=NORMAL + busy_timeout=5000ms + foreign_keys=ON + row_factory=Row.
"""

import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from db import PageIndexDB


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db_path(tmp_path):
    """Return a path for a temp db file (parent dir created by tmp_path)."""
    return str(tmp_path / "test_concurrency.db")


def _make_nodes(doc_id, count):
    """Build `count` node records (doc_id, node_id, title, summary, start, end, parent)."""
    return [
        (doc_id, f"node-{i}", f"Title {i}", f"Summary {i}", i, i + 1, None)
        for i in range(count)
    ]


def _make_pages(doc_id, count):
    """Build `count` page records (doc_id, page_number, content)."""
    return [(doc_id, i, f"page content {i}") for i in range(count)]


def _index_one(db, name, n_nodes=50, n_pages=20):
    """Insert one document + its nodes + pages (simulates index path)."""
    doc_id = db.insert_document(name, f"/tmp/{name}")
    db.insert_nodes(doc_id, _make_nodes(doc_id, n_nodes))
    db.insert_pages(doc_id, _make_pages(doc_id, n_pages))
    return doc_id


# ---------------------------------------------------------------------------
# T1 — RED: concurrent upload + search defect reproduction (proves P0-1 root cause)
# ---------------------------------------------------------------------------


def test_concurrent_upload_and_search_no_crash(tmp_db_path):
    """3 concurrent uploads + concurrent reads on the SAME PageIndexDB instance.

    With the current single cached connection shared across threads, this must
    crash (sqlite3.ProgrammingError) or corrupt data. After the W1 thread-local
    refactor, it must pass cleanly.
    """
    db = PageIndexDB(tmp_db_path)
    try:
        names = [f"doc_{i}.pdf" for i in range(3)]

        def upload(name):
            return _index_one(db, name)

        def reader():
            # Simulate _run_strategies read path: read while uploads happen.
            docs = db.get_all_documents()
            for d in docs:
                db.get_nodes_by_doc_id(d["id"])
            return len(docs)

        with ThreadPoolExecutor(max_workers=3) as pool:
            upload_futs = [pool.submit(upload, n) for n in names]
            read_futs = [pool.submit(reader) for _ in range(3)]
            uploaded_ids = [f.result() for f in upload_futs]
            [_ for f in read_futs for _ in [f.result()]]

        # Consistency assertions
        all_docs = db.get_all_documents()
        assert len(all_docs) == 3, f"expected 3 documents, got {len(all_docs)}"
        for doc_id, name in zip(sorted(uploaded_ids), names):
            nodes = db.get_nodes_by_doc_id(doc_id)
            assert len(nodes) == 50, f"doc {name}: expected 50 nodes, got {len(nodes)}"
        # foreign_key_check returns rows for violations; empty == OK
        conn = db._connect()
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        assert violations == [], f"foreign key violations: {violations}"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# T2 — RED: thread-local connection reuse + cleanup contract (locks T3 behavior)
# ---------------------------------------------------------------------------


def test_same_thread_reuses_connection(tmp_db_path):
    """Same thread calling _connect() twice must return the SAME connection object."""
    db = PageIndexDB(tmp_db_path)
    try:
        c1 = db._connect()
        c2 = db._connect()
        assert c1 is c2, "same thread must reuse the same connection (not new each call)"
    finally:
        db.close()


def test_different_threads_get_different_connections(tmp_db_path):
    """Two different threads must each get DISTINCT connections (thread isolation)."""
    db = PageIndexDB(tmp_db_path)
    try:
        results = {}

        def grab(key):
            results[key] = db._connect()

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(grab, "main_thread_conn_via_worker")
            # also grab one from another worker
            f2 = pool.submit(grab, "other_worker")
            f1.result()
            f2.result()

        # Two different worker threads -> two different connections
        conns = list(results.values())
        assert len(conns) == 2
        assert conns[0] is not conns[1], "different threads must get different connections"
    finally:
        db.close()


def test_close_closes_all_thread_local_connections(tmp_db_path):
    """close() must close ALL registered thread-local connections (R1/R3 leak guard)."""
    db = PageIndexDB(tmp_db_path)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = [pool.submit(db._connect) for _ in range(3)]
            [f.result() for f in futs]
        # After workers ran, _tls_connections should hold the 3 worker conns
        # (plus possibly the main-thread conn from ensure_schema)
        registered = list(db._tls_connections)
        assert len(registered) >= 3, (
            f"expected >=3 registered thread-local connections, got {len(registered)}"
        )
        db.close()
        # After close(), every registered connection must be closed
        for c in registered:
            assert _is_closed(c), f"connection {c} was not closed by db.close()"
        # registry cleared
        assert db._tls_connections == [], "registry not cleared after close()"
    finally:
        # db.close() already called above; ensure idempotent
        db.close()


def _is_closed(conn):
    """Return True if a sqlite3 connection is closed."""
    try:
        conn.execute("SELECT 1").fetchone()
        return False
    except Exception:
        return True


# ---------------------------------------------------------------------------
# T4 — GREEN: WAL + pragma assertions (FR2)
# ---------------------------------------------------------------------------


def test_pragmas_set_correctly(tmp_db_path):
    """FR2: journal_mode=wal, synchronous=1(NORMAL), busy_timeout=5000, foreign_keys=1."""
    db = PageIndexDB(tmp_db_path)
    try:
        conn = db._connect()
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        db.close()


def test_pragmas_set_on_each_thread_local_connection(tmp_db_path):
    """FR2: pragmas must be set on EVERY thread-local connection, not just the first."""
    db = PageIndexDB(tmp_db_path)
    try:
        results = {}

        def check_pragmas(key):
            conn = db._connect()
            results[key] = {
                "journal": conn.execute("PRAGMA journal_mode").fetchone()[0].lower(),
                "sync": conn.execute("PRAGMA synchronous").fetchone()[0],
                "busy": conn.execute("PRAGMA busy_timeout").fetchone()[0],
                "fk": conn.execute("PRAGMA foreign_keys").fetchone()[0],
            }

        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(check_pragmas, f"t{i}") for i in range(2)]
            [f.result() for f in futs]

        for key, p in results.items():
            assert p["journal"] == "wal", f"{key}: journal_mode={p['journal']}"
            assert p["sync"] == 1, f"{key}: synchronous={p['sync']}"
            assert p["busy"] == 5000, f"{key}: busy_timeout={p['busy']}"
            assert p["fk"] == 1, f"{key}: foreign_keys={p['fk']}"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# T5 — GREEN: indexes in ensure_schema single landing (FR3 + quality-gate #1)
# ---------------------------------------------------------------------------

_INDEX_NAMES = {
    "idx_nodes_doc_id",
    "idx_nodes_parent_node_id",
    "idx_pages_doc_id",
    "idx_closet_tags_token",
    "idx_doc_keywords",
}


def test_indexes_exist(tmp_db_path):
    """FR3: the 3 new indexes exist (+ existing idx_closet_tags_token, idx_doc_keywords)."""
    db = PageIndexDB(tmp_db_path)
    try:
        conn = db._connect()
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        ).fetchall()
        present = {r[0] for r in rows}
        missing = _INDEX_NAMES - present
        assert not missing, f"missing indexes: {missing}; present={present}"
    finally:
        db.close()


def test_indexes_created_via_ensure_schema_single_landing():
    """quality-gate #1: NO _ensure_indexes method; index DDL lives in ensure_schema.

    Guards against a dual-landing _ensure_indexes(conn) called from _connect that
    would double-execute index creation alongside ensure_schema.
    """
    import inspect

    import db as db_module

    # 1. _ensure_indexes must NOT exist on PageIndexDB
    assert not hasattr(PageIndexDB, "_ensure_indexes"), (
        "_ensure_indexes must not exist; indexes must live in ensure_schema's "
        "executescript (single landing, quality-gate #1)"
    )
    # 2. The 3 new index DDLs must be inside the ensure_schema source
    src = inspect.getsource(PageIndexDB.ensure_schema)
    for ddl_marker in (
        "CREATE INDEX IF NOT EXISTS idx_nodes_doc_id",
        "CREATE INDEX IF NOT EXISTS idx_nodes_parent_node_id",
        "CREATE INDEX IF NOT EXISTS idx_pages_doc_id",
    ):
        assert ddl_marker in src, f"index DDL not in ensure_schema source: {ddl_marker}"


# ---------------------------------------------------------------------------
# T6 — GREEN: indexes transparent to existing data (NFR2)
# ---------------------------------------------------------------------------


def test_indexes_transparent_to_existing_data(tmp_db_path):
    """NFR2: opening an existing db with data auto-creates indexes, no data loss."""
    # Pre-populate with data via a first instance
    db1 = PageIndexDB(tmp_db_path)
    try:
        doc_id = db1.insert_document("existing.pdf", "/tmp/existing.pdf")
        db1.insert_nodes(doc_id, _make_nodes(doc_id, 2))
        db1.insert_pages(doc_id, _make_pages(doc_id, 2))
    finally:
        db1.close()

    # Open a SECOND instance pointing at the same db file
    db2 = PageIndexDB(tmp_db_path)
    try:
        # Indexes auto-created (IF NOT EXISTS)
        conn = db2._connect()
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        ).fetchall()
        present = {r[0] for r in rows}
        assert _INDEX_NAMES <= present, f"missing indexes on reopen: {_INDEX_NAMES - present}"

        # Data intact: row counts unchanged
        docs = db2.get_all_documents()
        assert len(docs) == 1
        nodes = db2.get_nodes_by_doc_id(doc_id)
        assert len(nodes) == 2
        # pages via get_pages_in_range (page numbers 0,1)
        pages = db2.get_pages_in_range(doc_id, 0, 1)
        assert len(pages) == 2
    finally:
        db2.close()


# ---------------------------------------------------------------------------
# T7 — GREEN: concurrent integration tests (NFR1)
# ---------------------------------------------------------------------------


def test_concurrent_upload_foreign_key_integrity(tmp_db_path):
    """NFR1: 3 concurrent uploads leave no foreign key violations."""
    db = PageIndexDB(tmp_db_path)
    try:
        names = [f"fk_doc_{i}.pdf" for i in range(3)]

        def upload(name):
            return _index_one(db, name)

        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = [pool.submit(upload, n) for n in names]
            [f.result() for f in futs]

        conn = db._connect()
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        assert violations == [], f"foreign key violations: {violations}"
        assert len(db.get_all_documents()) == 3
    finally:
        db.close()


def test_concurrent_upload_and_search_data_consistency(tmp_db_path):
    """NFR1: 3 concurrent uploads of different files + concurrent reads stay consistent.

    Each document's nodes/pages row counts must match what was inserted.
    """
    db = PageIndexDB(tmp_db_path)
    try:
        # Different sizes per doc to catch cross-contamination
        specs = [("a.pdf", 50, 20), ("b.pdf", 30, 10), ("c.pdf", 70, 5)]
        expected = {}

        def upload(spec):
            name, n_nodes, n_pages = spec
            doc_id = _index_one(db, name, n_nodes=n_nodes, n_pages=n_pages)
            return name, doc_id, n_nodes, n_pages

        def reader():
            seen = []
            for d in db.get_all_documents():
                seen.append((d["id"], len(db.get_nodes_by_doc_id(d["id"]))))
            return seen

        with ThreadPoolExecutor(max_workers=3) as pool:
            up_futs = [pool.submit(upload, s) for s in specs]
            rd_futs = [pool.submit(reader) for _ in range(4)]
            for f in up_futs:
                name, doc_id, n_nodes, n_pages = f.result()
                expected[name] = (doc_id, n_nodes, n_pages)
            [_ for f in rd_futs for _ in [f.result()]]

        # Final consistency check (after all writes settled)
        assert len(db.get_all_documents()) == 3
        for name, (doc_id, n_nodes, n_pages) in expected.items():
            nodes = db.get_nodes_by_doc_id(doc_id)
            assert len(nodes) == n_nodes, (
                f"{name}: expected {n_nodes} nodes, got {len(nodes)}"
            )
            pages = db.get_pages_in_range(doc_id, 0, n_pages - 1)
            assert len(pages) == n_pages, (
                f"{name}: expected {n_pages} pages, got {len(pages)}"
            )
        conn = db._connect()
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        db.close()


# ---------------------------------------------------------------------------
# T9 — busy_timeout write latency benchmark (quality-gate #2)
# ---------------------------------------------------------------------------


def test_busy_timeout_covers_3x_write_latency(tmp_db_path, capsys):
    """quality-gate #2: busy_timeout(5000ms) >= 3 * single-write latency.

    Measures the median of 10 single writes (1 doc + ~50 nodes + ~20 pages),
    prints the benchmark, and asserts 5000 >= 3 * t_write_median. If this
    fails, busy_timeout must be raised (and spec §6.1 / tasks updated).
    """
    import statistics
    import time

    db = PageIndexDB(tmp_db_path)
    try:
        timings = []
        for i in range(10):
            t0 = time.perf_counter()
            _index_one(db, f"bench_{i}.pdf", n_nodes=50, n_pages=20)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # ms

        t_median_ms = statistics.median(timings)
        three_x = 3 * t_median_ms
        busy_timeout_ms = 5000
        with capsys.disabled():
            print(
                f"\n[T9 benchmark] t_write_median={t_median_ms:.2f}ms "
                f"3x={three_x:.2f}ms busy_timeout={busy_timeout_ms}ms "
                f"verdict={'PASS' if busy_timeout_ms >= three_x else 'NEEDS_ADJUSTMENT'}"
            )
        assert busy_timeout_ms >= three_x, (
            f"busy_timeout={busy_timeout_ms}ms < 3*t_write_median={three_x:.2f}ms; "
            "raise _BUSY_TIMEOUT_MS and update spec §6.1"
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# T10 — GREEN: disk overhead quantification (NFR4)
# ---------------------------------------------------------------------------


def test_disk_overhead_within_threshold(tmp_path, capsys):
    """NFR4: WAL + indexes total disk overhead <= 1.5x pure db size.

    Inserts 5 documents (each ~50 nodes + ~20 pages), checkpoints the WAL,
    then compares (db + wal + shm) total against db-only size.
    """
    db_path = str(tmp_path / "disk_test.db")
    db = PageIndexDB(db_path)
    try:
        for i in range(5):
            _index_one(db, f"disk_{i}.pdf", n_nodes=50, n_pages=20)
        # Checkpoint WAL into the main db so steady-state sizes are measured
        conn = db._connect()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    finally:
        db.close()

    def _size(suffix):
        p = db_path + suffix
        return os.path.getsize(p) if os.path.exists(p) else 0

    db_only = _size("")  # main db file
    wal = _size("-wal")
    shm = _size("-shm")
    total = db_only + wal + shm
    ratio = total / db_only if db_only else 0
    with capsys.disabled():
        print(
            f"\n[T10 disk] db_only={db_only/1024:.1f}KB wal={wal/1024:.1f}KB "
            f"shm={shm/1024:.1f}KB total={total/1024:.1f}KB ratio={ratio:.3f}"
        )
    assert ratio <= 1.5, (
        f"disk overhead ratio {ratio:.3f} > 1.5 "
        f"(db={db_only} wal={wal} shm={shm})"
    )
