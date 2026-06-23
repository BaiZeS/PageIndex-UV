#!/usr/bin/env python3
"""W2 FR5 — orphan cleanup migration.

Removes child-table rows whose ``doc_id`` no longer exists in ``documents``
(nodes/pages/closet_tags/doc_keywords). Idempotent: on a clean DB it deletes
0 rows. Uses batched DELETE (``rowid IN (SELECT rowid ... LIMIT N)``) in a
loop to avoid long locks on large databases — SQLite does NOT support
``DELETE ... LIMIT N`` directly, so the rowid-subselect form is required.

Entry point::

    python -m pageindex_mutil.migrations.cleanup_orphans --db-path <path>
    python -m pageindex_mutil.migrations.cleanup_orphans --db-path <path> --purge-kb-identity

No schema changes (NFR3); pure DML, single transaction wrapping all 4 tables
for atomicity (spec §6.2).
"""

import argparse
import logging
import sqlite3
import sys

logger = logging.getLogger("pageindex_mutil.migrations.cleanup_orphans")

DEFAULT_DB_PATH = "workspace/pageindex.db"
BATCH_SIZE = 1000
CHILD_TABLES = ("nodes", "pages", "closet_tags", "doc_keywords")


def _cleanup_table(conn, table, batch_size=BATCH_SIZE):
    """Batched-delete orphan rows from one child table.

    Returns the total number of rows deleted.
    """
    total = 0
    while True:
        cur = conn.execute(
            f"DELETE FROM {table} WHERE rowid IN ("
            f"  SELECT rowid FROM {table} "
            f"  WHERE doc_id NOT IN (SELECT id FROM documents) "
            f"  LIMIT ?"
            f")",
            (batch_size,),
        )
        deleted = cur.rowcount if cur.rowcount is not None else 0
        total += deleted
        if deleted < batch_size:
            break
    return total


def run_cleanup(db_path, purge_kb_identity=False, batch_size=BATCH_SIZE):
    """Run the orphan cleanup against ``db_path``.

    Single transaction wrapping all 4 child tables (atomic). Optionally
    purges kb_identity to force a rebuild (cleans historical fenced values).
    Returns a dict of {table: rows_deleted}.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        results = {}
        with conn:
            for table in CHILD_TABLES:
                results[table] = _cleanup_table(conn, table, batch_size)
            if purge_kb_identity:
                cur = conn.execute("DELETE FROM kb_identity")
                results["kb_identity"] = cur.rowcount if cur.rowcount else 0
        logger.info("cleanup_orphans complete: %s", results)
        return results
    finally:
        conn.close()


def main(argv=None):
    """Run cleanup. Accepts either a parsed argv list OR a db_path string.

    - ``main("/path/to/index.db")`` — convenience: run cleanup on that db.
    - ``main(["--db-path", "/path", "--purge-kb-identity"])`` — full CLI.
    - ``main()`` — CLI with defaults (for ``python -m ...``).
    """
    if isinstance(argv, str):
        # Convenience: caller passed a bare db path.
        results = run_cleanup(argv)
        total = sum(results.values())
        print(f"Removed {total} orphan row(s): {results}")
        return 0

    parser = argparse.ArgumentParser(
        description=(
            "Remove orphan child rows (nodes/pages/closet_tags/doc_keywords) "
            "whose doc_id no longer exists in documents. Idempotent."
        )
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Path to pageindex.db (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--purge-kb-identity",
        action="store_true",
        help="Also DELETE FROM kb_identity (force rebuild; cleans fenced values)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"DELETE batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    results = run_cleanup(
        args.db_path,
        purge_kb_identity=args.purge_kb_identity,
        batch_size=args.batch_size,
    )
    total = sum(results.values())
    print(f"Removed {total} orphan row(s): {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
