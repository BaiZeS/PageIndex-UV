import os
import sqlite3
import json
import threading


SQLITE_MAX_VARIABLE_NUMBER = 999

# WAL + synchronous=NORMAL + busy_timeout (spec §6.1).
# busy_timeout=5000ms covers ~3 concurrent uploads queueing depth (spec §5.2).
_BUSY_TIMEOUT_MS = 5000


class PageIndexDB:
    def __init__(self, db_path):
        self.db_path = db_path
        # Plan B (spec §4.3): thread-local connection pool. Each worker thread
        # gets its own connection via threading.local(); connections are
        # registered in _tls_connections so close() can iterate and close them
        # all (R1/R3 leak guard). check_same_thread=False only silences
        # sqlite3's default cross-thread guard — actual isolation is guaranteed
        # by thread-local ownership (one connection per thread).
        self._local = threading.local()
        self._tls_connections = []
        self._tls_lock = threading.Lock()
        # _conn kept as a backwards-compatible alias to the main-thread
        # thread-local connection (no caller relies on it, but it avoids
        # surprising AttributeError on legacy attribute access).
        self._conn = None
        self.ensure_schema()

    def _connect(self):
        """Return the calling thread's thread-local connection.

        Creates it on first use (per thread), applying the spec §6.1 pragmas
        (journal_mode=WAL, synchronous=NORMAL, busy_timeout, foreign_keys=ON)
        and row_factory=Row. All pragmas are idempotent. The connection is
        registered in _tls_connections (under _tls_lock) so close() can clean
        it up regardless of which thread created it.

        Callers use the returned connection as a context manager
        (``with self._connect() as conn:``) relying on sqlite3's native
        __enter__/__exit__ (commit/rollback) semantics — preserved here.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
            with self._tls_lock:
                self._tls_connections.append(conn)
            if threading.current_thread() is threading.main_thread():
                self._conn = conn
        return conn

    def close(self):
        """Close the current thread's connection and all registered connections.

        Iterates _tls_connections (thread-safe under _tls_lock), closing each
        (swallowing already-closed errors), then clears the registry. Safe to
        call multiple times (idempotent).
        """
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
            self._local.conn = None
        self._conn = None
        with self._tls_lock:
            for c in self._tls_connections:
                try:
                    c.close()
                except sqlite3.Error:
                    pass
            self._tls_connections.clear()

    def ensure_schema(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_name TEXT UNIQUE NOT NULL,
                    pdf_path TEXT NOT NULL,
                    tree_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    node_id TEXT NOT NULL,
                    title TEXT,
                    summary TEXT,
                    start_index INTEGER,
                    end_index INTEGER,
                    parent_node_id TEXT,
                    UNIQUE(doc_id, node_id)
                );

                CREATE TABLE IF NOT EXISTS pages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page_number INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    UNIQUE(doc_id, page_number)
                );

                CREATE TABLE IF NOT EXISTS closet_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    tag_text TEXT NOT NULL,
                    tag_token TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_closet_tags_token
                    ON closet_tags(doc_id, tag_token);

                CREATE TABLE IF NOT EXISTS doc_keywords (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    keyword TEXT NOT NULL,
                    field TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_doc_keywords ON doc_keywords(keyword, doc_id);

                CREATE INDEX IF NOT EXISTS idx_nodes_doc_id
                    ON nodes(doc_id);

                CREATE INDEX IF NOT EXISTS idx_nodes_parent_node_id
                    ON nodes(parent_node_id);

                CREATE INDEX IF NOT EXISTS idx_pages_doc_id
                    ON pages(doc_id);

                CREATE TABLE IF NOT EXISTS kb_identity (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    identity_text TEXT NOT NULL,
                    doc_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Entity and relationship tables for cross-document graph
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    aliases TEXT,
                    doc_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, entity_type)
                );
                CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
                CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);

                CREATE TABLE IF NOT EXISTS entity_mentions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    node_id TEXT,
                    context_snippet TEXT,
                    confidence REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_entity_mentions_doc ON entity_mentions(doc_id);
                CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id);

                CREATE TABLE IF NOT EXISTS entity_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    predicate TEXT NOT NULL,
                    object_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    doc_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
                    confidence REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_entity_relations_subject ON entity_relations(subject_id);
                CREATE INDEX IF NOT EXISTS idx_entity_relations_object ON entity_relations(object_id);
                """
            )
            # Migrate: add doc_description if missing
            try:
                conn.execute("SELECT doc_description FROM documents LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE documents ADD COLUMN doc_description TEXT")

    def insert_document(self, pdf_name, pdf_path, doc_description=None):
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO documents (pdf_name, pdf_path, doc_description)
                VALUES (?, ?, ?)
                ON CONFLICT(pdf_name) DO UPDATE SET
                    pdf_path = excluded.pdf_path,
                    doc_description = COALESCE(excluded.doc_description, documents.doc_description)
                RETURNING id
                """,
                (pdf_name, pdf_path, doc_description),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"Failed to insert or retrieve document for {pdf_name}")
            return row[0]

    def update_document_tree(self, doc_id, tree_json):
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET tree_json = ? WHERE id = ?",
                (tree_json, doc_id),
            )

    def update_document_description(self, doc_id, doc_description):
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET doc_description = ? WHERE id = ?",
                (doc_description, doc_id),
            )

    def insert_nodes(self, doc_id, records):
        with self._connect() as conn:
            conn.execute("DELETE FROM nodes WHERE doc_id = ?", (doc_id,))
            conn.executemany(
                """
                INSERT INTO nodes
                (doc_id, node_id, title, summary, start_index, end_index, parent_node_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )

    def insert_pages(self, doc_id, page_records, chunk_size=50):
        with self._connect() as conn:
            conn.execute("DELETE FROM pages WHERE doc_id = ?", (doc_id,))
            for i in range(0, len(page_records), chunk_size):
                chunk = page_records[i:i + chunk_size]
                conn.executemany(
                    "INSERT INTO pages (doc_id, page_number, content) VALUES (?, ?, ?)",
                    chunk,
                )

    def get_document_by_name(self, pdf_name):
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM documents WHERE pdf_name = ?", (pdf_name,)
        ).fetchone()
        return dict(row) if row else None

    def get_document_by_id(self, doc_id):
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_documents(self):
        conn = self._connect()
        rows = conn.execute("SELECT * FROM documents ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def get_node(self, doc_id, node_id):
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM nodes WHERE doc_id = ? AND node_id = ?",
            (doc_id, node_id),
        ).fetchone()
        return dict(row) if row else None

    def get_nodes_by_ids(self, doc_id, node_ids):
        if not node_ids:
            return []
        results = []
        conn = self._connect()
        for i in range(0, len(node_ids), SQLITE_MAX_VARIABLE_NUMBER):
            chunk = node_ids[i:i + SQLITE_MAX_VARIABLE_NUMBER]
            placeholders = ",".join("?" for _ in chunk)
            sql = f"SELECT * FROM nodes WHERE doc_id = ? AND node_id IN ({placeholders})"
            rows = conn.execute(sql, (doc_id, *chunk)).fetchall()
            results.extend(rows)
        return [dict(r) for r in results]

    def get_nodes_by_doc_id(self, doc_id):
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM nodes WHERE doc_id = ? ORDER BY start_index, id",
            (doc_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_top_level_nodes(self, doc_id):
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM nodes WHERE doc_id = ? AND parent_node_id IS NULL ORDER BY start_index, id",
            (doc_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_pages_in_range(self, doc_id, start_index, end_index):
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT page_number, content FROM pages
            WHERE doc_id = ? AND page_number >= ? AND page_number <= ?
            ORDER BY page_number
            """,
            (doc_id, start_index, end_index),
        ).fetchall()
        return [(r["page_number"], r["content"]) for r in rows]

    def get_pages_by_numbers(self, doc_id, page_numbers):
        if not page_numbers:
            return []
        results = []
        conn = self._connect()
        for i in range(0, len(page_numbers), SQLITE_MAX_VARIABLE_NUMBER):
            chunk = page_numbers[i:i + SQLITE_MAX_VARIABLE_NUMBER]
            placeholders = ",".join("?" for _ in chunk)
            sql = f"""
                SELECT page_number, content FROM pages
                WHERE doc_id = ? AND page_number IN ({placeholders})
                ORDER BY page_number
            """
            rows = conn.execute(sql, (doc_id, *chunk)).fetchall()
            results.extend(rows)
        return [(r["page_number"], r["content"]) for r in results]

    def insert_closet_tags(self, doc_id, records):
        if not records:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM closet_tags WHERE doc_id = ?", (doc_id,))
            for i in range(0, len(records), 100):
                chunk = records[i:i + 100]
                conn.executemany(
                    """
                    INSERT INTO closet_tags (doc_id, tag_text, tag_token, confidence, source)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    chunk,
                )

    def delete_closet_tags(self, doc_id):
        with self._connect() as conn:
            conn.execute("DELETE FROM closet_tags WHERE doc_id = ?", (doc_id,))

    def delete_document(self, doc_id: int) -> None:
        """Delete a document and cascade-delete its child rows.

        Relies on existing ``ON DELETE CASCADE`` foreign keys on
        nodes/pages/closet_tags/doc_keywords (see ensure_schema). The
        thread-local connection from ``self._connect()`` applies
        ``PRAGMA foreign_keys = ON`` per connection, so the cascade fires.
        Idempotent: deleting a non-existent id deletes 0 rows (no error).
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def match_closet_tags(self, tokens, top_k=5):
        if not tokens:
            return []
        results = []
        conn = self._connect()
        for i in range(0, len(tokens), SQLITE_MAX_VARIABLE_NUMBER):
            chunk = tokens[i:i + SQLITE_MAX_VARIABLE_NUMBER]
            placeholders = ",".join("?" for _ in chunk)
            sql = f"""
                SELECT doc_id, SUM(confidence) AS score
                FROM closet_tags
                WHERE tag_token IN ({placeholders})
                GROUP BY doc_id
            """
            rows = conn.execute(sql, chunk).fetchall()
            results.extend(rows)
        # Merge results from multiple chunks: re-aggregate scores by doc_id
        merged = {}
        for r in results:
            merged[r["doc_id"]] = merged.get(r["doc_id"], 0) + r["score"]
        # Sort by score descending, take top_k
        sorted_docs = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_docs

    def insert_doc_keywords(self, doc_id, records):
        if not records:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM doc_keywords WHERE doc_id = ?", (doc_id,))
            for i in range(0, len(records), 100):
                chunk = records[i:i + 100]
                conn.executemany(
                    "INSERT INTO doc_keywords (doc_id, keyword, field) VALUES (?, ?, ?)",
                    chunk,
                )

    def delete_doc_keywords(self, doc_id):
        with self._connect() as conn:
            conn.execute("DELETE FROM doc_keywords WHERE doc_id = ?", (doc_id,))

    def match_doc_keywords(self, tokens, top_k=10):
        if not tokens:
            return []
        results = []
        conn = self._connect()
        for i in range(0, len(tokens), SQLITE_MAX_VARIABLE_NUMBER):
            chunk = tokens[i:i + SQLITE_MAX_VARIABLE_NUMBER]
            placeholders = ",".join("?" for _ in chunk)
            sql = f"""
                SELECT doc_id, COUNT(*) AS score
                FROM doc_keywords
                WHERE keyword IN ({placeholders})
                GROUP BY doc_id
            """
            rows = conn.execute(sql, chunk).fetchall()
            results.extend(rows)
        merged = {}
        for r in results:
            merged[r["doc_id"]] = merged.get(r["doc_id"], 0) + r["score"]
        sorted_docs = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_docs

    def set_kb_identity(self, identity_text, doc_count):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO kb_identity (id, identity_text, doc_count, updated_at)
                VALUES (1, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    identity_text = excluded.identity_text,
                    doc_count = excluded.doc_count,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (identity_text, doc_count),
            )

    def get_kb_identity(self):
        conn = self._connect()
        row = conn.execute("SELECT identity_text FROM kb_identity WHERE id = 1").fetchone()
        return row["identity_text"] if row else None



    # ------------------------------------------------------------------
    # Entity and relationship methods
    # ------------------------------------------------------------------

    def insert_entity(self, entity_type: str, name: str, aliases: list = None) -> int:
        """Insert or get an entity. Returns entity ID."""
        aliases_json = json.dumps(aliases or [], ensure_ascii=False)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO entities (entity_type, name, aliases)
                VALUES (?, ?, ?)
                ON CONFLICT(name, entity_type) DO UPDATE SET
                    aliases = CASE 
                        WHEN excluded.aliases != '[]' THEN excluded.aliases 
                        ELSE entities.aliases 
                    END
                RETURNING id
                """,
                (entity_type, name, aliases_json)
            )
            row = cur.fetchone()
            if row is None:
                # Fallback: get existing
                row = conn.execute(
                    "SELECT id FROM entities WHERE name = ? AND entity_type = ?",
                    (name, entity_type)
                ).fetchone()
            return row[0] if row else None

    def insert_entity_mention(
        self, entity_id: int, doc_id: int, 
        node_id: str = None, context_snippet: str = None, 
        confidence: float = 0.5
    ) -> None:
        """Insert an entity mention."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO entity_mentions (entity_id, doc_id, node_id, context_snippet, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (entity_id, doc_id, node_id, context_snippet, confidence)
            )
            # Update doc_count
            conn.execute(
                """
                UPDATE entities SET doc_count = (
                    SELECT COUNT(DISTINCT doc_id) FROM entity_mentions WHERE entity_id = ?
                ) WHERE id = ?
                """,
                (entity_id, entity_id)
            )

    def insert_entity_relation(
        self, subject_id: int, predicate: str, object_id: int,
        doc_id: int = None, confidence: float = 0.5
    ) -> None:
        """Insert an entity relationship."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO entity_relations (subject_id, predicate, object_id, doc_id, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (subject_id, predicate, object_id, doc_id, confidence)
            )

    def get_entity_by_name(self, name: str) -> dict:
        """Get an entity by name."""
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM entities WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def search_entities(self, query: str, limit: int = 20) -> list:
        """Search entities by name (partial match)."""
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT * FROM entities 
            WHERE name LIKE ? OR aliases LIKE ?
            ORDER BY doc_count DESC
            LIMIT ?
            """,
            (f"%{query}%", f"%{query}%", limit)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_entity_relations(self, entity_id: int, direction: str = "both") -> list:
        """Get relations for an entity.
        
        Args:
            entity_id: Entity ID
            direction: "outgoing", "incoming", or "both"
        """
        conn = self._connect()
        results = []
        
        if direction in ("outgoing", "both"):
            rows = conn.execute(
                """
                SELECT er.*, e1.name as subject_name, e2.name as object_name
                FROM entity_relations er
                JOIN entities e1 ON er.subject_id = e1.id
                JOIN entities e2 ON er.object_id = e2.id
                WHERE er.subject_id = ?
                ORDER BY er.confidence DESC
                """,
                (entity_id,)
            ).fetchall()
            results.extend([dict(r) for r in rows])
        
        if direction in ("incoming", "both"):
            rows = conn.execute(
                """
                SELECT er.*, e1.name as subject_name, e2.name as object_name
                FROM entity_relations er
                JOIN entities e1 ON er.subject_id = e1.id
                JOIN entities e2 ON er.object_id = e2.id
                WHERE er.object_id = ?
                ORDER BY er.confidence DESC
                """,
                (entity_id,)
            ).fetchall()
            results.extend([dict(r) for r in rows])
        
        return results

    def get_document_entities(self, doc_id: int) -> list:
        """Get all entities mentioned in a document."""
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT e.*, em.context_snippet, em.confidence as mention_confidence
            FROM entities e
            JOIN entity_mentions em ON e.id = em.entity_id
            WHERE em.doc_id = ?
            ORDER BY em.confidence DESC
            """,
            (doc_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_entity_documents(self, entity_id: int) -> list:
        """Get all documents mentioning an entity."""
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT d.*, em.context_snippet, em.confidence
            FROM documents d
            JOIN entity_mentions em ON d.id = em.doc_id
            WHERE em.entity_id = ?
            ORDER BY em.confidence DESC
            """,
            (entity_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_related_documents(self, doc_id: int, limit: int = 10) -> list:
        """Get documents related through shared entities."""
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT d.*, COUNT(DISTINCT em.entity_id) as shared_entities
            FROM documents d
            JOIN entity_mentions em ON d.id = em.doc_id
            WHERE em.entity_id IN (
                SELECT entity_id FROM entity_mentions WHERE doc_id = ?
            )
            AND d.id != ?
            GROUP BY d.id
            ORDER BY shared_entities DESC
            LIMIT ?
            """,
            (doc_id, doc_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_entity(self, entity_id: int) -> None:
        """Delete an entity and its mentions/relations."""
        with self._connect() as conn:
            conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
