import os
import sqlite3


SQLITE_MAX_VARIABLE_NUMBER = 999


class PageIndexDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self._conn = None
        self.ensure_schema()

    def _connect(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

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

                CREATE TABLE IF NOT EXISTS kb_identity (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    identity_text TEXT NOT NULL,
                    doc_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
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
                ON CONFLICT(pdf_name) DO UPDATE SET pdf_name = pdf_name
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
            chunk = []
            for record in page_records:
                chunk.append(record)
                if len(chunk) >= chunk_size:
                    conn.executemany(
                        "INSERT INTO pages (doc_id, page_number, content) VALUES (?, ?, ?)",
                        chunk,
                    )
                    chunk = []
            if chunk:
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
    def ensure_closet_schema(self):
        # closet_tags is already created by ensure_schema(); keep this method
        # for callers that explicitly want to ensure the closet schema.
        pass

    def insert_closet_tags(self, doc_id, records):
        if not records:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM closet_tags WHERE doc_id = ?", (doc_id,))
            chunk = []
            for record in records:
                chunk.append(record)
                if len(chunk) >= 50:
                    conn.executemany(
                        """
                        INSERT INTO closet_tags (doc_id, tag_text, tag_token, confidence, source)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        chunk,
                    )
                    chunk = []
            if chunk:
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
            chunk = []
            for record in records:
                chunk.append(record)
                if len(chunk) >= 50:
                    conn.executemany(
                        "INSERT INTO doc_keywords (doc_id, keyword, field) VALUES (?, ?, ?)",
                        chunk,
                    )
                    chunk = []
            if chunk:
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

