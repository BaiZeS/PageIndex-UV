import os
import sqlite3
import json
import threading
import time


SQLITE_MAX_VARIABLE_NUMBER = 999

# WAL + synchronous=NORMAL + busy_timeout (spec §6.1).
# busy_timeout=5000ms covers ~3 concurrent uploads queueing depth (spec §5.2).
_BUSY_TIMEOUT_MS = 5000

# Common Chinese stopwords that don't help entity matching
_STOPWORDS = frozenset({
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
    "会", "着", "没有", "看", "好", "自己", "这", "他", "吗", "那",
    "被", "它", "把", "又", "对", "或者", "但", "而", "与", "及",
    "什么", "怎么", "哪些", "哪个", "如何", "为什么", "可以", "能",
    "请", "帮", "找", "查", "看看", "一下", "参与", "关于", "相关",
})


# Tokenization cache — jieba tokenization is deterministic for the same query.
_TOKENIZE_CACHE: dict = {}  # query -> (tokens, timestamp)
_TOKENIZE_CACHE_TTL = 300   # seconds (5 minutes)
_TOKENIZE_CACHE_MAX = 512   # max entries


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

                -- Node profiles (P1.2): per-TOC-node attribute signature
                -- (canonical entities / keywords / tags) for O(1) lookups.
                CREATE TABLE IF NOT EXISTS node_profiles (
                    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    node_id TEXT NOT NULL,
                    entities TEXT NOT NULL DEFAULT '[]',
                    keywords TEXT NOT NULL DEFAULT '[]',
                    tags TEXT NOT NULL DEFAULT '[]',
                    PRIMARY KEY (doc_id, node_id)
                );
                CREATE INDEX IF NOT EXISTS idx_node_profiles_doc
                    ON node_profiles(doc_id);

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
                CREATE INDEX IF NOT EXISTS idx_entity_relations_predicate ON entity_relations(predicate);

                -- Corpus tree (P1): unified corpus-level topic hierarchy.
                -- 文档→节点 is the real trunk; cluster layers above are inferred
                -- scaffolding. Soft membership makes doc→cluster a DAG, so a doc
                -- may attach to multiple clusters with weights (never hard-dropped).
                CREATE TABLE IF NOT EXISTS corpus_tree_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_id INTEGER REFERENCES corpus_tree_nodes(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    summary TEXT,
                    level INTEGER NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'cluster',
                    tag TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_corpus_tree_nodes_parent
                    ON corpus_tree_nodes(parent_id);

                CREATE TABLE IF NOT EXISTS corpus_tree_membership (
                    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    node_id INTEGER NOT NULL REFERENCES corpus_tree_nodes(id) ON DELETE CASCADE,
                    weight REAL NOT NULL DEFAULT 1.0,
                    PRIMARY KEY (doc_id, node_id)
                );
                CREATE INDEX IF NOT EXISTS idx_corpus_tree_membership_node
                    ON corpus_tree_membership(node_id);

                -- Canonical tag set + synonym mapping (tag normalization).
                CREATE TABLE IF NOT EXISTS corpus_tag_norm (
                    raw_tag TEXT PRIMARY KEY,
                    canonical_tag TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_corpus_tag_norm_canonical
                    ON corpus_tag_norm(canonical_tag);

                -- Disposition records for cluster merge/split decisions.
                CREATE TABLE IF NOT EXISTS corpus_tree_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id INTEGER,
                    event_type TEXT NOT NULL,
                    detail TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            # Migrate: add doc_description if missing
            try:
                conn.execute("SELECT doc_description FROM documents LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE documents ADD COLUMN doc_description TEXT")

            # One-shot backfill (idempotent): before a93ee95,
            # ClosetIndex.add_document hardcoded source="llm" for jieba
            # fallback tags, so legacy DBs carry fallback words mislabeled
            # "llm" that pollute the (now-gated) semantic channel. Collision
            # free: fallback tags have confidence exactly 0.3 while LLM tags
            # are stored only with conf >= 0.5 (_MIN_TAG_CONFIDENCE). No-op
            # once applied (source already "fallback") and on fresh DBs.
            conn.execute(
                "UPDATE closet_tags SET source = 'fallback' "
                "WHERE source = 'llm' AND abs(confidence - 0.3) < 1e-9"
            )

            # Migrate: add tf column for BM25 scoring (idempotent)
            try:
                conn.execute("SELECT tf FROM doc_keywords LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE doc_keywords ADD COLUMN tf INTEGER DEFAULT 1")

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

    def get_document_count(self) -> int:
        """Return total document count without materializing rows."""
        conn = self._connect()
        row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return row[0] if row else 0

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

    def get_document_content(self, doc_id):
        """Return the full text content of a document (concatenated pages)."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT content FROM pages WHERE doc_id = ? ORDER BY page_number",
            (doc_id,),
        ).fetchall()
        return "\n".join(r["content"] for r in rows if r["content"])

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

    def insert_closet_tags_batch(self, records: list[tuple]) -> None:
        """Batch insert closet tags across multiple documents.

        Each record = (doc_id, tag_text, tag_token, confidence, source).
        Deletes existing tags for each affected doc_id first, then bulk inserts.
        """
        if not records:
            return
        with self._connect() as conn:
            # Delete existing tags for all affected doc_ids
            doc_ids = {r[0] for r in records}
            for did in doc_ids:
                conn.execute("DELETE FROM closet_tags WHERE doc_id = ?", (did,))
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

    def get_doc_tags(self, doc_id, source=None):
        """返回某文档的语义标签列表：[{tag_text, confidence}]（按置信度降序）。

        source 可选过滤（[7.2] 分层）："llm" 只取 LLM 抽象语义标签（语义通道），
        "fallback" 只取 jieba 兜底原词（关键词层）；缺省返回全部来源。
        """
        sql = "SELECT tag_text, confidence FROM closet_tags WHERE doc_id = ?"
        params = [doc_id]
        if source is not None:
            sql += " AND source = ?"
            params.append(source)
        sql += " ORDER BY confidence DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [{"tag_text": r[0], "confidence": r[1]} for r in rows]

    def upsert_node_profiles(self, doc_id, profiles):
        """Replace the node profiles of a document (idempotent re-index).

        Each profile dict: {node_id, entities: [{name, type}], keywords: [...],
        tags: [...]}. Deletes the doc's existing rows first so stale nodes from
        a previous index don't survive, then bulk INSERT OR REPLACE.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM node_profiles WHERE doc_id = ?", (doc_id,))
            if not profiles:
                return
            records = [
                (
                    doc_id,
                    p["node_id"],
                    json.dumps(p.get("entities", []), ensure_ascii=False),
                    json.dumps(p.get("keywords", []), ensure_ascii=False),
                    json.dumps(p.get("tags", []), ensure_ascii=False),
                )
                for p in profiles
            ]
            for i in range(0, len(records), 100):
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO node_profiles
                    (doc_id, node_id, entities, keywords, tags)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    records[i:i + 100],
                )

    def get_node_profiles(self, doc_id):
        """Return a document's node profiles with JSON fields parsed."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT node_id, entities, keywords, tags FROM node_profiles "
            "WHERE doc_id = ? ORDER BY node_id",
            (doc_id,),
        ).fetchall()
        return [
            {
                "node_id": r["node_id"],
                "entities": json.loads(r["entities"] or "[]"),
                "keywords": json.loads(r["keywords"] or "[]"),
                "tags": json.loads(r["tags"] or "[]"),
            }
            for r in rows
        ]

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

    def match_closet_tags(self, tokens, top_k=5, source=None):
        """closet_tags 倒排匹配。source 可选过滤（[7.2] 分层）：
        语义通道传 "llm"（只认 LLM 抽象标签）；关键词层缺省匹配全部来源。"""
        if not tokens:
            return []
        src_clause = ""
        extra = []
        if source is not None:
            src_clause = " AND source = ?"
            extra = [source]
        conn = self._connect()
        # Reserve bind slots for the source filter / LIMIT inside SQLite's cap.
        chunk_size = SQLITE_MAX_VARIABLE_NUMBER - len(extra)
        if len(tokens) + 1 <= chunk_size:  # +1 for the LIMIT parameter
            placeholders = ",".join("?" for _ in tokens)
            sql = f"""
                SELECT doc_id, SUM(confidence) AS score
                FROM closet_tags
                WHERE tag_token IN ({placeholders}){src_clause}
                GROUP BY doc_id
                ORDER BY score DESC
                LIMIT ?
            """
            rows = conn.execute(sql, (*tokens, *extra, top_k)).fetchall()
            return [(r["doc_id"], r["score"]) for r in rows]
        results = []
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            sql = f"""
                SELECT doc_id, SUM(confidence) AS score
                FROM closet_tags
                WHERE tag_token IN ({placeholders}){src_clause}
                GROUP BY doc_id
            """
            rows = conn.execute(sql, (*chunk, *extra)).fetchall()
            results.extend(rows)
        merged = {}
        for r in results:
            merged[r["doc_id"]] = merged.get(r["doc_id"], 0) + r["score"]
        sorted_docs = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_docs

    def insert_doc_keywords(self, doc_id, records):
        if not records:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM doc_keywords WHERE doc_id = ?", (doc_id,))
            for i in range(0, len(records), 100):
                chunk = records[i:i + 100]
                # Backward compat: 3-element tuples (doc_id, keyword, field) → tf=1
                normalized = [r if len(r) == 4 else (*r, 1) for r in chunk]
                conn.executemany(
                    "INSERT INTO doc_keywords (doc_id, keyword, field, tf) VALUES (?, ?, ?, ?)",
                    normalized,
                )

    def delete_doc_keywords(self, doc_id):
        with self._connect() as conn:
            conn.execute("DELETE FROM doc_keywords WHERE doc_id = ?", (doc_id,))

    def match_doc_keywords(self, tokens, top_k=10):
        if not tokens:
            return []
        conn = self._connect()
        if len(tokens) <= SQLITE_MAX_VARIABLE_NUMBER:
            return self._bm25_query(conn, tokens, top_k)
        # Chunked path: compute BM25 per chunk, merge scores by summation
        merged = {}
        for i in range(0, len(tokens), SQLITE_MAX_VARIABLE_NUMBER):
            chunk = tokens[i:i + SQLITE_MAX_VARIABLE_NUMBER]
            for doc_id, score in self._bm25_query(conn, chunk, top_k=None):
                merged[doc_id] = merged.get(doc_id, 0.0) + score
        sorted_docs = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_docs

    def _bm25_query(self, conn, tokens, top_k):
        """BM25 scoring: k1=1.5, b=0.75, proper IDF, TF, doc-length normalization."""
        placeholders = ",".join("?" for _ in tokens)
        limit_clause = f"LIMIT ?" if top_k is not None else ""
        sql = f"""
            WITH doc_lens AS (
                SELECT doc_id, SUM(tf) AS doc_len FROM doc_keywords GROUP BY doc_id
            ),
            avgdl_val AS (
                SELECT AVG(doc_len) AS avgdl FROM doc_lens
            ),
            N_val AS (
                SELECT COUNT(DISTINCT id) AS N FROM documents
            ),
            tok_df AS (
                SELECT keyword, COUNT(DISTINCT doc_id) AS df
                FROM doc_keywords WHERE keyword IN ({placeholders}) GROUP BY keyword
            ),
            per_tok AS (
                SELECT doc_id, keyword, SUM(tf) AS tf_sum
                FROM doc_keywords WHERE keyword IN ({placeholders})
                GROUP BY doc_id, keyword
            )
            SELECT pt.doc_id,
                   SUM(
                     (pt.tf_sum * 2.5) / (pt.tf_sum + 1.5 * (1.0 - 0.75 + 0.75 * (dl.doc_len / av.avgdl)))
                     * LN(1.0 + (n.N - tdf.df + 0.5) / (tdf.df + 0.5))
                   ) AS score
            FROM per_tok pt
            JOIN doc_lens dl ON dl.doc_id = pt.doc_id
            CROSS JOIN avgdl_val av
            CROSS JOIN N_val n
            JOIN tok_df tdf ON tdf.keyword = pt.keyword
            GROUP BY pt.doc_id
            ORDER BY score DESC
            {limit_clause}
        """
        params = (*tokens, *tokens)
        if top_k is not None:
            params = (*params, top_k)
        rows = conn.execute(sql, params).fetchall()
        return [(r["doc_id"], r["score"]) for r in rows]

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
    # Corpus tree methods (P1)
    # ------------------------------------------------------------------

    def corpus_tree_clear(self):
        """Delete the whole corpus tree (nodes, memberships, events)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM corpus_tree_membership")
            conn.execute("DELETE FROM corpus_tree_nodes")
            conn.execute("DELETE FROM corpus_tree_events")

    def insert_corpus_tree_node(self, parent_id, title, summary, level,
                                kind="cluster", tag=None):
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO corpus_tree_nodes (parent_id, title, summary, level, kind, tag)
                VALUES (?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                (parent_id, title, summary, level, kind, tag),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("Failed to insert corpus tree node")
            return row[0]

    def update_corpus_tree_node(self, node_id, parent_id=None, title=None,
                                summary=None, level=None):
        """Update only the provided (non-None) fields of a corpus tree node."""
        fields, values = [], []
        if parent_id is not None:
            fields.append("parent_id = ?")
            values.append(parent_id)
        if title is not None:
            fields.append("title = ?")
            values.append(title)
        if summary is not None:
            fields.append("summary = ?")
            values.append(summary)
        if level is not None:
            fields.append("level = ?")
            values.append(level)
        if not fields:
            return
        values.append(node_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE corpus_tree_nodes SET {', '.join(fields)} WHERE id = ?",
                values,
            )

    def delete_corpus_tree_node(self, node_id):
        with self._connect() as conn:
            conn.execute("DELETE FROM corpus_tree_nodes WHERE id = ?", (node_id,))

    def get_corpus_tree_nodes(self):
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM corpus_tree_nodes ORDER BY level, id"
        ).fetchall()
        return [dict(r) for r in rows]

    def add_corpus_membership(self, doc_id, node_id, weight=1.0):
        """Upsert a soft doc→cluster edge (a doc may attach to many clusters)."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO corpus_tree_membership (doc_id, node_id, weight)
                VALUES (?, ?, ?)
                ON CONFLICT(doc_id, node_id) DO UPDATE SET weight = excluded.weight
                """,
                (doc_id, node_id, weight),
            )

    def delete_corpus_memberships_for_node(self, node_id):
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM corpus_tree_membership WHERE node_id = ?", (node_id,)
            )

    def get_corpus_node_docs(self, node_id):
        conn = self._connect()
        rows = conn.execute(
            "SELECT doc_id, weight FROM corpus_tree_membership WHERE node_id = ? "
            "ORDER BY doc_id",
            (node_id,),
        ).fetchall()
        return [(r["doc_id"], r["weight"]) for r in rows]

    def get_corpus_doc_memberships(self, doc_id):
        conn = self._connect()
        rows = conn.execute(
            "SELECT node_id, weight FROM corpus_tree_membership WHERE doc_id = ? "
            "ORDER BY node_id",
            (doc_id,),
        ).fetchall()
        return [(r["node_id"], r["weight"]) for r in rows]

    def get_all_corpus_memberships(self):
        conn = self._connect()
        rows = conn.execute(
            "SELECT doc_id, node_id, weight FROM corpus_tree_membership "
            "ORDER BY doc_id, node_id"
        ).fetchall()
        return [(r["doc_id"], r["node_id"], r["weight"]) for r in rows]

    def set_corpus_tag_norm_map(self, mapping):
        """Replace the whole raw_tag→canonical_tag mapping."""
        with self._connect() as conn:
            conn.execute("DELETE FROM corpus_tag_norm")
            conn.executemany(
                "INSERT INTO corpus_tag_norm (raw_tag, canonical_tag) VALUES (?, ?)",
                [(raw, canonical) for raw, canonical in mapping.items()],
            )

    def upsert_corpus_tag_norm(self, raw_tag, canonical_tag):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO corpus_tag_norm (raw_tag, canonical_tag) VALUES (?, ?)
                ON CONFLICT(raw_tag) DO UPDATE SET canonical_tag = excluded.canonical_tag
                """,
                (raw_tag, canonical_tag),
            )

    def get_corpus_tag_norm_map(self):
        conn = self._connect()
        rows = conn.execute("SELECT raw_tag, canonical_tag FROM corpus_tag_norm").fetchall()
        return {r["raw_tag"]: r["canonical_tag"] for r in rows}

    def get_corpus_canonical_tags(self):
        conn = self._connect()
        rows = conn.execute(
            "SELECT DISTINCT canonical_tag FROM corpus_tag_norm ORDER BY canonical_tag"
        ).fetchall()
        return [r["canonical_tag"] for r in rows]

    def remap_corpus_tag_norm(self, from_canonical, to_canonical):
        """Route every raw tag mapping to from_canonical at to_canonical.

        Used after a cluster merge so incremental docs carrying the victim's
        tag attach to the survivor instead of resurrecting the victim cluster.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE corpus_tag_norm SET canonical_tag = ? WHERE canonical_tag = ?",
                (to_canonical, from_canonical),
            )

    def insert_corpus_tree_event(self, node_id, event_type, detail=None):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO corpus_tree_events (node_id, event_type, detail) VALUES (?, ?, ?)",
                (node_id, event_type, detail),
            )

    def get_corpus_tree_events(self, event_type=None):
        conn = self._connect()
        if event_type:
            rows = conn.execute(
                "SELECT * FROM corpus_tree_events WHERE event_type = ? ORDER BY id",
                (event_type,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM corpus_tree_events ORDER BY id").fetchall()
        return [dict(r) for r in rows]

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
                        WHEN excluded.aliases != '[]' THEN
                            (SELECT json_group_array(DISTINCT value) FROM (
                                SELECT value FROM json_each(entities.aliases)
                                UNION
                                SELECT value FROM json_each(excluded.aliases)
                            ))
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

    def delete_entity_mentions(self, doc_id: int) -> None:
        """Delete all entity mentions of a document (pre-reindex cleanup)."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM entity_mentions WHERE doc_id = ?", (doc_id,)
            )

    def insert_entity_mentions_batch(self, records: list[tuple]) -> None:
        """Batch insert entity mentions.

        Each record = (entity_id, doc_id, node_id, context_snippet, confidence).
        Uses executemany for efficiency; updates doc_count per unique entity.
        """
        if not records:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO entity_mentions (entity_id, doc_id, node_id, context_snippet, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                records,
            )
            # Update doc_count for each unique entity_id
            entity_ids = {r[0] for r in records}
            for eid in entity_ids:
                conn.execute(
                    """
                    UPDATE entities SET doc_count = (
                        SELECT COUNT(DISTINCT doc_id) FROM entity_mentions WHERE entity_id = ?
                    ) WHERE id = ?
                    """,
                    (eid, eid),
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

    def get_entity_by_id(self, entity_id: int) -> dict:
        """Get an entity by ID."""
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_entity_mentions_by_doc(self, doc_id: int) -> list:
        """Get all entity mentions in a document."""
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT em.*, e.name as entity_name, e.entity_type
            FROM entity_mentions em
            JOIN entities e ON em.entity_id = e.id
            WHERE em.doc_id = ?
            ORDER BY em.confidence DESC, em.id
            """,
            (doc_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _tokenize_query(query: str) -> list[str]:
        """Tokenize query with jieba, filter stopwords and single-char tokens.

        Results are cached (same query → same tokens, TTL=5min).
        """
        now = time.monotonic()
        entry = _TOKENIZE_CACHE.get(query)
        if entry and (now - entry[1]) < _TOKENIZE_CACHE_TTL:
            return entry[0]
        if entry:
            del _TOKENIZE_CACHE[query]

        import jieba
        tokens = jieba.lcut(query)
        result = []
        for t in tokens:
            t = t.strip()
            if not t or len(t) < 2:
                continue
            if t in _STOPWORDS:
                continue
            result.append(t)

        if len(_TOKENIZE_CACHE) >= _TOKENIZE_CACHE_MAX:
            oldest_key = min(_TOKENIZE_CACHE, key=lambda k: _TOKENIZE_CACHE[k][1])
            del _TOKENIZE_CACHE[oldest_key]
        frozen = tuple(result)
        _TOKENIZE_CACHE[query] = (frozen, now)
        return frozen

    def search_entities(self, query: str, limit: int = 20) -> list:
        """Search entities by name/aliases using jieba tokenization.

        Tokenizes the query with jieba, filters stopwords and single-char
        tokens, then matches each token against entity name/aliases with OR.
        Results are deduplicated and ordered by doc_count DESC.
        """
        if not query or not query.strip():
            return []

        tokens = self._tokenize_query(query)
        if not tokens:
            return []

        conn = self._connect()
        # Build OR conditions for each token against name and aliases
        conditions = []
        params = []
        for token in tokens:
            conditions.append("(name LIKE ? OR aliases LIKE ?)")
            params.extend([f"%{token}%", f"%{token}%"])

        where_clause = " OR ".join(conditions)
        sql = f"""
            SELECT DISTINCT * FROM entities
            WHERE {where_clause}
            ORDER BY doc_count DESC
            LIMIT ?
        """
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
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

    def get_entities_by_relation(self, predicate: str, limit: int = 20) -> list:
        """Get entity pairs connected by a specific relation type."""
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT e1.name as subject_name, e2.name as object_name,
                   er.confidence, er.doc_id
            FROM entity_relations er
            JOIN entities e1 ON er.subject_id = e1.id
            JOIN entities e2 ON er.object_id = e2.id
            WHERE er.predicate = ?
            ORDER BY er.confidence DESC
            LIMIT ?
            """,
            (predicate, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_entities_by_type(self, entity_type: str) -> list:
        """Get all entities of a given type."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM entities WHERE entity_type = ?",
            (entity_type,)
        ).fetchall()
        return [dict(r) for r in rows]

    def merge_entity_aliases(self, entity_id: int, new_aliases: list) -> None:
        """Merge new aliases into an existing entity, deduplicating."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT aliases FROM entities WHERE id = ?", (entity_id,)
            ).fetchone()
            if not row:
                return
            existing = json.loads(row["aliases"] or "[]")
            merged = list(dict.fromkeys(existing + new_aliases))  # dedup, preserve order
            conn.execute(
                "UPDATE entities SET aliases = ? WHERE id = ?",
                (json.dumps(merged, ensure_ascii=False), entity_id)
            )

    def merge_entities(self, canonical_id: int, duplicate_id: int) -> None:
        """Merge duplicate entity into canonical: redirect mentions/relations, merge aliases, delete duplicate."""
        if canonical_id == duplicate_id:
            return
        with self._connect() as conn:
            # Redirect entity_mentions
            conn.execute(
                "UPDATE OR IGNORE entity_mentions SET entity_id = ? WHERE entity_id = ?",
                (canonical_id, duplicate_id),
            )
            # Redirect entity_relations (subject and object)
            conn.execute(
                "UPDATE OR IGNORE entity_relations SET subject_id = ? WHERE subject_id = ?",
                (canonical_id, duplicate_id),
            )
            conn.execute(
                "UPDATE OR IGNORE entity_relations SET object_id = ? WHERE object_id = ?",
                (canonical_id, duplicate_id),
            )
            # Remove self-referencing relations created by the redirect
            conn.execute(
                "DELETE FROM entity_relations WHERE subject_id = ? AND object_id = ?",
                (canonical_id, canonical_id),
            )
            # Merge aliases from duplicate into canonical
            row = conn.execute(
                "SELECT aliases FROM entities WHERE id = ?", (canonical_id,)
            ).fetchone()
            if row:
                existing = json.loads(row["aliases"] or "[]")
                dup_row = conn.execute(
                    "SELECT name, aliases FROM entities WHERE id = ?", (duplicate_id,)
                ).fetchone()
                if dup_row:
                    dup_aliases = json.loads(dup_row["aliases"] or "[]")
                    dup_name = dup_row["name"]
                    merged = list(dict.fromkeys(existing + [dup_name] + dup_aliases))
                    conn.execute(
                        "UPDATE entities SET aliases = ? WHERE id = ?",
                        (json.dumps(merged, ensure_ascii=False), canonical_id),
                    )
            # Update doc_count for canonical
            conn.execute(
                """UPDATE entities SET doc_count = (
                    SELECT COUNT(DISTINCT doc_id) FROM entity_mentions WHERE entity_id = ?
                ) WHERE id = ?""",
                (canonical_id, canonical_id),
            )
            # Delete duplicate
            conn.execute("DELETE FROM entities WHERE id = ?", (duplicate_id,))

    def delete_entity(self, entity_id: int) -> None:
        """Delete an entity and its mentions/relations."""
        with self._connect() as conn:
            conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
