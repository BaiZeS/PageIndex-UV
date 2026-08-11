import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from db import PageIndexDB, _TOKENIZE_CACHE


@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = PageIndexDB(path)
    yield db
    db.close()
    os.unlink(path)


class TestDocKeywords:
    def test_insert_and_match(self, tmp_db):
        doc_id = tmp_db.insert_document("test.pdf", "/tmp/test.pdf")
        tmp_db.insert_doc_keywords(doc_id, [
            (doc_id, "前端", "name"),
            (doc_id, "脚本", "name"),
            (doc_id, "开发", "description"),
        ])
        results = tmp_db.match_doc_keywords(["前端", "脚本"], top_k=5)
        assert len(results) == 1
        assert results[0][0] == doc_id

    def test_delete_doc_keywords(self, tmp_db):
        doc_id = tmp_db.insert_document("test.pdf", "/tmp/test.pdf")
        tmp_db.insert_doc_keywords(doc_id, [(doc_id, "test", "name")])
        tmp_db.delete_doc_keywords(doc_id)
        results = tmp_db.match_doc_keywords(["test"], top_k=5)
        assert len(results) == 0


class TestKBIdentity:
    def test_set_and_get(self, tmp_db):
        tmp_db.set_kb_identity("知识库共3个文档", 3)
        identity = tmp_db.get_kb_identity()
        assert identity == "知识库共3个文档"

    def test_get_missing_returns_none(self, tmp_db):
        assert tmp_db.get_kb_identity() is None


class TestClosetTags:
    def test_get_doc_tags(self, tmp_db):
        doc_id = tmp_db.insert_document("test.pdf", "/tmp/test.pdf")
        tmp_db.insert_closet_tags(doc_id, [
            (doc_id, "容器编排", "容器 编排", 0.95, "llm"),
            (doc_id, "微服务", "微服务", 0.8, "llm"),
        ])
        tags = tmp_db.get_doc_tags(doc_id)
        assert tags == [
            {"tag_text": "容器编排", "confidence": 0.95},
            {"tag_text": "微服务", "confidence": 0.8},
        ]

    def test_get_doc_tags_empty(self, tmp_db):
        doc_id = tmp_db.insert_document("test.pdf", "/tmp/test.pdf")
        assert tmp_db.get_doc_tags(doc_id) == []


class TestClosetTagsSourceBackfill:
    """遗留库回填：a93ee95 之前 ClosetIndex.add_document 对 jieba 兜底标签
    硬编码 source="llm"。ensure_schema 须在初始化时把 conf==0.3 的 "llm"
    行改标为 "fallback"（兜底词置信度恒为 0.3，LLM 标签只存 conf≥0.5，
    故回填无碰撞、幂等），真正的 LLM 行不受影响。"""

    @staticmethod
    def _rows_by_tag(db):
        rows = db._connect().execute(
            "SELECT tag_text, source FROM closet_tags"
        ).fetchall()
        return {r["tag_text"]: r["source"] for r in rows}

    def test_legacy_llm_conf03_rows_relabeled_on_init(self, tmp_path):
        path = str(tmp_path / "legacy.db")
        db = PageIndexDB(path)
        doc_id = db.insert_document("legacy.pdf", "/tmp/legacy.pdf")
        db.insert_closet_tags(doc_id, [
            (doc_id, "分布式存储", "分布式 存储", 0.3, "llm"),      # 遗留误标兜底词
            (doc_id, "容器编排", "容器 编排", 0.9, "llm"),           # 真 LLM 标签
            (doc_id, "存储", "存储", 0.3, "fallback"),               # 已正确的兜底行
        ])
        db.close()

        db2 = PageIndexDB(path)  # ensure_schema runs the backfill
        try:
            assert self._rows_by_tag(db2) == {
                "分布式存储": "fallback",
                "容器编排": "llm",
                "存储": "fallback",
            }
        finally:
            db2.close()

    def test_backfill_idempotent(self, tmp_path):
        path = str(tmp_path / "legacy.db")
        db = PageIndexDB(path)
        doc_id = db.insert_document("legacy.pdf", "/tmp/legacy.pdf")
        db.insert_closet_tags(doc_id, [(doc_id, "分布式存储", "分布式 存储", 0.3, "llm")])
        db.close()

        db2 = PageIndexDB(path)
        first = self._rows_by_tag(db2)
        db2.close()

        db3 = PageIndexDB(path)  # second run must be a no-op
        try:
            assert first == {"分布式存储": "fallback"}
            assert self._rows_by_tag(db3) == first
        finally:
            db3.close()


class TestMatchClosetTagsChunking:
    """bind-cap 回归：查询 token 超过 SQLite 绑定变量上限（999）时，
    match_closet_tags 必须分块执行而不抛错，且得分与不分块的小查询一致
    （带与不带 source 过滤均须覆盖）。"""

    @staticmethod
    def _seed(db):
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        d2 = db.insert_document("b.pdf", "/tmp/b.pdf")
        db.insert_closet_tags(d1, [
            (d1, "alpha标签", "alpha", 0.9, "llm"),
            (d1, "beta标签", "beta", 0.6, "fallback"),
        ])
        db.insert_closet_tags(d2, [(d2, "gamma标签", "gamma", 0.8, "llm")])
        return d1, d2

    @staticmethod
    def _big_query(hits, n=1200):
        tokens = [f"filler{i:04d}" for i in range(n)]
        for pos, tok in hits.items():
            tokens[pos] = tok
        return tokens

    @staticmethod
    def _expected_overlap(db, tokens, source=None):
        """Reference: plain Python token-overlap scoring over raw rows."""
        rows = db._connect().execute(
            "SELECT doc_id, tag_token, confidence, source FROM closet_tags"
        ).fetchall()
        token_set = set(tokens)
        scores = {}
        for r in rows:
            if r["tag_token"] not in token_set:
                continue
            if source is not None and r["source"] != source:
                continue
            scores[r["doc_id"]] = scores.get(r["doc_id"], 0) + r["confidence"]
        return scores

    # Hits land in chunk 1 (idx 500) and chunk 2 (idx 1100/1150) for both
    # chunk sizes (999 without source filter, 998 with one).
    HITS = {500: "gamma", 1100: "alpha", 1150: "beta"}

    def test_over_999_tokens_without_source_filter(self, tmp_db):
        d1, d2 = self._seed(tmp_db)
        tokens = self._big_query(self.HITS)
        results = tmp_db.match_closet_tags(tokens, top_k=5)  # must not raise
        expected = self._expected_overlap(tmp_db, tokens)
        assert expected == {d1: 0.9 + 0.6, d2: 0.8}
        assert {doc_id: score for doc_id, score in results} == pytest.approx(expected)
        # Small (non-chunked) query with the same overlap → identical scores.
        small = tmp_db.match_closet_tags(list(self.HITS.values()), top_k=5)
        assert dict(results) == pytest.approx(dict(small))
        assert [doc_id for doc_id, _ in results][0] == d1  # score-desc

    def test_over_999_tokens_with_source_filter(self, tmp_db):
        d1, d2 = self._seed(tmp_db)
        tokens = self._big_query(self.HITS)
        results = tmp_db.match_closet_tags(tokens, top_k=5, source="llm")
        expected = self._expected_overlap(tmp_db, tokens, source="llm")
        # beta 行是 fallback，语义通道只认 llm：doc1 只得 0.9
        assert expected == {d1: 0.9, d2: 0.8}
        assert {doc_id: score for doc_id, score in results} == pytest.approx(expected)
        small = tmp_db.match_closet_tags(list(self.HITS.values()), top_k=5, source="llm")
        assert dict(results) == pytest.approx(dict(small))


def _count_rows(db, table, doc_id):
    """Count child-table rows for a given doc_id (helper for cascade tests)."""
    conn = db._connect()
    row = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    return row[0]


class TestDeleteDocumentCascade:
    """W2 FR1/AC1.1 — delete_document cascades to nodes/pages/closet_tags/doc_keywords.

    Proves P0-2: currently PageIndexDB has no delete_document method, so this
    test must fail with AttributeError (RED) until FR1 is implemented.
    """

    def test_delete_document_cascades_children(self, tmp_db):
        doc_id = tmp_db.insert_document("cascade.pdf", "/tmp/cascade.pdf")
        # Populate all 4 child tables that declare ON DELETE CASCADE.
        tmp_db.insert_nodes(doc_id, [(doc_id, "n1", "title", "summary", 0, 10, None)])
        tmp_db.insert_pages(doc_id, [(doc_id, 1, "page one")])
        tmp_db.insert_closet_tags(doc_id, [(doc_id, "tag", "token", 0.9, "manual")])
        tmp_db.insert_doc_keywords(doc_id, [(doc_id, "keyword", "name")])

        # Pre-condition: child rows exist.
        assert _count_rows(tmp_db, "nodes", doc_id) == 1
        assert _count_rows(tmp_db, "pages", doc_id) == 1
        assert _count_rows(tmp_db, "closet_tags", doc_id) == 1
        assert _count_rows(tmp_db, "doc_keywords", doc_id) == 1

        tmp_db.delete_document(doc_id)

        # FR1/AC1.1: documents + all 4 child tables cleared via cascade.
        conn = tmp_db._connect()
        doc_count = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()[0]
        assert doc_count == 0
        assert _count_rows(tmp_db, "nodes", doc_id) == 0
        assert _count_rows(tmp_db, "pages", doc_id) == 0
        assert _count_rows(tmp_db, "closet_tags", doc_id) == 0
        assert _count_rows(tmp_db, "doc_keywords", doc_id) == 0

class TestInsertEntityAliasMerge:
    """C1: insert_entity UPSERT must MERGE aliases, not overwrite them."""

    def test_two_inserts_same_entity_different_aliases_merged(self, tmp_db):
        """Two docs extracting the same entity with different aliases → both aliases kept."""
        eid1 = tmp_db.insert_entity("PERSON", "Alice", ["Ali", "A"])
        eid2 = tmp_db.insert_entity("PERSON", "Alice", ["Ally", "A"])
        assert eid1 == eid2  # same entity
        row = tmp_db._connect().execute(
            "SELECT aliases FROM entities WHERE id = ?", (eid1,)
        ).fetchone()
        aliases = set(__import__("json").loads(row["aliases"]))
        # All four unique aliases must be present (A deduplicated)
        assert aliases == {"Ali", "A", "Ally"}

    def test_insert_with_empty_aliases_preserves_existing(self, tmp_db):
        """Inserting with aliases=[] must not discard existing aliases."""
        tmp_db.insert_entity("ORG", "Acme", ["Acme Corp"])
        tmp_db.insert_entity("ORG", "Acme", [])
        row = tmp_db._connect().execute(
            "SELECT aliases FROM entities WHERE name = 'Acme'"
        ).fetchone()
        aliases = __import__("json").loads(row["aliases"])
        assert "Acme Corp" in aliases


    def test_delete_document_idempotent_nonexistent(self, tmp_db):
        """NFR2/AC1.2 — deleting a non-existent id deletes 0 rows, no error."""
        # 999999 does not exist; DELETE matches 0 rows and returns normally.
        tmp_db.delete_document(999999)
        # Sanity: documents table still empty.
        conn = tmp_db._connect()
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


class TestGetDocumentCount:
    def test_empty_db_returns_zero(self, tmp_db):
        assert tmp_db.get_document_count() == 0

    def test_returns_correct_count(self, tmp_db):
        tmp_db.insert_document("a.pdf", "/tmp/a.pdf")
        tmp_db.insert_document("b.pdf", "/tmp/b.pdf")
        assert tmp_db.get_document_count() == 2

    def test_count_matches_get_all(self, tmp_db):
        tmp_db.insert_document("x.pdf", "/tmp/x.pdf")
        assert tmp_db.get_document_count() == len(tmp_db.get_all_documents())


class TestInsertEntityMentionsBatch:
    def test_batch_insert(self, tmp_db):
        eid = tmp_db.insert_entity("PERSON", "Alice")
        doc1 = tmp_db.insert_document("a.pdf", "/tmp/a.pdf")
        doc2 = tmp_db.insert_document("b.pdf", "/tmp/b.pdf")
        records = [
            (eid, doc1, None, "Alice in doc1", 0.9),
            (eid, doc2, None, "Alice in doc2", 0.8),
        ]
        tmp_db.insert_entity_mentions_batch(records)
        docs = tmp_db.get_entity_documents(eid)
        assert len(docs) == 2
        # Verify doc_count was updated
        entity = tmp_db.get_entity_by_name("Alice")
        assert entity["doc_count"] == 2

    def test_batch_empty_noop(self, tmp_db):
        tmp_db.insert_entity_mentions_batch([])  # should not raise


class TestInsertClosetTagsBatch:
    def test_batch_insert_single_doc(self, tmp_db):
        doc_id = tmp_db.insert_document("test.pdf", "/tmp/test.pdf")
        records = [
            (doc_id, "容器编排", "容器 编排", 0.9, "llm"),
            (doc_id, "微服务", "微服务", 0.8, "llm"),
        ]
        tmp_db.insert_closet_tags_batch(records)
        tags = tmp_db.get_doc_tags(doc_id)
        assert len(tags) == 2
        assert tags[0]["tag_text"] == "容器编排"

    def test_batch_insert_multi_doc(self, tmp_db):
        d1 = tmp_db.insert_document("a.pdf", "/tmp/a.pdf")
        d2 = tmp_db.insert_document("b.pdf", "/tmp/b.pdf")
        records = [
            (d1, "tag1", "token1", 0.9, "llm"),
            (d2, "tag2", "token2", 0.8, "llm"),
        ]
        tmp_db.insert_closet_tags_batch(records)
        assert len(tmp_db.get_doc_tags(d1)) == 1
        assert len(tmp_db.get_doc_tags(d2)) == 1

    def test_batch_replaces_existing(self, tmp_db):
        doc_id = tmp_db.insert_document("test.pdf", "/tmp/test.pdf")
        tmp_db.insert_closet_tags(doc_id, [(doc_id, "old", "old", 0.5, "llm")])
        tmp_db.insert_closet_tags_batch([(doc_id, "new", "new", 0.9, "llm")])
        tags = tmp_db.get_doc_tags(doc_id)
        assert len(tags) == 1
        assert tags[0]["tag_text"] == "new"

    def test_batch_empty_noop(self, tmp_db):
        tmp_db.insert_closet_tags_batch([])  # should not raise


class TestTokenizationCache:
    def setup_method(self):
        _TOKENIZE_CACHE.clear()

    def test_cache_hit(self, tmp_db):
        result1 = PageIndexDB._tokenize_query("人工智能技术")
        result2 = PageIndexDB._tokenize_query("人工智能技术")
        assert result1 == result2
        assert "人工智能技术" in _TOKENIZE_CACHE

    def test_cache_different_queries(self, tmp_db):
        result1 = PageIndexDB._tokenize_query("机器学习")
        result2 = PageIndexDB._tokenize_query("深度学习")
        assert "机器学习" in _TOKENIZE_CACHE
        assert "深度学习" in _TOKENIZE_CACHE

    def test_cache_ttl_expiry(self, tmp_db):
        _TOKENIZE_CACHE["test_query"] = (["old_tokens"], time.monotonic() - 600)
        result = PageIndexDB._tokenize_query("test_query")
        assert result != ["old_tokens"]  # should have been recomputed
        assert "test_query" in _TOKENIZE_CACHE

    def test_cache_max_eviction(self, tmp_db):
        # Fill cache to max
        for i in range(512):
            _TOKENIZE_CACHE[f"q{i}"] = (["tok"], time.monotonic())
        # One more should evict oldest
        PageIndexDB._tokenize_query("new_query")
        assert "new_query" in _TOKENIZE_CACHE
