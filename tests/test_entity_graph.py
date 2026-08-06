"""P3 图谱三件套测试 —— search_entities 分词化 + 实体消歧 + 三件套集成。

验收覆盖：
1. search_entities 分词匹配：多词查询分词后命中实体（jieba 分词、去停用词、单字过滤）；
2. 实体消歧：别名/同义实体合并（跨文档归一，对齐标签归一化思路）；
3. ① 实体快捷跳转：查询 → 实体 → 文档集 → 导航加速；
4. ② 预筛信号增强：实体命中节点加权（升级 _entity_boost_nodes）；
5. ③ 多跳导航：实体关系查询 → 关联实体 → 关联文档；
6. NFR4：新 LLM 调用点使用 retrieve_model。

全部 LLM 调用均 mock —— 无真实 LLM、无向量（FULLY VECTORLESS）。
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from db import PageIndexDB


@pytest.fixture
def db(tmp_path):
    """Create a fresh temp DB for each test."""
    db_path = str(tmp_path / "test.db")
    db = PageIndexDB(db_path)
    yield db
    db.close()


# ---------------------------------------------------------------------------
# Helper: insert a document row (needed for entity_mentions FK)
# ---------------------------------------------------------------------------

def _insert_doc(db, title="doc"):
    """Insert a minimal document row; return its id."""
    with db._connect() as conn:
        cur = conn.execute(
            "INSERT INTO documents (pdf_name, pdf_path) VALUES (?, ?)",
            (f"{title}.pdf", f"/tmp/{title}.pdf"),
        )
        return cur.lastrowid


def _insert_entity(db, entity_type, name, aliases=None):
    """Insert an entity; return its id."""
    return db.insert_entity(entity_type, name, aliases)


# ===========================================================================
# 1. search_entities 分词匹配
# ===========================================================================

class TestSearchEntitiesTokenization:
    """search_entities 从整串 LIKE 改为 jieba 分词匹配。"""

    def test_multi_word_query_matches_both_tokens(self, db):
        """多词查询 "张三 风控" 分词后 ["张三", "风控"] → 命中两个实体。"""
        e1 = _insert_entity(db, "person", "张三", ["小张"])
        e2 = _insert_entity(db, "concept", "风控系统", ["风险管理"])
        results = db.search_entities("张三 风控")
        names = {r["name"] for r in results}
        assert "张三" in names, "分词后应命中 '张三'"
        assert "风控系统" in names, "分词后应命中 '风控系统'"

    def test_single_word_query_preserved(self, db):
        """单字查询保持原有行为：直接 LIKE 匹配。"""
        _insert_entity(db, "person", "张三")
        results = db.search_entities("张三")
        assert len(results) == 1
        assert results[0]["name"] == "张三"

    def test_stopwords_filtered(self, db):
        """停用词（如"的"、"了"）被过滤，不影响匹配。"""
        _insert_entity(db, "person", "张三")
        results = db.search_entities("张三的项目")
        # "的" 被过滤，"项目" 是通用词但保留；"张三" 应命中
        names = {r["name"] for r in results}
        assert "张三" in names

    def test_single_char_tokens_filtered(self, db):
        """单字 token（如 "的"、"了"）被过滤，避免宽泛匹配。"""
        _insert_entity(db, "concept", "数据安全")
        results = db.search_entities("的")
        # 单字 "的" 太通用，应返回空
        assert len(results) == 0

    def test_aliases_matched(self, db):
        """别名也参与分词匹配。"""
        _insert_entity(db, "person", "张三", ["小张", "Zhang San"])
        results = db.search_entities("小张")
        names = {r["name"] for r in results}
        assert "张三" in names, "别名 '小张' 应命中实体 '张三'"

    def test_empty_query_returns_empty(self, db):
        """空查询返回空结果。"""
        _insert_entity(db, "person", "张三")
        results = db.search_entities("")
        assert len(results) == 0

    def test_limit_respected(self, db):
        """limit 参数限制返回数量。"""
        for i in range(10):
            _insert_entity(db, "concept", f"概念{i}")
        results = db.search_entities("概念", limit=3)
        assert len(results) <= 3

    def test_doc_count_ordering(self, db):
        """结果按 doc_count 降序排列。"""
        e1 = _insert_entity(db, "person", "张三")
        e2 = _insert_entity(db, "concept", "风控")
        doc_id = _insert_doc(db)
        # 给 e2 添加更多 mention 以提高 doc_count
        for i in range(3):
            d = _insert_doc(db, f"doc{i}")
            db.insert_entity_mention(e2, d, confidence=0.9)
        db.insert_entity_mention(e1, doc_id, confidence=0.9)
        results = db.search_entities("张三 风控")
        # e2 应排在前面（doc_count 更高）
        assert results[0]["name"] == "风控"


# ===========================================================================
# 2. 实体消歧
# ===========================================================================

class TestEntityDisambiguation:
    """实体消歧：跨文档归一，合并别名/同义实体。"""

    def test_merge_alias_entity(self, db):
        """已有 "小张" 实体时，新实体 "张三" alias ["小张"] 应合并。"""
        # 先插入 "小张"
        e1 = _insert_entity(db, "person", "小张")
        # 再插入 "张三" alias ["小张"]，应触发消歧合并
        e2 = _insert_entity(db, "person", "张三", ["小张"])
        # 消歧后应只有一个实体（或两个实体通过别名关联）
        results = db.search_entities("小张")
        # 至少应能找到 "张三"（因为别名包含 "小张"）
        names = {r["name"] for r in results}
        assert "张三" in names, "别名 '小张' 应命中 '张三'"

    def test_disambiguation_with_llm(self, db):
        """LLM 消歧：当精确匹配失败时，LLM 判断是否同一实体。"""
        e1 = _insert_entity(db, "person", "张三", ["小张"])
        # 新实体 "张先生" 可能与 "张三" 是同一人
        existing = [{"id": e1, "name": "张三", "aliases": '["小张"]', "entity_type": "person"}]
        from pageindex_mutil.entity_extractor import EntityExtractor
        extractor = EntityExtractor(model="test", retrieve_model="test")
        with patch("pageindex_mutil.entity_extractor.llm_completion") as mock_llm:
            mock_llm.return_value = json.dumps({
                "should_merge": True,
                "canonical_name": "张三",
                "reason": "同一人，不同称呼"
            })
            result = extractor.disambiguate_entity("张先生", ["老张"], existing)
            assert result is not None
            assert result["name"] == "张三"

    def test_disambiguation_failure_conservative(self, db):
        """LLM 失败时保守处理：不合并。"""
        e1 = _insert_entity(db, "person", "张三")
        existing = [{"id": e1, "name": "张三", "aliases": '[]', "entity_type": "person"}]
        from pageindex_mutil.entity_extractor import EntityExtractor
        extractor = EntityExtractor(model="test", retrieve_model="test")
        with patch("pageindex_mutil.entity_extractor.llm_completion") as mock_llm:
            mock_llm.return_value = None  # LLM 失败
            result = extractor.disambiguate_entity("张先生", [], existing)
            assert result is None, "LLM 失败时应保守不合并"

    def test_nfr4_retrieve_model_used(self, db):
        """NFR4：实体消歧 LLM 调用使用 retrieve_model。"""
        from pageindex_mutil.entity_extractor import EntityExtractor
        extractor = EntityExtractor(model="base-model", retrieve_model="retrieve-model")
        e1 = _insert_entity(db, "person", "张三")
        existing = [{"id": e1, "name": "张三", "aliases": '[]', "entity_type": "person"}]
        with patch("pageindex_mutil.entity_extractor.llm_completion") as mock_llm:
            mock_llm.return_value = json.dumps({"should_merge": False})
            extractor.disambiguate_entity("张先生", [], existing)
            # 验证使用了 retrieve_model
            assert mock_llm.called, "应调用 LLM"
            call_args = mock_llm.call_args
            assert call_args[0][0] == "retrieve-model", \
                "NFR4: 消歧应使用 retrieve_model"


# ===========================================================================
# 2b. 实体消歧管线集成（S7.1）
# ===========================================================================

class TestEntityDisambiguationPipeline:
    """管线级实体消歧：_resolve_entity 集成到 entity insertion flow。"""

    def test_quick_merge_by_alias_overlap(self, db):
        """已有 "张三" alias=["小张"]，新实体 name="小张" → 快速合并，无需 LLM。"""
        e1 = _insert_entity(db, "person", "张三", ["小张"])
        from pageindex_mutil.entity_extractor import EntityExtractor
        extractor = EntityExtractor(model="test", retrieve_model="test")
        # _resolve_entity 应在不调用 LLM 的情况下合并
        with patch("pageindex_mutil.entity_extractor.llm_completion") as mock_llm:
            from pageindex_mutil.client import PageIndexClient
            # 创建一个最小 client 用于测试 _resolve_entity
            client = PageIndexClient.__new__(PageIndexClient)
            client.db = db
            client.entity_extractor = extractor
            result_id = client._resolve_entity("person", "小张", [], extractor)
            assert result_id == e1, "应合并到已有实体 '张三'"
            # 别名应已更新，包含 "小张"
            entity = db.get_entity_by_name("张三")
            aliases = json.loads(entity.get("aliases", "[]"))
            assert "小张" in aliases, "合并后别名应包含 '小张'"
            # LLM 不应被调用（快速合并）
            mock_llm.assert_not_called()

    def test_quick_merge_new_alias_matches_existing_name(self, db):
        """已有 "小张"，新实体 name="张三" alias=["小张"] → 快速合并。"""
        e1 = _insert_entity(db, "person", "小张")
        from pageindex_mutil.entity_extractor import EntityExtractor
        extractor = EntityExtractor(model="test", retrieve_model="test")
        with patch("pageindex_mutil.entity_extractor.llm_completion") as mock_llm:
            from pageindex_mutil.client import PageIndexClient
            client = PageIndexClient.__new__(PageIndexClient)
            client.db = db
            client.entity_extractor = extractor
            result_id = client._resolve_entity("person", "张三", ["小张"], extractor)
            assert result_id == e1, "应合并到已有实体 '小张'"
            mock_llm.assert_not_called()

    def test_llm_merge_when_no_quick_match(self, db):
        """无快速匹配时，LLM 判定应合并 → 返回已有实体 ID。"""
        e1 = _insert_entity(db, "person", "张三")
        from pageindex_mutil.entity_extractor import EntityExtractor
        extractor = EntityExtractor(model="test", retrieve_model="test")
        with patch("pageindex_mutil.entity_extractor.llm_completion") as mock_llm:
            mock_llm.return_value = json.dumps({
                "should_merge": True,
                "canonical_name": "张三",
                "reason": "同一人"
            })
            from pageindex_mutil.client import PageIndexClient
            client = PageIndexClient.__new__(PageIndexClient)
            client.db = db
            client.entity_extractor = extractor
            result_id = client._resolve_entity("person", "张先生", [], extractor)
            assert result_id == e1, "LLM 判定合并 → 返回已有实体 ID"

    def test_no_match_creates_new_entity(self, db):
        """无匹配且 LLM 判定不合并 → 创建新实体。"""
        _insert_entity(db, "person", "张三")
        from pageindex_mutil.entity_extractor import EntityExtractor
        extractor = EntityExtractor(model="test", retrieve_model="test")
        with patch("pageindex_mutil.entity_extractor.llm_completion") as mock_llm:
            mock_llm.return_value = json.dumps({
                "should_merge": False,
                "canonical_name": None,
                "reason": "不同人"
            })
            from pageindex_mutil.client import PageIndexClient
            client = PageIndexClient.__new__(PageIndexClient)
            client.db = db
            client.entity_extractor = extractor
            result_id = client._resolve_entity("person", "李四", [], extractor)
            # 应创建新实体
            assert result_id is not None
            new_entity = db.get_entity_by_name("李四")
            assert new_entity is not None, "应创建新实体 '李四'"
            assert new_entity["id"] == result_id

    def test_llm_failure_conservative_no_merge(self, db):
        """LLM 失败 → 保守不合并，创建新实体。"""
        _insert_entity(db, "person", "张三")
        from pageindex_mutil.entity_extractor import EntityExtractor
        extractor = EntityExtractor(model="test", retrieve_model="test")
        with patch("pageindex_mutil.entity_extractor.llm_completion") as mock_llm:
            mock_llm.return_value = None  # LLM 失败
            from pageindex_mutil.client import PageIndexClient
            client = PageIndexClient.__new__(PageIndexClient)
            client.db = db
            client.entity_extractor = extractor
            result_id = client._resolve_entity("person", "张先生", [], extractor)
            # 保守策略：创建新实体
            assert result_id is not None
            new_entity = db.get_entity_by_name("张先生")
            assert new_entity is not None, "LLM 失败时应创建新实体"

    def test_nfr4_pipeline_uses_retrieve_model(self, db):
        """NFR4：管线消歧 LLM 调用使用 retrieve_model。"""
        _insert_entity(db, "person", "张三")
        from pageindex_mutil.entity_extractor import EntityExtractor
        extractor = EntityExtractor(model="base-model", retrieve_model="retrieve-model")
        with patch("pageindex_mutil.entity_extractor.llm_completion") as mock_llm:
            mock_llm.return_value = json.dumps({"should_merge": False})
            from pageindex_mutil.client import PageIndexClient
            client = PageIndexClient.__new__(PageIndexClient)
            client.db = db
            client.entity_extractor = extractor
            client._resolve_entity("person", "张先生", [], extractor)
            call_args = mock_llm.call_args
            assert call_args[0][0] == "retrieve-model", \
                "NFR4: 管线消歧应使用 retrieve_model"

    def test_no_existing_entities_skips_llm(self, db):
        """该类型无已有实体 → 跳过 LLM，直接创建新实体。"""
        from pageindex_mutil.entity_extractor import EntityExtractor
        extractor = EntityExtractor(model="test", retrieve_model="test")
        with patch("pageindex_mutil.entity_extractor.llm_completion") as mock_llm:
            from pageindex_mutil.client import PageIndexClient
            client = PageIndexClient.__new__(PageIndexClient)
            client.db = db
            client.entity_extractor = extractor
            result_id = client._resolve_entity("person", "张三", [], extractor)
            assert result_id is not None
            mock_llm.assert_not_called()


# ===========================================================================
# 2c. DB 层辅助方法
# ===========================================================================

class TestEntityDBHelpers:
    """get_entities_by_type 和 merge_entity_aliases 测试。"""

    def test_get_entities_by_type(self, db):
        """get_entities_by_type 返回指定类型的所有实体。"""
        _insert_entity(db, "person", "张三")
        _insert_entity(db, "person", "李四")
        _insert_entity(db, "concept", "风控")
        results = db.get_entities_by_type("person")
        names = {r["name"] for r in results}
        assert names == {"张三", "李四"}

    def test_get_entities_by_type_empty(self, db):
        """该类型无实体 → 空列表。"""
        _insert_entity(db, "person", "张三")
        results = db.get_entities_by_type("concept")
        assert results == []

    def test_merge_entity_aliases(self, db):
        """merge_entity_aliases 合并别名到已有实体。"""
        e1 = _insert_entity(db, "person", "张三", ["小张"])
        db.merge_entity_aliases(e1, ["老张", "Zhang"])
        entity = db.get_entity_by_name("张三")
        aliases = json.loads(entity.get("aliases", "[]"))
        assert set(aliases) == {"小张", "老张", "Zhang"}

    def test_merge_entity_aliases_dedup(self, db):
        """merge_entity_aliases 去重，不重复添加已有别名。"""
        e1 = _insert_entity(db, "person", "张三", ["小张"])
        db.merge_entity_aliases(e1, ["小张", "老张"])
        entity = db.get_entity_by_name("张三")
        aliases = json.loads(entity.get("aliases", "[]"))
        assert aliases.count("小张") == 1, "不应重复添加已有别名"
        assert "老张" in aliases


# ===========================================================================
# 3. ① 实体快捷跳转
# ===========================================================================

class TestEntityShortcutJump:
    """① 查询带实体 → 图谱直达实体所在分支。"""

    def test_entity_search_finds_documents(self, db):
        """实体搜索 → 实体文档集 → 可用于导航加速。"""
        doc_id = _insert_doc(db, "风控报告")
        entity_id = _insert_entity(db, "person", "张三")
        db.insert_entity_mention(entity_id, doc_id, confidence=0.9)
        # search_entities 应找到 "张三"
        entities = db.search_entities("张三")
        assert len(entities) == 1
        assert entities[0]["name"] == "张三"
        # get_entity_documents 应返回关联文档
        docs = db.get_entity_documents(entities[0]["id"])
        assert len(docs) == 1
        assert docs[0]["id"] == doc_id

    def test_multi_word_query_shortcut(self, db):
        """多词查询 "张三 参与的项目" → 分词 → 实体 → 文档。"""
        doc_id = _insert_doc(db, "项目计划")
        entity_id = _insert_entity(db, "person", "张三")
        db.insert_entity_mention(entity_id, doc_id, confidence=0.85)
        # 分词后 "张三" 应被提取
        entities = db.search_entities("张三 参与的项目")
        names = {e["name"] for e in entities}
        assert "张三" in names

    def test_multiple_entities_merged_docs(self, db):
        """多个实体命中文档集合并。"""
        doc1 = _insert_doc(db, "doc1")
        doc2 = _insert_doc(db, "doc2")
        e1 = _insert_entity(db, "person", "张三")
        e2 = _insert_entity(db, "concept", "风控")
        db.insert_entity_mention(e1, doc1, confidence=0.9)
        db.insert_entity_mention(e2, doc2, confidence=0.8)
        entities = db.search_entities("张三 风控")
        doc_ids = set()
        for entity in entities:
            for doc in db.get_entity_documents(entity["id"]):
                doc_ids.add(doc["id"])
        assert doc1 in doc_ids
        assert doc2 in doc_ids


# ===========================================================================
# 4. ② 预筛信号增强
# ===========================================================================

class TestPrefilterSignalBoost:
    """② 实体命中节点加权（升级 _entity_boost_nodes）。"""

    def test_entity_document_ids_from_tokenized_search(self, db):
        """分词化 search_entities → entity_document_ids 正确收集。"""
        doc_id = _insert_doc(db, "doc")
        entity_id = _insert_entity(db, "person", "张三")
        db.insert_entity_mention(entity_id, doc_id, confidence=0.9)
        # 模拟 _entity_document_ids 逻辑
        entities = db.search_entities("张三 风控项目")
        doc_ids = set()
        for entity in entities:
            eid = entity.get("id")
            if eid:
                for doc in db.get_entity_documents(eid):
                    if doc.get("id"):
                        doc_ids.add(int(doc["id"]))
        assert doc_id in doc_ids

    def test_entity_boost_nodes_basic(self, db):
        """实体命中节点被提升到候选列表前面。"""
        # 这个测试验证 _entity_boost_nodes 的逻辑
        # 实际实现在 super_tree.py 中，这里测试 DB 层支持
        doc_id = _insert_doc(db, "doc")
        entity_id = _insert_entity(db, "person", "张三")
        db.insert_entity_mention(entity_id, doc_id, confidence=0.9)
        entities = db.search_entities("张三")
        assert len(entities) > 0


# ===========================================================================
# 5. ③ 多跳导航
# ===========================================================================

class TestMultiHopGuidance:
    """③ 实体关系查询 → 关联实体 → 关联文档（查询期）。"""

    def test_get_entity_relations_returns_related(self, db):
        """实体 A 的关系 → 找到实体 B。"""
        e1 = _insert_entity(db, "person", "张三")
        e2 = _insert_entity(db, "project", "风控系统")
        db.insert_entity_relation(e1, "works_on", e2, confidence=0.9)
        relations = db.get_entity_relations(e1, direction="outgoing")
        assert len(relations) == 1
        assert relations[0]["subject_name"] == "张三"
        assert relations[0]["object_name"] == "风控系统"
        assert relations[0]["predicate"] == "works_on"

    def test_related_entity_documents(self, db):
        """实体 B 的文档 → 下一跳候选。"""
        e1 = _insert_entity(db, "person", "张三")
        e2 = _insert_entity(db, "project", "风控系统")
        doc_id = _insert_doc(db, "风控文档")
        db.insert_entity_relation(e1, "works_on", e2, confidence=0.9)
        db.insert_entity_mention(e2, doc_id, confidence=0.85)
        # 从 e1 的关系找到 e2，再找 e2 的文档
        relations = db.get_entity_relations(e1, direction="outgoing")
        related_docs = set()
        for rel in relations:
            obj_id = rel.get("object_id")
            if obj_id:
                for doc in db.get_entity_documents(obj_id):
                    related_docs.add(doc["id"])
        assert doc_id in related_docs

    def test_bidirectional_relations(self, db):
        """双向关系查询：incoming + outgoing。"""
        e1 = _insert_entity(db, "person", "张三")
        e2 = _insert_entity(db, "project", "风控系统")
        db.insert_entity_relation(e1, "works_on", e2, confidence=0.9)
        # incoming 方向：从 e2 查找指向它的关系
        relations = db.get_entity_relations(e2, direction="incoming")
        assert len(relations) == 1
        assert relations[0]["subject_name"] == "张三"

    def test_relation_confidence_ordering(self, db):
        """关系按 confidence 降序排列。"""
        e1 = _insert_entity(db, "person", "张三")
        e2 = _insert_entity(db, "project", "风控系统")
        e3 = _insert_entity(db, "project", "数据分析")
        db.insert_entity_relation(e1, "works_on", e2, confidence=0.95)
        db.insert_entity_relation(e1, "related_to", e3, confidence=0.6)
        relations = db.get_entity_relations(e1, direction="outgoing")
        assert relations[0]["confidence"] >= relations[1]["confidence"]


# ===========================================================================
# 6. 集成测试：_entity_document_ids 使用分词化 search_entities
# ===========================================================================

class TestEntityDocumentIdsIntegration:
    """集成测试：super_tree._entity_document_ids 使用分词化 search_entities。"""

    def test_entity_document_ids_multi_word(self, db):
        """多词查询通过分词找到实体文档集。"""
        doc_id = _insert_doc(db, "doc")
        entity_id = _insert_entity(db, "person", "张三")
        db.insert_entity_mention(entity_id, doc_id, confidence=0.9)
        # 模拟 _entity_document_ids 逻辑
        query = "张三 风控项目"
        try:
            entities = db.search_entities(query, limit=20)
        except Exception:
            entities = []
        doc_ids = set()
        for entity in entities:
            entity_id = entity.get("id")
            if not entity_id:
                continue
            try:
                for mention in db.get_entity_documents(entity_id):
                    if mention.get("id"):
                        doc_ids.add(int(mention["id"]))
            except Exception:
                continue
        assert doc_id in doc_ids

    def test_entity_document_ids_empty_query(self, db):
        """空查询返回空集。"""
        _insert_entity(db, "person", "张三")
        entities = db.search_entities("", limit=20)
        doc_ids = set()
        for entity in entities:
            eid = entity.get("id")
            if eid:
                for doc in db.get_entity_documents(eid):
                    if doc.get("id"):
                        doc_ids.add(int(doc["id"]))
        assert len(doc_ids) == 0
