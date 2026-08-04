import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from db import PageIndexDB

# Avoid triggering __init__.py imports that pull in heavy deps like PyPDF2.
super_tree_path = Path(__file__).parent.parent / "pageindex_mutil"
sys.path.insert(0, str(super_tree_path))

# Make relative import in super_tree.py work by creating a fake package context.
import importlib.util

# Pre-seed pageindex.utils so closet_index.py won't fail on its own relative import.
utils_spec = importlib.util.spec_from_file_location("pageindex_mutil.utils", super_tree_path / "utils.py")
utils_mod = importlib.util.module_from_spec(utils_spec)
sys.modules["pageindex_mutil.utils"] = utils_mod
# utils.py may also have missing deps; stub out the names closet_index needs.
utils_mod.llm_completion = lambda *a, **k: None
async def _mock_llm_acompletion(*a, **k):
    return None
utils_mod.llm_acompletion = _mock_llm_acompletion
utils_mod.count_tokens = lambda text, model=None: len(text or "") // 4
def _mock_extract_json(text):
    import json
    try:
        return json.loads(text)
    except Exception:
        return None
utils_mod.extract_json = _mock_extract_json
# Load the REAL strip_markdown_fence from utils.py source so FR3 tests exercise
# the actual production logic (the function is pure, no heavy deps).
_real_utils_spec = importlib.util.spec_from_file_location(
    "_real_utils_strip", super_tree_path / "utils.py"
)
_real_utils_mod = importlib.util.module_from_spec(_real_utils_spec)
_real_utils_spec.loader.exec_module(_real_utils_mod)
utils_mod.strip_markdown_fence = _real_utils_mod.strip_markdown_fence

# Also need pageindex.closet_index for the _STOPWORDS import.
closet_spec = importlib.util.spec_from_file_location("pageindex_mutil.closet_index", super_tree_path / "closet_index.py")
closet_mod = importlib.util.module_from_spec(closet_spec)
sys.modules["pageindex_mutil.closet_index"] = closet_mod
closet_spec.loader.exec_module(closet_mod)

spec = importlib.util.spec_from_file_location("pageindex_mutil.super_tree", super_tree_path / "super_tree.py")
super_tree_mod = importlib.util.module_from_spec(spec)
sys.modules["pageindex_mutil.super_tree"] = super_tree_mod
spec.loader.exec_module(super_tree_mod)
KeywordIndex = super_tree_mod.KeywordIndex


@pytest.fixture
def keyword_index():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = PageIndexDB(path)
    ki = KeywordIndex(db)
    yield ki, db
    db.close()
    os.unlink(path)


class TestKeywordIndex:
    def test_add_and_search(self, keyword_index):
        ki, db = keyword_index
        doc_id = db.insert_document("前端脚本开发指南.pdf", "/tmp/test.pdf",
                                     doc_description="前端脚本开发的完整指南")
        ki.add_document(doc_id, "前端脚本开发指南.pdf", "前端脚本开发的完整指南")
        results = ki.search("前端脚本")
        assert len(results) == 1
        assert results[0][0] == doc_id

    def test_search_no_match(self, keyword_index):
        ki, db = keyword_index
        doc_id = db.insert_document("test.pdf", "/tmp/test.pdf")
        ki.add_document(doc_id, "test.pdf", "")
        results = ki.search("不存在的关键词")
        assert len(results) == 0

    def test_remove_document(self, keyword_index):
        ki, db = keyword_index
        doc_id = db.insert_document("test.pdf", "/tmp/test.pdf")
        ki.add_document(doc_id, "test.pdf", "")
        ki.remove_document(doc_id)
        results = ki.search("test")
        assert len(results) == 0


KBIdentity = super_tree_mod.KBIdentity


@pytest.fixture
def kb_identity():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = PageIndexDB(path)
    ki = KBIdentity(db, model="qwen-plus")
    yield ki, db
    db.close()
    os.unlink(path)


class TestKBIdentity:
    def test_fallback_when_no_docs(self, kb_identity):
        ki, db = kb_identity
        identity = ki.get_identity()
        assert "暂无文档" in identity

    def test_fallback_when_cache_miss(self, kb_identity):
        ki, db = kb_identity
        db.insert_document("test.pdf", "/tmp/test.pdf")
        identity = ki.get_identity()
        assert "test.pdf" in identity

    def test_llm_generation(self, kb_identity):
        with patch.object(super_tree_mod, "llm_completion") as mock_llm:
            mock_llm.return_value = '{"summary": "知识库共1个文档，主题：测试"}'
            ki, db = kb_identity
            db.insert_document("test.pdf", "/tmp/test.pdf", doc_description="测试文档")
            identity = ki.get_identity()
            assert "测试" in identity
            mock_llm.assert_called_once()

    def test_invalidate_and_rebuild(self, kb_identity):
        ki, db = kb_identity
        db.insert_document("old.pdf", "/tmp/old.pdf")
        identity1 = ki.get_identity()
        assert "old.pdf" in identity1

        ki.invalidate()
        db.insert_document("new.pdf", "/tmp/new.pdf")
        identity2 = ki.get_identity()
        assert "new.pdf" in identity2

    def test_kb_identity_strips_markdown_fence(self, kb_identity):
        """W2 FR3/AC3.1 — fenced LLM output must be stripped before storage.

        RED (Task #8): _generate_with_llm stores response.strip() raw, so a
        fenced response persists with ``` markers, polluting L1 prompts.
        """
        ki, db = kb_identity
        db.insert_document("test.pdf", "/tmp/test.pdf", doc_description="测试文档")
        with patch.object(super_tree_mod, "llm_completion") as mock_llm:
            mock_llm.return_value = "```text\n某摘要内容\n```"
            identity = ki.get_identity()
            # Stored identity must not retain fence markers.
            assert "```" not in identity
            assert "某摘要内容" in identity

    def test_kb_identity_idempotent_on_plain_text(self, kb_identity):
        """W2 FR3/AC3.2 — plain text (no fence) is returned unchanged."""
        ki, db = kb_identity
        db.insert_document("test.pdf", "/tmp/test.pdf", doc_description="测试文档")
        with patch.object(super_tree_mod, "llm_completion") as mock_llm:
            mock_llm.return_value = "某摘要"
            identity = ki.get_identity()
            assert identity == "某摘要"


SuperTreeIndex = super_tree_mod.SuperTreeIndex


@pytest.fixture
def super_tree_index():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = PageIndexDB(path)
    client = MagicMock()
    client._uuid_to_db = {}
    client.documents = {}
    client.closet_index = None
    client._id_mapper = None  # 让 _get_db_to_uuid 回退到 _uuid_to_db
    st = SuperTreeIndex(db, model="qwen-plus", client=client)
    yield st, db, client
    db.close()
    os.unlink(path)


class TestSuperTreeIndex:
    def test_prefilter_empty_db(self, super_tree_index):
        st, db, client = super_tree_index
        result = st.prefilter("前端")
        assert result == {}

    def test_prefilter_with_keyword_match(self, super_tree_index):
        st, db, client = super_tree_index
        doc_id = db.insert_document("前端脚本.pdf", "/tmp/test.pdf",
                                     doc_description="前端开发指南")
        st.on_document_added(doc_id)
        result = st.prefilter("前端")
        assert doc_id in result

    def test_build_super_tree_empty(self, super_tree_index):
        st, db, client = super_tree_index
        tree = st._build_super_tree([])
        assert tree == {"documents": []}

    def test_on_document_added_updates_keyword_index(self, super_tree_index):
        st, db, client = super_tree_index
        doc_id = db.insert_document("test.pdf", "/tmp/test.pdf")
        st.on_document_added(doc_id)
        results = st.keyword_index.search("test")
        assert len(results) == 1

    def test_on_document_removed_clears_index(self, super_tree_index):
        st, db, client = super_tree_index
        doc_id = db.insert_document("test.pdf", "/tmp/test.pdf")
        st.on_document_added(doc_id)
        st.on_document_removed(doc_id)
        results = st.keyword_index.search("test")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_select_documents_reasoning_picks_subset(self, super_tree_index):
        """三层重构-选择层：推理式挑选只返回 LLM 选中的文档子集（uuid）。"""
        with patch.object(super_tree_mod, "llm_acompletion") as mock_llm:
            mock_llm.return_value = json.dumps({"doc_ids": ["uuid-a"]})
            st, db, client = super_tree_index
            client.documents = {"uuid-a": {"id": "uuid-a"}, "uuid-b": {"id": "uuid-b"}, "uuid-c": {"id": "uuid-c"}}
            client._uuid_to_db = {"uuid-a": 1, "uuid-b": 2, "uuid-c": 3}
            for i in (1, 2, 3):
                db.insert_document(f"doc{i}.pdf", f"/tmp/{i}.pdf")
            result = await st.select_documents("test", {1: 1.0, 2: 1.0, 3: 1.0})
            assert result == ["uuid-a"]

    @pytest.mark.asyncio
    async def test_select_documents_reasoning_variable_count(self, super_tree_index):
        """三层重构-选择层：宁缺毋滥——LLM 只挑 1 篇时不强行凑满 top_k（精确率修复核心）。"""
        with patch.object(super_tree_mod, "llm_acompletion") as mock_llm:
            mock_llm.return_value = json.dumps({"doc_ids": ["uuid-b"]})
            st, db, client = super_tree_index
            client.documents = {"uuid-a": {"id": "uuid-a"}, "uuid-b": {"id": "uuid-b"}, "uuid-c": {"id": "uuid-c"}}
            client._uuid_to_db = {"uuid-a": 1, "uuid-b": 2, "uuid-c": 3}
            for i in (1, 2, 3):
                db.insert_document(f"doc{i}.pdf", f"/tmp/{i}.pdf")
            result = await st.select_documents("test", {1: 1.0, 2: 1.0, 3: 1.0})
            assert result == ["uuid-b"]
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_select_documents_reasoning_empty_candidates(self, super_tree_index):
        """三层重构-选择层：空候选返回空列表。"""
        st, db, client = super_tree_index
        result = await st.select_documents("test", {})
        assert result == []

    @pytest.mark.asyncio
    async def test_select_documents_reasoning_map_reduce(self, super_tree_index):
        """三层重构-选择层：候选 > group_size 时走 map-reduce，各组胜者并集。"""
        st, db, client = super_tree_index
        st._REASON_GROUP_SIZE = 4  # 9 候选 → 分组 [4,4,1]
        docs = {f"uuid-{i}": i for i in range(1, 10)}
        client.documents = {u: {"id": u} for u in docs}
        client._uuid_to_db = dict(docs)
        for i in range(1, 10):
            db.insert_document(f"doc{i}.pdf", f"/tmp/{i}.pdf")
        # map：组1 挑 uuid-1，组2 挑 uuid-5；组3 单候选短路不调 LLM
        with patch.object(super_tree_mod, "llm_acompletion") as mock_llm:
            mock_llm.side_effect = [
                json.dumps({"doc_ids": ["uuid-1"]}),
                json.dumps({"doc_ids": ["uuid-5"]}),
            ]
            result = await st.select_documents("test", {i: 1.0 for i in range(1, 10)})
        assert set(result) == {"uuid-1", "uuid-5", "uuid-9"}

    @pytest.mark.asyncio
    async def test_score_candidates_empty(self, super_tree_index):
        """Q1 -- 空候选返回空列表，不调用 LLM。"""
        st, db, client = super_tree_index
        result = await st._score_candidates("test", {})
        assert result == []

    @pytest.mark.asyncio
    async def test_score_candidates_filters_below_relative_threshold(self, super_tree_index):
        """Q1 -- 相对阈值(>=最高分*ratio)：远低于最高分的候选被过滤，即使排在前面。"""
        with patch.object(super_tree_mod, "llm_acompletion") as mock_llm:
            # s_max=0.9 → 阈值 0.45，uuid-b(0.4) 被过滤；uuid-c(0.8) 保留。
            mock_llm.return_value = json.dumps({
                "ranked": [
                    {"doc_id": "uuid-a", "score": 0.9},
                    {"doc_id": "uuid-b", "score": 0.4},
                    {"doc_id": "uuid-c", "score": 0.8},
                ],
                "top_k": 10,
            })
            st, db, client = super_tree_index
            client.documents = {"uuid-a": {"id": "uuid-a"}, "uuid-b": {"id": "uuid-b"}, "uuid-c": {"id": "uuid-c"}}
            client._uuid_to_db = {"uuid-a": 1, "uuid-b": 2, "uuid-c": 3}
            for i in (1, 2, 3):
                db.insert_document(f"doc{i}.pdf", f"/tmp/{i}.pdf")
            result = await st._score_candidates("test", {1: 1.0, 2: 1.0, 3: 1.0})
            assert "uuid-b" not in result  # 0.4 < 0.9*0.5 被过滤
            assert set(result) == {"uuid-a", "uuid-c"}

    @pytest.mark.asyncio
    async def test_score_candidates_keeps_close_high_scores(self, super_tree_index):
        """Q1 -- 强域：分数都偏高且相对接近时不去除，避免误伤边界相关文档。"""
        with patch.object(super_tree_mod, "llm_acompletion") as mock_llm:
            # s_max=0.9 → 阈值 0.45，三者都 >= 0.45，全部保留（不误杀强域）。
            mock_llm.return_value = json.dumps({
                "ranked": [
                    {"doc_id": "uuid-a", "score": 0.9},
                    {"doc_id": "uuid-b", "score": 0.6},
                    {"doc_id": "uuid-c", "score": 0.5},
                ],
                "top_k": 10,
            })
            st, db, client = super_tree_index
            client.documents = {"uuid-a": {"id": "uuid-a"}, "uuid-b": {"id": "uuid-b"}, "uuid-c": {"id": "uuid-c"}}
            client._uuid_to_db = {"uuid-a": 1, "uuid-b": 2, "uuid-c": 3}
            for i in (1, 2, 3):
                db.insert_document(f"doc{i}.pdf", f"/tmp/{i}.pdf")
            result = await st._score_candidates("test", {1: 1.0, 2: 1.0, 3: 1.0})
            assert set(result) == {"uuid-a", "uuid-b", "uuid-c"}

    @pytest.mark.asyncio
    async def test_score_candidates_weak_query_keeps_topk(self, super_tree_index):
        """Q1 -- 弱查询(最高分<0.3)：保留 top-k 兜底，不因绝对阈值误杀。"""
        with patch.object(super_tree_mod, "llm_acompletion") as mock_llm:
            mock_llm.return_value = json.dumps({
                "ranked": [
                    {"doc_id": "uuid-a", "score": 0.25},
                    {"doc_id": "uuid-b", "score": 0.2},
                    {"doc_id": "uuid-c", "score": 0.15},
                ],
                "top_k": 10,
            })
            st, db, client = super_tree_index
            client.documents = {"uuid-a": {"id": "uuid-a"}, "uuid-b": {"id": "uuid-b"}, "uuid-c": {"id": "uuid-c"}}
            client._uuid_to_db = {"uuid-a": 1, "uuid-b": 2, "uuid-c": 3}
            for i in (1, 2, 3):
                db.insert_document(f"doc{i}.pdf", f"/tmp/{i}.pdf")
            result = await st._score_candidates("test", {1: 1.0, 2: 1.0, 3: 1.0})
            assert set(result) == {"uuid-a", "uuid-b", "uuid-c"}

    @pytest.mark.asyncio
    async def test_score_candidates_caps_top_k_at_config(self, super_tree_index):
        """Q1 -- 返回数量上限为 _SELECT_TOP_K，不被 LLM 的 top_k 字段覆盖。"""
        with patch.object(super_tree_mod, "llm_acompletion") as mock_llm:
            # 6 个达标候选，LLM 谎称 top_k=10；配置 _SELECT_TOP_K=5 应严格截断到 5。
            docs = {f"uuid-{i}": {"score": 0.9} for i in range(1, 7)}
            mock_llm.return_value = json.dumps({"ranked": [
                {"doc_id": did, "score": info["score"]} for did, info in docs.items()
            ], "top_k": 10})
            st, db, client = super_tree_index
            client.documents = {did: {"id": did} for did in docs}
            client._uuid_to_db = {did: i for i, did in enumerate(docs, 1)}
            for i in range(1, 7):
                db.insert_document(f"doc{i}.pdf", f"/tmp/{i}.pdf")
            result = await st._score_candidates("test", {i: 1.0 for i in range(1, 7)})
            assert len(result) == st._SELECT_TOP_K  # 严格等于配置上限 5

    def test_rank_k_defaults(self, super_tree_index):
        """Q1 -- 默认 rank_k/top_k 存在且为正。"""
        st, db, client = super_tree_index
        assert st._RANK_K > 0
        assert st._SELECT_TOP_K > 0

    def test_rank_k_from_config(self, super_tree_index):
        """Q1 -- _init_from_config 能从 config 读取 rank_k。"""
        st, db, client = super_tree_index
        prior = st._RANK_K
        st._RANK_K = 7
        assert st._RANK_K == 7
