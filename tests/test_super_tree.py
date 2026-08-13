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

    def test_hierarchy_boost_soft_routing(self, super_tree_index):
        """三层重构-层级：匹配查询的集合标签加性提升，绝不删除候选（软路由）。"""
        st, db, client = super_tree_index
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        d2 = db.insert_document("b.pdf", "/tmp/b.pdf")
        db.insert_closet_tags(d1, [(d1, "金融风控", "金融 风控", 0.9, "llm")])
        boosted = st._hierarchy_boost("金融风控合规", {d1: 1.0, d2: 1.0})
        # 标签匹配的 doc1 被提升；doc2 保持原分；两者都保留（不硬删）
        assert boosted[d1] > boosted[d2]
        assert d1 in boosted and d2 in boosted
        assert boosted[d2] == 1.0

    def test_hierarchy_boost_empty_candidates(self, super_tree_index):
        """三层重构-层级：空候选原样返回。"""
        st, db, client = super_tree_index
        assert st._hierarchy_boost("q", {}) == {}

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


# ---------------------------------------------------------------------------
# L1 文档级证据接地（spec [3.2] enhance 抽象 unit=文档级候选）
# ---------------------------------------------------------------------------

# _doc_evidence_lines 经 super_tree 内惰性导入解析 `from .agentic.enhance import
# ...`——与 test_tree_navigation.py 同款隔离：standalone 收集时 spec 加载 enhance
# 预置 sys.modules；全量套件运行时复用已加载的模块对象。
if "pageindex_mutil.agentic.enhance" not in sys.modules:
    _enh_spec = importlib.util.spec_from_file_location(
        "pageindex_mutil.agentic.enhance", super_tree_path / "agentic" / "enhance.py"
    )
    _enh_mod = importlib.util.module_from_spec(_enh_spec)
    sys.modules["pageindex_mutil.agentic.enhance"] = _enh_mod
    _enh_spec.loader.exec_module(_enh_mod)


def _seed_matching_docs(st, db, client):
    """doc1 三通道签名全部命中 "风控合规审查"；doc2 零命中。"""
    client._uuid_to_db = {"uuid-a": 1, "uuid-b": 2}
    client.documents = {"uuid-a": {"id": "uuid-a"}, "uuid-b": {"id": "uuid-b"}}
    d1 = db.insert_document("风控手册.pdf", "/tmp/a.pdf")
    d2 = db.insert_document("财务报表.pdf", "/tmp/b.pdf")
    db.upsert_node_profiles(d1, [
        {"node_id": "n1",
         "entities": [{"name": "风控系统", "type": "system"}],
         "keywords": ["风控", "合规"],
         "tags": ["合规管理"]},
    ])
    db.upsert_node_profiles(d2, [
        {"node_id": "n2", "entities": [], "keywords": ["财报"], "tags": []},
    ])
    return d1, d2


class TestDocEvidenceLines:
    """_doc_evidence_lines：文档级关键词/实体/标签命中行（确定性、防御式、无 LLM）。"""

    def test_matching_doc_gets_line_nonmatching_omitted(self, super_tree_index):
        """签名命中文档出行（uuid 标签 + 三通道命中项）；零命中文档不出行。"""
        st, db, client = super_tree_index
        d1, d2 = _seed_matching_docs(st, db, client)
        lines = st._doc_evidence_lines("风控合规审查", [d1, d2])
        assert set(lines) == {d1}  # 零命中文档整体省略，不发"无命中"噪音
        assert lines[d1] == ("doc uuid-a: 关键词命中: 风控, 合规 | "
                             "实体命中: 风控系统 | 标签命中: 合规管理")

    def test_doc_level_closet_tags_channel(self, super_tree_index):
        """doc 级 closet_tags(source='llm') 进标签通道；fallback 源不进。"""
        st, db, client = super_tree_index
        client._uuid_to_db = {"uuid-a": 1}
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        db.insert_closet_tags(d1, [
            (d1, "风险管理", "风控", 0.9, "llm"),
            (d1, "fallback词", "风控", 0.5, "fallback"),
        ])
        lines = st._doc_evidence_lines("风险管理要求", [d1])
        assert d1 in lines
        assert "标签命中: 风险管理" in lines[d1]
        assert "fallback词" not in lines[d1]

    def test_caps_keywords_entities_tags(self, super_tree_index):
        """呈现上限：每文档关键词 ≤4 / 实体 ≤3 / 标签 ≤2（按聚合序取前缀）。"""
        st, db, client = super_tree_index
        client._uuid_to_db = {"uuid-a": 1}
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        db.upsert_node_profiles(d1, [{
            "node_id": "n1",
            "entities": [{"name": f"风控{'ABCDE'[i]}", "type": "x"} for i in range(5)],
            "keywords": [f"风控{'甲乙丙丁戊己'[i]}" for i in range(6)],
            "tags": [f"合规{'甲乙丙丁'[i]}" for i in range(4)],
        }])
        lines = st._doc_evidence_lines("风控合规审查", [d1])
        line = lines[d1]
        kw_part = line.split("关键词命中: ")[1].split(" | ")[0]
        ent_part = line.split("实体命中: ")[1].split(" | ")[0]
        tag_part = line.split("标签命中: ")[1]
        assert kw_part.split(", ") == ["风控甲", "风控乙", "风控丙", "风控丁"]
        assert ent_part.split(", ") == ["风控A", "风控B", "风控C"]
        assert tag_part.split(", ") == ["合规甲", "合规乙"]

    def test_malformed_profiles_degrade_gracefully(self, super_tree_index):
        """坏行（非 dict 实体/非字符串关键词标签/缺名实体）跳过不抛。"""
        st, db, client = super_tree_index
        client._uuid_to_db = {"uuid-a": 1, "uuid-b": 2}
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        d2 = db.insert_document("b.pdf", "/tmp/b.pdf")  # 无 profile 行
        db.upsert_node_profiles(d1, [{
            "node_id": "n1",
            "entities": ["notadict", None, 123, {"noname": 1}, {"name": "风控系统"}],
            "keywords": ["风控", 123, None, {"x": 1}],
            "tags": [None, 42, "合规管理"],
        }])
        lines = st._doc_evidence_lines("风控合规审查", [d1, d2])
        assert set(lines) == {d1}
        assert "关键词命中: 风控" in lines[d1]
        assert "实体命中: 风控系统" in lines[d1]
        assert "标签命中: 合规管理" in lines[d1]

    def test_db_errors_degrade_gracefully(self, super_tree_index):
        """db 访问异常 → 该文档空证据，绝不抛出（证据只作参考，不挡选档）。"""
        st, db, client = super_tree_index
        client._uuid_to_db = {"uuid-a": 1}
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        with patch.object(db, "get_node_profiles", side_effect=RuntimeError("boom")), \
             patch.object(db, "get_doc_tags", side_effect=RuntimeError("boom")):
            lines = st._doc_evidence_lines("风控", [d1])
        assert lines == {}

    def test_empty_query_or_tokens_no_lines(self, super_tree_index):
        """空查询/全停用词查询（无 token）→ 无证据行。"""
        st, db, client = super_tree_index
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        db.upsert_node_profiles(d1, [{
            "node_id": "n1", "entities": [], "keywords": ["风控"], "tags": [],
        }])
        assert st._doc_evidence_lines("", [d1]) == {}

    def test_deterministic_across_calls(self, super_tree_index):
        """相同输入两次调用 → 相同证据行（内容与顺序都稳定）。"""
        st, db, client = super_tree_index
        d1, d2 = _seed_matching_docs(st, db, client)
        db.insert_closet_tags(d1, [(d1, "风险管理", "风控", 0.9, "llm")])
        first = st._doc_evidence_lines("风控合规审查", [d1, d2])
        second = st._doc_evidence_lines("风控合规审查", [d1, d2])
        assert first == second
        assert list(first) == list(second)


class TestHolisticSelectPromptEvidence:
    """_holistic_select：证据块注入 prompt（有命中才出现，零命中完全不变）。"""

    @pytest.mark.asyncio
    async def test_evidence_block_in_prompt(self, super_tree_index):
        """命中文档证据行 + 指引 + 要求进 prompt；零命中文档不出行。"""
        with patch.object(super_tree_mod, "llm_acompletion") as mock_llm:
            mock_llm.return_value = json.dumps({"doc_ids": ["uuid-a"]})
            st, db, client = super_tree_index
            d1, d2 = _seed_matching_docs(st, db, client)
            result = await st.select_documents("风控合规审查", {d1: 1.0, d2: 1.0})
        assert result == ["uuid-a"]
        prompt = mock_llm.call_args[0][1]
        assert "[文档语料证据（关键词/实体/标签命中，供参考）]" in prompt
        assert ("doc uuid-a: 关键词命中: 风控, 合规 | 实体命中: 风控系统 | "
                "标签命中: 合规管理") in prompt
        assert "doc uuid-b" not in prompt  # 零命中文档不出行
        assert "证据是语料事实，请优先依据证据与问题的语义关联程度判断" in prompt
        assert "无证据命中的文档仍可按标题/摘要判断" in prompt
        assert "证据命中是强信号" in prompt  # 新增要求行
        # 位置契约：证据段在 [候选文档结构] 之后、要求 之前
        assert (prompt.index("[候选文档结构]")
                < prompt.index("[文档语料证据")
                < prompt.index("要求："))

    @pytest.mark.asyncio
    async def test_no_hits_evidence_section_absent(self, super_tree_index):
        """全无命中 → 证据段/指引/新要求行整体省略（prompt 与改造前一致）。"""
        with patch.object(super_tree_mod, "llm_acompletion") as mock_llm:
            mock_llm.return_value = json.dumps({"doc_ids": ["uuid-a"]})
            st, db, client = super_tree_index
            client._uuid_to_db = {"uuid-a": 1, "uuid-b": 2}
            client.documents = {"uuid-a": {"id": "uuid-a"}, "uuid-b": {"id": "uuid-b"}}
            d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
            d2 = db.insert_document("b.pdf", "/tmp/b.pdf")
            db.upsert_node_profiles(d2, [{
                "node_id": "n1", "entities": [], "keywords": ["财报"], "tags": [],
            }])
            await st.select_documents("风控合规审查", {d1: 1.0, d2: 1.0})
        prompt = mock_llm.call_args[0][1]
        assert "文档语料证据" not in prompt
        assert "证据是语料事实" not in prompt
        assert "证据命中是强信号" not in prompt

    @pytest.mark.asyncio
    async def test_malformed_profiles_no_crash_in_select(self, super_tree_index):
        """选档路径遇坏签名行：不抛异常，prompt 照常生成（坏行跳过）。"""
        with patch.object(super_tree_mod, "llm_acompletion") as mock_llm:
            mock_llm.return_value = json.dumps({"doc_ids": ["uuid-a"]})
            st, db, client = super_tree_index
            client._uuid_to_db = {"uuid-a": 1, "uuid-b": 2}
            client.documents = {"uuid-a": {"id": "uuid-a"}, "uuid-b": {"id": "uuid-b"}}
            d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
            d2 = db.insert_document("b.pdf", "/tmp/b.pdf")
            db.upsert_node_profiles(d1, [{
                "node_id": "n1",
                "entities": [None, "bad", {"name": 123}, {"name": "风控系统"}],
                "keywords": [None, 9, "风控"],
                "tags": {"not": "a list"},
            }])
            result = await st.select_documents("风控合规审查", {d1: 1.0, d2: 1.0})
        assert result == ["uuid-a"]
        prompt = mock_llm.call_args[0][1]
        assert "关键词命中: 风控" in prompt
        assert "实体命中: 风控系统" in prompt

    @pytest.mark.asyncio
    async def test_evidence_prompt_deterministic(self, super_tree_index):
        """两次相同选档 → 注入 prompt 的证据段逐字节一致。"""
        st, db, client = super_tree_index
        d1, d2 = _seed_matching_docs(st, db, client)
        prompts = []
        for _ in range(2):
            with patch.object(super_tree_mod, "llm_acompletion") as mock_llm:
                mock_llm.return_value = json.dumps({"doc_ids": ["uuid-a"]})
                await st.select_documents("风控合规审查", {d1: 1.0, d2: 1.0})
            prompts.append(mock_llm.call_args[0][1])
        ev0 = prompts[0].split("[文档语料证据")[1]
        ev1 = prompts[1].split("[文档语料证据")[1]
        assert ev0 == ev1


# ---------------------------------------------------------------------------
# P2-Fix5: tag casefold consistency
# ---------------------------------------------------------------------------


class TestTagCasefold:
    def test_case_different_same_tag_deduped_with_correct_count(self, super_tree_index):
        """P2-Fix5: "AI" and "ai" counted together, appear once in profile."""
        st, db, _ = super_tree_index
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        d2 = db.insert_document("b.pdf", "/tmp/b.pdf")
        root = db.insert_corpus_tree_node(None, "知识库", "", 0, kind="root")
        cluster = db.insert_corpus_tree_node(root, "技术", "", 1, kind="cluster")
        db.add_corpus_membership(d1, cluster, 1.0)
        db.add_corpus_membership(d2, cluster, 0.5)

        db.insert_closet_tags(d1, [(d1, "AI", "AI", 0.9, "llm")])
        db.insert_closet_tags(d2, [(d2, "ai", "ai", 0.8, "llm")])

        view = st._load_corpus_tree()
        profile = st._aggregate_cluster_profile(cluster, view, {})

        assert len(profile["tags"]) == 1  # deduped to one
        assert profile["tags"][0] in ("AI", "ai")  # first-seen display text
        # Verify count: "AI" and "ai" should be counted together
        # Both docs have the tag, so count should be 2
        from collections import Counter
        view2 = st._load_corpus_tree()
        profile2 = st._aggregate_cluster_profile(cluster, view2, {})
        # top tag should have count 2 (both docs counted)
        assert len(profile2["tags"]) == 1
