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
            result, reasons = await st.select_documents("test", {1: 1.0, 2: 1.0, 3: 1.0})
            assert result == ["uuid-a"]
            assert reasons == {}

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
            result, reasons = await st.select_documents("test", {1: 1.0, 2: 1.0, 3: 1.0})
            assert result == ["uuid-b"]
            assert len(result) == 1
            assert reasons == {}

    @pytest.mark.asyncio
    async def test_select_documents_reasoning_empty_candidates(self, super_tree_index):
        """三层重构-选择层：空候选返回空列表。"""
        st, db, client = super_tree_index
        result, reasons = await st.select_documents("test", {})
        assert result == []
        assert reasons == {}

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
            result, reasons = await st.select_documents("test", {i: 1.0 for i in range(1, 10)})
        assert set(result) == {"uuid-1", "uuid-5", "uuid-9"}


# ---------------------------------------------------------------------------
# L1 文档级证据接地（evidence_bundle 直通 → prompt 证据块）
# ---------------------------------------------------------------------------

# _build_doc_entries 经 super_tree 内惰性导入解析 `from .agentic.evidence import
# ...`——standalone 收集时 spec 加载 evidence 预置 sys.modules；全量套件运行时
# 复用已加载的模块对象。
if "pageindex_mutil.agentic.evidence" not in sys.modules:
    _ev_spec = importlib.util.spec_from_file_location(
        "pageindex_mutil.agentic.evidence", super_tree_path / "agentic" / "evidence.py"
    )
    _ev_mod = importlib.util.module_from_spec(_ev_spec)
    sys.modules["pageindex_mutil.agentic.evidence"] = _ev_mod
    _ev_spec.loader.exec_module(_ev_mod)


def _seed_matching_docs(st, db, client):
    """doc1 三通道证据束命中；doc2 零命中。返回 (d1, d2, evidence_bundle)。"""
    client._uuid_to_db = {"uuid-a": 1, "uuid-b": 2}
    client.documents = {"uuid-a": {"id": "uuid-a"}, "uuid-b": {"id": "uuid-b"}}
    d1 = db.insert_document("风控手册.pdf", "/tmp/a.pdf")
    d2 = db.insert_document("财务报表.pdf", "/tmp/b.pdf")
    # 证据束直通（T13）：L1 证据块只来自证据束，不再有 _doc_evidence_lines 重算。
    bundle = {
        d1: {"channels": {
                 "keyword": [{"token": "风控", "field": "node_title"},
                             {"token": "合规", "field": "node_title"}],
                 "entity": [{"name": "风控系统", "type": "system"}],
                 "tag": [{"text": "合规管理", "confidence": 0.9}],
                 "vector": []},
             "graph": {"doc_entity_links": []}},
    }
    return d1, d2, bundle


class TestHolisticSelectPromptEvidence:
    """_holistic_select：证据（有命中才出现）+ 呈现单元（名称 + 摘要 + 证据行）。"""

    @pytest.mark.asyncio
    async def test_evidence_block_in_prompt(self, super_tree_index):
        """命中文档证据行进 prompt（呈现单元 JSON）；零命中文档无证据字段。"""
        with patch.object(super_tree_mod, "llm_acompletion") as mock_llm:
            mock_llm.return_value = json.dumps({"doc_ids": ["uuid-a"]})
            st, db, client = super_tree_index
            d1, d2, bundle = _seed_matching_docs(st, db, client)
            result, _ = await st.select_documents(
                "风控合规审查", {d1: 1.0, d2: 1.0}, evidence_bundle=bundle)
        assert result == ["uuid-a"]
        prompt = mock_llm.call_args[0][1]
        # 呈现单元：文档名 + 摘要 + 证据行（doc_summary 列未建 → doc_description 兜底为空）
        assert '"doc_id": "uuid-a"' in prompt
        assert ("关键词命中: 风控(node_title), 合规(node_title) | 实体命中: 风控系统（system） | "
                "标签命中: 合规管理") in prompt
        assert "证据是语料事实，请优先依据证据与问题的语义关联程度判断" in prompt
        assert "证据命中是强信号" in prompt  # 新增要求行（证据段）
        # 位置契约：证据段（候选文档结构）在 要求 之前
        assert prompt.index("[候选文档结构]") < prompt.index("要求：")

    @pytest.mark.asyncio
    async def test_prompt_drops_contradictory_instructions(self, super_tree_index):
        """[S6]#3：删除"基于标题摘要判断"引导与"可以少选甚至不选"判空授权句；
        保留"宁缺毋滥"；证据段固定句不带"仍可按标题/摘要判断"尾巴。"""
        with patch.object(super_tree_mod, "llm_acompletion") as mock_llm:
            mock_llm.return_value = json.dumps({"doc_ids": ["uuid-a"]})
            st, db, client = super_tree_index
            d1, d2, bundle = _seed_matching_docs(st, db, client)
            await st.select_documents("风控合规审查", {d1: 1.0, d2: 1.0},
                                      evidence_bundle=bundle)
        prompt = mock_llm.call_args[0][1]
        assert "可以少选甚至不选" not in prompt
        assert "基于文档的章节标题和摘要判断相关性" not in prompt
        assert "无证据命中的文档仍可按标题/摘要判断" not in prompt
        assert "宁缺毋滥" in prompt  # 保留
        assert ("证据是语料事实，请优先依据证据与问题的语义关联程度判断，"
                "而非简单计数命中个数。" in prompt)

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
            await st.select_documents("风控合规审查", {d1: 1.0, d2: 1.0})
        prompt = mock_llm.call_args[0][1]
        assert "证据是语料事实" not in prompt
        assert "证据命中是强信号" not in prompt

    @pytest.mark.asyncio
    async def test_evidence_prompt_deterministic(self, super_tree_index):
        """两次相同选档 → 注入 prompt 的候选文档块逐字节一致。"""
        st, db, client = super_tree_index
        d1, d2, bundle = _seed_matching_docs(st, db, client)
        prompts = []
        for _ in range(2):
            with patch.object(super_tree_mod, "llm_acompletion") as mock_llm:
                mock_llm.return_value = json.dumps({"doc_ids": ["uuid-a"]})
                await st.select_documents("风控合规审查", {d1: 1.0, d2: 1.0},
                                          evidence_bundle=bundle)
            prompts.append(mock_llm.call_args[0][1])
        assert prompts[0] == prompts[1]


class TestL1SelectKeep:
    """[S6]#6：终选 keep 上限取 l1_select_keep（默认 10，可配）。"""

    def test_l1_select_keep_default_ten(self, super_tree_index):
        st, db, client = super_tree_index
        assert st._L1_SELECT_KEEP == 10

    @pytest.mark.asyncio
    async def test_l1_keep_configurable(self, super_tree_index):
        st, db, client = super_tree_index
        st._L1_SELECT_KEEP = 3
        docs = {f"uuid-{i}": i for i in range(1, 6)}
        client._uuid_to_db = docs
        client.documents = {u: {"id": u} for u in docs}
        for i in range(1, 6):
            db.insert_document(f"doc{i}.pdf", f"/tmp/{i}.pdf")
        with patch.object(super_tree_mod, "llm_acompletion") as mock_llm:
            mock_llm.return_value = json.dumps({"doc_ids": ["uuid-1"]})
            await st.select_documents("test", {i: 1.0 for i in range(1, 6)})
        prompt = mock_llm.call_args[0][1]
        assert "最多 3 篇" in prompt  # keep 上限来自 l1_select_keep


class TestBudgetNoPop:
    """[S6]#4：预算超限先截摘要/退化弱候选，绝不静默 pop 文档。"""

    @pytest.mark.asyncio
    async def test_budget_truncates_summary_not_pop_docs(self, super_tree_index):
        st, db, client = super_tree_index
        st._MAX_SUPER_TREE_TOKENS = 30  # 极低预算，强制截短/降级
        n = 8
        docs = {}
        for i in range(n):
            did = db.insert_document(f"doc{i}.pdf", "/tmp/{i}.pdf",
                                     doc_description="摘要" * 200)
            docs[f"uuid-{i}"] = did
        client._uuid_to_db = docs
        client.documents = {u: {"id": u} for u in docs}
        with patch.object(super_tree_mod, "llm_acompletion") as mock_llm:
            mock_llm.return_value = json.dumps({"doc_ids": ["uuid-0"]})
            result, _ = await st.select_documents("查询", {d: 1.0 for d in docs.values()})
        prompt = mock_llm.call_args[0][1]
        # 不 pop 文档：全部候选名称仍在呈现中
        for i in range(n):
            assert f"doc{i}.pdf" in prompt
        # 完整 400 字符摘要已截短/清空（降级行只保留名称 + 证据摘要）
        assert "摘要" * 200 not in prompt


class TestReasonsNormalization:
    """[S6]#7：reasons 键规范化（db_id 键 → uuid；uuid 键原样；只含 keep 内条目）。"""

    @pytest.mark.asyncio
    async def test_db_id_keys_normalized_to_uuid_and_filtered(self, super_tree_index):
        st, db, client = super_tree_index
        client._uuid_to_db = {"uuid-a": 1, "uuid-b": 2, "uuid-c": 3}
        client.documents = {u: {"id": u} for u in ("uuid-a", "uuid-b", "uuid-c")}
        for i in (1, 2, 3):
            db.insert_document(f"doc{i}.pdf", f"/tmp/{i}.pdf")
        with patch.object(super_tree_mod, "llm_acompletion") as mock_llm:
            # LLM 回 db_id 键（JSON 序列化为字符串键）的理由；doc 3 未选中 → 应被过滤
            mock_llm.return_value = json.dumps({
                "doc_ids": ["uuid-a", "uuid-b"],
                "reasons": {"1": "理由A", "2": "理由B", "3": "未被选中"},
            })
            selected, reasons = await st._holistic_select("test", [1, 2, 3])
        assert selected == [1, 2]
        assert reasons == {"uuid-a": "理由A", "uuid-b": "理由B"}

    @pytest.mark.asyncio
    async def test_uuid_keys_passthrough_and_unselected_filtered(self, super_tree_index):
        st, db, client = super_tree_index
        client._uuid_to_db = {"uuid-a": 1, "uuid-b": 2}
        client.documents = {u: {"id": u} for u in ("uuid-a", "uuid-b")}
        for i in (1, 2):
            db.insert_document(f"doc{i}.pdf", f"/tmp/{i}.pdf")
        with patch.object(super_tree_mod, "llm_acompletion") as mock_llm:
            mock_llm.return_value = json.dumps({
                "doc_ids": ["uuid-a"],
                "reasons": {"uuid-a": "理由A", "uuid-b": "未选"},
            })
            selected, reasons = await st._holistic_select("test", [1, 2])
        assert selected == [1]
        assert reasons == {"uuid-a": "理由A"}  # uuid 键原样；未选的 uuid-b 被过滤

    @pytest.mark.asyncio
    async def test_map_reduce_mid_tier_reasons_aggregated(self, super_tree_index):
        """[S6]#7：11+ 候选 map-reduce 中间档位（winners ≤ keep、无 reduce）
        各组 reasons 必须聚合下传，绝不因 no-reduce 路径丢失 L1→L2 trace。"""
        st, db, client = super_tree_index
        st._REASON_GROUP_SIZE = 4  # 11 候选 → 3 组 [4,4,3]，每组 3 winners ≤ 10 → no-reduce
        docs = {f"uuid-{i}": i for i in range(1, 12)}
        client.documents = {u: {"id": u} for u in docs}
        client._uuid_to_db = dict(docs)
        for i in range(1, 12):
            db.insert_document(f"doc{i}.pdf", f"/tmp/{i}.pdf")

        # mock LLM 按 prompt 子串识别组别，各自返回 doc_ids + reasons（并发下
        # 与组序解耦，避免 side_effect 列表依赖 gather 调度顺序）。识别键取各组
        # 末位候选的 doc_id uuid：既不出现在 prompt 结尾示例 JSON（占位 uuid-1/
        # uuid-2），也不出现在 kb_identity（只列文档名，不含 uuid）。
        def _group_response(prompt):
            if "uuid-4" in prompt:
                return json.dumps({"doc_ids": ["uuid-1", "uuid-2"],
                                   "reasons": {"uuid-1": "理由1", "uuid-2": "理由2"}})
            if "uuid-8" in prompt:
                return json.dumps({"doc_ids": ["uuid-5"],
                                   "reasons": {"uuid-5": "理由5"}})
            if "uuid-11" in prompt:
                return json.dumps({"doc_ids": ["uuid-9", "uuid-10"],
                                   "reasons": {"uuid-9": "理由9", "uuid-10": "理由10"}})
            return json.dumps({"doc_ids": []})

        async def fake_llm(model, prompt, **kw):
            return _group_response(prompt)

        # sys.modules stub 陷阱：test_router.py 导入期会再次 clobber
        # sys.modules["pageindex_mutil.super_tree"] 为它自己的 stub 模块，故不能按
        # sys.modules[Cls.__module__] 反查（会拿到错误模块对象打空）。须直接打在本
        # 文件持有的 super_tree_mod（类方法 globals 真正所属的模块）上。
        with patch.object(super_tree_mod, "llm_acompletion", side_effect=fake_llm):
            result, reasons = await st.select_documents(
                "test", {i: 1.0 for i in range(1, 12)})

        assert set(result) == {"uuid-1", "uuid-2", "uuid-5", "uuid-9", "uuid-10"}
        # 中间档位 no-reduce：各组 reasons 聚合，键已规范化 uuid，只含 winners 条目
        assert reasons == {
            "uuid-1": "理由1", "uuid-2": "理由2", "uuid-5": "理由5",
            "uuid-9": "理由9", "uuid-10": "理由10",
        }


