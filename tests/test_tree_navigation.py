"""P2 语义树导航 + 量级自适应测试 —— 设计文档 [S5]/[S6]。

覆盖：
1. 量级档位探测（小直连 / 中单层路由 / 海量层级树，按文档数，阈值可配）；
2. 宽层语义预筛（jieba 关键词 + closet_tags 倒排，严格无向量，保召回）；
3. 图谱信号加权（L0 实体距离加权：查询实体 → 距离衰减×关系类型权重表 →
   节点子树查表打分，作为四通道之一（D 通道）；证据加权，不再强制捞回被切节点）；
4. LLM 逐层精挑（可变数量、宁缺毋滥，T9 H02 修复的泛化；retrieve_model 接线）；
5. 层级树导航（渐进披露：只展开选中分支；分支并行展开 asyncio.gather；
   软归属 doc_id 去重保留最大权重；语料树缺失优雅降级返回 None）；
6. select_documents 量级自适应入口（小=扁平直连、中=簇路由加权、海量=树导航，
   树导航落空回退扁平候选）。

全部 LLM 调用均 mock —— 无真实 LLM、无向量（FULLY VECTORLESS）。
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from db import PageIndexDB

pkg_path = Path(__file__).parent.parent / "pageindex_mutil"
sys.path.insert(0, str(pkg_path))

import importlib.util

# Load super_tree without triggering the heavy package __init__ (PyPDF2 etc.).
# IMPORTANT (test isolation): only seed a stub ``pageindex_mutil.utils`` when the
# real module is NOT already imported — same guard as test_corpus_tree.py, so a
# full-suite run never clobbers modules later test files depend on.
if "pageindex_mutil.utils" not in sys.modules:
    utils_spec = importlib.util.spec_from_file_location("pageindex_mutil.utils", pkg_path / "utils.py")
    utils_mod = importlib.util.module_from_spec(utils_spec)
    sys.modules["pageindex_mutil.utils"] = utils_mod
    utils_mod.llm_completion = lambda *a, **k: None

    async def _mock_llm_acompletion(*a, **k):
        return None

    utils_mod.llm_acompletion = _mock_llm_acompletion
    utils_mod.count_tokens = lambda text, model=None: len(text or "") // 4

    def _mock_extract_json(text):
        try:
            return json.loads(text)
        except Exception:
            return None

    utils_mod.extract_json = _mock_extract_json
    _real_utils_spec = importlib.util.spec_from_file_location(
        "_real_utils_strip_nav", pkg_path / "utils.py"
    )
    _real_utils_mod = importlib.util.module_from_spec(_real_utils_spec)
    _real_utils_spec.loader.exec_module(_real_utils_mod)
    utils_mod.strip_markdown_fence = _real_utils_mod.strip_markdown_fence

if "pageindex_mutil.closet_index" not in sys.modules:
    closet_spec = importlib.util.spec_from_file_location(
        "pageindex_mutil.closet_index", pkg_path / "closet_index.py"
    )
    closet_mod = importlib.util.module_from_spec(closet_spec)
    sys.modules["pageindex_mutil.closet_index"] = closet_mod
    closet_spec.loader.exec_module(closet_mod)

if "pageindex_mutil.super_tree" in sys.modules:
    # Full-suite run: reuse whatever module object is already registered (the
    # real module or an earlier spec-loaded copy) and patch attributes on it.
    super_tree_mod = sys.modules["pageindex_mutil.super_tree"]
else:
    spec = importlib.util.spec_from_file_location("pageindex_mutil.super_tree", pkg_path / "super_tree.py")
    super_tree_mod = importlib.util.module_from_spec(spec)
    sys.modules["pageindex_mutil.super_tree"] = super_tree_mod
    spec.loader.exec_module(super_tree_mod)

SuperTreeIndex = super_tree_mod.SuperTreeIndex

# Full-suite isolation: another test file (e.g. test_router.py) may have loaded
# pageindex_mutil.utils with a stub extract_json that always returns None.
# Patch the attribute on the already-loaded super_tree module so _select_nodes
# / _holistic_select / _score_candidates can parse LLM JSON responses correctly.
_orig_extract_json = getattr(super_tree_mod, "extract_json", None)
if _orig_extract_json is None or not callable(_orig_extract_json):
    pass  # will be set below
else:
    # Probe: if extract_json('{"a":1}') returns None, it's the broken stub.
    try:
        _probe = _orig_extract_json('{"a":1}')
    except Exception:
        _probe = None
    if _probe is None:
        def _fixed_extract_json(text):
            try:
                return json.loads(text)
            except Exception:
                return None
        super_tree_mod.extract_json = _fixed_extract_json

# T6.4/P2.7：逐层精挑统一走 enhance_and_select（_select_nodes 老路已移除）。
# 隔离策略与 super_tree 相同：standalone 收集时 spec 加载 enhance 预置到
# sys.modules；全量套件收集时复用已加载的模块。关键：其他测试文件会在收集期
# purge/重导入 pageindex_mutil.*，运行期生效的 enhance 模块对象可能不是收集期
# 那一个——而 super_tree.navigate_tree 是调用时经 sys.modules 惰性解析
# `from .agentic.enhance import ...`，因此所有 patch 必须经 _enhance_module()
# 惰性访问器取"当前生效"的模块对象（与 test_search_single_enhanced 同理）。
if "pageindex_mutil.agentic.enhance" not in sys.modules:
    enh_spec = importlib.util.spec_from_file_location(
        "pageindex_mutil.agentic.enhance", pkg_path / "agentic" / "enhance.py"
    )
    _enhance_mod_seeded = importlib.util.module_from_spec(enh_spec)
    sys.modules["pageindex_mutil.agentic.enhance"] = _enhance_mod_seeded
    enh_spec.loader.exec_module(_enhance_mod_seeded)


def _enhance_module():
    """当前生效的 enhance 模块（运行期 sys.modules 解析，与被测惰性导入同源）。

    顺带修复坏 extract_json stub：enhance 的 llm_completion/extract_json 绑定自
    其加载时刻的 utils 模块，可能是永远返回 None 的 stub。
    """
    m = sys.modules.get("pageindex_mutil.agentic.enhance")
    if m is None:
        import pageindex_mutil.agentic.enhance as _m
        m = _m
    ej = getattr(m, "extract_json", None)
    if ej is not None and callable(ej):
        try:
            probe = ej('{"a":1}')
        except Exception:
            probe = None
        if probe is None:
            def _fixed_enh_extract_json(text):
                try:
                    return json.loads(text)
                except Exception:
                    return None
            m.extract_json = _fixed_enh_extract_json
    return m


# Prompt marker for enhance_and_select 精挑（逐层分支选择）。
M_ENH = "检索增强专家"
# Prompt marker for T9 holistic document selection.
M_DOC = "文档检索专家"


@pytest.fixture
def nav_index():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = PageIndexDB(path)
    client = MagicMock()
    client._uuid_to_db = {}
    client._id_mapper = None
    client.documents = {}
    client.closet_index = None
    client.search_backend = MagicMock()
    st = SuperTreeIndex(db, model="qwen-plus", client=client)
    yield st, db, client
    db.close()
    os.unlink(path)


def _candidates_in_enhance_prompt(prompt):
    """Extract {title: node_id(str)} from an enhance_and_select 精挑 prompt.

    证据块格式（[3.2.2]）："候选节点 {nid}：\n标题：{title}\n..."——逐块解析，
    全局注记（"注：…"）不含"候选节点 "前缀，天然不干扰。
    """
    cands = {}
    for chunk in prompt.split("候选节点 ")[1:]:
        nid = chunk.split("：", 1)[0].strip()
        title = ""
        for line in chunk.splitlines():
            if line.startswith("标题："):
                title = line[len("标题："):]
                break
        cands[title] = nid
    return cands


def _enh_response(selected, pool_concern=False, concern_reason=""):
    return json.dumps({
        "selected_ids": selected,
        "pool_concern": pool_concern,
        "concern_reason": concern_reason,
    })


def _two_cluster_tree(db):
    """root → A(风控, docs下挂 A1/A2), B(审计, docs下挂 B1/B2)。返回节点 id 字典。

    先确保文档 1/2/3 存在（membership 的 doc_id 有外键约束）。
    B2 无文档归属，仅作为并行展开测试的第二子分支。
    """
    for i in (1, 2, 3):
        if db.get_document_by_id(i) is None:
            db.insert_document(f"doc{i}.pdf", f"/tmp/{i}.pdf")
    root = db.insert_corpus_tree_node(None, "知识库", "语料库根节点", 0, kind="root")
    a = db.insert_corpus_tree_node(root, "风控管理", "企业风险控制", 1, kind="cluster", tag="风控")
    b = db.insert_corpus_tree_node(root, "财务审计", "财务与审计", 1, kind="cluster", tag="审计")
    a1 = db.insert_corpus_tree_node(a, "风控制度", "风险控制制度文档", 2, kind="cluster", tag="风控")
    a2 = db.insert_corpus_tree_node(a, "风控案例", "风险案例汇编", 2, kind="cluster", tag="风控")
    b1 = db.insert_corpus_tree_node(b, "审计报告", "年度审计报告", 2, kind="cluster", tag="审计")
    b2 = db.insert_corpus_tree_node(b, "审计方法", "审计方法论", 2, kind="cluster", tag="审计")
    db.add_corpus_membership(1, a1, 0.8)
    db.add_corpus_membership(2, a2, 0.7)
    db.add_corpus_membership(3, b1, 0.9)
    return {"root": root, "a": a, "b": b, "a1": a1, "a2": a2, "b1": b1, "b2": b2}


# ---------------------------------------------------------------------------
# [S5] 量级自适应：档位探测
# ---------------------------------------------------------------------------


class TestScaleTier:
    def test_tier_by_doc_count(self, nav_index):
        st, db, _ = nav_index
        st._SMALL_MAX_DOCS = 3
        st._MASSIVE_MIN_DOCS = 5
        assert st.detect_scale_tier() == "small"
        db.insert_document("1.pdf", "/tmp/1.pdf")
        db.insert_document("2.pdf", "/tmp/2.pdf")
        assert st.detect_scale_tier() == "small"  # 2 < 3
        db.insert_document("3.pdf", "/tmp/3.pdf")
        assert st.detect_scale_tier() == "medium"  # 3 >= 3 且 < 5
        db.insert_document("4.pdf", "/tmp/4.pdf")
        db.insert_document("5.pdf", "/tmp/5.pdf")
        assert st.detect_scale_tier() == "massive"  # 5 >= 5

    def test_default_thresholds_sane(self, nav_index):
        st, _, _ = nav_index
        assert 0 < st._SMALL_MAX_DOCS < st._MASSIVE_MIN_DOCS

    def test_thresholds_overridable(self, nav_index):
        st, db, _ = nav_index
        db.insert_document("1.pdf", "/tmp/1.pdf")
        st._SMALL_MAX_DOCS = 1
        assert st.detect_scale_tier() != "small"


# ---------------------------------------------------------------------------
# [S6]① 语义预筛（宽层，严格无向量，保召回）
# ---------------------------------------------------------------------------


class TestNodePrefilter:
    def test_tag_match_ranks_first(self, nav_index):
        """closet_tags 倒排（节点 tag）命中 = 无向量语义匹配，信号强于词面。"""
        st, _, _ = nav_index
        nodes = [{"id": i, "title": f"节点{i}", "summary": "", "tag": None}
                 for i in range(1, 6)]
        nodes[2]["tag"] = "风控"
        out = st._prefilter_nodes("风控合规", nodes)
        assert out[0]["id"] == 3

    def test_wide_layer_no_hard_drop_signal_nodes_ranked_first(self, nav_index):
        """P2.9：宽层废除 top-k 硬截断——信号节点全部保留并排前，零丢弃；
        超限收窄由 enhance_and_select 的 union cap + 延迟池承担（[3.2.1]）。"""
        st, _, _ = nav_index
        nodes = [{"id": i, "title": f"节点{i}", "summary": "", "tag": None}
                 for i in range(1, 21)]
        signal_ids = {3, 7, 11, 15, 19, 20}
        for i in signal_ids:
            nodes[i - 1]["tag"] = "风控"
        out = st._prefilter_nodes("风控", nodes)
        assert len(out) == 20  # 无丢弃：全部节点保留
        assert {n["id"] for n in out[: len(signal_ids)]} == signal_ids  # 信号节点排前

    def test_no_signal_keeps_all(self, nav_index):
        """无任何匹配信号时不硬过滤，全量交给 LLM 精挑（保召回）。"""
        st, _, _ = nav_index
        nodes = [{"id": i, "title": f"节点{i}", "summary": "", "tag": None}
                 for i in range(1, 21)]
        out = st._prefilter_nodes("完全不相关的查询zzz", nodes)
        assert len(out) == 20

    def test_prefilter_is_vectorless(self, nav_index):
        """预筛绝不触碰向量后端（通道 C 不在树导航主路径，[S10]）。"""
        st, _, client = nav_index
        nodes = [{"id": 1, "title": "风控", "summary": "", "tag": "风控"}]
        st._prefilter_nodes("风控", nodes)
        client.search_backend.search.assert_not_called()


# ---------------------------------------------------------------------------
# [S6]② 图谱信号加权（L0 实体距离加权：D 通道）
# ---------------------------------------------------------------------------


class TestEntityBoost:
    """实体图谱信号 = 四通道打分中的 D 通道（现行 L0 实现）。

    四通道重构废弃了旧 API（`_entity_document_ids` 实体→文档集反查、
    `_entity_boost_nodes` 带切节点强制捞回）。现行链路：
    `_precompute_entity_distances` 以 search_entities 命中实体为种子（距离 0，
    权重 1.0）做一次 BFS（距离衰减 × 关系类型权重）生成 entity_table；
    `_get_node_entity_boost` 对节点子树文档的提及逐层查表取最大权重；
    `_score_nodes`/`_prefilter_nodes` 将其作为加权通道（最高贡献 0.2）。
    实体信号不再能强制捞回被截断的节点——硬捞回语义留待 P2 的
    并集 + 延迟池重构（spec [3.3]：实体信号重定义为证据）。
    """

    def test_entity_boost_weights_node_with_mention(self, nav_index):
        """查询实体以距离 0/权重 1.0 入 entity_table；仅含提及的节点获加权。"""
        st, db, _ = nav_index
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        d2 = db.insert_document("b.pdf", "/tmp/b.pdf")
        eid = db.insert_entity("person", "张三")
        db.insert_entity_mention(eid, d2, confidence=0.9)
        root = db.insert_corpus_tree_node(None, "知识库", "", 0, kind="root")
        n1 = db.insert_corpus_tree_node(root, "甲簇", "", 1, kind="cluster")
        n2 = db.insert_corpus_tree_node(root, "乙簇", "", 1, kind="cluster")
        db.add_corpus_membership(d1, n1, 1.0)
        db.add_corpus_membership(d2, n2, 1.0)
        view = st._load_corpus_tree()

        table = st._precompute_entity_distances("张三")
        assert table[eid]["distance"] == 0
        assert table[eid]["weight"] == 1.0
        assert table[eid]["name"] == "张三"

        boost2, info2 = st._get_node_entity_boost(n2, view, table)
        assert boost2 > 0  # 含查询实体提及的节点被加权
        assert info2 and info2["name"] == "张三"
        boost1, info1 = st._get_node_entity_boost(n1, view, table)
        assert boost1 == 0  # 无实体提及的兄弟节点不加权
        assert info1 is None
        # 查询未命中任何实体 → 空表（对应旧 test_entity_document_ids 的空查断言）
        assert st._precompute_entity_distances("无实体查询") == {}

    def test_entity_hit_node_ranked_first(self, nav_index):
        """D 通道加权使含实体节点在打分与预筛排序中升到零信号兄弟之前。"""
        st, db, _ = nav_index
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        d2 = db.insert_document("b.pdf", "/tmp/b.pdf")
        eid = db.insert_entity("person", "张三")
        db.insert_entity_mention(eid, d2, confidence=0.9)
        root = db.insert_corpus_tree_node(None, "知识库", "", 0, kind="root")
        n1 = db.insert_corpus_tree_node(root, "甲簇", "", 1, kind="cluster")
        n2 = db.insert_corpus_tree_node(root, "乙簇", "", 1, kind="cluster")
        db.add_corpus_membership(d1, n1, 1.0)
        db.add_corpus_membership(d2, n2, 1.0)
        view = st._load_corpus_tree()

        entity_table = st._precompute_entity_distances("张三")
        nodes = [view.nodes_by_id[n1], view.nodes_by_id[n2]]
        scores = st._score_nodes("张三", nodes, view, entity_table)
        assert scores[n2]["entity_boost"] > scores[n1]["entity_boost"]
        assert scores[n2]["total_score"] > scores[n1]["total_score"]
        out = st._prefilter_nodes("张三", nodes, view, entity_table)
        assert out[0]["id"] == n2  # 含查询实体的节点被加权提前

    def test_entity_hit_survives_without_hard_cut(self, nav_index):
        """P2.9 语义落地：预筛不再截断（旧 topk 硬切废除）——含实体证据的
        节点排序严格优先且零丢弃。超限候选的唯一溢出目标是 enhance_and_select
        的延迟池（deferred，可经 pool_concern 回捞），任何一层不再出现
        "分数硬截后直接丢弃"（[3.2.1] 收窄纪律；[3.3] 实体信号=证据）。
        """
        st, db, _ = nav_index
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        d2 = db.insert_document("b.pdf", "/tmp/b.pdf")
        eid = db.insert_entity("person", "张三")
        db.insert_entity_mention(eid, d2, confidence=0.9)
        root = db.insert_corpus_tree_node(None, "知识库", "", 0, kind="root")
        n_zero_a = db.insert_corpus_tree_node(root, "甲簇", "", 1, kind="cluster")
        n_zero_b = db.insert_corpus_tree_node(root, "乙簇", "", 1, kind="cluster")
        n_entity = db.insert_corpus_tree_node(root, "丙簇", "", 1, kind="cluster")
        db.add_corpus_membership(d1, n_zero_a, 1.0)  # d1 无实体提及 → 零信号
        db.add_corpus_membership(d2, n_entity, 1.0)  # d2 含张三 → 唯一证据
        view = st._load_corpus_tree()

        entity_table = st._precompute_entity_distances("张三")
        nodes = [view.nodes_by_id[n_zero_a], view.nodes_by_id[n_zero_b],
                 view.nodes_by_id[n_entity]]
        out = st._prefilter_nodes("张三", nodes, view, entity_table)
        assert len(out) == 3  # 零丢弃：无硬截断
        assert out[0]["id"] == n_entity  # 证据胜过无证据，实体节点排最前


# ---------------------------------------------------------------------------
# [S6]③ LLM 逐层精挑（可变数量、宁缺毋滥、retrieve_model 接线）
# ---------------------------------------------------------------------------


class TestLayerSelectionViaEnhance:
    """[3.2.1]/P2.7 层精挑统一走 enhance_and_select（_select_nodes 老路已移除）。"""

    @pytest.mark.asyncio
    async def test_variable_count(self, nav_index):
        """宁缺毋滥：LLM 只挑 1 个分支时不凑数。"""
        st, db, _ = nav_index
        _two_cluster_tree(db)
        prompts = []

        def fake(model, prompt, **kw):
            if M_ENH not in prompt:
                return ""
            prompts.append(prompt)
            titles = _candidates_in_enhance_prompt(prompt)
            if "风控管理" in titles:
                return _enh_response([titles["风控管理"]])  # 只挑 1 个
            if "风控制度" in titles:
                return _enh_response([titles["风控制度"]])
            return _enh_response([])

        with patch.object(_enhance_module(), "llm_completion", side_effect=fake):
            docs = await st.navigate_tree("风控制度")
        assert set(docs) == {1}  # 仅风控制度分支下的文档
        assert any("财务审计" in _candidates_in_enhance_prompt(p) for p in prompts[:1])

    @pytest.mark.asyncio
    async def test_select_none(self, nav_index):
        """L1 全拒 → 导航空结果（调用方回退扁平候选）。"""
        st, db, _ = nav_index
        _two_cluster_tree(db)
        with patch.object(_enhance_module(), "llm_completion",
                          return_value=_enh_response([])):
            docs = await st.navigate_tree("风控")
        assert docs == {}

    @pytest.mark.asyncio
    async def test_invalid_ids_filtered(self, nav_index):
        """LLM 返回未知/非法 id 被过滤，合法 id 照常展开。"""
        st, db, _ = nav_index
        _two_cluster_tree(db)

        def fake(model, prompt, **kw):
            if M_ENH not in prompt:
                return ""
            titles = _candidates_in_enhance_prompt(prompt)
            if "风控管理" in titles:
                return _enh_response([999, titles["风控管理"], "x", titles["风控管理"]])
            return _enh_response(list(titles.values()))

        with patch.object(_enhance_module(), "llm_completion", side_effect=fake):
            docs = await st.navigate_tree("风控")
        assert set(docs) == {1, 2}  # A 分支照常展开（非法 id 忽略，重复去重）

    @pytest.mark.asyncio
    async def test_llm_failure_degrades_union_passes_through(self, nav_index):
        """[7.7] LLM 失效不做启发式裁剪：union 候选放行，导航仍有产出。"""
        st, db, _ = nav_index
        _two_cluster_tree(db)
        with patch.object(_enhance_module(), "llm_completion", return_value=""):
            docs = await st.navigate_tree("风控")
        # 零信号 → 全量 union → LLM 空回复降级放行 → 两簇及子簇展开
        assert set(docs) == {1, 2, 3}

    @pytest.mark.asyncio
    async def test_layer_selection_uses_retrieve_model(self, nav_index):
        """NFR4：层精挑 LLM（enhance 内）用 retrieve_model（model 兜底）。"""
        st, db, client = nav_index
        _two_cluster_tree(db)
        st2 = SuperTreeIndex(db, model="m", client=client, retrieve_model="r-model")
        models = []

        def fake(model, prompt, **kw):
            if M_ENH in prompt:
                models.append(model)
            return _enh_response([])

        with patch.object(_enhance_module(), "llm_completion", side_effect=fake):
            await st2.navigate_tree("风控")
        assert models and all(m == "r-model" for m in models)

    @pytest.mark.asyncio
    async def test_layer_selection_falls_back_to_model(self, nav_index):
        """NFR4：retrieve_model=None → 用 model。"""
        st, db, client = nav_index
        _two_cluster_tree(db)
        st2 = SuperTreeIndex(db, model="m", client=client, retrieve_model=None)
        models = []

        def fake(model, prompt, **kw):
            if M_ENH in prompt:
                models.append(model)
            return _enh_response([])

        with patch.object(_enhance_module(), "llm_completion", side_effect=fake):
            await st2.navigate_tree("风控")
        assert models and all(m == "m" for m in models)


# ---------------------------------------------------------------------------
# [S6] 层级树导航：渐进披露 / 并行展开 / 软归属去重 / 优雅降级 / 无向量
# ---------------------------------------------------------------------------


class TestNavigateTree:
    @pytest.mark.asyncio
    async def test_empty_tree_returns_none(self, nav_index):
        """语料树未建 → 返回 None，调用方降级扁平管线（行为不变）。"""
        st, _, _ = nav_index
        assert await st.navigate_tree("任意查询") is None

    @pytest.mark.asyncio
    async def test_progressive_disclosure(self, nav_index):
        """只展开选中分支：B 子树从不进入任何精挑 prompt。"""
        st, db, _ = nav_index
        _two_cluster_tree(db)
        prompts = []

        def fake(model, prompt, *a, **k):
            if M_ENH not in prompt:
                return ""
            prompts.append(prompt)
            titles = _candidates_in_enhance_prompt(prompt)
            if "风控管理" in titles and "财务审计" in titles:
                return _enh_response([titles["风控管理"]])
            if "风控制度" in titles:
                return _enh_response([titles["风控制度"]])
            return _enh_response([])

        with patch.object(_enhance_module(), "llm_completion", side_effect=fake):
            docs = await st.navigate_tree("风控制度")
        assert set(docs) == {1}
        assert len(prompts) == 2  # 仅 level1 + A 的 level2
        assert not any("审计报告" in p for p in prompts)  # B 分支未展开

    @pytest.mark.asyncio
    async def test_parallel_branch_expansion(self, nav_index):
        """延迟纪律：一层选中多个分支后，下一层导航并行展开（asyncio.gather）。

        enhance 的 LLM 调用经 asyncio.to_thread（线程池），层间并行不变。
        """
        st, db, _ = nav_index
        _two_cluster_tree(db)
        active = {"n": 0, "max": 0}

        def fake(model, prompt, *a, **k):
            if M_ENH not in prompt:
                return ""
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
            time.sleep(0.02)
            active["n"] -= 1
            titles = _candidates_in_enhance_prompt(prompt)
            return _enh_response(list(titles.values()))  # 全选

        with patch.object(_enhance_module(), "llm_completion", side_effect=fake):
            docs = await st.navigate_tree("风控 审计")
        assert set(docs) == {1, 2, 3}
        assert active["max"] >= 2  # A/B 两支的下一层导航并行执行

    @pytest.mark.asyncio
    async def test_soft_membership_dedup_max_weight(self, nav_index):
        """软归属 DAG：同一文档经多簇命中 → 按 doc_id 去重，保留最大权重。"""
        st, db, _ = nav_index
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        d2 = db.insert_document("b.pdf", "/tmp/b.pdf")
        root = db.insert_corpus_tree_node(None, "知识库", "", 0, kind="root")
        ca = db.insert_corpus_tree_node(root, "风控管理", "", 1, kind="cluster", tag="风控")
        cb = db.insert_corpus_tree_node(root, "风险合规", "", 1, kind="cluster", tag="合规")
        db.add_corpus_membership(d1, ca, 0.4)
        db.add_corpus_membership(d1, cb, 0.9)
        db.add_corpus_membership(d2, ca, 0.6)

        def fake(model, prompt, *a, **k):
            if M_ENH in prompt:
                titles = _candidates_in_enhance_prompt(prompt)
                return _enh_response(list(titles.values()))  # 两簇都选
            return ""

        with patch.object(_enhance_module(), "llm_completion", side_effect=fake):
            docs = await st.navigate_tree("风控 合规")
        assert set(docs) == {d1, d2}
        assert docs[d1] == 0.9  # 去重后保留最大权重
        assert docs[d2] == 0.6

    @pytest.mark.asyncio
    async def test_never_calls_vector_backend(self, nav_index):
        """[S10] 树导航主路径严格无向量：全程不触碰 search_backend（通道 C）。"""
        st, db, client = nav_index
        _two_cluster_tree(db)

        def fake(model, prompt, *a, **k):
            if M_ENH in prompt:
                titles = _candidates_in_enhance_prompt(prompt)
                return _enh_response(list(titles.values()))
            return ""

        with patch.object(_enhance_module(), "llm_completion", side_effect=fake):
            docs = await st.navigate_tree("风控")
        assert docs
        client.search_backend.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_narrow_layer_all_siblings_reach_llm(self, nav_index):
        """窄层（兄弟数 ≤ 阈值）跳过信号排序，全部直达 enhance 精挑。"""
        st, db, _ = nav_index
        st._NARROW_LAYER_MAX = 8
        root = db.insert_corpus_tree_node(None, "知识库", "", 0, kind="root")
        titles = [f"主题簇{i}" for i in range(1, 6)]
        for t in titles:
            db.insert_corpus_tree_node(root, t, "", 1, kind="cluster")
        seen = []

        def fake(model, prompt, *a, **k):
            if M_ENH in prompt:
                seen.append(prompt)
            return _enh_response([])

        with patch.object(_enhance_module(), "llm_completion", side_effect=fake):
            await st.navigate_tree("完全无关查询")
        assert len(seen) == 1
        got = _candidates_in_enhance_prompt(seen[0])
        assert set(got) == set(titles)  # 5 个兄弟全部直达精挑

    @pytest.mark.asyncio
    async def test_wide_layer_union_narrowing_via_profiles(self, nav_index):
        """P2.9：宽层收窄 = 高召回 union——签名命中节点经标签通道进 union，
        无签名兄弟不进 union（union 收窄，非 top-k 硬截）；命中节点必达精挑。"""
        st, db, _ = nav_index
        st._NARROW_LAYER_MAX = 3
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        root = db.insert_corpus_tree_node(None, "知识库", "", 0, kind="root")
        for i in range(1, 7):
            db.insert_corpus_tree_node(root, f"无关簇{i}", "", 1, kind="cluster")
        hit = db.insert_corpus_tree_node(root, "风控专区", "", 1, kind="cluster", tag="风控")
        db.add_corpus_membership(d1, hit, 1.0)
        db.insert_closet_tags(d1, [(d1, "风控", "风控", 0.9, "llm")])
        seen = []

        def fake(model, prompt, *a, **k):
            if M_ENH in prompt:
                seen.append(prompt)
            return _enh_response([])

        with patch.object(_enhance_module(), "llm_completion", side_effect=fake):
            await st.navigate_tree("风控")
        got = _candidates_in_enhance_prompt(seen[0])
        assert "风控专区" in got  # 签名命中节点必在 union
        assert len(got) == 1  # 无签名兄弟被 union 收窄（非硬截丢弃——零信号保护见专项）
        _ = hit

    @pytest.mark.asyncio
    async def test_zero_signal_wide_layer_passes_all_siblings(self, nav_index):
        """零信号保护：宽层全无签名时全量兄弟进精挑（[1.1] 保召回，无硬丢弃）。"""
        st, db, _ = nav_index
        st._NARROW_LAYER_MAX = 2
        root = db.insert_corpus_tree_node(None, "知识库", "", 0, kind="root")
        titles = [f"普通簇{i}" for i in range(1, 8)]
        for t in titles:
            db.insert_corpus_tree_node(root, t, "", 1, kind="cluster")
        seen = []

        def fake(model, prompt, *a, **k):
            if M_ENH in prompt:
                seen.append(prompt)
            return _enh_response([])

        with patch.object(_enhance_module(), "llm_completion", side_effect=fake):
            await st.navigate_tree("完全无关的查询zz")
        assert len(seen) == 1
        got = _candidates_in_enhance_prompt(seen[0])
        assert set(got) == set(titles)  # 7 个兄弟全部进精挑，无一硬丢


# ---------------------------------------------------------------------------
# [S5]+[S6] select_documents 量级自适应入口
# ---------------------------------------------------------------------------


class TestAdaptiveSelectDocuments:
    @pytest.mark.asyncio
    async def test_small_tier_flat_path_no_tree_navigation(self, nav_index):
        """小语料直连：跳过层级遍历，走扁平管线（无树导航 LLM 调用）。"""
        st, db, client = nav_index
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        d2 = db.insert_document("b.pdf", "/tmp/b.pdf")
        client._uuid_to_db = {"uuid-1": d1, "uuid-2": d2}
        prompts = []

        async def fake(model, prompt, *a, **k):
            prompts.append(prompt)
            if M_DOC in prompt:
                return json.dumps({"doc_ids": ["uuid-1"]})
            return ""

        with patch.object(super_tree_mod, "llm_acompletion", side_effect=fake), \
                patch.object(_enhance_module(), "llm_completion",
                             side_effect=lambda m, p, **k: _enh_response([])) as mock_enh:
            result = await st.select_documents("q", {d1: 1.0, d2: 1.0})
        assert result == ["uuid-1"]
        assert not any(M_ENH in p for p in prompts)
        mock_enh.assert_not_called()  # 小语料直连：不触发树导航精挑

    @pytest.mark.asyncio
    async def test_massive_tier_uses_tree_navigation(self, nav_index):
        """海量档：树导航产出候选 → 文档级推理挑选（宁缺毋滥，可变数量）。"""
        st, db, client = nav_index
        st._SMALL_MAX_DOCS = 1
        st._MASSIVE_MIN_DOCS = 3
        ids = _two_cluster_tree(db)
        client._uuid_to_db = {"uuid-1": 1, "uuid-2": 2, "uuid-3": 3}

        async def fake_acompletion(model, prompt, *a, **k):
            if M_DOC in prompt:
                return json.dumps({"doc_ids": ["uuid-2"]})  # 只挑 1 篇（宁缺毋滥）
            return ""

        def fake_completion(model, prompt, *a, **k):
            if M_ENH in prompt:
                titles = _candidates_in_enhance_prompt(prompt)
                if "风控管理" in titles and "财务审计" in titles:
                    return _enh_response([titles["风控管理"]])
                return _enh_response(list(titles.values()))
            return ""

        with patch.object(super_tree_mod, "llm_acompletion", side_effect=fake_acompletion), \
                patch.object(_enhance_module(), "llm_completion", side_effect=fake_completion):
            result = await st.select_documents("风控制度", {1: 1.0, 2: 1.0, 3: 1.0})
        assert result == ["uuid-2"]
        _ = ids

    @pytest.mark.asyncio
    async def test_massive_no_tree_falls_back_flat(self, nav_index):
        """海量档但语料树未建 → 优雅降级扁平管线（现有部署行为不变）。"""
        st, db, client = nav_index
        st._SMALL_MAX_DOCS = 1
        st._MASSIVE_MIN_DOCS = 2
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        d2 = db.insert_document("b.pdf", "/tmp/b.pdf")
        client._uuid_to_db = {"uuid-1": d1, "uuid-2": d2}
        prompts = []

        async def fake(model, prompt, *a, **k):
            prompts.append(prompt)
            if M_DOC in prompt:
                return json.dumps({"doc_ids": ["uuid-2"]})
            return ""

        with patch.object(super_tree_mod, "llm_acompletion", side_effect=fake):
            result = await st.select_documents("q", {d1: 1.0, d2: 1.0})
        assert result == ["uuid-2"]
        assert not any(M_ENH in p for p in prompts)  # 无树 → 无逐层精挑

    @pytest.mark.asyncio
    async def test_massive_nav_empty_falls_back_flat(self, nav_index):
        """树导航落空（精挑无人命中）→ 回退扁平候选，防级联误判。"""
        st, db, client = nav_index
        st._SMALL_MAX_DOCS = 1
        st._MASSIVE_MIN_DOCS = 2
        _two_cluster_tree(db)
        client._uuid_to_db = {"uuid-1": 1, "uuid-2": 2, "uuid-3": 3}

        async def fake_acompletion(model, prompt, *a, **k):
            if M_DOC in prompt:
                return json.dumps({"doc_ids": ["uuid-3"]})
            return ""

        with patch.object(super_tree_mod, "llm_acompletion", side_effect=fake_acompletion), \
                patch.object(_enhance_module(), "llm_completion",
                             side_effect=lambda m, p, **k: _enh_response([])):
            result = await st.select_documents("q", {1: 1.0, 2: 1.0, 3: 1.0})
        assert result == ["uuid-3"]

    def test_medium_cluster_route_boost(self, nav_index):
        """中档单层领域路由：命中领域的候选加性提升；软路由绝不删候选。"""
        st, db, _ = nav_index
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        d2 = db.insert_document("b.pdf", "/tmp/b.pdf")
        d3 = db.insert_document("c.pdf", "/tmp/c.pdf")
        root = db.insert_corpus_tree_node(None, "知识库", "", 0, kind="root")
        ca = db.insert_corpus_tree_node(root, "风控管理", "风险控制", 1, kind="cluster", tag="风控")
        cb = db.insert_corpus_tree_node(root, "财务审计", "审计", 1, kind="cluster", tag="审计")
        db.add_corpus_membership(d1, ca, 1.0)
        db.add_corpus_membership(d2, ca, 1.0)
        db.add_corpus_membership(d3, cb, 1.0)
        boosted = st._cluster_route_boost("风控合规", {d1: 1.0, d2: 1.0, d3: 1.0})
        assert boosted[d1] > 1.0
        assert boosted[d2] > 1.0
        assert boosted[d3] == 1.0  # 未命中领域保持原分
        assert set(boosted) == {d1, d2, d3}  # 无硬删

    def test_medium_cluster_route_boost_no_tree(self, nav_index):
        """语料树缺失 → 原样返回候选（优雅降级）。"""
        st, db, _ = nav_index
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        out = st._cluster_route_boost("q", {d1: 1.0})
        assert out == {d1: 1.0}

    @pytest.mark.asyncio
    async def test_medium_tier_routes_through_cluster_boost(self, nav_index):
        """中档端到端：select_documents 走单层领域路由再加推理挑选。"""
        st, db, client = nav_index
        st._SMALL_MAX_DOCS = 1
        st._MASSIVE_MIN_DOCS = 100
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        d2 = db.insert_document("b.pdf", "/tmp/b.pdf")
        client._uuid_to_db = {"uuid-1": d1, "uuid-2": d2}
        root = db.insert_corpus_tree_node(None, "知识库", "", 0, kind="root")
        ca = db.insert_corpus_tree_node(root, "风控管理", "", 1, kind="cluster", tag="风控")
        db.add_corpus_membership(d1, ca, 1.0)
        db.add_corpus_membership(d2, ca, 1.0)

        async def fake(model, prompt, *a, **k):
            if M_DOC in prompt:
                return json.dumps({"doc_ids": ["uuid-1"]})
            return ""

        with patch.object(super_tree_mod, "llm_acompletion", side_effect=fake), \
             patch.object(st, "_cluster_route_boost",
                          side_effect=st._cluster_route_boost) as spy:
            result = await st.select_documents("风控", {d1: 1.0, d2: 1.0})
        assert result == ["uuid-1"]
        spy.assert_called_once()


# ---------------------------------------------------------------------------
# T6.4 簇聚合签名（unit=层分支节点；子树 top 实体/关键词/标签 + run 缓存）
# ---------------------------------------------------------------------------


class TestClusterProfileAggregation:
    """查询时簇聚合签名：真实 DB 夹具验证实体/关键词/标签聚合与上限。"""

    def test_aggregates_entities_keywords_tags_from_subtree(self, nav_index):
        st, db, _ = nav_index
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        d2 = db.insert_document("b.pdf", "/tmp/b.pdf")
        root = db.insert_corpus_tree_node(None, "知识库", "", 0, kind="root")
        cluster = db.insert_corpus_tree_node(root, "风控管理", "", 1, kind="cluster")
        db.add_corpus_membership(d1, cluster, 1.0)
        db.add_corpus_membership(d2, cluster, 0.5)

        eid_zs = db.insert_entity("person", "张三")
        eid_ls = db.insert_entity("person", "李四")
        db.insert_entity_mention(eid_zs, d1, confidence=0.9)
        db.insert_entity_mention(eid_zs, d2, confidence=0.8)
        db.insert_entity_mention(eid_ls, d1, confidence=0.7)
        db.upsert_node_profiles(d1, [
            {"node_id": "n1", "entities": [], "keywords": ["风控", "合规"], "tags": []},
        ])
        db.upsert_node_profiles(d2, [
            {"node_id": "n2", "entities": [], "keywords": ["风控", "审计"], "tags": []},
        ])
        db.insert_closet_tags(d1, [
            (d1, "风险管理", "风控", 0.9, "llm"),
            (d1, "原词兜底", "兜底", 0.5, "fallback"),  # fallback 源不进语义漏斗
        ])
        db.insert_closet_tags(d2, [(d2, "风险管理", "风控", 0.8, "llm")])

        view = st._load_corpus_tree()
        profile = st._aggregate_cluster_profile(cluster, view, {})

        # 实体：跨文档频次排序（张三 2 篇 > 李四 1 篇），canonical 名来自 entities 表
        assert [e["name"] for e in profile["entities"]] == ["张三", "李四"]
        # 关键词：风控 2 次排前
        assert profile["keywords"][0] == "风控"
        assert set(profile["keywords"]) == {"风控", "合规", "审计"}
        # 标签：只认 source='llm'（closet_tags 源门控，[7.2]）
        assert profile["tags"] == ["风险管理"]

    def test_entity_table_distance_annotation(self, nav_index):
        """[3.3] L0 实体距离作证据：命中 entity_table 的实体注释距离/关系类型。"""
        st, db, _ = nav_index
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        root = db.insert_corpus_tree_node(None, "知识库", "", 0, kind="root")
        cluster = db.insert_corpus_tree_node(root, "簇", "", 1, kind="cluster")
        db.add_corpus_membership(d1, cluster, 1.0)
        eid = db.insert_entity("person", "张三")
        db.insert_entity_mention(eid, d1, confidence=0.9)
        view = st._load_corpus_tree()
        table = {eid: {"distance": 1, "relation_type": "part_of",
                       "weight": 0.56, "name": "张三"}}
        profile = st._aggregate_cluster_profile(cluster, view, table)
        assert profile["entities"][0]["name"] == "张三"
        assert "图谱距离1" in profile["entities"][0]["type"]
        assert "part_of" in profile["entities"][0]["type"]

    def test_caps_entities_keywords_tags(self, nav_index):
        """聚合上限：实体 ≤5 / 关键词 ≤8 / 标签 ≤3。"""
        st, db, _ = nav_index
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        root = db.insert_corpus_tree_node(None, "知识库", "", 0, kind="root")
        cluster = db.insert_corpus_tree_node(root, "簇", "", 1, kind="cluster")
        db.add_corpus_membership(d1, cluster, 1.0)
        db.upsert_node_profiles(d1, [
            {"node_id": "n1", "entities": [],
             "keywords": [f"关键词{i}" for i in range(10)], "tags": []},
        ])
        for i in range(6):
            db.insert_entity_mention(db.insert_entity("c", f"实体{i}"), d1, confidence=0.9)
        db.insert_closet_tags(
            d1, [(d1, f"标签{i}", f"标签{i}", 0.9, "llm") for i in range(5)],
        )
        view = st._load_corpus_tree()
        profile = st._aggregate_cluster_profile(cluster, view, {})
        assert len(profile["entities"]) == 5
        assert len(profile["keywords"]) == 8
        assert len(profile["tags"]) == 3

    def test_doc_sampling_cap(self, nav_index):
        """子树成本守卫：只采样排序前 _CLUSTER_PROFILE_MAX_DOCS 篇成员文档。"""
        st, db, _ = nav_index
        st._CLUSTER_PROFILE_MAX_DOCS = 2
        root = db.insert_corpus_tree_node(None, "知识库", "", 0, kind="root")
        cluster = db.insert_corpus_tree_node(root, "簇", "", 1, kind="cluster")
        for i in range(3):
            d = db.insert_document(f"d{i}.pdf", f"/tmp/{i}.pdf")
            db.add_corpus_membership(d, cluster, 1.0)
            db.upsert_node_profiles(d, [
                {"node_id": "n1", "entities": [], "keywords": [f"kw{i}"], "tags": []},
            ])
        view = st._load_corpus_tree()
        profile = st._aggregate_cluster_profile(cluster, view, {})
        # 成员文档按 id 排序取前 2 篇 → 第 3 篇的 kw2 不被采样
        assert "kw2" not in profile["keywords"]
        assert len(profile["keywords"]) == 2

    @pytest.mark.asyncio
    async def test_profile_cache_avoids_recomputation(self, nav_index):
        """run-scoped 缓存：同一簇二次 _navigate_level 不重复聚合查询。"""
        st, db, _ = nav_index
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        root = db.insert_corpus_tree_node(None, "知识库", "", 0, kind="root")
        c1 = db.insert_corpus_tree_node(root, "簇甲", "", 1, kind="cluster")
        db.add_corpus_membership(d1, c1, 1.0)

        calls = {"n": 0}
        real_mentions = db.get_entity_mentions_by_doc

        def spy(doc_id):
            calls["n"] += 1
            return real_mentions(doc_id)

        db.get_entity_mentions_by_doc = spy
        view = st._load_corpus_tree()
        enhancer = _enhance_module().UnifiedNodeEnhancement("m")
        cache = {}
        siblings = view.children_of(root)
        with patch.object(_enhance_module(), "llm_completion",
                          side_effect=lambda m, p, **k: _enh_response([])):
            await st._navigate_level("q", siblings, view, {}, enhancer, [], cache)
            first = calls["n"]
            await st._navigate_level("q", siblings, view, {}, enhancer, [], cache)
            second = calls["n"]
        assert first >= 1          # 首次惰性聚合
        assert second == first     # 第二次全命中缓存，无新聚合查询

    @pytest.mark.asyncio
    async def test_profile_evidence_drives_layer_selection_end_to_end(self, nav_index):
        """聚合签名驱动逐层精挑：实体证据接地进 prompt，LLM 依证据选分支。"""
        st, db, _ = nav_index
        d1 = db.insert_document("a.pdf", "/tmp/a.pdf")
        d2 = db.insert_document("b.pdf", "/tmp/b.pdf")
        root = db.insert_corpus_tree_node(None, "知识库", "", 0, kind="root")
        ca = db.insert_corpus_tree_node(root, "人物档案", "", 1, kind="cluster")
        cb = db.insert_corpus_tree_node(root, "财务报表", "", 1, kind="cluster")
        db.add_corpus_membership(d1, ca, 1.0)
        db.add_corpus_membership(d2, cb, 1.0)
        eid = db.insert_entity("person", "张三")
        db.insert_entity_mention(eid, d1, confidence=0.9)
        prompts = []

        def fake(model, prompt, **kw):
            if M_ENH in prompt:
                prompts.append(prompt)
                if "实体匹配：张三" in prompt:
                    titles = _candidates_in_enhance_prompt(prompt)
                    return _enh_response([titles["人物档案"]])
            return _enh_response([])

        with patch.object(_enhance_module(), "llm_completion", side_effect=fake):
            docs = await st.navigate_tree("张三的档案")
        assert docs == {d1: 1.0}
        assert "实体匹配：张三" in prompts[0]


# ---------------------------------------------------------------------------
# T6.4/[3.2.1] 延迟池 + pool_concern 层重试（cap×2，只重跑本层）
# ---------------------------------------------------------------------------


class TestLayerDeferredAndRetry:
    """收窄纪律：超限候选进延迟池（暴露可回捞）；pool_concern + deferred →
    本层放宽 cap×2 重跑 enhance 一次（不回溯上层）。"""

    def _four_tagged_clusters(self, db):
        root = db.insert_corpus_tree_node(None, "知识库", "", 0, kind="root")
        for i in range(1, 5):
            d = db.insert_document(f"d{i}.pdf", f"/tmp/{i}.pdf")
            c = db.insert_corpus_tree_node(root, f"风控簇{i}", "", 1, kind="cluster")
            db.add_corpus_membership(d, c, 1.0)
            db.insert_closet_tags(d, [(d, "风控", "风控", 0.9, "llm")])
        return root

    @pytest.mark.asyncio
    async def test_overflow_deferred_surfaced_and_retry_relaxes_cap(self, nav_index):
        from pageindex_mutil.agentic.enhance import POOL_CONCERN_RETRY_CAP_MULTIPLIER
        st, db, _ = nav_index
        self._four_tagged_clusters(db)

        UnifiedNodeEnhancement = _enhance_module().UnifiedNodeEnhancement
        calls, results = [], []

        class SpyEnhancer(UnifiedNodeEnhancement):
            def __init__(self, model, retrieve_model=None):
                super().__init__(model, retrieve_model=retrieve_model)
                self.union_max_candidates = 2  # union=4 > cap=2 → deferred 2

            async def enhance_and_select(self, query, candidates, profiles,
                                         query_entities=None, node_budget=None,
                                         token_budget=None, max_candidates=None):
                calls.append(max_candidates)
                result = await super().enhance_and_select(
                    query, candidates, profiles, query_entities=query_entities,
                    node_budget=node_budget, token_budget=token_budget,
                    max_candidates=max_candidates,
                )
                results.append(result)
                return result

        def fake(model, prompt, **kw):
            if M_ENH in prompt:
                titles = _candidates_in_enhance_prompt(prompt)
                return _enh_response(list(titles.values()), pool_concern=True,
                                     concern_reason="疑似漏掉分支")
            return ""

        with patch.object(_enhance_module(), "UnifiedNodeEnhancement", SpyEnhancer), \
                patch.object(_enhance_module(), "llm_completion", side_effect=fake):
            docs = await st.navigate_tree("风控")

        # union=4 > cap=2 → 被截 2 个进延迟池（暴露可回捞，不硬丢）
        assert len(results[0]["deferred"]) == 2
        # pool_concern + deferred → 重跑本层一次：cap ×POOL_CONCERN_RETRY_CAP_MULTIPLIER
        assert calls[0] is None
        assert calls[1] == 2 * POOL_CONCERN_RETRY_CAP_MULTIPLIER
        assert len(calls) == 2  # 只重跑一次
        # 重跑后 union 全量 4 簇进精挑并被选中展开
        assert len(docs) == 4

    @pytest.mark.asyncio
    async def test_no_retry_when_union_fits(self, nav_index):
        """union 未超限（deferred 为空）→ 仅 pool_concern 不触发层重试。"""
        st, db, _ = nav_index
        root = db.insert_corpus_tree_node(None, "知识库", "", 0, kind="root")
        d = db.insert_document("d.pdf", "/tmp/d.pdf")
        c = db.insert_corpus_tree_node(root, "风控簇", "", 1, kind="cluster")
        db.add_corpus_membership(d, c, 1.0)
        db.insert_closet_tags(d, [(d, "风控", "风控", 0.9, "llm")])

        UnifiedNodeEnhancement = _enhance_module().UnifiedNodeEnhancement
        calls = []

        class SpyEnhancer(UnifiedNodeEnhancement):
            async def enhance_and_select(self, query, candidates, profiles,
                                         query_entities=None, node_budget=None,
                                         token_budget=None, max_candidates=None):
                calls.append(max_candidates)
                return await super().enhance_and_select(
                    query, candidates, profiles, query_entities=query_entities,
                    node_budget=node_budget, token_budget=token_budget,
                    max_candidates=max_candidates,
                )

        def fake(model, prompt, **kw):
            if M_ENH in prompt:
                titles = _candidates_in_enhance_prompt(prompt)
                return _enh_response(list(titles.values()), pool_concern=True,
                                     concern_reason="证据偏弱")
            return ""

        with patch.object(_enhance_module(), "UnifiedNodeEnhancement", SpyEnhancer), \
                patch.object(_enhance_module(), "llm_completion", side_effect=fake):
            docs = await st.navigate_tree("风控")
        assert calls == [None]  # 无被截候选 → 不重跑
        assert len(docs) == 1
