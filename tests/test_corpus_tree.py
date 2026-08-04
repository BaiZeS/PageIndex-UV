"""P1 语料树构建（无向量管线）测试 —— 设计文档 [S3]/[S3.1]。

验收覆盖：
1. 产出可检视语料树，结构合理（get_tree 嵌套结构）；
2. 文档覆盖率 100%（每篇至少挂 1 簇，软归属可多挂；无标签文档挂"未分类"）；
3. 簇大小分布落在卡界区间（越界簇有合并/拆分处置记录）；
4. 标签归一一致性（规范集内无同义标签并存，如 风控/风险管理）。

另覆盖：确定性倒排分组（不走 LLM）、增量更新两点细化（标签先匹配已有规范集、
簇卡界每次插入时评估）、client.index() 增量钩子接线、retrieve_model 接线（NFR4）。

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

# Avoid triggering __init__.py imports that pull in heavy deps like PyPDF2.
pkg_path = Path(__file__).parent.parent / "pageindex_mutil"
sys.path.insert(0, str(pkg_path))

import importlib.util

# Load corpus_tree without triggering the heavy package __init__ (PyPDF2 etc.).
# IMPORTANT (test isolation): only seed a stub ``pageindex_mutil.utils`` when the
# real module is NOT already imported. In the full suite, test_config.py imports
# the real utils first, and test_delete_path.py (collected after this file) needs
# the real ``get_llm_config`` — clobbering utils here would break its collection.
# corpus_tree.py binds llm_completion/extract_json into its own namespace and we
# patch ``corpus_tree_mod.llm_completion`` directly, so the stub is only needed to
# satisfy the relative import on standalone runs.
if "pageindex_mutil.utils" not in sys.modules:
    utils_spec = importlib.util.spec_from_file_location("pageindex_mutil.utils", pkg_path / "utils.py")
    utils_mod = importlib.util.module_from_spec(utils_spec)
    sys.modules["pageindex_mutil.utils"] = utils_mod
    utils_mod.llm_completion = lambda *a, **k: None
    utils_mod.count_tokens = lambda text, model=None: len(text or "") // 4

    def _mock_extract_json(text):
        try:
            return json.loads(text)
        except Exception:
            return None

    utils_mod.extract_json = _mock_extract_json

if "pageindex_mutil.corpus_tree" in sys.modules:
    # Full-suite run: the real module was already imported (via the package
    # __init__). Reuse it so we don't shadow the real module for other tests.
    corpus_tree_mod = sys.modules["pageindex_mutil.corpus_tree"]
else:
    spec = importlib.util.spec_from_file_location("pageindex_mutil.corpus_tree", pkg_path / "corpus_tree.py")
    corpus_tree_mod = importlib.util.module_from_spec(spec)
    sys.modules["pageindex_mutil.corpus_tree"] = corpus_tree_mod
    spec.loader.exec_module(corpus_tree_mod)

CorpusTreeBuilder = corpus_tree_mod.CorpusTreeBuilder

# Prompt markers used to route mocked LLM responses.
M_NORM = "标签归一化"
M_SPLIT = "簇拆分"
M_MERGE = "簇合并裁定"
M_ATTACH = "挂簇裁定"
M_UPPER = "上层结构"
M_SIMILAR = "过相似"


@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = PageIndexDB(path)
    yield db
    db.close()
    os.unlink(path)


def _add_doc(db, name, tags, description=""):
    """Insert a document with closet tags. tags = [(tag_text, confidence)]."""
    doc_id = db.insert_document(name, f"/tmp/{name}", doc_description=description)
    if tags:
        db.insert_closet_tags(
            doc_id, [(doc_id, t, t, conf, "llm") for t, conf in tags]
        )
    return doc_id


def _route_llm(handlers):
    """Build a fake llm_completion routing canned responses by prompt marker."""
    calls = []

    def fake(model, prompt, chat_history=None, return_finish_reason=False):
        calls.append((model, prompt))
        for marker, resp in handlers.items():
            if marker in prompt:
                return resp(prompt) if callable(resp) else resp
        return ""

    fake.calls = calls
    return fake


def _leaf_clusters(tree):
    """Collect cluster nodes holding documents (direct membership) from a tree dict."""
    out = []

    def walk(node):
        if node.get("docs"):
            out.append(node)
        for c in node.get("children", []):
            walk(c)

    if tree:
        walk(tree)
    return out


def _all_doc_ids(tree):
    ids = set()
    for c in _leaf_clusters(tree):
        ids.update(d["doc_id"] for d in c["docs"])
    return ids


# ---------------------------------------------------------------------------
# 存储层（corpus_tree 表/字段）
# ---------------------------------------------------------------------------


class TestCorpusTreeStorage:
    def test_schema_tables_created(self, tmp_db):
        rows = tmp_db._connect().execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r[0] for r in rows}
        assert "corpus_tree_nodes" in names
        assert "corpus_tree_membership" in names
        assert "corpus_tag_norm" in names
        assert "corpus_tree_events" in names

    def test_norm_map_set_get_and_upsert(self, tmp_db):
        tmp_db.set_corpus_tag_norm_map({"风控": "风险管理", "风险管理": "风险管理"})
        m = tmp_db.get_corpus_tag_norm_map()
        assert m == {"风控": "风险管理", "风险管理": "风险管理"}
        # upsert overwrites existing raw tag, inserts new ones
        tmp_db.upsert_corpus_tag_norm("风控", "企业风险控制")
        tmp_db.upsert_corpus_tag_norm("合规", "合规")
        m = tmp_db.get_corpus_tag_norm_map()
        assert m["风控"] == "企业风险控制"
        assert m["合规"] == "合规"

    def test_canonical_tags_distinct(self, tmp_db):
        tmp_db.set_corpus_tag_norm_map({"风控": "风险管理", "风险管理": "风险管理", "合规": "合规"})
        assert sorted(tmp_db.get_corpus_canonical_tags()) == ["合规", "风险管理"]

    def test_membership_upsert_and_queries(self, tmp_db):
        doc_id = tmp_db.insert_document("a.pdf", "/tmp/a.pdf")
        node_id = tmp_db.insert_corpus_tree_node(None, "风险管理", "s", 1, kind="cluster", tag="风险管理")
        tmp_db.add_corpus_membership(doc_id, node_id, 0.9)
        tmp_db.add_corpus_membership(doc_id, node_id, 0.7)  # upsert, no duplicate
        docs = tmp_db.get_corpus_node_docs(node_id)
        assert docs == [(doc_id, 0.7)]
        mem = tmp_db.get_corpus_doc_memberships(doc_id)
        assert mem == [(node_id, 0.7)]
        all_mem = tmp_db.get_all_corpus_memberships()
        assert all_mem == [(doc_id, node_id, 0.7)]

    def test_events_record_and_filter(self, tmp_db):
        tmp_db.insert_corpus_tree_event(1, "split", '{"from": 5}')
        tmp_db.insert_corpus_tree_event(2, "merge", '{"into": "x"}')
        assert len(tmp_db.get_corpus_tree_events()) == 2
        splits = tmp_db.get_corpus_tree_events(event_type="split")
        assert len(splits) == 1
        assert splits[0]["detail"] == '{"from": 5}'

    def test_corpus_tree_clear(self, tmp_db):
        doc_id = tmp_db.insert_document("a.pdf", "/tmp/a.pdf")
        node_id = tmp_db.insert_corpus_tree_node(None, "t", "s", 1)
        tmp_db.add_corpus_membership(doc_id, node_id, 1.0)
        tmp_db.insert_corpus_tree_event(node_id, "split", "{}")
        tmp_db.corpus_tree_clear()
        assert tmp_db.get_corpus_tree_nodes() == []
        assert tmp_db.get_all_corpus_memberships() == []
        assert tmp_db.get_corpus_tree_events() == []


# ---------------------------------------------------------------------------
# 标签归一化（验收 4：规范集内无同义标签并存）
# ---------------------------------------------------------------------------


class TestTagNormalization:
    def test_synonyms_merged_into_single_canonical(self, tmp_db):
        """风控/风险管理 同义 → 归一为同一规范标签，且两篇文档进入同一簇。"""
        d1 = _add_doc(tmp_db, "风控手册.pdf", [("风控", 0.9)])
        d2 = _add_doc(tmp_db, "风险管理指南.pdf", [("风险管理", 0.9)])
        fake = _route_llm({
            M_NORM: json.dumps({"groups": [
                {"canonical": "风险管理", "synonyms": ["风控", "风险管理"]},
            ]}),
        })
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            b = CorpusTreeBuilder(tmp_db, model="m", cluster_min=1)
            tree = b.rebuild()
        norm = tmp_db.get_corpus_tag_norm_map()
        assert norm["风控"] == norm["风险管理"] == "风险管理"
        # 同义标签的两篇文档落入同一簇
        leaves = _leaf_clusters(tree)
        together = [c for c in leaves if {d["doc_id"] for d in c["docs"]} == {d1, d2}]
        assert len(together) == 1

    def test_norm_map_covers_every_raw_tag(self, tmp_db):
        """归一结果必须覆盖全部原始标签（LLM 漏掉的标签保持自身为规范标签）。"""
        _add_doc(tmp_db, "a.pdf", [("风控", 0.9), ("合规", 0.8)])
        fake = _route_llm({
            # LLM 只处理了"风控"，漏掉"合规"
            M_NORM: json.dumps({"groups": [
                {"canonical": "风险管理", "synonyms": ["风控"]},
            ]}),
        })
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            CorpusTreeBuilder(tmp_db, model="m", cluster_min=1).rebuild()
        norm = tmp_db.get_corpus_tag_norm_map()
        assert norm["风控"] == "风险管理"
        assert norm["合规"] == "合规"

    def test_llm_failure_falls_back_to_identity(self, tmp_db):
        """LLM 不可用时归一退化为恒等映射，建树不崩溃、覆盖不丢。"""
        d1 = _add_doc(tmp_db, "a.pdf", [("风控", 0.9)])
        with patch.object(corpus_tree_mod, "llm_completion", return_value=""):
            b = CorpusTreeBuilder(tmp_db, model="m", cluster_min=1)
            tree = b.rebuild()
        norm = tmp_db.get_corpus_tag_norm_map()
        assert norm == {"风控": "风控"}
        assert _all_doc_ids(tree) == {d1}


# ---------------------------------------------------------------------------
# 确定性倒排分组（③ 不走 LLM）
# ---------------------------------------------------------------------------


class TestDeterministicGrouping:
    def test_same_canonical_same_cluster_different_tags_separated(self, tmp_db):
        d1 = _add_doc(tmp_db, "a.pdf", [("风险管理", 0.9)])
        d2 = _add_doc(tmp_db, "b.pdf", [("风险管理", 0.8)])
        d3 = _add_doc(tmp_db, "c.pdf", [("前端开发", 0.8)])
        fake = _route_llm({M_NORM: json.dumps({"groups": [
            {"canonical": "风险管理", "synonyms": ["风险管理"]},
            {"canonical": "前端开发", "synonyms": ["前端开发"]},
        ]})})
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            tree = CorpusTreeBuilder(tmp_db, model="m", cluster_min=1).rebuild()
        leaves = _leaf_clusters(tree)
        by_title = {c["title"]: {d["doc_id"] for d in c["docs"]} for c in leaves}
        assert by_title["风险管理"] == {d1, d2}
        assert by_title["前端开发"] == {d3}

    def test_grouping_issues_no_llm_calls(self, tmp_db):
        """分组是纯倒排 group-by：除标签归一与合并保护外不产生 LLM 调用。"""
        for i in range(4):
            _add_doc(tmp_db, f"a{i}.pdf", [("风险管理", 0.9)])
            _add_doc(tmp_db, f"b{i}.pdf", [("前端开发", 0.9)])
        fake = _route_llm({M_NORM: json.dumps({"groups": [
            {"canonical": "风险管理", "synonyms": ["风险管理"]},
            {"canonical": "前端开发", "synonyms": ["前端开发"]},
        ]})})
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            CorpusTreeBuilder(tmp_db, model="m", cluster_min=1).rebuild()
        # 只发生归一化与合并保护（过相似合并）两类 LLM 调用
        # （无拆分/过小合并/分组/挂簇裁定）
        assert len(fake.calls) >= 1
        for _, prompt in fake.calls:
            assert M_NORM in prompt or M_SIMILAR in prompt


# ---------------------------------------------------------------------------
# 树结构与覆盖率（验收 1/2）
# ---------------------------------------------------------------------------


class TestTreeBuildAndCoverage:
    def test_get_tree_shape_inspectable(self, tmp_db):
        _add_doc(tmp_db, "a.pdf", [("风险管理", 0.9)])
        fake = _route_llm({M_NORM: json.dumps({"groups": [
            {"canonical": "风险管理", "synonyms": ["风险管理"]}]})})
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            b = CorpusTreeBuilder(tmp_db, model="m", cluster_min=1)
            tree = b.rebuild()
        assert tree["title"] == "知识库"
        assert tree["kind"] == "root"
        assert tree["level"] == 0
        child = tree["children"][0]
        assert child["title"] == "风险管理"
        assert child["kind"] == "cluster"
        assert child["parent_id"] == tree["node_id"]
        assert child["docs"][0]["doc_name"] == "a.pdf"
        # get_tree() 可重复读取已持久化的树
        again = b.get_tree()
        assert again["node_id"] == tree["node_id"]

    def test_full_coverage_including_tagless_doc(self, tmp_db):
        """验收 2：每篇文档至少挂 1 簇；无标签文档挂"未分类"兜底簇。"""
        d1 = _add_doc(tmp_db, "a.pdf", [("风险管理", 0.9)])
        d2 = _add_doc(tmp_db, "notags.pdf", [])
        fake = _route_llm({M_NORM: json.dumps({"groups": [
            {"canonical": "风险管理", "synonyms": ["风险管理"]}]})})
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            tree = CorpusTreeBuilder(tmp_db, model="m", cluster_min=1).rebuild()
        assert _all_doc_ids(tree) == {d1, d2}
        leaves = _leaf_clusters(tree)
        fallback = [c for c in leaves if c["title"] == "未分类"]
        assert len(fallback) == 1
        assert {d["doc_id"] for d in fallback[0]["docs"]} == {d2}
        # 覆盖率：membership 表层面每篇文档至少一条
        for did in (d1, d2):
            assert len(tmp_db.get_corpus_doc_memberships(did)) >= 1

    def test_soft_membership_doc_in_multiple_clusters(self, tmp_db):
        """软归属：一篇跨主题文档挂多个簇（DAG），不被硬切碎。"""
        d1 = _add_doc(tmp_db, "cross.pdf", [("风险管理", 0.9), ("前端开发", 0.7)])
        fake = _route_llm({M_NORM: json.dumps({"groups": [
            {"canonical": "风险管理", "synonyms": ["风险管理"]},
            {"canonical": "前端开发", "synonyms": ["前端开发"]},
        ]})})
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            tree = CorpusTreeBuilder(tmp_db, model="m", cluster_min=1).rebuild()
        mem = tmp_db.get_corpus_doc_memberships(d1)
        assert len(mem) == 2
        weights = {node_id: w for node_id, w in mem}
        nodes = {n["id"]: n for n in tmp_db.get_corpus_tree_nodes()}
        assert {nodes[nid]["title"] for nid in weights} == {"风险管理", "前端开发"}


# ---------------------------------------------------------------------------
# 簇大小双向卡界（验收 3：越界簇有合并/拆分处置记录）
# ---------------------------------------------------------------------------


class TestClusterSizeBounds:
    def test_oversized_cluster_split_by_llm(self, tmp_db):
        """过大簇 → LLM 裁定拆分；拆后各簇 ≤ max，且有 split 处置记录。"""
        doc_ids = [_add_doc(tmp_db, f"d{i}.pdf", [("风险管理", 0.9)]) for i in range(5)]
        fake = _route_llm({
            M_NORM: json.dumps({"groups": [
                {"canonical": "风险管理", "synonyms": ["风险管理"]}]}),
            M_SPLIT: json.dumps({"clusters": [
                {"title": "风险识别", "summary": "s1", "doc_ids": doc_ids[:3]},
                {"title": "风险处置", "summary": "s2", "doc_ids": doc_ids[3:]},
            ]}),
        })
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            b = CorpusTreeBuilder(tmp_db, model="m", cluster_min=1, cluster_max=3)
            tree = b.rebuild()
        leaves = _leaf_clusters(tree)
        assert all(len(c["docs"]) <= 3 for c in leaves)
        assert _all_doc_ids(tree) == set(doc_ids)
        splits = tmp_db.get_corpus_tree_events(event_type="split")
        assert len(splits) == 1

    def test_split_deterministic_fallback_when_llm_invalid(self, tmp_db):
        """LLM 拆分结果非法（漏文档）→ 确定性均分兜底，仍满足 ≤ max。"""
        doc_ids = [_add_doc(tmp_db, f"d{i}.pdf", [("风险管理", 0.9)]) for i in range(5)]
        fake = _route_llm({
            M_NORM: json.dumps({"groups": [
                {"canonical": "风险管理", "synonyms": ["风险管理"]}]}),
            M_SPLIT: json.dumps({"clusters": [
                {"title": "只有一半", "summary": "s", "doc_ids": doc_ids[:2]},
            ]}),
        })
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            tree = CorpusTreeBuilder(tmp_db, model="m", cluster_min=1, cluster_max=3).rebuild()
        leaves = _leaf_clusters(tree)
        assert all(len(c["docs"]) <= 3 for c in leaves)
        assert _all_doc_ids(tree) == set(doc_ids)
        assert len(tmp_db.get_corpus_tree_events(event_type="split")) >= 1

    def test_small_cluster_merged_into_sibling(self, tmp_db):
        """过小簇 → LLM 裁定并入兄弟簇；merge 处置记录；覆盖不丢。"""
        d_big = [_add_doc(tmp_db, f"big{i}.pdf", [("风险管理", 0.9)]) for i in range(3)]
        d_small = _add_doc(tmp_db, "small.pdf", [("合规", 0.9)])
        fake = _route_llm({
            M_NORM: json.dumps({"groups": [
                {"canonical": "风险管理", "synonyms": ["风险管理"]},
                {"canonical": "合规", "synonyms": ["合规"]}]}),
            M_MERGE: json.dumps({"target": "风险管理"}),
        })
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            tree = CorpusTreeBuilder(tmp_db, model="m", cluster_min=2, cluster_max=50).rebuild()
        assert _all_doc_ids(tree) == set(d_big) | {d_small}
        merges = tmp_db.get_corpus_tree_events(event_type="merge")
        assert len(merges) == 1
        # 合规簇已被合并消失
        assert all(c["title"] != "合规" for c in _leaf_clusters(tree))

    def test_small_cluster_no_candidates_kept_with_record(self, tmp_db):
        """过小且无可并入对象 → 保留，但留下处置记录（不静默）。"""
        _add_doc(tmp_db, "solo.pdf", [("孤本主题", 0.9)])
        fake = _route_llm({M_NORM: json.dumps({"groups": [
            {"canonical": "孤本主题", "synonyms": ["孤本主题"]}]})})
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            tree = CorpusTreeBuilder(tmp_db, model="m", cluster_min=5, cluster_max=50).rebuild()
        assert len(_leaf_clusters(tree)) == 1
        events = tmp_db.get_corpus_tree_events()
        assert any(e["event_type"] == "merge_skipped" for e in events)


# ---------------------------------------------------------------------------
# 合并保护：过相似的兄弟簇合并（[S3.1] 细而不碎·主动闸门）
# ---------------------------------------------------------------------------


class TestSimilarMergeProtection:
    """过相似兄弟簇合并：LLM 语义裁定，宁缺毋滥；失败/无把握时不合并。"""

    def _two_adequate_siblings(self, tmp_db):
        """构造两个规模达标（≥cluster_min）的兄弟簇，各 2 篇文档。"""
        d1 = [_add_doc(tmp_db, f"a{i}.pdf", [("风险管理", 0.9)]) for i in range(2)]
        d2 = [_add_doc(tmp_db, f"b{i}.pdf", [("风控管理", 0.9)]) for i in range(2)]
        norm = json.dumps({"groups": [
            {"canonical": "风险管理", "synonyms": ["风险管理"]},
            {"canonical": "风控管理", "synonyms": ["风控管理"]},
        ]})
        return d1, d2, norm

    def test_similar_pair_merged_memberships_moved_event_recorded(self, tmp_db):
        """(a) LLM 返回过相似簇对 → 合并、membership 迁移、merge_similar 记录、覆盖不丢。"""
        d1, d2, norm = self._two_adequate_siblings(tmp_db)
        fake = _route_llm({
            M_NORM: norm,
            M_SIMILAR: json.dumps({"pairs": [["风险管理", "风控管理"]]}),
        })
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            tree = CorpusTreeBuilder(tmp_db, model="m", cluster_min=2, cluster_max=50).rebuild()
        # 两个过相似簇合并为一个，全部文档迁入幸存簇
        leaves = _leaf_clusters(tree)
        assert len(leaves) == 1
        assert len(leaves[0]["docs"]) == 4
        assert _all_doc_ids(tree) == set(d1) | set(d2)
        # 记录 merge_similar 处置事件
        events = tmp_db.get_corpus_tree_events(event_type="merge_similar")
        assert len(events) == 1
        # 每个文档仍有归属（覆盖不丢）
        for did in list(d1) + list(d2):
            assert len(tmp_db.get_corpus_doc_memberships(did)) >= 1

    def test_similar_empty_pairs_no_merge(self, tmp_db):
        """(b) LLM 返回空列表 → 不合并（宁缺毋滥）。"""
        d1, d2, norm = self._two_adequate_siblings(tmp_db)
        fake = _route_llm({
            M_NORM: norm,
            M_SIMILAR: json.dumps({"pairs": []}),
        })
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            tree = CorpusTreeBuilder(tmp_db, model="m", cluster_min=2, cluster_max=50).rebuild()
        # 两簇保持独立
        leaves = _leaf_clusters(tree)
        assert len(leaves) == 2
        assert _all_doc_ids(tree) == set(d1) | set(d2)
        assert tmp_db.get_corpus_tree_events(event_type="merge_similar") == []

    def test_similar_llm_failure_no_merge_no_crash(self, tmp_db):
        """(c) LLM 失败（抛异常）→ 不合并、不崩溃（自动造的树错了比没有树更糟）。"""
        d1, d2, norm = self._two_adequate_siblings(tmp_db)

        def _boom(prompt):
            raise RuntimeError("LLM unavailable")

        fake = _route_llm({
            M_NORM: norm,
            M_SIMILAR: _boom,
        })
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            tree = CorpusTreeBuilder(tmp_db, model="m", cluster_min=2, cluster_max=50).rebuild()
        # 两簇保持独立，建树不崩溃
        leaves = _leaf_clusters(tree)
        assert len(leaves) == 2
        assert _all_doc_ids(tree) == set(d1) | set(d2)
        assert tmp_db.get_corpus_tree_events(event_type="merge_similar") == []


# ---------------------------------------------------------------------------
# 上层结构递归生成（④ generate_toc_init/continue 模式）
# ---------------------------------------------------------------------------


class TestUpperStructure:
    def test_grouping_init_continue_builds_parent_level(self, tmp_db):
        """簇数超扇出上限 → LLM 分批（init+continue）生成上层分组。"""
        tags = ["主题甲", "主题乙", "主题丙", "主题丁", "主题戊"]
        doc_ids = []
        for i, t in enumerate(tags):
            doc_ids.append(_add_doc(tmp_db, f"d{i}.pdf", [(t, 0.9)]))
        fake = _route_llm({
            M_NORM: json.dumps({"groups": [
                {"canonical": t, "synonyms": [t]} for t in tags
            ]}),
            M_UPPER: lambda prompt: json.dumps({"groups": [
                {"title": "分组AB", "summary": "甲乙", "members": ["主题甲", "主题乙"]},
                {"title": "分组CD", "summary": "丙丁", "members": ["主题丙", "主题丁"]},
            ]}),
        })
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            b = CorpusTreeBuilder(tmp_db, model="m", cluster_min=1, cluster_max=50)
            b._MAX_ROOT_FANOUT = 2
            b._GROUP_BATCH_SIZE = 3  # 5 簇 → init(3) + continue(2)
            tree = b.rebuild()
        # init 与 continue 两种调用都发生过
        upper_calls = [p for _, p in fake.calls if M_UPPER in p]
        assert len(upper_calls) >= 2
        titles = {c["title"] for c in tree["children"]}
        assert "分组AB" in titles and "分组CD" in titles
        # 未归组的"主题戊"仍是 ROOT 直接孩子
        assert "主题戊" in titles
        # 分组节点是簇的父级
        ab = next(c for c in tree["children"] if c["title"] == "分组AB")
        assert {gc["title"] for gc in ab["children"]} == {"主题甲", "主题乙"}
        # 覆盖不丢
        assert _all_doc_ids(tree) == set(doc_ids)


# ---------------------------------------------------------------------------
# 增量更新（[S3] 增量两点细化）
# ---------------------------------------------------------------------------


class TestIncrementalUpdate:
    def _build_base(self, db, **kw):
        _add_doc(db, "a.pdf", [("风险管理", 0.9)])
        fake = _route_llm({M_NORM: json.dumps({"groups": [
            {"canonical": "风险管理", "synonyms": ["风险管理"]}]})})
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            b = CorpusTreeBuilder(db, model="m", **kw)
            b.rebuild()
        return b

    def test_existing_tag_matches_without_renormalization(self, tmp_db):
        """增量①：新文档标签命中已有规范集 → 直接挂簇，不重跑全库归一。"""
        b = self._build_base(tmp_db)
        d_new = _add_doc(tmp_db, "new.pdf", [("风险管理", 0.8)])
        fake = _route_llm({})  # 任何 LLM 调用都返回空 → 若误调归一会退化
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            b.update_for_document(d_new)
        assert not any(M_NORM in p for _, p in fake.calls)
        mem = tmp_db.get_corpus_doc_memberships(d_new)
        assert len(mem) == 1
        node = next(n for n in tmp_db.get_corpus_tree_nodes() if n["id"] == mem[0][0])
        assert node["title"] == "风险管理"

    def test_new_tag_single_adjudication_merges_into_existing(self, tmp_db):
        """增量①：未命中标签 → LLM 单点裁定并入已有规范标签（不重跑全库）。"""
        b = self._build_base(tmp_db)
        d_new = _add_doc(tmp_db, "new.pdf", [("风控", 0.8)])
        fake = _route_llm({
            M_NORM: json.dumps({"canonical": "风险管理"}),
        })
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            b.update_for_document(d_new)
        # 单点裁定：只对这一个标签一次归一裁定调用
        norm_calls = [p for _, p in fake.calls if M_NORM in p]
        assert len(norm_calls) == 1
        assert tmp_db.get_corpus_tag_norm_map()["风控"] == "风险管理"
        mem = tmp_db.get_corpus_doc_memberships(d_new)
        node = next(n for n in tmp_db.get_corpus_tree_nodes() if n["id"] == mem[0][0])
        assert node["title"] == "风险管理"

    def test_new_tag_opens_new_canonical_and_cluster(self, tmp_db):
        """增量：裁定新开规范标签 → 挂簇裁定新开簇或并入现有簇。"""
        b = self._build_base(tmp_db)
        d_new = _add_doc(tmp_db, "new.pdf", [("前端开发", 0.8)])
        fake = _route_llm({
            M_NORM: json.dumps({"canonical": "前端开发"}),
            M_ATTACH: json.dumps({"action": "create", "title": "前端开发", "summary": "前端相关"}),
        })
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            b.update_for_document(d_new)
        assert tmp_db.get_corpus_tag_norm_map()["前端开发"] == "前端开发"
        mem = tmp_db.get_corpus_doc_memberships(d_new)
        assert len(mem) == 1
        node = next(n for n in tmp_db.get_corpus_tree_nodes() if n["id"] == mem[0][0])
        assert node["title"] == "前端开发"
        assert any(e["event_type"] == "new_cluster" for e in tmp_db.get_corpus_tree_events())

    def test_insert_evaluates_bounds_split_on_overflow(self, tmp_db):
        """增量②：每次插入评估卡界 —— 挂簇后超上限立即拆分。"""
        b = self._build_base(tmp_db, cluster_min=1, cluster_max=2)
        d2 = _add_doc(tmp_db, "b.pdf", [("风险管理", 0.9)])
        d3 = _add_doc(tmp_db, "c.pdf", [("风险管理", 0.9)])
        fake = _route_llm({
            M_NORM: json.dumps({"canonical": "风险管理"}),
            M_SPLIT: lambda prompt: json.dumps({"clusters": []}),  # 逼确定性兜底
        })
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            b.update_for_document(d2)
            b.update_for_document(d3)  # 第 3 篇挂入 → 簇规模 3 > max=2 → 拆分
        leaves = _leaf_clusters(b.get_tree())
        assert all(len(c["docs"]) <= 2 for c in leaves)
        assert _all_doc_ids(b.get_tree()) >= {d2, d3}
        assert len(tmp_db.get_corpus_tree_events(event_type="split")) >= 1

    def test_incremental_tagless_doc_fallback(self, tmp_db):
        """增量：无标签文档挂"未分类"兜底簇。"""
        b = self._build_base(tmp_db)
        d_new = _add_doc(tmp_db, "notags.pdf", [])
        with patch.object(corpus_tree_mod, "llm_completion", return_value=""):
            b.update_for_document(d_new)
        mem = tmp_db.get_corpus_doc_memberships(d_new)
        assert len(mem) == 1
        node = next(n for n in tmp_db.get_corpus_tree_nodes() if n["id"] == mem[0][0])
        assert node["title"] == "未分类"


# ---------------------------------------------------------------------------
# retrieve_model 接线（NFR4）
# ---------------------------------------------------------------------------


class TestRetrieveModelWiring:
    def _two_tag_setup(self, tmp_db):
        _add_doc(tmp_db, "a.pdf", [("风险管理", 0.9)])
        _add_doc(tmp_db, "b.pdf", [("合规审查", 0.8)])
        return _route_llm({M_NORM: json.dumps({"groups": [
            {"canonical": "风险管理", "synonyms": ["风险管理"]},
            {"canonical": "合规审查", "synonyms": ["合规审查"]},
        ]})})

    def test_uses_retrieve_model_when_set(self, tmp_db):
        fake = self._two_tag_setup(tmp_db)
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            CorpusTreeBuilder(tmp_db, model="m", retrieve_model="r-model",
                              cluster_min=1).rebuild()
        assert fake.calls
        assert all(model == "r-model" for model, _ in fake.calls)

    def test_falls_back_to_model(self, tmp_db):
        fake = self._two_tag_setup(tmp_db)
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            CorpusTreeBuilder(tmp_db, model="m", retrieve_model=None,
                              cluster_min=1).rebuild()
        assert fake.calls
        assert all(model == "m" for model, _ in fake.calls)


# ---------------------------------------------------------------------------
# client.index() 增量钩子接线
# ---------------------------------------------------------------------------


def _import_real_client():
    """Purge stubbed pageindex_mutil modules and import the real client."""
    sys.modules.setdefault("PyPDF2", MagicMock())
    for k in list(sys.modules):
        if k == "pageindex_mutil" or k.startswith("pageindex_mutil."):
            del sys.modules[k]
    from pageindex_mutil.client import PageIndexClient
    return PageIndexClient


class TestClientWiring:
    def test_client_initializes_corpus_tree_with_db(self, tmp_path):
        PageIndexClient = _import_real_client()
        # search_backend="keyword" keeps this test vectorless (no embedding load).
        client = PageIndexClient(db_path=str(tmp_path / "t.db"), search_backend="keyword")
        try:
            assert client.corpus_tree is not None
            assert type(client.corpus_tree).__name__ == "CorpusTreeBuilder"
        finally:
            client.close()

    def test_client_corpus_tree_none_without_db(self):
        PageIndexClient = _import_real_client()
        client = PageIndexClient()
        try:
            assert client.corpus_tree is None
        finally:
            client.close()

    def test_index_calls_corpus_tree_update_after_tags(self, tmp_path):
        """index() 在 closet 标签落库后调用 corpus_tree.update_for_document。"""
        PageIndexClient = _import_real_client()
        client = PageIndexClient(db_path=str(tmp_path / "t.db"), search_backend="keyword")
        try:
            client.corpus_tree.update_for_document = MagicMock()
            client.super_tree_index.on_document_added = MagicMock()
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
                f.write("# Test\n\nbody\n")
                md_path = f.name
            try:
                mock_structure = [{"node_id": "n1", "title": "T", "text": "x",
                                   "summary": "s", "level": 1}]
                with patch("pageindex_mutil.client.md_to_tree") as mock_md, \
                     patch.object(client.closet_index, "add_document"):
                    mock_md.return_value = {
                        "doc_name": "test.md", "doc_description": "d",
                        "line_count": 2, "structure": mock_structure,
                    }
                    client.index(md_path, mode="md")
                client.corpus_tree.update_for_document.assert_called_once_with(1)
            finally:
                os.unlink(md_path)
        finally:
            client.close()
