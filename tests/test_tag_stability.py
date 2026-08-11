"""T5.3 / P1.3: closet_tags 稳定性与可控性。

覆盖设计文档 [7.2] 三项改造：
1. fallback 分层降级：_fallback_tags 产物 source="fallback"，只进关键词层，
   不进语义标签通道（语义通道只认 source="llm" 的抽象标签，conf≥0.5）；
2. 增量归一锚定：新抽取 LLM 标签先与 corpus_tag_norm 已有规范集比对——
   语义近似复用既有 canonical 名（单点 LLM 裁定，复用 corpus_tree 逻辑），
   映射持久化 corpus_tag_norm；真新概念才新开；
3. 幂等：重索引同一文档产出相同标签集（temp=0 + 归一锚定），无重复行。

另覆盖确定性抽取的 K 上限（_MAX_TAGS_PER_DOC=5）与 retrieve_model 接线（NFR4）。

全部 LLM 调用均 mock，真实 SQLite。与 test_node_profiles.py 相同模式：
purge stubs → import 真模块并持有引用 → 一律 patch.object（不串 patch 路径），
避免其他测试文件 re-seed sys.modules 造成 patch 目标漂移。
"""
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

for _mod in list(sys.modules):
    if _mod == "pageindex_mutil" or _mod.startswith("pageindex_mutil."):
        del sys.modules[_mod]

import pageindex_mutil.closet_index as closet_mod
import pageindex_mutil.corpus_tree as corpus_tree_mod
from db import PageIndexDB
from pageindex_mutil.closet_index import ClosetIndex
from pageindex_mutil.corpus_tree import CorpusTreeBuilder

# Prompt markers used to route mocked LLM responses.
M_EXTRACT = "语义标签提取专家"
M_ARBITRATE = "增量单标签裁定"


@pytest.fixture
def db(tmp_path):
    d = PageIndexDB(str(tmp_path / "tags.db"))
    yield d
    d.close()


@pytest.fixture
def closet(db):
    return ClosetIndex(db, model="m", retrieve_model=None)


def _raw_rows(db, doc_id):
    rows = db._connect().execute(
        "SELECT tag_text, tag_token, confidence, source FROM closet_tags "
        "WHERE doc_id = ? ORDER BY id",
        (doc_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _route_llm(handlers):
    """Fake llm_completion routing canned responses by prompt marker."""
    calls = []

    def fake(model, prompt, chat_history=None, return_finish_reason=False, thinking_disabled=True):
        calls.append((model, prompt))
        for marker, resp in handlers.items():
            if marker in prompt:
                return resp(prompt) if callable(resp) else resp
        return ""

    fake.calls = calls
    return fake


def _live_corpus_tree():
    """The corpus_tree module object ``ClosetIndex._anchor_tags`` resolves at
    call time via its lazy ``from .corpus_tree import ...``.

    Other test files re-seed ``sys.modules`` during collection, so the live
    module may differ from the ``corpus_tree_mod`` captured at import time.
    """
    import importlib
    return importlib.import_module("pageindex_mutil.corpus_tree")


@contextmanager
def _patch_llm(fake):
    """Patch llm_completion on every module the code under test can reach.

    - ``closet_mod`` (captured): ClosetIndex._extract_tags globals.
    - ``corpus_tree_mod`` (captured): CorpusTreeBuilder._resolve_new_tag path.
    - live corpus_tree: ClosetIndex._anchor_tags' lazy import target.

    Dedup by id so double-patching the same object is avoided.
    """
    import contextlib
    targets = {}
    for m in (closet_mod, corpus_tree_mod, _live_corpus_tree()):
        targets[id(m)] = m
    with contextlib.ExitStack() as stack:
        for m in targets.values():
            stack.enter_context(patch.object(m, "llm_completion", side_effect=fake))
        yield


def _arb_calls(fake):
    return [p for _, p in fake.calls if M_ARBITRATE in p]


# ---------------------------------------------------------------------------
# 1. fallback 分层降级
# ---------------------------------------------------------------------------


class TestFallbackSourceDemotion:
    def test_llm_tags_stored_with_source_llm(self, db, closet):
        doc_id = db.insert_document("a.pdf", "/tmp/a.pdf")
        fake = _route_llm({M_EXTRACT: json.dumps([
            {"tag": "容器编排", "confidence": 0.9},
            {"tag": "微服务治理", "confidence": 0.8},
        ])})
        with _patch_llm(fake):
            closet.add_document(doc_id, "K8s实践", "容器平台", [{"title": "部署"}])
        rows = _raw_rows(db, doc_id)
        assert {(r["tag_text"], r["source"]) for r in rows} == {
            ("容器编排", "llm"), ("微服务治理", "llm")}

    def test_fallback_tags_stored_with_source_fallback(self, db, closet):
        """LLM 无产出 → jieba 兜底标签，source="fallback"、conf=0.3（不再冒充 llm）。"""
        doc_id = db.insert_document("b.pdf", "/tmp/b.pdf")
        fake = _route_llm({})  # 抽取返回空 → 走 fallback
        with _patch_llm(fake):
            closet.add_document(doc_id, "分布式存储实践", "存储架构说明", [{"title": "x"}])
        rows = _raw_rows(db, doc_id)
        assert rows, "fallback tags should exist"
        assert all(r["source"] == "fallback" for r in rows)
        assert all(abs(r["confidence"] - 0.3) < 1e-9 for r in rows)

    def test_semantic_channel_reads_only_llm_tags(self, db, closet):
        """语义通道（ClosetIndex.search）只认 source="llm"；关键词层保留 fallback。"""
        d_llm = db.insert_document("llm.pdf", "/tmp/llm.pdf")
        d_fb = db.insert_document("fb.pdf", "/tmp/fb.pdf")
        # doc1: LLM 抽象标签（与查询词面不重叠）
        fake = _route_llm({M_EXTRACT: json.dumps([{"tag": "数据治理", "confidence": 0.9}])})
        with _patch_llm(fake):
            closet.add_document(d_llm, "治理白皮书", "", [{"title": "x"}])
        # doc2: LLM 失败 → fallback 原词（含查询词"分布式/存储"）
        fake2 = _route_llm({})
        with _patch_llm(fake2):
            closet.add_document(d_fb, "分布式存储实践", "", [{"title": "x"}])
        assert all(r["source"] == "fallback" for r in _raw_rows(db, d_fb))

        # 语义通道：fallback 原词不得命中
        assert closet.search("分布式存储实践") == []
        # 关键词层（不带 source 过滤的倒排）：fallback 仍可命中
        keyword_hits = db.match_closet_tags(["分布式", "存储"])
        assert d_fb in [doc_id for doc_id, _ in keyword_hits]

    def test_get_doc_tags_source_filter(self, db):
        """DB 读取侧：source 过滤（默认返回全部，保持兼容）。"""
        doc_id = db.insert_document("c.pdf", "/tmp/c.pdf")
        db.insert_closet_tags(doc_id, [
            (doc_id, "容器编排", "容器 编排", 0.9, "llm"),
            (doc_id, "容器", "容器", 0.3, "fallback"),
        ])
        assert sorted(t["tag_text"] for t in db.get_doc_tags(doc_id)) == ["容器", "容器编排"]
        llm_only = db.get_doc_tags(doc_id, source="llm")
        assert [t["tag_text"] for t in llm_only] == ["容器编排"]
        assert db.get_doc_tags(doc_id, source="fallback")[0]["tag_text"] == "容器"


# ---------------------------------------------------------------------------
# 2. 增量归一锚定（复用 corpus_tree 单点裁定）
# ---------------------------------------------------------------------------


class TestCanonicalAnchoring:
    def test_similar_tag_reuses_existing_canonical_and_persists(self, db, closet):
        """新标签与已有规范语义近似 → 复用 canonical 名落库；映射持久化。"""
        db.upsert_corpus_tag_norm("容器编排", "容器编排")
        doc_id = db.insert_document("a.pdf", "/tmp/a.pdf")
        fake = _route_llm({
            M_EXTRACT: json.dumps([{"tag": "容器调度", "confidence": 0.9}]),
            M_ARBITRATE: json.dumps({"canonical": "容器编排"}),
        })
        with _patch_llm(fake):
            closet.add_document(doc_id, "调度系统", "", [{"title": "x"}])
        rows = _raw_rows(db, doc_id)
        assert [(r["tag_text"], r["source"]) for r in rows] == [("容器编排", "llm")]
        assert db.get_corpus_tag_norm_map()["容器调度"] == "容器编排"
        assert len(_arb_calls(fake)) == 1  # 单点裁定：仅一次

    def test_genuinely_new_tag_opens_new_canonical(self, db, closet):
        """真新概念 → 自身成为新规范标签并持久化映射。"""
        db.upsert_corpus_tag_norm("风险管理", "风险管理")
        doc_id = db.insert_document("a.pdf", "/tmp/a.pdf")
        fake = _route_llm({
            M_EXTRACT: json.dumps([{"tag": "前端开发", "confidence": 0.85}]),
            M_ARBITRATE: json.dumps({"canonical": "前端开发"}),
        })
        with _patch_llm(fake):
            closet.add_document(doc_id, "前端手册", "", [{"title": "x"}])
        rows = _raw_rows(db, doc_id)
        assert [(r["tag_text"], r["source"]) for r in rows] == [("前端开发", "llm")]
        assert db.get_corpus_tag_norm_map()["前端开发"] == "前端开发"

    def test_arbitration_llm_failure_falls_back_to_raw(self, db, closet):
        """裁定 LLM 无响应 → 退回原标签新开（匹配 corpus_tree 保守行为），不崩不丢。"""
        db.upsert_corpus_tag_norm("风险管理", "风险管理")
        doc_id = db.insert_document("a.pdf", "/tmp/a.pdf")
        fake = _route_llm({M_EXTRACT: json.dumps([{"tag": "前端开发", "confidence": 0.8}])})
        with _patch_llm(fake):
            closet.add_document(doc_id, "前端手册", "", [{"title": "x"}])
        assert [r["tag_text"] for r in _raw_rows(db, doc_id)] == ["前端开发"]
        assert db.get_corpus_tag_norm_map()["前端开发"] == "前端开发"

    def test_known_raw_tag_hits_norm_map_without_arbitration(self, db, closet):
        """已有 raw→canonical 映射 → 直接复用，不再调用裁定 LLM。"""
        db.upsert_corpus_tag_norm("风控", "风险管理")
        db.upsert_corpus_tag_norm("风险管理", "风险管理")
        doc_id = db.insert_document("a.pdf", "/tmp/a.pdf")
        fake = _route_llm({M_EXTRACT: json.dumps([{"tag": "风控", "confidence": 0.9}])})
        with _patch_llm(fake):
            closet.add_document(doc_id, "风控手册", "", [{"title": "x"}])
        assert [r["tag_text"] for r in _raw_rows(db, doc_id)] == ["风险管理"]
        assert _arb_calls(fake) == []

    def test_exact_canonical_hit_without_arbitration(self, db, closet):
        """抽取结果恰为已有规范标签 → 直接复用，无裁定调用。"""
        db.upsert_corpus_tag_norm("风险管理", "风险管理")
        doc_id = db.insert_document("a.pdf", "/tmp/a.pdf")
        fake = _route_llm({M_EXTRACT: json.dumps([{"tag": "风险管理", "confidence": 0.9}])})
        with _patch_llm(fake):
            closet.add_document(doc_id, "风控手册", "", [{"title": "x"}])
        assert [r["tag_text"] for r in _raw_rows(db, doc_id)] == ["风险管理"]
        assert _arb_calls(fake) == []
        assert db.get_corpus_tag_norm_map()["风险管理"] == "风险管理"

    def test_synonymous_raw_tags_dedupe_to_single_canonical(self, db, closet):
        """同文档两个近义原始标签并入同一规范 → 只落一行，置信度取最大。"""
        db.upsert_corpus_tag_norm("容器编排", "容器编排")
        doc_id = db.insert_document("a.pdf", "/tmp/a.pdf")
        fake = _route_llm({
            M_EXTRACT: json.dumps([
                {"tag": "容器调度", "confidence": 0.7},
                {"tag": "容器管理", "confidence": 0.9},
            ]),
            M_ARBITRATE: json.dumps({"canonical": "容器编排"}),
        })
        with _patch_llm(fake):
            closet.add_document(doc_id, "容器平台", "", [{"title": "x"}])
        rows = _raw_rows(db, doc_id)
        assert [(r["tag_text"], r["confidence"]) for r in rows] == [("容器编排", 0.9)]

    def test_corpus_tree_reuses_anchor_without_extra_arbitration(self, db, closet):
        """锚定落库后 corpus_tree 增量挂簇命中 norm_map，不再产生裁定调用。"""
        db.upsert_corpus_tag_norm("风险管理", "风险管理")
        root = db.insert_corpus_tree_node(None, "知识库", "", 0, kind="root")
        cluster = db.insert_corpus_tree_node(
            root, "风险管理", "", 1, kind="cluster", tag="风险管理")
        builder = CorpusTreeBuilder(db, model="m", cluster_min=1)
        doc_id = db.insert_document("a.pdf", "/tmp/a.pdf")
        fake = _route_llm({
            M_EXTRACT: json.dumps([{"tag": "风控", "confidence": 0.8}]),
            M_ARBITRATE: json.dumps({"canonical": "风险管理"}),
        })
        with _patch_llm(fake):
            closet.add_document(doc_id, "风控手册", "", [{"title": "x"}])
            # 锚定后落库即为规范名，供 corpus_tree 直接复用
            assert [r["tag_text"] for r in _raw_rows(db, doc_id)] == ["风险管理"]
            arb_after_add = len(_arb_calls(fake))
            assert arb_after_add == 1  # 仅 add_document 锚定一次
            builder.update_for_document(doc_id)
        # corpus_tree 挂簇复用 norm_map，不产生新的裁定调用
        assert len(_arb_calls(fake)) == arb_after_add
        mem = db.get_corpus_doc_memberships(doc_id)
        assert [node_id for node_id, _ in mem] == [cluster]

    def test_arbitration_uses_retrieve_model(self, db, tmp_path):
        """NFR4：锚定裁定调用接 retrieve_model。"""
        index = ClosetIndex(
            PageIndexDB(str(tmp_path / "rm.db")), model="m", retrieve_model="r-model")
        try:
            index.db.upsert_corpus_tag_norm("风险管理", "风险管理")
            doc_id = index.db.insert_document("a.pdf", "/tmp/a.pdf")
            fake = _route_llm({
                M_EXTRACT: json.dumps([{"tag": "风控", "confidence": 0.9}]),
                M_ARBITRATE: json.dumps({"canonical": "风险管理"}),
            })
            with _patch_llm(fake):
                index.add_document(doc_id, "风控手册", "", [{"title": "x"}])
            arb_models = [m for m, p in fake.calls if M_ARBITRATE in p]
            assert arb_models == ["r-model"]
        finally:
            index.db.close()


# ---------------------------------------------------------------------------
# 3. 幂等 + 确定性
# ---------------------------------------------------------------------------


class TestIdempotencyAndDeterminism:
    def test_reindex_same_doc_yields_identical_tag_set(self, db, closet):
        """重索引同一文档：标签集完全一致、无重复行、裁定不重跑。"""
        db.upsert_corpus_tag_norm("风险管理", "风险管理")
        doc_id = db.insert_document("a.pdf", "/tmp/a.pdf")
        fake = _route_llm({
            M_EXTRACT: json.dumps([
                {"tag": "风控", "confidence": 0.8},
                {"tag": "合规审查", "confidence": 0.7},
            ]),
            # "风控"并入既有规范；"合规审查"新开
            M_ARBITRATE: lambda prompt: json.dumps(
                {"canonical": "风险管理" if "风控" in prompt else "合规审查"}),
        })
        with _patch_llm(fake):
            closet.add_document(doc_id, "风控手册", "", [{"title": "x"}])
            first = _raw_rows(db, doc_id)
            closet.add_document(doc_id, "风控手册", "", [{"title": "x"}])
            second = _raw_rows(db, doc_id)
        assert first == second
        assert [(r["tag_text"], r["source"]) for r in second] == [
            ("风险管理", "llm"), ("合规审查", "llm")]
        # 两个新标签各裁定一次；第二轮全部命中 norm_map，无新增裁定
        assert len(_arb_calls(fake)) == 2

    def test_reindex_fallback_idempotent(self, db, closet):
        doc_id = db.insert_document("a.pdf", "/tmp/a.pdf")
        fake = _route_llm({})
        with _patch_llm(fake):
            closet.add_document(doc_id, "分布式存储实践", "", [{"title": "x"}])
            first = _raw_rows(db, doc_id)
            closet.add_document(doc_id, "分布式存储实践", "", [{"title": "x"}])
            second = _raw_rows(db, doc_id)
        assert first == second
        assert all(r["source"] == "fallback" for r in second)

    def test_extract_caps_at_five_tags(self, db, closet):
        """确定性抽取 K 上限：超出 _MAX_TAGS_PER_DOC 的标签被截断。"""
        assert closet._MAX_TAGS_PER_DOC == 5
        doc_id = db.insert_document("a.pdf", "/tmp/a.pdf")
        tags = [{"tag": f"标签{i}", "confidence": 0.9} for i in range(7)]
        fake = _route_llm({M_EXTRACT: json.dumps(tags)})
        with _patch_llm(fake):
            closet.add_document(doc_id, "多标签文档", "", [{"title": "x"}])
        rows = _raw_rows(db, doc_id)
        assert [r["tag_text"] for r in rows] == [f"标签{i}" for i in range(5)]
