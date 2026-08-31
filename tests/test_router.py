import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Avoid triggering __init__.py imports that pull in heavy deps like PyPDF2.
pageindex_path = Path(__file__).parent.parent / "pageindex_mutil"
sys.path.insert(0, str(pageindex_path))

# T32.1 真实 import（包 __init__ 已 PEP 562 惰性化，重依赖只在属性访问时加载）：
# 历史的 spec_from_file_location 预置桩机器（utils stub/closet/super_tree/planner/
# verifier/multi_hop/evidence/router 逐个 spec 加载 + evidence_module/
# _ensure_utils_configloader 两套规避手法）整体退役——那套机器的存在前提是
# `import pageindex_mutil` 触发重依赖链，现已不成立。
import pageindex_mutil.agentic.router as router_mod
import pageindex_mutil.super_tree as super_tree_mod
from pageindex_mutil.agentic import evidence as evidence_module

AgenticRouter = router_mod.AgenticRouter


def _bundle(entries):
    """构造证据束 + 空 ctx（build_evidence_bundle patch 返回值）。"""
    return (entries, {"tokens": [], "query_entities": []})


def _mount_real_engine(router):
    """T32.2：_act_tree_search/_recall_nodes_for_doc 实现已移入 SuperTreeIndex。
    给 router 挂一个引擎宿主（client/model/retrieve_model 跟随 router，与
    生产构造入参同名），绑定真实 act/recall——测试在引擎实例属性上替换单步
    即可（act 体内 self.recall_nodes_for_doc 派发到同一宿主）。"""
    import types as _types
    from pageindex_mutil.super_tree import SuperTreeIndex
    sti = _types.SimpleNamespace(
        client=router.client,
        model=router.model,
        retrieve_model=router.retrieve_model,
    )
    sti.recall_nodes_for_doc = _types.MethodType(SuperTreeIndex.recall_nodes_for_doc, sti)
    sti.act_tree_search = _types.MethodType(SuperTreeIndex.act_tree_search, sti)
    router.super_tree_index = sti
    return sti


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.documents = {}
    client.closet_index = None
    client.super_tree_index = None
    client._uuid_to_db = {}
    client.db = None
    return client


@pytest.fixture
def router(mock_client):
    return AgenticRouter(mock_client, model="qwen-plus")


class TestSearchSuperTree:
    @pytest.mark.asyncio
    async def test_empty_bundle_returns_graceful_empty(self, router):
        """L0 证据束空 → 候选集空 → 优雅空响应（替代 prefilter 空语义）。"""
        mock_st = MagicMock()
        mock_st.select_documents = AsyncMock(return_value=([], {}))
        router.super_tree_index = mock_st
        result = await router._search_super_tree("test query", top_k=3)
        assert result["answer"] == "No relevant documents found."
        assert result["confidence"] == "low"

    @pytest.mark.asyncio
    async def test_candidate_set_derived_from_bundle(self, router):
        """候选集 = bundle.keys() 派生（derive_evidence_score 作标量），不再走 prefilter。"""
        mock_st = MagicMock()
        mock_st.select_documents = AsyncMock(return_value=([], {}))
        router.super_tree_index = mock_st
        router.client.db = MagicMock()
        router.client._id_mapper = None
        router.client._uuid_to_db = {"u1": 1, "u2": 2}
        bundle = {
            1: {"channels": {"keyword": [{"token": "a"}], "tag": [], "entity": [], "vector": []}, "graph": {}},
            2: {"channels": {"keyword": [], "tag": [{"text": "t"}], "entity": [], "vector": []}, "graph": {}},
        }
        with patch.object(evidence_module, "build_evidence_bundle",
                          return_value=_bundle(bundle)):
            result = await router._search_super_tree("test query", top_k=3)
        mock_st.select_documents.assert_awaited_once_with(
            "test query", {1: 1.0, 2: 2.0}, evidence_bundle=bundle)

    @pytest.mark.asyncio
    async def test_select_documents_returns_empty(self, router):
        mock_st = MagicMock()
        mock_st.select_documents = AsyncMock(return_value=([], {}))
        router.super_tree_index = mock_st
        router.client.db = MagicMock()
        router.client._id_mapper = None
        router.client._uuid_to_db = {"u1": 1, "u2": 2}
        bundle = {
            1: {"channels": {"keyword": [{"token": "a"}, {"token": "b"}], "tag": [], "entity": [], "vector": []}, "graph": {}},
            2: {"channels": {"keyword": [], "tag": [{"text": "t"}], "entity": [], "vector": []}, "graph": {}},
        }
        with patch.object(evidence_module, "build_evidence_bundle",
                          return_value=_bundle(bundle)):
            result = await router._search_super_tree("test query", top_k=3)
        assert result["answer"] == "Super-Tree selection returned no documents."
        assert result["confidence"] == "low"
        mock_st.select_documents.assert_awaited_once_with(
            "test query", {1: 2.0, 2: 2.0}, evidence_bundle=bundle)

    @pytest.mark.asyncio
    async def test_full_super_tree_path(self, router):
        mock_st = MagicMock()
        mock_st.select_documents = AsyncMock(return_value=(["uuid-1"], {}))
        router.super_tree_index = mock_st
        router.client.db = MagicMock()
        router.client._id_mapper = None
        router.client._uuid_to_db = {"uuid-1": 1}
        bundle = {
            1: {"channels": {"keyword": [], "tag": [], "entity": [], "vector": []}, "graph": {}},
        }

        # Mock act_tree_search to return context（T6.4：回填证据派生分数；
        # T32.2：_search_super_tree 经 super_tree_index.act_tree_search 调引擎）
        async def fake_act(query, docs, node_matches=None, doc_scores_out=None,
                           evidence_bundle=None, evidence_ctx=None):
            if doc_scores_out is not None:
                # 节点召回覆盖度（evidence-derived，(0,1]）
                doc_scores_out["uuid-1"] = 1.0
            return (
                "some context",           # ctx
                [{"node_id": "n1", "title": "Section 1"}],  # nodes
                1,                        # src_docs
                1,                        # cov_nodes
                {"uuid-1": [1, 2]},       # doc_pages_map
                [{"doc_id": "uuid-1", "page": 1}],  # pages_with_text
            )

        mock_st.act_tree_search = AsyncMock(side_effect=fake_act)

        # Mock verifier
        mock_verify_result = MagicMock()
        mock_verify_result.action = "answer"
        router.verifier.verify = MagicMock(return_value=mock_verify_result)

        # Mock _load_main_funcs
        with patch.object(router, '_load_main_funcs', return_value={
            "generate_answer": lambda q, ctx: "test answer"
        }), patch.object(evidence_module, "build_evidence_bundle",
                         return_value=_bundle(bundle)):
            result = await router._search_super_tree("test query", top_k=3)

        assert result["answer"] == "test answer"
        assert result["confidence"] == "high"
        # matched_docs score = 证据分（无命中给 0），不再硬编码/覆盖度
        assert result["matched_docs"] == [{"doc_id": "uuid-1", "score": 0.0}]
        assert len(result["selected_nodes"]) == 1
        assert result["selected_nodes"][0]["node_id"] == "n1"
        assert result["selected_nodes"][0]["title"] == "Section 1"
        assert len(result["pages"]) == 1
        assert result["pages"][0]["doc_id"] == "uuid-1"

    @pytest.mark.asyncio
    async def test_act_phase_failure(self, router):
        mock_st = MagicMock()
        mock_st.select_documents = AsyncMock(return_value=(["uuid-1"], {}))
        router.super_tree_index = mock_st
        router.client.db = MagicMock()
        router.client._id_mapper = None
        router.client._uuid_to_db = {"uuid-1": 1}
        bundle = {
            1: {"channels": {"keyword": [], "tag": [], "entity": [], "vector": []}, "graph": {}},
        }

        mock_st.act_tree_search = AsyncMock(side_effect=RuntimeError("boom"))

        with patch.object(evidence_module, "build_evidence_bundle",
                          return_value=_bundle(bundle)):
            result = await router._search_super_tree("test query", top_k=3)
        assert "Failed to retrieve content" in result["answer"]
        assert result["confidence"] == "unknown"
        # T6.4 score 语义统一：Act 失败无节点级证据接地 → 不虚报匹配
        assert result["matched_docs"] == []

    @pytest.mark.asyncio
    async def test_verifier_refuse(self, router):
        mock_st = MagicMock()
        mock_st.select_documents = AsyncMock(return_value=(["uuid-1"], {}))
        router.super_tree_index = mock_st
        router.client.db = MagicMock()
        router.client._id_mapper = None
        router.client._uuid_to_db = {"uuid-1": 1}
        bundle = {
            1: {"channels": {"keyword": [], "tag": [], "entity": [], "vector": []}, "graph": {}},
        }

        mock_st.act_tree_search = AsyncMock(return_value=(
            "some context",
            [{"node_id": "n1", "title": "Section 1"}],
            1, 1, {"uuid-1": [1]},
            [{"doc_id": "uuid-1", "page": 1}],
        ))

        mock_verify_result = MagicMock()
        mock_verify_result.action = "refuse"
        router.verifier.verify = MagicMock(return_value=mock_verify_result)

        with patch.object(router, '_load_main_funcs', return_value={
            "generate_answer": lambda q, ctx: "test answer"
        }), patch.object(evidence_module, "build_evidence_bundle",
                         return_value=_bundle(bundle)):
            result = await router._search_super_tree("test query", top_k=3)

        assert result["answer"] == "I don't know."
        assert result["confidence"] == "low"


class TestSearchRouting:
    @pytest.mark.asyncio
    async def test_uses_super_tree_when_available(self, router):
        mock_st = MagicMock()
        mock_st.select_documents = AsyncMock(return_value=(["uuid-1"], {}))
        router.super_tree_index = mock_st
        router.client.db = MagicMock()
        router.client._id_mapper = None
        router.client._uuid_to_db = {"uuid-1": 1}
        bundle = {
            1: {"channels": {"keyword": [], "tag": [], "entity": [], "vector": []}, "graph": {}},
        }

        mock_st.act_tree_search = AsyncMock(return_value=(
            "ctx", [{"node_id": "n1", "title": "T"}], 1, 1, {"uuid-1": [1]},
            [{"doc_id": "uuid-1", "page": 1}],
        ))

        mock_verify_result = MagicMock()
        mock_verify_result.action = "answer"
        router.verifier.verify = MagicMock(return_value=mock_verify_result)

        with patch.object(router, '_load_main_funcs', return_value={
            "generate_answer": lambda q, ctx: "ans"
        }), patch.object(evidence_module, "build_evidence_bundle",
                         return_value=_bundle(bundle)):
            result = await router.search("test query", top_k=3)

        assert result["answer"] == "ans"

    @pytest.mark.asyncio
    async def test_super_tree_failure_returns_graceful_empty(self, router):
        """单链（[S4]/T13）：super_tree 异常 → 优雅空响应（不再回退 v2）。"""
        mock_st = MagicMock()
        mock_st.select_documents = AsyncMock(side_effect=RuntimeError("selection failed"))
        router.super_tree_index = mock_st
        router.client.db = MagicMock()
        router.client._id_mapper = None
        router.client._uuid_to_db = {"uuid-1": 1}
        bundle = {
            1: {"channels": {"keyword": [], "tag": [], "entity": [], "vector": []}, "graph": {}},
        }

        with patch.object(evidence_module, "build_evidence_bundle",
                          return_value=_bundle(bundle)):
            result = await router.search("test query", top_k=3)

        assert result["answer"] == (
            "Router not available. Initialise PageIndexClient with db_path= to enable multi-document search."
        )
        assert result["confidence"] == "unknown"
        assert result["matched_docs"] == []

    @pytest.mark.asyncio
    async def test_no_super_tree_returns_graceful_empty(self, router):
        """单链（[S4]/T13）：无 super_tree_index → 优雅空响应（不再回退 v2）。"""
        router.super_tree_index = None

        result = await router.search("test query", top_k=3)

        assert result["answer"] == (
            "Router not available. Initialise PageIndexClient with db_path= to enable multi-document search."
        )
        assert result["confidence"] == "unknown"
        assert result["matched_docs"] == []

    @pytest.mark.asyncio
    async def test_single_chain_no_multi_hop_pre_gate(self, router):
        """单链（[S4]/T13/T31.3）：search 不走 multi_hop 前置门——super_tree_index
        存在时直接 _search_super_tree；MultiHopReasoner 死实例化已删（结构性保证，
        不再有 multi_hop_reasoner 属性可被误调）。"""
        router.super_tree_index = MagicMock()  # truthy → 走 super_tree 单链
        assert not hasattr(router, "multi_hop_reasoner")  # T31.3 死实例化删除守卫
        router._search_super_tree = AsyncMock(return_value={"answer": "super"})

        result = await router.search("q", top_k=3)

        assert result["answer"] == "super"
        router._search_super_tree.assert_awaited_once_with("q", 3)


class TestActTreeSearchBudget:
    """P0: 多文档上下文 token 预算——按相关度降序，预算满即停。"""

    @pytest.mark.asyncio
    async def test_context_budget_caps_docs(self, router, monkeypatch):
        import sys
        import types

        # 其他测试文件（test_retrieve_model_wiring）运行时会把
        # sys.modules["pageindex_mutil.utils"] 换成真实模块（tiktoken count_tokens，
        # "x"*400 只算 50 token），导致预算截停失效。在当前模块对象上钉住 len//4
        # stub，使本测试与执行顺序无关。
        monkeypatch.setattr(
            sys.modules["pageindex_mutil.utils"], "count_tokens",
            lambda text, model=None: len(text or "") // 4,
        )

        # count_tokens mock 为 len//4 → 每篇上下文 400 字符 = 100 token
        # 预算设 150 → 只容得下 1 篇，第 2 篇会超 → 被预算截停
        reasoning_stub = types.ModuleType("pageindex_mutil.reasoning")
        reasoning_stub._get_max_context_tokens = lambda: 150
        # T32.1：monkeypatch.setitem 自动还原——直接赋值会残留 stub 污染后续
        # 文件（此前被防御性 purge 掩盖，purge 退役后裸奔成顺序性污染源）。
        monkeypatch.setitem(sys.modules, "pageindex_mutil.reasoning", reasoning_stub)

        monkeypatch.setattr(super_tree_mod, "_REASON_FUNCS", {
            "build_context_for_doc": lambda doc, selected, pages: "x" * 400,
            "spans_from_nodes": lambda n: [1],
        })

        async def fake_recall(query, doc_id, matched_info=None):
            return {
                "doc_id": doc_id,
                "doc": {"doc_name": doc_id, "type": "md"},
                "structure": [{"node_id": "n1"}],
                "selected": [{"node_id": "n1", "text": "x" * 400}],
                "pages": [1],
                "relevance_score": 1.0,
            }

        sti = _mount_real_engine(router)
        sti.recall_nodes_for_doc = fake_recall

        _ctx, _nodes, src_docs, _cov, _dpm, _pwt = await router._act_tree_search(
            "q", ["d1", "d2", "d3"]
        )
        # 预算 150、每篇 100 token → 仅 1 篇入上下文，其余被预算截停
        assert src_docs == 1

    @pytest.mark.asyncio
    async def test_first_doc_over_budget_admitted_then_stops(self, router, monkeypatch):
        """P0 残留修复：首篇不再绕过预算检查——超大单篇仍准入（否则无上下文），
        但准入后立即停止（不再构建/考虑后续文档）。"""
        import sys
        import types

        monkeypatch.setattr(
            sys.modules["pageindex_mutil.utils"], "count_tokens",
            lambda text, model=None: len(text or "") // 4,
        )
        reasoning_stub = types.ModuleType("pageindex_mutil.reasoning")
        reasoning_stub._get_max_context_tokens = lambda: 150
        # T32.1：monkeypatch.setitem 自动还原——直接赋值会残留 stub 污染后续
        # 文件（此前被防御性 purge 掩盖，purge 退役后裸奔成顺序性污染源）。
        monkeypatch.setitem(sys.modules, "pageindex_mutil.reasoning", reasoning_stub)

        # d1 = 800 字符 = 200 token（单篇即超预算 150）；d2 = 400 字符 = 100 token
        build_ctx = MagicMock(
            side_effect=lambda doc, selected, pages:
                "x" * 800 if doc["doc_name"] == "d1" else "x" * 400
        )
        monkeypatch.setattr(super_tree_mod, "_REASON_FUNCS", {
            "build_context_for_doc": build_ctx,
            "spans_from_nodes": lambda n: [1],
        })

        async def fake_recall(query, doc_id, matched_info=None):
            return {
                "doc_id": doc_id,
                "doc": {"doc_name": doc_id, "type": "md"},
                "structure": [{"node_id": "n1"}],
                "selected": [{"node_id": "n1", "text": "x"}],
                "pages": [1],
                "relevance_score": 1.0,
            }

        sti = _mount_real_engine(router)
        sti.recall_nodes_for_doc = fake_recall

        ctx, _nodes, src_docs, _cov, dpm, _pwt = await router._act_tree_search(
            "q", ["d1", "d2"]
        )
        assert src_docs == 1            # 超大首篇被准入
        assert "d1" in dpm and "d2" not in dpm
        # 准入首篇后立即停止：d2 的上下文根本不再构建（旧实现会先构建再截停）
        assert build_ctx.call_count == 1

    @pytest.mark.asyncio
    async def test_docs_within_budget_all_admitted(self, router, monkeypatch):
        """预算内的多篇文档全部准入（新预算逻辑不过度截停）。"""
        import sys
        import types

        monkeypatch.setattr(
            sys.modules["pageindex_mutil.utils"], "count_tokens",
            lambda text, model=None: len(text or "") // 4,
        )
        reasoning_stub = types.ModuleType("pageindex_mutil.reasoning")
        reasoning_stub._get_max_context_tokens = lambda: 250
        # T32.1：monkeypatch.setitem 自动还原——直接赋值会残留 stub 污染后续
        # 文件（此前被防御性 purge 掩盖，purge 退役后裸奔成顺序性污染源）。
        monkeypatch.setitem(sys.modules, "pageindex_mutil.reasoning", reasoning_stub)

        monkeypatch.setattr(super_tree_mod, "_REASON_FUNCS", {
            "build_context_for_doc": lambda doc, selected, pages: "x" * 400,
            "spans_from_nodes": lambda n: [1],
        })

        async def fake_recall(query, doc_id, matched_info=None):
            return {
                "doc_id": doc_id,
                "doc": {"doc_name": doc_id, "type": "md"},
                "structure": [{"node_id": "n1"}],
                "selected": [{"node_id": "n1", "text": "x"}],
                "pages": [1],
                "relevance_score": 1.0,
            }

        sti = _mount_real_engine(router)
        sti.recall_nodes_for_doc = fake_recall

        _ctx, _nodes, src_docs, _cov, _dpm, _pwt = await router._act_tree_search(
            "q", ["d1", "d2"]
        )
        assert src_docs == 2  # 每篇 100 token，共 200 <= 250，全部准入

    @pytest.mark.asyncio
    async def test_entity_context_block_counted_against_budget(self, router, monkeypatch):
        """P0 残留修复：循环后追加的实体关系上下文块计入同一预算（超余量则不追加）。"""
        import sys
        import types

        monkeypatch.setattr(
            sys.modules["pageindex_mutil.utils"], "count_tokens",
            lambda text, model=None: len(text or "") // 4,
        )
        reasoning_stub = types.ModuleType("pageindex_mutil.reasoning")
        reasoning_stub._get_max_context_tokens = lambda: 150
        # T32.1：monkeypatch.setitem 自动还原——直接赋值会残留 stub 污染后续
        # 文件（此前被防御性 purge 掩盖，purge 退役后裸奔成顺序性污染源）。
        monkeypatch.setitem(sys.modules, "pageindex_mutil.reasoning", reasoning_stub)

        # 单篇 400 字符 = 100 token → 余量仅 50 token
        monkeypatch.setattr(super_tree_mod, "_REASON_FUNCS", {
            "build_context_for_doc": lambda doc, selected, pages: "x" * 400,
            "spans_from_nodes": lambda n: [1],
        })

        async def fake_recall(query, doc_id, matched_info=None):
            return {
                "doc_id": doc_id,
                "doc": {"doc_name": doc_id, "type": "md"},
                "structure": [{"node_id": "n1"}],
                "selected": [{"node_id": "n1", "text": "x"}],
                "pages": [1],
                "relevance_score": 1.0,
            }

        sti = _mount_real_engine(router)
        sti.recall_nodes_for_doc = fake_recall

        # 实体块 ≈ 350+ 字符 ≈ 88 token > 余量 50 → 不得追加
        mock_db = MagicMock()
        mock_db.search_entities.return_value = [{"id": 1, "name": "实体甲"}]
        mock_db.get_entity_relations.return_value = [
            {"subject_name": "主体名" * 10, "predicate": "关联", "object_name": "对象名" * 10}
        ] * 5
        router.client.db = mock_db

        ctx, _nodes, src_docs, _cov, _dpm, _pwt = await router._act_tree_search(
            "q", ["d1"]
        )
        assert src_docs == 1
        assert "=== Entity:" not in ctx  # 实体块计入预算，超余量被跳过

    @pytest.mark.asyncio
    async def test_entity_context_block_admitted_when_budget_allows(self, router, monkeypatch):
        """预算充足时实体关系块照常追加（计入预算但放得下）。"""
        import sys
        import types

        monkeypatch.setattr(
            sys.modules["pageindex_mutil.utils"], "count_tokens",
            lambda text, model=None: len(text or "") // 4,
        )
        reasoning_stub = types.ModuleType("pageindex_mutil.reasoning")
        reasoning_stub._get_max_context_tokens = lambda: 2000
        # T32.1：monkeypatch.setitem 自动还原——直接赋值会残留 stub 污染后续
        # 文件（此前被防御性 purge 掩盖，purge 退役后裸奔成顺序性污染源）。
        monkeypatch.setitem(sys.modules, "pageindex_mutil.reasoning", reasoning_stub)

        monkeypatch.setattr(super_tree_mod, "_REASON_FUNCS", {
            "build_context_for_doc": lambda doc, selected, pages: "x" * 400,
            "spans_from_nodes": lambda n: [1],
        })

        async def fake_recall(query, doc_id, matched_info=None):
            return {
                "doc_id": doc_id,
                "doc": {"doc_name": doc_id, "type": "md"},
                "structure": [{"node_id": "n1"}],
                "selected": [{"node_id": "n1", "text": "x"}],
                "pages": [1],
                "relevance_score": 1.0,
            }

        sti = _mount_real_engine(router)
        sti.recall_nodes_for_doc = fake_recall

        mock_db = MagicMock()
        mock_db.search_entities.return_value = [{"id": 1, "name": "实体甲"}]
        mock_db.get_entity_relations.return_value = [
            {"subject_name": "A", "predicate": "关联", "object_name": "B"}
        ]
        router.client.db = mock_db

        ctx, _nodes, src_docs, _cov, _dpm, _pwt = await router._act_tree_search(
            "q", ["d1"]
        )
        assert src_docs == 1
        assert "=== Entity:" in ctx


class TestActTreeSearchDedup:
    """[S6] 软归属去重：同一文档经多个簇分支命中 → 召回/预算只计一次。"""

    @pytest.mark.asyncio
    async def test_candidate_docs_deduped_before_recall(self, router, monkeypatch):
        calls = []

        async def fake_recall(query, doc_id, matched_info=None):
            calls.append(doc_id)
            return {
                "doc_id": doc_id,
                "doc": {"doc_name": doc_id, "type": "md"},
                "structure": [{"node_id": "n1"}],
                "selected": [{"node_id": "n1", "text": "x"}],
                "pages": [1],
                "relevance_score": 1.0,
            }

        sti = _mount_real_engine(router)
        sti.recall_nodes_for_doc = fake_recall
        monkeypatch.setattr(super_tree_mod, "_REASON_FUNCS", {
            "build_context_for_doc": lambda doc, selected, pages: "ctx",
            "spans_from_nodes": lambda n: [1],
        })

        _ctx, _nodes, src_docs, _cov, dpm, _pwt = await router._act_tree_search(
            "q", ["d1", "d1", "d2", "d1"]
        )
        assert len(calls) == 2
        assert set(calls) == {"d1", "d2"}
        assert src_docs == 2
        assert set(dpm) == {"d1", "d2"}


class TestActTreeSearchEvidenceSort:
    """[S8] 上下文组装排序：证据分（derive_evidence_score）降序主键，平分按
    candidate_docs 裁定序（次键）——与 matched_docs 同口径。"""

    def _setup_recall(self, router, monkeypatch):
        monkeypatch.setattr(super_tree_mod, "_REASON_FUNCS", {
            "build_context_for_doc": lambda doc, selected, pages: "ctx",
            "spans_from_nodes": lambda n: [1],
        })

        async def fake_recall(query, doc_id, matched_info=None):
            return {
                "doc_id": doc_id,
                "doc": {"doc_name": doc_id, "type": "md"},
                "structure": [{"node_id": "n1"}],
                "selected": [{"node_id": "n1", "text": "x"}],
                "pages": [1],
                "relevance_score": 1.0,
            }

        sti = _mount_real_engine(router)
        sti.recall_nodes_for_doc = fake_recall

    @pytest.mark.asyncio
    async def test_sorts_by_evidence_score_desc(self, router, monkeypatch):
        """证据分不同 → 按证据分降序（覆盖度分/裁定序皆被证据分压过）。"""
        self._setup_recall(router, monkeypatch)
        router.client._id_mapper = {"docA": 101, "docB": 102}
        evidence_bundle = {
            101: {"channels": {"entity": [{"name": "甲"}], "tag": [], "keyword": []}},
            102: {"channels": {"entity": [], "tag": [], "keyword": []}},
        }
        # candidate_docs 裁定序把低分 docB 排前——证据分排序必须把它压到后面
        _ctx, _nodes, _src, _cov, dpm, _pwt = await router._act_tree_search(
            "q", ["docB", "docA"], evidence_bundle=evidence_bundle
        )
        assert list(dpm) == ["docA", "docB"]

    @pytest.mark.asyncio
    async def test_tie_broken_by_candidate_docs_order(self, router, monkeypatch):
        """证据分平分 → 保持 candidate_docs（L1 裁定序）顺序。"""
        self._setup_recall(router, monkeypatch)
        router.client._id_mapper = {"docA": 101, "docB": 102}
        evidence_bundle = {
            101: {"channels": {"entity": [], "tag": [{"text": "x"}], "keyword": []}},
            102: {"channels": {"entity": [], "tag": [{"text": "x"}], "keyword": []}},
        }
        _ctx, _nodes, _src, _cov, dpm, _pwt = await router._act_tree_search(
            "q", ["docB", "docA"], evidence_bundle=evidence_bundle
        )
        assert list(dpm) == ["docB", "docA"]
