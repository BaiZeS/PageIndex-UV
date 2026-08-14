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

import importlib.util

# Pre-seed pageindex.utils so imports won't fail
utils_spec = importlib.util.spec_from_file_location("pageindex_mutil.utils", pageindex_path / "utils.py")
utils_mod = importlib.util.module_from_spec(utils_spec)
sys.modules["pageindex_mutil.utils"] = utils_mod
utils_mod.llm_completion = lambda *a, **k: None
async def _mock_llm_acompletion(*a, **k):
    return None
utils_mod.llm_acompletion = _mock_llm_acompletion
utils_mod.count_tokens = lambda text, model=None: len(text or "") // 4
utils_mod.extract_json = lambda *a, **k: None
# Load the REAL strip_markdown_fence from utils.py source so super_tree.py's
# `from .utils import ..., strip_markdown_fence` resolves (W2 FR3). The function
# is pure, no heavy deps.
_real_utils_spec = importlib.util.spec_from_file_location(
    "_real_utils_strip_rt", pageindex_path / "utils.py"
)
_real_utils_mod = importlib.util.module_from_spec(_real_utils_spec)
_real_utils_spec.loader.exec_module(_real_utils_mod)
utils_mod.strip_markdown_fence = _real_utils_mod.strip_markdown_fence

# 补齐 page_index.py / page_index_md.py / retrieve.py / client.py 等模块从 utils
# 导入的其余符号（stub 缺 generate_summaries_for_structure 等，触发
# pageindex_mutil/__init__ 链导入时报 ImportError）。显式逐名 setattr——不遍历
# dir(_real_utils_mod)（那会把 logging/json/PyPDF2/openai 等导入的子模块一并拷进
# stub，过宽）。已 stub 的 llm_* / count_tokens / extract_json / strip_markdown_fence
# 保留不覆盖。
for _name in (
    "ConfigLoader", "JsonLogger",
    "add_node_text", "add_preface_if_needed", "configure_llm",
    "convert_page_to_int", "convert_physical_index_to_int",
    "create_clean_structure_for_description", "create_node_mapping",
    "format_structure", "generate_doc_description", "generate_node_summary",
    "generate_summaries_for_structure", "get_json_content",
    "get_number_of_pages", "get_page_tokens", "get_pdf_name",
    "post_processing", "print_json", "print_toc",
    "remove_fields", "remove_structure_text", "structure_to_list", "write_node_id",
):
    setattr(utils_mod, _name, getattr(_real_utils_mod, _name))

# Pre-seed pageindex.closet_index for _STOPWORDS
closet_spec = importlib.util.spec_from_file_location("pageindex_mutil.closet_index", pageindex_path / "closet_index.py")
closet_mod = importlib.util.module_from_spec(closet_spec)
sys.modules["pageindex_mutil.closet_index"] = closet_mod
closet_spec.loader.exec_module(closet_mod)

# Pre-seed pageindex.super_tree
super_tree_spec = importlib.util.spec_from_file_location("pageindex_mutil.super_tree", pageindex_path / "super_tree.py")
super_tree_mod = importlib.util.module_from_spec(super_tree_spec)
sys.modules["pageindex_mutil.super_tree"] = super_tree_mod
super_tree_spec.loader.exec_module(super_tree_mod)

# Pre-seed pageindex.agentic.planner
planner_spec = importlib.util.spec_from_file_location("pageindex_mutil.agentic.planner", pageindex_path / "agentic" / "planner.py")
planner_mod = importlib.util.module_from_spec(planner_spec)
sys.modules["pageindex_mutil.agentic.planner"] = planner_mod
planner_spec.loader.exec_module(planner_mod)

# Pre-seed pageindex.agentic.strategies
strategies_spec = importlib.util.spec_from_file_location("pageindex_mutil.agentic.strategies", pageindex_path / "agentic" / "strategies.py")
strategies_mod = importlib.util.module_from_spec(strategies_spec)
sys.modules["pageindex_mutil.agentic.strategies"] = strategies_mod
strategies_spec.loader.exec_module(strategies_mod)

# Pre-seed pageindex.agentic.verifier
verifier_spec = importlib.util.spec_from_file_location("pageindex_mutil.agentic.verifier", pageindex_path / "agentic" / "verifier.py")
verifier_mod = importlib.util.module_from_spec(verifier_spec)
sys.modules["pageindex_mutil.agentic.verifier"] = verifier_mod
verifier_spec.loader.exec_module(verifier_mod)

# Pre-seed pageindex.agentic.multi_hop (router imports it at module level)
multi_hop_spec = importlib.util.spec_from_file_location("pageindex_mutil.agentic.multi_hop", pageindex_path / "agentic" / "multi_hop.py")
multi_hop_mod = importlib.util.module_from_spec(multi_hop_spec)
sys.modules["pageindex_mutil.agentic.multi_hop"] = multi_hop_mod
multi_hop_spec.loader.exec_module(multi_hop_mod)

# Now load the router
router_spec = importlib.util.spec_from_file_location("pageindex_mutil.agentic.router", pageindex_path / "agentic" / "router.py")
router_mod = importlib.util.module_from_spec(router_spec)
sys.modules["pageindex_mutil.agentic.router"] = router_mod
router_spec.loader.exec_module(router_mod)
AgenticRouter = router_mod.AgenticRouter


class TestWeightedRRF:
    def test_single_strategy(self):
        results = {"metadata": [("doc1", 1), ("doc2", 2)]}
        weights = {"metadata": 1.0}
        fused = AgenticRouter._weighted_rrf(results, weights)
        assert len(fused) == 2
        assert fused[0][0] == "doc1"
        assert fused[0][1] > fused[1][1]

    def test_multiple_strategies(self):
        results = {
            "metadata": [("doc1", 1)],
            "semantics": [("doc1", 1), ("doc2", 1)],
        }
        weights = {"metadata": 1.0, "semantics": 1.5}
        fused = AgenticRouter._weighted_rrf(results, weights)
        assert len(fused) == 2
        # doc1 appears in both, so it should score higher
        assert fused[0][0] == "doc1"

    def test_empty_results(self):
        assert AgenticRouter._weighted_rrf({}, {}) == []


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


class TestContentRankConversion:
    """P0-Bug2: content 策略命中数必须转换为真实 rank 后再喂 RRF。"""

    @pytest.mark.asyncio
    async def test_content_results_are_ranks_not_hit_counts(self, router):
        router.metadata_strategy.search = MagicMock(return_value=[])

        match_a = [{"node_id": "nA", "keyword": "k", "context": "ctxA"}]
        match_b = [{"node_id": "nB", "keyword": "k", "context": "ctxB"}]
        router.content_strategy = MagicMock()
        # ContentStrategy 保证按命中数降序返回 (doc_id, hit_count, matches)
        router.content_strategy.search = MagicMock(
            return_value=[("docA", 5, match_a), ("docB", 1, match_b)]
        )

        results, node_matches = await router._run_strategies(
            "test query", [], {"content": 1.0}
        )

        # 必须是真实的 1-based rank（最优在前），而不是原始命中数
        assert results["content"] == [("docA", 1), ("docB", 2)]
        # node_matches 仍按 doc_id 携带匹配节点信息
        assert node_matches == {"docA": match_a, "docB": match_b}

    @pytest.mark.asyncio
    async def test_fusion_ranks_more_hits_higher(self, router):
        """端到端融合断言：命中更多的文档在 RRF 融合后排名更高。"""
        router.metadata_strategy.search = MagicMock(return_value=[])
        router.content_strategy = MagicMock()
        router.content_strategy.search = MagicMock(
            return_value=[
                ("docA", 8, [{"node_id": "a"}]),
                ("docB", 1, [{"node_id": "b"}]),
            ]
        )

        results, _ = await router._run_strategies(
            "test query", [], {"content": 1.0}
        )
        fused = AgenticRouter._weighted_rrf(results, {"content": 1.0})
        assert [doc_id for doc_id, _ in fused] == ["docA", "docB"]


class TestSearchSuperTree:
    @pytest.mark.asyncio
    async def test_prefilter_returns_empty(self, router):
        mock_st = MagicMock()
        mock_st.prefilter.return_value = {}
        router.super_tree_index = mock_st

        result = await router._search_super_tree("test query", top_k=3)
        assert result["answer"] == "No relevant documents found in prefilter."
        assert result["confidence"] == "low"
        mock_st.prefilter.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_select_documents_returns_empty(self, router):
        mock_st = MagicMock()
        mock_st.prefilter.return_value = {1: 2.0, 2: 1.0}
        mock_st.select_documents = AsyncMock(return_value=[])
        router.super_tree_index = mock_st

        result = await router._search_super_tree("test query", top_k=3)
        assert result["answer"] == "Super-Tree selection returned no documents."
        assert result["confidence"] == "low"
        mock_st.prefilter.assert_called_once_with("test query")
        mock_st.select_documents.assert_awaited_once_with("test query", {1: 2.0, 2: 1.0})

    @pytest.mark.asyncio
    async def test_full_super_tree_path(self, router):
        mock_st = MagicMock()
        mock_st.prefilter.return_value = {1: 1.0}
        mock_st.select_documents = AsyncMock(return_value=["uuid-1"])
        router.super_tree_index = mock_st

        # Mock planner to avoid LLM call for HyDE
        from pageindex_mutil.agentic.planner import PlanResult
        router.planner.plan = AsyncMock(return_value=PlanResult(
            queries=["test query"], weights={}, query_type="factual"
        ))

        # Mock _act_tree_search to return context（T6.4：回填证据派生分数）
        async def fake_act(query, docs, node_matches=None, doc_scores_out=None):
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

        router._act_tree_search = AsyncMock(side_effect=fake_act)

        # Mock verifier
        mock_verify_result = MagicMock()
        mock_verify_result.action = "answer"
        router.verifier.verify = MagicMock(return_value=mock_verify_result)

        # Mock _load_main_funcs
        with patch.object(router, '_load_main_funcs', return_value={
            "generate_answer": lambda q, ctx: "test answer"
        }):
            result = await router._search_super_tree("test query", top_k=3)

        assert result["answer"] == "test answer"
        assert result["confidence"] == "high"
        # matched_docs score = 召回覆盖度（evidence-derived），不再硬编码
        assert result["matched_docs"] == [{"doc_id": "uuid-1", "score": 1.0}]
        assert len(result["selected_nodes"]) == 1
        assert result["selected_nodes"][0]["node_id"] == "n1"
        assert result["selected_nodes"][0]["title"] == "Section 1"
        assert len(result["pages"]) == 1
        assert result["pages"][0]["doc_id"] == "uuid-1"

    @pytest.mark.asyncio
    async def test_act_phase_failure(self, router):
        mock_st = MagicMock()
        mock_st.prefilter.return_value = {1: 1.0}
        mock_st.select_documents = AsyncMock(return_value=["uuid-1"])
        router.super_tree_index = mock_st

        router._act_tree_search = AsyncMock(side_effect=RuntimeError("boom"))

        result = await router._search_super_tree("test query", top_k=3)
        assert "Failed to retrieve content" in result["answer"]
        assert result["confidence"] == "unknown"
        # T6.4 score 语义统一：Act 失败无节点级证据接地 → 不虚报匹配
        assert result["matched_docs"] == []

    @pytest.mark.asyncio
    async def test_verifier_refuse(self, router):
        mock_st = MagicMock()
        mock_st.prefilter.return_value = {1: 1.0}
        mock_st.select_documents = AsyncMock(return_value=["uuid-1"])
        router.super_tree_index = mock_st

        # Mock planner to avoid LLM call for HyDE
        from pageindex_mutil.agentic.planner import PlanResult
        router.planner.plan = AsyncMock(return_value=PlanResult(
            queries=["test query"], weights={}, query_type="factual"
        ))

        router._act_tree_search = AsyncMock(return_value=(
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
        }):
            result = await router._search_super_tree("test query", top_k=3)

        assert result["answer"] == "I don't know."
        assert result["confidence"] == "low"


class TestSearchRouting:
    @pytest.mark.asyncio
    async def test_uses_super_tree_when_available(self, router):
        mock_st = MagicMock()
        mock_st.prefilter.return_value = {1: 1.0}
        mock_st.select_documents = AsyncMock(return_value=["uuid-1"])
        router.super_tree_index = mock_st

        router._act_tree_search = AsyncMock(return_value=(
            "ctx", [{"node_id": "n1", "title": "T"}], 1, 1, {"uuid-1": [1]},
            [{"doc_id": "uuid-1", "page": 1}],
        ))

        mock_verify_result = MagicMock()
        mock_verify_result.action = "answer"
        router.verifier.verify = MagicMock(return_value=mock_verify_result)

        with patch.object(router, '_load_main_funcs', return_value={
            "generate_answer": lambda q, ctx: "ans"
        }):
            result = await router.search("test query", top_k=3)

        assert result["answer"] == "ans"

    @pytest.mark.asyncio
    async def test_fallback_to_v2_on_super_tree_failure(self, router):
        mock_st = MagicMock()
        mock_st.prefilter.side_effect = RuntimeError("prefilter failed")
        router.super_tree_index = mock_st

        router._search_v2 = AsyncMock(return_value={
            "query": "test query",
            "mode": "multi",
            "answer": "v2 answer",
            "confidence": "high",
            "matched_docs": [],
            "selected_nodes": [],
            "pages": [],
        })

        result = await router.search("test query", top_k=3)
        assert result["answer"] == "v2 answer"
        router._search_v2.assert_awaited_once_with("test query", 3)

    @pytest.mark.asyncio
    async def test_uses_v2_when_no_super_tree(self, router):
        router.super_tree_index = None

        router._search_v2 = AsyncMock(return_value={
            "query": "test query",
            "mode": "multi",
            "answer": "v2 answer",
            "confidence": "high",
            "matched_docs": [],
            "selected_nodes": [],
            "pages": [],
        })

        result = await router.search("test query", top_k=3)
        assert result["answer"] == "v2 answer"
        router._search_v2.assert_awaited_once_with("test query", 3)

    @pytest.mark.asyncio
    async def test_single_chain_no_multi_hop_pre_gate(self, router):
        """单链（[S4]）：search 不再走 multi_hop 前置门——super_tree_index 存在时
        直接 _search_super_tree，multi_hop_reasoner.execute 不得被调用。"""
        router.super_tree_index = MagicMock()  # truthy → 走 super_tree 单链
        router.multi_hop_reasoner.execute = AsyncMock(return_value={"answer": "hop"})
        router._search_super_tree = AsyncMock(return_value={"answer": "super"})

        result = await router.search("q", top_k=3)

        assert result["answer"] == "super"
        router.multi_hop_reasoner.execute.assert_not_awaited()
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
        sys.modules["pageindex_mutil.reasoning"] = reasoning_stub

        router._main_funcs = {
            "build_context_for_doc": lambda doc, selected, pages: "x" * 400,
            "pages_from_nodes": lambda n: [1],
        }

        async def fake_recall(query, doc_id, matched_info=None):
            return {
                "doc_id": doc_id,
                "doc": {"doc_name": doc_id, "type": "md"},
                "structure": [{"node_id": "n1"}],
                "selected": [{"node_id": "n1", "text": "x" * 400}],
                "pages": [1],
                "relevance_score": 1.0,
            }

        router._recall_nodes_for_doc = fake_recall

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
        sys.modules["pageindex_mutil.reasoning"] = reasoning_stub

        # d1 = 800 字符 = 200 token（单篇即超预算 150）；d2 = 400 字符 = 100 token
        build_ctx = MagicMock(
            side_effect=lambda doc, selected, pages:
                "x" * 800 if doc["doc_name"] == "d1" else "x" * 400
        )
        router._main_funcs = {
            "build_context_for_doc": build_ctx,
            "pages_from_nodes": lambda n: [1],
        }

        async def fake_recall(query, doc_id, matched_info=None):
            return {
                "doc_id": doc_id,
                "doc": {"doc_name": doc_id, "type": "md"},
                "structure": [{"node_id": "n1"}],
                "selected": [{"node_id": "n1", "text": "x"}],
                "pages": [1],
                "relevance_score": 1.0,
            }

        router._recall_nodes_for_doc = fake_recall

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
        sys.modules["pageindex_mutil.reasoning"] = reasoning_stub

        router._main_funcs = {
            "build_context_for_doc": lambda doc, selected, pages: "x" * 400,
            "pages_from_nodes": lambda n: [1],
        }

        async def fake_recall(query, doc_id, matched_info=None):
            return {
                "doc_id": doc_id,
                "doc": {"doc_name": doc_id, "type": "md"},
                "structure": [{"node_id": "n1"}],
                "selected": [{"node_id": "n1", "text": "x"}],
                "pages": [1],
                "relevance_score": 1.0,
            }

        router._recall_nodes_for_doc = fake_recall

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
        sys.modules["pageindex_mutil.reasoning"] = reasoning_stub

        # 单篇 400 字符 = 100 token → 余量仅 50 token
        router._main_funcs = {
            "build_context_for_doc": lambda doc, selected, pages: "x" * 400,
            "pages_from_nodes": lambda n: [1],
        }

        async def fake_recall(query, doc_id, matched_info=None):
            return {
                "doc_id": doc_id,
                "doc": {"doc_name": doc_id, "type": "md"},
                "structure": [{"node_id": "n1"}],
                "selected": [{"node_id": "n1", "text": "x"}],
                "pages": [1],
                "relevance_score": 1.0,
            }

        router._recall_nodes_for_doc = fake_recall

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
        sys.modules["pageindex_mutil.reasoning"] = reasoning_stub

        router._main_funcs = {
            "build_context_for_doc": lambda doc, selected, pages: "x" * 400,
            "pages_from_nodes": lambda n: [1],
        }

        async def fake_recall(query, doc_id, matched_info=None):
            return {
                "doc_id": doc_id,
                "doc": {"doc_name": doc_id, "type": "md"},
                "structure": [{"node_id": "n1"}],
                "selected": [{"node_id": "n1", "text": "x"}],
                "pages": [1],
                "relevance_score": 1.0,
            }

        router._recall_nodes_for_doc = fake_recall

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
    async def test_candidate_docs_deduped_before_recall(self, router):
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

        router._recall_nodes_for_doc = fake_recall
        router._main_funcs = {
            "build_context_for_doc": lambda doc, selected, pages: "ctx",
            "pages_from_nodes": lambda n: [1],
        }

        _ctx, _nodes, src_docs, _cov, dpm, _pwt = await router._act_tree_search(
            "q", ["d1", "d1", "d2", "d1"]
        )
        assert len(calls) == 2
        assert set(calls) == {"d1", "d2"}
        assert src_docs == 2
        assert set(dpm) == {"d1", "d2"}
