import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Avoid triggering __init__.py imports that pull in heavy deps like PyPDF2.
pageindex_path = Path(__file__).parent.parent.parent / "PageIndex" / "pageindex"
sys.path.insert(0, str(pageindex_path))

import importlib.util

# Pre-seed pageindex.utils so imports won't fail
utils_spec = importlib.util.spec_from_file_location("pageindex.utils", pageindex_path / "utils.py")
utils_mod = importlib.util.module_from_spec(utils_spec)
sys.modules["pageindex.utils"] = utils_mod
utils_mod.llm_completion = lambda *a, **k: None
async def _mock_llm_acompletion(*a, **k):
    return None
utils_mod.llm_acompletion = _mock_llm_acompletion
utils_mod.count_tokens = lambda text, model=None: len(text or "") // 4
utils_mod.extract_json = lambda *a, **k: None

# Pre-seed pageindex.closet_index for _STOPWORDS
closet_spec = importlib.util.spec_from_file_location("pageindex.closet_index", pageindex_path / "closet_index.py")
closet_mod = importlib.util.module_from_spec(closet_spec)
sys.modules["pageindex.closet_index"] = closet_mod
closet_spec.loader.exec_module(closet_mod)

# Pre-seed pageindex.super_tree
super_tree_spec = importlib.util.spec_from_file_location("pageindex.super_tree", pageindex_path / "super_tree.py")
super_tree_mod = importlib.util.module_from_spec(super_tree_spec)
sys.modules["pageindex.super_tree"] = super_tree_mod
super_tree_spec.loader.exec_module(super_tree_mod)

# Pre-seed pageindex.agentic.planner
planner_spec = importlib.util.spec_from_file_location("pageindex.agentic.planner", pageindex_path / "agentic" / "planner.py")
planner_mod = importlib.util.module_from_spec(planner_spec)
sys.modules["pageindex.agentic.planner"] = planner_mod
planner_spec.loader.exec_module(planner_mod)

# Pre-seed pageindex.agentic.strategies
strategies_spec = importlib.util.spec_from_file_location("pageindex.agentic.strategies", pageindex_path / "agentic" / "strategies.py")
strategies_mod = importlib.util.module_from_spec(strategies_spec)
sys.modules["pageindex.agentic.strategies"] = strategies_mod
strategies_spec.loader.exec_module(strategies_mod)

# Pre-seed pageindex.agentic.verifier
verifier_spec = importlib.util.spec_from_file_location("pageindex.agentic.verifier", pageindex_path / "agentic" / "verifier.py")
verifier_mod = importlib.util.module_from_spec(verifier_spec)
sys.modules["pageindex.agentic.verifier"] = verifier_mod
verifier_spec.loader.exec_module(verifier_mod)

# Now load the router
router_spec = importlib.util.spec_from_file_location("pageindex.agentic.router", pageindex_path / "agentic" / "router.py")
router_mod = importlib.util.module_from_spec(router_spec)
sys.modules["pageindex.agentic.router"] = router_mod
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

        # Mock _act_tree_search to return context
        router._act_tree_search = AsyncMock(return_value=(
            "some context",           # ctx
            [{"node_id": "n1", "title": "Section 1"}],  # nodes
            1,                        # src_docs
            1,                        # cov_nodes
            {"uuid-1": [1, 2]},       # doc_pages_map
        ))

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
        assert result["matched_docs"] == [{"doc_id": "uuid-1", "score": 1.0}]
        assert result["selected_nodes"] == [{"node_id": "n1", "title": "Section 1"}]
        assert result["pages"] == [{"doc_id": "uuid-1", "pages": [1, 2]}]

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
        assert result["matched_docs"] == [{"doc_id": "uuid-1", "score": 1.0}]

    @pytest.mark.asyncio
    async def test_verifier_refuse(self, router):
        mock_st = MagicMock()
        mock_st.prefilter.return_value = {1: 1.0}
        mock_st.select_documents = AsyncMock(return_value=["uuid-1"])
        router.super_tree_index = mock_st

        router._act_tree_search = AsyncMock(return_value=(
            "some context",
            [{"node_id": "n1", "title": "Section 1"}],
            1, 1, {"uuid-1": [1]}
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
            "ctx", [{"node_id": "n1", "title": "T"}], 1, 1, {"uuid-1": [1]}
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
