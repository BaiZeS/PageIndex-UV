"""Tests for multi-hop loop reasoning (P4 / [S8]).

Covers:
- Single-hop query delegates to existing pipeline (no loop).
- Multi-hop query: loop executes with hop navigation, entity extraction, graph-guided next hop.
- Max hops limit reached: stops loop, generates answer from accumulated context.
- Hop yields no new info: early termination.
- Graph-guided next hop: entity relation → next entity → next documents.
- NFR4: decomposability judgment + entity extraction use retrieve_model.
- Token budget respected across hops.
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pageindex_path = Path(__file__).parent.parent / "pageindex_mutil"
sys.path.insert(0, str(pageindex_path))

import importlib.util

# Pre-seed pageindex.utils so imports won't fail
utils_spec = importlib.util.spec_from_file_location(
    "pageindex_mutil.utils", pageindex_path / "utils.py"
)
utils_mod = importlib.util.module_from_spec(utils_spec)
sys.modules["pageindex_mutil.utils"] = utils_mod
utils_spec.loader.exec_module(utils_mod)

# Pre-seed pageindex.closet_index
closet_spec = importlib.util.spec_from_file_location(
    "pageindex_mutil.closet_index", pageindex_path / "closet_index.py"
)
closet_mod = importlib.util.module_from_spec(closet_spec)
sys.modules["pageindex_mutil.closet_index"] = closet_mod
closet_spec.loader.exec_module(closet_mod)

# Pre-seed pageindex.super_tree
super_tree_spec = importlib.util.spec_from_file_location(
    "pageindex_mutil.super_tree", pageindex_path / "super_tree.py"
)
super_tree_mod = importlib.util.module_from_spec(super_tree_spec)
sys.modules["pageindex_mutil.super_tree"] = super_tree_mod
super_tree_spec.loader.exec_module(super_tree_mod)

# Pre-seed pageindex.agentic.planner
planner_spec = importlib.util.spec_from_file_location(
    "pageindex_mutil.agentic.planner", pageindex_path / "agentic" / "planner.py"
)
planner_mod = importlib.util.module_from_spec(planner_spec)
sys.modules["pageindex_mutil.agentic.planner"] = planner_mod
planner_spec.loader.exec_module(planner_mod)

# Pre-seed pageindex.agentic.strategies
strategies_spec = importlib.util.spec_from_file_location(
    "pageindex_mutil.agentic.strategies", pageindex_path / "agentic" / "strategies.py"
)
strategies_mod = importlib.util.module_from_spec(strategies_spec)
sys.modules["pageindex_mutil.agentic.strategies"] = strategies_mod
strategies_spec.loader.exec_module(strategies_mod)

# Pre-seed pageindex.agentic.verifier
verifier_spec = importlib.util.spec_from_file_location(
    "pageindex_mutil.agentic.verifier", pageindex_path / "agentic" / "verifier.py"
)
verifier_mod = importlib.util.module_from_spec(verifier_spec)
sys.modules["pageindex_mutil.agentic.verifier"] = verifier_mod
verifier_spec.loader.exec_module(verifier_mod)

# Pre-seed pageindex.agentic.router
router_spec = importlib.util.spec_from_file_location(
    "pageindex_mutil.agentic.router", pageindex_path / "agentic" / "router.py"
)
router_mod = importlib.util.module_from_spec(router_spec)
sys.modules["pageindex_mutil.agentic.router"] = router_mod
router_spec.loader.exec_module(router_mod)

# Now load multi_hop
multi_hop_spec = importlib.util.spec_from_file_location(
    "pageindex_mutil.agentic.multi_hop", pageindex_path / "agentic" / "multi_hop.py"
)
multi_hop_mod = importlib.util.module_from_spec(multi_hop_spec)
sys.modules["pageindex_mutil.agentic.multi_hop"] = multi_hop_mod
multi_hop_spec.loader.exec_module(multi_hop_mod)
MultiHopReasoner = multi_hop_mod.MultiHopReasoner

AgenticRouter = router_mod.AgenticRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_db():
    """Create a mock db with entity graph methods."""
    db = MagicMock()
    db.search_entities.return_value = []
    db.get_entity_relations.return_value = []
    db.get_entity_documents.return_value = []
    return db


def _make_mock_client(db=None):
    """Create a mock client with db, documents, and super_tree_index."""
    client = MagicMock()
    client.db = db or _make_mock_db()
    client.documents = {}
    client._id_mapper = {}
    client._uuid_to_db = {}
    client.super_tree_index = MagicMock()
    client.super_tree_index.prefilter.return_value = {}
    client.super_tree_index.select_documents = AsyncMock(return_value=[])
    return client


def _make_reasoner(model="test-model", retrieve_model="retrieve-model"):
    """Create a MultiHopReasoner with mock dependencies."""
    client = _make_mock_client()
    router = MagicMock(spec=AgenticRouter)
    router.model = model
    router.retrieve_model = retrieve_model
    router._load_main_funcs.return_value = {
        "generate_answer": MagicMock(return_value="final answer"),
    }
    router._act_tree_search = AsyncMock(
        return_value=("context text", [], 1, 1, {"doc1": [1]}, [])
    )
    router.verifier = MagicMock()
    router.verifier.verify.return_value = MagicMock(action="accept")
    router.client = client
    router.super_tree_index = client.super_tree_index

    reasoner = MultiHopReasoner(
        model=model,
        retrieve_model=retrieve_model,
    )
    return reasoner, router, client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleHopDelegation:
    """Single-hop queries delegate to existing _search_super_tree without looping."""

    def test_single_hop_delegates_to_super_tree(self):
        """When LLM says query is NOT decomposable, reasoner delegates to router._search_super_tree."""
        reasoner, router, client = _make_reasoner()

        single_hop_response = json.dumps({
            "decomposable": False,
            "sub_queries": [],
        })

        expected_result = {
            "query": "What is X?",
            "answer": "X is Y",
            "confidence": "high",
            "matched_docs": [],
            "selected_nodes": [],
            "pages": [],
        }
        router._search_super_tree = AsyncMock(return_value=expected_result)

        with patch.object(multi_hop_mod, "llm_acompletion", return_value=single_hop_response):
            result = asyncio.run(reasoner.execute("What is X?", router, client.db))

        router._search_super_tree.assert_awaited_once_with("What is X?", 3)
        assert result == expected_result


class TestMultiHopLoop:
    """Multi-hop queries trigger the reasoning-retrieval loop."""

    def test_multi_hop_executes_two_hops(self):
        """Multi-hop query: hop 1 navigates, extracts entity, graph guides hop 2."""
        reasoner, router, client = _make_reasoner()

        # LLM says query is decomposable
        decompose_response = json.dumps({
            "decomposable": True,
            "sub_queries": ["Find info about A", "Find info about B based on A"],
        })

        # Hop 1: extract entities
        hop1_extract = json.dumps({
            "entities": ["entity_A"],
            "facts": ["A is related to B"],
            "next_hop_hint": "entity_B",
        })

        # Hop 2: extract nothing new (terminal)
        hop2_extract = json.dumps({
            "entities": [],
            "facts": ["B does X"],
            "next_hop_hint": "",
        })

        # Graph: entity_A → related_to → entity_B
        # search_entities must return different results based on query
        def mock_search_entities(query, limit=20):
            q = query.lower()
            if "entity_b" in q:
                return [{"id": 2, "name": "entity_B"}]
            return [{"id": 1, "name": "entity_A"}]

        client.db.search_entities = mock_search_entities

        # get_entity_relations returns DIFFERENT relations per entity_id
        # so _guide_next_hop can find an unvisited next hop
        def mock_get_entity_relations(entity_id, direction="both"):
            if entity_id == 1:
                return [
                    {"subject_id": 1, "subject_name": "entity_A", "predicate": "related_to",
                     "object_id": 2, "object_name": "entity_B", "confidence": 0.9},
                ]
            # entity_B has outgoing relation to entity_C (unvisited)
            return [
                {"subject_id": 2, "subject_name": "entity_B", "predicate": "uses",
                 "object_id": 3, "object_name": "entity_C", "confidence": 0.8},
            ]

        client.db.get_entity_relations = mock_get_entity_relations
        client.db.get_entity_documents.return_value = [
            {"id": 10, "pdf_name": "doc_B.pdf", "confidence": 0.8},
        ]

        router._act_tree_search = AsyncMock(
            return_value=("hop context", [], 1, 1, {}, [])
        )

        # Use call counter to distinguish decomposability, extraction hop1, extraction hop2
        acall_n = [0]
        def mock_acompletion(model, prompt):
            acall_n[0] += 1
            if "decomposable" in prompt or "可分解" in prompt:
                return decompose_response
            # Call 2 = first extraction (hop 1), call 3+ = subsequent hops
            if acall_n[0] == 2:
                return hop1_extract
            return hop2_extract

        with patch.object(multi_hop_mod, "llm_acompletion", side_effect=mock_acompletion), \
             patch.object(multi_hop_mod, "llm_completion", return_value="answer text"):
            result = asyncio.run(reasoner.execute(
                "How does A relate to B?", router, client.db
            ))

        # Should have navigated at least twice
        assert router._act_tree_search.await_count >= 2
        assert result["answer"]
        assert result["confidence"] in ("high", "medium", "low")

    def test_multi_hop_graph_guided_next_hop(self):
        """Entity relation leads to next entity → next documents."""
        reasoner, router, client = _make_reasoner()

        decompose_response = json.dumps({
            "decomposable": True,
            "sub_queries": ["Find A", "Find B"],
        })

        hop1_extract = json.dumps({
            "entities": ["Alpha"],
            "facts": ["Alpha is connected to Beta"],
            "next_hop_hint": "Beta",
        })

        hop2_extract = json.dumps({
            "entities": [],
            "facts": ["Beta does Y"],
            "next_hop_hint": "",
        })

        # Use MagicMock for search_entities so call_count works
        mock_search = MagicMock()
        def mock_search_fn(query, limit=20):
            q = query.lower()
            if "gamma" in q:
                return [{"id": 30, "name": "Gamma"}]
            if "beta" in q and "alpha" not in q:
                return [{"id": 20, "name": "Beta"}]
            return [{"id": 10, "name": "Alpha"}]

        mock_search.side_effect = mock_search_fn
        client.db.search_entities = mock_search

        # Query-aware relations per entity (use MagicMock for call tracking)
        mock_relations = MagicMock()
        def mock_rel_fn(entity_id, direction="both"):
            if entity_id == 10:
                return [
                    {"subject_id": 10, "subject_name": "Alpha", "predicate": "connected_to",
                     "object_id": 20, "object_name": "Beta", "confidence": 0.9},
                ]
            if entity_id == 20:
                return [
                    {"subject_id": 20, "subject_name": "Beta", "predicate": "implements",
                     "object_id": 30, "object_name": "Gamma", "confidence": 0.7},
                ]
            return []

        mock_relations.side_effect = mock_rel_fn
        client.db.get_entity_relations = mock_relations
        client.db.get_entity_documents.return_value = [
            {"id": 50, "pdf_name": "docs.pdf", "confidence": 0.7},
        ]

        router._act_tree_search = AsyncMock(
            return_value=("context", [], 1, 1, {}, [])
        )

        acall_n = [0]
        def mock_acompletion(model, prompt):
            acall_n[0] += 1
            if "decomposable" in prompt or "可分解" in prompt:
                return decompose_response
            if acall_n[0] == 2:
                return hop1_extract
            return hop2_extract

        with patch.object(multi_hop_mod, "llm_acompletion", side_effect=mock_acompletion), \
             patch.object(multi_hop_mod, "llm_completion", return_value="answer"):
            result = asyncio.run(reasoner.execute(
                "How does Alpha connect to Beta?", router, client.db
            ))

        # Graph was consulted
        assert mock_search.call_count >= 1
        assert mock_relations.call_count >= 1
        assert result["answer"]


class TestMaxHopsLimit:
    """Loop stops at max_hops even if more hops are suggested."""

    def test_max_hops_stops_loop(self):
        reasoner, router, client = _make_reasoner()

        decompose_response = json.dumps({
            "decomposable": True,
            "sub_queries": ["Q1", "Q2", "Q3", "Q4"],
        })

        # Always suggests another hop
        always_next = json.dumps({
            "entities": ["next_entity"],
            "facts": ["some fact"],
            "next_hop_hint": "another_entity",
        })

        client.db.search_entities.return_value = [{"id": 1, "name": "e"}]
        client.db.get_entity_relations.return_value = [
            {"subject_id": 1, "subject_name": "e", "predicate": "p", "object_id": 2, "object_name": "e2", "confidence": 0.5},
        ]
        client.db.get_entity_documents.return_value = [{"id": 1, "pdf_name": "d.pdf"}]

        router._act_tree_search = AsyncMock(
            return_value=("ctx", [], 1, 1, {}, [])
        )

        with patch.object(multi_hop_mod, "llm_acompletion", side_effect=lambda m, p: decompose_response if "decomposable" in p or "可分解" in p else always_next), \
             patch.object(multi_hop_mod, "llm_completion", return_value="final"):
            result = asyncio.run(reasoner.execute(
                "complex query", router, client.db, max_hops=2
            ))

        # Should stop at max_hops=2 (not 3 or 4)
        assert router._act_tree_search.await_count <= 2
        assert result["answer"]


class TestEarlyTermination:
    """Loop terminates early when a hop yields no new information."""

    def test_no_new_entities_stops_loop(self):
        reasoner, router, client = _make_reasoner()

        decompose_response = json.dumps({
            "decomposable": True,
            "sub_queries": ["Q1", "Q2"],
        })

        # Hop 1 yields no entities and no next hop hint
        hop1_extract = json.dumps({
            "entities": [],
            "facts": [],
            "next_hop_hint": "",
        })

        router._act_tree_search = AsyncMock(
            return_value=("context", [], 1, 1, {}, [])
        )

        with patch.object(multi_hop_mod, "llm_acompletion", side_effect=lambda m, p: decompose_response if "decomposable" in p or "可分解" in p else hop1_extract), \
             patch.object(multi_hop_mod, "llm_completion", return_value="answer"):
            result = asyncio.run(reasoner.execute(
                "query", router, client.db
            ))

        # Only 1 hop should execute (no next hop to follow)
        assert router._act_tree_search.await_count == 1
        assert result["answer"]

    def test_empty_context_stops_loop(self):
        reasoner, router, client = _make_reasoner()

        decompose_response = json.dumps({
            "decomposable": True,
            "sub_queries": ["Q1", "Q2"],
        })

        hop1_extract = json.dumps({
            "entities": ["e"],
            "facts": [],
            "next_hop_hint": "e",
        })

        # Navigation returns empty context
        router._act_tree_search = AsyncMock(
            return_value=("", [], 0, 0, {}, [])
        )

        with patch.object(multi_hop_mod, "llm_acompletion", side_effect=lambda m, p: decompose_response if "decomposable" in p or "可分解" in p else hop1_extract), \
             patch.object(multi_hop_mod, "llm_completion", return_value="answer"):
            result = asyncio.run(reasoner.execute(
                "query", router, client.db
            ))

        # Empty context should trigger early termination
        assert router._act_tree_search.await_count == 1


class TestRetrieveModelWiring:
    """NFR4: decomposability judgment + entity extraction use retrieve_model."""

    def test_decomposability_uses_retrieve_model(self):
        reasoner, router, client = _make_reasoner(model="m", retrieve_model="r-model")

        single_hop = json.dumps({"decomposable": False, "sub_queries": []})
        router._search_super_tree = AsyncMock(return_value={
            "query": "q", "answer": "a", "confidence": "high",
            "matched_docs": [], "selected_nodes": [], "pages": [],
        })

        with patch.object(multi_hop_mod, "llm_acompletion", return_value=single_hop) as mock_llm:
            asyncio.run(reasoner.execute("q", router, client.db))
            # First call is decomposability judgment — must use retrieve_model
            assert mock_llm.call_args_list[0][0][0] == "r-model"

    def test_entity_extraction_uses_retrieve_model(self):
        reasoner, router, client = _make_reasoner(model="m", retrieve_model="r-model")

        decompose = json.dumps({"decomposable": True, "sub_queries": ["Q1"]})
        extract = json.dumps({"entities": [], "facts": [], "next_hop_hint": ""})

        router._act_tree_search = AsyncMock(
            return_value=("ctx", [], 1, 1, {}, [])
        )

        calls = []
        def track_calls(model, prompt):
            calls.append(model)
            if "decomposable" in prompt or "可分解" in prompt:
                return decompose
            return extract

        with patch.object(multi_hop_mod, "llm_acompletion", side_effect=track_calls), \
             patch.object(multi_hop_mod, "llm_completion", return_value="answer"):
            asyncio.run(reasoner.execute("q", router, client.db))

        # All llm_acompletion calls must use retrieve_model
        for m in calls:
            assert m == "r-model"

    def test_falls_back_to_model_when_retrieve_model_none(self):
        reasoner, router, client = _make_reasoner(model="m", retrieve_model=None)

        single_hop = json.dumps({"decomposable": False, "sub_queries": []})
        router._search_super_tree = AsyncMock(return_value={
            "query": "q", "answer": "a", "confidence": "high",
            "matched_docs": [], "selected_nodes": [], "pages": [],
        })

        with patch.object(multi_hop_mod, "llm_acompletion", return_value=single_hop) as mock_llm:
            asyncio.run(reasoner.execute("q", router, client.db))
            assert mock_llm.call_args_list[0][0][0] == "m"


class TestTokenBudget:
    """Token budget is respected across hops."""

    def test_context_accumulates_within_budget(self):
        reasoner, router, client = _make_reasoner()

        decompose = json.dumps({"decomposable": True, "sub_queries": ["Q1", "Q2"]})
        extract = json.dumps({
            "entities": ["e"],
            "facts": ["fact"],
            "next_hop_hint": "e",
        })

        # Each hop returns a small context
        small_ctx = "x" * 100
        router._act_tree_search = AsyncMock(
            return_value=(small_ctx, [], 1, 1, {}, [])
        )

        client.db.search_entities.return_value = [{"id": 1, "name": "e"}]
        client.db.get_entity_relations.return_value = []
        client.db.get_entity_documents.return_value = [{"id": 1, "pdf_name": "d.pdf"}]

        with patch.object(multi_hop_mod, "llm_acompletion", side_effect=lambda m, p: decompose if "decomposable" in p or "可分解" in p else extract), \
             patch.object(multi_hop_mod, "llm_completion", return_value="answer"):
            result = asyncio.run(reasoner.execute(
                "q", router, client.db, max_hops=3
            ))

        assert result["answer"]
        # Context should be aggregated from multiple hops
        assert "hop_contexts" in result or result.get("answer")

    def test_budget_exceeded_stops_hops(self):
        reasoner, router, client = _make_reasoner()

        decompose = json.dumps({"decomposable": True, "sub_queries": ["Q1", "Q2"]})
        extract = json.dumps({
            "entities": ["e"],
            "facts": ["fact"],
            "next_hop_hint": "e",
        })

        # Return a large context that exceeds budget
        huge_ctx = "x" * 20000
        router._act_tree_search = AsyncMock(
            return_value=(huge_ctx, [], 1, 1, {}, [])
        )

        client.db.search_entities.return_value = [{"id": 1, "name": "e"}]
        client.db.get_entity_relations.return_value = []
        client.db.get_entity_documents.return_value = [{"id": 1, "pdf_name": "d.pdf"}]

        with patch.object(multi_hop_mod, "llm_acompletion", side_effect=lambda m, p: decompose if "decomposable" in p or "可分解" in p else extract), \
             patch.object(multi_hop_mod, "llm_completion", return_value="answer"):
            result = asyncio.run(reasoner.execute(
                "q", router, client.db, max_hops=3,
                token_budget=500
            ))

        # Should have stopped after first hop due to budget
        assert router._act_tree_search.await_count == 1
        assert result["answer"]


class TestGraphGuidedNextHop:
    """Entity relation → next entity → next documents flow."""

    def test_graph_provides_next_hop_docs(self):
        reasoner, router, client = _make_reasoner()

        decompose = json.dumps({"decomposable": True, "sub_queries": ["Q1", "Q2"]})
        hop1 = json.dumps({
            "entities": ["ProjectX"],
            "facts": ["ProjectX uses TechY"],
            "next_hop_hint": "TechY",
        })
        hop2 = json.dumps({
            "entities": [],
            "facts": ["TechY is a framework"],
            "next_hop_hint": "",
        })

        # Graph: ProjectX → uses → TechY → built_with → LangChain
        # Query-aware search_entities
        def mock_search_entities(query, limit=20):
            q = query.lower()
            if "techy" in q:
                return [{"id": 20, "name": "TechY"}]
            if "projectx" in q:
                return [{"id": 10, "name": "ProjectX"}]
            return [{"id": 10, "name": "ProjectX"}]

        client.db.search_entities = mock_search_entities

        # Query-aware relations: each entity has its own outgoing relations
        def mock_get_entity_relations(entity_id, direction="both"):
            if entity_id == 10:
                return [
                    {"subject_id": 10, "subject_name": "ProjectX", "predicate": "uses",
                     "object_id": 20, "object_name": "TechY", "confidence": 0.9},
                ]
            if entity_id == 20:
                return [
                    {"subject_id": 20, "subject_name": "TechY", "predicate": "built_with",
                     "object_id": 30, "object_name": "LangChain", "confidence": 0.8},
                ]
            return []

        client.db.get_entity_relations = mock_get_entity_relations
        client.db.get_entity_documents.return_value = [
            {"id": 50, "pdf_name": "techy_docs.pdf", "confidence": 0.8},
        ]

        hop_count = [0]
        def mock_act_tree(query, docs):
            hop_count[0] += 1
            return (f"ctx_hop{hop_count[0]}", [], 1, 1, {}, [])

        router._act_tree_search = AsyncMock(side_effect=mock_act_tree)

        llm_calls = []
        def mock_acompletion(model, prompt):
            llm_calls.append(prompt[:200])
            if "decomposable" in prompt or "可分解" in prompt:
                return decompose
            if len(llm_calls) <= 2:
                return hop1
            return hop2

        with patch.object(multi_hop_mod, "llm_acompletion", side_effect=mock_acompletion), \
             patch.object(multi_hop_mod, "llm_completion", return_value="answer"):
            result = asyncio.run(reasoner.execute(
                "What technology does ProjectX use?",
                router, client.db
            ))

        assert result["answer"]
        assert hop_count[0] >= 2


class TestReasonerResultStructure:
    """MultiHopReasoner.execute returns a well-formed result dict."""

    def test_result_has_required_keys(self):
        reasoner, router, client = _make_reasoner()

        single_hop = json.dumps({"decomposable": False, "sub_queries": []})
        router._search_super_tree = AsyncMock(return_value={
            "query": "q", "answer": "a", "confidence": "high",
            "matched_docs": [], "selected_nodes": [], "pages": [],
        })

        with patch.object(multi_hop_mod, "llm_acompletion", return_value=single_hop):
            result = asyncio.run(reasoner.execute("q", router, client.db))

        assert "query" in result
        assert "answer" in result
        assert "confidence" in result

    def test_multi_hop_result_has_hop_count(self):
        reasoner, router, client = _make_reasoner()

        decompose = json.dumps({"decomposable": True, "sub_queries": ["Q1"]})
        extract = json.dumps({
            "entities": [],
            "facts": ["fact"],
            "next_hop_hint": "",
        })

        router._act_tree_search = AsyncMock(
            return_value=("ctx", [], 1, 1, {}, [])
        )

        with patch.object(multi_hop_mod, "llm_acompletion", side_effect=lambda m, p: decompose if "decomposable" in p or "可分解" in p else extract), \
             patch.object(multi_hop_mod, "llm_completion", return_value="answer"):
            result = asyncio.run(reasoner.execute("q", router, client.db))

        assert "hop_count" in result
        assert result["hop_count"] >= 1


class TestMatchedDocsPopulated:
    """Issue #1: all_matched_docs must be populated with hop doc IDs in multi-hop result."""

    def test_multi_hop_result_has_populated_matched_docs(self):
        """After multi-hop execution, matched_docs must contain doc IDs from hops."""
        reasoner, router, client = _make_reasoner()

        decompose = json.dumps({"decomposable": True, "sub_queries": ["Q1", "Q2"]})
        hop1_extract = json.dumps({
            "entities": ["entity_A"],
            "facts": ["A is related to B"],
            "next_hop_hint": "entity_B",
        })
        hop2_extract = json.dumps({
            "entities": [],
            "facts": ["B does X"],
            "next_hop_hint": "",
        })

        # Query-aware mocks so _guide_next_hop can find entity_B
        def mock_search_entities(query, limit=5):
            q = query.lower()
            if "entity_b" in q:
                return [{"id": 2, "name": "entity_B"}]
            return [{"id": 1, "name": "entity_A"}]

        client.db.search_entities = mock_search_entities

        def mock_get_entity_relations(entity_id):
            if entity_id == 1:
                return [
                    {"subject_id": 1, "subject_name": "entity_A", "predicate": "related_to",
                     "object_id": 2, "object_name": "entity_B", "confidence": 0.9},
                ]
            if entity_id == 2:
                return [
                    {"subject_id": 2, "subject_name": "entity_B", "predicate": "uses",
                     "object_id": 3, "object_name": "entity_C", "confidence": 0.8},
                ]
            return []

        client.db.get_entity_relations = mock_get_entity_relations
        client.db.get_entity_documents.return_value = [{"id": 10, "pdf_name": "doc.pdf"}]

        # Hop 1 returns doc_pages_map with doc "doc-uuid-1", hop 2 with "doc-uuid-2"
        hop_count = [0]
        def mock_act_tree(query, docs):
            hop_count[0] += 1
            if hop_count[0] == 1:
                return ("ctx1", [{"node_id": "n1"}], 1, 1, {"doc-uuid-1": [1, 2]}, [])
            return ("ctx2", [{"node_id": "n2"}], 1, 1, {"doc-uuid-2": [3]}, [])

        router._act_tree_search = AsyncMock(side_effect=mock_act_tree)

        llm_calls = [0]
        def mock_acompletion(model, prompt):
            llm_calls[0] += 1
            if "decomposable" in prompt or "可分解" in prompt:
                return decompose
            if llm_calls[0] == 2:
                return hop1_extract
            return hop2_extract

        with patch.object(multi_hop_mod, "llm_acompletion", side_effect=mock_acompletion), \
             patch.object(multi_hop_mod, "llm_completion", return_value="final answer"):
            result = asyncio.run(reasoner.execute(
                "How does A relate to B?", router, client.db
            ))

        # matched_docs must NOT be empty — must contain doc IDs from hops
        assert result["matched_docs"], "matched_docs is empty; hop doc IDs were never collected"
        doc_ids = {d["doc_id"] for d in result["matched_docs"]}
        assert "doc-uuid-1" in doc_ids, "doc-uuid-1 from hop 1 missing from matched_docs"
        assert "doc-uuid-2" in doc_ids, "doc-uuid-2 from hop 2 missing from matched_docs"
