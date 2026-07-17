"""Tests for retrieve_model wiring (W6 FR2 / NFR4).

Verifies that the retrieval path LLM call sites use ``retrieve_model`` when set
and fall back to ``model`` when retrieve_model is None/empty (NFR4).

Covers two paths:
  - PageIndexClient path: ClosetIndex, SuperTreeIndex, RetrievalPlanner,
    DescriptionStrategy, CRAGVerifier (6 call sites).
  - main.py path: _call_llm_json, generate_answer (2 call sites).
"""
import asyncio
import importlib
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# NOTE: test_router.py (collected in the same full-suite run) pre-seeds
# sys.modules with stub pageindex_mutil.* modules whose llm_completion/
# llm_acompletion are no-op lambdas. We purge any pre-seeded pageindex_mutil
# stubs at import time so our `from pageindex_mutil... import ...` resolves the
# REAL modules. We then hold module-object references and patch via
# `patch.object(module, ...)` (NOT string paths) so the patch always targets
# the same module object the class under test references, regardless of any
# later sys.modules mutation by other test files. This is a test-isolation guard
# scoped to THIS file only.
for _mod in list(sys.modules):
    if _mod == "pageindex_mutil" or _mod.startswith("pageindex_mutil."):
        del sys.modules[_mod]

import pageindex_mutil.closet_index as closet_index_mod
import pageindex_mutil.super_tree as super_tree_mod
import pageindex_mutil.agentic.planner as planner_mod
import pageindex_mutil.agentic.strategies as strategies_mod
import pageindex_mutil.agentic.verifier as verifier_mod
from pageindex_mutil.closet_index import ClosetIndex
from pageindex_mutil.super_tree import SuperTreeIndex
from pageindex_mutil.agentic.planner import RetrievalPlanner
from pageindex_mutil.agentic.strategies import DescriptionStrategy
from pageindex_mutil.agentic.verifier import CRAGVerifier


# ---------------------------------------------------------------------------
# PageIndexClient path — 6 retrieval call sites
# ---------------------------------------------------------------------------


class TestRetrieveModelWiringPageIndexClient:
    """Each retrieval call site must pass retrieve_model (or model fallback)."""

    def test_closet_index_uses_retrieve_model_when_set(self):
        """closet_index.py:72 _extract_tags -> llm_completion(retrieve_model or model)."""
        idx = ClosetIndex(db=MagicMock(), model="m", retrieve_model="r-model")
        with patch.object(closet_index_mod, "llm_completion", return_value="[]") as mock_llm:
            idx._extract_tags("doc", "desc", ["t1"])
            assert mock_llm.call_args[0][0] == "r-model"

    def test_closet_index_falls_back_to_model_when_retrieve_model_none(self):
        """NFR4: retrieve_model=None -> llm_completion(model)."""
        idx = ClosetIndex(db=MagicMock(), model="m", retrieve_model=None)
        with patch.object(closet_index_mod, "llm_completion", return_value="[]") as mock_llm:
            idx._extract_tags("doc", "desc", ["t1"])
            assert mock_llm.call_args[0][0] == "m"

    def test_super_tree_select_documents_uses_retrieve_model(self):
        """super_tree.py:234 select_documents -> llm_acompletion(retrieve_model or model)."""
        fake_client = MagicMock()
        fake_client._uuid_to_db = {}
        st = SuperTreeIndex(db=MagicMock(), model="m", client=fake_client, retrieve_model="r-model")
        st.kb_identity = MagicMock()
        st.kb_identity.get_identity.return_value = "kb"
        with patch.object(super_tree_mod, "llm_acompletion", return_value='{"doc_ids":[]}') as mock_llm:
            asyncio.run(st.select_documents("q", {1: 1.0}))
            assert mock_llm.call_args[0][0] == "r-model"

    def test_super_tree_select_documents_falls_back_to_model(self):
        """NFR4: retrieve_model=None -> llm_acompletion(model)."""
        fake_client = MagicMock()
        fake_client._uuid_to_db = {}
        st = SuperTreeIndex(db=MagicMock(), model="m", client=fake_client, retrieve_model=None)
        st.kb_identity = MagicMock()
        st.kb_identity.get_identity.return_value = "kb"
        with patch.object(super_tree_mod, "llm_acompletion", return_value='{"doc_ids":[]}') as mock_llm:
            asyncio.run(st.select_documents("q", {1: 1.0}))
            assert mock_llm.call_args[0][0] == "m"

    def test_planner_uses_retrieve_model(self):
        """planner.py:41 plan -> llm_acompletion(retrieve_model or model)."""
        planner = RetrievalPlanner(model="m", retrieve_model="r-model")
        with patch.object(planner_mod, "llm_acompletion", return_value='{"query_type":"factual","hyde_answer":"","query_variants":[],"weights":{}}') as mock_llm:
            asyncio.run(planner.plan("q"))
            assert mock_llm.call_args[0][0] == "r-model"

    def test_planner_falls_back_to_model(self):
        """NFR4: retrieve_model=None -> llm_acompletion(model)."""
        planner = RetrievalPlanner(model="m", retrieve_model=None)
        with patch.object(planner_mod, "llm_acompletion", return_value='{"query_type":"factual","hyde_answer":"","query_variants":[],"weights":{}}') as mock_llm:
            asyncio.run(planner.plan("q"))
            assert mock_llm.call_args[0][0] == "m"

    def test_description_strategy_uses_retrieve_model(self):
        """strategies.py:87 fallback -> llm_completion(retrieve_model or model)."""
        ds = DescriptionStrategy(model="m", retrieve_model="r-model")
        ds._main_get_relevant = None  # force fallback branch
        with patch.object(strategies_mod, "llm_completion", return_value='{"doc_ids":["d1"]}') as mock_llm:
            ds.search("q", [{"doc_id": "d1", "doc_name": "n", "description": "d"}])
            assert mock_llm.call_args[0][0] == "r-model"

    def test_description_strategy_falls_back_to_model(self):
        """NFR4: retrieve_model=None -> llm_completion(model)."""
        ds = DescriptionStrategy(model="m", retrieve_model=None)
        ds._main_get_relevant = None
        with patch.object(strategies_mod, "llm_completion", return_value='{"doc_ids":["d1"]}') as mock_llm:
            ds.search("q", [{"doc_id": "d1", "doc_name": "n", "description": "d"}])
            assert mock_llm.call_args[0][0] == "m"

    def test_verifier_uses_retrieve_model(self):
        """verifier.py:78 verify -> llm_completion(retrieve_model or model)."""
        v = CRAGVerifier(model="m", retrieve_model="r-model")
        with patch.object(verifier_mod, "llm_completion", return_value='{"based_on_context":true,"sufficient":true,"confidence":0.9}') as mock_llm:
            v.verify("ans", "ctx", "q", 1, 1)
            assert mock_llm.call_args[0][0] == "r-model"

    def test_verifier_falls_back_to_model(self):
        """NFR4: retrieve_model=None -> llm_completion(model)."""
        v = CRAGVerifier(model="m", retrieve_model=None)
        with patch.object(verifier_mod, "llm_completion", return_value='{"based_on_context":true,"sufficient":true,"confidence":0.9}') as mock_llm:
            v.verify("ans", "ctx", "q", 1, 1)
            assert mock_llm.call_args[0][0] == "m"

    def test_super_tree_kb_identity_uses_retrieve_model(self):
        """super_tree.py:103 KBIdentity._generate_with_llm -> llm_completion(retrieve_model or model)."""
        # KBIdentity is constructed inside SuperTreeIndex with model + retrieve_model;
        # we verify it uses retrieve_model via the SuperTreeIndex wiring.
        fake_db = MagicMock()
        fake_db.get_kb_identity.return_value = None  # force rebuild (no cache)
        fake_db.get_all_documents.return_value = [{"id": 1, "pdf_name": "d", "doc_description": "x"}]
        fake_db.get_top_level_nodes.return_value = []
        fake_client = MagicMock()
        fake_client._uuid_to_db = {}
        st = SuperTreeIndex(db=fake_db, model="m", client=fake_client, retrieve_model="r-model")
        with patch.object(super_tree_mod, "llm_completion", return_value="summary") as mock_llm:
            st.kb_identity.get_identity()
            assert mock_llm.call_args[0][0] == "r-model"


# ---------------------------------------------------------------------------
# main.py path — 2 retrieval call sites
# ---------------------------------------------------------------------------


class TestRetrieveModelWiringMainPy:
    """_call_llm_json and generate_answer must use RETRIEVE_MODEL_NAME or MODEL_NAME."""

    def _import_main_with_env(self, monkeypatch, retrieve_model=None, model="m"):
        """Import (or reload) main with specific env, returning the module."""
        monkeypatch.setenv("MODEL_NAME", model)
        if retrieve_model is not None:
            monkeypatch.setenv("RETRIEVE_MODEL_NAME", retrieve_model)
        else:
            monkeypatch.delenv("RETRIEVE_MODEL_NAME", raising=False)
        # Clear cached main module so MODEL_NAME/RETRIEVE_MODEL_NAME re-resolve.
        mods_to_del = [k for k in list(sys.modules) if k == "main" or k.startswith("pageindex_mutil.utils")]
        for k in mods_to_del:
            del sys.modules[k]
        import pageindex_mutil.utils as utils_mod
        importlib.reload(utils_mod)
        import main as main_mod
        importlib.reload(main_mod)
        return main_mod

    def _mock_client(self, main_mod, content):
        """Stub get_llm_client so _call_llm_json/generate_answer use a mock
        client capturing the `model` kwarg. main.py resolves the shared client
        per-call via get_llm_client() (no module-level snapshot), so we patch
        that accessor rather than a captured global."""
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = content
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        main_mod.get_llm_client = lambda: mock_client
        return mock_client

    def test_call_llm_json_uses_retrieve_model_when_set(self, monkeypatch):
        """main.py:145 _call_llm_json -> model=RETRIEVE_MODEL_NAME or MODEL_NAME."""
        main_mod = self._import_main_with_env(monkeypatch, retrieve_model="r-model", model="m")
        # Mock the OpenAI client's chat.completions.create
        mock_client = self._mock_client(main_mod, '{"key": ["val"]}')
        main_mod._call_llm_json("prompt", extract_key="key")
        assert mock_client.chat.completions.create.call_args[1]["model"] == "r-model"

    def test_call_llm_json_falls_back_to_model_when_retrieve_unset(self, monkeypatch):
        """NFR4: RETRIEVE_MODEL_NAME None/empty -> model=MODEL_NAME (the `or` fallback).

        The fallback semantics are `RETRIEVE_MODEL_NAME or MODEL_NAME`. When
        retrieve_model resolves to None/empty (e.g. config.yaml retrieve_model
        is null and RETRIEVE_MODEL_NAME env unset), the call site must use
        MODEL_NAME. We test the `or` fallback directly by setting
        RETRIEVE_MODEL_NAME to None on the module (simulating the null-yaml case).
        """
        main_mod = self._import_main_with_env(monkeypatch, retrieve_model="r-model", model="m")
        # Simulate the null/empty retrieve_model case (config.yaml retrieve_model
        # could be null, or RETRIEVE_MODEL_NAME env empty).
        main_mod.RETRIEVE_MODEL_NAME = None
        mock_client = self._mock_client(main_mod, '{"key": ["val"]}')
        main_mod._call_llm_json("prompt", extract_key="key")
        assert mock_client.chat.completions.create.call_args[1]["model"] == "m"

    def test_generate_answer_uses_retrieve_model_when_set(self, monkeypatch):
        """main.py:292 generate_answer -> model=RETRIEVE_MODEL_NAME or MODEL_NAME."""
        main_mod = self._import_main_with_env(monkeypatch, retrieve_model="r-model", model="m")
        mock_client = self._mock_client(main_mod, "answer")
        main_mod.generate_answer("q", "ctx")
        assert mock_client.chat.completions.create.call_args[1]["model"] == "r-model"

    def test_generate_answer_falls_back_to_model_when_retrieve_unset(self, monkeypatch):
        """NFR4: RETRIEVE_MODEL_NAME None/empty -> model=MODEL_NAME (the `or` fallback)."""
        main_mod = self._import_main_with_env(monkeypatch, retrieve_model="r-model", model="m")
        main_mod.RETRIEVE_MODEL_NAME = None
        mock_client = self._mock_client(main_mod, "answer")
        main_mod.generate_answer("q", "ctx")
        assert mock_client.chat.completions.create.call_args[1]["model"] == "m"
