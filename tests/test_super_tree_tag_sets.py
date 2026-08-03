import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from db import PageIndexDB

# Preload pageindex_mutil.utils minimal stub (same pattern as test_super_tree.py)
_mutil = Path(__file__).parent.parent / "pageindex_mutil"
sys.path.insert(0, str(_mutil))

import importlib.util

utils_spec = importlib.util.spec_from_file_location("pageindex_mutil.utils", _mutil / "utils.py")
utils_mod = importlib.util.module_from_spec(utils_spec)
sys.modules["pageindex_mutil.utils"] = utils_mod
utils_mod.llm_completion = lambda *a, **k: None
async def _mock_llm_acompletion(*a, **k):
    return None
utils_mod.llm_acompletion = _mock_llm_acompletion
utils_mod.count_tokens = lambda text, model=None: len(text or "") // 4
utils_mod.extract_json = lambda text: None
utils_mod.strip_markdown_fence = lambda text: text

closet_spec = importlib.util.spec_from_file_location("pageindex_mutil.closet_index", _mutil / "closet_index.py")
closet_mod = importlib.util.module_from_spec(closet_spec)
sys.modules["pageindex_mutil.closet_index"] = closet_mod
closet_spec.loader.exec_module(closet_mod)

spec = importlib.util.spec_from_file_location("pageindex_mutil.super_tree", _mutil / "super_tree.py")
st_mod = importlib.util.module_from_spec(spec)
sys.modules["pageindex_mutil.super_tree"] = st_mod
spec.loader.exec_module(st_mod)
SuperTreeIndex = st_mod.SuperTreeIndex


@pytest.fixture
def super_tree_index():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = PageIndexDB(path)
    client = MagicMock()
    client._uuid_to_db = {}
    client.documents = {}
    client.closet_index = None
    st = SuperTreeIndex(db, model="qwen-plus", client=client)
    yield st, db, client
    db.close()
    os.unlink(path)


class TestSelectTagSets:
    @pytest.mark.asyncio
    async def test_select_tag_sets_keeps_related_docs(self, super_tree_index):
        """阶段3 -- LLM 选出相关标签，只保留命中该标签的候选文档。"""
        st, db, client = super_tree_index
        # 两个文档，标签不同
        d1 = db.insert_document("金融风控.pdf", "/tmp/1.pdf")
        d2 = db.insert_document("医疗诊断.pdf", "/tmp/2.pdf")
        db.insert_closet_tags(d1, [(d1, "金融风控", "金融 风控", 0.9, "llm")])
        db.insert_closet_tags(d2, [(d2, "医疗诊断", "医疗 诊断", 0.9, "llm")])
        client._uuid_to_db = {"u1": d1, "u2": d2}
        client.documents = {"u1": {"id": "u1"}, "u2": {"id": "u2"}}

        candidates = {d1: 1.0, d2: 1.0}
        with patch.object(st_mod, "llm_acompletion") as mock_llm, \
             patch.object(st_mod, "extract_json", side_effect=json.loads):
            mock_llm.return_value = json.dumps({"tags": ["金融风控"]})
            result = await st._select_tag_sets("金融风险", candidates)
            # 只保留金融文档
            assert d1 in result
            assert d2 not in result

    @pytest.mark.asyncio
    async def test_select_tag_sets_empty_candidates(self, super_tree_index):
        """阶段3 -- 空候选返回空。"""
        st, db, client = super_tree_index
        result = await st._select_tag_sets("q", {})
        assert result == {}

    @pytest.mark.asyncio
    async def test_select_tag_sets_llm_failure_keeps_all(self, super_tree_index):
        """阶段3 -- LLM 失败(空响应)时保留全部候选，不误伤。"""
        st, db, client = super_tree_index
        d1 = db.insert_document("金融风控.pdf", "/tmp/1.pdf")
        db.insert_closet_tags(d1, [(d1, "金融风控", "金融 风控", 0.9, "llm")])
        client._uuid_to_db = {"u1": d1}
        client.documents = {"u1": {"id": "u1"}}
        candidates = {d1: 1.0}
        with patch.object(st_mod, "llm_acompletion") as mock_llm:
            mock_llm.return_value = None
            result = await st._select_tag_sets("q", candidates)
            assert d1 in result