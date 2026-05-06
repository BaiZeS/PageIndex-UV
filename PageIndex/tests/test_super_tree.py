import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from db import PageIndexDB

# Avoid triggering __init__.py imports that pull in heavy deps like PyPDF2.
super_tree_path = Path(__file__).parent.parent.parent / "PageIndex" / "pageindex"
sys.path.insert(0, str(super_tree_path))

# Make relative import in super_tree.py work by creating a fake package context.
import importlib.util

# Pre-seed pageindex.utils so closet_index.py won't fail on its own relative import.
utils_spec = importlib.util.spec_from_file_location("pageindex.utils", super_tree_path / "utils.py")
utils_mod = importlib.util.module_from_spec(utils_spec)
sys.modules["pageindex.utils"] = utils_mod
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

# Also need pageindex.closet_index for the _STOPWORDS import.
closet_spec = importlib.util.spec_from_file_location("pageindex.closet_index", super_tree_path / "closet_index.py")
closet_mod = importlib.util.module_from_spec(closet_spec)
sys.modules["pageindex.closet_index"] = closet_mod
closet_spec.loader.exec_module(closet_mod)

spec = importlib.util.spec_from_file_location("pageindex.super_tree", super_tree_path / "super_tree.py")
super_tree_mod = importlib.util.module_from_spec(spec)
sys.modules["pageindex.super_tree"] = super_tree_mod
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

    def test_build_super_tree_empty(self, super_tree_index):
        st, db, client = super_tree_index
        tree = st._build_super_tree([])
        assert tree == {"documents": []}

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
    async def test_select_documents(self, super_tree_index):
        with patch.object(super_tree_mod, "llm_acompletion") as mock_llm:
            mock_llm.return_value = '{"doc_ids": ["uuid-1"]}'
            st, db, client = super_tree_index
            client.documents = {"uuid-1": {"id": "uuid-1", "doc_name": "test.pdf"}}
            client._uuid_to_db = {"uuid-1": 1}
            db.insert_document("test.pdf", "/tmp/test.pdf")

            result = await st.select_documents("test", {1: 1.0})
            assert "uuid-1" in result
