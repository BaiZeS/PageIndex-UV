"""P2（[S3] 语料树简化替代后）corpus_tree 模块测试。

聚类层级构建（LLM 聚类 + 簇命名）已按 [S3]/[S10] 删除；本文件只覆盖幸存面：
1. 语料树表结构 + corpus_tag_norm 读写（迁移安全、标签锚定写入者 ClosetIndex 依赖）；
2. resolve_new_tag 单点标签裁定（共享入口，ClosetIndex._anchor_tags 复用）。

全部 LLM 调用均 mock —— 无真实 LLM、无向量。
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

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
# the real utils first, and test_delete_path.py needs the real ``get_llm_config`` —
# clobbering utils here would break its collection. corpus_tree.py binds
# llm_completion/extract_json into its own namespace and we patch
# ``corpus_tree_mod.llm_completion`` directly, so the stub is only needed to
# satisfy the relative import on standalone runs.
if "pageindex_mutil.utils" not in sys.modules:
    utils_spec = importlib.util.spec_from_file_location("pageindex_mutil.utils", pkg_path / "utils.py")
    utils_mod = importlib.util.module_from_spec(utils_spec)
    sys.modules["pageindex_mutil.utils"] = utils_mod
    utils_mod.llm_completion = lambda *a, **k: None

    def _mock_extract_json(text):
        try:
            return json.loads(text)
        except Exception:
            return None

    utils_mod.extract_json = _mock_extract_json

if "pageindex_mutil.corpus_tree" in sys.modules:
    # Full-suite run: the real module was already imported. Reuse it.
    corpus_tree_mod = sys.modules["pageindex_mutil.corpus_tree"]
else:
    spec = importlib.util.spec_from_file_location("pageindex_mutil.corpus_tree", pkg_path / "corpus_tree.py")
    corpus_tree_mod = importlib.util.module_from_spec(spec)
    sys.modules["pageindex_mutil.corpus_tree"] = corpus_tree_mod
    spec.loader.exec_module(corpus_tree_mod)

# Prompt marker used to route the mocked LLM response for the single-tag
# adjudication call (resolve_new_tag).
M_ARBITRATE = "增量单标签裁定"


@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = PageIndexDB(path)
    yield db
    db.close()
    os.unlink(path)


def _route_llm(handlers):
    """Build a fake llm_completion routing canned responses by prompt marker."""
    calls = []

    def fake(model, prompt, chat_history=None, return_finish_reason=False, thinking_disabled=True):
        calls.append((model, prompt))
        for marker, resp in handlers.items():
            if marker in prompt:
                return resp(prompt) if callable(resp) else resp
        return ""

    fake.calls = calls
    return fake


def _seed_canonical(db, tags):
    """Seed the canonical tag set (identity mapping)."""
    for t in tags:
        db.upsert_corpus_tag_norm(t, t)


# ---------------------------------------------------------------------------
# 存储层（corpus_tree 表/字段 + corpus_tag_norm 读写，迁移安全保留）
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

    def test_remap_corpus_tag_norm(self, tmp_db):
        """簇合并后规范标签改道：原指向 victim 规范的原始标签全部改指幸存规范。"""
        tmp_db.set_corpus_tag_norm_map({
            "风控": "风控管理", "风控管理": "风控管理", "风险管理": "风险管理",
        })
        tmp_db.remap_corpus_tag_norm("风控管理", "风险管理")
        m = tmp_db.get_corpus_tag_norm_map()
        assert m["风控"] == "风险管理"
        assert m["风控管理"] == "风险管理"
        assert m["风险管理"] == "风险管理"
        assert sorted(tmp_db.get_corpus_canonical_tags()) == ["风险管理"]

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
# resolve_new_tag（共享单点标签裁定；ClosetIndex._anchor_tags 复用）
# ---------------------------------------------------------------------------


class TestResolveNewTag:
    def test_returns_raw_when_no_canonical_tags(self, tmp_db):
        """无已有规范集 → 不做 LLM 调用，原样退回（新开）。"""
        fake = _route_llm({})
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            assert corpus_tree_mod.resolve_new_tag(tmp_db, "m", "新标签") == "新标签"
        assert fake.calls == []

    def test_reuses_existing_canonical_when_semantically_close(self, tmp_db):
        """LLM 裁定并入已有规范标签 → 复用该规范名。"""
        _seed_canonical(tmp_db, ["风险管理"])
        fake = _route_llm({M_ARBITRATE: json.dumps({"canonical": "风险管理"})})
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            assert corpus_tree_mod.resolve_new_tag(tmp_db, "m", "风控") == "风险管理"
        assert len(fake.calls) == 1

    def test_hallucinated_canonical_falls_back_to_raw(self, tmp_db):
        """LLM 裁定返回不在已有规范集的标签名（幻觉）→ 退回原标签新开。"""
        _seed_canonical(tmp_db, ["风险管理"])
        fake = _route_llm({M_ARBITRATE: json.dumps({"canonical": "不存在的幻觉标签"})})
        with patch.object(corpus_tree_mod, "llm_completion", side_effect=fake):
            assert corpus_tree_mod.resolve_new_tag(tmp_db, "m", "风控") == "风控"

    def test_llm_empty_response_falls_back_to_raw(self, tmp_db):
        """LLM 无响应（空回复）→ 退回原标签，不破坏确定性。"""
        _seed_canonical(tmp_db, ["风险管理"])
        with patch.object(corpus_tree_mod, "llm_completion", return_value=""):
            assert corpus_tree_mod.resolve_new_tag(tmp_db, "m", "风控") == "风控"

    def test_llm_exception_falls_back_to_raw(self, tmp_db):
        """LLM 抛异常 → 退回原标签，不崩溃。"""
        _seed_canonical(tmp_db, ["风险管理"])

        def _boom(model, prompt, **kw):
            raise RuntimeError("LLM unavailable")

        with patch.object(corpus_tree_mod, "llm_completion", side_effect=_boom):
            assert corpus_tree_mod.resolve_new_tag(tmp_db, "m", "风控") == "风控"
