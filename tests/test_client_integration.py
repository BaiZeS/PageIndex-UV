import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from db import PageIndexDB


class TestPageIndexClientSuperTree:
    def test_super_tree_index_initialized_with_db(self):
        """PageIndexClient with db_path should initialize super_tree_index."""
        # We need to mock PyPDF2 since client.py imports it at top level
        sys.modules["PyPDF2"] = MagicMock()

        from pageindex_mutil.client import PageIndexClient

        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            client = PageIndexClient(db_path=db_path)
            assert hasattr(client, "super_tree_index")
            assert client.super_tree_index is not None
            assert client.router is not None
            assert client.router.super_tree_index is client.super_tree_index
        finally:
            client.close()
            os.unlink(db_path)

    def test_super_tree_index_none_without_db(self):
        """PageIndexClient without db_path should not have super_tree_index."""
        sys.modules["PyPDF2"] = MagicMock()

        from pageindex_mutil.client import PageIndexClient

        client = PageIndexClient()
        assert hasattr(client, "super_tree_index")
        assert client.super_tree_index is None
        assert client.router is None

    def test_on_document_added_called_during_index(self):
        """index() should call super_tree_index.on_document_added after DB insert."""
        sys.modules["PyPDF2"] = MagicMock()

        from pageindex_mutil.client import PageIndexClient

        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            client = PageIndexClient(db_path=db_path)

            # Mock super_tree_index.on_document_added
            client.super_tree_index.on_document_added = MagicMock()

            # Create a temp markdown file to index
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
                f.write("# Test Document\n\nThis is a test.\n")
                md_path = f.name

            try:
                # Mock md_to_tree to avoid async complexity and LLM calls
                mock_structure = [
                    {
                        "node_id": "n1",
                        "title": "Test Document",
                        "text": "This is a test.",
                        "summary": "A test doc",
                        "level": 1,
                    }
                ]
                with patch("pageindex_mutil.client.md_to_tree") as mock_md:
                    mock_md.return_value = {
                        "doc_name": "test.md",
                        "doc_description": "A test markdown file",
                        "line_count": 3,
                        "structure": mock_structure,
                    }
                    doc_id = client.index(md_path, mode="md")

                # Verify on_document_added was called
                assert client.super_tree_index.on_document_added.called
                # It should be called with the db_doc_id (which is 1 for first insert)
                call_args = client.super_tree_index.on_document_added.call_args
                assert call_args[0][0] == 1  # First document gets db_id = 1
            finally:
                os.unlink(md_path)
        finally:
            client.close()
            os.unlink(db_path)


@pytest.mark.asyncio
async def test_single_doc_goes_through_unified_chain(tmp_path):
    """单文档不再走 _search_single 分支——统一链（router.search → super_tree）接管。

    [S4] 单链：候选=1 不再短路 _search_single；client.search 一律经 router。
    _search_single 若被调用会记录 doc_id（应保持为空）。
    """
    import types

    sys.modules["PyPDF2"] = MagicMock()

    from pageindex_mutil.client import PageIndexClient
    from pageindex_mutil.agentic.router import AgenticRouter

    client = PageIndexClient(db_path=str(tmp_path / "t.db"), search_backend="keyword")
    try:
        client.documents = {
            "d1": {
                "doc_name": "单文档", "doc_description": "", "type": "md",
                "structure": [{
                    "node_id": "0001", "title": "t", "summary": "s",
                    "text": "浴血内容", "span_kind": "line",
                    "line_num": 1, "end_line": 2, "nodes": [],
                }],
            }
        }

        calls = []

        async def fake_single(self, q, doc_id):
            calls.append(doc_id)
            return {"mode": "single"}

        client._search_single = types.MethodType(fake_single, client)

        router = AgenticRouter(client, "m")
        router.super_tree_index = MagicMock()  # truthy → 单链走 _search_super_tree

        async def fake_unified(q, top_k):
            calls.append(("unified", q))
            return {"mode": "multi"}

        router._search_super_tree = fake_unified
        client.router = router

        await client.search("浴血")

        # 统一链（super_tree）被调用；_search_single 未被调用
        assert calls == [("unified", "浴血")]
    finally:
        client.close()


# ---------------------------------------------------------------------------
# T8 审查收尾：单文档词面命中 / 未命中契约（真实 router 链 + 真实 super_tree_index）
# ---------------------------------------------------------------------------


def _select_json(selected, pool_concern=False, concern_reason=""):
    """enhance_and_select 的 LLM 响应 JSON（与 test_search_single_enhanced 同构）。"""
    return json.dumps({
        "selected_ids": selected,
        "pool_concern": pool_concern,
        "concern_reason": concern_reason,
    })


def _single_doc_client(tmp_path, doc_name, doc_description, index_body=None, pages=None):
    """真实 PageIndexClient(db) + 单文档入 DB/documents/关键词索引。

    index_body: 传入则作为 doc_keywords 正文索引（on_document_added content=...）；
    None 时 doc_keywords 仅收录 doc_name/description（nodes 表无行，故不含节点标题）。
    """
    sys.modules["PyPDF2"] = MagicMock()
    from pageindex_mutil.client import PageIndexClient

    client = PageIndexClient(db_path=str(tmp_path / "t.db"), search_backend="keyword")
    # 关闭 vector/keyword search_backend（prefilter 通道 C）：其 BM25 结果与
    # 通道 B（KeywordIndex）同源，且 KeywordSearchBackend 持类级缓存（key 仅
    # query+top_k，跨测试 DB 会串味）——单文档 prefilter 契约测试只依赖通道 A/B/D。
    client.search_backend = None
    db_doc_id = client.db.insert_document(
        doc_name, f"/tmp/{doc_name}", doc_description=doc_description,
    )
    client._id_mapper.register("d1", db_doc_id)
    client.documents["d1"] = {
        "doc_name": doc_name,
        "type": "md",
        "structure": [{
            "node_id": "n1", "title": "浴血值获取", "summary": "浴血值玩法",
            "text": "浴血值可以通过完成日常任务获得。",
            "span_kind": "line", "line_num": 1, "end_line": 2,
            "keywords": ["浴血值"],
        }],
        "pages": pages or [],
    }
    client.super_tree_index.on_document_added(db_doc_id, content=index_body)
    return client, db_doc_id


@pytest.mark.asyncio
async def test_single_doc_lexical_hit_end_to_end(tmp_path):
    """(T8 审查) 单文档词面命中端到端：真实 router 链（super_tree_index 可用、
    查询词命中 doc_keywords）+ mock LLM → client.search 返回非空 answer 且键完整。"""
    from pageindex_mutil.agentic.planner import PlanResult
    import pageindex_mutil.agentic.enhance as enhance
    import pageindex_mutil.reasoning as reasoning

    client, _ = _single_doc_client(
        tmp_path, "浴血值获取指南.md", "浴血值玩法说明",
        index_body="浴血值可以通过完成日常任务获得",
    )
    try:
        router = client.router
        # 规划 / 校验 mock（LLM 客户端未初始化时不依赖真实客户端状态，确定性）
        router.planner.plan = AsyncMock(return_value=PlanResult(
            queries=["浴血值怎么获得"], weights={}, query_type="factual",
        ))
        router.verifier.verify = MagicMock(return_value=MagicMock(action="answer"))

        captured = []

        def fake_llm(model, prompt, **kwargs):
            captured.append(prompt)
            # 按 prompt 子串路由：只有看到 浴血 证据才选 n1
            return _select_json(["n1"]) if "浴血" in prompt else _select_json([])

        with patch.object(enhance, "llm_completion", side_effect=fake_llm), \
                patch.object(reasoning, "generate_answer",
                             MagicMock(return_value="最终答案")):
            result = await client.search("浴血值怎么获得")

        assert set(result.keys()) == {
            "query", "mode", "answer", "confidence",
            "matched_docs", "selected_nodes", "pages",
        }
        assert result["answer"] == "最终答案"
        assert result["confidence"] == "high"
        # matched_docs score = 证据分（词面命中加权 > 0），非硬编码/覆盖度
        assert [d["doc_id"] for d in result["matched_docs"]] == ["d1"]
        assert all(d["score"] > 0 for d in result["matched_docs"])
        assert [n["node_id"] for n in result["selected_nodes"]] == ["n1"]
    finally:
        client.close()


@pytest.mark.asyncio
async def test_single_doc_lexical_miss_returns_prefilter_empty(tmp_path):
    """(T8 审查/T13) 单文档词面未命中契约 [S4]：查询词不命中任何 doc_keywords →
    统一链 L0 prefilter 空 → 优雅空响应（confidence low、matched_docs 空、不抛异常）。
    （旧 _content_fallback 正文 BM25 兜底已随 T13 删除。）"""
    client, _ = _single_doc_client(
        tmp_path, "操作手册.md", "",
        index_body=None,  # doc_keywords 只收 doc_name/description，不含正文
        pages=[{"page": 1, "content": "浴血值可以通过完成日常任务获得。"}],
    )
    try:
        # 确定性：规划 mock（.env 配了真实 LLM 客户端，未 mock 会真调 API）
        from pageindex_mutil.agentic.planner import PlanResult
        client.router.planner.plan = AsyncMock(return_value=PlanResult(
            queries=["浴血值怎么获得"], weights={}, query_type="factual",
        ))

        result = await client.search("浴血值怎么获得")

        assert set(result.keys()) == {
            "query", "mode", "answer", "confidence",
            "matched_docs", "selected_nodes", "pages",
        }
        # [S4]/T13 单链契约：prefilter 空 → 优雅空响应（不再走 content_fallback）
        assert result["confidence"] == "low"
        assert result["matched_docs"] == []
        assert result["selected_nodes"] == []
    finally:
        client.close()
