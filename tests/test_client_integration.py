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

    def test_warm_restart_registers_uuid_db_mappings(self, tmp_path):
        """Warm start（索引缓存使能前提）：既有 workspace+db 二次构造 PageIndexClient
        必须注册 uuid↔db 映射并回载 tree_json——修复前 _load_workspace 先于 self.db
        赋值执行，映射注册分支是死代码，重启后 L1 的 db_to_uuid 为空 → 检索恒返回
        "Super-Tree selection returned no documents"（cfce7d4 意图被构造顺序废掉）。"""
        sys.modules["PyPDF2"] = MagicMock()
        from pageindex_mutil.client import PageIndexClient, META_INDEX

        workspace = tmp_path / "ws"
        workspace.mkdir()
        db_path = str(tmp_path / "pageindex.db")

        # 造一个已索引状态：db 行（含 tree_json）+ workspace _meta.json 同名匹配
        db = PageIndexDB(db_path)
        db_doc_id = db.insert_document(pdf_name="warm.md", pdf_path="/tmp/warm.md")
        tree = [{"node_id": "0000", "title": "章节", "nodes": []}]
        db.update_document_tree(db_doc_id, json.dumps(tree, ensure_ascii=False))
        doc_uuid = "11111111-2222-3333-4444-555555555555"
        (workspace / META_INDEX).write_text(json.dumps({
            doc_uuid: {"type": "md", "path": "/tmp/warm.md",
                       "doc_name": "warm.md", "doc_description": "d"},
        }, ensure_ascii=False), encoding="utf-8")

        client = PageIndexClient(workspace=str(workspace), db_path=db_path)
        try:
            # 根因断言：uuid↔db 映射已注册（修复前为空）
            assert client._id_mapper.to_db(doc_uuid) == db_doc_id
            assert client.super_tree_index._get_db_to_uuid().get(db_doc_id) == doc_uuid
            # tree_json 回载：结构就绪，检索链无需重建索引
            doc = client.documents.get(doc_uuid)
            assert doc is not None and doc.get("structure") == tree
        finally:
            client.close()
            db.close()

    def test_warm_restart_db_only_branch(self, tmp_path):
        """Warm start db-only 分支（评测缓存形态）：workspace 无 _meta.json
        （index_batch 从不写它）→ documents 空 → db-only 走 uuid.uuid4() 新建
        条目。修复前 for-循环变量 `uuid` 遮蔽模块级导入，documents 空时该分支
        必抛 UnboundLocalError 且被宽 except 吞掉 → 热启动静默失败（缓存 HIT
        加载 0 documents，91782bc warm-start 修复在这条路径上从未走通）。"""
        sys.modules["PyPDF2"] = MagicMock()
        from pageindex_mutil.client import PageIndexClient

        workspace = tmp_path / "ws_dbonly"
        workspace.mkdir()  # 空 workspace：无 _meta.json
        db_path = str(tmp_path / "pageindex.db")

        db = PageIndexDB(db_path)
        db_doc_id = db.insert_document(pdf_name="cached.md", pdf_path="/tmp/cached.md")
        tree = [{"node_id": "0000", "title": "章节", "nodes": []}]
        db.update_document_tree(db_doc_id, json.dumps(tree, ensure_ascii=False))

        client = PageIndexClient(workspace=str(workspace), db_path=db_path)
        try:
            # db-only 分支应注册映射（修复前整段被 UnboundLocalError 打断）
            uuids = [u for u, d in client._id_mapper.items()]
            assert len(uuids) == 1, "db-only 分支应为 db 行注册一个 uuid"
            loaded = client.documents.get(uuids[0])
            assert loaded is not None
            assert loaded["doc_name"] == "cached.md"
            assert loaded.get("structure") == tree  # tree_json 回载
        finally:
            client.close()
            db.close()

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


# ---------------------------------------------------------------------------
# T32.1 收尾：热启动落库契约——tree_json 必须保留 span 字段与 text
# （修复前 index/index_batch 用 create_clean_structure_for_description 白名单
# 落库，span_kind/line_num/end_line/start_index/end_index/text 全剥 → server
# 重启 / 评测缓存 HIT 回载后 reasoning.spans_from_nodes 全空 →
# router._recall_nodes_for_doc 报 "selected nodes yield no spans (legacy
# index?); dropped" → 检索空。注意：修复前已落库的存量缓存实例（如 899 条
# 评测缓存）仍是旧 schema（无 span/text），其行为不变——需父代理重建缓存。
# ---------------------------------------------------------------------------


def test_structure_for_persistence_keeps_span_and_text():
    """白名单单测：保留语义字段 + span 字段 + text，剥离其余键，递归防御同构。"""
    from pageindex_mutil.utils import structure_for_persistence

    structure = [{
        "title": "第一章", "node_id": "0001", "summary": "s1",
        "prefix_summary": "p1", "span_kind": "line",
        "line_num": 1, "end_line": 10, "text": "第一章正文",
        "level": 1,                      # 白名单外 → 必须被剥
        "nodes": [{
            "title": "1.1", "node_id": "0002",
            "span_kind": "page", "start_index": 2, "end_index": 3,
            "text": "body", "entities": ["junk"],  # 白名单外 → 剥
            "nodes": [],
        }],
    }]
    out = structure_for_persistence(structure)
    assert isinstance(out, list) and len(out) == 1
    root = out[0]
    for key in ("title", "node_id", "summary", "prefix_summary",
                "span_kind", "line_num", "end_line", "text"):
        assert root[key] == structure[0][key]
    assert "level" not in root
    child = root["nodes"][0]
    for key in ("title", "node_id", "span_kind", "start_index", "end_index", "text"):
        assert child[key] == structure[0]["nodes"][0][key]
    assert "entities" not in child
    assert "nodes" not in child          # 空 children 不落（同 create_clean 语义）
    # 防御非 dict / 非 list（与 create_clean_structure_for_description 同构）
    assert structure_for_persistence("scalar") == "scalar"
    assert structure_for_persistence(None) is None
    # span 字段可被 spans_from_nodes 消费：双跨度皆存活
    from pageindex_mutil.reasoning import spans_from_nodes
    spans = spans_from_nodes([root, child])
    assert spans["lines"] == [("0001", 1, 10)]
    assert set(spans["pages"]) == {2, 3}


def test_index_md_persists_span_and_text_survives_hot_restart(tmp_path):
    """集成：真实 MD 管线 index()（仅 LLM 边界 mock）→ db.tree_json 含 span/text
    → 二次构造 client（server 重启等价）回载后 spans_from_nodes 非空。"""
    sys.modules["PyPDF2"] = MagicMock()
    import pageindex_mutil.page_index_md as page_index_md_mod
    import pageindex_mutil.utils as utils_mod
    from pageindex_mutil.client import PageIndexClient
    from pageindex_mutil.reasoning import spans_from_nodes
    from pageindex_mutil.utils import structure_to_list

    md = tmp_path / "game-guide.md"
    md.write_text(
        "# 每日任务\n\n完成日常任务可获得浴血值奖励。\n\n"
        "# 装备强化\n\n强化需要消耗玄铁与金币。\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    db_path = str(tmp_path / "hot.db")

    # search_backend="keyword"：默认 hybrid 的向量通道会触 HF/HTTP 网络重试。
    client = PageIndexClient(workspace=str(workspace), db_path=db_path,
                             search_backend="keyword")
    client.entity_extractor = None  # 跳实体抽取（LLM），落库断言不依赖它
    client.super_tree_index.on_document_added = MagicMock()  # 词面索引 LLM 通道关闭
    try:
        # generate_doc_description / doc_summary 走各自命名空间的 llm_completion，
        # 两处都 mock——小文档节点 <200 token，摘要走 text 直取，不触 LLM。
        with patch("pageindex_mutil.client.llm_completion",
                   return_value="mock doc summary"), \
             patch("pageindex_mutil.utils.llm_completion",
                   return_value="mock description"):
            doc_id = client.index(str(md), mode="md")

        # 落库面契约：tree_json 节点保留 span 字段与 text
        db_doc_id = client._id_mapper.to_db(doc_id)
        row = client.db.get_document_by_id(db_doc_id)
        assert row and row.get("tree_json")
        persisted = json.loads(row["tree_json"])
        nodes = structure_to_list(persisted)
        assert len(nodes) == 2
        for node in nodes:
            assert node.get("span_kind") == "line"
            assert isinstance(node.get("line_num"), int)
            assert isinstance(node.get("end_line"), int)
            assert node.get("text")
            assert "level" not in node  # 白名单外的内存字段不进 tree_json
        assert spans_from_nodes(nodes)["lines"] == [
            (n["node_id"], n["line_num"], n["end_line"]) for n in nodes
        ]
    finally:
        client.close()

    # 热启动面契约（T32.1 根因场景）：新进程形态二次构造 →
    # documents[uuid]["structure"] 由 tree_json 回载 → span 消费方直接可用，
    # router._recall_nodes_for_doc 不再触发 "yield no spans (legacy index?)"。
    client2 = PageIndexClient(workspace=str(workspace), db_path=db_path,
                              search_backend="keyword")
    try:
        uuids = [u for u, _ in client2._id_mapper.items()]
        assert len(uuids) == 1
        reloaded = client2.documents[uuids[0]]["structure"]
        assert reloaded == persisted
        spans = spans_from_nodes(structure_to_list(reloaded))
        assert spans["lines"], "热启动回载后 span 全空——落库契约再次断裂"
    finally:
        client2.close()


def test_batch_parse_and_insert_persists_span_and_text(tmp_path):
    """batch 写入点（_parse_and_insert_doc，index_batch Phase 1 调用）同样走
    structure_for_persistence——mock md_to_tree（patch 自动识别 async）注入含
    span/text 的结构，断言落库存活、白名单外键剥离。"""
    sys.modules["PyPDF2"] = MagicMock()
    from pageindex_mutil.client import PageIndexClient
    from pageindex_mutil.utils import structure_to_list

    md = tmp_path / "batch.md"
    md.write_text("# Batch Doc\n\nbody text\n", encoding="utf-8")
    client = PageIndexClient(db_path=str(tmp_path / "batch.db"),
                             search_backend="keyword")
    try:
        mock_structure = [{
            "node_id": "0001", "title": "Batch Doc", "text": "body text",
            "summary": "s", "span_kind": "line", "line_num": 1,
            "end_line": 2, "level": 1, "nodes": [],
        }]
        with patch("pageindex_mutil.client.md_to_tree", return_value={
            "doc_name": "batch.md", "doc_description": "d",
            "line_count": 3, "structure": mock_structure,
        }):
            doc_id, db_doc_id, _ = client._parse_and_insert_doc(str(md), mode="md")
        row = client.db.get_document_by_id(db_doc_id)
        persisted = structure_to_list(json.loads(row["tree_json"]))
        assert len(persisted) == 1
        node = persisted[0]
        assert node["span_kind"] == "line"
        assert node["line_num"] == 1 and node["end_line"] == 2
        assert node["text"] == "body text"
        assert "level" not in node
    finally:
        client.close()


@pytest.mark.asyncio
async def test_single_doc_lexical_miss_returns_prefilter_empty(tmp_path):
    """(T8 审查/T13) 单文档词面未命中契约 [S4]：查询词不命中任何 doc_keywords →
    统一链 L0 证据束空 → 优雅空响应（confidence low、matched_docs 空、不抛异常）。
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
        # [S4]/T13 单链契约：证据束空 → 优雅空响应（不再走 content_fallback）
        assert result["confidence"] == "low"
        assert result["matched_docs"] == []
        assert result["selected_nodes"] == []
    finally:
        client.close()
