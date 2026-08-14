"""P2 单链语义生产链路级整合回归（spec [S4]/[S12]）。

用真实 PageIndexClient + 真实 db/索引流程，仅在 LLM 边界 mock（planner 计划 /
L1 选档 / L2 节点精挑 / 答案生成 / verifier 校验），贯穿 client.search →
router.search → _search_super_tree 全链：

1. 候选=1 短路：单文档语料 L1 候选=1 时零 L1 LLM 调用（单文档成本≈0），answer 非空；
2. 扩召点名接线：verifier 判 expand 且 need 点名 doc2 → recall_loop 续接轮只补点名文档；
3. MD line span locator：span_kind/line_num/end_line 经全链 → selected_nodes/pages 行区间，
   上下文组装用节点 text（无页码臆造）；
4. LLM 不可用降级：节点精挑 LLM 抛异常 → [7.7] 放行 union → 单链不崩、confidence=low、
   matched_docs 非空（证据束接地未丢）。

sys.modules 污染陷阱（test_router.py / test_super_tree.py / test_agentic_recall.py
收集/运行期会 clobber pageindex_mutil.* 为 stub 或重载为全新对象）：不按
sys.modules[Cls.__module__] 反查打补丁，也不在模块层持有跨文件可能失效的引用——
每次测试由 _fresh_modules() 干净重载真实模块并直接持有模块对象，patch.object
打到类方法 globals 真正所属的模块上；client 也由同一批重载的模块构建，保证一致。
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# 查询/语料词面（与 test_client_integration 的 T8 单文档契约同源）：jieba 切出
# 「浴血」命中 doc_keywords content 通道，保证 L0 prefilter 稳定召回。
QUERY = "浴血值怎么获得"
NODE_TEXT = "浴血值可以通过完成日常任务获得。"


def _fresh_modules():
    """清理 pageindex_mutil.* stub 后干净重载真实模块，返回持有引用的命名空间。"""
    for _m in list(sys.modules):
        if _m == "pageindex_mutil" or _m.startswith("pageindex_mutil."):
            del sys.modules[_m]
    import pageindex_mutil.client as client_mod
    import pageindex_mutil.super_tree as super_tree_mod
    import pageindex_mutil.agentic.enhance as enhance_mod
    import pageindex_mutil.reasoning as reasoning_mod
    from pageindex_mutil.agentic.verifier import VerifyResult
    from pageindex_mutil.agentic.planner import PlanResult
    return SimpleNamespace(
        client_mod=client_mod,
        super_tree_mod=super_tree_mod,
        enhance_mod=enhance_mod,
        reasoning_mod=reasoning_mod,
        VerifyResult=VerifyResult,
        PlanResult=PlanResult,
    )


@pytest.fixture
def env(tmp_path):
    """真实 PageIndexClient(db) + 干净重载的真实模块（每测试独立，防 sys.modules 串味）。"""
    m = _fresh_modules()
    client = m.client_mod.PageIndexClient(
        db_path=str(tmp_path / "t.db"), search_backend="keyword",
    )
    # 关闭 vector/keyword search_backend（prefilter 通道 C）：其 BM25 结果与
    # 通道 B（KeywordIndex）同源且 KeywordSearchBackend 持类级缓存（key 仅
    # query+top_k，跨测试 DB 会串味）——本套用例只依赖通道 A/B/D。
    client.search_backend = None
    m.client = client
    yield m
    client.close()


def _index_doc(client, uuid_id, doc_name, description, nodes, index_body=None):
    """把一篇 MD 文档入 DB/documents/关键词索引（span_kind=line 的扁平结构）。"""
    db_doc_id = client.db.insert_document(
        doc_name, f"/tmp/{doc_name}", doc_description=description,
    )
    client._id_mapper.register(uuid_id, db_doc_id)
    client.documents[uuid_id] = {
        "doc_name": doc_name,
        "doc_description": description,
        "type": "md",
        "structure": nodes,
        "pages": [],
    }
    client.super_tree_index.on_document_added(db_doc_id, content=index_body)
    return db_doc_id


def _line_node(node_id, title, text, keywords=None):
    """MD 行跨度节点（span_kind/line_num/end_line，无页码）。"""
    node = {
        "node_id": node_id,
        "title": title,
        "summary": title,
        "text": text,
        "span_kind": "line",
        "line_num": 1,
        "end_line": 2,
        "nodes": [],
    }
    if keywords:
        node["keywords"] = keywords
    return node


def _select_json(selected, pool_concern=False, concern_reason=""):
    """enhance_and_select 的 LLM 响应 JSON（与既有单文档/多文档测试同构）。"""
    return json.dumps({
        "selected_ids": selected,
        "pool_concern": pool_concern,
        "concern_reason": concern_reason,
    })


def _route_enhance(handlers):
    """按 prompt 子串路由 enhance 节点精挑响应（同步，供 asyncio.to_thread 调用）。"""
    def fake(model, prompt, **kw):
        for marker, selected in handlers.items():
            if marker in prompt:
                return _select_json(selected)
        return _select_json([])
    return fake


def _mock_plan(router, env, query=QUERY):
    """确定性规划（单查询，无 HyDE），避免 planner 真调 LLM。"""
    router.planner.plan = AsyncMock(return_value=env.PlanResult(
        queries=[query], weights={}, query_type="factual",
    ))


# ---------------------------------------------------------------------------
# 1. 候选=1 短路：单文档语料全链零 L1 LLM 调用（单文档成本≈0）
# ---------------------------------------------------------------------------


async def test_candidate_singleton_shortcircuit_via_production_path(env):
    """单文档 → client.search 全链：L1 候选=1 短路（无 L1 LLM 调用），answer 非空。"""
    client = env.client
    _index_doc(
        client, "d1", "浴血值获取指南.md", "浴血值玩法说明",
        [_line_node("n1", "浴血值获取", NODE_TEXT, keywords=["浴血值"])],
        index_body=NODE_TEXT,
    )
    router = client.router
    _mock_plan(router, env)

    l1_calls = []

    async def _l1(model, prompt, **kw):
        l1_calls.append(prompt)
        return json.dumps({"doc_ids": ["d1"]})

    with patch.object(env.super_tree_mod, "llm_acompletion", new=_l1), \
         patch.object(env.enhance_mod, "llm_completion",
                      new=_route_enhance({"n1": ["n1"]})), \
         patch.object(env.reasoning_mod, "generate_answer",
                      new=lambda q, ctx: "最终答案"):
        router.verifier.verify = MagicMock(
            return_value=env.VerifyResult(1.0, "answer"))
        result = await client.search(QUERY)

    # [S4] 候选=1 → _holistic_select 短路：零 L1 推理 LLM 调用
    assert l1_calls == []
    assert result["answer"] == "最终答案"
    assert result["confidence"] == "high"
    assert [d["doc_id"] for d in result["matched_docs"]] == ["d1"]
    assert [n["node_id"] for n in result["selected_nodes"]] == ["n1"]


# ---------------------------------------------------------------------------
# 2. 扩召点名接线：verifier 判 expand + need 点名 doc2 → 续接轮只补点名文档
# ---------------------------------------------------------------------------


async def test_super_tree_expand_wires_recall_loop_named_fetch(env):
    """多文档、verifier 判 expand 且 need 点名 doc2：rounds_used≥2，轮 2 act 只处理点名文档。"""
    client = env.client
    _index_doc(
        client, "d1", "浴血值获取指南.md", "浴血值玩法说明",
        [_line_node("n1", "浴血值获取", NODE_TEXT, keywords=["浴血值"])],
        index_body=NODE_TEXT,
    )
    _index_doc(
        client, "d2", "浴血值副本获取指南.md", "浴血值副本说明",
        [_line_node("n2", "副本获取", "浴血值可以在周末副本中获得双倍。", keywords=["浴血值"])],
        index_body="浴血值可以在周末副本中获得双倍",
    )
    # 预置 KB 概览缓存，避免 _holistic_select 经 KBIdentity 真调同步 llm_completion
    client.db.set_kb_identity("知识库含 2 个文档。", 2)

    router = client.router
    _mock_plan(router, env)

    # L1 只选 d1（d2 留给 verifier 点名补召回）
    async def _l1(model, prompt, **kw):
        return json.dumps({"doc_ids": ["d1"]})

    # 记录 _act_tree_search 每轮的候选文档序列（保留真实生产 act）
    real_act = router._act_tree_search
    act_calls = []

    async def recording_act(query, candidate_docs, **kwargs):
        act_calls.append(list(candidate_docs))
        return await real_act(query, candidate_docs, **kwargs)

    router._act_tree_search = recording_act

    with patch.object(env.super_tree_mod, "llm_acompletion", new=_l1), \
         patch.object(env.enhance_mod, "llm_completion",
                      new=_route_enhance({"n1": ["n1"], "n2": ["n2"]})), \
         patch.object(env.reasoning_mod, "generate_answer",
                      new=lambda q, ctx: "综合答案"):
        router.verifier.verify = MagicMock(side_effect=[
            env.VerifyResult(0.5, "expand",
                             need=[{"doc_id": "d2", "reason": "缺该文档证据"}]),
            env.VerifyResult(0.9, "answer"),
        ])
        result = await client.search(QUERY)

    # [S8] 点名扩召：轮 2 只补 need 点名的 d2（不按分数序滑窗补其它文档）
    assert result["rounds_used"] == 2
    assert act_calls == [["d1"], ["d2"]]
    assert result["confidence"] == "high"
    assert result["answer"] == "综合答案"


# ---------------------------------------------------------------------------
# 3. MD line span locator：行区间端到端 + 上下文用节点 text（无页码臆造）
# ---------------------------------------------------------------------------


async def test_md_line_span_context_end_to_end(env):
    """MD 文档（span_kind/line_num/end_line）经全链：selected_nodes 行区间 + 节点 text 组装。"""
    client = env.client
    _index_doc(
        client, "d1", "浴血值获取指南.md", "浴血值玩法说明",
        [_line_node("n1", "浴血值获取", NODE_TEXT, keywords=["浴血值"])],
        index_body=NODE_TEXT,
    )
    router = client.router
    _mock_plan(router, env)

    captured = {}

    def _generate(q, ctx):
        captured["ctx"] = ctx
        return "答案"

    with patch.object(env.enhance_mod, "llm_completion",
                      new=_route_enhance({"n1": ["n1"]})), \
         patch.object(env.reasoning_mod, "generate_answer", new=_generate):
        router.verifier.verify = MagicMock(
            return_value=env.VerifyResult(1.0, "answer"))
        result = await client.search(QUERY)

    # [S10] span 分派：line 节点输出行区间，无页码臆造
    nodes = result["selected_nodes"]
    assert len(nodes) == 1
    assert nodes[0]["node_id"] == "n1"
    assert nodes[0]["start_line"] == 1
    assert nodes[0]["end_line"] == 2
    assert "pages" not in nodes[0]  # MD line 节点不输出 page 区间

    pages = result["pages"]
    assert len(pages) == 1
    assert pages[0]["doc_id"] == "d1"
    assert pages[0]["start_line"] == 1
    assert pages[0]["end_line"] == 2
    assert "page" not in pages[0]  # UI 溯源不臆造页码

    # 上下文组装用节点 text（无 "--- Page" 页码标记）
    assert NODE_TEXT in captured["ctx"]
    assert "--- Page" not in captured["ctx"]


# ---------------------------------------------------------------------------
# 4. LLM 不可用降级：[7.7] 放行 union → 单链不崩、confidence=low、matched_docs 非空
# ---------------------------------------------------------------------------


async def test_llm_unavailable_degrades_to_union(env):
    """节点精挑 LLM 抛异常 → [7.7] 放行 union（不裁剪）→ 证据束接地不丢，低置信响应。"""
    client = env.client
    _index_doc(
        client, "d1", "浴血值获取指南.md", "浴血值玩法说明",
        [_line_node("n1", "浴血值获取", NODE_TEXT, keywords=["浴血值"])],
        index_body=NODE_TEXT,
    )
    router = client.router
    _mock_plan(router, env)

    enhance_calls = []

    def _boom(model, prompt, **kw):
        enhance_calls.append(prompt)
        raise RuntimeError("LLM unavailable")

    with patch.object(env.enhance_mod, "llm_completion", new=_boom), \
         patch.object(env.reasoning_mod, "generate_answer",
                      new=lambda q, ctx: "Error: OpenAI client not initialized."):
        # 校验 LLM 亦不可用 → 无法确认接地 → 诚实拒答（low），但 matched_docs 仍非空
        router.verifier.verify = MagicMock(
            return_value=env.VerifyResult(0.0, "refuse"))
        result = await client.search(QUERY)

    # [7.7] 节点精挑 LLM 确实被调用并抛异常（而非被短路）
    assert len(enhance_calls) >= 1
    # 单链不崩：降级放行 union → 文档接地未丢 → matched_docs 非空
    assert result["confidence"] == "low"
    assert [d["doc_id"] for d in result["matched_docs"]] == ["d1"]
    assert result["answer"]  # 有响应（拒绝作答而非崩溃）
