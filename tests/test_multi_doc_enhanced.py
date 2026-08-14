"""T6.4 多文档路径统一接入 enhance_and_select 测试（spec [3.2.1]/[3.4]/P2.7-P2.9）。

验收覆盖：
1. _recall_nodes_for_doc：enhance_and_select 精挑（unit=节点）——选择顺序保持、
   证据接地进 prompt（浴血值式断言）、DB node_profiles 优先/structure 键兜底、
   matched_info 内容命中词并入关键词证据；
2. 启发式关键词兜底已移除：LLM 精挑为空即无召回（LLM 唯一裁剪者，[7.7]）；
3. pool_concern 重选（至多一次）：有 deferred → 放宽 union 上限；
   无 deferred → force-all 全池（同步放宽 cap；×2 约定）；
4. matched_docs score 语义统一：_search_super_tree 用节点召回覆盖度
   （evidence-derived，确定性 (0,1]），无召回证据的文档不进 matched；
   Act 失败 → matched 为空（不虚报）；响应形状键不变；
5. multi_hop matched_docs：覆盖度分数（doc_scores_out 回填）；缺失防御回退 1.0。

全部 LLM 调用均 mock —— 无真实 LLM。
"""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# 测试隔离守卫（与 test_search_single_enhanced 同理）：收集期清理预置 stub 并
# 干净加载；运行期由 _real_modules fixture 保证真实模块在场；patch 一律
# patch.object(module, ...)（模块对象经惰性访问器取）。
for _mod in list(sys.modules):
    if _mod == "pageindex_mutil" or _mod.startswith("pageindex_mutil."):
        del sys.modules[_mod]
sys.modules.setdefault("PyPDF2", MagicMock())  # client 链顶层导入 PyPDF2

import pageindex_mutil.client  # noqa: F401  首次干净加载


@pytest.fixture(autouse=True)
def _real_modules():
    for _m in list(sys.modules):
        if _m == "pageindex_mutil" or _m.startswith("pageindex_mutil."):
            del sys.modules[_m]
    import pageindex_mutil.client  # noqa: F401
    import pageindex_mutil.agentic.enhance  # noqa: F401
    import pageindex_mutil.agentic.router  # noqa: F401
    import pageindex_mutil.agentic.multi_hop  # noqa: F401
    import pageindex_mutil.reasoning  # noqa: F401
    yield


def _enhance_mod():
    import pageindex_mutil.agentic.enhance as m
    return m


def _router_mod():
    import pageindex_mutil.agentic.router as m
    return m


def _multi_hop_mod():
    import pageindex_mutil.agentic.multi_hop as m
    return m


def _reasoning_mod():
    import pageindex_mutil.reasoning as m
    return m


def _evidence_mod():
    import pageindex_mutil.agentic.evidence as m
    return m


def _node(nid, title="标题", summary="摘要", text="正文", **extra):
    node = {"node_id": nid, "title": title, "summary": summary, "text": text,
            "start_index": 0, "end_index": 0}
    node.update(extra)
    return node


def _select_json(selected, pool_concern=False, concern_reason=""):
    return json.dumps({
        "selected_ids": selected,
        "pool_concern": pool_concern,
        "concern_reason": concern_reason,
    })


def _router_with_doc(structure, doc_id="doc1", model="m-model", retrieve_model="r-model",
                     doc_type="md"):
    """Build a real AgenticRouter over a MagicMock client holding one in-memory doc."""
    AgenticRouter = _router_mod().AgenticRouter
    client = MagicMock()
    client.documents = {
        doc_id: {
            "doc_name": f"test.{doc_type}",
            "type": doc_type,
            "structure": structure,
            "pages": [],
        }
    }
    client.closet_index = None
    client.super_tree_index = None
    client.db = None
    client._uuid_to_db = {}
    router = AgenticRouter(client, model=model, retrieve_model=retrieve_model)
    return router, client


def _patch_enhance_llm(**kwargs):
    return patch.object(_enhance_mod(), "llm_completion", **kwargs)


def _patch_generate_answer(return_value="ANSWER"):
    return patch.object(
        _reasoning_mod(), "generate_answer",
        MagicMock(return_value=return_value),
    )


# ---------------------------------------------------------------------------
# 1. _recall_nodes_for_doc：enhance_and_select 精挑（unit=节点）
# ---------------------------------------------------------------------------


class TestRecallNodesForDoc:
    def test_enhance_selection_order_preserved_and_coverage_score(self):
        """LLM 精挑序保持（无重排）；relevance_score = 召回覆盖度 (0,1]。"""
        router, _ = _router_with_doc([_node(f"n{i}") for i in range(4)])
        with _patch_enhance_llm(return_value=_select_json(["n2", "n0"])):
            result = asyncio.run(router._recall_nodes_for_doc("q", "doc1"))
        assert result is not None
        assert [n["node_id"] for n in result["selected"]] == ["n2", "n0"]
        # 2 / 4 候选 = 0.5（覆盖度，而非硬编码/len(顶层)）
        assert result["relevance_score"] == 0.5
        assert result["pages"]

    def test_evidence_grounding_reaches_prompt_and_selected(self):
        """浴血值式断言：签名关键词作为证据进精挑 prompt，LLM 依据证据选中。"""
        router, client = _router_with_doc([
            _node("nodeA", title="浴血值获取", text="浴血值可以通过日常任务获得。",
                  keywords=["浴血值", "日常任务"]),
            _node("nodeB", title="天气", text="今天天气不错。", keywords=["天气"]),
        ])

        captured = []

        def fake_llm(model, prompt, **kwargs):
            captured.append(prompt)
            # 模拟 LLM：只有看到 浴血值 证据才选 nodeA
            return _select_json(["nodeA"]) if "浴血值" in prompt else _select_json([])

        with _patch_enhance_llm(side_effect=fake_llm):
            result = asyncio.run(router._recall_nodes_for_doc("浴血值怎么获得", "doc1"))

        assert len(captured) == 1
        assert "关键词命中：浴血值" in captured[0]
        assert "候选节点 nodeB" not in captured[0]  # nodeB 无命中不进 union
        assert [n["node_id"] for n in result["selected"]] == ["nodeA"]

    def test_db_profiles_preferred_via_id_mapper(self):
        """签名解析 DB 优先：经 _id_mapper.to_db 的整数 id 查 node_profiles。"""
        router, client = _router_with_doc([_node("n1", keywords=["浴血值"])])
        fake_db = MagicMock()
        fake_db.get_node_profiles.return_value = [
            {"node_id": "n1", "entities": [], "keywords": ["声望"], "tags": []},
        ]
        fake_db.search_entities.return_value = []
        client.db = fake_db
        from pageindex_mutil.client import DocIdMapper
        client._id_mapper = DocIdMapper()
        client._id_mapper.register("doc1", 7)

        captured = []

        def fake_llm(model, prompt, **kwargs):
            captured.append(prompt)
            return _select_json(["n1"])

        with _patch_enhance_llm(side_effect=fake_llm):
            result = asyncio.run(router._recall_nodes_for_doc("声望值", "doc1"))

        fake_db.get_node_profiles.assert_called_once_with(7)  # db 整数 id
        assert "关键词命中：声望" in captured[0]
        assert "浴血值" not in captured[0]  # DB 签名优先，structure 键被忽略
        assert [n["node_id"] for n in result["selected"]] == ["n1"]

    def test_matched_info_keyword_folded_into_evidence(self):
        """词面接地保全：调用方 matched_info 命中词并入该节点关键词证据。"""
        router, _ = _router_with_doc([
            _node("n1", title="玩法总览", text="……"),
            _node("n2", title="其他", text="……"),
        ])
        captured = []

        def fake_llm(model, prompt, **kwargs):
            captured.append(prompt)
            return _select_json(["n1"]) if "帮会战" in prompt else _select_json([])

        with _patch_enhance_llm(side_effect=fake_llm):
            result = asyncio.run(router._recall_nodes_for_doc(
                "帮会战奖励", "doc1",
                matched_info=[{"node_id": "n1", "keyword": "帮会战", "context": "ctx"}],
            ))

        assert "关键词命中：帮会战" in captured[0]
        assert [n["node_id"] for n in result["selected"]] == ["n1"]

    def test_content_text_channel_rescues_unprofiled_node(self):
        """P2.6 正文内容通道：节点存储签名缺失/被垃圾词淹没时，正文含查询词
        即进 union；命中词以关键词命中呈现于 prompt，精挑选中后 text 可组装上下文。

        n_sig 经 matched_info 命中词保持 union 非空（排除零信号兜底解释）；
        n_zero 无任何信号也不含查询词 → 不得进 union。
        """
        router, _ = _router_with_doc([
            _md_node("n_rel", title="玩法", text="浴血值可以通过日常任务获得。"),
            _md_node("n_sig", title="任务", text="任务系统说明。"),
            _md_node("n_zero", title="无关", text="完全无关的内容。"),
        ])

        captured = []

        def fake_llm(model, prompt, **kwargs):
            captured.append(prompt)
            # 模拟 LLM：只有看到 浴血 正文证据才选 n_rel
            return _select_json(["n_rel"]) if "浴血" in prompt else _select_json([])

        with _patch_enhance_llm(side_effect=fake_llm):
            result = asyncio.run(router._recall_nodes_for_doc(
                "浴血值怎么获得", "doc1",
                matched_info=[{"node_id": "n_sig", "keyword": "获得"}],
            ))

        assert len(captured) == 1
        assert "候选节点 n_rel" in captured[0]
        assert "候选节点 n_sig" in captured[0]
        assert "候选节点 n_zero" not in captured[0]  # 无信号不进 union
        kw_block = captured[0].split("候选节点 n_rel：", 1)[1].split("候选节点", 1)[0]
        assert "关键词命中" in kw_block and "浴血" in kw_block
        assert result is not None
        assert [n["node_id"] for n in result["selected"]] == ["n_rel"]
        assert result["selected"][0]["text"] == "浴血值可以通过日常任务获得。"

    def test_keyword_fallback_removed_llm_sole_pruner(self):
        """旧启发式关键词兜底已移除：LLM 精挑为空即无召回。

        节点正文含查询词（旧 _keyword_select_nodes 会兜底选中），但签名零信号
        全量送 LLM 后 LLM 判空 → None（LLM 唯一裁剪者，[7.7]）。
        """
        router, _ = _router_with_doc([
            _node("n1", title="无关节点", text="浴血值在这里出现但签名无记录。"),
        ])
        with _patch_enhance_llm(return_value=_select_json([])):
            result = asyncio.run(router._recall_nodes_for_doc("浴血值怎么获得", "doc1"))
        assert result is None

    def test_missing_doc_or_structure_returns_none(self):
        router, client = _router_with_doc([_node("n1")])
        assert asyncio.run(router._recall_nodes_for_doc("q", "absent")) is None
        client.documents["empty"] = {"structure": []}
        assert asyncio.run(router._recall_nodes_for_doc("q", "empty")) is None

    def test_query_entities_resolved_and_reach_entity_channel(self):
        """query_entities 接线：db 实体命中 → 实体通道证据进 prompt。"""
        router, client = _router_with_doc([
            _node("n1", entities=[{"name": "张三", "type": "人物"}]),
            _node("n2"),
        ])
        fake_db = MagicMock()
        fake_db.get_node_profiles.return_value = []
        fake_db.search_entities.return_value = [{"name": "张三", "aliases": '["张生"]'}]
        client.db = fake_db

        captured = []

        def fake_llm(model, prompt, **kwargs):
            captured.append(prompt)
            return _select_json(["n1"])

        with _patch_enhance_llm(side_effect=fake_llm):
            result = asyncio.run(router._recall_nodes_for_doc("张三的事", "doc1"))

        assert "实体匹配：张三（人物）" in captured[0]
        assert [n["node_id"] for n in result["selected"]] == ["n1"]


# ---------------------------------------------------------------------------
# 2. pool_concern 重选（[3.2.1]：放宽 union 上限，×2 约定）
# ---------------------------------------------------------------------------


def _instrumented_enhancer(select_results, model="m-model", retrieve_model="r-model"):
    """真实 UnifiedNodeEnhancement + 记录式 enhance_and_select 替身。"""
    enh = _enhance_mod().UnifiedNodeEnhancement(model, retrieve_model=retrieve_model)
    calls = []

    async def fake_select(query, candidates, profiles, query_entities=None,
                          node_budget=None, token_budget=None, max_candidates=None,
                          force_all_candidates=False):
        calls.append({
            "query": query, "candidates": candidates, "profiles": profiles,
            "query_entities": query_entities, "max_candidates": max_candidates,
            "force_all_candidates": force_all_candidates,
        })
        return select_results[len(calls) - 1]

    enh.enhance_and_select = fake_select
    return enh, calls


class TestRecallPoolConcernRetry:
    def test_retry_relaxes_max_candidates_keeps_candidates(self):
        from pageindex_mutil.agentic.enhance import POOL_CONCERN_RETRY_CAP_MULTIPLIER
        router, _ = _router_with_doc([_node("n0"), _node("n2")])
        results = [
            {"selected_ids": ["n0"], "pool_concern": True,
             "concern_reason": "疑似漏掉分支", "deferred": ["n2"]},
            {"selected_ids": ["n0", "n2"], "pool_concern": False,
             "concern_reason": "", "deferred": []},
        ]
        enh, calls = _instrumented_enhancer(results)
        with patch.object(_enhance_mod(), "UnifiedNodeEnhancement",
                          lambda model, retrieve_model=None: enh):
            result = asyncio.run(router._recall_nodes_for_doc("两处证据", "doc1"))

        assert len(calls) == 2
        # 第一次按配置上限；第二次放宽 ×POOL_CONCERN_RETRY_CAP_MULTIPLIER
        assert calls[0]["max_candidates"] is None
        assert calls[1]["max_candidates"] == (
            enh.union_max_candidates * POOL_CONCERN_RETRY_CAP_MULTIPLIER
        )
        # 候选/签名不变（deferred 经 union 自然回池）
        assert calls[1]["candidates"] is calls[0]["candidates"]
        assert calls[1]["profiles"] == calls[0]["profiles"]
        assert [n["node_id"] for n in result["selected"]] == ["n0", "n2"]

    def test_full_pool_retry_when_pool_concern_and_deferred_empty(self):
        """P2.6：pool_concern 且 deferred 为空（判据①关键概念无命中）→
        force_all_candidates=True 全量重选一次；第二次结果生效；至多重试一次。
        审查加固：全池重选同步放宽 cap——候选数超 cap 时零信号候选才不会
        按分降序垫底被再次截掉（否则全池重选退化为 pass-1）。"""
        from pageindex_mutil.agentic.enhance import POOL_CONCERN_RETRY_CAP_MULTIPLIER
        router, _ = _router_with_doc([_node("n0"), _node("n1")])
        results = [
            {"selected_ids": [], "pool_concern": True,
             "concern_reason": "关键概念无命中", "deferred": []},
            {"selected_ids": ["n1"], "pool_concern": False,
             "concern_reason": "", "deferred": []},
            # 第三次结果永远不该被消费（无循环）
            {"selected_ids": ["n0", "n1"], "pool_concern": False,
             "concern_reason": "", "deferred": []},
        ]
        enh, calls = _instrumented_enhancer(results)
        with patch.object(_enhance_mod(), "UnifiedNodeEnhancement",
                          lambda model, retrieve_model=None: enh):
            result = asyncio.run(router._recall_nodes_for_doc("浴血值怎么获得", "doc1"))

        assert len(calls) == 2  # 至多一次重试，无循环
        assert calls[0]["force_all_candidates"] is False
        assert calls[1]["force_all_candidates"] is True
        # 全池重选同样放宽 cap ×倍数（候选/签名不变，准入面真正扩大）
        assert calls[1]["max_candidates"] == (
            enh.union_max_candidates * POOL_CONCERN_RETRY_CAP_MULTIPLIER
        )
        assert calls[1]["candidates"] is calls[0]["candidates"]
        assert calls[1]["profiles"] == calls[0]["profiles"]
        assert result is not None
        assert [n["node_id"] for n in result["selected"]] == ["n1"]

    def test_full_pool_retry_not_repeated_when_still_concerned(self):
        """全池重选后仍 pool_concern → 接受结果（medium 语义由调用方决定），不再重试。"""
        router, _ = _router_with_doc([_node("n0")])
        results = [
            {"selected_ids": [], "pool_concern": True,
             "concern_reason": "关键概念无命中", "deferred": []},
            {"selected_ids": ["n0"], "pool_concern": True,
             "concern_reason": "仍偏弱", "deferred": []},
        ]
        enh, calls = _instrumented_enhancer(results)
        with patch.object(_enhance_mod(), "UnifiedNodeEnhancement",
                          lambda model, retrieve_model=None: enh):
            result = asyncio.run(router._recall_nodes_for_doc("q", "doc1"))
        assert len(calls) == 2
        assert result is not None
        assert [n["node_id"] for n in result["selected"]] == ["n0"]


# ---------------------------------------------------------------------------
# 3. _search_super_tree：matched_docs score = 召回覆盖度（evidence-derived）
# ---------------------------------------------------------------------------


def _plan_result(queries=None):
    from pageindex_mutil.agentic.planner import PlanResult
    return PlanResult(queries=queries or ["q"], weights={}, query_type="factual")


class TestSuperTreeMatchedScores:
    def test_matched_scores_are_evidence_score(self):
        """端到端：prefilter→选档→enhance 节点召回→matched score = 证据分
        （derive_evidence_score，通道命中加权标量，[S6]#8）。"""
        router, client = _router_with_doc([_node("n0")])

        mock_st = MagicMock()
        mock_st.prefilter.return_value = {1: 1.0}
        mock_st.select_documents = AsyncMock(return_value=(["doc1"], {}))
        router.super_tree_index = mock_st
        router.planner.plan = AsyncMock(return_value=_plan_result())
        router.verifier.verify = MagicMock(return_value=MagicMock(action="answer"))

        # 触发 bundle 构建：db 非 None；recall 路径经空签名回退结构（安全）
        client.db = MagicMock()
        client.db.get_node_profiles.return_value = []
        client.db.search_entities.return_value = []
        client._id_mapper = None
        client._uuid_to_db = {"doc1": 7}
        bundle = {7: {"channels": {
            "keyword": [{"token": "x"}, {"token": "y"}], "tag": [], "entity": [],
        }, "graph": {}}}
        with patch.object(_evidence_mod(), "build_evidence_bundle", return_value=bundle), \
                _patch_enhance_llm(return_value=_select_json(["n0"])), \
                _patch_generate_answer("最终答案"):
            result = asyncio.run(router._search_super_tree("q", top_k=3))

        # 证据分 = 1*2 keyword = 2.0（不再是硬编码 1.0 / 覆盖度）
        assert result["matched_docs"] == [{"doc_id": "doc1", "score": 2.0}]
        assert result["answer"] == "最终答案"
        assert result["confidence"] == "high"
        assert set(result.keys()) == {
            "query", "mode", "answer", "confidence",
            "matched_docs", "selected_nodes", "pages",
        }

    def test_doc_without_recall_evidence_excluded_from_matched(self):
        """LLM 精挑为空（无召回证据）的文档不进 matched（不虚报匹配）。"""
        router, _ = _router_with_doc([_node("n0")])

        mock_st = MagicMock()
        mock_st.prefilter.return_value = {1: 1.0}
        mock_st.select_documents = AsyncMock(return_value=(["doc1"], {}))
        router.super_tree_index = mock_st
        router.planner.plan = AsyncMock(return_value=_plan_result())
        router.verifier.verify = MagicMock(return_value=MagicMock(action="answer"))

        with _patch_enhance_llm(return_value=_select_json([])), \
                _patch_generate_answer():
            result = asyncio.run(router._search_super_tree("q", top_k=3))

        assert result["matched_docs"] == []
        assert result["confidence"] == "low"

    def test_act_failure_matched_empty_no_fake_evidence(self):
        """Act 阶段异常 → 无证据接地 → matched 为空（不再带硬编码 1.0）。"""
        router, _ = _router_with_doc([_node("n0")])

        mock_st = MagicMock()
        mock_st.prefilter.return_value = {1: 1.0}
        mock_st.select_documents = AsyncMock(return_value=(["doc1"], {}))
        router.super_tree_index = mock_st
        router.planner.plan = AsyncMock(return_value=_plan_result())
        router._act_tree_search = AsyncMock(side_effect=RuntimeError("boom"))

        result = asyncio.run(router._search_super_tree("q", top_k=3))
        assert "Failed to retrieve content" in result["answer"]
        assert result["matched_docs"] == []


class TestActTreeSearchScoresOut:
    """doc_scores_out：证据派生分数回填（预算截断前全量记录）。"""

    @pytest.mark.asyncio
    async def test_doc_scores_out_filled_for_all_recall_successes(self):
        router, _ = _router_with_doc([_node("n0")])
        router._main_funcs = {
            "build_context_for_doc": lambda doc, selected, pages: "ctx",
            "spans_from_nodes": lambda n: [1],
        }

        async def fake_recall(query, doc_id, matched_info=None):
            if doc_id == "d2":
                return None  # 召回失败：不进分数回填
            return {
                "doc_id": doc_id,
                "doc": {"doc_name": doc_id, "type": "md"},
                "structure": [{"node_id": "n0"}],
                "selected": [{"node_id": "n0", "text": "x"}],
                "pages": [1],
                "relevance_score": 0.3333,
            }

        router._recall_nodes_for_doc = fake_recall
        scores = {}
        await router._act_tree_search("q", ["d1", "d2"], doc_scores_out=scores)
        assert scores == {"d1": 0.3333}

    @pytest.mark.asyncio
    async def test_doc_scores_out_optional_backward_compatible(self):
        """不传 doc_scores_out 行为不变（recall_loop/multi_hop 旧式调用兼容）。"""
        router, _ = _router_with_doc([_node("n0")])
        router._main_funcs = {
            "build_context_for_doc": lambda doc, selected, pages: "ctx",
            "spans_from_nodes": lambda n: [1],
        }

        async def fake_recall(query, doc_id, matched_info=None):
            return {
                "doc_id": doc_id,
                "doc": {"doc_name": doc_id, "type": "md"},
                "structure": [{"node_id": "n0"}],
                "selected": [{"node_id": "n0", "text": "x"}],
                "pages": [1],
                "relevance_score": 1.0,
            }

        router._recall_nodes_for_doc = fake_recall
        ctx, nodes, src_docs, cov, dpm, pwt = await router._act_tree_search("q", ["d1"])
        assert src_docs == 1


# ---------------------------------------------------------------------------
# 4. multi_hop matched_docs：覆盖度分数（doc_scores_out 回填）
# ---------------------------------------------------------------------------


class _FakeIdMapper:
    def __init__(self, db_to_uuid):
        self._db_to_uuid = dict(db_to_uuid)

    def to_uuid(self, db_id):
        return self._db_to_uuid.get(db_id)


class TestMultiHopMatchedScores:
    def _run(self, act_side_effect):
        """Run a 1-hop multi-hop execution with the given _act_tree_search."""
        reasoner = _multi_hop_mod().MultiHopReasoner(model="m", retrieve_model="r")

        client = MagicMock()
        db = MagicMock()
        db.search_entities.return_value = [{"id": 1, "name": "X"}]
        db.get_entity_documents.return_value = [{"id": 10}]
        client.db = db
        client._id_mapper = _FakeIdMapper({10: "uuid-10"})

        router = MagicMock(spec=_router_mod().AgenticRouter)
        router.model = "m"
        router.retrieve_model = "r"
        router.client = client
        router._load_main_funcs.return_value = {
            "generate_answer": MagicMock(return_value="final"),
        }
        router._act_tree_search = AsyncMock(side_effect=act_side_effect)
        router.verifier = MagicMock()
        router.verifier.verify.return_value = MagicMock(action="answer")

        decompose = json.dumps({"decomposable": True})
        extract = json.dumps({"entities": [], "facts": [], "next_hop_hint": ""})
        with patch.object(_multi_hop_mod(), "llm_acompletion",
                          side_effect=lambda m, p, **kw: decompose if "可分解" in p or "decomposable" in p else extract), \
                patch.object(_multi_hop_mod(), "llm_completion", return_value="ans"):
            return asyncio.run(reasoner.execute("q", router, db))

    def test_matched_scores_are_coverage_from_doc_scores_out(self):
        def fake_act(query, docs, node_matches=None, doc_scores_out=None):
            if doc_scores_out is not None:
                doc_scores_out["uuid-10"] = 0.25
            return ("ctx", [{"node_id": "n1"}], 1, 1, {"uuid-10": [1]}, [])

        result = self._run(fake_act)
        assert result["matched_docs"] == [{"doc_id": "uuid-10", "score": 0.25}]
        assert all(0 < d["score"] <= 1.0 for d in result["matched_docs"])
        # 响应形状：仍为 [{doc_id, score}] 列表
        assert set(result["matched_docs"][0].keys()) == {"doc_id", "score"}

    def test_matched_score_defensive_fallback_when_unreported(self):
        """doc_pages_map 有文档但 scores 未回填（防御场景）→ 回退 1.0，(0,1]。"""
        def fake_act(query, docs, node_matches=None, doc_scores_out=None):
            return ("ctx", [{"node_id": "n1"}], 1, 1, {"uuid-10": [1]}, [])

        result = self._run(fake_act)
        assert result["matched_docs"] == [{"doc_id": "uuid-10", "score": 1.0}]

    def test_dedup_across_hops_keeps_first_score(self):
        """跨 hop 去重：同一文档只记一次（保留首见分数）。"""
        n = [0]

        def fake_act(query, docs, node_matches=None, doc_scores_out=None):
            n[0] += 1
            if doc_scores_out is not None:
                doc_scores_out["uuid-10"] = 0.5 if n[0] == 1 else 0.9
            return ("ctx", [{"node_id": "n1"}], 1, 1, {"uuid-10": [1]}, [])

        reasoner = _multi_hop_mod().MultiHopReasoner(model="m", retrieve_model="r")
        client = MagicMock()
        db = MagicMock()
        db.search_entities.return_value = [{"id": 1, "name": "X"}]
        db.get_entity_documents.return_value = [{"id": 10}]
        client.db = db
        client._id_mapper = _FakeIdMapper({10: "uuid-10"})
        router = MagicMock(spec=_router_mod().AgenticRouter)
        router.model = "m"
        router.retrieve_model = "r"
        router.client = client
        router._load_main_funcs.return_value = {
            "generate_answer": MagicMock(return_value="final"),
        }
        router._act_tree_search = AsyncMock(side_effect=fake_act)
        router.verifier = None  # 走启发式 confidence，聚焦 matched 断言

        decompose = json.dumps({"decomposable": True})
        hop1 = json.dumps({"entities": ["A"], "facts": [], "next_hop_hint": "B"})
        hop2 = json.dumps({"entities": [], "facts": [], "next_hop_hint": ""})
        calls = [0]

        def fake_acompletion(model, prompt, **kw):
            calls[0] += 1
            if "可分解" in prompt or "decomposable" in prompt:
                return decompose
            if calls[0] == 2:
                return hop1
            return hop2

        db.get_entity_relations.return_value = []
        with patch.object(_multi_hop_mod(), "llm_acompletion", side_effect=fake_acompletion), \
                patch.object(_multi_hop_mod(), "llm_completion", return_value="ans"):
            result = asyncio.run(reasoner.execute("q", router, db))

        assert result["matched_docs"] == [{"doc_id": "uuid-10", "score": 0.5}]


# ---------------------------------------------------------------------------
# 5. T20：MD 文档节点无页码索引——pages 门槛不得拦截召回，
#    多文档 MD 语料上下文组装走节点 text（build_context_for_doc 的 md 分支）
# ---------------------------------------------------------------------------


def _md_node(nid, title="标题", summary="摘要", text="正文", **extra):
    """真实 MD 节点形态：page_index_md.py 写 span_kind/line_num/end_line 行跨度，
    不写 start_index/end_index——因此 spans_from_nodes 派 line 跨度（pages 恒空，
    lines 恒有）。T20 回归守卫的靶子形状。"""
    node = {"node_id": nid, "title": title, "summary": summary, "text": text,
            "span_kind": "line", "line_num": 1, "end_line": 2}
    node.update(extra)
    return node


class TestMdNodeRecallNoPagesGate:
    def test_md_recall_returns_selected_with_empty_pages(self):
        """MD 文档：节点行跨度 → pages 为空但 lines 非空 → 召回成功（dict 而非 None）。"""
        router, _ = _router_with_doc([
            _md_node("n1", title="浴血值获取",
                     text="浴血值可以通过日常任务获得。", keywords=["浴血值"]),
            _md_node("n2", title="天气", text="今天天气不错。"),
        ])
        with _patch_enhance_llm(return_value=_select_json(["n1"])):
            result = asyncio.run(router._recall_nodes_for_doc("浴血值怎么获得", "doc1"))
        assert result is not None
        assert [n["node_id"] for n in result["selected"]] == ["n1"]
        assert result["pages"] == []  # MD 节点无页码索引
        assert result["lines"] == [("n1", 1, 2)]  # line 跨度接地（纯 span 门槛）
        assert result["relevance_score"] == 0.5  # 1/2 候选覆盖度

    def test_pdf_without_pages_still_gated(self):
        """纯 span 门槛：无任何 locator（无 span_kind/start_index/line_num）的 PDF
        节点 → pages 与 lines 皆空 → 仍返回 None（doc type hack 已移除）。"""
        router, _ = _router_with_doc([
            {"node_id": "n1", "title": "标题", "summary": "摘要", "text": "正文"},
        ], doc_type="pdf")
        with _patch_enhance_llm(return_value=_select_json(["n1"])):
            result = asyncio.run(router._recall_nodes_for_doc("q", "doc1"))
        assert result is None

    @pytest.mark.asyncio
    async def test_act_tree_search_builds_context_from_md_node_text(self):
        """_act_tree_search：MD 召回结果经节点 text 组装非空上下文，
        覆盖度进 doc_scores_out；空页列表被 doc_pages_map/pages_with_text 容忍。"""
        router, _ = _router_with_doc([
            _md_node("n1", title="浴血值获取",
                     text="浴血值可以通过日常任务获得。", keywords=["浴血值"]),
            _md_node("n2", title="天气", text="今天天气不错。"),
        ])
        scores = {}
        with _patch_enhance_llm(return_value=_select_json(["n1"])):
            ctx, nodes, src_docs, cov, dpm, pwt = await router._act_tree_search(
                "浴血值怎么获得", ["doc1"], doc_scores_out=scores,
            )
        assert ctx  # 非空上下文
        assert "浴血值可以通过日常任务获得。" in ctx  # 节点 text 进上下文
        assert "浴血值获取" in ctx  # 节点标题进上下文
        assert [n["node_id"] for n in nodes] == ["n1"]
        assert src_docs == 1
        assert scores == {"doc1": 0.5}
        assert dpm.get("doc1") == []  # 空页列表容忍，不抛错

    def test_super_tree_end_to_end_md_corpus_matched_and_context(self):
        """端到端：MD 语料经 _search_super_tree → matched_docs 非空（证据分，
        无 bundle 条目给 0），且传给答案 LLM 的上下文含节点 text（不再 No relevant content）。"""
        router, _ = _router_with_doc([
            _md_node("n1", title="浴血值获取",
                     text="浴血值可以通过日常任务获得。", keywords=["浴血值"]),
            _md_node("n2", title="天气", text="今天天气不错。"),
        ])

        mock_st = MagicMock()
        mock_st.prefilter.return_value = {1: 1.0}
        mock_st.select_documents = AsyncMock(return_value=(["doc1"], {}))
        router.super_tree_index = mock_st
        router.planner.plan = AsyncMock(return_value=_plan_result())
        router.verifier.verify = MagicMock(return_value=MagicMock(action="answer"))

        gen_calls = []

        def fake_generate(query, ctx):
            gen_calls.append((query, ctx))
            return "最终答案"

        with _patch_enhance_llm(return_value=_select_json(["n1"])), \
                patch.object(_reasoning_mod(), "generate_answer", fake_generate):
            result = asyncio.run(router._search_super_tree("浴血值怎么获得", top_k=3))

        # 召回未被 pages 门槛拦截 → 证据分进 matched（db=None → bundle 空 → 0.0）
        assert result["matched_docs"] == [{"doc_id": "doc1", "score": 0.0}]
        assert result["answer"] == "最终答案"
        assert result["confidence"] == "high"
        # 传给答案 LLM 的上下文确实含节点 text（证据接地闭环）
        assert len(gen_calls) == 1
        assert "浴血值可以通过日常任务获得。" in gen_calls[0][1]


# ---------------------------------------------------------------------------
# P2-Fix4: NFR4 router-site retrieve_model test
# ---------------------------------------------------------------------------


class TestNFR4RouterSite:
    def test_router_enhance_uses_retrieve_model(self):
        """NFR4：router 构造时传入 retrieve_model，enhance 内 llm_completion 用 retrieve_model。"""
        import asyncio
        from pageindex_mutil.agentic.router import AgenticRouter

        models = []

        mock_client = MagicMock()
        mock_client._ensure_doc_loaded = MagicMock()
        mock_client.documents = {
            "doc1": {
                "doc_name": "test", "type": "md",
                "structure": [_md_node("n0")], "pages": [],
            }
        }
        mock_client._id_mapper = MagicMock()
        mock_client._id_mapper.to_db = MagicMock(return_value=1)
        mock_client.db = None

        router = AgenticRouter(mock_client, model="m", retrieve_model="r-model")

        def fake_llm(model, prompt, **kw):
            models.append(model)
            return _select_json(["n0"], False, "")

        async def call_recall():
            with patch.object(_enhance_mod(), "llm_completion", side_effect=fake_llm):
                return await router._recall_nodes_for_doc("test", "doc1")

        asyncio.run(call_recall())
        assert models and all(m == "r-model" for m in models)
