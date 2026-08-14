"""T6.3/T11 Agentic 多轮召回循环测试（spec [3.5]/[7.5]/[7.6]/[S8]）。

覆盖：
1. 首轮即答快速路径：verifier answer → 一轮返回，不再二次 Route；
2. expand → 轮 2：只补 verifier 点名（need）的 doc_id 对象（[S8] 删除滑窗）；
3. refuse → 立即拒答；
4. max_rounds 耗尽 → best_effort：confidence low、接地再挑选、引用来源标注；
5. 累积池为空 → 诚实拒答，不编造（不调用 generate_answer）；
6. 延迟预算：极小 agentic_max_latency_ms → 轮 2 前截停 → best_effort；
7. 单轮超时：轮 2 挂死 → 有界降级 best_effort，不死锁；
8. token 总账上限 → best_effort；
9. need 空/无有效对象 → no_target 终止进 best_effort；
10. 组件容错：verifier/生成器抛错 → 降级 best_effort，不击穿；
11. 回归：_search_v2 expand 分支不再抛 pages_with_text2.items() AttributeError，且按 need 点名委派；
12. node_matches 续接转发：轮 1 Route 节点命中经委派传入续接轮；
13. max_rounds=1：轮 1 expand 无处可扩 → 直接 best_effort；
14. 续接池复用：仅给 fused（ctx_state=None）→ 以调用方池起轮 1，不重新 Route；
15. 小工具鲁棒性：_normalize_fused 滤除非有限分数；_load_settings 容忍非法配置；
16. 点名补召回：轮 ≥2 只补 need 点名对象（非按分数序滑窗补入其它文档）。

全部 LLM 调用均 mock —— 无真实 LLM。
"""
import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# 测试隔离守卫（与 test_search_single_enhanced 同理）：收集期清理其他测试文件
# 预置的 pageindex_mutil.* stub，干净加载真实模块；运行期由 _real_modules
# fixture 保证真实模块在场，断言/patch 一律经惰性访问器取当前生效模块对象。
for _mod in list(sys.modules):
    if _mod == "pageindex_mutil" or _mod.startswith("pageindex_mutil."):
        del sys.modules[_mod]

import pageindex_mutil.agentic.router  # noqa: F401  首次干净加载
from pageindex_mutil.agentic.verifier import VerifyResult
from pageindex_mutil.agentic.planner import PlanResult


@pytest.fixture(autouse=True)
def _real_modules():
    for _m in list(sys.modules):
        if _m == "pageindex_mutil" or _m.startswith("pageindex_mutil."):
            del sys.modules[_m]
    import pageindex_mutil.agentic.router  # noqa: F401
    import pageindex_mutil.agentic.recall_loop  # noqa: F401
    import pageindex_mutil.agentic.verifier  # noqa: F401
    yield


def _router_mod():
    import pageindex_mutil.agentic.router as m
    return m


def _recall_loop_mod():
    import pageindex_mutil.agentic.recall_loop as m
    return m


BASE_KEYS = {"query", "mode", "answer", "confidence", "matched_docs", "selected_nodes", "pages"}


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
    return _router_mod().AgenticRouter(mock_client, model="qwen-plus", retrieve_model="qwen-flash")


def _loop(router):
    return _recall_loop_mod().AgenticRecallLoop(router)


def _setup_route(router, n_docs=12):
    """独立模式 Route mock：metadata 单策略 → 融合序 d1..dN（分数降序）。"""
    router._build_docs_info = MagicMock(
        return_value=[{"doc_id": f"d{i + 1}"} for i in range(n_docs)]
    )
    router.planner.plan = AsyncMock(return_value=PlanResult(
        queries=["q"], weights={"metadata": 1.0}, query_type="factual"
    ))
    router._run_strategies = AsyncMock(return_value=(
        {"metadata": [(f"d{i + 1}", i + 1) for i in range(n_docs)]},
        {},
    ))


def _act_outcome(ctx="ctx-data", docs=None):
    docs = docs or ["d1"]
    return (
        ctx,
        [{"node_id": "n1", "title": "T", "summary": "S", "text": "X",
          "start_index": 1, "end_index": 2}],
        len(docs),
        len(docs),
        {d: [1] for d in docs},
        [{"doc_id": d, "page": 1, "text": "t"} for d in docs],
    )


def _r1_ctx_state():
    return {
        "ctx": "round-1-ctx",
        "nodes": [{"node_id": "n0", "title": "T0"}],
        "src_docs": 1,
        "cov_nodes": 1,
        "doc_pages_map": {"d1": [1]},
        "pages_with_text": [{"doc_id": "d1", "page": 1}],
        "matched": [{"doc_id": "d1", "score": 1.0}],
    }


def _fused_pool(n=12):
    return [(f"d{i}", 1.0 / i) for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# 1. 首轮即答快速路径（⑧）
# ---------------------------------------------------------------------------


class TestRoundOneFastPath:
    @pytest.mark.asyncio
    async def test_answer_verdict_returns_after_one_round(self, router):
        _setup_route(router, n_docs=12)
        router._act_tree_search = AsyncMock(return_value=_act_outcome(docs=["d1", "d2", "d3"]))
        router.verifier.verify = MagicMock(return_value=VerifyResult(0.9, "answer"))
        router._main_funcs = {"generate_answer": MagicMock(return_value="ANS")}

        result = await _loop(router).retrieve("q", top_k=3)

        assert result["confidence"] == "high"
        assert result["answer"] == "ANS"
        assert result["rounds_used"] == 1
        assert result["mode"] == "multi"
        assert BASE_KEYS <= set(result.keys())
        # 一次 Route（轮内并发召回）+ 一次 Act + 一次校验 —— 无第二轮开销
        assert router._run_strategies.call_count == 1
        assert router._act_tree_search.await_count == 1
        router.verifier.verify.assert_called_once()


# ---------------------------------------------------------------------------
# 2. expand → 轮 2：只补 verifier 点名（need）的 doc_id 对象（[S8] 删除滑窗）
# ---------------------------------------------------------------------------


class TestExpandNamedFetch:
    @pytest.mark.asyncio
    async def test_expand_fetches_only_named_docs(self, router):
        """[S8] 点名补召回：轮 2 只补 need 点名的 doc_id，而非按分数序滑窗补入其它文档。"""
        act_calls = []

        async def fake_act(query, candidates, node_matches=None):
            act_calls.append(list(candidates))
            return _act_outcome(docs=candidates[:1])

        router._act_tree_search = fake_act
        router.verifier.verify = MagicMock(return_value=VerifyResult(0.9, "answer"))
        router._main_funcs = {"generate_answer": MagicMock(return_value="ANS")}

        result = await _loop(router).retrieve(
            "q", top_k=3,
            first_round_fused=[("d1", 1.0), ("d3", 0.9), ("d4", 0.8),
                               ("d2", 0.7), ("d5", 0.6), ("d6", 0.5)],
            first_round_ctx_state=_r1_ctx_state(),
            expand_need=[{"doc_id": "d2", "reason": "缺该文档"}],
        )

        assert result["confidence"] == "high"
        assert result["rounds_used"] == 2
        # 点名补召回：只补 d2；d5/d6 虽在融合池内但未被点名，不得入候选
        assert act_calls == [["d2"]]

    @pytest.mark.asyncio
    async def test_expand_with_empty_need_stops_no_target(self, router):
        """[S8] need 为空 → 无点名对象 → no_target 终止进 best_effort（不再滑窗扩召）。"""
        _setup_route(router, n_docs=12)
        router._act_tree_search = AsyncMock(return_value=_act_outcome(docs=["d1", "d2", "d3"]))
        router.verifier.verify = MagicMock(return_value=VerifyResult(0.5, "expand", need=[]))
        router._main_funcs = {"generate_answer": MagicMock(return_value="ANS")}

        result = await _loop(router).retrieve("q", top_k=3)

        assert result["confidence"] == "low"
        assert "尽力作答" in result["note"]
        assert result["rounds_used"] == 1  # 轮 2 未开：need 空 → no_target
        assert router._act_tree_search.await_count == 1  # 仅轮 1；best_effort 复用轮 1 状态

    @pytest.mark.asyncio
    async def test_expand_named_need_skips_already_retrieved(self, router):
        """[S8] 点名对象已在召回集（去重）→ 无新对象 → no_target 终止。"""
        _setup_route(router, n_docs=12)
        act_calls = []

        async def fake_act(query, candidates, node_matches=None):
            act_calls.append(list(candidates))
            return _act_outcome(docs=candidates[:1])

        router._act_tree_search = fake_act
        router.verifier.verify = MagicMock(return_value=VerifyResult(0.5, "expand"))
        router._main_funcs = {"generate_answer": MagicMock(return_value="ANS")}

        result = await _loop(router).retrieve("q", top_k=3)

        assert result["confidence"] == "low"
        assert result["rounds_used"] == 1
        assert act_calls == [["d1", "d2", "d3"]]  # 仅轮 1；expand 无 need → 无轮 2


# ---------------------------------------------------------------------------
# 3. refuse 立即拒答（③ 复用 CRAG action）
# ---------------------------------------------------------------------------


class TestRefuse:
    @pytest.mark.asyncio
    async def test_refuse_returns_immediately(self, router):
        _setup_route(router)
        router._act_tree_search = AsyncMock(return_value=_act_outcome(docs=["d1"]))
        router.verifier.verify = MagicMock(return_value=VerifyResult(0.1, "refuse"))
        router._main_funcs = {"generate_answer": MagicMock(return_value="ANS")}

        result = await _loop(router).retrieve("q", top_k=3)

        assert result["answer"] == "I don't know."
        assert result["confidence"] == "low"
        router._act_tree_search.assert_awaited_once()


# ---------------------------------------------------------------------------
# 4. max_rounds 耗尽 → best_effort（[7.6]）
# ---------------------------------------------------------------------------


class TestRoundsExhaustedBestEffort:
    @pytest.mark.asyncio
    async def test_best_effort_low_confidence_grounded_with_citations(self, router):
        _setup_route(router, n_docs=30)
        act_calls = []

        async def fake_act(query, candidates, node_matches=None):
            act_calls.append(list(candidates))
            return _act_outcome(ctx="ctx-" + ",".join(candidates), docs=candidates)

        router._act_tree_search = fake_act
        # 逐轮 need 点名：轮 2 补 d4-d6，轮 3 补 d7，轮 3 后 need 空 → 循环耗尽
        router.verifier.verify = MagicMock(side_effect=[
            VerifyResult(0.5, "expand", need=[{"doc_id": f"d{i}", "reason": "缺"} for i in range(4, 7)]),
            VerifyResult(0.5, "expand", need=[{"doc_id": "d7", "reason": "缺"}]),
            VerifyResult(0.5, "expand", need=[]),
        ])
        gen = MagicMock(return_value="BEST")
        router._main_funcs = {"generate_answer": gen}

        loop = _loop(router)
        loop.max_rounds = 3
        result = await loop.retrieve("q", top_k=3)

        assert result["confidence"] == "low"
        assert result["rounds_used"] == 3
        assert "尽力作答" in result["note"]
        # 引用来源标注（grounding 后 doc_pages_map 键序取累积池头部）
        assert "引用来源" in result["note"] and "d1" in result["note"]
        # 轮 1 d1-d3 / 轮 2 d4-d6（点名）/ 轮 3 d7（点名）+ best_effort 接地再挑选
        assert len(act_calls) == 4
        assert act_calls[0] == ["d1", "d2", "d3"]
        assert act_calls[1] == ["d4", "d5", "d6"]
        assert act_calls[2] == ["d7"]
        assert act_calls[3] == [f"d{i}" for i in range(1, 8)]  # 全累积池再选择
        # matched_docs 按融合分数降序（确定性）
        assert [m["doc_id"] for m in result["matched_docs"]] == [f"d{i}" for i in range(1, 8)]

    @pytest.mark.asyncio
    async def test_max_rounds_one_expands_straight_to_best_effort(self, router):
        """max_rounds=1：轮 1 expand 判定但无处可扩 → 直接 best_effort。"""
        _setup_route(router, n_docs=12)
        router._act_tree_search = AsyncMock(return_value=_act_outcome(docs=["d1", "d2", "d3"]))
        router.verifier.verify = MagicMock(return_value=VerifyResult(0.5, "expand"))
        router._main_funcs = {"generate_answer": MagicMock(return_value="ANS")}

        result = await _loop(router).retrieve("q", top_k=3, max_rounds=1)

        assert result["confidence"] == "low"
        assert "尽力作答" in result["note"]
        assert result["rounds_used"] == 1
        assert result["answer"] == "ANS"
        # 轮 2 未开；best_effort 走轮 1 状态快捷路径（未重跑 Act）
        assert router._act_tree_search.await_count == 1


# ---------------------------------------------------------------------------
# 5. 空累积池 → 诚实拒答，不编造（[7.6]）
# ---------------------------------------------------------------------------


class TestHonestRefusalOnEmptyPool:
    @pytest.mark.asyncio
    async def test_no_evidence_honest_refusal_no_fabrication(self, router):
        _setup_route(router, n_docs=3)
        router._run_strategies = AsyncMock(return_value=({"metadata": []}, {}))
        gen = MagicMock(return_value="MADE-UP ANSWER")
        router._main_funcs = {"generate_answer": gen}

        result = await _loop(router).retrieve("q", top_k=3)

        assert result["confidence"] == "low"
        assert "未在语料中找到相关证据" in result["answer"]
        assert result["matched_docs"] == []
        assert "诚实拒答" in result["note"]
        gen.assert_not_called()  # 无证据 → 不生成、不编造


# ---------------------------------------------------------------------------
# 6. 延迟预算（[7.5]c）
# ---------------------------------------------------------------------------


class TestLatencyBudget:
    @pytest.mark.asyncio
    async def test_tiny_latency_budget_stops_before_round2(self, router):
        router._act_tree_search = AsyncMock(return_value=_act_outcome())
        router._main_funcs = {"generate_answer": MagicMock(return_value="R1ANS")}

        loop = _loop(router)
        loop.max_latency_ms = 1  # 极小预算：已用时 + 预估必然超
        result = await loop.retrieve(
            "q", top_k=3,
            first_round_fused=_fused_pool(), first_round_ctx_state=_r1_ctx_state(),
        )

        assert result["confidence"] == "low"
        assert "尽力作答" in result["note"]
        assert result["rounds_used"] == 1
        assert result["answer"] == "R1ANS"  # 轮 1 状态直接兜底（未重跑 Act）
        router._act_tree_search.assert_not_awaited()


# ---------------------------------------------------------------------------
# 7. 单轮超时 → 有界降级（[7.5]c）
# ---------------------------------------------------------------------------


class TestRoundTimeout:
    @pytest.mark.asyncio
    async def test_hung_round2_downgrades_to_best_effort_bounded(self, router):
        async def hang(query, candidates, node_matches=None):
            await asyncio.sleep(5)
            return _act_outcome()

        router._act_tree_search = hang
        router.verifier.verify = MagicMock(return_value=VerifyResult(0.9, "answer"))
        router._main_funcs = {"generate_answer": MagicMock(return_value="R1ANS")}

        loop = _loop(router)
        loop.round_timeout_s = 0.05

        t0 = time.monotonic()
        result = await loop.retrieve(
            "q", top_k=3,
            first_round_fused=_fused_pool(), first_round_ctx_state=_r1_ctx_state(),
            expand_need=[{"doc_id": "d4", "reason": "缺"}],
        )
        elapsed = time.monotonic() - t0

        assert elapsed < 2.0  # 有界：无死等
        assert result["confidence"] == "low"
        assert "尽力作答" in result["note"]
        assert result["answer"] == "R1ANS"  # 超时轮未产出 → 退回轮 1 状态


# ---------------------------------------------------------------------------
# 8. token 总账天花板（[7.5]b）
# ---------------------------------------------------------------------------


class TestTokenLedgerCap:
    @pytest.mark.asyncio
    async def test_over_budget_stops_before_next_round(self, router):
        _setup_route(router, n_docs=12)
        router._act_tree_search = AsyncMock(return_value=_act_outcome(
            ctx="x" * 2000, docs=["d1", "d2", "d3"],
        ))
        router.verifier.verify = MagicMock(return_value=VerifyResult(0.5, "expand"))
        router._main_funcs = {"generate_answer": MagicMock(return_value="A" * 2000)}

        loop = _loop(router)
        loop.max_total_tokens = 5  # 轮 1 上下文即爆账
        result = await loop.retrieve("q", top_k=3)

        assert result["confidence"] == "low"
        assert "尽力作答" in result["note"]
        assert result["rounds_used"] == 1
        assert router._act_tree_search.await_count == 1  # 轮 2 未开


# ---------------------------------------------------------------------------
# 9. 组件容错：verifier 缺失/抛错 → 降级 best_effort，不击穿（约束：never crash）
# ---------------------------------------------------------------------------


class TestComponentTolerance:
    @pytest.mark.asyncio
    async def test_verifier_error_degrades_to_best_effort(self, router):
        _setup_route(router)
        router._act_tree_search = AsyncMock(return_value=_act_outcome(docs=["d1"]))
        router.verifier.verify = MagicMock(side_effect=RuntimeError("verifier exploded"))
        router._main_funcs = {"generate_answer": MagicMock(return_value="ANS")}

        result = await _loop(router).retrieve("q", top_k=3)

        assert result["confidence"] == "low"
        assert "尽力作答" in result["note"]
        assert result["rounds_used"] == 1

    @pytest.mark.asyncio
    async def test_generator_exception_contained_to_best_effort(self, router):
        """回归：生成器抛错不得击穿 retrieve()——已有接地证据时降级为形态完整的 best_effort。"""
        _setup_route(router)
        router._act_tree_search = AsyncMock(return_value=_act_outcome(docs=["d1", "d2", "d3"]))
        router.verifier.verify = MagicMock(return_value=VerifyResult(0.9, "answer"))
        router._main_funcs = {
            "generate_answer": MagicMock(side_effect=RuntimeError("generator exploded"))
        }

        result = await _loop(router).retrieve("q", top_k=3)

        assert BASE_KEYS <= set(result.keys())
        assert result["confidence"] == "low"
        assert "尽力作答" in result["note"]
        assert result["rounds_used"] == 1
        assert result["answer"] == ""                     # 兜底接地同样生成失败 → 空答案仍是良构响应
        assert [m["doc_id"] for m in result["matched_docs"]] == ["d1", "d2", "d3"]
        router.verifier.verify.assert_not_called()        # 生成先于校验失败，不再触发校验


# ---------------------------------------------------------------------------
# 10b. node_matches 续接转发（续接轮不再以空 node_matches 召回）
# ---------------------------------------------------------------------------


class TestNodeMatchesForwarding:
    @pytest.mark.asyncio
    async def test_continuation_adopts_first_round_node_matches(self, router):
        """续接模式承接调用方轮 1 的 node_matches → 轮 2 树搜索复用节点命中。"""
        nm = {"d5": [{"node_id": "n5", "keyword": "kw", "context": "c"}]}
        seen_matches = []

        async def fake_act(query, candidates, node_matches=None):
            seen_matches.append(dict(node_matches or {}))
            return _act_outcome(docs=candidates[:1])

        router._act_tree_search = fake_act
        router.verifier.verify = MagicMock(return_value=VerifyResult(0.9, "answer"))
        router._main_funcs = {"generate_answer": MagicMock(return_value="ANS")}

        result = await _loop(router).retrieve(
            "q", top_k=3,
            first_round_fused=_fused_pool(), first_round_ctx_state=_r1_ctx_state(),
            first_round_node_matches=nm,
            expand_need=[{"doc_id": "d4", "reason": "缺"}],
        )

        assert result["confidence"] == "high"
        assert result["rounds_used"] == 2
        # 轮 2 Act 拿到轮 1 节点命中（未修复处此处为 {} → 节点召回弱化）
        assert seen_matches == [nm]


# ---------------------------------------------------------------------------
# 10c. 续接池复用：仅给 fused（ctx_state=None）→ 以调用方融合池起轮 1
# ---------------------------------------------------------------------------


class TestContinuationPoolReuse:
    @pytest.mark.asyncio
    async def test_first_round_fused_without_ctx_state_starts_at_round1(self, router):
        """仅给 first_round_fused（ctx_state=None）⇒ start_round=1：
        用调用方融合池切轮 1 候选，不自行 Plan/Route。"""
        act_calls = []

        async def fake_act(query, candidates, node_matches=None):
            act_calls.append(list(candidates))
            return _act_outcome(docs=candidates[:1])

        router._act_tree_search = fake_act
        # 若误走自行 Route，空 docs_info 会打空融合池 → 结果必然不是 high
        router._build_docs_info = MagicMock(return_value=[])
        router.planner.plan = AsyncMock(return_value=PlanResult(
            queries=["q"], weights={"metadata": 1.0}, query_type="factual"
        ))
        router.verifier.verify = MagicMock(return_value=VerifyResult(0.9, "answer"))
        router._main_funcs = {"generate_answer": MagicMock(return_value="ANS")}

        result = await _loop(router).retrieve("q", top_k=3, first_round_fused=_fused_pool())

        assert result["confidence"] == "high"
        assert result["rounds_used"] == 1
        assert act_calls == [["d1", "d2", "d3"]]   # 调用方 fused[:top_k]
        router.planner.plan.assert_not_awaited()
        router._build_docs_info.assert_not_called()


# ---------------------------------------------------------------------------
# 10d. 小工具鲁棒性：非有限分数过滤 + 非法配置容忍
# ---------------------------------------------------------------------------


class TestHelperRobustness:
    def test_normalize_fused_drops_non_finite_scores(self):
        norm = _recall_loop_mod().AgenticRecallLoop._normalize_fused
        out = norm([
            ("d1", 0.5), ("d2", float("nan")), ("d3", float("inf")),
            ("d4", "-inf"), ("d5", "0.25"), "bad", ("d6",),
        ])
        assert out == [("d1", 0.5), ("d5", 0.25)]

    def test_named_candidates_extracts_doc_id_subset(self):
        """[S8] _named_candidates 最小实现：只取 doc_id 子集（去重、保序、排除已召回），
        node_id 条目/非 dict/无 doc_id 跳过，page 不参与文档级补召回。"""
        named = _recall_loop_mod().AgenticRecallLoop._named_candidates
        need = [
            {"doc_id": "d2", "reason": "缺该文档"},
            {"node_id": "n9", "reason": "缺节点"},          # 无 doc_id → 跳过
            {"doc_id": "d2", "reason": "重复"},              # 去重
            {"doc_id": "d4", "page": 3, "reason": "缺页"},   # page 不参与文档级补召回
            "bad",                                            # 非 dict → 跳过
            {"reason": "既无 doc_id 也无 node_id"},          # 无对象键 → 跳过
        ]
        assert named(need, retrieved={"d1"}) == ["d2", "d4"]

    def test_named_candidates_coerces_numeric_doc_id_to_str(self):
        """[Fix] LLM 回数字 doc_id（如 5）→ 强转字符串，避免 int 流入 List[str]，
        且数字版与其字符串版（"5"）判重不重复召回。"""
        named = _recall_loop_mod().AgenticRecallLoop._named_candidates
        need = [
            {"doc_id": 5, "reason": "缺该文档"},           # 数字 doc_id
            {"doc_id": "5", "reason": "同一文档字符串版"},  # 强转后与上条判重
            {"doc_id": "d7", "reason": "缺"},
            {"doc_id": "", "reason": "空串跳过"},           # 空串跳过
        ]
        out = named(need, retrieved={"d1"})
        assert out == ["5", "d7"]
        assert all(isinstance(x, str) for x in out)

    def test_load_settings_tolerates_malformed_values(self, monkeypatch):
        """非法配置值逐字段回退默认——回归：不得从 __init__ 抛出。"""
        import pageindex_mutil.utils as utils_mod

        class _BadCfg:
            agentic_max_rounds = "three"     # → ValueError
            agentic_max_latency_ms = "fast"  # → ValueError
            # agentic_round_timeout_s 缺失 → 默认
            agentic_max_total_tokens = []    # int([]) → TypeError

        class _FakeLoader:
            def load(self, path):
                return _BadCfg()

        monkeypatch.setattr(utils_mod, "ConfigLoader", _FakeLoader)
        rl = _recall_loop_mod()
        loop = rl.AgenticRecallLoop(MagicMock())

        assert loop.max_rounds == rl.AgenticRecallLoop.DEFAULT_MAX_ROUNDS
        assert loop.max_latency_ms == rl.AgenticRecallLoop.DEFAULT_MAX_LATENCY_MS
        assert loop.round_timeout_s == rl.AgenticRecallLoop.DEFAULT_ROUND_TIMEOUT_S
        assert loop.max_total_tokens == rl.AgenticRecallLoop.DEFAULT_MAX_TOTAL_TOKENS


# ---------------------------------------------------------------------------
# 11. 回归：_search_v2 expand 委派（旧 pages_with_text2.items() AttributeError）
# ---------------------------------------------------------------------------


class TestSearchV2ExpandIntegration:
    async def _expand_router(self, router, n_docs=12, need=None):
        router.planner.plan = AsyncMock(return_value=PlanResult(
            queries=["q"], weights={"metadata": 1.0}, query_type="factual"
        ))
        router._build_docs_info = MagicMock(
            return_value=[{"doc_id": f"d{i + 1}"} for i in range(n_docs)]
        )
        router._run_strategies = AsyncMock(return_value=(
            {"metadata": [(f"d{i + 1}", i + 1) for i in range(n_docs)]},
            {},
        ))
        router._act_tree_search = AsyncMock(return_value=_act_outcome(docs=["d1", "d2", "d3"]))
        router.verifier.verify = MagicMock(return_value=VerifyResult(0.5, "expand", need=need or []))
        router._main_funcs = {"generate_answer": MagicMock(return_value="ANS")}

    @pytest.mark.asyncio
    async def test_expand_path_executes_without_attribute_error(self, router):
        """[S8] expand 分支按 v.need 点名委派——旧实现在此抛 pages_with_text2 AttributeError。"""
        await self._expand_router(
            router, n_docs=30,
            need=[{"doc_id": "d5", "reason": "缺"}, {"doc_id": "d9", "reason": "缺"}],
        )
        act_calls = []

        async def fake_act(query, candidates, node_matches=None):
            act_calls.append(list(candidates))
            return _act_outcome(docs=candidates[:1])

        router._act_tree_search = fake_act

        result = await router._search_v2("q", top_k=3)

        assert BASE_KEYS <= set(result.keys())
        assert result["mode"] == "multi"
        assert result["confidence"] == "low"  # 点名耗尽 → best_effort
        assert result["rounds_used"] == 2
        # 轮 1 (v2) d1-d3；轮 2 只补点名 d5,d9；轮 3 无新点名 → no_target → best_effort
        assert act_calls[0] == ["d1", "d2", "d3"]
        assert act_calls[1] == ["d5", "d9"]
        assert act_calls[2] == ["d1", "d2", "d3", "d5", "d9"]  # best_effort 累积池再选择

    @pytest.mark.asyncio
    async def test_expand_delegation_forwards_node_matches(self, router):
        """Route 拿到节点命中 → expand 委派随 first_round_node_matches 传入，续接轮召回不弱化。"""
        await self._expand_router(router, n_docs=30, need=[{"doc_id": "d5", "reason": "缺"}])
        nm = {"d5": [{"node_id": "n5", "keyword": "kw", "context": "c"}]}
        router._run_strategies = AsyncMock(return_value=(
            {"metadata": [(f"d{i + 1}", i + 1) for i in range(30)]},
            nm,
        ))
        seen_matches = []
        real_act = router._act_tree_search

        async def fake_act(query, candidates, node_matches=None):
            seen_matches.append(dict(node_matches or {}))
            return await real_act(query, candidates, node_matches=node_matches)

        router._act_tree_search = fake_act

        result = await router._search_v2("q", top_k=3)

        assert result["rounds_used"] == 2
        # 轮 1 Act（v2 本体）+ 循环轮 2（点名 d5）+ best_effort 接地，全部携带 node_matches
        assert seen_matches == [nm] * 3

    @pytest.mark.asyncio
    async def test_expand_not_delegated_when_pool_exhausted(self, router):
        """融合池已吃满（无文档可扩召）→ 保持原 medium 响应，不进循环。"""
        await self._expand_router(router, n_docs=2)
        result = await router._search_v2("q", top_k=3)
        assert result["confidence"] == "medium"
        assert result["answer"] == "ANS"
        assert "rounds_used" not in result

    @pytest.mark.asyncio
    async def test_loop_failure_falls_back_to_round1_answer(self, router):
        """循环自身异常不得击穿搜索——回退轮 1 medium 响应。"""
        await self._expand_router(router)
        boom = _recall_loop_mod().AgenticRecallLoop

        class _BoomLoop(boom):
            async def retrieve(self, *args, **kwargs):
                raise RuntimeError("loop exploded")

        import pageindex_mutil.agentic.recall_loop as rl_mod
        router_mod = _router_mod()
        # _search_v2 在分支内惰性 `from .recall_loop import AgenticRecallLoop`
        rl_mod.AgenticRecallLoop = _BoomLoop
        try:
            result = await router._search_v2("q", top_k=3)
        finally:
            rl_mod.AgenticRecallLoop = boom

        assert result["confidence"] == "medium"
        assert result["answer"] == "ANS"
