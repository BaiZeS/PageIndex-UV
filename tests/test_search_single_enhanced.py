"""T6.2 _search_single 接入 enhance_and_select 测试（spec [3.4]/[3.4.1]/[3.2.1]）。

验收覆盖：
1. 选择顺序来自 enhance_and_select（LLM 精挑序），无 len(summary) 重排；
2. matched_docs score = 选择覆盖度（selected/candidates），无硬编码 1.0；
   confidence: 无 pool_concern → high，有 → medium；
3. 浴血值类盲区：节点签名关键词作为证据进入精挑 prompt，LLM 选中后
   答案上下文取自该节点正文（真实 enhance 管线 + mocked LLM）；
4. 多范围取数不丢段：LLM 选两节点 → 上下文含两段正文（[3.4.1]①③）；
5. pool_concern + deferred → 放宽 max_candidates 重选（断言第二次调用参数）；
   pool_concern 但无 deferred → 不重选，confidence medium；
6. profiles 解析序：DB node_profiles 优先（经 _id_mapper 的 db 整数 id）；
   无 db → structure 节点字典键兜底；全无 → 空证据仍可端到端工作；
7. NFR4：enhancer 以 (self.model, retrieve_model=self.retrieve_model) 构造；
8. 响应形状键不变（app/server 消费）。

全部 LLM 调用均 mock —— 无真实 LLM。
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# 测试隔离守卫（与 test_unified_enhance / test_retrieve_model_wiring 同理）：
# 收集期清理预置 stub 并干净加载；运行期由 _real_modules fixture 保证真实
# 模块在场，patch 一律 patch.object(module, ...)（模块对象经惰性访问器取）。
for _mod in list(sys.modules):
    if _mod == "pageindex_mutil" or _mod.startswith("pageindex_mutil."):
        del sys.modules[_mod]
sys.modules.setdefault("PyPDF2", MagicMock())  # client 链顶层导入 PyPDF2

import pageindex_mutil.client  # noqa: F401  首次干净加载
from db import PageIndexDB


# ---------------------------------------------------------------------------
# 公共夹具
# ---------------------------------------------------------------------------
# 重要：test_router 等会在运行期向 sys.modules 注入 pageindex_mutil.* stub
# 模块并残留；_search_single 在调用时经 sys.modules 解析 .agentic.enhance /
# .reasoning。因此每个用例前 purge 残留 stub 并重新加载真实模块（autouse
# fixture _real_modules），patch 目标再经惰性访问器取当前生效模块，保证与被
# 测代码解析到同一对象。


@pytest.fixture(autouse=True)
def _real_modules():
    for _m in list(sys.modules):
        if _m == "pageindex_mutil" or _m.startswith("pageindex_mutil."):
            del sys.modules[_m]
    import pageindex_mutil.client  # noqa: F401
    import pageindex_mutil.agentic.enhance  # noqa: F401
    import pageindex_mutil.reasoning  # noqa: F401
    yield


def _enhance_mod():
    import pageindex_mutil.agentic.enhance as m
    return m


def _reasoning_mod():
    import pageindex_mutil.reasoning as m
    return m


def _client(model="m-model", retrieve_model="r-model"):
    from pageindex_mutil.client import PageIndexClient
    return PageIndexClient(model=model, retrieve_model=retrieve_model)


def _node(nid, title="标题", summary="摘要", text="正文", **extra):
    node = {"node_id": nid, "title": title, "summary": summary, "text": text}
    node.update(extra)
    return node


def _add_doc(client, structure, doc_id="doc1", doc_type="md"):
    client.documents[doc_id] = {
        "doc_name": "test.md",
        "type": doc_type,
        "structure": structure,
        "pages": [],
    }
    return doc_id


def _select_json(selected, pool_concern=False, concern_reason=""):
    return json.dumps({
        "selected_ids": selected,
        "pool_concern": pool_concern,
        "concern_reason": concern_reason,
    })


def _run_search_single(client, query="浴血值怎么获得", doc_id="doc1"):
    return asyncio.run(client._search_single(query, doc_id))


def _patch_generate_answer(return_value="ANSWER"):
    """generate_answer 在 _search_single 内按调用时从 reasoning 模块导入，
    patch 当前生效模块的属性即命中。"""
    return patch.object(
        _reasoning_mod(), "generate_answer",
        MagicMock(return_value=return_value),
    )


def _patch_enhance_llm(**kwargs):
    return patch.object(_enhance_mod(), "llm_completion", **kwargs)


# ---------------------------------------------------------------------------
# 1. 选择顺序：LLM 精挑序，无 len(summary) 重排（[3.4]）
# ---------------------------------------------------------------------------


class TestSelectionOrder:
    def test_llm_order_preserved_despite_summary_lengths(self):
        """旧逻辑按 len(summary) 降序 → long 会排前；新逻辑保持 LLM 顺序。"""
        client = _client()
        _add_doc(client, [
            _node("short", summary="s", text="短文本"),
            _node("long", summary="x" * 200, text="长文本"),
        ])
        with _patch_enhance_llm(
            return_value=_select_json(["short", "long"]),
        ), _patch_generate_answer():
            result = _run_search_single(client)
        assert [n["node_id"] for n in result["selected_nodes"]] == ["short", "long"]

    def test_unselected_longer_summary_node_absent(self):
        client = _client()
        _add_doc(client, [
            _node("short", summary="s", text="短文本"),
            _node("long", summary="x" * 200, text="长文本"),
        ])
        with _patch_enhance_llm(
            return_value=_select_json(["short"]),
        ), _patch_generate_answer():
            result = _run_search_single(client)
        assert [n["node_id"] for n in result["selected_nodes"]] == ["short"]


# ---------------------------------------------------------------------------
# 2. score/confidence：选择覆盖度 + pool_concern 派生（[3.4] 移除硬编码）
# ---------------------------------------------------------------------------


class TestScoreAndConfidence:
    def test_score_is_selection_coverage_not_hardcoded(self):
        client = _client()
        _add_doc(client, [_node(f"n{i}") for i in range(4)])
        with _patch_enhance_llm(
            return_value=_select_json(["n0"]),
        ), _patch_generate_answer():
            result = _run_search_single(client)
        # 1 / 4 候选 = 0.25（而非硬编码 1.0）
        assert result["matched_docs"] == [{"doc_id": "doc1", "score": 0.25}]
        assert result["confidence"] == "high"

    def test_score_all_selected_is_one(self):
        client = _client()
        _add_doc(client, [_node("n0"), _node("n1")])
        with _patch_enhance_llm(
            return_value=_select_json(["n0", "n1"]),
        ), _patch_generate_answer():
            result = _run_search_single(client)
        assert result["matched_docs"][0]["score"] == 1.0

    def test_empty_selection_graceful_without_hardcoded_score(self):
        client = _client()
        _add_doc(client, [_node("n0")])
        with _patch_enhance_llm(
            return_value=_select_json([]),
        ), _patch_generate_answer() as mock_answer:
            result = _run_search_single(client)
        assert result["answer"] == "No relevant sections found."
        assert result["confidence"] == "low"
        assert result["matched_docs"] == []  # 无证据接地 → 不虚报匹配
        assert result["selected_nodes"] == []
        mock_answer.assert_not_called()


# ---------------------------------------------------------------------------
# 3. 浴血值类盲区：签名证据进 prompt → LLM 精挑 → 答案接地（真实管线）
# ---------------------------------------------------------------------------


class TestEvidenceGrounding:
    def test_keyword_evidence_reaches_prompt_and_answer_uses_node_text(self):
        """structure 键兜底 profiles（db=None）：浴血值 关键词命中证据进精挑
        prompt；mock LLM 依据证据选 nodeA；答案上下文含 nodeA 正文。"""
        client = _client()
        assert client.db is None
        _add_doc(client, [
            _node("nodeA", title="浴血值获取", summary="浴血值玩法",
                  text="浴血值可以通过完成日常任务获得。",
                  keywords=["浴血值", "日常任务"]),
            _node("nodeB", title="天气", summary="天气系统",
                  text="今天天气不错。", keywords=["天气"]),
        ])

        captured_prompts = []

        def fake_llm(model, prompt, **kwargs):
            captured_prompts.append(prompt)
            # 模拟 LLM：只有看到 浴血值 证据才选 nodeA
            if "浴血值" in prompt:
                return _select_json(["nodeA"])
            return _select_json([])

        with _patch_enhance_llm(side_effect=fake_llm), \
                _patch_generate_answer("答案：日常任务") as mock_answer:
            result = _run_search_single(client, query="浴血值怎么获得")

        # 证据接地：关键词命中进入精挑 prompt
        assert len(captured_prompts) == 1
        assert "关键词命中：浴血值" in captured_prompts[0]
        assert "候选节点 nodeB" not in captured_prompts[0]  # nodeB 无命中不进 union
        # LLM 仍是决策者：选中 nodeA
        assert [n["node_id"] for n in result["selected_nodes"]] == ["nodeA"]
        # 答案由 nodeA 正文合成
        ctx = mock_answer.call_args[0][1]
        assert "浴血值可以通过完成日常任务获得。" in ctx
        assert "今天天气不错。" not in ctx
        assert result["answer"] == "答案：日常任务"

    def test_content_channel_rescues_node_without_signature(self):
        """P2.6 正文内容通道：节点无任何签名（关键词被垃圾词淹没/缺失）但正文含
        查询词 → 进 union，命中词作为关键词命中进 prompt，LLM 选中后正文进答案。"""
        client = _client()
        assert client.db is None
        # nodeA 无 keywords/entities/tags——签名完全缺失，正文是唯一接地
        _add_doc(client, [
            _node("nodeA", title="获取方式", summary="获取方式",
                  text="浴血值可以通过完成日常任务获得。"),
            _node("nodeB", title="天气", summary="天气", text="今天天气不错。"),
        ])

        captured_prompts = []

        def fake_llm(model, prompt, **kwargs):
            captured_prompts.append(prompt)
            return _select_json(["nodeA"]) if "浴血" in prompt else _select_json([])

        with _patch_enhance_llm(side_effect=fake_llm), \
                _patch_generate_answer("答案：日常任务") as mock_answer:
            result = _run_search_single(client, query="浴血值怎么获得")

        assert len(captured_prompts) == 1
        assert "候选节点 nodeA" in captured_prompts[0]
        assert "候选节点 nodeB" not in captured_prompts[0]  # 无命中不进 union
        kw_block = captured_prompts[0].split("候选节点 nodeA：", 1)[1].split("候选节点", 1)[0]
        assert "关键词命中" in kw_block and "浴血" in kw_block
        assert [n["node_id"] for n in result["selected_nodes"]] == ["nodeA"]
        ctx = mock_answer.call_args[0][1]
        assert "浴血值可以通过完成日常任务获得。" in ctx
        assert result["answer"] == "答案：日常任务"

    def test_candidates_without_text_unchanged(self):
        """后向兼容：节点无 text 字段时内容通道不生效，行为与之前一致。"""
        client = _client()
        # nodeA 无 text（PDF 风格节点），仅标题/摘要提到查询词；无签名 → 无命中通道
        _add_doc(client, [
            _node("nodeA", title="声望值获取", summary="声望值获取方式", text=""),
            _node("nodeB", title="声望", summary="声望", text="", keywords=["声望"]),
        ])

        captured_prompts = []

        def fake_llm(model, prompt, **kwargs):
            captured_prompts.append(prompt)
            return _select_json(["nodeA", "nodeB"])

        with _patch_enhance_llm(side_effect=fake_llm), \
                _patch_generate_answer():
            result = _run_search_single(client, query="声望值")

        # nodeB 经关键词通道进 union；nodeA 无 text 无签名 → 不进 union
        # （nodeB 已使 union 非空，零信号兜底不触发）
        assert "候选节点 nodeB" in captured_prompts[0]
        assert "候选节点 nodeA" not in captured_prompts[0]
        assert [n["node_id"] for n in result["selected_nodes"]] == ["nodeB"]


# ---------------------------------------------------------------------------
# 4. 多范围取数不丢段（[3.4.1]①③：跨段融合）
# ---------------------------------------------------------------------------


class TestMultiSpan:
    def test_two_selected_nodes_both_reach_answer_context(self):
        client = _client()
        _add_doc(client, [
            _node("n1", title="上篇", text="第一段证据：规则说明。"),
            _node("n2", title="下篇", text="第二段证据：数值表格。"),
            _node("n3", title="无关", text="无关内容。"),
        ])
        with _patch_enhance_llm(
            return_value=_select_json(["n1", "n2"]),
        ), _patch_generate_answer() as mock_answer:
            result = _run_search_single(client)
        assert [n["node_id"] for n in result["selected_nodes"]] == ["n1", "n2"]
        ctx = mock_answer.call_args[0][1]
        assert "第一段证据：规则说明。" in ctx
        assert "第二段证据：数值表格。" in ctx
        assert "无关内容。" not in ctx

    def test_generate_answer_prompt_has_cross_span_fusion_guidance(self):
        """[3.4.1]②：答案合成 prompt 显式引导综合多处证据。"""
        client = _client()
        _add_doc(client, [_node("n1", text="证据甲。"), _node("n2", text="证据乙。")])
        with _patch_enhance_llm(
            return_value=_select_json(["n1", "n2"]),
        ), patch.object(_reasoning_mod(), "get_llm_client") as mock_client:
            mock_llm_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "综合答案"
            mock_llm_client.chat.completions.create.return_value = mock_resp
            mock_client.return_value = mock_llm_client
            result = _run_search_single(client)
        prompt = mock_llm_client.chat.completions.create.call_args[1]["messages"][1]["content"]
        assert "如证据分布在多个段落，请综合多处证据作答" in prompt
        assert "证据甲。" in prompt and "证据乙。" in prompt
        assert result["answer"] == "综合答案"


# ---------------------------------------------------------------------------
# 5. pool_concern 重选（[3.2.1]：放宽 max_candidates）
# ---------------------------------------------------------------------------


def _instrumented_enhancer(select_results):
    """真实 UnifiedNodeEnhancement + 记录式 enhance_and_select 替身。"""
    from pageindex_mutil.agentic.enhance import UnifiedNodeEnhancement
    enh = UnifiedNodeEnhancement("m-model", retrieve_model="r-model")
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


class TestPoolConcernRetry:
    def test_retry_relaxes_max_candidates_keeps_candidates(self):
        from pageindex_mutil.client import POOL_CONCERN_RETRY_CAP_MULTIPLIER
        client = _client()
        _add_doc(client, [_node("n0"), _node("n2")])
        results = [
            {"selected_ids": ["n0"], "pool_concern": True,
             "concern_reason": "疑似漏掉分支", "deferred": ["n2"]},
            {"selected_ids": ["n0", "n2"], "pool_concern": False,
             "concern_reason": "", "deferred": []},
        ]
        enh, calls = _instrumented_enhancer(results)
        with patch.object(_enhance_mod(), "UnifiedNodeEnhancement",
                          lambda model, retrieve_model=None: enh), \
                _patch_generate_answer():
            result = _run_search_single(client, query="两处证据")

        assert len(calls) == 2
        # 第一次按配置上限；第二次放宽 ×POOL_CONCERN_RETRY_CAP_MULTIPLIER
        assert calls[0]["max_candidates"] is None
        assert calls[1]["max_candidates"] == (
            enh.union_max_candidates * POOL_CONCERN_RETRY_CAP_MULTIPLIER
        )
        # 候选/签名不变（deferred 经 union 自然回池）
        assert calls[1]["candidates"] is calls[0]["candidates"]
        assert calls[1]["profiles"] == calls[0]["profiles"]
        # 重选结果生效：n2 回池并被选中；pool_concern 解除 → high
        assert [n["node_id"] for n in result["selected_nodes"]] == ["n0", "n2"]
        assert result["confidence"] == "high"

    def test_full_pool_retry_when_pool_concern_and_deferred_empty(self):
        """P2.6：pool_concern 且 deferred 为空 → force_all_candidates=True 全量
        重选一次；第二次结果生效；至多重试一次（无循环）。"""
        client = _client()
        _add_doc(client, [_node("n0"), _node("n1")])
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
                          lambda model, retrieve_model=None: enh), \
                _patch_generate_answer():
            result = _run_search_single(client)
        assert len(calls) == 2  # 至多一次重试，无循环
        assert calls[0]["force_all_candidates"] is False
        assert calls[1]["force_all_candidates"] is True
        assert calls[1]["max_candidates"] is None  # 全池重选不抬 cap，走全量直通
        assert calls[1]["candidates"] is calls[0]["candidates"]
        assert calls[1]["profiles"] == calls[0]["profiles"]
        assert [n["node_id"] for n in result["selected_nodes"]] == ["n1"]
        assert result["confidence"] == "high"  # 重选后 pool_concern 解除

    def test_full_pool_retry_not_repeated_when_still_concerned(self):
        """全池重选后仍 pool_concern → 不再重试（至多一次）。"""
        client = _client()
        _add_doc(client, [_node("n0")])
        results = [
            {"selected_ids": [], "pool_concern": True,
             "concern_reason": "关键概念无命中", "deferred": []},
            {"selected_ids": ["n0"], "pool_concern": True,
             "concern_reason": "仍偏弱", "deferred": []},
        ]
        enh, calls = _instrumented_enhancer(results)
        with patch.object(_enhance_mod(), "UnifiedNodeEnhancement",
                          lambda model, retrieve_model=None: enh), \
                _patch_generate_answer():
            result = _run_search_single(client)
        assert len(calls) == 2
        assert result["confidence"] == "medium"  # pool_concern 留存 → medium

    def test_retry_still_concerned_yields_medium(self):
        client = _client()
        _add_doc(client, [_node("n0"), _node("n2")])
        results = [
            {"selected_ids": ["n0"], "pool_concern": True,
             "concern_reason": "x", "deferred": ["n2"]},
            {"selected_ids": ["n0", "n2"], "pool_concern": True,
             "concern_reason": "仍偏弱", "deferred": []},
        ]
        enh, calls = _instrumented_enhancer(results)
        with patch.object(_enhance_mod(), "UnifiedNodeEnhancement",
                          lambda model, retrieve_model=None: enh), \
                _patch_generate_answer():
            result = _run_search_single(client)
        assert len(calls) == 2
        assert result["confidence"] == "medium"


# ---------------------------------------------------------------------------
# 6. profiles 解析序：DB 优先 → structure 键兜底 → 空证据可用（[3.4]）
# ---------------------------------------------------------------------------


class TestProfileResolution:
    def test_db_profiles_preferred_over_structure_keys(self):
        client = _client()
        doc_id = _add_doc(client, [
            _node("n1", keywords=["浴血值"]),  # structure 键：不应被采用
        ])
        fake_db = MagicMock()
        fake_db.get_node_profiles.return_value = [
            {"node_id": "n1", "entities": [], "keywords": ["声望"], "tags": []},
        ]
        fake_db.search_entities.return_value = []
        client.db = fake_db
        client._id_mapper.register(doc_id, 7)

        captured = []

        def fake_llm(model, prompt, **kwargs):
            captured.append(prompt)
            return _select_json(["n1"])

        with _patch_enhance_llm(side_effect=fake_llm), \
                _patch_generate_answer():
            result = _run_search_single(client, query="声望值")

        fake_db.get_node_profiles.assert_called_once_with(7)  # db 整数 id
        # 采用 DB 签名（声望），structure 键（浴血值）被忽略
        assert "关键词命中：声望" in captured[0]
        assert "浴血值" not in captured[0]
        assert result["selected_nodes"][0]["node_id"] == "n1"

    def test_real_db_node_profiles_end_to_end(self):
        """真实 PageIndexDB：node_profiles 表供证，entities 表（空）经
        resolve_query_entities 真实路径解析（真 enhance 管线）。"""
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            db = PageIndexDB(db_path)
            db_doc_id = db.insert_document("test.md", "/tmp/test.md")
            client = _client()
            client.db = db
            doc_id = _add_doc(client, [_node("n1", title="浴血值获取",
                                             text="浴血值通过日常任务获得。")])
            client._id_mapper.register(doc_id, db_doc_id)
            db.upsert_node_profiles(db_doc_id, [
                {"node_id": "n1", "entities": [], "keywords": ["浴血值"], "tags": []},
            ])

            captured = []

            def fake_llm(model, prompt, **kwargs):
                captured.append(prompt)
                return _select_json(["n1"]) if "浴血值" in prompt else _select_json([])

            with _patch_enhance_llm(side_effect=fake_llm), \
                    _patch_generate_answer() as mock_answer:
                result = _run_search_single(client, query="浴血值怎么获得")

            assert captured, "enhance pipeline was not invoked"
            assert "关键词命中：浴血值" in captured[0]
            assert [n["node_id"] for n in result["selected_nodes"]] == ["n1"]
            assert "浴血值通过日常任务获得。" in mock_answer.call_args[0][1]
            db.close()
        finally:
            os.unlink(db_path)

    def test_no_profiles_anywhere_empty_evidence_still_works(self):
        """无 db、structure 无签名键 → 空证据零信号全量送 LLM，端到端可用。"""
        client = _client()
        assert client.db is None
        _add_doc(client, [_node("n1", text="纯正文，无签名。")])
        with _patch_enhance_llm(
            return_value=_select_json(["n1"]),
        ), _patch_generate_answer() as mock_answer:
            result = _run_search_single(client, query="任意问题")
        assert [n["node_id"] for n in result["selected_nodes"]] == ["n1"]
        assert "纯正文，无签名。" in mock_answer.call_args[0][1]
        assert result["confidence"] == "high"


# ---------------------------------------------------------------------------
# 7. NFR4 接线 + 响应形状
# ---------------------------------------------------------------------------


class TestNFR4AndShape:
    def test_enhancer_constructed_with_model_and_retrieve_model(self):
        client = _client(model="m-model", retrieve_model="r-model")
        _add_doc(client, [_node("n0")])
        ctor_calls = []
        UnifiedNodeEnhancement = _enhance_mod().UnifiedNodeEnhancement

        class SpyEnhancer(UnifiedNodeEnhancement):
            def __init__(self, model, retrieve_model=None):
                ctor_calls.append((model, retrieve_model))
                super().__init__(model, retrieve_model=retrieve_model)

            async def enhance_and_select(self, *args, **kwargs):
                return {"selected_ids": ["n0"], "pool_concern": False,
                        "concern_reason": "", "deferred": []}

        with patch.object(_enhance_mod(), "UnifiedNodeEnhancement", SpyEnhancer), \
                _patch_generate_answer():
            _run_search_single(client)
        assert ctor_calls == [("m-model", "r-model")]

    def test_response_shape_keys_unchanged_via_search_dispatch(self):
        client = _client()
        _add_doc(client, [_node("n0", start_index=0, end_index=1)])
        with _patch_enhance_llm(
            return_value=_select_json(["n0"]),
        ), _patch_generate_answer():
            result = asyncio.run(client.search("q"))  # 单文档 → _search_single
        assert set(result.keys()) == {
            "query", "mode", "answer", "confidence",
            "matched_docs", "selected_nodes", "pages",
        }
        assert result["mode"] == "single"
        assert result["selected_nodes"][0]["node_id"] == "n0"
