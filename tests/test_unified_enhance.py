"""P2 UnifiedNodeEnhancement.enhance_and_select 测试（spec [3.2]/[1.1]/[1.2]/[7.3]-[7.7]）。

验收覆盖：
1. 四通道 union 各通道召回（keyword/tag/entity 命中各自带入节点；纯 Python，无 LLM）；
2. union 为空 → 全量候选送 LLM（零信号不裁剪，[1.1]）；
3. cap 超限 → 多信号加权收缩 + deferred 集合正确；全零分 → 禁止按 score 收缩，
   输入顺序 + 绝对上限兜底（[1.2] 零值泛滥防护）；
4. 多信号加权：多命中节点排在单命中节点之前；
5. 单节点证据封顶（实体≤3、关键词≤5、标签≤2）；全局注记去重（命中 >3 节点）
   且只保留在最高分节点展开（[7.4]）；
6. 跨节点总预算超限 → 最弱候选退化为"标题+摘要"一行（强候选保留富证据）；
7. prompt 含 [3.2.2] 指引行 + pool_concern 三条判据（原文）+ 预算块（给预算时）；
8. JSON passthrough（selected_ids/pool_concern/concern_reason）；
   非法 selected_ids（未知 id/重复）被过滤；空选择合法（宁缺毋滥）；
9. LLM 失效降级：不做启发式裁剪，放行 union 全部候选（[7.7]）；
10. NFR4 retrieve_model 接线；async API 可经 asyncio.run 使用。
11. 审查修复：全局注记按 distinct 节点计数（大小写变体/重复条目不得重复计入，
    注记文本节点 id 无重复）；cap 钳位 ≥1 且非法配置逐级回退（永不抛出）；
    平分确定性裁决；顶层 list JSON 降级；query_entities=[] ≡ None；
    非正预算视为未给；不可哈希 node_id 不抛出。

全部 LLM 调用均 mock —— 无真实 LLM。
"""
import asyncio
import inspect
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# 测试隔离守卫（与 test_retrieve_model_wiring / test_entity_disambiguation 同理）：
# test_router 等会在运行期向 sys.modules 预置 pageindex_mutil.* stub 模块；此处
# 清理预置 stub、导入真实模块并持有模块对象引用，patch 一律用 patch.object(module, ...)，
# 保证命中被测类实际引用的模块。
for _mod in list(sys.modules):
    if _mod == "pageindex_mutil" or _mod.startswith("pageindex_mutil."):
        del sys.modules[_mod]

import pageindex_mutil.agentic.enhance as enhance_mod
from pageindex_mutil.agentic.enhance import (
    UnifiedNodeEnhancement,
    WEIGHT_ENTITY,
    WEIGHT_TAG,
    WEIGHT_KEYWORD,
    EVIDENCE_MAX_ENTITIES_PER_NODE,
    EVIDENCE_MAX_KEYWORDS_PER_NODE,
    EVIDENCE_MAX_TAGS_PER_NODE,
    GLOBAL_NOTE_THRESHOLD,
    ABSOLUTE_CEILING_MULTIPLIER,
)


def _cand(nid, title="标题", summary="摘要"):
    return {"node_id": nid, "title": title, "summary": summary}


def _resp(selected, pool_concern=False, concern_reason=""):
    return json.dumps({
        "selected_ids": selected,
        "pool_concern": pool_concern,
        "concern_reason": concern_reason,
    })


def _enhancer(model="m", retrieve_model=None, cap=None, budget=None):
    enh = UnifiedNodeEnhancement(model, retrieve_model=retrieve_model)
    if cap is not None:
        enh.union_max_candidates = cap
    if budget is not None:
        enh.evidence_max_chars = budget
    return enh


def _select_call(enh, return_value=None, side_effect=None, **call_kwargs):
    """Run enhance_and_select with mocked llm_completion; return (result, mock)."""
    kwargs = {}
    if side_effect is not None:
        kwargs["side_effect"] = side_effect
    else:
        kwargs["return_value"] = return_value if return_value is not None else ""
    with patch.object(enhance_mod, "llm_completion", **kwargs) as mock_llm:
        result = asyncio.run(enh.enhance_and_select(**call_kwargs))
    return result, mock_llm


def _evidence_section(prompt):
    """Extract the evidence block region of the prompt (between header and guidance)."""
    return prompt.split("候选节点证据：", 1)[1].split("判断指引", 1)[0]


def _prompt_of(mock_llm):
    return mock_llm.call_args[0][1]


# ===========================================================================
# 1. 四通道 union 各通道召回（纯 Python，无 LLM）
# ===========================================================================


class TestUnionChannels:
    """每个通道各自把命中节点带进 union；未命中节点不进 union。"""

    def test_keyword_tag_entity_channels_each_admit_their_node(self):
        candidates = [_cand("n_kw"), _cand("n_tag"), _cand("n_ent"), _cand("n_miss")]
        profiles = {
            "n_kw": {"entities": [], "keywords": ["浴血"], "tags": []},
            "n_tag": {"entities": [], "keywords": [], "tags": ["游戏机制"]},
            "n_ent": {"entities": [{"name": "张三", "type": "person"}], "keywords": [], "tags": []},
            "n_miss": {"entities": [{"name": "气象站", "type": "object"}],
                       "keywords": ["天气"], "tags": ["气象"]},
        }
        enh = _enhancer()
        result, mock_llm = _select_call(
            enh,
            return_value=_resp(["n_kw", "n_tag", "n_ent", "n_miss"]),
            query="浴血怎么获得，游戏机制如何",
            candidates=candidates,
            profiles=profiles,
            query_entities=["张三"],
        )
        # n_miss 不在 union：即使 LLM"选了"也被校验过滤
        assert result["selected_ids"] == ["n_kw", "n_tag", "n_ent"]
        assert result["deferred"] == []
        prompt = _prompt_of(mock_llm)
        for nid in ("n_kw", "n_tag", "n_ent"):
            assert f"候选节点 {nid}" in prompt
        assert "候选节点 n_miss" not in prompt

    def test_keyword_substring_inclusion_for_multi_char_tokens(self):
        """query token 浴血 ⊂ 节点关键词 浴血值 → 命中（jieba 把 浴血值 拆成 浴血+值）。"""
        candidates = [_cand("n1"), _cand("n2")]
        profiles = {
            "n1": {"entities": [], "keywords": ["浴血值"], "tags": []},
            "n2": {"entities": [], "keywords": ["声望"], "tags": []},
        }
        enh = _enhancer()
        result, _ = _select_call(
            enh,
            return_value=_resp(["n1", "n2"]),
            query="浴血值怎么获得",
            candidates=candidates,
            profiles=profiles,
        )
        assert result["selected_ids"] == ["n1"]

    def test_entity_channel_casefold_substring(self):
        """实体通道：casefold + 子串放行。"""
        candidates = [_cand("a"), _cand("b")]
        profiles = {
            "a": {"entities": [{"name": "Albert Einstein", "type": "person"}], "keywords": [], "tags": []},
            "b": {"entities": [{"name": "Newton", "type": "person"}], "keywords": [], "tags": []},
        }
        enh = _enhancer()
        # query entity "einstein" 与 "Albert Einstein" casefold 子串命中
        result, _ = _select_call(
            enh,
            return_value=_resp(["a", "b"]),
            query="相对论的提出者",
            candidates=candidates,
            profiles=profiles,
            query_entities=["einstein"],
        )
        assert result["selected_ids"] == ["a"]


# ===========================================================================
# 2. union 为空 → 全量候选送 LLM（零信号不裁剪）
# ===========================================================================


class TestUnionEmpty:
    def test_zero_signal_sends_all_candidates_to_llm(self):
        candidates = [_cand("n0"), _cand("n1"), _cand("n2")]
        profiles = {
            "n0": {"entities": [], "keywords": ["苹果"], "tags": ["水果"]},
            "n1": {"entities": [{"name": "李四", "type": "person"}], "keywords": [], "tags": []},
        }
        enh = _enhancer()
        result, mock_llm = _select_call(
            enh,
            return_value=_resp(["n0", "n1", "n2"]),
            query="天马流星拳",
            candidates=candidates,
            profiles=profiles,
        )
        # 全量放行：三个 id 都在 union 内，校验全部通过
        assert result["selected_ids"] == ["n0", "n1", "n2"]
        assert result["deferred"] == []
        prompt = _prompt_of(mock_llm)
        for nid in ("n0", "n1", "n2"):
            assert f"候选节点 {nid}" in prompt

    def test_missing_profile_node_still_selectable_with_empty_evidence(self):
        """[7.7] 签名缺失退化：无 profile 节点空证据仍可被选中。"""
        candidates = [_cand("n1"), _cand("n2")]
        profiles = {"n1": {"entities": [], "keywords": ["苹果"], "tags": []}}  # n2 无 profile
        enh = _enhancer()
        result, mock_llm = _select_call(
            enh,
            return_value=_resp(["n2"]),
            query="天马流星拳",
            candidates=candidates,
            profiles=profiles,
        )
        assert result["selected_ids"] == ["n2"]
        prompt = _prompt_of(mock_llm)
        assert "候选节点 n2" in prompt  # 空证据块仍然呈现（标题+摘要）
        assert "标题：标题" in prompt


# ===========================================================================
# 3. cap 超限收缩 + deferred；零值绝对上限
# ===========================================================================


class TestCapAndDeferred:
    def test_union_over_cap_shrinks_by_score_and_defers_rest(self):
        # 输入序刻意打乱：c(kw=1) b(tag=2) a(ent=3) d(未命中)
        candidates = [_cand("c"), _cand("b"), _cand("a"), _cand("d")]
        profiles = {
            "a": {"entities": [{"name": "张三", "type": "person"}], "keywords": [], "tags": []},
            "b": {"entities": [], "keywords": [], "tags": ["游戏机制"]},
            "c": {"entities": [], "keywords": ["浴血"], "tags": []},
            "d": {"entities": [], "keywords": [], "tags": []},
        }
        enh = _enhancer(cap=2)
        result, mock_llm = _select_call(
            enh,
            return_value=_resp(["a", "b", "c", "d"]),
            query="浴血 游戏机制",
            candidates=candidates,
            profiles=profiles,
            query_entities=["张三"],
        )
        # a(3.0) b(2.0) 保留，c(1.0) 进延迟池，d 本就未进 union
        assert result["selected_ids"] == ["a", "b"]
        assert result["deferred"] == ["c"]
        prompt = _prompt_of(mock_llm)
        assert "候选节点 a" in prompt
        assert "候选节点 b" in prompt
        assert "候选节点 c" not in prompt
        assert "候选节点 d" not in prompt

    def test_zero_signal_absolute_ceiling_in_input_order(self):
        """[1.2] 零值泛滥防护：全零分禁止按 score 收缩 → 输入顺序 + 绝对上限兜底。"""
        candidates = [_cand(f"n{i}") for i in range(6)]
        enh = _enhancer(cap=2)  # 绝对上限 = 2 × 2 = 4
        result, mock_llm = _select_call(
            enh,
            return_value=_resp([f"n{i}" for i in range(6)]),
            query="天马流星拳",
            candidates=candidates,
            profiles={},
        )
        assert result["selected_ids"] == ["n0", "n1", "n2", "n3"]  # 输入顺序准入
        assert result["deferred"] == ["n4", "n5"]
        prompt = _prompt_of(mock_llm)
        for nid in ("n0", "n1", "n2", "n3"):
            assert f"候选节点 {nid}" in prompt
        assert "候选节点 n4" not in prompt
        assert "候选节点 n5" not in prompt

    def test_multi_signal_node_ranks_above_single_hit(self):
        """多信号加权：entity+keyword(4.0) > tag(2.0) > keyword(1.0)。"""
        candidates = [_cand("kw_only"), _cand("tag_only"), _cand("multi")]
        profiles = {
            "multi": {"entities": [{"name": "张三", "type": "person"}],
                      "keywords": ["浴血"], "tags": []},
            "tag_only": {"entities": [], "keywords": [], "tags": ["游戏机制"]},
            "kw_only": {"entities": [], "keywords": ["获得"], "tags": []},
        }
        enh = _enhancer(cap=2)
        result, mock_llm = _select_call(
            enh,
            return_value=_resp(["multi", "tag_only"]),
            query="浴血怎么获得，游戏机制",
            candidates=candidates,
            profiles=profiles,
            query_entities=["张三"],
        )
        assert result["deferred"] == ["kw_only"]
        prompt = _prompt_of(mock_llm)
        # 证据块按分数降序呈现：multi(4.0) 在 tag_only(2.0) 之前
        assert prompt.index("候选节点 multi") < prompt.index("候选节点 tag_only")
        assert "候选节点 kw_only" not in prompt


# ===========================================================================
# 4. 证据组装：单节点封顶 + 全局注记 + 预算退化
# ===========================================================================


class TestEvidenceAssembly:
    def test_per_node_entity_cap(self):
        candidates = [_cand("n1")]
        profiles = {
            "n1": {"entities": [{"name": f"e{i}", "type": "concept"} for i in range(1, 6)],
                   "keywords": [], "tags": []},
        }
        enh = _enhancer()
        _, mock_llm = _select_call(
            enh,
            return_value=_resp(["n1"]),
            query="实体查询",
            candidates=candidates,
            profiles=profiles,
            query_entities=[f"e{i}" for i in range(1, 6)],
        )
        section = _evidence_section(_prompt_of(mock_llm))
        for i in range(1, EVIDENCE_MAX_ENTITIES_PER_NODE + 1):  # 前 3 个
            assert f"e{i}（concept）" in section
        assert "e4" not in section  # 第 4、5 个被封顶
        assert "e5" not in section

    def test_per_node_keyword_cap(self):
        words = ["苹果", "电脑", "手机", "桌子", "椅子", "窗户", "地板"]
        candidates = [_cand("n1")]
        profiles = {"n1": {"entities": [], "keywords": list(words), "tags": []}}
        enh = _enhancer()
        _, mock_llm = _select_call(
            enh,
            return_value=_resp(["n1"]),
            query=" ".join(words),
            candidates=candidates,
            profiles=profiles,
        )
        section = _evidence_section(_prompt_of(mock_llm))
        for w in words[:EVIDENCE_MAX_KEYWORDS_PER_NODE]:  # 前 5 个
            assert w in section
        assert "窗户" not in section  # 第 6、7 个被封顶
        assert "地板" not in section

    def test_per_node_tag_cap(self):
        candidates = [_cand("n1")]
        profiles = {"n1": {"entities": [], "keywords": [], "tags": ["算法", "结构", "网络"]}}
        enh = _enhancer()
        _, mock_llm = _select_call(
            enh,
            return_value=_resp(["n1"]),
            query="算法 结构 网络",
            candidates=candidates,
            profiles=profiles,
        )
        section = _evidence_section(_prompt_of(mock_llm))
        assert "标签命中：算法、结构" in section
        assert "网络" not in section  # 第 3 个标签被封顶

    def test_global_note_dedup_over_threshold_and_keeper_node(self):
        """>GLOBAL_NOTE_THRESHOLD 个节点共享同一实体 → 一条全局注记，
        且只保留在最高分节点逐节点展开。"""
        candidates = [_cand("n0"), _cand("n1"), _cand("n2"), _cand("n3")]
        profiles = {
            nid: {"entities": [{"name": "张三", "type": "person"}], "keywords": [], "tags": []}
            for nid in ("n0", "n1", "n2", "n3")
        }
        profiles["n2"]["keywords"] = ["武功"]  # n2 多一档 → 唯一最高分 keeper
        enh = _enhancer()
        _, mock_llm = _select_call(
            enh,
            return_value=_resp(["n0", "n1", "n2", "n3"]),
            query="武功秘籍",
            candidates=candidates,
            profiles=profiles,
            query_entities=["张三"],
        )
        prompt = _prompt_of(mock_llm)
        section = _evidence_section(prompt)
        assert section.count("注：实体 张三") == 1
        note_line = next(l for l in section.splitlines() if l.startswith("注：实体 张三"))
        for nid in ("n0", "n1", "n2", "n3"):
            assert nid in note_line
        # 逐节点展开只在 keeper(n2) 保留一次
        assert section.count("实体匹配：张三（person）") == 1
        keeper_block = section.split("候选节点 n2：", 1)[1].split("候选节点", 1)[0] \
            if "候选节点 n2：" in section else ""
        assert "张三" in keeper_block

    def test_no_global_note_at_threshold(self):
        """命中节点数 == 阈值（不超过）→ 不做全局注记，各节点保留。"""
        assert GLOBAL_NOTE_THRESHOLD == 3
        candidates = [_cand("n0"), _cand("n1"), _cand("n2")]
        profiles = {
            nid: {"entities": [{"name": "张三", "type": "person"}], "keywords": [], "tags": []}
            for nid in ("n0", "n1", "n2")
        }
        enh = _enhancer()
        _, mock_llm = _select_call(
            enh,
            return_value=_resp(["n0", "n1", "n2"]),
            query="人物查询",
            candidates=candidates,
            profiles=profiles,
            query_entities=["张三"],
        )
        section = _evidence_section(_prompt_of(mock_llm))
        assert "命中于节点" not in section
        assert section.count("实体匹配：张三（person）") == 3

    def test_budget_degrades_weakest_nodes_to_one_line(self):
        """[7.4] 跨节点总预算：最弱候选退化为"标题+摘要"一行，强候选保留富证据。"""
        strong = _cand("strong", title="强节点", summary="强" * 30)
        weak = _cand("weak", title="弱节点", summary="弱" * 30)
        profiles = {
            "strong": {"entities": [{"name": "张三", "type": "person"}], "keywords": [], "tags": []},
            "weak": {"entities": [], "keywords": ["武功"], "tags": []},
        }
        # 全量两块 ≈129 chars；强块(69)+弱单行(50)=119 —— budget 取中间值 125：
        # 只退化最弱节点即可达标，强节点保留完整证据。
        enh = _enhancer(budget=125)
        _, mock_llm = _select_call(
            enh,
            return_value=_resp(["strong", "weak"]),
            query="武功秘籍",
            candidates=[strong, weak],
            profiles=profiles,
            query_entities=["张三"],
        )
        section = _evidence_section(_prompt_of(mock_llm))
        # 弱节点退化为单行（标题与摘要同行，无证据行）
        assert "候选节点 weak：标题：弱节点｜摘要：" in section
        assert "关键词命中" not in section
        # 强节点保留富证据块
        assert "实体匹配：张三（person）" in section
        weak_line = next(l for l in section.splitlines() if l.startswith("候选节点 weak"))
        assert "摘要：" in weak_line  # 单行含标题+摘要


# ===========================================================================
# 5. prompt 内容：指引 + pool_concern 判据 + 预算块
# ===========================================================================


class TestPromptContent:
    def _capture(self, **kwargs):
        candidates = [_cand("n0")]
        profiles = {"n0": {"entities": [], "keywords": ["浴血"], "tags": []}}
        enh = _enhancer()
        call = dict(query="浴血", candidates=candidates, profiles=profiles)
        call.update(kwargs)
        _, mock_llm = _select_call(enh, return_value=_resp(["n0"]), **call)
        return _prompt_of(mock_llm)

    def test_prompt_contains_guidance_criteria_and_schema(self):
        prompt = self._capture()
        # [3.2.2] 指引行
        assert "实体和关键词匹配是语料事实，请优先依据它们与问题的语义关联程度判断，而非简单计数命中个数" in prompt
        # pool_concern 三条判据（原文）+ 判据之外一律 false
        assert "查询里的关键概念（实体/核心关键词）没有命中任何候选的证据" in prompt
        assert "命中的候选数明显偏少（如仅 1 个）且证据偏弱" in prompt
        assert "选中节点间主题互斥/矛盾，疑似漏掉真正的分支" in prompt
        assert "一律 false" in prompt
        # 输出 schema + 宁缺毋滥
        assert "selected_ids" in prompt
        assert "pool_concern" in prompt
        assert "concern_reason" in prompt
        assert "宁缺毋滥" in prompt

    def test_budget_block_present_when_budgets_given(self):
        prompt = self._capture(node_budget=3, token_budget=4000)
        # [7.5]b 预算转 prompt 指令
        assert "本轮最多选 3 个节点" in prompt
        assert "约 4000 token" in prompt
        assert "超出预算优先选证据最充分的" in prompt
        assert "selected_ids 个数不得超过 3" in prompt

    def test_no_budget_block_when_budgets_absent(self):
        prompt = self._capture()
        assert "预算约束" not in prompt

    def test_non_positive_or_invalid_budgets_treated_as_absent(self):
        """#7：node_budget/token_budget ≤0（或非法值）→ 视为未给，不渲染预算块。"""
        for kwargs in (
            dict(node_budget=0),
            dict(token_budget=-1),
            dict(node_budget=0, token_budget=0),
            dict(node_budget="not-a-number"),
        ):
            prompt = self._capture(**kwargs)
            assert "预算约束" not in prompt
            assert "最多选 0 个节点" not in prompt


# ===========================================================================
# 6. JSON passthrough 与校验过滤
# ===========================================================================


class TestSelectParsing:
    def test_json_passthrough_all_fields(self):
        candidates = [_cand("n0"), _cand("n1")]
        profiles = {"n0": {"entities": [], "keywords": ["浴血"], "tags": []},
                    "n1": {"entities": [], "keywords": [], "tags": []}}
        enh = _enhancer()
        result, _ = _select_call(
            enh,
            return_value=_resp(["n0"], pool_concern=True,
                               concern_reason="关键概念X未命中任何候选证据"),
            query="浴血与X",
            candidates=candidates,
            profiles=profiles,
        )
        assert result["selected_ids"] == ["n0"]
        assert result["pool_concern"] is True
        assert result["concern_reason"] == "关键概念X未命中任何候选证据"

    def test_invalid_and_duplicate_ids_filtered(self):
        candidates = [_cand("n0"), _cand("n1")]
        enh = _enhancer()
        result, _ = _select_call(
            enh,
            return_value=json.dumps({"selected_ids": ["n1", "n1", "ghost", "n0"],
                                     "pool_concern": False, "concern_reason": ""}),
            query="天马流星拳",  # 零信号：全部候选进 union
            candidates=candidates,
            profiles={},
        )
        # 去重 + 未知 id 过滤，保留 LLM 给出的相对顺序
        assert result["selected_ids"] == ["n1", "n0"]

    def test_empty_selection_is_valid_not_degradation(self):
        """宁缺毋滥：LLM 明确一个都不选是合法结果，不是失效降级。"""
        candidates = [_cand("n0")]
        profiles = {"n0": {"entities": [], "keywords": ["浴血"], "tags": []}}
        enh = _enhancer()
        result, _ = _select_call(
            enh,
            return_value=_resp([]),
            query="浴血",
            candidates=candidates,
            profiles=profiles,
        )
        assert result["selected_ids"] == []
        assert result["pool_concern"] is False
        assert result["concern_reason"] == ""  # 不是 llm_unavailable

    def test_empty_candidates_short_circuits(self):
        enh = _enhancer()
        with patch.object(enhance_mod, "llm_completion") as mock_llm:
            result = asyncio.run(enh.enhance_and_select("q", [], {}))
        assert result == {"selected_ids": [], "pool_concern": False,
                          "concern_reason": "", "deferred": []}
        mock_llm.assert_not_called()  # 无候选不调 LLM


# ===========================================================================
# 7. LLM 失效降级（[7.7]：放行证据，union 即选中）
# ===========================================================================


class TestDegradation:
    def test_llm_empty_response_passes_union_through(self):
        candidates = [_cand("n_hit"), _cand("n_miss")]
        profiles = {"n_hit": {"entities": [], "keywords": ["浴血"], "tags": []},
                    "n_miss": {"entities": [], "keywords": ["天气"], "tags": []}}
        enh = _enhancer()
        result, _ = _select_call(
            enh,
            return_value="",
            query="浴血",
            candidates=candidates,
            profiles=profiles,
        )
        assert result["selected_ids"] == ["n_hit"]  # union 全部放行
        assert result["pool_concern"] is False
        assert result["concern_reason"] == "llm_unavailable"

    def test_llm_exception_degrades_and_keeps_deferred(self):
        candidates = [_cand("a"), _cand("b"), _cand("c")]
        profiles = {
            "a": {"entities": [{"name": "张三", "type": "person"}], "keywords": [], "tags": []},
            "b": {"entities": [], "keywords": ["浴血"], "tags": []},
            "c": {"entities": [], "keywords": [], "tags": ["游戏机制"]},
        }
        enh = _enhancer(cap=2)
        result, _ = _select_call(
            enh,
            side_effect=RuntimeError("boom"),
            query="浴血 游戏机制",
            candidates=candidates,
            profiles=profiles,
            query_entities=["张三"],
        )
        # union=[a,b,c]，分数 a=3.0(实体) c=2.0(标签) b=1.0(关键词)；
        # cap=2 → 保留 a,c；deferred=[b]；降级路径必须原样透传 deferred
        assert result["selected_ids"] == ["a", "c"]
        assert result["deferred"] == ["b"]
        assert result["pool_concern"] is False
        assert result["concern_reason"] == "llm_unavailable"

    def test_llm_malformed_json_degrades(self):
        candidates = [_cand("n0")]
        profiles = {"n0": {"entities": [], "keywords": ["浴血"], "tags": []}}
        enh = _enhancer()
        result, _ = _select_call(
            enh,
            return_value="这不是JSON",
            query="浴血",
            candidates=candidates,
            profiles=profiles,
        )
        assert result["selected_ids"] == ["n0"]
        assert result["concern_reason"] == "llm_unavailable"

    def test_llm_toplevel_list_json_degrades(self):
        """顶层 JSON 数组（非约定对象）→ 按 LLM 失效降级，放行 union。"""
        candidates = [_cand("n_hit"), _cand("n_miss")]
        profiles = {"n_hit": {"entities": [], "keywords": ["浴血"], "tags": []},
                    "n_miss": {"entities": [], "keywords": ["天气"], "tags": []}}
        enh = _enhancer()
        result, _ = _select_call(
            enh,
            return_value=json.dumps(["n_hit", "n_miss"]),
            query="浴血",
            candidates=candidates,
            profiles=profiles,
        )
        assert result["selected_ids"] == ["n_hit"]
        assert result["pool_concern"] is False
        assert result["concern_reason"] == "llm_unavailable"


# ===========================================================================
# 8. NFR4 retrieve_model 接线 + async API
# ===========================================================================


class TestRetrieveModelWiringNFR4:
    def test_uses_retrieve_model_when_set(self):
        enh = UnifiedNodeEnhancement("m", retrieve_model="r-model")
        _, mock_llm = _select_call(
            enh, return_value=_resp([]),
            query="q", candidates=[_cand("n0")], profiles={},
        )
        assert mock_llm.call_args[0][0] == "r-model"

    def test_falls_back_to_model_when_retrieve_model_none(self):
        enh = UnifiedNodeEnhancement("m", retrieve_model=None)
        _, mock_llm = _select_call(
            enh, return_value=_resp([]),
            query="q", candidates=[_cand("n0")], profiles={},
        )
        assert mock_llm.call_args[0][0] == "m"


class TestAsyncAPIAndConfig:
    def test_enhance_and_select_is_async_runnable_via_asyncio_run(self):
        assert inspect.iscoroutinefunction(UnifiedNodeEnhancement.enhance_and_select)
        enh = _enhancer()
        with patch.object(enhance_mod, "llm_completion", return_value=_resp([])):
            result = asyncio.run(
                enh.enhance_and_select("q", [_cand("n0")], {})
            )
        assert set(result.keys()) == {"selected_ids", "pool_concern", "concern_reason", "deferred"}


# ===========================================================================
# T6.2: max_candidates 上限覆盖（pool_concern 放宽重选用，[3.2.1]）
# ===========================================================================


class TestMaxCandidatesOverride:
    """max_candidates=None → 配置上限；显式传值 → 本次调用覆盖（不改实例配置）。"""

    @staticmethod
    def _fixture(cap):
        candidates = [_cand("a"), _cand("b"), _cand("c"), _cand("d")]
        profiles = {
            "a": {"entities": [{"name": "张三", "type": "person"}], "keywords": [], "tags": []},
            "b": {"entities": [], "keywords": [], "tags": ["游戏机制"]},
            "c": {"entities": [], "keywords": ["浴血"], "tags": []},
            "d": {"entities": [], "keywords": [], "tags": []},
        }
        kwargs = dict(
            query="浴血 游戏机制",
            candidates=candidates,
            profiles=profiles,
            query_entities=["张三"],
        )
        return _enhancer(cap=cap), kwargs

    def test_none_uses_configured_cap(self):
        enh, kwargs = self._fixture(cap=2)
        result, _ = _select_call(
            enh, return_value=_resp(["a", "b", "c"]), **kwargs
        )
        assert result["deferred"] == ["c"]
        assert result["selected_ids"] == ["a", "b"]

    def test_override_raises_cap_for_this_call(self):
        enh, kwargs = self._fixture(cap=2)
        result, mock_llm = _select_call(
            enh, return_value=_resp(["a", "b", "c"]),
            max_candidates=3, **kwargs
        )
        # 上限放宽到 3：c 不再进延迟池，LLM 可选全部 union 成员
        assert result["deferred"] == []
        assert result["selected_ids"] == ["a", "b", "c"]
        prompt = _prompt_of(mock_llm)
        assert "候选节点 c" in prompt

    def test_override_does_not_mutate_instance_config(self):
        enh, kwargs = self._fixture(cap=2)
        _select_call(enh, return_value=_resp([]), max_candidates=3, **kwargs)
        assert enh.union_max_candidates == 2  # 实例配置不变


# ===========================================================================
# T6.2: resolve_query_entities（共享助手；router T6.4 复用）
# ===========================================================================


class TestResolveQueryEntities:
    def test_names_and_aliases_flattened_in_order(self):
        from pageindex_mutil.agentic.enhance import resolve_query_entities
        db = MagicMock()
        db.search_entities.return_value = [
            {"name": "张三", "aliases": '["张生", "张三丰"]'},
            {"name": "李四", "aliases": "[]"},
        ]
        assert resolve_query_entities(db, "张三和李四") == ["张三", "张生", "张三丰", "李四"]
        db.search_entities.assert_called_once_with("张三和李四", limit=5)

    def test_custom_limit_forwarded(self):
        from pageindex_mutil.agentic.enhance import resolve_query_entities
        db = MagicMock()
        db.search_entities.return_value = []
        resolve_query_entities(db, "q", limit=3)
        db.search_entities.assert_called_once_with("q", limit=3)

    def test_casefold_dedup_keeps_first_spelling(self):
        from pageindex_mutil.agentic.enhance import resolve_query_entities
        db = MagicMock()
        db.search_entities.return_value = [
            {"name": "Hero", "aliases": '["hero", "HERO 2", "Hero"]'},
        ]
        assert resolve_query_entities(db, "hero") == ["Hero", "HERO 2"]

    def test_malformed_aliases_json_skipped_not_raised(self):
        from pageindex_mutil.agentic.enhance import resolve_query_entities
        db = MagicMock()
        db.search_entities.return_value = [
            {"name": "A", "aliases": "这不是JSON"},
            {"name": "B", "aliases": None},
            {"name": "C"},  # 无 aliases 键
            {"name": "D", "aliases": '["d1"]'},
        ]
        assert resolve_query_entities(db, "q") == ["A", "B", "C", "D", "d1"]

    def test_aliases_as_list_object_also_accepted(self):
        from pageindex_mutil.agentic.enhance import resolve_query_entities
        db = MagicMock()
        db.search_entities.return_value = [{"name": "A", "aliases": ["a1"]}]
        assert resolve_query_entities(db, "q") == ["A", "a1"]

    def test_none_db_or_blank_query_returns_empty(self):
        from pageindex_mutil.agentic.enhance import resolve_query_entities
        db = MagicMock()
        assert resolve_query_entities(None, "q") == []
        assert resolve_query_entities(db, "") == []
        assert resolve_query_entities(db, "   ") == []
        db.search_entities.assert_not_called()

    def test_search_entities_exception_degrades_to_empty(self):
        from pageindex_mutil.agentic.enhance import resolve_query_entities
        db = MagicMock()
        db.search_entities.side_effect = RuntimeError("boom")
        assert resolve_query_entities(db, "q") == []

    def test_real_db_roundtrip_name_and_alias(self):
        """真实 PageIndexDB：实体名与别名都能被解析出来。"""
        import os
        import tempfile
        from db import PageIndexDB
        from pageindex_mutil.agentic.enhance import resolve_query_entities

        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            db = PageIndexDB(db_path)
            db.insert_entity("concept", "浴血值", ["浴血点数"])
            names = resolve_query_entities(db, "浴血值怎么获得")
            assert "浴血值" in names
            assert "浴血点数" in names
            db.close()
        finally:
            os.unlink(db_path)

    def test_config_defaults_wired(self):
        """config.yaml 新增键生效（union_max_candidates/evidence_max_chars）。"""
        enh = UnifiedNodeEnhancement("m")
        assert enh.union_max_candidates == 80
        assert enh.evidence_max_chars == 6000

    def test_weights_follow_spec_ordering(self):
        """[1.2]③ w_e > w_t > w_k。"""
        assert WEIGHT_ENTITY > WEIGHT_TAG > WEIGHT_KEYWORD
        assert ABSOLUTE_CEILING_MULTIPLIER == 2


# ===========================================================================
# 11. P2 审查修复回归
# ===========================================================================


class TestGlobalNoteDistinctNodes:
    """#1 回归：全局注记计数 distinct 节点——同 profile 的大小写变体/重复条目
    不得把同一节点重复计入，注记文本节点 id 无重复。"""

    def test_case_variant_entries_do_not_trigger_premature_note(self):
        # 2 节点 × 2 个大小写变体 = 4 条目 > 阈值，但 distinct 节点仅 2 → 不得出注记
        candidates = [_cand("n0"), _cand("n1")]
        profiles = {
            nid: {"entities": [{"name": "Einstein", "type": "person"},
                               {"name": "einstein", "type": "person"}],
                  "keywords": [], "tags": []}
            for nid in ("n0", "n1")
        }
        enh = _enhancer()
        _, mock_llm = _select_call(
            enh,
            return_value=_resp(["n0", "n1"]),
            query="物理学家的贡献",
            candidates=candidates,
            profiles=profiles,
            query_entities=["Einstein"],
        )
        section = _evidence_section(_prompt_of(mock_llm))
        assert "命中于节点" not in section
        # 两个节点各自保留实体展开（未发生 keeper 折叠）
        assert section.count("实体匹配") == 2

    def test_note_text_lists_each_node_id_once(self):
        # 4 节点各带大小写变体 + 重复关键词条目 → 注记合法触发，但每个节点 id 只出现一次
        candidates = [_cand(f"n{i}") for i in range(4)]
        profiles = {
            f"n{i}": {"entities": [{"name": "Einstein", "type": "person"},
                                    {"name": "einstein", "type": "person"}],
                      "keywords": ["武功", "武功"], "tags": []}
            for i in range(4)
        }
        enh = _enhancer()
        _, mock_llm = _select_call(
            enh,
            return_value=_resp([f"n{i}" for i in range(4)]),
            query="武功秘籍",
            candidates=candidates,
            profiles=profiles,
            query_entities=["Einstein"],
        )
        section = _evidence_section(_prompt_of(mock_llm))
        ent_note = next(l for l in section.splitlines() if l.startswith("注：实体"))
        kw_note = next(l for l in section.splitlines() if l.startswith("注：关键词"))
        for note in (ent_note, kw_note):
            for i in range(4):
                assert note.count(f"n{i}") == 1, f"{note} 中 n{i} 出现多次"


class TestCapRobustness:
    """#2 回归：cap ≤0 钳位 ≥1；非法覆盖/实例配置逐级回退——enhance_and_select 永不抛出。"""

    def test_zero_and_negative_configured_cap_clamps_to_one(self):
        for bad_cap in (0, -3):
            enh, kwargs = TestMaxCandidatesOverride._fixture(cap=bad_cap)
            result, _ = _select_call(enh, return_value="", **kwargs)
            # 钳位到 1：保留分数最高的 a(3.0)，b/c 进延迟池
            assert result["selected_ids"] == ["a"], bad_cap
            assert result["deferred"] == ["b", "c"], bad_cap

    def test_invalid_max_candidates_falls_back_to_configured_cap(self):
        enh, kwargs = TestMaxCandidatesOverride._fixture(cap=2)
        for bad in ("not-a-number", object(), []):
            result, _ = _select_call(
                enh, return_value=_resp(["a", "b", "c"]),
                max_candidates=bad, **kwargs,
            )
            assert result["deferred"] == ["c"], bad
            assert result["selected_ids"] == ["a", "b"], bad

    def test_invalid_instance_cap_falls_back_to_class_default(self):
        candidates = [_cand(f"n{i}") for i in range(4)]
        profiles = {
            f"n{i}": {"entities": [], "keywords": ["武功"], "tags": []} for i in range(4)
        }
        enh = _enhancer()
        enh.union_max_candidates = "oops"
        result, _ = _select_call(
            enh,
            return_value="",
            query="武功秘籍",
            candidates=candidates,
            profiles=profiles,
        )
        assert result["deferred"] == []
        assert result["selected_ids"] == [f"n{i}" for i in range(4)]


class TestTieBreakDeterminism:
    """平分裁决确定性：同多信号分 → 输入顺序保留，重复运行输出一致。"""

    def test_equal_scores_preserve_input_order_and_are_deterministic(self):
        candidates = [_cand("n0"), _cand("n1"), _cand("n2")]
        profiles = {
            f"n{i}": {"entities": [], "keywords": ["武功"], "tags": []} for i in range(3)
        }
        call = dict(query="武功秘籍", candidates=candidates, profiles=profiles)
        enh = _enhancer(cap=2)
        r1, _ = _select_call(enh, return_value="", **call)
        r2, _ = _select_call(enh, return_value="", **call)
        assert r1 == r2
        # 平分（各 1.0）→ 稳定排序按输入序：保留 n0/n1，延迟 n2
        assert r1["selected_ids"] == ["n0", "n1"]
        assert r1["deferred"] == ["n2"]


class TestQueryEntitiesEquivalence:
    """query_entities=[] 与 None 完全等价（无实体通道信号）。"""

    def test_empty_list_behaves_identically_to_none(self):
        candidates = [_cand("n_ent"), _cand("n_kw")]
        profiles = {
            "n_ent": {"entities": [{"name": "张三", "type": "person"}], "keywords": [], "tags": []},
            "n_kw": {"entities": [], "keywords": ["浴血"], "tags": []},
        }
        call = dict(query="浴血", candidates=candidates, profiles=profiles)
        enh = _enhancer()
        r_none, m_none = _select_call(
            enh, return_value=_resp(["n_ent", "n_kw"]), query_entities=None, **call
        )
        r_empty, m_empty = _select_call(
            enh, return_value=_resp(["n_ent", "n_kw"]), query_entities=[], **call
        )
        assert r_none == r_empty
        assert _prompt_of(m_none) == _prompt_of(m_empty)
        # [] 不带实体 → n_ent 不进 union，LLM 选了也被校验过滤
        assert r_none["selected_ids"] == ["n_kw"]


class TestUnhashableNodeId:
    """#4 回归：不可哈希 node_id（如 list）不抛出——profiles 查表回退 str 键。"""

    def test_unhashable_node_id_does_not_raise(self):
        candidates = [
            {"node_id": ["bad"], "title": "脏", "summary": "脏"},
            _cand("n0"),
        ]
        profiles = {"n0": {"entities": [], "keywords": ["浴血"], "tags": []}}
        enh = _enhancer()
        result, _ = _select_call(
            enh,
            return_value=_resp(["n0"]),
            query="浴血",
            candidates=candidates,
            profiles=profiles,
        )
        assert result["selected_ids"] == ["n0"]
