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
14. retry_on_pool_concern（[3.2.1] 共享重选助手）：无 concern 透传；deferred/force-all
    两个互斥分支都放宽 cap×2；回归——候选 > cap 时 force-all 重试零信号节点保持可见
    （不全被再次截入延迟池）；至多一次重试。

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

    def test_empty_selection_falls_back_to_union(self):
        """[S13] 空选保底（局部复归，T31.2）：LLM 显式 [] 且无 pool_concern →
        放行 union 信号最强子集作召回下限，不再是"合法空选丢弃整篇文档"。"""
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
        assert result["selected_ids"] == ["n0"]  # union 信号子集放行
        assert result["pool_concern"] is False
        assert result["concern_reason"] == "empty_selection_fallback"  # 过裁标记（非 llm_unavailable）

    def test_empty_candidates_short_circuits(self):
        enh = _enhancer()
        with patch.object(enhance_mod, "llm_completion") as mock_llm:
            result = asyncio.run(enh.enhance_and_select("q", [], {}))
        assert result == {"selected_ids": [], "pool_concern": False,
                          "concern_reason": "", "deferred": [],
                          "selection_fallback": False}
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
        assert set(result.keys()) == {"selected_ids", "pool_concern",
                                      "concern_reason", "deferred",
                                      "selection_fallback"}


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
# T6.4: resolve_node_profiles（共享助手——单文档/多文档路径共用）
# ===========================================================================


class TestResolveNodeProfiles:
    def test_db_profiles_preferred_over_structure_keys(self):
        from pageindex_mutil.agentic.enhance import resolve_node_profiles
        db = MagicMock()
        db.get_node_profiles.return_value = [
            {"node_id": "n1", "entities": [{"name": "E", "type": "t"}],
             "keywords": ["k1"], "tags": None},
        ]
        mapping = {"n1": {"keywords": ["不该被采用"]}}
        profiles = resolve_node_profiles(db, 7, mapping)
        db.get_node_profiles.assert_called_once_with(7)  # db 整数 id
        assert profiles == {
            "n1": {"entities": [{"name": "E", "type": "t"}],
                   "keywords": ["k1"], "tags": []}
        }

    def test_db_empty_rows_falls_back_to_structure_keys(self):
        from pageindex_mutil.agentic.enhance import resolve_node_profiles
        db = MagicMock()
        db.get_node_profiles.return_value = []
        mapping = {
            "n1": {"keywords": ["k"], "entities": None},
            "n2": {},  # 无签名键 → 不进 profiles
        }
        assert resolve_node_profiles(db, 1, mapping) == {"n1": {"keywords": ["k"]}}

    def test_db_none_or_doc_id_none_uses_structure_keys(self):
        from pageindex_mutil.agentic.enhance import resolve_node_profiles
        mapping = {"n1": {"tags": ["t1"]}}
        assert resolve_node_profiles(None, 1, mapping) == {"n1": {"tags": ["t1"]}}
        db = MagicMock()
        assert resolve_node_profiles(db, None, mapping) == {"n1": {"tags": ["t1"]}}
        db.get_node_profiles.assert_not_called()

    def test_db_exception_degrades_to_structure_keys(self):
        from pageindex_mutil.agentic.enhance import resolve_node_profiles
        db = MagicMock()
        db.get_node_profiles.side_effect = RuntimeError("boom")
        mapping = {"n1": {"keywords": ["k"]}}
        assert resolve_node_profiles(db, 9, mapping) == {"n1": {"keywords": ["k"]}}

    def test_nothing_available_returns_empty(self):
        from pageindex_mutil.agentic.enhance import resolve_node_profiles
        assert resolve_node_profiles(None, None, {"n1": {}}) == {}
        assert resolve_node_profiles(None, None, {}) == {}

    def test_all_malformed_db_rows_falls_back_to_structure_keys(self):
        """P2-Fix6: when all DB rows are malformed (non-dict), fall through to structure keys."""
        from pageindex_mutil.agentic.enhance import resolve_node_profiles
        db = MagicMock()
        db.get_node_profiles.return_value = ["not a dict", 123, None]
        mapping = {"n1": {"keywords": ["structure_kw"], "tags": ["structure_tag"]}}
        profiles = resolve_node_profiles(db, 1, mapping)
        assert profiles == {"n1": {"keywords": ["structure_kw"], "tags": ["structure_tag"]}}


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


# ===========================================================================
# 12. P2.6 union 正文内容通道：直接内容接地（不依赖存储签名）
# ===========================================================================


class TestContentTextChannel:
    """候选可选携带 `text` 字段：query token 命中正文 → 进 union，命中词记入
    关键词证据（LLM 视为关键词命中，WEIGHT_KEYWORD 计分）。无 `text` 字段的
    候选（如语料树簇节点）通道关闭，行为与之前完全一致。"""

    def test_content_hit_admits_node_without_profile(self):
        """无存储签名但正文含查询词的节点进 union；匹配词以关键词命中呈现。"""
        candidates = [
            {"node_id": "n_body", "title": "数值", "summary": "数值",
             "text": "浴血值可以通过日常任务获得。"},
            _cand("n_miss"),
        ]
        enh = _enhancer()
        result, mock_llm = _select_call(
            enh,
            return_value=_resp(["n_body"]),
            query="浴血值怎么获得",
            candidates=candidates,
            profiles={},
        )
        assert result["selected_ids"] == ["n_body"]
        ev = _evidence_section(_prompt_of(mock_llm))
        assert "候选节点 n_body" in ev
        assert "候选节点 n_miss" not in ev  # 无任何信号 → 不进 union
        kw_lines = [l for l in ev.splitlines() if l.startswith("关键词命中：")]
        assert kw_lines and "浴血" in kw_lines[0]  # 正文命中词进关键词证据

    def test_content_channel_rescues_node_drowned_by_junk_signature(self):
        """P2 评测回归：存储 keywords 被引用垃圾（08/2015/官网/引用/日期）淹没、
        查询概念不在 top-K → 正文通道按节点正文直接接地，节点重新进 union。"""
        junk = ["08", "2015", "官网", "引用", "日期"]
        candidates = [
            {"node_id": "n_junk", "title": "参考资料", "summary": "参考资料",
             "text": "浴血值可以通过日常任务获得。引用 日期 官网 08 2015"},
            {"node_id": "n_sig", "title": "其他", "summary": "其他",
             "text": "完全无关的内容。"},
        ]
        profiles = {
            "n_junk": {"entities": [], "keywords": junk, "tags": []},
            "n_sig": {"entities": [], "keywords": ["获得"], "tags": []},  # 保 union 非空
        }
        enh = _enhancer()
        result, mock_llm = _select_call(
            enh,
            return_value=_resp(["n_junk"]),
            query="浴血值怎么获得",
            candidates=candidates,
            profiles=profiles,
        )
        assert result["selected_ids"] == ["n_junk"]
        ev = _evidence_section(_prompt_of(mock_llm))
        assert "候选节点 n_junk" in ev
        # 正文命中词与签名词并列呈现；浴血 必须可见
        kw_block = ev.split("候选节点 n_junk：", 1)[1].split("候选节点", 1)[0]
        assert "浴血" in kw_block
        # 垃圾词不作为命中证据呈现（签名通道本就未命中）
        assert "关键词命中：08" not in ev

    def test_no_text_field_content_channel_inactive(self):
        """后向兼容：无 `text` 字段的候选行为不变——即使标题/摘要含查询词，
        也不经内容通道进 union（另一候选有信号，零信号兜底不触发）。"""
        candidates = [
            _cand("n_sig"),
            _cand("n_title_only", title="声望值获取", summary="声望值获取方式"),
        ]
        profiles = {"n_sig": {"entities": [], "keywords": ["声望"], "tags": []}}
        enh = _enhancer()
        result, mock_llm = _select_call(
            enh,
            return_value=_resp(["n_sig", "n_title_only"]),
            query="声望值怎么获得",
            candidates=candidates,
            profiles=profiles,
        )
        assert result["selected_ids"] == ["n_sig"]  # n_title_only 不在 union
        ev = _evidence_section(_prompt_of(mock_llm))
        assert "候选节点 n_sig" in ev
        assert "候选节点 n_title_only" not in ev

    def test_non_str_text_defensive_no_raise(self):
        """防御性：`text` 为非字符串（None/数字等脏数据）时内容通道关闭，绝不抛出。"""
        enh = _enhancer()
        assert UnifiedNodeEnhancement._content_hits(["浴血"], None) == []
        assert UnifiedNodeEnhancement._content_hits(["浴血"], 123) == []
        assert UnifiedNodeEnhancement._content_hits(["浴血"], {"a": 1}) == []
        candidates = [
            {"node_id": "n_sig"},
            {"node_id": "n_bad", "title": "t", "summary": "s", "text": 42},
        ]
        profiles = {"n_sig": {"entities": [], "keywords": ["浴血"], "tags": []}}
        result, mock_llm = _select_call(
            enh, return_value=_resp(["n_sig"]),
            query="浴血值", candidates=candidates, profiles=profiles,
        )
        assert result["selected_ids"] == ["n_sig"]  # 坏 text 节点走零信号路径、不抛错

    def test_content_hit_dedups_with_stored_keyword_hit(self):
        """同一词的签名命中与正文命中合并呈现一次（casefold 去重，不重复计分）。"""
        candidates = [
            {"node_id": "n1", "title": "t", "summary": "s",
             "text": "浴血值是游戏内的一种货币。"},
        ]
        profiles = {"n1": {"entities": [], "keywords": ["浴血"], "tags": []}}
        enh = _enhancer()
        result, mock_llm = _select_call(
            enh,
            return_value=_resp(["n1"]),
            query="浴血值怎么获得",
            candidates=candidates,
            profiles=profiles,
        )
        ev = _evidence_section(_prompt_of(mock_llm))
        kw_lines = [l for l in ev.splitlines() if l.startswith("关键词命中：")]
        assert kw_lines == ["关键词命中：浴血"]  # 单条，无重复词

    def test_content_hit_casefold_ascii(self):
        """ASCII 正文/查询大小写不敏感。"""
        candidates = [
            {"node_id": "n1", "title": "t", "summary": "s",
             "text": "支持 CUDA 加速计算。"},
            _cand("n2"),
        ]
        enh = _enhancer()
        result, mock_llm = _select_call(
            enh,
            return_value=_resp(["n1"]),
            query="cuda加速吗",
            candidates=candidates,
            profiles={"n2": {"entities": [], "keywords": ["天气"], "tags": []}},
        )
        assert result["selected_ids"] == ["n1"]
        ev = _evidence_section(_prompt_of(mock_llm))
        kw_lines = [l for l in ev.splitlines() if l.startswith("关键词命中：")]
        assert kw_lines and "cuda" in kw_lines[0]

    def test_content_hits_count_toward_keyword_weight_in_cap_ordering(self):
        """正文命中按关键词权重计分：cap 截断时 签名+正文 多信号节点优先保留。"""
        candidates = [
            {"node_id": "n_body_only", "title": "t", "summary": "s",
             "text": "浴血值在这里。"},
            {"node_id": "n_both", "title": "t", "summary": "s",
             "text": "浴血值也在这里。"},
        ]
        # n_both：标签信号(2) + 正文命中关键词信号(1) = 3 > n_body_only 正文单命中(1)
        profiles = {"n_both": {"entities": [], "keywords": [], "tags": ["浴血值"]}}
        enh = _enhancer(cap=1)
        result, _ = _select_call(
            enh,
            return_value=_resp(["n_both"]),
            query="浴血值",
            candidates=candidates,
            profiles=profiles,
        )
        assert result["selected_ids"] == ["n_both"]
        assert result["deferred"] == ["n_body_only"]


# ===========================================================================
# 13. P2.6 force_all_candidates：pool_concern 空池全量重选（零信号直通）
# ===========================================================================


class TestForceAllCandidates:
    """force_all_candidates=True → union 视为全量候选准入（含零信号节点），
    供 pool_concern 且无被截候选时的全池重选（[3.2.1]）；默认 False 行为不变。"""

    def test_force_all_admits_zero_signal_nodes_when_union_nonempty(self):
        candidates = [
            _cand("n_sig"),
            _cand("n_weak", title="弱候选标题", summary="弱候选摘要"),
        ]
        profiles = {"n_sig": {"entities": [], "keywords": ["声望"], "tags": []}}
        enh = _enhancer()
        result, mock_llm = _select_call(
            enh,
            return_value=_resp(["n_sig", "n_weak"]),
            query="声望值怎么获得",
            candidates=candidates,
            profiles=profiles,
            force_all_candidates=True,
        )
        ev = _evidence_section(_prompt_of(mock_llm))
        assert "候选节点 n_sig" in ev
        assert "候选节点 n_weak" in ev  # 零信号节点也进 union
        assert "弱候选标题" in ev  # 标题+摘要可见
        assert result["selected_ids"] == ["n_sig", "n_weak"]

    def test_default_false_keeps_union_admission_rules(self):
        candidates = [
            _cand("n_sig"),
            _cand("n_weak", title="弱候选标题", summary="弱候选摘要"),
        ]
        profiles = {"n_sig": {"entities": [], "keywords": ["声望"], "tags": []}}
        enh = _enhancer()
        result, mock_llm = _select_call(
            enh,
            return_value=_resp(["n_sig", "n_weak"]),
            query="声望值怎么获得",
            candidates=candidates,
            profiles=profiles,
        )
        ev = _evidence_section(_prompt_of(mock_llm))
        assert "候选节点 n_sig" in ev
        assert "候选节点 n_weak" not in ev
        assert result["selected_ids"] == ["n_sig"]  # 不在 union → 被过滤

    def test_force_all_still_respects_cap_and_defers_overflow(self):
        """全量准入仍受 union cap 约束：超限按信号分截断进延迟池。"""
        candidates = [
            _cand("n_sig"),
            _cand("n_weak"),
        ]
        profiles = {"n_sig": {"entities": [], "keywords": ["声望"], "tags": []}}
        enh = _enhancer(cap=1)
        result, _ = _select_call(
            enh,
            return_value=_resp(["n_sig"]),
            query="声望值怎么获得",
            candidates=candidates,
            profiles=profiles,
            force_all_candidates=True,
        )
        assert result["selected_ids"] == ["n_sig"]
        assert result["deferred"] == ["n_weak"]


# ===========================================================================
# 14. retry_on_pool_concern：pool_concern 重选共享助手（审查加固）
#     ——单文档 _search_single / 多文档 _recall_nodes_for_doc 共用；
#     force-all 分支同样放宽 cap（否则零信号候选按分截断垫底再次被截，
#     全池重选退化为 pass-1，判据①救不到任何节点）。
# ===========================================================================


class TestRetryOnPoolConcern:
    """至多一次重试、二选一互斥分支、两分支均放宽 cap×POOL_CONCERN_RETRY_CAP_MULTIPLIER。"""

    @staticmethod
    def _recording_enhancer(cap=80, results=None):
        """真实实例 + 记录式 enhance_and_select 替身（观察重试调用参数）。"""
        from pageindex_mutil.agentic.enhance import (
            UnifiedNodeEnhancement, POOL_CONCERN_RETRY_CAP_MULTIPLIER,
        )
        enh = UnifiedNodeEnhancement("m", retrieve_model="r-model")
        enh.union_max_candidates = cap
        calls = []
        canned = list(results or [])

        async def fake_select(query, candidates, profiles, query_entities=None,
                              node_budget=None, token_budget=None, max_candidates=None,
                              force_all_candidates=False, l1_reasons=None,
                              query_tokens=None, node_entities=None):
            calls.append({
                "query": query, "candidates": candidates, "profiles": profiles,
                "query_entities": query_entities, "max_candidates": max_candidates,
                "force_all_candidates": force_all_candidates,
                "query_tokens": query_tokens, "node_entities": node_entities,
            })
            return canned[len(calls) - 1]

        enh.enhance_and_select = fake_select
        return enh, calls, POOL_CONCERN_RETRY_CAP_MULTIPLIER

    def test_no_concern_returns_same_result_without_retry(self):
        from pageindex_mutil.agentic.enhance import retry_on_pool_concern
        enh, calls, _ = self._recording_enhancer()
        result = {"selected_ids": ["n0"], "pool_concern": False,
                  "concern_reason": "", "deferred": []}
        with patch.object(enhance_mod, "llm_completion",
                          side_effect=AssertionError("no retry expected")):
            out = asyncio.run(retry_on_pool_concern(
                enh, result, "q", [_cand("n0")], {}))
        assert out is result  # 无 concern → 原样透传，零额外 LLM 调用
        assert calls == []

    def test_deferred_branch_relaxes_cap_without_force_all(self):
        from pageindex_mutil.agentic.enhance import retry_on_pool_concern
        enh, calls, mult = self._recording_enhancer(cap=5, results=[
            {"selected_ids": ["n0", "n2"], "pool_concern": False,
             "concern_reason": "", "deferred": []},
        ])
        pass1 = {"selected_ids": ["n0"], "pool_concern": True,
                 "concern_reason": "疑似漏掉分支", "deferred": ["n2"]}
        out = asyncio.run(retry_on_pool_concern(
            enh, pass1, "q", [_cand("n0"), _cand("n2")], {}))
        assert len(calls) == 1  # 至多一次重试
        assert calls[0]["max_candidates"] == 5 * mult
        assert calls[0]["force_all_candidates"] is False
        assert out["selected_ids"] == ["n0", "n2"]

    def test_force_all_branch_also_relaxes_cap(self):
        """审查加固：force-all 分支同步放宽 cap（防零信号候选垫底再截）。"""
        from pageindex_mutil.agentic.enhance import retry_on_pool_concern
        enh, calls, mult = self._recording_enhancer(cap=5, results=[
            {"selected_ids": ["n1"], "pool_concern": False,
             "concern_reason": "", "deferred": []},
        ])
        pass1 = {"selected_ids": [], "pool_concern": True,
                 "concern_reason": "关键概念无命中", "deferred": []}
        out = asyncio.run(retry_on_pool_concern(
            enh, pass1, "q", [_cand("n0"), _cand("n1")], {}))
        assert len(calls) == 1  # 至多一次重试
        assert calls[0]["max_candidates"] == 5 * mult  # 放宽 cap，不再 None
        assert calls[0]["force_all_candidates"] is True
        assert out["selected_ids"] == ["n1"]

    def test_retry_at_most_once_when_still_concerned(self):
        from pageindex_mutil.agentic.enhance import retry_on_pool_concern
        enh, calls, _ = self._recording_enhancer(results=[
            {"selected_ids": ["n0"], "pool_concern": True,
             "concern_reason": "仍偏弱", "deferred": []},
        ])
        pass1 = {"selected_ids": [], "pool_concern": True,
                 "concern_reason": "关键概念无命中", "deferred": []}
        out = asyncio.run(retry_on_pool_concern(
            enh, pass1, "q", [_cand("n0")], {}))
        assert len(calls) == 1  # 重选仍 concern 也接受，绝无第二次重试
        assert out["pool_concern"] is True

    def test_force_all_retry_keeps_zero_signal_nodes_visible(self):
        """Important#1 回归：候选数 > cap 时 force-all 重试必须同步放宽 cap。

        cap=3、6 候选（2 有信号 + 4 零信号）：pass-1 仅 2 信号节点准入且未被截
        （deferred 空）→ pool_concern 走 force-all 分支。若重试不放宽 cap，全量
        准入后按分降序截断，零信号节点垫底全部/大部再次被截——等价 pass-1，判据①
        救不到任何节点。放宽后（3×2=6 ≥ 6）零信号节点全部留在 LLM 证据池。
        """
        from pageindex_mutil.agentic.enhance import retry_on_pool_concern
        enh = _enhancer(cap=3)
        candidates = [
            _cand("n_sig1"), _cand("n_sig2"),
            _cand("n_z1"), _cand("n_z2"), _cand("n_z3"), _cand("n_z4"),
        ]
        profiles = {
            "n_sig1": {"entities": [], "keywords": ["浴血"], "tags": []},
            "n_sig2": {"entities": [], "keywords": ["声望"], "tags": []},
        }
        query = "浴血声望怎么获得"
        pass1_resp = _resp([], pool_concern=True, concern_reason="关键概念命中偏弱")
        retry_resp = _resp(["n_sig1", "n_z3"])
        with patch.object(enhance_mod, "llm_completion",
                          side_effect=[pass1_resp, retry_resp]) as mock_llm:
            result1 = asyncio.run(
                enh.enhance_and_select(query, candidates, profiles))
            result2 = asyncio.run(retry_on_pool_concern(
                enh, result1, query, candidates, profiles))
        assert mock_llm.call_count == 2

        # pass-1：仅信号节点可见；union 未超限 → deferred 空（走 force-all 分支）
        assert result1["deferred"] == []
        ev1 = _evidence_section(mock_llm.call_args_list[0][0][1])
        assert "候选节点 n_sig1" in ev1 and "候选节点 n_sig2" in ev1
        for zid in ("n_z1", "n_z2", "n_z3", "n_z4"):
            assert f"候选节点 {zid}" not in ev1

        # 重试：cap 放宽至 3×2=6 → 6 候选全量可见，零信号节点未被再次截掉
        ev2 = _evidence_section(mock_llm.call_args_list[1][0][1])
        for nid in ("n_sig1", "n_sig2", "n_z1", "n_z2", "n_z3", "n_z4"):
            assert f"候选节点 {nid}" in ev2
        assert result2["deferred"] == []
        # 零信号节点不仅可见，且可被 LLM 选中
        assert result2["selected_ids"] == ["n_sig1", "n_z3"]


# ===========================================================================
# 15. [S6]#7/[S7] l1_reasons trace：L1 选中理由注入 L2 节点裁定 prompt（防锚定）
# ===========================================================================


class TestL1ReasonsInjection:
    def test_l1_reason_injected_and_labeled(self):
        """未传 l1_reasons 时 prompt 不含该段；传入时带"判断而非事实"标注注入。"""
        enh = _enhancer()
        p1 = enh._build_prompt("查询Q", "证据", 2, None)
        assert "上级选档依据" not in p1  # 未传理由时 prompt 不含该段
        p2 = enh._build_prompt("查询Q", "证据", 2, None, l1_reasons={"d1": "正文命中"})
        assert "上级选档依据（判断而非事实，供参考，可推翻）" in p2
        assert "正文命中" in p2

    def test_enhance_and_select_forwards_l1_reasons_to_prompt(self):
        """enhance_and_select(l1_reasons=...) 透传至 _build_prompt，标注段入 prompt。"""
        enh = _enhancer()
        _, mock_llm = _select_call(
            enh,
            return_value=_resp(["n0"]),
            query="浴血",
            candidates=[_cand("n0")],
            profiles={"n0": {"entities": [], "keywords": ["浴血"], "tags": []}},
            l1_reasons={"d1": "正文命中"},
        )
        prompt = _prompt_of(mock_llm)
        assert "上级选档依据（判断而非事实，供参考，可推翻）" in prompt
        assert "正文命中" in prompt


# ===========================================================================
# 16. [S7] query_tokens 直通：非 None 时复用 L0 共享物，不再内部 tokenize
# ===========================================================================


class TestQueryTokensReuse:
    def test_enhance_reuses_provided_query_tokens(self, monkeypatch):
        """义务：query_tokens 非 None 时不再内部 tokenize（复用 L0 共享物）。"""
        enh = UnifiedNodeEnhancement("m", retrieve_model="r")
        calls = {"n": 0}
        real = UnifiedNodeEnhancement._tokenize

        def spy(cls, text):
            calls["n"] += 1
            return real(text)

        monkeypatch.setattr(UnifiedNodeEnhancement, "_tokenize", classmethod(spy))
        with patch.object(enhance_mod, "llm_completion", return_value=_resp(["n1"])):
            asyncio.run(enh.enhance_and_select(
                "q", [{"node_id": "n1", "title": "t", "summary": "s", "text": "浴血"}],
                {}, query_tokens=["浴血"]))
        assert calls["n"] == 0  # 提供 tokens 后零内部 tokenize


# ===========================================================================
# 17. [S7]/[S1]① L2 正文命中上下文直通：证据渲染命中处上下文补 LLM 知识盲区
# ===========================================================================


class TestContentHitContexts:
    def test_content_hit_contexts_extracts_window(self):
        from pageindex_mutil.agentic.enhance import UnifiedNodeEnhancement
        text = "浴血值可以通过完成日常任务获得，是帮会系统中重要的成长数值。"
        ctxs = UnifiedNodeEnhancement._content_hit_contexts(["浴血"], text)
        assert ctxs and "浴血值" in ctxs[0]  # 命中处上下文，而非裸 token

    def test_evidence_renders_content_context(self):
        """正文命中 → 节点证据块含命中上下文（非只裸 token），补 LLM 知识盲区。"""
        from pageindex_mutil.agentic.enhance import UnifiedNodeEnhancement
        enh = UnifiedNodeEnhancement("m", retrieve_model="r")
        text = "浴血值可以通过完成日常任务获得。"
        cand = {"node_id": "n1", "title": "t", "summary": "s", "text": text}
        signals = {"n1": {"entities": [], "keywords": [], "tags": [], "score": 0, "pos": 0,
                          "content_contexts": ["浴血值可以通过完成日常任务获得"]}}
        evidence = enh._assemble_evidence(["n1"], signals, {"n1": cand})
        assert "浴血值可以通过完成日常任务获得" in evidence

    def test_enhance_wires_content_contexts_into_signals(self, monkeypatch):
        """claim(b) 接线回归：正文命中 → node_signals 落 content_contexts（删接线即断特性）。"""
        from pageindex_mutil.agentic import enhance as emod
        enh = emod.UnifiedNodeEnhancement("m", retrieve_model="r")
        captured = {}
        real_assemble = enh._assemble_evidence

        def spy_assemble(union, node_signals, cand_by_id):
            captured["signals"] = node_signals
            return real_assemble(union, node_signals, cand_by_id)

        monkeypatch.setattr(enh, "_assemble_evidence", spy_assemble)
        monkeypatch.setattr(
            emod, "llm_completion",
            lambda *a, **k: '{"selected_ids": ["n1"], "pool_concern": false, "concern_reason": ""}',
        )
        import asyncio
        asyncio.run(enh.enhance_and_select(
            "浴血",
            [{"node_id": "n1", "title": "t", "summary": "s",
              "text": "浴血值可以通过完成日常任务获得。"}],
            {}, query_tokens=["浴血"]))
        assert captured["signals"]["n1"].get("content_contexts")


# ===========================================================================
# 18. T31.1 密度优先窗口：判别性证据段（多 token 密集）优先于泛化词首现位置
# ===========================================================================


class TestContentHitContextDensity:
    """_content_hit_contexts 密度优先（#5/#8 诊断根因①修复）：
    旧实现按 query token 顺序取前 2 个命中的 ±60 窗口——永远渲染最泛化词
    （提升/自身）的窗口，判别性答案段（灭世双头龙/丸带句）从未进入证据。"""

    def test_dense_window_beats_generic_first_position(self):
        """泛化词早处散布、判别句晚处密集 → 第一段必须落判别句窗口。"""
        tokens = ["提升", "自身", "攻击力", "灭世", "双头"]
        generic = "提升自身属性的方式有很多，日常任务、帮会活动都可以提升自身能力。" * 3
        dense = "某技能在攻击回合中会大幅提升攻击力，据说能够一次秒杀一头灭世双头龙。"
        filler = "。" + "无关内容填充" * 40 + "。"
        text = generic + filler + dense
        ctxs = UnifiedNodeEnhancement._content_hit_contexts(tokens, text)
        assert ctxs, "应至少返回一个窗口"
        # 密集判别句含 灭世/双头（泛化词窗口不含）——密度优先的判别力所在
        assert "灭世" in ctxs[0] and "双头" in ctxs[0], (
            f"第一段应落判别句窗口，实际：{ctxs[0][:80]}"
        )

    def test_tie_break_longest_token_wins(self):
        """同 distinct 数的平分窗口：锚点 token 最长者优先（稀有度代理，
        防泛化短词堆密度）。"""
        text = ("ab 出现一次。" + "填充" * 60 + "。cdef 也出现一次。更多填充内容" * 5)
        ctxs = UnifiedNodeEnhancement._content_hit_contexts(["ab", "cdef"], text)
        assert ctxs
        assert "cdef" in ctxs[0], "平分时长 token 窗口应优先"

    def test_tie_break_leftmost_when_equal(self):
        """distinct 与最长 token 均平分 → 起点最左窗口优先（确定性）。"""
        text = ("LEFT marker ab here。" + "填充" * 60 + "。RIGHT marker ab there。" + "尾部" * 40)
        ctxs = UnifiedNodeEnhancement._content_hit_contexts(["ab"], text)
        assert ctxs
        assert "LEFT" in ctxs[0], "完全平分时最左窗口优先"

    def test_second_window_disjoint_from_first(self):
        """两个密集簇相距足够远 → 两段分别落两簇（保持 max_hits=2 上限）。"""
        cluster1 = "alpha beta gamma 同现于第一簇"
        cluster2 = "alpha beta gamma 同现于第二簇"
        gap = "。" + "间隔填充" * 80 + "。"
        text = cluster1 + gap + cluster2
        ctxs = UnifiedNodeEnhancement._content_hit_contexts(
            ["alpha", "beta", "gamma"], text)
        assert 1 <= len(ctxs) <= 2
        if len(ctxs) == 2:
            # 第二段必须与第一段字符区间不重叠（各自含自己的簇标记）
            assert "第一簇" in ctxs[0]
            assert "第二簇" in ctxs[1]

    def test_short_text_yields_single_window(self):
        """所有命中都在一个窗口范围内 → 只出 1 段（无合法不重叠第二窗口）。"""
        text = "甲乙丙丁 alpha beta gamma 戊己庚辛"
        ctxs = UnifiedNodeEnhancement._content_hit_contexts(
            ["alpha", "beta", "gamma"], text)
        assert len(ctxs) == 1
        assert "alpha" in ctxs[0]

    def test_many_occurrences_bounded(self):
        """锚点护栏：token 大量重复出现不失控（≤max_hits 段、不抛出）。"""
        text = "重复token到处都是 " * 200 + "尾部 uniqueword 出现一次"
        ctxs = UnifiedNodeEnhancement._content_hit_contexts(
            ["token", "uniqueword"], text)
        assert len(ctxs) <= 2

    def test_casefold_length_mismatch_safe(self):
        """casefold 长度漂移（ß→ss）不得因下标错位抛出/错切——回退安全匹配。"""
        text = "前缀填充内容若干 " * 5 + "straße 灭世双头龙 结尾"
        ctxs = UnifiedNodeEnhancement._content_hit_contexts(["灭世", "双头"], text)
        assert ctxs and "灭世" in ctxs[0]

    def test_boundary_anchor_not_counted_in_window(self):
        """off-by-one 回归（审查 Minor-3，快审复核构造）：起点恰在窗口右缘
        end+window 的锚点不在渲染切片 text[lo:hi) 内，不得参与 distinct/
        tie-break——否则隐形 token 虚增密度、可翻转窗口胜者。
        构造（精确字符算术）：head = "甲乙"+"f"*60 → 长 62；甲乙@0 窗 [0,62)；
        丙丁@62 **恰在右缘**（不在切片内）；壬癸@124（62+2+60）。
        旧实现（bisect_right）：丙丁@62 计入甲乙窗（distinct 虚增 2），且壬癸@124
        计入丙丁窗（distinct 3）→ ctx[0] 落丙丁窗（不含壬癸，错）；
        新实现：甲乙/丙丁窗 distinct=1，壬癸窗含 {壬癸,子丑} distinct=2 胜出。"""
        text = "甲乙" + "f" * 60 + "丙丁" + "y" * 60 + "壬癸子丑"
        ctxs = UnifiedNodeEnhancement._content_hit_contexts(
            ["甲乙", "丙丁", "壬癸", "子丑"], text)
        assert ctxs
        # 修复后：甲乙窗 distinct=1（丙丁不越界计入）；远处 {壬癸,子丑} 簇
        # distinct=2 胜出——第一段必须落远处簇而非被虚增的甲乙窗。
        assert "壬癸" in ctxs[0], "边界锚点不得虚增甲乙窗密度改变胜者"

    def test_defensive_inputs(self):
        assert UnifiedNodeEnhancement._content_hit_contexts(["x"], None) == []
        assert UnifiedNodeEnhancement._content_hit_contexts(["x"], "") == []
        assert UnifiedNodeEnhancement._content_hit_contexts([], "text") == []
        assert UnifiedNodeEnhancement._content_hit_contexts(["不存在词"], "文本内容") == []
        assert UnifiedNodeEnhancement._content_hit_contexts(["a"], "a text") == []  # len<2 token


# ===========================================================================
# 19. T31.2 [S13] 空选保底（局部复归）：合法空选放行 union 信号最强子集
# ===========================================================================


class TestEmptySelectionFallback:
    """修复 B：合法空选（selected 最终为空且 pool_concern=False）→ 放行 union
    按 (score 降序, pos 升序) top-k 作召回下限。区分两种空因；零信号护栏。"""

    def _profiles(self):
        return {
            "n1": {"entities": [], "keywords": ["浴血"], "tags": []},      # score 1
            "n2": {"entities": [{"name": "张三", "type": "person"}], "keywords": [], "tags": []},  # 3
            "n3": {"entities": [], "keywords": [], "tags": ["游戏机制"]},  # 2
        }

    def test_explicit_empty_falls_back_score_ordered(self):
        """LLM 显式 [] → union 按分数降序放行，concern_reason=empty_selection_fallback。"""
        enh = _enhancer()
        result, _ = _select_call(
            enh,
            return_value=_resp([]),
            query="浴血 游戏机制",
            candidates=[_cand("n1"), _cand("n2"), _cand("n3")],
            profiles=self._profiles(),
            query_entities=["张三"],
        )
        assert result["selected_ids"] == ["n2", "n3", "n1"]  # 3.0 > 2.0 > 1.0
        assert result["concern_reason"] == "empty_selection_fallback"
        assert result["pool_concern"] is False
        assert result["selection_fallback"] is True  # 独立布尔标记（Minor-2）

    def test_fallback_topk_respects_config(self):
        """empty_fallback_topk 截断：10 节点 union 只放行信号最强 top-3。"""
        candidates = [_cand(f"n{i}") for i in range(10)]
        profiles = {
            f"n{i}": {"entities": [], "keywords": ["浴血"], "tags": []}
            for i in range(10)
        }
        enh = _enhancer()
        enh.empty_fallback_topk = 3
        result, _ = _select_call(
            enh,
            return_value=_resp([]),
            query="浴血",
            candidates=candidates,
            profiles=profiles,
        )
        assert result["selected_ids"] == ["n0", "n1", "n2"]  # 命中同分按输入序，截 3

    def test_off_union_selection_distinct_marker(self):
        """LLM 选了但全被 union 过滤（键位漂移）→ selection_off_union 标记。"""
        enh = _enhancer()
        result, _ = _select_call(
            enh,
            return_value=_resp(["ghost", "phantom"]),  # 全在 union 外
            query="浴血 游戏机制",
            candidates=[_cand("n1"), _cand("n2")],
            profiles={
                "n1": {"entities": [], "keywords": ["浴血"], "tags": []},      # 1
                "n2": {"entities": [], "keywords": [], "tags": ["游戏机制"]},  # 2
            },
        )
        assert result["selected_ids"] == ["n2", "n1"]  # 分数降序放行
        assert result["concern_reason"] == "selection_off_union"

    def test_zero_signal_union_keeps_empty(self):
        """护栏：union 全零信号（零值泛滥直通）→ 维持空选——LLM 判空与零证据
        互证，放行=注入无接地的目录头部节点，与保底初衷背离。"""
        enh = _enhancer()
        result, _ = _select_call(
            enh,
            return_value=_resp([]),
            query="完全无关查询",  # 零信号：union=全量候选，全零分
            candidates=[_cand("n0"), _cand("n1")],
            profiles={},
        )
        assert result["selected_ids"] == []
        assert result["concern_reason"] == ""  # 无 fallback 标记
        assert result["pool_concern"] is False

    def test_pool_concern_true_empty_not_fallback(self):
        """pool_concern=True 的空选不经保底（retry_on_pool_concern 链路管）。"""
        enh = _enhancer()
        result, _ = _select_call(
            enh,
            return_value=_resp([], pool_concern=True, concern_reason="池子过窄"),
            query="浴血",
            candidates=[_cand("n0")],
            profiles={"n0": {"entities": [], "keywords": ["浴血"], "tags": []}},
        )
        assert result["selected_ids"] == []
        assert result["pool_concern"] is True
        assert result["concern_reason"] == "池子过窄"


# ===========================================================================
# 20. T31.3 retry_on_pool_concern 透传 query 共享物（罕见路径不再重复分词/实体回退）
# ===========================================================================


class TestRetryForwardsQueryArtifacts:
    def _spy_enhancer(self, captured):
        from pageindex_mutil.agentic import enhance as emod
        enh = emod.UnifiedNodeEnhancement("m", retrieve_model="r")

        async def fake_eas(query, candidates, profiles, query_entities=None,
                           query_tokens=None, node_budget=None, token_budget=None,
                           max_candidates=None, force_all_candidates=False,
                           l1_reasons=None, node_entities=None):
            captured.append({
                "query_tokens": query_tokens,
                "node_entities": node_entities,
                "max_candidates": max_candidates,
                "force_all": force_all_candidates,
            })
            return {"selected_ids": ["n1"], "pool_concern": False,
                    "concern_reason": "", "deferred": []}

        enh.enhance_and_select = fake_eas
        return enh

    def test_retry_deferred_branch_forwards_artifacts(self):
        from pageindex_mutil.agentic.enhance import retry_on_pool_concern
        captured = []
        enh = self._spy_enhancer(captured)
        result = asyncio.run(retry_on_pool_concern(
            enh,
            {"selected_ids": [], "pool_concern": True, "concern_reason": "r",
             "deferred": ["n2"]},
            "查询", [_cand("n1")], {},
            query_entities=["张三"], query_tokens=["查询", "词"],
            node_entities={"n1": [{"name": "张三", "type": "person"}]},
        ))
        assert captured, "deferred 分支应触发重选"
        assert captured[0]["query_tokens"] == ["查询", "词"]
        assert captured[0]["node_entities"] == {"n1": [{"name": "张三", "type": "person"}]}
        assert result["selected_ids"] == ["n1"]

    def test_retry_force_all_branch_forwards_artifacts(self):
        from pageindex_mutil.agentic.enhance import retry_on_pool_concern
        captured = []
        enh = self._spy_enhancer(captured)
        asyncio.run(retry_on_pool_concern(
            enh,
            {"selected_ids": [], "pool_concern": True, "concern_reason": "r",
             "deferred": []},
            "查询", [_cand("n1")], {},
            query_entities=["张三"], query_tokens=["查询", "词"],
            node_entities={"n1": [{"name": "张三", "type": "person"}]},
        ))
        assert captured and captured[0]["force_all"] is True
        assert captured[0]["query_tokens"] == ["查询", "词"]
        assert captured[0]["node_entities"] == {"n1": [{"name": "张三", "type": "person"}]}
