"""json_repair：qwen 系列 LLM 输出 JSON 修复钩子的单测 + 检索链接线冒烟。

覆盖三类断言：
1. **真实错误模式**（评测日志 58 例的分布）：缺逗号 / Python 字面量 / 全角标点 /
   截断 / 围栏与前导文本 / 裸键与单引号 —— 逐级修复后可解析，且 applied_fixes
   指明生效级别（可观测性）。
2. **回归（零改动直通）**：合法 JSON 文本一字不改、applied_fixes 为空、
   extract_json_robust 结果与 json.loads 完全一致。
3. **返回契约**：失败返 {}（不抛异常），None/空串/非字符串安全。
4. **接线**：4 个检索链调用点确实走 extract_json_robust（reasoning / verifier
   走功能级冒烟，enhance / super_tree 走源码级断言 —— 完整功能由全套件回归背书）。
"""

import inspect
import json
import re

import pytest

from pageindex_mutil.json_repair import (
    ESCALATED_STAGES,
    REPAIR_STAGES,
    clip_json_fragment,
    close_truncated,
    convert_single_quotes,
    escape_control_chars_in_strings,
    extract_json_robust,
    fix_python_literals,
    insert_missing_commas,
    normalize_fullwidth,
    quote_bare_keys,
    remove_trailing_commas,
    repair_json_text,
    strip_code_fence,
)


def _repair(text):
    """repair_json_text 的文本投影（丢弃 fixes，用于断言"改坏了什么"）。"""
    return repair_json_text(text)[0]


def _fixed(text):
    """修复后必须可解析，返回 parsed；不可解析直接失败（测试不宜静默放过）。"""
    repaired, fixes = repair_json_text(text)
    obj = json.loads(repaired)  # 抛 JSONDecodeError 即测试失败
    assert isinstance(fixes, list)
    return obj, fixes


# ---------------------------------------------------------------------------
# 1. 真实错误模式
# ---------------------------------------------------------------------------

class TestRealErrorPatterns:
    def test_missing_comma_in_array(self):
        """'Expecting ',' delimiter' —— 数组内相邻字符串漏逗号（qwen 高频）。"""
        src = '{"selected_ids": ["0001" "0002"], "reason": "x"}'
        obj, fixes = _fixed(src)
        assert obj == {"selected_ids": ["0001", "0002"], "reason": "x"}
        assert "missing_commas" in fixes

    def test_missing_comma_between_pairs(self):
        src = '{"doc_ids": ["a"], "reasons": {"a": "理由一" "b": "理由二"}}'
        obj, fixes = _fixed(src)
        assert obj == {"doc_ids": ["a"], "reasons": {"a": "理由一", "b": "理由二"}}
        assert "missing_commas" in fixes

    def test_missing_comma_after_object_value(self):
        """题述保守子集：`}` 后紧跟新键。"""
        src = '{"a": {"b": 1} "c": 2}'
        obj, _ = _fixed(src)
        assert obj == {"a": {"b": 1}, "c": 2}

    def test_python_bool_literals(self):
        """'Expecting value' —— Python True/False。"""
        src = '{"based_on_context": True, "sufficient": False}'
        obj, fixes = _fixed(src)
        assert obj == {"based_on_context": True, "sufficient": False}
        assert "python_literals" in fixes

    def test_fullwidth_punctuation_outside_strings_only(self):
        """字符串外的全角逗号/冒号归一；**字符串内的全角标点必须原样保留**。"""
        src = '{"a": "文本，含全角：标点"， "b": 1}'
        obj, fixes = _fixed(src)
        assert obj == {"a": "文本，含全角：标点", "b": 1}
        assert "fullwidth" in fixes

    def test_fullwidth_quotes_and_colon(self):
        src = '{“selected_ids”: [“0001”]}'
        obj, fixes = _fixed(src)
        assert obj == {"selected_ids": ["0001"]}
        assert "fullwidth" in fixes

    def test_truncated_array_and_object(self):
        """'Unterminated' —— 括号栈逆序补齐。"""
        obj, fixes = _fixed('{"selected_ids": ["0001", "0002"')
        assert obj == {"selected_ids": ["0001", "0002"]}
        assert "close_truncated" in fixes

    def test_truncated_inside_string(self):
        obj, _ = _fixed('{"reason": "这条证据来自第三章')
        assert obj == {"reason": "这条证据来自第三章"}

    def test_truncated_after_comma_not_left_illegal(self):
        obj, _ = _fixed('{"ids": [1, 2,')
        assert obj == {"ids": [1, 2]}

    def test_truncated_dangling_key_is_dropped_not_fabricated(self):
        """截断在 `"key":` / 悬空键：回退删除残缺片段，**不虚构 null 值**。"""
        obj, _ = _fixed('{"a": 1, "b":')
        assert obj == {"a": 1}
        obj, _ = _fixed('{"a": "x", "b"')
        assert obj == {"a": "x"}
        # 值完整的截断不受影响（冒号后的字符串是完整值，只补 `}`）
        obj, _ = _fixed('{"a": "x"')
        assert obj == {"a": "x"}

    def test_markdown_fence_variants(self):
        for src in (
            '```json\n{"a": 1, "b": [1, 2,]}\n```',
            '```text\n{"a": 1, "b": [1, 2,]}\n```',
            '```\n{"a": 1, "b": [1, 2,]}\n```',
        ):
            obj, fixes = _fixed(src)
            assert obj == {"a": 1, "b": [1, 2]}
            assert "strip_fence" in fixes

    def test_leading_prose_clipped(self):
        obj, fixes = _fixed('好的，结果是：\n{"selected_ids": ["0001"]}\n希望有帮助')
        assert obj == {"selected_ids": ["0001"]}
        assert "clip_fragment" in fixes

    def test_none_word_boundary_does_not_corrupt_string_content(self):
        """旧实现盲 replace('None','null') 会把字符串里的 NoneType 改坏；新钩子不会。"""
        src = '{"a": None, "text": "NoneType 错误"}'
        obj, fixes = _fixed(src)
        assert obj == {"a": None, "text": "NoneType 错误"}
        assert "python_literals" in fixes
        assert "NoneType" in obj["text"]
        # 文本层：字符串字面量内一字未动
        assert fix_python_literals(src) == '{"a": null, "text": "NoneType 错误"}'

    def test_none_inside_key_string_preserved(self):
        src = '{"NoneNode": None, "s": "aNoneb"}'
        obj, _ = _fixed(src)
        assert obj == {"NoneNode": None, "s": "aNoneb"}

    def test_trailing_comma(self):
        obj, fixes = _fixed('{"ids": [1, 2,], "ok": true,}')
        assert obj == {"ids": [1, 2], "ok": True}
        assert "trailing_commas" in fixes

    def test_bare_keys_and_single_quotes(self):
        obj, fixes = _fixed("{selected_ids: ['0001', '0002']}")
        assert obj == {"selected_ids": ["0001", "0002"]}
        assert "bare_keys" in fixes and "single_quotes" in fixes

    def test_escaped_single_quote_in_single_quoted_string(self):
        obj, _ = _fixed("{'a': 'it\\'s'}")
        assert obj == {"a": "it's"}

    def test_raw_newline_inside_string(self):
        """json.loads(strict) 不允许字符串内裸控制字符——旧实现靠全局抹掉换行绕过。"""
        obj, fixes = _fixed('{"quote": "第一行\n第二行"}')
        assert obj == {"quote": "第一行\n第二行"}
        assert "control_chars" in fixes

    def test_combined_multi_failure(self):
        """围栏 + Python bool + 截断（真实 qwen 输出的复合形态）。"""
        src = '结果如下：\n```json\n{"sufficient": True, "need": [{"doc_id": "x"}\n```'
        obj, fixes = _fixed(src)
        assert obj == {"sufficient": True, "need": [{"doc_id": "x"}]}
        assert {"strip_fence", "python_literals", "close_truncated"} <= set(fixes)

    def test_applied_fixes_is_observable_and_ordered(self):
        _, fixes = repair_json_text('```json\n{"a": None, "b": [1,]}\n```')
        assert fixes == ["strip_fence", "python_literals", "trailing_commas"]

    def test_pipeline_gate_order_constants(self):
        """保守级在前、激进级（截断/裸键/单引号）在后 —— 契约稳定。"""
        conservative = [n for n, _ in REPAIR_STAGES]
        escalated = [n for n, _ in ESCALATED_STAGES]
        assert conservative[:2] == ["strip_fence", "clip_fragment"]
        assert escalated == ["close_truncated", "bare_keys", "single_quotes"]


# ---------------------------------------------------------------------------
# 2. 回归：合法 JSON 零改动直通
# ---------------------------------------------------------------------------

VALID_JSON_SAMPLES = [
    '{"a": 1}',
    '{"a": {"b": [1, 2, {"c": "d"}]}}',
    '{"selected_ids": ["0001", "0002"], "reason": "含，全角：标点与: ASCII"}',
    '{"嵌套": {"列表": [1, 2, "含，逗号"], "空": []}, "None字样": "NoneType 错误"}',
    '{"ok": true, "bad": false, "nothing": null}',
    # 字符串内含被转义的双引号 + 全角标点（用 json.dumps 生成合法文本，避免手写转义出错）
    json.dumps({"quote": '他说 "你好"，然后离开 “某某”'}, ensure_ascii=False),
    '{"path": "C:\\\\tmp\\\\a", "slash": "a/b"}',
    '{"multi": "line1\\nline2\\ttab"}',
    '[{"id": 1}, {"id": 2}]',
    '[]',
    '{}',
    '{"num": -1.5e3, "zero": 0}',
    '{"long": "中文' + "证据" * 40 + '"}',
    '  {"padded": true}  ',
    '[1, 2, 3]',
]


class TestRegressions:
    @pytest.mark.parametrize("src", VALID_JSON_SAMPLES)
    def test_valid_json_text_untouched(self, src):
        repaired, fixes = repair_json_text(src)
        assert repaired == src, "合法 JSON 不允许被改写"
        assert fixes == []

    @pytest.mark.parametrize("src", VALID_JSON_SAMPLES)
    def test_robust_matches_json_loads(self, src):
        assert extract_json_robust(src) == json.loads(src)

    @pytest.mark.parametrize("src", VALID_JSON_SAMPLES)
    def test_passthrough_zero_rewrite_even_wrapped(self, src):
        """直通路径不经过任何改写（首步 json.loads 命中即返回）。"""
        assert extract_json_robust(src) == json.loads(src.strip())

    def test_markdown_fenced_valid_json_stripped_correctly(self):
        src = '```json\n' + VALID_JSON_SAMPLES[2] + '\n```'
        assert strip_code_fence(src) == VALID_JSON_SAMPLES[2]
        assert extract_json_robust(src) == json.loads(VALID_JSON_SAMPLES[2])

    def test_comma_inside_string_never_split(self):
        src = '{"reason": "因为A，所以B，且C, D 都相关", "ok": true}'
        assert _repair(src) == src
        assert extract_json_robust(src)["reason"].count("，") == 2

    def test_single_quote_inside_double_quoted_string_untouched(self):
        src = '{"quote": "user\'s file", "b": 1}'
        assert _repair(src) == src
        assert extract_json_robust(src) == json.loads(src)

    def test_repair_is_idempotent_on_repaired_output(self):
        src = '```json\n{"a": True, "ids": [1, 2,]}\n```'
        once, _ = repair_json_text(src)
        twice, fixes2 = repair_json_text(once)
        assert twice == once
        assert fixes2 == []

    def test_array_root_repair_preserved(self):
        obj, _ = _fixed('```json\n["0001" "0002"]\n```')
        assert obj == ["0001", "0002"]


# ---------------------------------------------------------------------------
# 3. 单级函数独立性
# ---------------------------------------------------------------------------

class TestStageIndependence:
    def test_strip_code_fence_no_fence_is_identity(self):
        assert strip_code_fence('{"a": 1}') == '{"a": 1}'
        assert strip_code_fence('  {"a": 1}') == '  {"a": 1}'

    def test_clip_json_fragment_pairs_and_truncation(self):
        assert clip_json_fragment('prose {"a": {"b": 1}} tail') == '{"a": {"b": 1}}'
        assert clip_json_fragment('prose {"a": [1') == '{"a": [1'
        assert clip_json_fragment('no json here') == 'no json here'
        # 字符串内的括号不参与配对
        assert clip_json_fragment('x {"a": "}"} y') == '{"a": "}"}'

    def test_normalize_fullwidth_only_outside_strings(self):
        assert normalize_fullwidth('{"a": "b，c"，"d"： 1}') == '{"a": "b，c","d": 1}'

    def test_fix_python_literals_boundaries(self):
        assert fix_python_literals('{"a": None, "b": True, "c": "None True"}') == \
            '{"a": null, "b": true, "c": "None True"}'
        # 已是小写 → 不动
        assert fix_python_literals('{"a": null}') == '{"a": null}'
        # Noneless / Trueish 之类的词不被词边界误伤
        assert fix_python_literals('{"a": Noneless}') == '{"a": Noneless}'

    def test_remove_trailing_commas(self):
        assert remove_trailing_commas('{"a": [1, 2,], "b": {},}') == '{"a": [1, 2], "b": {}}'
        assert remove_trailing_commas('{"a": "x,]"}') == '{"a": "x,]"}'

    def test_insert_missing_commas_no_false_positive_on_colon(self):
        src = '{"a": "b"}'
        assert insert_missing_commas(src) == src

    def test_close_truncated_stack_order(self):
        assert json.loads(close_truncated('{"a": [{"b": 1}, {"c": "d"')) == \
            {"a": [{"b": 1}, {"c": "d"}]}
        assert close_truncated('{"a": 1}') == '{"a": 1}'

    def test_escape_control_chars_in_strings(self):
        assert escape_control_chars_in_strings('{"a": "x\ny"}') == '{"a": "x\\ny"}'
        assert escape_control_chars_in_strings('{"a": "xy"}') == '{"a": "xy"}'

    def test_quote_bare_keys_nested_and_lists(self):
        assert json.loads(quote_bare_keys('{a: 1, b: [c: 0]}'.replace('[c: 0]', '[{"c": 0}]'))) == \
            {"a": 1, "b": [{"c": 0}]}

    def test_convert_single_quotes_skips_inner_double_quote(self):
        """含双引号的单引号串需要转义，风险高 → 保守跳过该串（不改坏语义）。"""
        src = "{'a': 'say \"hi\"'}"
        out = convert_single_quotes(src)
        assert '"say \\"hi\\""' not in out  # 值串保持原样，未被半路改坏
        assert "'say" in out
        assert extract_json_robust(src) == {}  # 修不动 → 契约返回 {}


# ---------------------------------------------------------------------------
# 4. 返回契约
# ---------------------------------------------------------------------------

class TestContract:
    @pytest.mark.parametrize("bad", [
        None, "", "   ", "这不是 JSON", "{", "{]", "random prose 123",
        '{"unclosed": tru', 123, {"already": "dict"}, ["list"],
    ])
    def test_bad_input_returns_empty_dict(self, bad):
        assert extract_json_robust(bad) == {}

    def test_no_exception_for_binary_garbage(self):
        for src in ['{{{{', ']]]]', '"just a string', '{"a": tru', 'x' * 500]:
            assert isinstance(extract_json_robust(src), (dict, list, str, int, float, type(None)))

    def test_scalar_json_passthrough(self):
        assert extract_json_robust('"文本"') == "文本"
        assert extract_json_robust("123") == 123

    def test_failure_logs_warning(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            assert extract_json_robust("完全不是 JSON") == {}
        assert any("extract_json_robust" in r.getMessage() for r in caplog.records)

    def test_repair_json_text_accepts_non_string_safely(self):
        text, fixes = repair_json_text(None)
        assert text == "" and fixes == []

    def test_stage_exceptions_do_not_escape(self, monkeypatch):
        """任何一级内部炸掉都不能冒到检索链（防御：级别函数抛异常 → 跳过该级）。"""
        import pageindex_mutil.json_repair as jr

        def boom(_text):
            raise RuntimeError("stage exploded")

        stages = list(jr.REPAIR_STAGES) + [("probe", boom)] + list(jr.ESCALATED_STAGES)
        monkeypatch.setattr(jr, "REPAIR_STAGES", tuple(stages))
        obj, _ = _fixed('```json\n{"a": None}\n```')
        assert obj == {"a": None}


# ---------------------------------------------------------------------------
# 5. 接线冒烟（检索链 4 个解析点）
# ---------------------------------------------------------------------------

WIRED_MODULES = [
    "pageindex_mutil.reasoning",
    "pageindex_mutil.super_tree",
    "pageindex_mutil.agentic.enhance",
    "pageindex_mutil.agentic.verifier",
]


@pytest.mark.parametrize("mod_name", WIRED_MODULES)
def test_call_site_uses_robust_hook(mod_name):
    """4 个调用点必须 import 并使用 extract_json_robust，且不再直调旧 extract_json。"""
    import importlib
    mod = importlib.import_module(mod_name)
    src = inspect.getsource(mod)
    assert "extract_json_robust" in src, mod_name
    assert re.search(r"(?<![\w.])extract_json\(", src) is None, mod_name
    assert getattr(mod, "extract_json_robust") is extract_json_robust


def test_reasoning_wiring_repairs_python_bools():
    from pageindex_mutil import reasoning

    class _Msg:
        content = '```json\n{"ids": [1, None, True]}\n```'

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def create(self, *a, **k):
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    orig_client = reasoning.get_llm_client
    orig_model = reasoning._get_retrieve_model_name
    try:
        reasoning.get_llm_client = lambda: _Client()
        reasoning._get_retrieve_model_name = lambda: "fake-model"
        # 旧 extract_json 在 True/None 上返 {} → []；修复钩子应给出字符串化的 id
        # None 由 _call_llm_json 既有逻辑过滤；True 已修复为 true 并被字符串化
        assert reasoning._call_llm_json("p", extract_key="ids") == ["1", "True"]
    finally:
        reasoning.get_llm_client = orig_client
        reasoning._get_retrieve_model_name = orig_model


def test_verifier_wiring_repairs_python_bools():
    from pageindex_mutil.agentic import verifier as vmod

    calls = []

    def fake_llm(model, prompt, **kw):
        calls.append(prompt)
        # 判据不可解析时（旧实现）会走 action="answer" 启发式回退，need 恒为 []
        return ('{"based_on_context": True, "sufficient": False, '
                '"evidence_quote": "上下文实锤", "confidence": True, "need": [{"doc_id": "d1"}]}')

    orig = vmod.llm_completion
    try:
        vmod.llm_completion = fake_llm
        v = vmod.CRAGVerifier(model="fake-model")
        res = v.verify("答案", "上下文" * 50, "问题", source_docs=3, covered_nodes=10)
        assert res.need == [{"doc_id": "d1", "reason": ""}]
        assert res.action != "answer", "未修复时才会走的启发式回退路径被命中"
    finally:
        vmod.llm_completion = orig


def test_enhance_wiring_repairs_missing_comma(monkeypatch):
    """enhance_and_select 的 LLM 精挑输出缺逗号 → 修复后按 LLM 选择裁剪（非降级放行）。

    旧实现在这里返 {} → concern_reason="llm_unavailable"（union 全放行），
    LLM 的精挑被整条丢掉 —— 正是本次要修的检索质量损伤。
    """
    import asyncio

    from pageindex_mutil.agentic import enhance as emod

    captured = {}

    def fake_llm(model, prompt, **kw):
        captured["prompt"] = prompt
        return '{"selected_ids": ["0001" "0002"], "pool_concern": False}'

    monkeypatch.setattr(emod, "llm_completion", fake_llm)
    enh = emod.UnifiedNodeEnhancement("fake-model")
    candidates = [
        {"node_id": "0001", "title": "第一章", "summary": "甲", "text": "甲正文"},
        {"node_id": "0002", "title": "第二章", "summary": "乙", "text": "乙正文"},
        {"node_id": "0003", "title": "第三章", "summary": "丙", "text": "丙正文"},
    ]
    result = asyncio.run(
        enh.enhance_and_select("乙", candidates, {}, node_budget=3)
    )
    assert captured.get("prompt"), "LLM 精挑未被调用"
    assert result["concern_reason"] != "llm_unavailable", "解析失败触发降级：钩子未生效"
    assert result["selected_ids"] == ["0001", "0002"]
