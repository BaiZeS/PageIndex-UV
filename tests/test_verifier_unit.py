"""CRAGVerifier 单元测试：偏严取向（[7.5]a）+ 阈值路由 + 启发式回退。

覆盖：
1. _to_bool 归一化（既有行为保留）；
2. _score_retrieval 启发式计算（既有行为保留）；
3. 新判据（[7.5]a）："上下文是否支撑答案"，要求逐字引用 evidence_quote；
   引用不出上下文实锤 → 视同未接地，高置信 answer 路径必须持有 evidence_quote；
4. tau_high/tau_low 由 config.yaml 覆盖（默认 tau_high 收紧为 0.75）；
5. LLM 失败/沉默/坏 JSON → 纯启发式 s_ret 回退（既有行为保留）；
6. NFR4：verifier 的 LLM 调用走 retrieve_model or model。

隔离说明：其他测试模块（test_router.py / test_agentic_recall.py）会在运行期
purge 并重新 import pageindex_mutil.*，收集期绑定的类会过期。因此本文件一律
经惰性访问器（_verifier_mod/_verifier_cls）取当前生效模块与类，patch 目标同理。
"""

import json
import sys

import pytest

# 收集期清理跨文件残留 stub，干净加载真实模块。
for _mod_name in list(sys.modules):
    if _mod_name == "pageindex_mutil" or _mod_name.startswith("pageindex_mutil."):
        del sys.modules[_mod_name]

import pageindex_mutil.agentic.verifier  # noqa: F401  首次干净加载


@pytest.fixture(autouse=True)
def _fresh_verifier_module():
    """每个用例前 purge 残留模块并重新加载，保证 patch 与断言命中同一对象。"""
    for _m in list(sys.modules):
        if _m == "pageindex_mutil" or _m.startswith("pageindex_mutil."):
            del sys.modules[_m]
    import pageindex_mutil.agentic.verifier  # noqa: F401
    yield


def _verifier_mod():
    import pageindex_mutil.agentic.verifier as m
    return m


def _verifier_cls():
    return _verifier_mod().CRAGVerifier


def long_context():
    return "x " * 4000


def test_to_bool_normalization():
    cls = _verifier_cls()
    assert cls._to_bool(True) is True
    assert cls._to_bool(False) is False
    assert cls._to_bool("true") is True
    assert cls._to_bool("yes") is True
    assert cls._to_bool("是") is True
    assert cls._to_bool("1") is True
    assert cls._to_bool("y") is True
    assert cls._to_bool("false") is False
    assert cls._to_bool("no") is False
    assert cls._to_bool("否") is False
    assert cls._to_bool(0) is False
    assert cls._to_bool(None) is False


def test_score_retrieval_computation():
    v = _verifier_cls()("qwen-plus")
    # Empty context → token_score = 0
    score = v._score_retrieval("", source_docs=0, covered_nodes=0)
    assert score == 0.0

    # Maxed out values
    score = v._score_retrieval(long_context(), source_docs=3, covered_nodes=10)
    assert score == 1.0


def test_verify_returns_verify_result_without_llm(monkeypatch):
    monkeypatch.setattr(_verifier_mod(), "llm_completion", lambda *a, **k: None)
    v = _verifier_cls()("qwen-plus")
    result = v.verify("ans", long_context(), "q", source_docs=3, covered_nodes=10)
    assert isinstance(result, _verifier_mod().VerifyResult)
    assert result.action in ("answer", "expand", "refuse")


class TestStrictPromptCriteria:
    """[7.5]a：判据为"上下文是否支撑答案"，evidence_quote 为高置信必要条件。"""

    def _mock_llm(self, monkeypatch, payload):
        captured = {}

        def fake_llm(model, prompt, **kwargs):
            captured["model"] = model
            captured["prompt"] = prompt
            return payload

        monkeypatch.setattr(_verifier_mod(), "llm_completion", fake_llm)
        return captured

    def test_prompt_judges_context_support_and_demands_evidence_quote(self, monkeypatch):
        captured = self._mock_llm(monkeypatch, json.dumps({
            "based_on_context": True,
            "sufficient": True,
            "evidence_quote": "上下文中的原文片段",
            "confidence": 0.95,
        }))
        v = _verifier_cls()("qwen-plus")
        result = v.verify("ans", long_context(), "问题", source_docs=3, covered_nodes=10)

        prompt = captured["prompt"]
        assert "上下文是否支撑答案" in prompt
        assert "evidence_quote" in prompt
        assert "宁可触发补充召回" in prompt
        # s_ret=1.0、s_cov=0.95、有实锤引用 → combined=0.965 ≥ 0.75 → answer
        assert result.action == "answer"
        assert result.confidence >= v.TAU_HIGH

    def test_missing_evidence_quote_blocks_answer_path(self, monkeypatch):
        """引用不出上下文实锤 → 视同未接地（×0.5），s_cov 再高也不判 answer。"""
        self._mock_llm(monkeypatch, json.dumps({
            "based_on_context": True,
            "sufficient": True,
            "evidence_quote": "",
            "confidence": 0.95,
        }))
        v = _verifier_cls()("qwen-plus")
        result = v.verify("ans", long_context(), "q", source_docs=3, covered_nodes=10)
        # combined = (1.0*0.3 + 0.95*0.7) * 0.5 = 0.4825 → [tau_low, tau_high) → expand
        assert result.action == "expand"
        assert result.confidence < v.TAU_HIGH

    def test_insufficient_context_demotes(self, monkeypatch):
        self._mock_llm(monkeypatch, json.dumps({
            "based_on_context": True,
            "sufficient": False,
            "evidence_quote": "有引用但不充分",
            "confidence": 0.9,
        }))
        v = _verifier_cls()("qwen-plus")
        result = v.verify("ans", long_context(), "q", source_docs=3, covered_nodes=10)
        # combined = (0.3 + 0.63) * 0.5 = 0.465 < 0.75 → expand
        assert result.action == "expand"

    def test_retrieve_model_wired(self, monkeypatch):
        """NFR4：verifier LLM 调用走 retrieve_model or model。"""
        captured = self._mock_llm(monkeypatch, None)
        v = _verifier_cls()("m-model", retrieve_model="r-model")
        v.verify("ans", long_context(), "q")
        assert captured["model"] == "r-model"


class TestThresholdConfig:
    def test_defaults_tightened_from_config_yaml(self):
        v = _verifier_cls()("qwen-plus")
        assert v.TAU_HIGH == 0.75  # config.yaml tau_high：偏严收紧
        assert v.TAU_LOW == 0.4

    def test_config_override_respected(self, monkeypatch):
        from types import SimpleNamespace

        import pageindex_mutil.utils as utils_mod

        class _FakeLoader:
            def load(self, user_opt=None):
                return SimpleNamespace(tau_high=0.9, tau_low=0.2)

        monkeypatch.setattr(utils_mod, "ConfigLoader", _FakeLoader)
        v = _verifier_cls()("qwen-plus")
        assert v.TAU_HIGH == 0.9
        assert v.TAU_LOW == 0.2

    def test_threshold_routing(self, monkeypatch):
        """combined ≥ tau_high → answer；[tau_low, tau_high) → expand；< tau_low → refuse。"""
        captured = {}

        def fake_llm(model, prompt, **kwargs):
            return captured["payload"]

        monkeypatch.setattr(_verifier_mod(), "llm_completion", fake_llm)
        v = _verifier_cls()("qwen-plus")
        v.TAU_HIGH, v.TAU_LOW = 0.7, 0.4

        def with_conf(conf):
            return json.dumps({
                "based_on_context": True, "sufficient": True,
                "evidence_quote": "实锤", "confidence": conf,
            })

        captured["payload"] = with_conf(0.6)  # 0.3 + 0.42 = 0.72 ≥ 0.7
        assert v.verify("a", long_context(), "q", 3, 10).action == "answer"

        captured["payload"] = with_conf(0.5)  # 0.3 + 0.35 = 0.65 → expand
        assert v.verify("a", long_context(), "q", 3, 10).action == "expand"

        captured["payload"] = with_conf(0.5)  # 空上下文 s_ret=0 → 0.35 < 0.4 → refuse
        assert v.verify("a", "", "q", 0, 0).action == "refuse"


class TestHeuristicFallback:
    """LLM 失败/沉默/坏 JSON → 保留纯启发式 s_ret 回退（不走偏严闸门）。"""

    def test_llm_silent_falls_back_to_s_ret(self, monkeypatch):
        monkeypatch.setattr(_verifier_mod(), "llm_completion", lambda *a, **k: None)
        v = _verifier_cls()("qwen-plus")
        ctx = long_context()
        result = v.verify("ans", ctx, "q", source_docs=3, covered_nodes=10)
        expected = v._score_retrieval(ctx, 3, 10)
        assert result.action == "answer"
        assert abs(result.confidence - expected) < 1e-9

    def test_llm_raises_falls_back_to_s_ret(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("llm unavailable")

        monkeypatch.setattr(_verifier_mod(), "llm_completion", boom)
        v = _verifier_cls()("qwen-plus")
        ctx = long_context()
        result = v.verify("ans", ctx, "q", source_docs=1, covered_nodes=2)
        expected = v._score_retrieval(ctx, 1, 2)
        assert result.action == "answer"
        assert abs(result.confidence - expected) < 1e-9

    def test_malformed_json_falls_back_to_s_ret(self, monkeypatch):
        monkeypatch.setattr(
            _verifier_mod(), "llm_completion", lambda *a, **k: "这不是JSON"
        )
        v = _verifier_cls()("qwen-plus")
        ctx = long_context()
        result = v.verify("ans", ctx, "q", source_docs=3, covered_nodes=10)
        expected = v._score_retrieval(ctx, 3, 10)
        assert result.action == "answer"
        assert abs(result.confidence - expected) < 1e-9


class TestVerifierContextBudgetAndNeed:
    """[S8] verifier 上下文预算可配（verifier_context_chars）+ need 点名输出。"""

    def test_verifier_context_budget_truncates_prompt(self, monkeypatch):
        """ctx_budget 生效（不依赖 config.yaml）：context[:ctx_budget] 进 prompt。"""
        v = _verifier_cls()("qwen-plus", retrieve_model="r")
        monkeypatch.setattr(v, "ctx_budget", 300)
        captured = {}
        monkeypatch.setattr(
            _verifier_mod(), "llm_completion",
            lambda *a, **k: captured.setdefault("p", a[1]),
        )
        long_ctx = "证据内容。" * 600  # 3000 字符
        v.verify("答案", long_ctx, "查询", 2, 3)
        # 只截取前 300 字符进 prompt，全文不得出现
        assert long_ctx[:300] in captured["p"]
        assert long_ctx not in captured["p"]

    def test_verifier_parses_need_field(self, monkeypatch):
        """sufficient=false 且给出 need 列表 → 规整进 VerifyResult.need。"""
        v = _verifier_cls()("qwen-plus", retrieve_model="r")
        monkeypatch.setattr(
            _verifier_mod(), "llm_completion",
            lambda *a, **k: (
                '{"based_on_context": false, "sufficient": false, '
                '"evidence_quote": "", "confidence": 0.3, '
                '"need": [{"doc_id": "d2", "reason": "缺该文档证据"}]}'
            ),
        )
        res = v.verify("答案", "上下文", "查询", 1, 2)
        assert res.need == [{"doc_id": "d2", "reason": "缺该文档证据"}]

    def test_verifier_normalizes_need_skips_invalid_entries(self):
        """_normalize_need：跳过非 dict / 缺对象键条目，保留 node_id/page。"""
        norm = _verifier_cls()._normalize_need
        raw = [
            {"doc_id": "d1", "reason": "缺 d1"},
            {"node_id": "n3", "page": 5},
            {"doc_id": "d2"},
            {"reason": "缺对象键，应跳过"},
            "not-a-dict",
            None,
        ]
        assert norm(raw) == [
            {"doc_id": "d1", "reason": "缺 d1"},
            {"node_id": "n3", "page": 5, "reason": ""},
            {"doc_id": "d2", "reason": ""},
        ]
        # 非法整体输入 → []
        assert norm(None) == []
        assert norm("need") == []
        assert norm({"doc_id": "d1"}) == []

    def test_verifier_normalizes_page_str_and_none_reason(self):
        """page 数字字符串规整为 int（非法省略）；reason None 规整为空串。"""
        norm = _verifier_cls()._normalize_need
        raw = [
            {"doc_id": "d1", "page": "5", "reason": None},
            {"node_id": "n2", "page": "abc", "reason": "缺 n2"},
            {"doc_id": "d3", "page": 7},
        ]
        assert norm(raw) == [
            {"doc_id": "d1", "page": 5, "reason": ""},
            {"node_id": "n2", "reason": "缺 n2"},
            {"doc_id": "d3", "page": 7, "reason": ""},
        ]


class TestCoerceCtxBudget:
    """[S8] _coerce_ctx_budget 规整：正整数原样返回，0/负/非数值/溢出回退 8000。"""

    def test_valid_value_passthrough(self):
        coerce = _verifier_cls()._coerce_ctx_budget
        assert coerce(4000) == 4000
        assert coerce("4000") == 4000

    def test_invalid_falls_back_to_8000(self):
        coerce = _verifier_cls()._coerce_ctx_budget
        assert coerce(0) == 8000
        assert coerce(-5) == 8000
        assert coerce("abc") == 8000
        assert coerce(None) == 8000
        assert coerce(float("inf")) == 8000
