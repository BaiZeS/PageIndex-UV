import logging
from dataclasses import dataclass

from ..utils import llm_completion, extract_json, count_tokens


@dataclass
class VerifyResult:
    confidence: float
    action: str  # "answer" | "expand" | "refuse"


class CRAGVerifier:
    # 偏严取向（[7.5]a）：tau_high 收紧，更难判 answer，倾向触发 expand 补充召回。
    # config.yaml 的 tau_high/tau_low 可覆盖（见 _init_from_config）。
    TAU_HIGH = 0.75
    TAU_LOW = 0.4

    # Retrieval scoring constants
    _TOKEN_BUDGET = 4000
    _MAX_SOURCE_DOCS = 3
    _MAX_COVERED_NODES = 10
    _TOKEN_WEIGHT = 0.4
    _DOC_WEIGHT = 0.3
    _NODE_WEIGHT = 0.3

    def __init__(self, model: str, retrieve_model: str = None):
        self.model = model
        self.retrieve_model = retrieve_model
        self._init_from_config()

    def _init_from_config(self):
        """Override class defaults with config.yaml values if present."""
        try:
            from ..utils import ConfigLoader
            cfg = ConfigLoader().load(None)
            self.TAU_HIGH = getattr(cfg, "tau_high", self.TAU_HIGH)
            self.TAU_LOW = getattr(cfg, "tau_low", self.TAU_LOW)
        except Exception:
            pass

    def _score_retrieval(
        self, context: str, source_docs: int, covered_nodes: int
    ) -> float:
        tokens = count_tokens(context)
        token_score = min(tokens / self._TOKEN_BUDGET, 1.0)
        doc_score = min(source_docs / self._MAX_SOURCE_DOCS, 1.0)
        node_score = min(covered_nodes / self._MAX_COVERED_NODES, 1.0)
        return (
            token_score * self._TOKEN_WEIGHT
            + doc_score * self._DOC_WEIGHT
            + node_score * self._NODE_WEIGHT
        )

    @staticmethod
    def _to_bool(val) -> bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "yes", "是", "1", "y")
        return bool(val)

    def verify(
        self,
        answer: str,
        context: str,
        query: str,
        source_docs: int = 0,
        covered_nodes: int = 0,
    ) -> VerifyResult:
        s_ret = self._score_retrieval(context, source_docs, covered_nodes)

        # [7.5]a 偏严取向：判据是"上下文是否支撑答案"，不做答案自评。
        # 要求从上下文逐字引用支撑证据（evidence_quote）；引用不出实锤视同未接地。
        prompt = f"""你是一个证据充分性评审专家。你的任务不是评价答案写得好不好，而是判断"上下文是否支撑答案"：答案中的关键论断必须能在上下文中找到依据。

问题: {query}

检索到的上下文（部分）:
{context[:2000]}

生成的答案:
{answer}

请严格评估:
1. based_on_context: 答案的关键论断是否都能在上下文中找到依据？（true/false）
2. sufficient: 上下文是否足以准确回答该问题？（true/false）
3. evidence_quote: 从上下文中逐字引用一段支撑答案关键论断的证据片段；找不到可引用的实锤则返回空字符串 ""
4. confidence: 上下文对答案的支撑度（0.0-1.0）

判据从严：答案若引用不出上下文实锤（evidence_quote 为空），based_on_context 应为 false，且 confidence 应相应降低。宁可触发补充召回，也不放行没有上下文支撑的答案。

返回JSON格式: {{"based_on_context": true, "sufficient": true, "evidence_quote": "上下文中的原文片段", "confidence": 0.85}}
直接返回JSON，不要其他内容。
"""
        try:
            response = llm_completion(self.retrieve_model or self.model, prompt, thinking_disabled=False)
            if not response:
                return VerifyResult(confidence=s_ret, action="answer")

            data = extract_json(response)
            # LLM 失败/空/坏 JSON → 纯启发式 s_ret 回退（保留既有行为）
            if not isinstance(data, dict) or not any(
                k in data for k in ("based_on_context", "sufficient", "confidence")
            ):
                return VerifyResult(confidence=s_ret, action="answer")

            s_cov = float(data.get("confidence", s_ret))
            based = self._to_bool(data.get("based_on_context", True))
            sufficient = self._to_bool(data.get("sufficient", True))
            evidence_quote = str(data.get("evidence_quote") or "").strip()
            # 高置信判 answer 的必要条件：能引用出上下文实锤；引用不出视同未接地。
            if not evidence_quote:
                based = False

            combined = s_ret * 0.3 + s_cov * 0.7
            if not based or not sufficient:
                combined *= 0.5

            if combined >= self.TAU_HIGH:
                return VerifyResult(confidence=combined, action="answer")
            elif combined >= self.TAU_LOW:
                return VerifyResult(confidence=combined, action="expand")
            else:
                return VerifyResult(confidence=combined, action="refuse")
        except Exception as e:
            logging.warning(f"Verification failed: {e}")
            return VerifyResult(confidence=s_ret, action="answer")
