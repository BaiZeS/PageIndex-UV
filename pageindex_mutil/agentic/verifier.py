import logging
from dataclasses import dataclass

from ..utils import llm_completion, extract_json, count_tokens


@dataclass
class VerifyResult:
    confidence: float
    action: str  # "answer" | "expand" | "refuse"


class CRAGVerifier:
    TAU_HIGH = 0.7
    TAU_LOW = 0.4

    # Retrieval scoring constants
    _TOKEN_BUDGET = 4000
    _MAX_SOURCE_DOCS = 3
    _MAX_COVERED_NODES = 10
    _TOKEN_WEIGHT = 0.4
    _DOC_WEIGHT = 0.3
    _NODE_WEIGHT = 0.3

    def __init__(self, model: str):
        self.model = model

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

        prompt = f"""你是一个答案质量评估专家。基于以下信息判断答案的置信度。

问题: {query}

检索到的上下文（部分）:
{context[:2000]}

生成的答案:
{answer}

请评估:
1. 答案是否基于上下文中的事实？（是/否/部分）
2. 上下文是否充分回答了问题？（充分/不充分）
3. 整体置信度（0.0-1.0）

返回JSON格式: {{"based_on_context": true, "sufficient": true, "confidence": 0.85}}
直接返回JSON，不要其他内容。
"""
        try:
            response = llm_completion(self.model, prompt)
            if not response:
                return VerifyResult(confidence=s_ret, action="answer")

            data = extract_json(response)
            if not isinstance(data, dict):
                return VerifyResult(confidence=s_ret, action="answer")

            s_cov = float(data.get("confidence", s_ret))
            based = self._to_bool(data.get("based_on_context", True))
            sufficient = self._to_bool(data.get("sufficient", True))

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
