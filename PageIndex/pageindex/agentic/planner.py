import logging
from dataclasses import dataclass
from typing import List, Dict

from ..utils import llm_acompletion, extract_json


@dataclass
class PlanResult:
    queries: List[str]
    weights: Dict[str, float]
    query_type: str


class RetrievalPlanner:
    def __init__(self, model: str):
        self.model = model

    async def plan(self, query: str) -> PlanResult:
        prompt = f"""你是一个检索策略规划专家。分析以下用户问题，决定最佳检索策略。

用户问题: {query}

请完成以下任务:
1. 判断问题类型: factual(事实查询) / analytical(分析推理) / comparative(对比查询) / vague(模糊查询)
2. 生成假设答案(HyDE): 基于问题生成一个简短假设答案（1-2句话）
3. 基于假设答案提取2个不同角度的查询变体（更具体或更宽泛的表述）
4. 分配三策略权重(0-1之间): metadata(文档名/描述关键词匹配), semantics(语义标签索引), description(LLM描述相关性判断)

返回JSON格式:
{{
    "query_type": "factual",
    "hyde_answer": "假设答案...",
    "query_variants": ["变体1", "变体2"],
    "weights": {{"metadata": 0.2, "semantics": 0.5, "description": 0.3}}
}}

直接返回JSON，不要其他内容。
"""
        try:
            response = await llm_acompletion(self.model, prompt)
            if not response:
                return self._default_plan(query)
            data = extract_json(response)
            if not isinstance(data, dict):
                return self._default_plan(query)

            query_type = data.get("query_type", "factual")
            hyde = data.get("hyde_answer", "")
            variants = data.get("query_variants", [])
            weights = data.get("weights", {})

            queries = [query]
            if hyde:
                queries.append(hyde)
            for v in variants:
                if v and v != query and v not in queries:
                    queries.append(v)

            default_weights = {"metadata": 0.2, "semantics": 0.5, "description": 0.3}
            for k, v in default_weights.items():
                weights.setdefault(k, v)

            # Normalize weights to sum to 1.0
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

            return PlanResult(queries=queries, weights=weights, query_type=query_type)
        except Exception as e:
            logging.warning(f"Planner failed: {e}")
            return self._default_plan(query)

    def _default_plan(self, query: str) -> PlanResult:
        return PlanResult(
            queries=[query],
            weights={"metadata": 0.2, "semantics": 0.5, "description": 0.3},
            query_type="factual",
        )
