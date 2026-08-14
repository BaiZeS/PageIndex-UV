"""语料级标签归一化共享入口（P2 简化后）。

[S3] 语料树简化替代（用户 2026-08-13 拍板）：删除聚类层级构建（LLM 聚类 + 簇命名——
评审实证在统一单链下零推理消费者），改产文档级接地摘要 ``doc_summary``（索引期 LLM 生成，
见 client._generate_doc_summary）。

本模块保留语料级标签归一化的共享入口 ``resolve_new_tag``：
- 写入者 ``ClosetIndex._anchor_tags``（[7.2] 标签词表锚定）独立幸存；
- ``corpus_tag_norm`` 读写与语料树表结构保留（迁移安全）；
- 推理期 super_tree 的 navigate_tree/_aggregate_cluster_profile 路径由 T13 另行删除。
"""
import json
import logging
from typing import Optional

from .utils import llm_completion, extract_json


def resolve_new_tag(db, model: Optional[str], raw_tag: str) -> str:
    """增量标签归一（共享）：单点 LLM 裁定"并入已有规范标签 or 新开"。

    语料级标签集稳定的公共入口（[7.2]）：新文档抽取出的新标签先与已有
    canonical 集比对，语义近似则复用既有 canonical 名，不新造。被
    ClosetIndex._anchor_tags 复用。

    LLM 返回的 canonical 必须命中已有规范集（否则视为幻觉）；LLM 失败、
    无响应或幻觉一律退回原标签（新开），不破坏确定性。
    """
    canonical_tags = db.get_corpus_canonical_tags()
    if not canonical_tags:
        return raw_tag
    prompt = f"""你是一个语义标签归一化专家（标签归一化·增量单标签裁定）。已有规范标签集：
{json.dumps(canonical_tags, ensure_ascii=False)}

新文档产生了一个新标签："{raw_tag}"
请裁定：该标签应并入哪个已有规范标签（同义/近义），还是作为新的规范标签？

要求：
1. 与已有规范标签同义或高度近义时，并入该规范标签
2. 否则作为新的规范标签（可直接使用原标签名）

返回JSON格式：{{"canonical": "规范标签名"}}
直接返回最终JSON结构，不要输出其他内容。"""
    try:
        response = llm_completion(model, prompt, thinking_disabled=True)
        data = extract_json(response) if response else None
    except Exception as e:
        logging.warning("resolve_new_tag LLM call failed: %s", e)
        return raw_tag
    if isinstance(data, dict):
        canonical = str(data.get("canonical", "")).strip()
        # 只接受已有规范标签（并入）；其他名字视为幻觉，退回原标签新开
        if canonical and canonical in canonical_tags:
            return canonical
    return raw_tag
