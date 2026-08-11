"""Entity extraction module for cross-document relationship graph.

Extracts entities (people, projects, organizations, concepts) from documents
and builds a relationship graph for cross-document queries.
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .utils import llm_completion, extract_json

logger = logging.getLogger(__name__)

# --- 实体消歧 blocking（轻量候选预裁剪）参数 ---
# 批归一分块大小：单次 LLM prompt 最多喂入的名字数，防止大语料爆上下文
NORMALIZE_BATCH_CHUNK_SIZE = 200
# 批归一类型分期：主动批归一先只覆盖 person/project/organization，concept 后置。
# 注意：增量单实体消歧 disambiguate_entity 不受此限制，所有类型仍走 _resolve_entity。
BATCH_NORMALIZE_ENTITY_TYPES = ["person", "project", "organization"]
# 字符集 Jaccard 阈值：≥ 该值视为疑似同簇
BLOCKING_JACCARD_THRESHOLD = 0.5
# 前缀匹配最小长度：单字前缀太宽泛，不作为信号
BLOCKING_MIN_PREFIX_LEN = 2
# 送入 LLM 裁定的候选上限（按信号强度取 top-N）
BLOCKING_MAX_CANDIDATES = 20


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein 编辑距离（两行 DP，无外部依赖）。"""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _blocking_score(query_names: List[str], candidate_names: List[str]) -> Optional[float]:
    """blocking 轻量信号：新实体名/别名 vs 候选实体名/别名，任一命中返回强度分，否则 None。

    只做 LLM 前置裁剪（疑似同簇才送 LLM 裁定），不直接决定合并；
    保守原则：信号宁缺毋滥，漏掉的疑似对由后续批归一兜底。

    信号（case-insensitive，strip 后比较）：
      1. 精确匹配            → 1.0
      2. 前缀匹配（≥2 字）   → 0.9
      3. 字符集 Jaccard ≥ BLOCKING_JACCARD_THRESHOLD → jaccard 值
      4. 编辑距离 ≤ max(2, 较长串长 // 4) 且至少共享一个字符 → 强度分
         （附加共享字符约束：极短串的编辑距离 ≤2 可能只是整串替换，
           不构成相似信号；"小张" vs "张三" 距离=2 且共享 "张" → 疑似对）
    """
    q = [n.strip().casefold() for n in query_names if n and n.strip()]
    c = [n.strip().casefold() for n in candidate_names if n and n.strip()]
    if not q or not c:
        return None

    best: Optional[float] = None
    for a in q:
        for b in c:
            score: Optional[float] = None
            if a == b:
                score = 1.0
            else:
                prefix_len = len(a) if b.startswith(a) else (len(b) if a.startswith(b) else 0)
                if prefix_len >= BLOCKING_MIN_PREFIX_LEN:
                    score = 0.9
            if score is None:
                set_a, set_b = set(a), set(b)
                inter = len(set_a & set_b)
                if inter:
                    jaccard = inter / len(set_a | set_b)
                    if jaccard >= BLOCKING_JACCARD_THRESHOLD:
                        score = jaccard
                    threshold = max(2, max(len(a), len(b)) // 4)
                    dist = _edit_distance(a, b)
                    if dist <= threshold:
                        # 强度下限 0.1：命中 blocking 即有最低强度，避免等长全替换对得 0 分
                        edit_score = max(0.1, 1.0 - dist / max(len(a), len(b), 1))
                        score = max(score or 0.0, edit_score)
            if score is not None and (best is None or score > best):
                best = score
    return best


@dataclass
class Entity:
    """Represents an extracted entity."""
    name: str
    entity_type: str  # person, project, organization, concept
    aliases: List[str] = None
    confidence: float = 0.0


@dataclass
class EntityRelation:
    """Represents a relationship between entities."""
    subject: str
    predicate: str  # works_on, authored, related_to, mentions, etc.
    object: str
    confidence: float = 0.0


class EntityExtractor:
    """Extract entities and relationships from document content."""

    def __init__(self, model: str, retrieve_model: str = None):
        self.model = model
        self.retrieve_model = retrieve_model

    def extract_entities(
        self, 
        doc_name: str, 
        doc_description: str, 
        node_titles: List[str],
        node_summaries: List[str] = None
    ) -> List[Entity]:
        """Extract entities from document metadata.
        
        Args:
            doc_name: Document filename
            doc_description: Document description
            node_titles: List of section titles
            node_summaries: Optional list of section summaries
            
        Returns:
            List of extracted entities
        """
        # Prepare context
        titles_text = "\n".join("- " + t for t in node_titles[:20])  # Limit to 20 titles
        summaries_text = ""
        if node_summaries:
            summaries_text = "\n".join("- " + s for s in node_summaries[:10])  # Limit to 10 summaries
        
        # Build prompt sections
        desc_part = doc_description[:500] if doc_description else "无"
        summary_section = ""
        if summaries_text:
            summary_section = "\n章节摘要:\n" + summaries_text
        
        prompt = (
            "你是一个实体提取专家。从以下文档信息中提取所有重要的实体（人物、项目、组织、概念）。\n\n"
            "文档名: " + doc_name + "\n"
            "文档描述: " + desc_part + "\n"
            "主要章节:\n" + titles_text + "\n"
            + summary_section + "\n\n"
            "要求:\n"
            "1. 提取所有可以识别的实体\n"
            "2. 每个实体包含名称和类型（person/project/organization/concept）\n"
            "3. 如果有别名或简称，一并列出\n"
            "4. 给出每个实体的置信度（0.0-1.0）\n"
            "5. 使用中文输出\n\n"
            "返回JSON格式:\n"
            "[\n"
            '    {"name": "实体名", "type": "person", "aliases": ["别名"], "confidence": 0.9},\n'
            "    ...\n"
            "]\n"
            "直接返回JSON数组，不要其他内容。"
        )

        try:
            response = llm_completion(self.retrieve_model or self.model, prompt, thinking_disabled=True)
            if not response:
                return []
            
            data = extract_json(response)
            if not isinstance(data, list):
                return []
            
            entities = []
            for item in data:
                if isinstance(item, dict):
                    name = item.get("name", "").strip()
                    if not name:
                        continue
                    
                    entity_type = item.get("type", "concept").lower()
                    if entity_type not in ["person", "project", "organization", "concept"]:
                        entity_type = "concept"
                    
                    aliases = item.get("aliases", [])
                    if isinstance(aliases, str):
                        aliases = [aliases]
                    
                    confidence = float(item.get("confidence", 0.5))
                    
                    entities.append(Entity(
                        name=name,
                        entity_type=entity_type,
                        aliases=aliases,
                        confidence=confidence
                    ))
            
            return entities
            
        except Exception as e:
            logger.warning("Entity extraction failed: %s", e)
            return []

    def extract_relations(
        self,
        doc_name: str,
        entities: List[Entity],
        node_titles: List[str],
        node_summaries: List[str] = None
    ) -> List[EntityRelation]:
        """Extract relationships between entities.
        
        Args:
            doc_name: Document filename
            entities: List of extracted entities
            node_titles: List of section titles
            node_summaries: Optional list of section summaries
            
        Returns:
            List of entity relationships
        """
        if len(entities) < 2:
            return []
        
        # Prepare entity list
        entity_names = [e.name for e in entities[:30]]  # Limit to 30 entities
        entities_text = ", ".join(entity_names)
        
        # Prepare context
        titles_text = "\n".join("- " + t for t in node_titles[:20])
        
        prompt = (
            "你是一个关系提取专家。从以下文档信息中提取实体之间的关系。\n\n"
            "文档名: " + doc_name + "\n"
            "文档中的实体: " + entities_text + "\n\n"
            "主要章节:\n" + titles_text + "\n\n"
            "要求:\n"
            "1. 识别实体之间的关系\n"
            "2. 关系类型包括: works_on, authored, related_to, mentions, part_of, manages, etc.\n"
            "3. 只提取有明确证据的关系\n"
            "4. 给出每个关系的置信度（0.0-1.0）\n\n"
            "返回JSON格式:\n"
            "[\n"
            '    {"subject": "实体A", "predicate": "works_on", "object": "实体B", "confidence": 0.8},\n'
            "    ...\n"
            "]\n"
            "直接返回JSON数组，不要其他内容。"
        )

        try:
            response = llm_completion(self.retrieve_model or self.model, prompt, thinking_disabled=True)
            if not response:
                return []
            
            data = extract_json(response)
            if not isinstance(data, list):
                return []
            
            # Validate entities exist
            valid_names = set(entity_names)
            relations = []
            
            for item in data:
                if isinstance(item, dict):
                    subject = item.get("subject", "").strip()
                    predicate = item.get("predicate", "").strip()
                    obj = item.get("object", "").strip()
                    
                    if not all([subject, predicate, obj]):
                        continue
                    
                    # Check if entities are valid (allow fuzzy matching)
                    if subject not in valid_names:
                        # Try to find close match
                        subject = self._fuzzy_match(subject, valid_names) or subject
                    if obj not in valid_names:
                        obj = self._fuzzy_match(obj, valid_names) or obj
                    
                    confidence = float(item.get("confidence", 0.5))
                    
                    relations.append(EntityRelation(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        confidence=confidence
                    ))
            
            return relations
            
        except Exception as e:
            logger.warning("Relation extraction failed: %s", e)
            return []

    def _fuzzy_match(self, name: str, valid_names: set) -> Optional[str]:
        """Fuzzy match a name against valid entity names."""
        name_lower = name.lower()
        for valid_name in valid_names:
            if name_lower in valid_name.lower() or valid_name.lower() in name_lower:
                return valid_name
            # Check for high overlap
            if len(set(name_lower) & set(valid_name.lower())) / max(len(name_lower), 1) > 0.7:
                return valid_name
        return None

    def extract_from_document(
        self,
        doc_name: str,
        doc_description: str,
        structure: List[Dict]
    ) -> Tuple[List[Entity], List[EntityRelation]]:
        """Extract entities and relations from a document structure.

        Args:
            doc_name: Document filename
            doc_description: Document description
            structure: Document tree structure

        Returns:
            Tuple of (entities, relations)
        """
        # Flatten structure to get titles, summaries, and context snippets
        node_titles = []
        node_summaries = []
        node_contexts = []  # (title, summary) pairs for context extraction

        def flatten_nodes(nodes, depth=0):
            for node in nodes:
                title = node.get("title", "")
                summary = node.get("summary", "")
                if title:
                    node_titles.append(title)
                if summary:
                    node_summaries.append(summary)
                if title or summary:
                    node_contexts.append(f"{title}: {summary}")

                children = node.get("nodes", [])
                if children and depth < 3:
                    flatten_nodes(children, depth + 1)

        flatten_nodes(structure)

        # Extract entities
        entities = self.extract_entities(
            doc_name, doc_description, node_titles, node_summaries
        )

        # Extract relations
        relations = self.extract_relations(
            doc_name, entities, node_titles, node_summaries
        )

        return entities, relations, node_contexts

    def disambiguate_entity(
        self,
        new_name: str,
        new_aliases: List[str],
        existing_entities: List[Dict],
    ) -> Optional[Dict]:
        """Decide whether a new entity should merge with an existing one.

        Blocking 预裁剪 + LLM 裁定（retrieve_model per NFR4）:
        1. 轻量信号（_blocking_score）从同类型已有实体中筛出疑似同簇候选；
           无候选命中 → 直接跳过 LLM（保守不合并）。
        2. 仅对信号强度 top-N 的候选调 LLM 裁是否合并。
        返回应合入的 canonical 实体 dict，或 None（不确定/失败一律不合并）。
        """
        if not existing_entities:
            return None

        # --- Blocking: 只把疑似同簇候选喂给 LLM，调用量 O(全量) → O(疑似对) ---
        query_names = [new_name] + list(new_aliases or [])
        blocked: List[Tuple[float, Dict, List[str]]] = []
        for entity in existing_entities:
            aliases_raw = entity.get("aliases", "[]") or "[]"
            if isinstance(aliases_raw, str):
                try:
                    cand_aliases = json.loads(aliases_raw)
                    if not isinstance(cand_aliases, list):
                        cand_aliases = []
                except ValueError:
                    cand_aliases = []
            else:
                cand_aliases = list(aliases_raw)
            score = _blocking_score(query_names, [entity["name"]] + cand_aliases)
            if score is not None:
                blocked.append((score, entity, cand_aliases))

        if not blocked:
            # 无疑似候选 → 跳过 LLM，保守不合并
            return None

        blocked.sort(key=lambda item: item[0], reverse=True)
        candidates = blocked[:BLOCKING_MAX_CANDIDATES]

        candidates_text = json.dumps(
            [{"name": e["name"], "aliases": aliases}
             for _, e, aliases in candidates],
            ensure_ascii=False,
        )

        prompt = (
            "你是实体消歧专家。给定一个新实体和已有实体列表，判断新实体是否与某个已有实体是同一人/事/物。\n\n"
            f"新实体名称: {new_name}\n"
            f"新实体别名: {json.dumps(new_aliases, ensure_ascii=False)}\n\n"
            f"已有实体列表:\n{candidates_text}\n\n"
            "要求:\n"
            "1. 如果新实体与某个已有实体是同一人/事/物，返回该已有实体的名称\n"
            "2. 如果不确定或不是同一实体，返回 null\n"
            "3. 考虑别名、简称、同义词等因素\n\n"
            '返回JSON: {"should_merge": true/false, "canonical_name": "实体名或null", "reason": "理由"}\n'
            "直接返回JSON，不要其他内容。"
        )

        try:
            response = llm_completion(self.retrieve_model or self.model, prompt, thinking_disabled=True)
            if not response:
                return None

            data = extract_json(response)
            if not isinstance(data, dict):
                return None

            if not data.get("should_merge"):
                return None

            canonical_name = data.get("canonical_name")
            if not canonical_name:
                return None

            # Find the matching candidate (LLM 只见过 blocking 幸存者)
            for _, entity, _aliases in candidates:
                if entity["name"] == canonical_name:
                    return entity

            return None

        except Exception as e:
            logger.warning("Entity disambiguation failed: %s", e)
            return None

    def normalize_entities_batch(self, db) -> None:
        """Batch entity normalization: LLM merges synonyms per type group.

        Mirrors the _normalize_tags pattern in corpus_tree.py:
        - Collect all unique entities grouped by type
        - For each type: LLM merges synonyms → canonical entity map
        - Apply: merge duplicate entities, update mentions
        - Conservative on failure (identity mapping)
        """
        # 类型分期：主动批归一先只覆盖 person/project/organization，concept 后置
        # （增量单实体消歧 disambiguate_entity 不受限，仍覆盖所有类型）
        for etype in BATCH_NORMALIZE_ENTITY_TYPES:
            entities = db.get_entities_by_type(etype)
            if len(entities) <= 1:
                continue
            names = [e["name"] for e in entities]
            mapping = self._normalize_entities_llm(names)
            # Apply canonical map: group by canonical → merge duplicates
            canonical_groups: dict[str, list] = {}
            for entity in entities:
                canonical = mapping.get(entity["name"], entity["name"])
                canonical_groups.setdefault(canonical, []).append(entity)
            for canonical, group in canonical_groups.items():
                if len(group) <= 1:
                    continue
                # Find or use the first entity as canonical
                canonical_entity = None
                for e in group:
                    if e["name"] == canonical:
                        canonical_entity = e
                        break
                if canonical_entity is None:
                    canonical_entity = group[0]
                for e in group:
                    if e["id"] == canonical_entity["id"]:
                        continue
                    try:
                        db.merge_entities(canonical_entity["id"], e["id"])
                    except Exception as ex:
                        logger.warning("Entity merge failed (%s → %s): %s",
                                       e["name"], canonical, ex)

    def _normalize_entities_llm(self, names: list[str]) -> dict[str, str]:
        """LLM merges synonym entity names → raw→canonical mapping.

        批归一分块（bounded chunk）+ map-reduce：
        - map：按 NORMALIZE_BATCH_CHUNK_SIZE 分块，块内 LLM 归一；
        - reduce：多块时对各块归一后的代表元（canonical 名）再做一轮合并，
          杜绝大语料单 prompt 爆上下文。
        Mirrors _normalize_tags in corpus_tree.py. Conservative on failure.
        """
        mapping = {n: n for n in names}
        if len(names) <= 1:
            return mapping
        chunks = [names[i:i + NORMALIZE_BATCH_CHUNK_SIZE]
                  for i in range(0, len(names), NORMALIZE_BATCH_CHUNK_SIZE)]
        if len(chunks) == 1:
            return self._normalize_chunk_llm(names)
        # map: 各块独立归一
        for chunk in chunks:
            mapping.update(self._normalize_chunk_llm(chunk))
        # reduce: 块间代表元合一轮（保守：合并结果以 LLM 返回为准）
        representatives = list(dict.fromkeys(mapping.values()))
        if len(representatives) > 1:
            consolidation = self._normalize_chunk_llm(representatives)
            mapping = {raw: consolidation.get(canonical, canonical)
                       for raw, canonical in mapping.items()}
        return mapping

    def _normalize_chunk_llm(self, names: list[str]) -> dict[str, str]:
        """单次 LLM 归一（单块或 reduce 合并轮）→ raw→canonical mapping。

        Conservative on failure: 恒等映射（不合并）。
        """
        mapping = {n: n for n in names}
        if len(names) <= 1:
            return mapping
        prompt = (
            "你是一个实体归一化专家。以下是从语料库全部文档中抽取的实体名称集合，"
            "其中可能存在同义或近义实体（例如\"张三\"与\"张先生\"同义）。\n"
            "请将同义/近义实体合并，输出规范实体集。\n\n"
            f"实体全集：\n{json.dumps(names, ensure_ascii=False)}\n\n"
            "要求：\n"
            "1. 含义相同或高度相近的实体必须合并为同一个规范实体\n"
            "2. 规范实体名选用其中最清晰、最通用的表述\n"
            "3. 含义不同的实体保持各自独立，自成规范实体\n"
            "4. 每个输入实体都必须归入一个规范实体，不得遗漏\n\n"
            '返回JSON格式：{"groups": [{"canonical": "规范实体", "synonyms": ["原实体1", "原实体2"]}, ...]}\n'
            "直接返回最终JSON结构，不要输出其他内容。"
        )
        try:
            response = llm_completion(self.retrieve_model or self.model, prompt, thinking_disabled=True)
            if not response:
                return mapping
            data = extract_json(response)
            if not isinstance(data, dict):
                return mapping
            for group in data.get("groups", []):
                if not isinstance(group, dict):
                    continue
                canonical = str(group.get("canonical", "")).strip()
                if not canonical:
                    continue
                for syn in group.get("synonyms", []):
                    syn = str(syn).strip()
                    if syn in mapping:
                        mapping[syn] = canonical
            return mapping
        except Exception as e:
            logger.warning("Entity batch normalization failed: %s", e)
            return mapping
