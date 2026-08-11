import logging
from typing import List, Tuple

try:
    import jieba
except ImportError:
    jieba = None

from .closet_index import _STOPWORDS


class KeywordIndex:
    """jieba-based inverted index for document name and description."""

    def __init__(self, db):
        self.db = db
        if jieba is None:
            logging.warning("jieba not installed; KeywordIndex will be unavailable")

    def _tokenize(self, text: str) -> List[str]:
        if not text or jieba is None:
            return []
        tokens = jieba.lcut(text)
        return [
            t.strip().lower()
            for t in tokens
            if len(t.strip()) > 1 and t.strip().lower() not in _STOPWORDS
        ]

    def add_document(self, doc_id: int, doc_name: str, doc_description: str,
                     node_titles: List[str] = None, content: str = None) -> None:
        records = []
        for token in self._tokenize(doc_name):
            records.append((doc_id, token, "name"))
        for token in self._tokenize(doc_description or ""):
            records.append((doc_id, token, "description"))
        if node_titles:
            for title in node_titles:
                for token in self._tokenize(title):
                    records.append((doc_id, token, "node_title"))
        # Index full document body (unique tokens) so keyword search covers content,
        # not just titles/descriptions — critical for short-passage retrieval.
        if content:
            for token in set(self._tokenize(content)):
                records.append((doc_id, token, "content"))
        self.db.insert_doc_keywords(doc_id, records)

    def remove_document(self, doc_id: int) -> None:
        self.db.delete_doc_keywords(doc_id)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        tokens = self._tokenize(query)
        if not tokens:
            return []
        return self.db.match_doc_keywords(tokens, top_k)


import json
from .utils import llm_completion, strip_markdown_fence


class KBIdentity:
    """Lazy-generated knowledge base identity summary."""

    def __init__(self, db, model: str, retrieve_model: str = None):
        self.db = db
        self.model = model
        self.retrieve_model = retrieve_model

    def get_identity(self) -> str:
        cached = self.db.get_kb_identity()
        if cached:
            return cached
        return self._build()

    def invalidate(self) -> None:
        with self.db._connect() as conn:
            conn.execute("DELETE FROM kb_identity WHERE id = 1")

    def _build(self) -> str:
        docs = self.db.get_all_documents()
        if not docs:
            return "知识库中暂无文档。"

        try:
            return self._generate_with_llm(docs)
        except Exception as e:
            logging.warning("KB Identity LLM generation failed: %s", e)
            return self._build_fallback(docs)

    def _generate_with_llm(self, docs) -> str:
        doc_list = []
        for doc in docs:
            top_nodes = self.db.get_top_level_nodes(doc["id"])
            sections = [n.get("title", "") for n in top_nodes if n.get("title")]
            sections_str = "、".join(sections[:5]) if sections else "无章节信息"
            doc_list.append({
                "name": doc.get("pdf_name", ""),
                "sections": sections_str,
            })

        prompt = f"""你是一个知识库管理员。给定以下文档列表，请生成一段简短的摘要（不超过200字），描述知识库的整体内容和主要主题。

文档列表：
{json.dumps(doc_list, ensure_ascii=False, indent=2)}

要求：
1. 说明文档总数
2. 概括主要主题领域
3. 不要列出每个文档的详细内容

直接返回纯文本摘要，不要输出 JSON 或其他格式。"""

        response = llm_completion(self.retrieve_model or self.model, prompt, thinking_disabled=False)
        if response:
            cleaned = strip_markdown_fence(response)
            self.db.set_kb_identity(cleaned, len(docs))
            return cleaned
        raise RuntimeError("LLM returned empty response")

    def _build_fallback(self, docs) -> str:
        names = [doc.get("pdf_name", "") for doc in docs]
        text = f"知识库共 {len(docs)} 个文档：" + "、".join(names[:10])
        if len(names) > 10:
            text += " 等"
        self.db.set_kb_identity(text, len(docs))
        return text


import asyncio
from collections import defaultdict
from typing import Set, Dict, List, Optional

from .utils import llm_acompletion, count_tokens, extract_json

# [S5] 量级自适应档位
TIER_SMALL = "small"      # 小语料直连：跳过层级，扁平语义匹配
TIER_MEDIUM = "medium"    # 中语料：单层领域路由 → 域内匹配
TIER_MASSIVE = "massive"  # 海量：完整层级树导航 + 宽层语义预筛 + 图谱加权


class _CorpusTreeView:
    """语料树内存视图：孩子/文档索引 + 子树文档集合缓存（导航期只读）。"""

    def __init__(self, nodes: List[dict], memberships: List[tuple]):
        self.nodes_by_id = {n["id"]: n for n in nodes}
        self.root_id = next(
            (n["id"] for n in nodes if n.get("kind") == "root"), None)
        if self.root_id is None:
            self.root_id = next(
                (n["id"] for n in nodes if n.get("parent_id") is None), None)
        self._children: Dict[int, List[dict]] = defaultdict(list)
        for n in nodes:
            if n.get("parent_id") is not None:
                self._children[n["parent_id"]].append(n)
        self._docs: Dict[int, List[tuple]] = defaultdict(list)
        for doc_id, node_id, weight in memberships:
            self._docs[node_id].append((doc_id, weight))
        self._subtree_cache: Dict[int, set] = {}

    def children_of(self, node_id: int) -> List[dict]:
        return self._children.get(node_id, [])

    def docs_of(self, node_id: int) -> List[tuple]:
        return self._docs.get(node_id, [])

    def subtree_doc_ids(self, node_id: int) -> set:
        """节点子树（含自身）的全部文档 id——软归属下文档可多次出现，集合天然去重。"""
        if node_id not in self._subtree_cache:
            ids = {d for d, _ in self._docs.get(node_id, [])}
            for child in self._children.get(node_id, []):
                ids |= self.subtree_doc_ids(child["id"])
            self._subtree_cache[node_id] = ids
        return self._subtree_cache[node_id]


class SuperTreeIndex:
    """L0 dual-channel prefilter + L1 Super-Tree document selection."""

    # Defaults (overridden by config.yaml via _init_from_config)
    _MAX_TOP_NODES_PER_DOC = 8
    _MAX_CANDIDATE_DOCS = 50
    _MAX_SUPER_TREE_TOKENS = 6000
    _SUMMARY_MAX_LEN = 100
    _RANK_K = 12
    _SELECT_TOP_K = 5
    _SCORE_RATIO = 0.5
    # 三层重构-选择层：map-reduce 推理选择参数
    _REASON_GROUP_SIZE = 10   # 候选 <= 该值走单次整体挑选；否则分组 map-reduce
    _REASON_KEEP_PER_GROUP = 3  # map 阶段每组保留的篇数
    # 三层重构-L0：各通道召回 top-k（并集召回，宁多勿漏，精排交给选择层）
    _L0_CHANNEL_TOPK = 30
    # 三层重构-层级：集合标签匹配的软路由加权（加性提升，绝不硬删候选）
    _HIERARCHY_BOOST_WEIGHT = 1.0
    # P2 量级自适应（[S5]）：档位文档数阈值（标定目标，可配置覆盖）
    _SMALL_MAX_DOCS = 50     # 文档数 < 该值 → 小语料直连（跳过层级遍历）
    _MASSIVE_MIN_DOCS = 500  # 文档数 >= 该值 → 海量层级树导航
    # P2 语义树导航（[S6]）
    _NODE_PREFILTER_TOPK = 20   # 宽层语义预筛 top-k
    _NARROW_LAYER_MAX = 16      # 兄弟数 <= 该值视为窄层：跳过预筛直接给 LLM
    _ENTITY_BOOST_WEIGHT = 1.0  # 实体命中加权强度（图谱信号，[S6]②）

    def __init__(self, db, model: str, client, retrieve_model: str = None):
        self.db = db
        self.model = model
        self.retrieve_model = retrieve_model
        self.client = client
        self.keyword_index = KeywordIndex(db)
        self.kb_identity = KBIdentity(db, model, retrieve_model)
        self._init_from_config()
        self._backfill_existing_docs()

    def _init_from_config(self):
        """Override class defaults with config.yaml values if present."""
        try:
            from .utils import ConfigLoader
            cfg = ConfigLoader().load(None)
            self._MAX_TOP_NODES_PER_DOC = getattr(cfg, "max_top_nodes_per_doc", self._MAX_TOP_NODES_PER_DOC)
            self._MAX_CANDIDATE_DOCS = getattr(cfg, "max_candidate_docs", self._MAX_CANDIDATE_DOCS)
            self._MAX_SUPER_TREE_TOKENS = getattr(cfg, "max_super_tree_tokens", self._MAX_SUPER_TREE_TOKENS)
            self._SUMMARY_MAX_LEN = getattr(cfg, "summary_max_len", self._SUMMARY_MAX_LEN)
            self._RANK_K = getattr(cfg, "rank_k", self._RANK_K)
            self._SELECT_TOP_K = getattr(cfg, "select_top_k", self._SELECT_TOP_K)
            self._SCORE_RATIO = getattr(cfg, "score_ratio", self._SCORE_RATIO)
            self._REASON_GROUP_SIZE = getattr(cfg, "reason_group_size", self._REASON_GROUP_SIZE)
            self._REASON_KEEP_PER_GROUP = getattr(cfg, "reason_keep_per_group", self._REASON_KEEP_PER_GROUP)
            self._L0_CHANNEL_TOPK = getattr(cfg, "l0_channel_topk", self._L0_CHANNEL_TOPK)
            self._HIERARCHY_BOOST_WEIGHT = getattr(cfg, "hierarchy_boost_weight", self._HIERARCHY_BOOST_WEIGHT)
            self._SMALL_MAX_DOCS = getattr(cfg, "scale_small_max_docs", self._SMALL_MAX_DOCS)
            self._MASSIVE_MIN_DOCS = getattr(cfg, "scale_massive_min_docs", self._MASSIVE_MIN_DOCS)
            self._NODE_PREFILTER_TOPK = getattr(cfg, "node_prefilter_topk", self._NODE_PREFILTER_TOPK)
            self._NARROW_LAYER_MAX = getattr(cfg, "narrow_layer_max", self._NARROW_LAYER_MAX)
            self._ENTITY_BOOST_WEIGHT = getattr(cfg, "entity_boost_weight", self._ENTITY_BOOST_WEIGHT)
        except Exception:
            pass

    def _get_db_to_uuid(self) -> Dict[int, str]:
        """Build reverse mapping from db_id -> uuid."""
        id_mapper = getattr(self.client, "_id_mapper", None)
        if id_mapper:
            return {db: uuid for uuid, db in id_mapper.items()}
        return {v: k for k, v in getattr(self.client, "_uuid_to_db", {}).items()}

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def on_document_added(self, db_doc_id: int) -> None:
        doc = self.db.get_document_by_id(db_doc_id)
        if doc:
            # Collect node titles from DB for keyword indexing
            node_titles = []
            try:
                top_nodes = self.db.get_top_level_nodes(db_doc_id)
                node_titles = [n.get("title", "") for n in top_nodes if n.get("title")]
            except Exception:
                pass
            # Collect full document body for content-level keyword indexing
            content = ""
            try:
                content = self.db.get_document_content(db_doc_id)
            except Exception:
                pass
            self.keyword_index.add_document(
                db_doc_id, doc.get("pdf_name", ""),
                doc.get("doc_description", ""), node_titles, content=content
            )
        self.kb_identity.invalidate()

    def on_document_removed(self, db_doc_id: int) -> None:
        self.keyword_index.remove_document(db_doc_id)
        self.kb_identity.invalidate()

    # ------------------------------------------------------------------
    # L0: Dual-channel prefilter
    # ------------------------------------------------------------------
    def prefilter(self, query: str) -> Dict[int, float]:
        """Return candidate doc_ids with cumulative channel scores.

        Channels:
          A: ClosetIndex semantic tag matching
          B: KeywordIndex inverted index
          C: Vector search via ChromaDB — explicit opt-in enhancement only
             ([S10]): NOT part of the tree-navigation main path, which is
             strictly vectorless (jieba + closet_tags inverted). Active only
             when a vector search_backend is configured (SEARCH_BACKEND=
             hybrid/chroma; deployed default is keyword).
          D: Entity graph matching
        """
        scores: Dict[int, float] = {}
        topk = self._L0_CHANNEL_TOPK

        # Channel A: tag matching (ClosetIndex)
        if hasattr(self.client, "closet_index") and self.client.closet_index:
            try:
                tag_results = self.client.closet_index.search(query, top_k=topk)
                for doc_id, score in tag_results:
                    scores[int(doc_id)] = scores.get(int(doc_id), 0.0) + float(score)
            except Exception as e:
                logging.warning("Tag matching failed: %s", e)

        # Channel B: keyword inverted index
        try:
            keyword_results = self.keyword_index.search(query, top_k=topk)
            for doc_id, score in keyword_results:
                scores[int(doc_id)] = scores.get(int(doc_id), 0.0) + float(score)
        except Exception as e:
            logging.warning("Keyword search failed: %s", e)

        # Channel C: vector search (ChromaDB)
        if hasattr(self.client, "search_backend") and self.client.search_backend:
            try:
                vector_results = self.client.search_backend.search(query, top_k=topk)
                for doc_id, score in vector_results:
                    # Weight vector results higher for semantic understanding
                    scores[int(doc_id)] = scores.get(int(doc_id), 0.0) + float(score) * 1.5
            except Exception as e:
                logging.warning("Vector search failed in prefilter: %s", e)

        # Channel D: entity graph matching
        if hasattr(self.client, "db") and self.client.db:
            try:
                entities = self.client.db.search_entities(query, limit=topk)
                for entity in entities:
                    entity_id = entity.get("id")
                    if entity_id:
                        # Find documents mentioning this entity
                        mentions = self.client.db.get_entity_documents(entity_id)
                        for mention in mentions:
                            doc_id = mention.get("id")
                            if doc_id:
                                mention_conf = mention.get("confidence", 0.5)
                                scores[int(doc_id)] = scores.get(int(doc_id), 0.0) + float(mention_conf)
            except Exception as e:
                logging.warning("Entity graph search failed in prefilter: %s", e)

        return scores

    def _truncate_candidates(self, scores: Dict[int, float]) -> list[int]:
        if len(scores) <= self._MAX_CANDIDATE_DOCS:
            return list(scores.keys())

        sorted_docs = sorted(
            scores.keys(), key=lambda d: scores.get(d, 0.0), reverse=True
        )
        return sorted_docs[:self._MAX_CANDIDATE_DOCS]

    # ------------------------------------------------------------------
    # L1: Super-Tree document selection
    # ------------------------------------------------------------------
    async def _select_tag_sets(self, query: str, candidate_db_ids: Dict[int, float]) -> Dict[int, float]:
        """阶段3 形态1：按领域标签把候选文档聚类，LLM 选相关集合，返回命中文档。

        复用 closet_tags 已有的领域级语义标签，把 L0 候选按标签聚成集合，
        让 LLM 先选相关集合，再只对命中集合做 L1 精排——减少难负样本干扰。
        LLM 失败时保留全部候选（不误伤）。
        """
        if not candidate_db_ids:
            return {}

        # 聚合同一标签下的候选文档
        tag_to_docs: Dict[str, list] = {}
        for db_id in candidate_db_ids:
            try:
                # 领域级语义标签只认 LLM 抽象标签（fallback 原词不进语义漏斗）
                tags = self.db.get_doc_tags(db_id, source="llm")
            except Exception:
                tags = []
            for t in tags:
                tag_to_docs.setdefault(t["tag_text"], []).append(db_id)

        if not tag_to_docs:
            return candidate_db_ids

        tag_list = "\n".join(f"- {tag}（{len(docs)}篇）" for tag, docs in tag_to_docs.items())
        prompt = f"""你是一个文档检索领域选择器。给定查询和候选领域标签，选出与查询最相关的 1-3 个领域。

[用户问题]
{query}

[候选领域标签]
{tag_list}

返回JSON格式：
{{"tags": ["领域标签1", "领域标签2"]}}
只返回与查询相关的领域，直接返回JSON，不要其他内容。"""

        response = await llm_acompletion(self.retrieve_model or self.model, prompt, thinking_disabled=False)
        if not response:
            return candidate_db_ids

        data = extract_json(response)
        if not isinstance(data, dict):
            return candidate_db_ids

        selected_tags = data.get("tags", [])
        if not isinstance(selected_tags, list) or not selected_tags:
            return candidate_db_ids

        keep = set()
        for tag in selected_tags:
            for db_id in tag_to_docs.get(str(tag), []):
                keep.add(db_id)
        if not keep:
            return candidate_db_ids
        return {db_id: candidate_db_ids[db_id] for db_id in keep}

    async def _score_candidates(self, query: str, candidate_db_ids: Dict[int, float]) -> list[str]:
        """Q1：对候选文档做显式相关性打分，返回降序列出的 top-k uuid。

        相较于旧逻辑让 LLM 直接吐 doc_id，这里让 LLM 对每篇给出
        相关性分数与理由，按分数取前 _SELECT_TOP_K，提升难负样本下的精确率。
        """
        if not candidate_db_ids:
            return []

        # 粗选：限到 _RANK_K 再打分，控制打分 prompt 体积。
        truncated = self._truncate_candidates(candidate_db_ids)[: self._RANK_K]

        super_tree = self._build_super_tree(truncated)
        kb_identity = self.kb_identity.get_identity()

        # 预算保护：超限时缩减候选，避免 L1 打分 prompt 过长。
        tree_json = json.dumps(super_tree, ensure_ascii=False)
        total_tokens = count_tokens(tree_json) + count_tokens(kb_identity) + count_tokens(query)
        while total_tokens > self._MAX_SUPER_TREE_TOKENS and len(super_tree["documents"]) > 5:
            super_tree["documents"].pop()
            tree_json = json.dumps(super_tree, ensure_ascii=False)
            total_tokens = count_tokens(tree_json) + count_tokens(kb_identity) + count_tokens(query)

        prompt = f"""你是一个文档检索相关性评分专家。给定用户问题、知识库概览和候选文档结构，请为每个候选文档给出 0.0-1.0 的相关性分数。

[知识库概览]
{kb_identity}

[用户问题]
{query}

[候选文档结构]
{json.dumps(super_tree, ensure_ascii=False)}

要求：
1. 分数越高代表文档越可能包含答案；只给 >=0.5 的候选保留，其余可在 ranked 中省略。
2. 用一句话说明理由。

返回JSON格式：
{{
  "ranked": [
    {{"doc_id": "uuid-1", "score": 0.9, "reason": "直接相关"}},
    ...
  ],
  "top_k": {self._SELECT_TOP_K}
}}
直接返回最终JSON结构，不要输出其他内容。"""

        response = await llm_acompletion(self.retrieve_model or self.model, prompt, thinking_disabled=False)
        if not response:
            return []

        data = extract_json(response)
        if not isinstance(data, dict):
            return []

        ranked = data.get("ranked", [])
        if not isinstance(ranked, list) or not ranked:
            return []

        scored = []
        for item in ranked:
            if not isinstance(item, dict):
                continue
            doc_id = item.get("doc_id")
            if not doc_id:
                continue
            try:
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            scored.append((doc_id, score))

        if not scored:
            return []

        scored.sort(key=lambda x: x[1], reverse=True)
        s_max = scored[0][1]
        # 自适应相对阈值：保留分数 >= 最高分*ratio 的候选。
        # 若最高分过低（整个查询弱），则保留 top-k 兜底，避免误杀。
        if s_max < 0.3:
            result = [doc_id for doc_id, _s in scored[: self._SELECT_TOP_K]]
        else:
            threshold = s_max * self._SCORE_RATIO
            result = [doc_id for doc_id, s in scored if s >= threshold]
        # top_k 固定为配置值，不被 LLM 返回值控制，保证契约"取前 _SELECT_TOP_K"
        return result[: self._SELECT_TOP_K]

    async def _holistic_select(self, query: str, db_ids: list[int], keep: int = None) -> list[int]:
        """推理式整体挑选：LLM 从 db_ids 中挑选最相关的文档（可变数量，宁缺毋滥）。

        与独立打分不同，这里让 LLM 横向比较后"挑选"，只返回真正可能相关的文档，
        从机制上避免硬负样本稀释精确率。返回 db_id 列表。
        """
        if not db_ids:
            return []
        if len(db_ids) == 1:
            return list(db_ids)

        keep = keep or self._SELECT_TOP_K
        super_tree = self._build_super_tree(db_ids)
        kb_identity = self.kb_identity.get_identity()

        # 预算保护：超限时缩减候选，避免挑选 prompt 过长。
        tree_json = json.dumps(super_tree, ensure_ascii=False)
        total_tokens = count_tokens(tree_json) + count_tokens(kb_identity) + count_tokens(query)
        while total_tokens > self._MAX_SUPER_TREE_TOKENS and len(super_tree["documents"]) > 2:
            super_tree["documents"].pop()
            tree_json = json.dumps(super_tree, ensure_ascii=False)
            total_tokens = count_tokens(tree_json) + count_tokens(kb_identity) + count_tokens(query)

        prompt = f"""你是一个文档检索专家。给定用户问题、知识库概览和候选文档结构，请挑选出最可能包含答案的文档（最多 {keep} 篇）。

[知识库概览]
{kb_identity}

[用户问题]
{query}

[候选文档结构]
{tree_json}

要求：
1. 只挑真正可能包含答案的文档；若没有足够相关的，可以少选甚至不选。
2. 宁缺毋滥：不要为凑数而挑选不相关的文档。
3. 基于文档的章节标题和摘要判断相关性。

返回JSON格式：
{{"doc_ids": ["uuid-1", "uuid-2"]}}
直接返回JSON，不要其他内容。"""

        response = await llm_acompletion(self.retrieve_model or self.model, prompt, thinking_disabled=False)
        if not response:
            return []
        data = extract_json(response)
        if not isinstance(data, dict):
            return []
        picked = data.get("doc_ids", [])
        if not isinstance(picked, list):
            return []

        uuid_to_db = {v: k for k, v in self._get_db_to_uuid().items()}
        db_id_set = set(db_ids)
        selected = []
        for did in picked:
            db_id = None
            if isinstance(did, int):
                db_id = did
            elif isinstance(did, str):
                db_id = uuid_to_db.get(did)
            if db_id is not None and db_id in db_id_set and db_id not in selected:
                selected.append(db_id)
        return selected[:keep]

    async def _select_documents_reasoning(self, query: str, candidate_db_ids: Dict[int, float]) -> list[str]:
        """三层重构-选择层：map-reduce 推理式选档。

        小候选集(<= _REASON_GROUP_SIZE)走单次整体挑选；大候选集先分组 map
        (每组挑选 top-_REASON_KEEP_PER_GROUP)，再 reduce 精选出最终 top-k。
        全程"推理挑选"而非"独立打分"，返回 uuid 列表。
        """
        truncated = self._truncate_candidates(candidate_db_ids)
        if not truncated:
            return []

        if len(truncated) <= self._REASON_GROUP_SIZE:
            selected_dbids = await self._holistic_select(query, truncated)
        else:
            groups = [
                truncated[i:i + self._REASON_GROUP_SIZE]
                for i in range(0, len(truncated), self._REASON_GROUP_SIZE)
            ]
            map_tasks = [
                self._holistic_select(query, g, keep=self._REASON_KEEP_PER_GROUP)
                for g in groups
            ]
            map_results = await asyncio.gather(*map_tasks, return_exceptions=True)
            winners = []
            for r in map_results:
                if isinstance(r, list):
                    for db_id in r:
                        if db_id not in winners:
                            winners.append(db_id)
            if not winners:
                return []
            if len(winners) <= self._SELECT_TOP_K:
                selected_dbids = winners
            else:
                selected_dbids = await self._holistic_select(query, winners)

        db_to_uuid = self._get_db_to_uuid()
        return [db_to_uuid[d] for d in selected_dbids if d in db_to_uuid]

    def _hierarchy_boost(self, query: str, candidate_db_ids: Dict[int, float]) -> Dict[int, float]:
        """三层重构-层级：集合标签软路由。

        以 closet_tags 为"集合"层（每个标签=一个集合，文档软归属多个集合）。
        对候选文档，若其集合标签与查询词有重叠，则按标签置信度加性提升分数。
        软路由：只加权、绝不删除候选（吸取 H03 查询期硬过滤误删的教训）。
        """
        if not candidate_db_ids or jieba is None:
            return candidate_db_ids

        query_tokens = {
            t.strip().lower() for t in jieba.lcut(query)
            if len(t.strip()) > 1 and t.strip().lower() not in _STOPWORDS
        }
        if not query_tokens:
            return candidate_db_ids

        boosted: Dict[int, float] = {}
        for db_id, score in candidate_db_ids.items():
            boost = 0.0
            try:
                # 集合层软路由只认 LLM 抽象标签（fallback 原词不进语义漏斗）
                tags = self.db.get_doc_tags(db_id, source="llm")
            except Exception:
                tags = []
            for t in tags:
                tag_text = t.get("tag_text", "")
                if not tag_text:
                    continue
                tag_tokens = {
                    x.strip().lower() for x in jieba.lcut(tag_text)
                    if len(x.strip()) > 1
                }
                if query_tokens & tag_tokens:
                    boost += float(t.get("confidence", 0.5))
            boosted[db_id] = float(score) + boost * self._HIERARCHY_BOOST_WEIGHT
        return boosted

    async def select_documents(self, query: str, candidate_db_ids: Dict[int, float]) -> list[str]:
        """量级自适应入口（[S5]）：小=直连 / 中=单层领域路由 / 海量=层级树导航。

        语料树未建、导航落空或导航候选挑选为空时，一律降级回扁平管线
        （标签软路由 + map-reduce 推理选档），行为与改造前一致。
        """
        if not candidate_db_ids:
            return []
        tier = self.detect_scale_tier()
        if tier == TIER_MASSIVE:
            nav_docs = await self.navigate_tree(query)
            if nav_docs:
                # 树导航候选（已按 doc_id 去重、留最大权重）→ 软路由加权 → 推理选档
                boosted = self._hierarchy_boost(query, nav_docs)
                selected = await self._select_documents_reasoning(query, boosted)
                if selected:
                    logging.info("[SuperTree] massive tier: tree navigation selected %d docs", len(selected))
                    return selected
                logging.info("[SuperTree] tree-nav selection empty; falling back to flat candidates")
            else:
                logging.info("[SuperTree] tree navigation yielded nothing (or tree absent); flat fallback")
        elif tier == TIER_MEDIUM:
            candidate_db_ids = self._cluster_route_boost(query, candidate_db_ids)
        # 小语料直连 / 中语料域内匹配 / 海量兜底：扁平管线
        boosted = self._hierarchy_boost(query, candidate_db_ids)
        return await self._select_documents_reasoning(query, boosted)

    # ------------------------------------------------------------------
    # P2 量级自适应引擎（[S5]）
    # ------------------------------------------------------------------
    def detect_scale_tier(self) -> str:
        """探测语料规模（文档数）→ 导航档位。阈值可配置，实测标定。"""
        try:
            n_docs = int(self.db.get_document_count())
        except Exception:
            n_docs = 0
        if n_docs < self._SMALL_MAX_DOCS:
            return TIER_SMALL
        if n_docs < self._MASSIVE_MIN_DOCS:
            return TIER_MEDIUM
        return TIER_MASSIVE

    def _load_corpus_tree(self) -> Optional[_CorpusTreeView]:
        """加载语料树视图；未建树返回 None（调用方降级扁平管线）。"""
        try:
            nodes = self.db.get_corpus_tree_nodes()
            memberships = self.db.get_all_corpus_memberships()
        except Exception:
            return None
        if not nodes:
            return None
        return _CorpusTreeView(nodes, memberships)

    # ------------------------------------------------------------------
    # P2 语义树导航（[S6]）：每层 = 预筛 → 加权 → 精挑
    # ------------------------------------------------------------------
    def _score_nodes(self, query: str, nodes: List[dict],
                     view: _CorpusTreeView = None,
                     entity_table: Dict[int, Dict] = None) -> Dict[int, Dict]:
        """四通道节点打分：标签/BM25/向量/实体图谱。

        Returns:
            {node_id: {"tag_score": float, "bm25_score": float,
                       "vector_score": float, "entity_boost": float,
                       "total_score": float, "entity_info": dict}}
        """
        query_tokens = set(self.keyword_index._tokenize(query or ""))
        q_lower = (query or "").lower()

        scores: Dict[int, Dict] = {}
        for node in nodes:
            node_scores = {
                "tag_score": 0.0,
                "bm25_score": 0.0,
                "vector_score": 0.0,
                "entity_boost": 0.0,
                "entity_info": None,
            }

            # Channel A: closet_tags 语义标签匹配
            tag = (node.get("tag") or "").strip().lower()
            if tag:
                if tag in q_lower or q_lower in tag:
                    node_scores["tag_score"] = 2.0
                elif query_tokens & set(self.keyword_index._tokenize(tag)):
                    node_scores["tag_score"] = 1.5

            # Channel B: BM25 关键词匹配（jieba 分词）
            if query_tokens:
                text_tokens = set(self.keyword_index._tokenize(
                    f"{node.get('title', '')} {node.get('summary', '')}"))
                overlap = query_tokens & text_tokens
                if overlap:
                    # BM25-like scoring: overlap / query_length
                    node_scores["bm25_score"] = len(overlap) / len(query_tokens)

            # Channel C: 向量相似度（如果可用）
            if hasattr(self, '_vector_score') and self._vector_score:
                try:
                    node_scores["vector_score"] = self._vector_score(query, node)
                except Exception:
                    pass

            # Channel D: 实体图谱加权（查表）
            if entity_table and view:
                boost, entity_info = self._get_node_entity_boost(
                    node["id"], view, entity_table
                )
                node_scores["entity_boost"] = boost
                node_scores["entity_info"] = entity_info

            # 归一化总分：各通道权重
            node_scores["total_score"] = (
                node_scores["tag_score"] * 0.3 +
                node_scores["bm25_score"] * 0.3 +
                node_scores["vector_score"] * 0.2 +
                node_scores["entity_boost"] * 0.2
            )

            scores[node["id"]] = node_scores

        return scores

    def _prefilter_nodes(self, query: str, nodes: List[dict],
                         view: _CorpusTreeView = None,
                         entity_table: Dict[int, Dict] = None) -> List[dict]:
        """[S6]① 宽层语义预筛：四通道打分，筛到 top-k。

        保召回：无任何匹配信号时原样全量返回（绝不硬过滤，由 LLM 精挑裁决）。
        """
        if not nodes:
            return []
        scores = self._score_nodes(query, nodes, view, entity_table)
        # 零信号保召回：四个通道全部无命中时不截断，全量交给 LLM 精挑
        if not any(s["total_score"] > 0 for s in scores.values()):
            return list(nodes)
        # Sort by total_score descending
        ranked = sorted(nodes,
                       key=lambda n: scores.get(n["id"], {}).get("total_score", 0.0),
                       reverse=True)
        return ranked[: self._NODE_PREFILTER_TOPK]

    # ------------------------------------------------------------------
    # Entity distance precomputation (L0, one BFS, level-by-level lookup)
    # ------------------------------------------------------------------
    # Relation type weights (causal > part_of > related_to > other)
    _RELATION_TYPE_WEIGHTS = {
        "causal": 1.0, "causes": 1.0, "effect": 1.0,
        "part_of": 0.8, "contains": 0.8, "has_part": 0.8, "belongs_to": 0.8,
        "related_to": 0.6, "associated": 0.6, "similar": 0.6,
        "_default": 0.4,
    }
    # Distance decay
    _DISTANCE_DECAY = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.2}
    _MAX_ENTITY_HOP = 3

    def _precompute_entity_distances(self, query: str) -> Dict[int, Dict]:
        """L0 预计算实体距离（一次 BFS，逐层查表）。

        Returns:
            entity_table: {entity_id: {distance, relation_type, weight, name}}
        """
        try:
            query_entities = self.db.search_entities(query, limit=self._L0_CHANNEL_TOPK)
        except Exception:
            return {}

        entity_table = {}
        queue = []

        # Query entities: distance 0, weight 1.0
        for entity in query_entities:
            eid = entity.get("id")
            if not eid:
                continue
            entity_table[eid] = {
                "distance": 0,
                "relation_type": "direct",
                "weight": 1.0,
                "name": entity.get("name", "")
            }
            queue.append((eid, 0))

        visited = set(entity_table.keys())

        # BFS to compute distances
        while queue:
            current_id, current_dist = queue.pop(0)
            if current_dist >= self._MAX_ENTITY_HOP:
                continue

            try:
                relations = self.db.get_entity_relations(current_id)
            except Exception:
                continue

            for rel in relations:
                neighbor_id = (rel["object_id"] if rel["subject_id"] == current_id
                              else rel["subject_id"])

                if neighbor_id in visited:
                    continue

                # Compute weight: distance decay × relation type weight
                distance_weight = self._DISTANCE_DECAY.get(current_dist + 1, 0.1)
                relation_type = rel.get("predicate", "_default")
                relation_weight = self._RELATION_TYPE_WEIGHTS.get(
                    relation_type, self._RELATION_TYPE_WEIGHTS["_default"]
                )
                total_weight = distance_weight * relation_weight

                try:
                    neighbor_entity = self.db.get_entity_by_id(neighbor_id)
                    name = neighbor_entity.get("name", "") if neighbor_entity else ""
                except Exception:
                    name = ""

                entity_table[neighbor_id] = {
                    "distance": current_dist + 1,
                    "relation_type": relation_type,
                    "weight": total_weight,
                    "name": name
                }

                visited.add(neighbor_id)
                queue.append((neighbor_id, current_dist + 1))

        return entity_table

    def _get_node_entity_boost(self, node_id: int, view: _CorpusTreeView,
                                entity_table: Dict[int, Dict]) -> Tuple[float, Optional[Dict]]:
        """逐层图谱加权：查表，不重新 BFS。

        Returns:
            (boost_weight, entity_info) or (0, None)
        """
        if not entity_table:
            return 0, None

        # Get entities in this node's subtree
        doc_ids = view.subtree_doc_ids(node_id)
        if not doc_ids:
            return 0, None

        best_boost = 0
        best_entity = None

        # Check each document's entities against the entity table
        for doc_id in doc_ids:
            try:
                mentions = self.db.get_entity_mentions_by_doc(doc_id)
            except Exception:
                continue
            for mention in mentions:
                entity_id = mention.get("entity_id")
                if entity_id and entity_id in entity_table:
                    info = entity_table[entity_id]
                    if info["weight"] > best_boost:
                        best_boost = info["weight"]
                        best_entity = info

        return best_boost, best_entity

    async def _select_nodes(self, query: str, nodes: List[dict],
                            node_scores: Dict[int, Dict] = None) -> List[int]:
        """[S6]③ LLM 推理精挑：读节点摘要 + 四通道分数 + 图谱加权证据，可变数量挑选、宁缺毋滥。

        T9 推理挑选（H02 修复）泛化到树分支；返回选中的节点 id 列表。
        """
        if not nodes:
            return []
        if len(nodes) == 1:
            return [nodes[0]["id"]]

        # Build node info with evidence
        node_list = []
        for n in nodes:
            info = {
                "node_id": n["id"],
                "title": n.get("title", ""),
                "summary": (n.get("summary") or "")[: self._SUMMARY_MAX_LEN]
            }

            # Add channel scores if available
            if node_scores and n["id"] in node_scores:
                scores = node_scores[n["id"]]
                if scores.get("tag_score", 0) > 0:
                    info["tag_score"] = round(scores["tag_score"], 2)
                if scores.get("bm25_score", 0) > 0:
                    info["bm25_score"] = round(scores["bm25_score"], 2)
                if scores.get("vector_score", 0) > 0:
                    info["vector_score"] = round(scores["vector_score"], 2)

                # Add entity boost evidence if available
                if scores.get("entity_boost", 0) > 0:
                    info["entity_boost"] = round(scores["entity_boost"], 2)
                    if scores.get("entity_info"):
                        entity = scores["entity_info"]
                        info["entity_relation"] = {
                            "name": entity.get("name", ""),
                            "type": entity.get("relation_type", ""),
                            "distance": entity.get("distance", -1)
                        }

            node_list.append(info)

        prompt = f"""你是一个知识库树导航精挑专家（语义树导航·逐层精挑）。给定用户问题与当前层的候选分支节点（含标题、摘要和四通道分数），请挑选出可能包含答案的分支（可变数量）。

[用户问题]
{query}

[候选分支节点]
{json.dumps(node_list, ensure_ascii=False)}

要求：
1. 只挑真正可能包含答案的分支；若没有足够相关的，可以少选甚至不选。
2. 宁缺毋滥：不要为凑数而挑选不相关的分支。
3. 基于分支标题与摘要判断相关性。
4. 综合考虑四通道分数：
   - tag_score: 标签匹配（语义相关性）
   - bm25_score: 关键词匹配（词面匹配）
   - vector_score: 向量相似度（语义相似）
   - entity_relation: 实体关系（图谱加权）
     - distance=0: 直接包含查询实体
     - distance=1: 包含与查询实体有直接关系的实体
     - relation_type: causal(因果) > part_of(组成) > related_to(相关)

返回JSON格式：{{"node_ids": [1, 2]}}
直接返回JSON，不要其他内容。"""
        response = await llm_acompletion(self.retrieve_model or self.model, prompt, thinking_disabled=False)
        if not response:
            return []
        data = extract_json(response)
        if not isinstance(data, dict):
            return []
        picked = data.get("node_ids", [])
        if not isinstance(picked, list):
            return []
        valid = {n["id"] for n in nodes}
        selected = []
        for nid in picked:
            try:
                nid = int(nid)
            except (TypeError, ValueError):
                continue
            if nid in valid and nid not in selected:
                selected.append(nid)
        return selected

    async def navigate_tree(self, query: str) -> Optional[Dict[int, float]]:
        """[S6] 层级树导航：逐层预筛→加权→精挑，渐进披露，分支并行展开。

        返回 doc_id→weight（软归属按 doc_id 去重、保留最大权重）；
        语料树未建返回 None（调用方降级扁平管线）；树存在但无命中返回 {}。
        """
        view = self._load_corpus_tree()
        if view is None or view.root_id is None:
            return None
        top_nodes = view.children_of(view.root_id)
        if not top_nodes:
            return {}

        # L0: Precompute entity distances (one BFS, level-by-level lookup)
        entity_table = self._precompute_entity_distances(query)

        return await self._navigate_level(query, top_nodes, view, entity_table)

    async def _navigate_level(self, query: str, siblings: List[dict],
                              view: _CorpusTreeView, entity_table: Dict[int, Dict]) -> Dict[int, float]:
        """单层导航：四通道初筛→图谱加权(查表)→LLM精挑→只展开选中分支。"""
        if not siblings:
            return {}

        # ① 四通道初筛（缩小当前层候选节点范围）
        if len(siblings) > self._NARROW_LAYER_MAX:
            candidates = self._prefilter_nodes(query, siblings, view, entity_table)
        else:
            candidates = list(siblings)

        # ② 图谱加权（查表，不重新 BFS）- 已集成到 _score_nodes
        # 获取节点分数信息用于 LLM
        node_scores = self._score_nodes(query, candidates, view, entity_table)

        # ③ LLM精挑（看到完整证据做决策）
        picked_ids = set(await self._select_nodes(query, candidates, node_scores))
        picked = [n for n in candidates if n["id"] in picked_ids]

        docs: Dict[int, float] = {}
        subtasks = []
        for node in picked:
            for doc_id, weight in view.docs_of(node["id"]):
                docs[doc_id] = max(docs.get(doc_id, 0.0), float(weight))
            children = view.children_of(node["id"])
            if children:
                subtasks.append(self._navigate_level(query, children, view, entity_table))
        if subtasks:
            # 延迟纪律（[S6]）：一层选中多个分支后，下一层导航并行展开
            results = await asyncio.gather(*subtasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logging.warning("Tree navigation branch failed: %s", r)
                    continue
                for doc_id, weight in r.items():
                    docs[doc_id] = max(docs.get(doc_id, 0.0), weight)
        return docs

    def _cluster_route_boost(self, query: str, candidate_db_ids: Dict[int, float]) -> Dict[int, float]:
        """[S5] 中语料单层领域路由：对根孩子节点（领域）无向量打分，
        命中领域子树内的候选加性提升。软路由：只加权、绝不删除候选。"""
        if not candidate_db_ids:
            return candidate_db_ids
        view = self._load_corpus_tree()
        if view is None or view.root_id is None:
            return candidate_db_ids
        top_nodes = view.children_of(view.root_id)
        if not top_nodes:
            return candidate_db_ids
        scores = self._score_nodes(query, top_nodes)
        if not scores:
            return candidate_db_ids
        boosted = dict(candidate_db_ids)
        for node in top_nodes:
            s = scores.get(node["id"], {}).get("total_score", 0.0)
            if s <= 0:
                continue
            for doc_id in view.subtree_doc_ids(node["id"]):
                if doc_id in boosted:
                    boosted[doc_id] = float(boosted[doc_id]) + s * self._HIERARCHY_BOOST_WEIGHT
        return boosted

    # ------------------------------------------------------------------
    # Super-Tree builder
    # ------------------------------------------------------------------
    def _build_super_tree(self, doc_ids: list[int]) -> Dict:
        documents = []
        db_to_uuid = self._get_db_to_uuid()

        # Batch-fetch all documents once (avoids N+1 query per doc)
        all_docs = self.db.get_all_documents()
        docs_by_id = {d["id"]: d for d in all_docs}

        for db_doc_id in doc_ids:
            doc = docs_by_id.get(db_doc_id)
            if not doc:
                continue

            uuid = db_to_uuid.get(db_doc_id, str(db_doc_id))
            top_nodes = self.db.get_top_level_nodes(db_doc_id)

            # Batch-fetch child counts for all top nodes in one query
            top_node_ids = [n.get("node_id") for n in top_nodes if n.get("node_id")]
            child_counts = {}
            if top_node_ids:
                with self.db._connect() as conn:
                    placeholders = ",".join("?" * len(top_node_ids))
                    rows = conn.execute(
                        f"SELECT parent_node_id, COUNT(*) FROM nodes "
                        f"WHERE parent_node_id IN ({placeholders}) GROUP BY parent_node_id",
                        top_node_ids,
                    ).fetchall()
                    child_counts = {row[0]: row[1] for row in rows}

            nodes_with_children = [
                (node, child_counts.get(node.get("node_id"), 0))
                for node in top_nodes
            ]

            nodes_with_children.sort(key=lambda x: x[1], reverse=True)
            selected = nodes_with_children[:self._MAX_TOP_NODES_PER_DOC]

            top_nodes_out = []
            for node, _ in selected:
                summary = node.get("summary", "") or ""
                if len(summary) > self._SUMMARY_MAX_LEN:
                    summary = summary[:self._SUMMARY_MAX_LEN] + "..."
                node_entry = {
                    "title": node.get("title", ""),
                    "summary": summary,
                }
                # Enrich with depth=2 children titles for finer granularity
                child_titles = []
                for child in node.get("nodes", [])[:5]:
                    child_title = child.get("title", "")
                    if child_title:
                        child_titles.append(child_title)
                if child_titles:
                    node_entry["children"] = child_titles
                top_nodes_out.append(node_entry)

            documents.append({
                "doc_id": uuid,
                "db_id": db_doc_id,
                "doc_name": doc.get("pdf_name", ""),
                "description": doc.get("doc_description", ""),
                "top_nodes": top_nodes_out,
            })

        return {"documents": documents}

    # ------------------------------------------------------------------
    # Backfill existing documents on first init
    # ------------------------------------------------------------------
    def _backfill_existing_docs(self) -> None:
        try:
            with self.db._connect() as conn:
                row = conn.execute("SELECT COUNT(*) FROM doc_keywords").fetchone()
                if row and row[0] > 0:
                    return
        except Exception:
            return

        for doc in self.db.get_all_documents():
            try:
                node_titles = []
                try:
                    top_nodes = self.db.get_top_level_nodes(doc["id"])
                    node_titles = [n.get("title", "") for n in top_nodes if n.get("title")]
                except Exception:
                    pass
                self.keyword_index.add_document(
                    doc["id"],
                    doc.get("pdf_name", ""),
                    doc.get("doc_description", ""),
                    node_titles,
                )
            except Exception as e:
                logging.warning("Backfill failed for doc %s: %s", doc.get("pdf_name"), e)
