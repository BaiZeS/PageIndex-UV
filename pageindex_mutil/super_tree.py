import logging
from collections import Counter
from typing import Dict, List, Tuple

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
        """Tokenize text with jieba (self is unused — safe to call as KeywordIndex._tokenize(None, text))."""
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
        for token, count in Counter(self._tokenize(doc_name)).items():
            records.append((doc_id, token, "name", count))
        for token, count in Counter(self._tokenize(doc_description or "")).items():
            records.append((doc_id, token, "description", count))
        if node_titles:
            for title in node_titles:
                for token, count in Counter(self._tokenize(title)).items():
                    records.append((doc_id, token, "node_title", count))
        # Index full document body with term frequencies for BM25 scoring.
        if content:
            for token, count in Counter(self._tokenize(content)).items():
                records.append((doc_id, token, "content", count))
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

from .utils import llm_acompletion, count_tokens, extract_json


# ---------------------------------------------------------------------------
# Lazy import of reasoning helpers（T32.2：自 AgenticRouter._load_main_funcs
# 上提为模块级，同样的惰性导入 + 缓存 + ImportError 容错模式）。
# 与 router 版的一处必要差异：缓存是模块全局，import 失败不写缓存（返回空
# dict、下次调用重试）——测试可用 sys.modules 临时替换 reasoning，粘住的
# 空缓存会跨测试泄漏；成功导入才缓存。
# 红线：本模块顶层永不 import agentic.enhance / agentic.evidence
# （enhance.py 模块级 `from ..super_tree import KeywordIndex`，直引即成环）。
# ---------------------------------------------------------------------------
_REASON_FUNCS = None


def _load_reason_funcs():
    global _REASON_FUNCS
    if _REASON_FUNCS is None:
        try:
            from .reasoning import (
                build_context_with_budget,
                generate_answer,
                spans_from_nodes,
                build_context_for_doc,
            )
        except ImportError:
            return {}
        _REASON_FUNCS = {
            "build_context_with_budget": build_context_with_budget,
            "generate_answer": generate_answer,
            "spans_from_nodes": spans_from_nodes,
            "build_context_for_doc": build_context_for_doc,
        }
    return _REASON_FUNCS


class SuperTreeIndex:
    """L0 evidence bundle（见 agentic/evidence.py）+ L1 Super-Tree document selection."""

    # Defaults (overridden by config.yaml via _init_from_config)
    _MAX_CANDIDATE_DOCS = 50
    _MAX_SUPER_TREE_TOKENS = 6000
    _L1_SELECT_KEEP = 10  # L1 终选 keep 上限（对齐评测 R@10 口径，[S6]#6）
    # 三层重构-选择层：map-reduce 推理选择参数
    _REASON_GROUP_SIZE = 10   # 候选 <= 该值走单次整体挑选；否则分组 map-reduce
    _REASON_KEEP_PER_GROUP = 3  # map 阶段每组保留的篇数
    # 三层重构-L0：各通道召回 top-k（并集召回，宁多勿漏，精排交给选择层）
    _L0_CHANNEL_TOPK = 30
    # L1 预算守卫（[S6]#4）：先截短 doc_summary（最短 _DOC_SUMMARY_MIN_LEN），
    # 仍超才退化弱候选为一行"名称 + 证据摘要"（证据摘要截断上限）。
    _DOC_SUMMARY_MIN_LEN = 50
    _DOC_EVIDENCE_SUMMARY_MAX_LEN = 40

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
            self._MAX_CANDIDATE_DOCS = getattr(cfg, "max_candidate_docs", self._MAX_CANDIDATE_DOCS)
            self._MAX_SUPER_TREE_TOKENS = getattr(cfg, "max_super_tree_tokens", self._MAX_SUPER_TREE_TOKENS)
            self._L1_SELECT_KEEP = getattr(cfg, "l1_select_keep", self._L1_SELECT_KEEP)
            self._REASON_GROUP_SIZE = getattr(cfg, "reason_group_size", self._REASON_GROUP_SIZE)
            self._REASON_KEEP_PER_GROUP = getattr(cfg, "reason_keep_per_group", self._REASON_KEEP_PER_GROUP)
            self._L0_CHANNEL_TOPK = getattr(cfg, "l0_channel_topk", self._L0_CHANNEL_TOPK)
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
    def on_document_added(self, db_doc_id: int, content: str = None) -> None:
        doc = self.db.get_document_by_id(db_doc_id)
        if doc:
            # Collect node titles from DB for keyword indexing
            node_titles = []
            try:
                top_nodes = self.db.get_top_level_nodes(db_doc_id)
                node_titles = [n.get("title", "") for n in top_nodes if n.get("title")]
            except Exception:
                pass
            # Collect full document body for content-level keyword indexing.
            # Prefer caller-provided content (in-memory structure text): MD docs
            # indexed via client.index_batch never populate the pages table, so
            # get_document_content() returns empty for them (body-blind BM25).
            body = content if content else ""
            if not body:
                try:
                    body = self.db.get_document_content(db_doc_id)
                except Exception:
                    body = ""
            self.keyword_index.add_document(
                db_doc_id, doc.get("pdf_name", ""),
                doc.get("doc_description", ""), node_titles, content=body
            )
        self.kb_identity.invalidate()

    def on_document_removed(self, db_doc_id: int) -> None:
        self.keyword_index.remove_document(db_doc_id)
        self.kb_identity.invalidate()

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
    def _build_doc_entries(self, query: str, db_ids: List[int],
                           evidence_bundle: dict = None,
                           all_docs: List[dict] = None) -> List[dict]:
        """L1 候选呈现单元：文档名 + doc_summary(or doc_description) + 证据行。

        证据行只来自证据束（render_doc_evidence 渲染，evidence_bundle 直通；
        缺失时为空，不再有 _doc_evidence_lines 旧路 node_profiles 重算）。
        doc_summary 列由 T12 落库——读 doc.get("doc_summary") 兜底 doc_description。

        all_docs 由调用方一次取好传入（map-reduce 各组共用，避免每组重查
        get_all_documents() SELECT *）；默认 None 时内部获取一次（单次挑选
        或直接调用路径）。
        """
        if all_docs is None:
            try:
                all_docs = self.db.get_all_documents()
            except Exception:
                all_docs = []
        docs_by_id = {d["id"]: d for d in all_docs if isinstance(d, dict) and "id" in d}
        db_to_uuid = self._get_db_to_uuid()

        from .agentic.evidence import render_doc_evidence, derive_evidence_score
        bundle = evidence_bundle or {}

        def _evidence(db_id):
            line = render_doc_evidence(bundle, [db_id])
            prefix = f"doc {db_id}: "
            if line.startswith(prefix):
                line = line[len(prefix):]
            return "" if line == "无通道命中" else line

        def _score(db_id):
            return derive_evidence_score(bundle.get(int(db_id)))

        entries = []
        for db_id in db_ids:
            doc = docs_by_id.get(db_id) or {}
            entries.append({
                "db_id": db_id,
                "uuid": db_to_uuid.get(db_id, str(db_id)),
                "name": doc.get("pdf_name") or "",
                "summary": doc.get("doc_summary") or doc.get("doc_description") or "",
                "evidence": _evidence(db_id),
                "evidence_score": _score(db_id),
                "degraded": False,
            })
        return entries

    def _render_doc_block(self, entries: List[dict]) -> str:
        """候选文档块 JSON：{doc_id, name, summary, evidence}（退化条目无 summary）。"""
        documents = []
        for e in entries:
            item = {"doc_id": e["uuid"], "name": e["name"]}
            if not e["degraded"]:
                item["summary"] = e["summary"]
            if e["evidence"]:
                item["evidence"] = e["evidence"]
            documents.append(item)
        return json.dumps({"documents": documents}, ensure_ascii=False)

    def _fit_budget(self, entries: List[dict], kb_identity: str, query: str) -> List[dict]:
        """预算守卫（[S6]#4，不 pop 文档）：先逐条截短 doc_summary（最短
        _DOC_SUMMARY_MIN_LEN 字符），仍超才退化弱候选（证据分最低者优先）为
        一行"名称 + 证据摘要"。返回调整后的 entries（原地改）。"""
        def _tokens():
            return (count_tokens(self._render_doc_block(entries))
                    + count_tokens(kb_identity) + count_tokens(query))

        if _tokens() <= self._MAX_SUPER_TREE_TOKENS:
            return entries

        # 阶段 1：逐条截短摘要（保序轮转最长者，最短 50 字符）
        for _ in range(len(entries) * 4 + 1):
            if _tokens() <= self._MAX_SUPER_TREE_TOKENS:
                return entries
            truncatable = [
                e for e in entries
                if not e["degraded"] and len(e["summary"]) > self._DOC_SUMMARY_MIN_LEN
            ]
            if not truncatable:
                break
            e = max(truncatable, key=lambda x: len(x["summary"]))
            e["summary"] = e["summary"][:max(self._DOC_SUMMARY_MIN_LEN, len(e["summary"]) // 2)]

        if _tokens() <= self._MAX_SUPER_TREE_TOKENS:
            return entries

        # 阶段 2：退化弱候选（证据分最低、db_id 小者优先）为一行"名称 + 证据摘要"
        degradable = sorted(
            [e for e in entries if not e["degraded"]],
            key=lambda x: (x["evidence_score"], x["db_id"]),
        )
        for e in degradable:
            if _tokens() <= self._MAX_SUPER_TREE_TOKENS:
                break
            e["degraded"] = True
            e["summary"] = ""
            if e["evidence"]:
                e["evidence"] = e["evidence"][:self._DOC_EVIDENCE_SUMMARY_MAX_LEN]
        return entries

    def _reason_key_to_uuid(self, k, db_to_uuid: Dict[int, str],
                            uuid_to_db: Dict[str, int]):
        """理由键规范化（[S6]#7/[S7]）：uuid 键原样；int db_id 键 / 数字字符串
        db_id 键 → uuid；无法识别者原样返回（随后由 selected 过滤丢弃）。"""
        if isinstance(k, int):
            return db_to_uuid.get(k)
        if isinstance(k, str):
            if k in uuid_to_db:
                return k
            try:
                db_id = int(k)
            except (TypeError, ValueError):
                db_id = None
            if db_id is not None and db_id in db_to_uuid:
                return db_to_uuid[db_id]
            return k
        return None

    def _normalize_reasons(self, raw_reasons, selected: List[int]) -> Dict[str, str]:
        """规整 LLM 选中理由：键统一为 uuid（db_id 键 → uuid；uuid 键原样），
        只保留 selected（已截 keep）内的条目。缺失/非 dict/空值条目降级跳过。"""
        if not isinstance(raw_reasons, dict):
            return {}
        db_to_uuid = self._get_db_to_uuid()
        uuid_to_db = {v: k for k, v in db_to_uuid.items()}
        selected_uuids = {db_to_uuid.get(db_id, str(db_id)) for db_id in selected}
        reasons: Dict[str, str] = {}
        for k, v in raw_reasons.items():
            if not v:
                continue
            key = self._reason_key_to_uuid(k, db_to_uuid, uuid_to_db)
            if key is None or key not in selected_uuids:
                continue
            reasons[key] = str(v)
        return reasons

    async def _holistic_select(self, query: str, db_ids: list[int],
                               keep: int = None, evidence_bundle: dict = None,
                               all_docs: List[dict] = None):
        """推理式整体挑选：LLM 从 db_ids 中挑选最相关的文档（可变数量，宁缺毋滥）。

        与独立打分不同，这里让 LLM 横向比较后"挑选"，只返回真正可能相关的文档，
        从机制上避免硬负样本稀释精确率。返回 (db_id 列表, reasons dict)——
        reasons 键已规范化为 uuid、只含 selected[:keep] 内条目（[S6]#7/[S7]
        L1→L2 trace 下传）；候选呈现单元 = 文档名 + doc_summary + 证据行（[S6]#2）。
        """
        if not db_ids:
            return [], {}
        if len(db_ids) == 1:
            return list(db_ids), {}

        keep = keep or self._L1_SELECT_KEEP
        kb_identity = self.kb_identity.get_identity()

        entries = self._build_doc_entries(query, db_ids, evidence_bundle, all_docs=all_docs)
        entries = self._fit_budget(entries, kb_identity, query)
        doc_block = self._render_doc_block(entries)

        has_evidence = any(e["evidence"] for e in entries)
        if has_evidence:
            evidence_note = (
                "\n证据是语料事实，请优先依据证据与问题的语义关联程度判断，"
                "而非简单计数命中个数。\n"
            )
            evidence_requirement = "3. 证据命中是强信号，但选档仍须宁缺毋滥，不因命中而硬选不相关文档。\n"
        else:
            evidence_note = ""
            evidence_requirement = ""
        idx = 4 if evidence_requirement else 3

        prompt = f"""你是一个文档检索专家。给定用户问题、知识库概览和候选文档，请挑选出最可能包含答案的文档（最多 {keep} 篇）。

[知识库概览]
{kb_identity}

[用户问题]
{query}

[候选文档结构]
{doc_block}
{evidence_note}
要求：
1. 只挑真正可能包含答案的文档。
2. 宁缺毋滥：不要为凑数而挑选不相关的文档。
{evidence_requirement}{idx}. 为每个选中的文档给出一句话选中理由，填入 reasons 字段（doc_id → 理由）；理由基于该文档的证据/标题/摘要。
返回JSON格式：
{{"doc_ids": ["uuid-1", "uuid-2"], "reasons": {{"uuid-1": "一句话选中理由"}}}}
直接返回JSON，不要其他内容。"""

        response = await llm_acompletion(self.retrieve_model or self.model, prompt, thinking_disabled=False)
        if not response:
            return [], {}
        data = extract_json(response)
        if not isinstance(data, dict):
            return [], {}
        picked = data.get("doc_ids", [])
        if not isinstance(picked, list):
            return [], {}

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

        selected = selected[:keep]
        reasons = self._normalize_reasons(data.get("reasons"), selected)
        return selected, reasons

    async def _select_documents_reasoning(self, query: str, candidate_db_ids: Dict[int, float],
                                          evidence_bundle: dict = None):
        """三层重构-选择层：map-reduce 推理式选档。

        小候选集(<= _REASON_GROUP_SIZE)走单次整体挑选；大候选集先分组 map
        (每组挑选 top-_REASON_KEEP_PER_GROUP)，再 reduce 精选出最终 top-k。
        全程"推理挑选"而非"独立打分"。返回 (uuid 列表, reasons_by_uuid)——
        reduce 阶段阈值与终选 keep 同源（均取 _L1_SELECT_KEEP，[S6]#5/#6）。
        证据束直通 _holistic_select（生产链路生效，[S6]#2）。
        """
        truncated = self._truncate_candidates(candidate_db_ids)
        if not truncated:
            return [], {}

        # 文档列表一次取好（map-reduce 各组共用，避免每组重查 get_all_documents()
        # SELECT *；单次 select_documents 内只查一次）。
        try:
            all_docs = self.db.get_all_documents()
        except Exception:
            all_docs = []

        if len(truncated) <= self._REASON_GROUP_SIZE:
            selected_dbids, reasons = await self._holistic_select(
                query, truncated, evidence_bundle=evidence_bundle, all_docs=all_docs)
        else:
            groups = [
                truncated[i:i + self._REASON_GROUP_SIZE]
                for i in range(0, len(truncated), self._REASON_GROUP_SIZE)
            ]
            map_tasks = [
                self._holistic_select(query, g, keep=self._REASON_KEEP_PER_GROUP,
                                      evidence_bundle=evidence_bundle,
                                      all_docs=all_docs)
                for g in groups
            ]
            map_results = await asyncio.gather(*map_tasks, return_exceptions=True)
            winners = []
            merged_reasons: Dict[str, str] = {}
            for r in map_results:
                selected, group_reasons = [], {}
                if isinstance(r, tuple):
                    selected, group_reasons = r
                elif isinstance(r, list):
                    selected = r
                if isinstance(selected, list):
                    for db_id in selected:
                        if db_id not in winners:
                            winners.append(db_id)
                if isinstance(group_reasons, dict):
                    merged_reasons.update(group_reasons)
            if not winners:
                return [], {}
            if len(winners) <= self._L1_SELECT_KEEP:
                selected_dbids = winners
                # 中间档位（2–3 组 ⇒ winners ≤ keep）无 reduce：聚合各组 reasons
                # （仅 winners 内条目，键规范化复用 _normalize_reasons），
                # 保住 L1→L2 trace（[S6]#7）。
                reasons = self._normalize_reasons(merged_reasons, winners)
            else:
                selected_dbids, reasons = await self._holistic_select(
                    query, winners, evidence_bundle=evidence_bundle,
                    all_docs=all_docs)

        db_to_uuid = self._get_db_to_uuid()
        selected_uuids = [db_to_uuid[d] for d in selected_dbids if d in db_to_uuid]
        return selected_uuids, reasons

    async def select_documents(self, query: str, candidate_db_ids: Dict[int, float],
                               evidence_bundle: dict = None):
        """L1 选档入口（T13 收拢）：删 tier 树导航与软路由，直接推理选档。

        返回 (selected_uuids, reasons_by_uuid)；evidence_bundle 直通推理选档（[S6]#2）。
        """
        if not candidate_db_ids:
            return [], {}
        return await self._select_documents_reasoning(
            query, candidate_db_ids, evidence_bundle=evidence_bundle)

    # ------------------------------------------------------------------
    # 图谱距离衰减 / 关系类型权重常量（与 db.get_entity_distances_cte 同源，
    # test_entity_distances_cte.py 引用这些常量做逐项对照）
    # ------------------------------------------------------------------
    _RELATION_TYPE_WEIGHTS = {
        "causal": 1.0, "causes": 1.0, "effect": 1.0,
        "part_of": 0.8, "contains": 0.8, "has_part": 0.8, "belongs_to": 0.8,
        "related_to": 0.6, "associated": 0.6, "similar": 0.6,
        "_default": 0.4,
    }
    # Distance decay
    _DISTANCE_DECAY = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.2}
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

    # ------------------------------------------------------------------
    # Act — tree search + context assembly (parallelized)
    # ------------------------------------------------------------------
    async def recall_nodes_for_doc(self, query: str, doc_id: str,
                                      matched_info: List[Dict] = None,
                                      l1_reasons: Dict[str, str] = None,
                                      query_entities=None, query_tokens=None,
                                      node_entities=None):
        """Recall relevant nodes for a single document (runs in thread).

        [3.2.1] unit = 节点：enhance_and_select 统一接入——四通道 union 宽召回 +
        证据接地 + LLM 精挑（唯一裁剪者）。node_profiles 签名（DB 优先）与
        查询实体作为证据注入；内容策略命中词并入关键词证据（只喂证据，不替代
        精挑）。候选携带节点正文（P2.6 正文内容通道：query token 命中正文即准入
        union——存储签名被垃圾词淹没/缺失时的直接内容接地）。LLM 失效时不做
        启发式兜底裁剪（[7.7] 放行 union）——不再有 get_relevant_nodes 旧路，
        也不再有启发式关键词兜底。

        l1_reasons: {doc_id: 一句话选中理由}（[S6]#7/[S7] L1→L2 trace）。仅把
        本文档自己的理由子集下传 L2 节点裁定（防锚定，标注为判断而非事实）；
        None/缺该文档 → 传 None，行为与之前完全一致。

        query_entities/query_tokens: [S7] L0 证据束共享物（_act_tree_search 经
        evidence_ctx 提取下传）。非 None 时直接使用，不再 db.search_entities /
        内部 tokenize；None 时回退既有解析（v2 路径 / 无证据束调用方兼容）。

        node_entities: {node_id: [{"name","type","confidence"}]}（[S7]/[S5] 节点级
        实体直通）。非 None 时透传 enhance_and_select，节点实体证据改用证据束
        派生值（替代 resolve_node_profiles 的实体重读）；None 时行为不变。
        """
        funcs = _load_reason_funcs()
        spans_from_nodes = funcs.get("spans_from_nodes")

        if hasattr(self.client, "_ensure_doc_loaded"):
            self.client._ensure_doc_loaded(doc_id)
        doc = self.client.documents.get(doc_id)
        if not doc:
            logging.info("[Recall] doc=%s not loaded; skip", doc_id)
            return None
        structure = doc.get("structure", [])
        if not structure:
            logging.info("[Recall] doc=%s has empty structure; skip", doc_id)
            return None

        from .utils import create_node_mapping
        from .agentic.enhance import (
            UnifiedNodeEnhancement,
            resolve_query_entities,
            resolve_node_profiles,
            retry_on_pool_concern,
        )

        mapping = create_node_mapping(structure)
        candidates = [
            {
                "node_id": nid,
                "title": node.get("title") or "",
                "summary": node.get("summary") or "",
                # 正文内容通道（P2.6）：直接内容接地，存储签名淹没/缺失时保召回
                "text": node.get("text") or "",
            }
            for nid, node in mapping.items()
        ]

        # 签名解析（[3.4] 共享助手）：DB node_profiles 优先 → structure 键兜底
        db = getattr(self.client, "db", None)
        id_mapper = getattr(self.client, "_id_mapper", None)
        db_doc_id = None
        if db is not None and id_mapper is not None and hasattr(id_mapper, "to_db"):
            db_doc_id = id_mapper.to_db(doc_id)
        profiles = resolve_node_profiles(db, db_doc_id, mapping)
        if query_entities is None:
            query_entities = resolve_query_entities(db, query, limit=5) if db else []

        # 内容策略命中词并入关键词证据（v2 词面接地的保全，仅证据不裁决）。
        # 写时拷贝：structure 兜底签名的 list 与内存文档节点共享引用，
        # 不得把查询期命中词污染进在内存文档结构。
        if matched_info:
            for m in matched_info:
                if not isinstance(m, dict):
                    continue
                nid, kw = m.get("node_id"), m.get("keyword")
                if not nid or not isinstance(kw, str) or not kw.strip():
                    continue
                prof = profiles.get(nid)
                prof = dict(prof) if prof else {"entities": [], "keywords": [], "tags": []}
                kws = list(prof.get("keywords") or [])
                if kw.strip() not in kws:
                    kws.append(kw.strip())
                prof["keywords"] = kws
                profiles[nid] = prof

        # NFR4: 检索 LLM 调用点用 retrieve_model（model 兜底）
        enhancer = UnifiedNodeEnhancement(self.model, retrieve_model=self.retrieve_model)
        # [S6]#7/[S7]：只把本文档自己的 L1 理由子集下传（防锚定——理由仅为参考判断，
        # L2 以本层证据为准）；无理由时按旧签名调用（不传 l1_reasons），行为与改造前一致。
        call_kwargs = {}
        if l1_reasons:
            reason = l1_reasons.get(doc_id)
            if reason:
                call_kwargs["l1_reasons"] = {doc_id: reason}  # 仅本文档理由子集
        if query_tokens is not None:
            call_kwargs["query_tokens"] = query_tokens
        if node_entities is not None:
            call_kwargs["node_entities"] = node_entities
        result = await enhancer.enhance_and_select(
            query, candidates, profiles, query_entities=query_entities, **call_kwargs
        )

        # [3.2.1] pool_concern 重选（至多一次，二选一分支）走共享助手：
        # ① 有被截候选 → 放宽 union 上限重选；② 无被截候选 → force-all 全池
        # 直通重选（同样放宽 cap，防零信号候选垫底再截）。详见 retry_on_pool_concern。
        # T31.3：query_tokens/node_entities 与首选调用同源透传——重选路径不再
        # 重复分词/实体回退 node_profiles。
        result = await retry_on_pool_concern(
            enhancer, result, query, candidates, profiles,
            query_entities=query_entities,
            query_tokens=query_tokens,
            node_entities=node_entities,
        )

        selected_ids = result["selected_ids"]
        if not selected_ids:
            logging.info("[Recall] doc=%s empty selection (candidates=%d, concern_reason=%r)",
                         doc_id, len(candidates), result.get("concern_reason", ""))
            return None

        # 保持 LLM 精挑顺序（无重排）
        selected = [mapping[nid] for nid in selected_ids if nid in mapping]
        selected = [n for n in selected if n]
        if not selected:
            logging.info("[Recall] doc=%s selected ids missed mapping: %s", doc_id, selected_ids[:5])
            return None

        # [S10] 统一 span：spans_from_nodes（page/line 双跨度分派）。
        spans = spans_from_nodes(selected)
        pages = spans["pages"]
        lines = spans["lines"]
        # 统一 span 门槛（[S4]）：纯 span 判定，彻底移除 doc type hack——
        # page 跨度与 line 跨度皆空（无任何 locator）→ 无法接地取文本 → 拦截；
        # 有任一跨度即放行（PDF 凭 page、MD 凭 line，节点 text 直接组装）。
        if not spans["pages"] and not spans["lines"]:
            # 零跨度可诊断痕迹：旧索引（T7 前落库、无 line_num/end_line）的 MD 文档
            # 会被静默排除——落告警以便排查"选中了文档却无任何 locator"。
            logging.warning(
                "[Recall] doc=%s selected nodes yield no spans (legacy index?); dropped",
                doc_id,
            )
            return None

        # 相关度 = 召回覆盖度（selected / 全部候选节点），确定性 (0,1]，
        # 与单文档 _search_single 的 matched_docs score 语义统一。
        relevance_score = min(round(len(selected) / max(len(candidates), 1), 4), 1.0)

        return {
            "doc_id": doc_id,
            "doc": doc,
            "structure": structure,
            "selected": selected,
            "pages": pages,
            "lines": lines,
            "relevance_score": relevance_score,
            # [S13] 空选保底（T31.2）诊断标记：本篇 L2 经保底放行（union 信号
            # 最强子集）而非 LLM 精挑——消费方可观测、评测可统计。读 enhance 的
            # 独立布尔（审查 Minor-2：不词汇判定 concern_reason，防模型复述误标）。
            "l2_fallback": bool(result.get("selection_fallback")),
        }

    async def act_tree_search(
        self, query: str, candidate_docs: List[str],
        node_matches: Dict[str, List[Dict]] = None,
        doc_scores_out: Dict[str, float] = None,
        l1_reasons: Dict[str, str] = None,
        evidence_bundle: Dict = None,
        evidence_ctx: Dict = None,
    ) -> Tuple[str, List[dict], int, int, Dict[str, List[int]], List[dict]]:
        """Act 阶段树搜索。doc_scores_out 非 None 时回填每篇召回成功文档的
        证据派生分数（节点召回覆盖度 (0,1]）——供调用方构造 matched_docs，
        不再硬编码 1.0（T6.4 score 语义统一）。

        l1_reasons: {doc_id: 一句话选中理由}（[S6]#7/[S7] L1→L2 trace 预留槽位，
        默认 None；真正注入在 T9 L1 裁定改造）。None 时不向下传理由，行为不变。

        evidence_bundle: [S8] 证据束（{db_id: entry}）；非 None 时上下文组装排序
        主键改为 derive_evidence_score（次键 L1 裁定序/candidate_docs 顺序），
        与 matched_docs 同口径——保证支撑段可见，替代覆盖度分排序。None（v2 路径
        无证据束）时回退既有覆盖度分排序，保持既有调用方兼容。

        evidence_ctx: [S5]/[S7] 证据束上下文（{"tokens", "query_entities"}）。
        本任务在方法体内提取 ctx_qe/ctx_qt 并透传 _recall_nodes_for_doc → 最终
        enhance_and_select 复用 query_tokens、_recall_nodes_for_doc 不再重复
        resolve_query_entities。
        """
        funcs = _load_reason_funcs()
        spans_from_nodes = funcs.get("spans_from_nodes")
        if not spans_from_nodes:
            raise RuntimeError("main.py helpers not available")

        # [S6] 软归属去重：同一文档可经多个簇分支命中（软归属 ⇒ DAG），
        # 召回与预算只计一次。
        seen_docs = set()
        unique_docs = []
        for doc_id in candidate_docs:
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                unique_docs.append(doc_id)

        # [S7] L2 复用证据束 query 物：从 evidence_ctx 提取 query tokens/entities
        # 下传 _recall_nodes_for_doc（直通 enhance_and_select），消除重复 tokenize/
        # search_entities。None（v2 路径/无证据束）时各自回退既有解析。
        ctx_qe = evidence_ctx.get("query_entities") if evidence_ctx else None
        ctx_qt = evidence_ctx.get("tokens") if evidence_ctx else None

        # [S7] 节点级实体直通：从 evidence_bundle 按 doc 聚合 {node_id: [实体条目]}，
        # 透传 L2（替代 resolve_node_profiles 的实体重读）。需 db_id→uuid 反查——
        # 提前重建映射，不依赖下方排序段才建的 uuid_to_db。
        node_entities_by_doc: Dict[str, Dict[str, list]] = {}
        if evidence_bundle is not None:
            id_mapper = getattr(self.client, "_id_mapper", None)
            if id_mapper is not None:
                db_to_uuid = {int(db): uuid for uuid, db in id_mapper.items()}
            else:
                db_to_uuid = {
                    int(v): k
                    for k, v in (getattr(self.client, "_uuid_to_db", {}) or {}).items()
                }
            for db_id, e in evidence_bundle.items():
                uuid_id = db_to_uuid.get(int(db_id))
                if not uuid_id:
                    continue
                node_map = node_entities_by_doc.setdefault(uuid_id, {})
                for ent in ((e.get("channels") or {}).get("entity") or []):
                    nid = ent.get("node_id")
                    if not nid:
                        continue
                    node_map.setdefault(str(nid), []).append({
                        "name": ent.get("name"),
                        "type": ent.get("type"),
                        "confidence": ent.get("confidence"),
                    })

        # Parallel node recall across documents (with match info if available)
        recall_tasks = []
        for doc_id in unique_docs:
            call_kwargs = {
                "matched_info": node_matches.get(doc_id) if node_matches else None,
            }
            if l1_reasons is not None:
                call_kwargs["l1_reasons"] = l1_reasons
            if ctx_qe is not None:
                call_kwargs["query_entities"] = ctx_qe
            if ctx_qt is not None:
                call_kwargs["query_tokens"] = ctx_qt
            if evidence_bundle is not None:
                ne = node_entities_by_doc.get(doc_id)
                if ne:
                    call_kwargs["node_entities"] = ne
            recall_tasks.append(self.recall_nodes_for_doc(query, doc_id, **call_kwargs))
        recall_results = await asyncio.gather(*recall_tasks, return_exceptions=True)

        # Filter out failures and sort by relevance score (descending).
        # gather(return_exceptions=True) 会静默吞掉召回异常——显式记录，
        # 否则整批召回失败时只剩空上下文，无从定位。
        doc_results = []
        for doc_id, r in zip(unique_docs, recall_results):
            if isinstance(r, dict):
                doc_results.append(r)
            elif isinstance(r, Exception):
                logging.warning("[Act] node recall raised for doc=%s: %s: %s",
                                doc_id, type(r).__name__, r)

        # [S8] 上下文组装排序：有证据束时主键证据分（derive_evidence_score）、
        # 次键 L1 裁定序（candidate_docs 顺序）——与 matched_docs 同口径，替代
        # 覆盖度分排序（多文档上下文下保证支撑段靠前可见，verifier 才能引用实锤）。
        # 无证据束（v2 路径）回退既有覆盖度分排序，保持既有调用方兼容。
        if evidence_bundle is not None:
            from .agentic.evidence import derive_evidence_score
            id_mapper = getattr(self.client, "_id_mapper", None)
            if id_mapper is not None:
                uuid_to_db = dict(id_mapper.items())
            else:
                uuid_to_db = getattr(self.client, "_uuid_to_db", {}) or {}
            order = {doc_id: i for i, doc_id in enumerate(candidate_docs)}

            def _ev_sort_key(r):
                db_id = uuid_to_db.get(r.get("doc_id"))
                score = derive_evidence_score(evidence_bundle.get(db_id))
                return (-score, order.get(r.get("doc_id"), len(candidate_docs)))

            doc_results.sort(key=_ev_sort_key)
        else:
            doc_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        # 证据派生分数回填（预算截断前全量记录——召回并发完成，与准入无关）
        if doc_scores_out is not None:
            for r in doc_results:
                doc_scores_out[r["doc_id"]] = float(r.get("relevance_score", 0.0))

        contexts = []
        all_nodes = []
        source_docs = 0
        doc_pages_map: Dict[str, List[int]] = {}
        doc_lines_map: Dict[str, List[tuple]] = {}

        build_ctx = _load_reason_funcs().get("build_context_for_doc")

        # P0: 多文档上下文 token 预算——doc_results 已按相关度降序，预算满即停，
        # 保住最相关文档，避免多文档全文拼接冲爆上下文窗口。
        from .utils import count_tokens as _count_tokens
        try:
            from .reasoning import _get_max_context_tokens as _get_max_ctx
            _max_ctx_tokens = _get_max_ctx()
        except Exception:
            _max_ctx_tokens = 16000
        _used_ctx_tokens = 0

        for result in doc_results:
            doc_id = result["doc_id"]
            doc = result["doc"]
            structure = result["structure"]
            selected = result["selected"]
            pages = result["pages"]
            lines = result.get("lines", [])

            if build_ctx:
                context = build_ctx(doc, selected, pages)
            else:
                # Fallback: inline context assembly
                ctx_parts = [f"\n=== Document: {doc.get('doc_name', '')} ===\n"]
                if doc.get("type") == "pdf" and doc.get("pages"):
                    page_map = {p["page"]: p["content"] for p in doc["pages"]}
                    for p in sorted(set(pages)):
                        text = page_map.get(p, "")
                        if text:
                            ctx_parts.append(f"\n--- Page {p} ---\n{text}")
                elif doc.get("type") == "md" and structure:
                    for node in selected:
                        txt = node.get("text", "")
                        if txt:
                            ctx_parts.append(f"\n--- {node.get('title', '')} ---\n{txt}")
                context = "".join(ctx_parts) if len(ctx_parts) > 1 else ""

            if context:
                _ctx_tokens = _count_tokens(context)
                _over_budget = _used_ctx_tokens + _ctx_tokens > _max_ctx_tokens
                # 已有上下文且加本篇会超预算 → 停止（doc_results 按相关度降序）
                if _over_budget and contexts:
                    logging.info("[SuperTree] context budget reached; stop adding docs (%d kept)", len(contexts))
                    break
                contexts.append(context)
                _used_ctx_tokens += _ctx_tokens
                all_nodes.extend(selected)
                source_docs += 1
                doc_pages_map[doc_id] = sorted(set(pages))
                doc_lines_map[doc_id] = lines
                if _over_budget:
                    # P0 残留修复（[S9]）：首篇不再绕过预算检查——超大单篇
                    # 仍准入（否则完全没有上下文），但准入后立即停止。
                    logging.info("[SuperTree] first doc alone exceeds context budget; admitted and stopping")
                    break

        # Enrich context with entity relationships if available
        if hasattr(self.client, "db") and self.client.db and contexts:
            try:
                query_entities = self.client.db.search_entities(query, limit=5)
                entity_context_parts = []
                for entity in query_entities[:3]:
                    entity_id = entity.get("id")
                    if entity_id:
                        relations = self.client.db.get_entity_relations(entity_id)
                        if relations:
                            rel_text = f"\n=== Entity: {entity.get('name', '')} ===\n"
                            for rel in relations[:5]:
                                rel_text += f"- {rel.get('subject_name', '')} --{rel.get('predicate', '')}--> {rel.get('object_name', '')}\n"
                            entity_context_parts.append(rel_text)
                if entity_context_parts:
                    # P0 残留修复（[S9]）：实体关系上下文块计入同一预算，
                    # 剩余预算不足则跳过（不再无预算追加）。
                    entity_block = "\n".join(entity_context_parts)
                    if _used_ctx_tokens + _count_tokens(entity_block) <= _max_ctx_tokens:
                        contexts.append(entity_block)
                    else:
                        logging.info("[SuperTree] entity context block exceeds remaining budget; skipped")
            except Exception:
                pass

        # Build pages/lines with text content for UI display（[S10] span 分派：
        # page 走 page_map；line 输出节点标题 + 行区间，不再空文本）。
        pages_with_text = []
        node_by_id = {n.get("node_id"): n for n in all_nodes}
        for doc_id, page_nums in doc_pages_map.items():
            doc = self.client.documents.get(doc_id)
            if doc and doc.get("type") == "pdf" and doc.get("pages"):
                page_map = {p["page"]: p["content"] for p in doc["pages"]}
                for p in page_nums:
                    pages_with_text.append({
                        "doc_id": doc_id,
                        "page": p,
                        "text": (page_map.get(p, "") or "")[:500]
                    })
            else:
                for node_id, start_line, end_line in doc_lines_map.get(doc_id, []):
                    node = node_by_id.get(node_id, {})
                    pages_with_text.append({
                        "doc_id": doc_id,
                        "title": node.get("title", ""),
                        "start_line": start_line,
                        "end_line": end_line,
                    })

        return "\n\n".join(contexts), all_nodes, source_docs, len(all_nodes), doc_pages_map, pages_with_text
