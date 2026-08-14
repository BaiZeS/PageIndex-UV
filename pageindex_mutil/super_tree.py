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
