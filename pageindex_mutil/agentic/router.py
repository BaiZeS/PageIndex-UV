import asyncio
import logging
from typing import List, Tuple, Dict

from .planner import RetrievalPlanner
from .recall_loop import _node_payload
from .verifier import CRAGVerifier
from ..super_tree import SuperTreeIndex


class AgenticRouter:
    """Orchestrate Plan -> Route -> Act -> Verify."""

    def __init__(self, client, model: str, retrieve_model: str = None):
        self.client = client
        self.model = model
        self.retrieve_model = retrieve_model
        self.planner = RetrievalPlanner(model, retrieve_model)
        self.verifier = CRAGVerifier(model, retrieve_model)
        self._main_funcs = None

        self.super_tree_index = None
        if hasattr(client, "super_tree_index") and client.super_tree_index:
            self.super_tree_index = client.super_tree_index

    # ------------------------------------------------------------------
    # Lazy import of reasoning helpers (avoid circular deps at import time)
    # ------------------------------------------------------------------
    def _load_main_funcs(self):
        if self._main_funcs is None:
            try:
                from ..reasoning import (
                    build_context_with_budget,
                    generate_answer,
                    spans_from_nodes,
                    build_context_for_doc,
                )
                self._main_funcs = {
                    "build_context_with_budget": build_context_with_budget,
                    "generate_answer": generate_answer,
                    "spans_from_nodes": spans_from_nodes,
                    "build_context_for_doc": build_context_for_doc,
                }
            except ImportError:
                self._main_funcs = {}
        return self._main_funcs

    # ------------------------------------------------------------------
    # Act — tree search + context assembly (parallelized)
    # ------------------------------------------------------------------
    async def _recall_nodes_for_doc(self, query: str, doc_id: str,
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
        funcs = self._load_main_funcs()
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

        from ..utils import create_node_mapping
        from .enhance import (
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

    async def _act_tree_search(
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
        funcs = self._load_main_funcs()
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
            recall_tasks.append(self._recall_nodes_for_doc(query, doc_id, **call_kwargs))
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
            from .evidence import derive_evidence_score
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

        build_ctx = self._load_main_funcs().get("build_context_for_doc")

        # P0: 多文档上下文 token 预算——doc_results 已按相关度降序，预算满即停，
        # 保住最相关文档，避免多文档全文拼接冲爆上下文窗口。
        from ..utils import count_tokens as _count_tokens
        try:
            from ..reasoning import _get_max_context_tokens as _get_max_ctx
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

    # ------------------------------------------------------------------
    # Super-Tree search
    # ------------------------------------------------------------------
    async def _search_super_tree(self, query: str, top_k: int = 3) -> Dict:
        """L0 证据束 → L1 Super-Tree selection → L2/L3 Act → Verify.

        CRAG expand 判定接入 AgenticRecallLoop（[S8] 点名扩召）：以本轮 L1 选中
        文档的证据分作为轮 1 融合序、本轮 Act 产出作为轮 1 上下文状态，续接轮只
        补 verifier 点名（need）对象。循环失败/异常时回退原 medium 响应。
        """
        logging.info("[SuperTree] query=%r top_k=%d", query, top_k)

        # L0 = 证据束（[S5] prefilter 改造）：build_evidence_bundle 是唯一召回源，
        # query tokens/entities/图谱距离一次计算全链引用。候选集 = bundle.keys()
        # 派生（derive_evidence_score 作排序标量），替代独立 prefilter 四通道打分。
        # HyDE 前置调用随 prefilter 一并移除（[S11]#1 / P2 审查 FOLLOWUP④）。
        from .evidence import build_evidence_bundle, derive_evidence_score
        db = getattr(self.client, "db", None)
        bundle: dict = {}
        evidence_ctx = None
        if db is not None:
            try:
                from ..utils import ConfigLoader
                cfg = ConfigLoader().load(None)
                bundle, evidence_ctx = build_evidence_bundle(
                    self.client, db, query,
                    topk=getattr(cfg, "l0_channel_topk", 30),
                )
            except Exception as e:
                logging.warning("[SuperTree] evidence bundle build failed: %s", e)

        candidate_db_ids = {
            db_id: derive_evidence_score(entry)
            for db_id, entry in bundle.items()
        }
        logging.info("[SuperTree] L0 candidates=%d", len(candidate_db_ids))

        if not candidate_db_ids:
            return {
                "query": query,
                "mode": "multi",
                "answer": "No relevant documents found.",
                "confidence": "low",
                "matched_docs": [],
                "selected_nodes": [],
                "pages": [],
            }

        # L1: Super-Tree LLM selection——证据束直通（[S6]#2）。
        selected_uuids, l1_reasons = await self.super_tree_index.select_documents(
            query, candidate_db_ids, evidence_bundle=bundle)
        logging.info("[SuperTree] L1 selected=%d docs: %s", len(selected_uuids), selected_uuids)
        if not selected_uuids:
            return {
                "query": query,
                "mode": "multi",
                "answer": "Super-Tree selection returned no documents.",
                "confidence": "low",
                "matched_docs": [],
                "selected_nodes": [],
                "pages": [],
            }

        # L2/L3: Act — tree search on selected documents (reuse existing _act_tree_search)
        doc_scores: Dict[str, float] = {}
        # [S6]#7 端到端：L1 选中理由经 conditional-kwarg 下传（有理由才传，行为不变）。
        act_kwargs = {"doc_scores_out": doc_scores}
        if l1_reasons:
            act_kwargs["l1_reasons"] = l1_reasons
        # [S8] 上下文组装排序同口径：证据束直通 _act_tree_search（T9 已构建）。
        # 空束（无 db/构建失败/无命中）不传——_act_tree_search 回退覆盖度分排序，
        # 与既有调用方（v2 路径）行为一致。
        if bundle:
            act_kwargs["evidence_bundle"] = bundle
        if evidence_ctx:
            act_kwargs["evidence_ctx"] = evidence_ctx
        try:
            ctx, nodes, src_docs, cov_nodes, doc_pages_map, pages_with_text = await self._act_tree_search(
                query, selected_uuids, **act_kwargs
            )
            logging.info("[SuperTree] L2/L3 context_len=%d src_docs=%d nodes=%d",
                        len(ctx), src_docs, cov_nodes)
        except Exception as e:
            logging.warning("Act phase failed: %s", e)
            return {
                "query": query,
                "mode": "multi",
                "answer": f"Failed to retrieve content: {e}",
                "confidence": "unknown",
                # 无节点级证据接地 → 不虚报匹配（与单文档空选择语义一致）
                "matched_docs": [],
                "selected_nodes": [],
                "pages": [],
            }

        # matched_docs score = 证据分（derive_evidence_score，通道命中加权标量；
        # 无 bundle 条目给 0），同分按 selected_uuids 顺序（L1 裁定序）排列（[S6]#8）。
        # 召回无果（精挑选空且无保底——零信号护栏拦截 / pool_concern 重试后仍空）
        # 的文档不进 matched（不虚报匹配）；[S13] 空选保底放行的文档进 matched
        # （带证据分，l2_fallback 标记可观测）。
        id_mapper = getattr(self.client, "_id_mapper", None)
        if id_mapper is not None:
            uuid_to_db = dict(id_mapper.items())
        else:
            uuid_to_db = getattr(self.client, "_uuid_to_db", {}) or {}
        from .evidence import derive_evidence_score
        matched = [
            {"doc_id": doc_id,
             "score": round(derive_evidence_score(bundle.get(uuid_to_db.get(doc_id))), 4)}
            for doc_id in selected_uuids if doc_id in doc_scores
        ]

        if not ctx:
            return {
                "query": query,
                "mode": "multi",
                "answer": "No relevant content found.",
                "confidence": "low",
                "matched_docs": matched,
                "selected_nodes": [],
                "pages": [],
            }

        # Generate answer
        funcs = self._load_main_funcs()
        generate_answer = funcs.get("generate_answer")
        if not generate_answer:
            return {
                "query": query,
                "mode": "multi",
                "answer": "Answer generation not available.",
                "confidence": "unknown",
                "matched_docs": matched,
                "selected_nodes": [],
                "pages": [],
            }

        answer = generate_answer(query, ctx)

        # Verify
        v = await asyncio.to_thread(
            self.verifier.verify, answer, ctx, query, src_docs, cov_nodes
        )
        if v.action == "refuse":
            return {
                "query": query,
                "mode": "multi",
                "answer": "I don't know.",
                "confidence": "low",
                "matched_docs": matched,
                "selected_nodes": _node_payload(nodes),
                "pages": [
                    {"doc_id": d, "pages": p}
                    for d, p in doc_pages_map.items()
                ],
            }

        # CRAG expand → Agentic 多轮召回循环（[S8] 点名扩召）。以本轮 L1 选中
        # 文档的证据分作轮 1 融合序、本轮 Act 产出作轮 1 上下文状态，续接轮只补
        # verifier 点名（need）对象。循环失败/异常回退原 medium 响应。
        if v.action == "expand":
            try:
                from .recall_loop import AgenticRecallLoop
                loop = AgenticRecallLoop(self)
                return await loop.retrieve(
                    query, top_k=top_k,
                    first_round_fused=[(doc_id, float(doc_scores.get(doc_id, 0.0))) for doc_id in selected_uuids],
                    first_round_ctx_state={"ctx": ctx, "nodes": nodes, "src_docs": src_docs,
                                           "cov_nodes": cov_nodes, "doc_pages_map": doc_pages_map,
                                           "pages_with_text": pages_with_text},
                    expand_need=getattr(v, "need", None),
                )
            except Exception as e:
                logging.warning("Agentic recall loop failed: %s", e)

        conf = "high" if v.action == "answer" else "medium"
        return {
            "query": query,
            "mode": "multi",
            "answer": answer,
            "confidence": conf,
            "matched_docs": matched,
            "selected_nodes": _node_payload(nodes),
            "pages": pages_with_text,
        }

    # ------------------------------------------------------------------
    # Public search
    # ------------------------------------------------------------------
    async def search(self, query: str, top_k: int = 3) -> Dict:
        """Unified single chain ([S4]): direct Super-Tree search only.

        The multi_hop pre-gate and _search_v2 fallback are removed (P2 T13).
        Without a super_tree_index (no-db / router-only), return a graceful
        empty response — no LLM call — preserving the upstream "Router not
        available" semantics of client.search.
        """
        # Direct Super-Tree search (single chain, no multi_hop pre-gate)
        if self.super_tree_index:
            try:
                result = await self._search_super_tree(query, top_k)
                logging.info("[Router] Super-Tree confidence=%s docs=%d",
                            result.get("confidence"), len(result.get("matched_docs", [])))
                return result
            except Exception as e:
                logging.warning("Super-Tree search failed: %s", e)
        return {
            "query": query,
            "mode": "multi",
            "answer": "Router not available. Initialise PageIndexClient with db_path= to enable multi-document search.",
            "confidence": "unknown",
            "matched_docs": [],
            "selected_nodes": [],
            "pages": [],
        }
