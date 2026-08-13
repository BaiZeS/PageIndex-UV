import asyncio
import logging
from typing import List, Tuple, Dict

from .planner import RetrievalPlanner
from .recall_loop import _node_payload
from .strategies import MetadataStrategy, SemanticsStrategy, ContentStrategy, DescriptionStrategy
from .verifier import CRAGVerifier
from .multi_hop import MultiHopReasoner
from ..super_tree import SuperTreeIndex


class AgenticRouter:
    """Orchestrate Plan -> Route -> Act -> Verify."""

    def __init__(self, client, model: str, retrieve_model: str = None):
        self.client = client
        self.model = model
        self.retrieve_model = retrieve_model
        self.planner = RetrievalPlanner(model, retrieve_model)
        self.metadata_strategy = MetadataStrategy()
        self.content_strategy = ContentStrategy(client)
        self.semantics_strategy = None
        self.description_strategy = DescriptionStrategy(model, retrieve_model)
        self.verifier = CRAGVerifier(model, retrieve_model)
        self.multi_hop_reasoner = MultiHopReasoner(model, retrieve_model)
        self._main_funcs = None

        if hasattr(client, "closet_index") and client.closet_index:
            self.semantics_strategy = SemanticsStrategy(client.closet_index)

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
                    pages_from_nodes,
                    build_context_for_doc,
                )
                self._main_funcs = {
                    "build_context_with_budget": build_context_with_budget,
                    "generate_answer": generate_answer,
                    "pages_from_nodes": pages_from_nodes,
                    "build_context_for_doc": build_context_for_doc,
                }
            except ImportError:
                self._main_funcs = {}
        return self._main_funcs

    # ------------------------------------------------------------------
    # Docs info
    # ------------------------------------------------------------------
    def _build_docs_info(self) -> List[Dict]:
        docs_info = []
        # Prefer db, but only include docs that are loaded in memory
        # (Act phase needs in-memory structure/pages)
        if hasattr(self.client, "db") and self.client.db:
            try:
                # Build reverse mapping: db_id -> uuid
                id_mapper = getattr(self.client, "_id_mapper", None)
                if id_mapper:
                    db_to_uuid = {db: uuid for uuid, db in id_mapper.items()}
                else:
                    db_to_uuid = {v: k for k, v in getattr(self.client, "_uuid_to_db", {}).items()}
                for doc in self.client.db.get_all_documents():
                    doc_id_int = doc["id"]
                    if doc_id_int not in db_to_uuid:
                        continue
                    doc_id = db_to_uuid[doc_id_int]
                    if doc_id not in self.client.documents:
                        continue
                    top = self.client.db.get_top_level_nodes(doc["id"])
                    docs_info.append(
                        {
                            "doc_id": doc_id,
                            "doc_name": doc.get("pdf_name", ""),
                            "description": doc.get("doc_description", ""),
                            "top_level_sections": [
                                n.get("title") for n in top if n.get("title")
                            ],
                        }
                    )
                if docs_info:
                    return docs_info
            except Exception:
                pass

        # Fallback to in-memory documents
        for doc_id, doc in self.client.documents.items():
            docs_info.append(
                {
                    "doc_id": doc_id,
                    "doc_name": doc.get("doc_name", ""),
                    "description": doc.get("doc_description", ""),
                    "top_level_sections": [],
                }
            )
        return docs_info

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------
    # RRF constant; see "Reciprocal Rank Fusion outperforms BM25 and Vector Search"
    _RRF_K = 60

    @staticmethod
    def _weighted_rrf(
        results_dict: Dict[str, List[Tuple[str, int]]],
        weights: Dict[str, float],
        k: int = None,
    ) -> List[Tuple[str, float]]:
        if k is None:
            k = AgenticRouter._RRF_K
        scores: Dict[str, float] = {}
        for strategy, results in results_dict.items():
            weight = weights.get(strategy, 1.0)
            for doc_id, rank in results:
                scores[doc_id] = scores.get(doc_id, 0.0) + weight * (
                    1.0 / (k + rank)
                )
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Route — parallel strategies
    # ------------------------------------------------------------------
    async def _run_strategies(
        self, query: str, docs_info: List[Dict], weights: Dict[str, float]
    ) -> Tuple[Dict[str, List[Tuple[str, int]]], Dict[str, List[Dict]]]:
        """Run retrieval strategies and return results + node match info.

        Returns:
            results: {strategy_name: [(doc_id, rank)]}
            node_matches: {doc_id: [{"node_id", "keyword", "context"}]}
        """
        tasks = {}
        tasks["metadata"] = asyncio.to_thread(
            self.metadata_strategy.search, query, docs_info
        )
        # Content-based search: always run (cheap, keyword-only)
        tasks["content"] = asyncio.to_thread(
            self.content_strategy.search, query, docs_info
        )
        if self.semantics_strategy and weights.get("semantics", 0) > 0:
            tasks["semantics"] = asyncio.to_thread(
                self.semantics_strategy.search, query, docs_info
            )
        if weights.get("description", 0) > 0:
            tasks["description"] = asyncio.to_thread(
                self.description_strategy.search, query, docs_info
            )

        results: Dict[str, List[Tuple[str, int]]] = {}
        node_matches: Dict[str, List[Dict]] = {}

        if tasks:
            done = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for name, res in zip(tasks.keys(), done):
                if isinstance(res, Exception):
                    logging.warning("Strategy %s failed: %s", name, res)
                    results[name] = []
                elif name == "content" and isinstance(res, list):
                    # ContentStrategy returns (doc_id, hit_count, matched_nodes)
                    content_results = []
                    for item in res:
                        if len(item) == 3:
                            doc_id, score, matches = item
                            content_results.append((doc_id, score))
                            node_matches[doc_id] = matches
                        else:
                            content_results.append(item)
                    # 命中数不是 rank：按命中数降序（已排序，防御性再排一次）
                    # 转换为真实 1-based rank 后再喂 RRF，否则命中越多分越低。
                    content_results.sort(key=lambda t: t[1], reverse=True)
                    results[name] = [
                        (doc_id, rank + 1)
                        for rank, (doc_id, _count) in enumerate(content_results)
                    ]
                else:
                    results[name] = res

        return results, node_matches

    # ------------------------------------------------------------------
    # Act — tree search + context assembly (parallelized)
    # ------------------------------------------------------------------
    async def _recall_nodes_for_doc(self, query: str, doc_id: str,
                                      matched_info: List[Dict] = None,
                                      l1_reasons: Dict[str, str] = None):
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
        """
        funcs = self._load_main_funcs()
        pages_from_nodes = funcs.get("pages_from_nodes")

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
        reason_subset = None
        if l1_reasons:
            reason = l1_reasons.get(doc_id)
            if reason:
                reason_subset = {doc_id: reason}
        if reason_subset is not None:
            result = await enhancer.enhance_and_select(
                query, candidates, profiles, query_entities=query_entities,
                l1_reasons=reason_subset,
            )
        else:
            result = await enhancer.enhance_and_select(
                query, candidates, profiles, query_entities=query_entities,
            )

        # [3.2.1] pool_concern 重选（至多一次，二选一分支）走共享助手：
        # ① 有被截候选 → 放宽 union 上限重选；② 无被截候选 → force-all 全池
        # 直通重选（同样放宽 cap，防零信号候选垫底再截）。详见 retry_on_pool_concern。
        result = await retry_on_pool_concern(
            enhancer, result, query, candidates, profiles,
            query_entities=query_entities,
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

        pages = pages_from_nodes(selected)
        if not pages:
            if doc.get("type") == "pdf":
                return None
            # MD 文档节点无页码索引；上下文由 build_context_for_doc 走节点 text 组装

        # 相关度 = 召回覆盖度（selected / 全部候选节点），确定性 (0,1]，
        # 与单文档 _search_single 的 matched_docs score 语义统一。
        relevance_score = min(round(len(selected) / max(len(candidates), 1), 4), 1.0)

        return {
            "doc_id": doc_id,
            "doc": doc,
            "structure": structure,
            "selected": selected,
            "pages": pages,
            "relevance_score": relevance_score,
        }

    async def _act_tree_search(
        self, query: str, candidate_docs: List[str],
        node_matches: Dict[str, List[Dict]] = None,
        doc_scores_out: Dict[str, float] = None,
        l1_reasons: Dict[str, str] = None,
    ) -> Tuple[str, List[dict], int, int, Dict[str, List[int]], List[dict]]:
        """Act 阶段树搜索。doc_scores_out 非 None 时回填每篇召回成功文档的
        证据派生分数（节点召回覆盖度 (0,1]）——供调用方构造 matched_docs，
        不再硬编码 1.0（T6.4 score 语义统一）。

        l1_reasons: {doc_id: 一句话选中理由}（[S6]#7/[S7] L1→L2 trace 预留槽位，
        默认 None；真正注入在 T9 L1 裁定改造）。None 时不向下传理由，行为不变。
        """
        funcs = self._load_main_funcs()
        pages_from_nodes = funcs.get("pages_from_nodes")
        if not pages_from_nodes:
            raise RuntimeError("main.py helpers not available")

        # [S6] 软归属去重：同一文档可经多个簇分支命中（软归属 ⇒ DAG），
        # 召回与预算只计一次。
        seen_docs = set()
        unique_docs = []
        for doc_id in candidate_docs:
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                unique_docs.append(doc_id)

        # Parallel node recall across documents (with match info if available)
        recall_tasks = []
        for doc_id in unique_docs:
            call_kwargs = {
                "matched_info": node_matches.get(doc_id) if node_matches else None,
            }
            if l1_reasons is not None:
                call_kwargs["l1_reasons"] = l1_reasons
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

        doc_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        # 证据派生分数回填（预算截断前全量记录——召回并发完成，与准入无关）
        if doc_scores_out is not None:
            for r in doc_results:
                doc_scores_out[r["doc_id"]] = float(r.get("relevance_score", 0.0))

        contexts = []
        all_nodes = []
        source_docs = 0
        doc_pages_map: Dict[str, List[int]] = {}

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

        # Build pages with text content for UI display
        pages_with_text = []
        for doc_id, page_nums in doc_pages_map.items():
            doc = self.client.documents.get(doc_id)
            if not doc or doc.get("type") != "pdf" or not doc.get("pages"):
                for p in page_nums:
                    pages_with_text.append({"doc_id": doc_id, "page": p})
                continue
            page_map = {p["page"]: p["content"] for p in doc["pages"]}
            for p in page_nums:
                pages_with_text.append({
                    "doc_id": doc_id,
                    "page": p,
                    "text": (page_map.get(p, "") or "")[:500]
                })

        return "\n\n".join(contexts), all_nodes, source_docs, len(all_nodes), doc_pages_map, pages_with_text

    # ------------------------------------------------------------------
    # Content-based lexical fallback (robustness for structureless docs)
    # ------------------------------------------------------------------
    def _content_fallback(self, query: str, top_k: int) -> list:
        """BM25 over raw document content; used when the structure-based pipeline
        yields nothing (e.g. short/structureless passages), so search never returns
        empty just because a document has no hierarchical structure."""
        import math, os
        docs = getattr(self.client, "documents", {})
        items = []
        for uuid_id, info in docs.items():
            content = ""
            try:
                pages = info.get("pages")
                if pages:
                    content = "\n".join(p.get("content", "") for p in pages if isinstance(p, dict))
            except Exception:
                pass
            if not content:
                path = info.get("path")
                if path and os.path.exists(path):
                    try:
                        with open(path, encoding="utf-8") as f:
                            content = f.read()
                    except Exception:
                        content = ""
            if not content:
                content = f"{info.get('doc_name','') or ''} {info.get('doc_description','') or ''}"
            items.append((uuid_id, content))
        if not items:
            return []
        try:
            import jieba
            tok = lambda t: [x for x in jieba.lcut(t or "") if x.strip()]
        except Exception:
            tok = lambda t: (t or "").split()
        qt = tok(query)
        dt = [tok(c) for _, c in items]
        lens = [len(x) for x in dt]
        avgdl = sum(lens) / len(lens) if lens else 1
        df = {}
        for toks in dt:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = len(items); k1 = 1.5; b = 0.75
        def idf(t):
            return math.log((n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1)
        scored = []
        for i, (uuid_id, _c) in enumerate(items):
            tf = {}
            for t in dt[i]:
                tf[t] = tf.get(t, 0) + 1
            s = sum(idf(q) * (tf.get(q, 0) * (k1 + 1)) / (tf.get(q, 0) + k1 * (1 - b + b * lens[i] / avgdl))
                    for q in qt if tf.get(q, 0) > 0)
            scored.append((uuid_id, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{"doc_id": u, "score": float(s)} for u, s in scored[:top_k] if s > 0]

    # ------------------------------------------------------------------
    # Super-Tree search
    # ------------------------------------------------------------------
    async def _search_super_tree(self, query: str, top_k: int = 3) -> Dict:
        """L0 prefilter → L1 Super-Tree selection → L2/L3 Act → Verify.

        Note (T6.3): 超树路径未接入 AgenticRecallLoop——expand 判定在此仅回退为
        medium 置信响应。原因：本路径无策略融合池（候选来自 prefilter+LLM 选择，
        非 _run_strategies/_weighted_rrf），循环的"逐轮放宽融合切窗"无对称语义。
        v2 路径的 expand 委派见 _search_v2（spec [3.5]）。
        """
        logging.info("[SuperTree] query=%r top_k=%d", query, top_k)

        # HyDE: generate hypothetical answer for query expansion
        hyde_answer = None
        try:
            plan = await self.planner.plan(query)
            if plan.queries and len(plan.queries) > 1:
                hyde_answer = plan.queries[1]  # First variant after original query
                logging.info("[SuperTree] HyDE answer=%r", hyde_answer)
        except Exception as e:
            logging.warning("[SuperTree] HyDE planning failed: %s", e)

        # L0: Dual-channel prefilter (with optional HyDE query expansion)
        candidate_db_ids = self.super_tree_index.prefilter(query)

        # If HyDE generated a different query, also run prefilter on it
        if hyde_answer and hyde_answer != query:
            hyde_scores = self.super_tree_index.prefilter(hyde_answer)
            for doc_id, score in hyde_scores.items():
                # Boost docs that match both original and HyDE queries
                existing = candidate_db_ids.get(doc_id, 0.0)
                candidate_db_ids[doc_id] = existing + score * 0.5

        logging.info("[SuperTree] L0 candidates=%d", len(candidate_db_ids))

        if not candidate_db_ids:
            fallback = self._content_fallback(query, top_k)
            if fallback:
                logging.info("[SuperTree] prefilter empty; content fallback -> %d docs", len(fallback))
                return {
                    "query": query,
                    "mode": "multi",
                    "answer": "",
                    "confidence": "low",
                    "matched_docs": fallback,
                    "selected_nodes": [],
                    "pages": [],
                }
            return {
                "query": query,
                "mode": "multi",
                "answer": "No relevant documents found in prefilter.",
                "confidence": "low",
                "matched_docs": [],
                "selected_nodes": [],
                "pages": [],
            }

        # L1: Super-Tree LLM selection
        selected_uuids = await self.super_tree_index.select_documents(query, candidate_db_ids)
        logging.info("[SuperTree] L1 selected=%d docs: %s", len(selected_uuids), selected_uuids)
        if not selected_uuids:
            fallback = self._content_fallback(query, top_k)
            if fallback:
                logging.info("[SuperTree] L1 empty; content fallback -> %d docs", len(fallback))
                return {
                    "query": query,
                    "mode": "multi",
                    "answer": "",
                    "confidence": "low",
                    "matched_docs": fallback,
                    "selected_nodes": [],
                    "pages": [],
                }
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
        try:
            ctx, nodes, src_docs, cov_nodes, doc_pages_map, pages_with_text = await self._act_tree_search(
                query, selected_uuids, doc_scores_out=doc_scores
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

        # matched_docs score = 节点召回覆盖度（evidence-derived，(0,1]），
        # 取代旧硬编码 1.0；召回无果（LLM 精挑为空）的文档不进 matched。
        matched = [
            {"doc_id": doc_id, "score": round(doc_scores[doc_id], 4)}
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
    # v2 search (Plan -> Route -> Act -> Verify)
    # ------------------------------------------------------------------
    async def _search_v2(self, query: str, top_k: int = 3) -> Dict:
        logging.info("[v2] query=%r top_k=%d", query, top_k)
        # Plan
        plan = await self.planner.plan(query)
        logging.info("[v2] Plan: type=%s queries=%d weights=%s",
                    plan.query_type, len(plan.queries), plan.weights)

        # Docs info
        docs_info = self._build_docs_info()
        if not docs_info:
            return {
                "query": query,
                "mode": "multi",
                "answer": "No documents indexed.",
                "confidence": "unknown",
                "matched_docs": [],
                "selected_nodes": [],
                "pages": [],
            }

        # Route
        results, node_matches = await self._run_strategies(
            plan.queries[0], docs_info, plan.weights
        )

        # Run semantics on query variants too
        if self.semantics_strategy and len(plan.queries) > 1:
            best_sem: Dict[str, int] = {}
            # Seed with original semantics results
            for doc_id, rank in results.get("semantics", []):
                best_sem[doc_id] = rank
            for q in plan.queries[1:]:
                try:
                    r = await asyncio.to_thread(
                        self.semantics_strategy.search, q, docs_info
                    )
                    for doc_id, rank in r:
                        if doc_id not in best_sem or rank < best_sem[doc_id]:
                            best_sem[doc_id] = rank
                except Exception:
                    pass
            if best_sem:
                results["semantics"] = sorted(
                    best_sem.items(), key=lambda x: x[1]
                )

        # RRF
        fused = self._weighted_rrf(results, plan.weights)
        if not fused:
            return {
                "query": query,
                "mode": "multi",
                "answer": "No relevant documents found.",
                "confidence": "unknown",
                "matched_docs": [],
                "selected_nodes": [],
                "pages": [],
            }

        candidates = [doc_id for doc_id, _ in fused[:top_k]]
        matched = [
            {"doc_id": doc_id, "score": round(score, 4)}
            for doc_id, score in fused[:top_k]
        ]

        # Act (with node matches from ContentStrategy)
        try:
            ctx, nodes, src_docs, cov_nodes, doc_pages_map, pages_with_text = await self._act_tree_search(
                query, candidates, node_matches=node_matches
            )
        except Exception as e:
            logging.warning("Act phase failed: %s", e)
            return {
                "query": query,
                "mode": "multi",
                "answer": f"Failed to retrieve content: {e}",
                "confidence": "unknown",
                "matched_docs": matched,
                "selected_nodes": [],
                "pages": [],
            }

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

        # Expand on medium confidence → Agentic 多轮召回循环（spec [3.5]）。
        # 替换旧的单次 expand 分支（pages_with_text2.items() AttributeError）：
        # 逐轮放宽召回 + verifier 判停 + 延迟/预算保护。轮 1（本次）的融合序与
        # 已召回文档作为排除种子传入，循环从轮 2 继续并返回最终响应。
        # fused 池已被轮 1 吃满（无更多文档可扩召）时保持原 medium 响应。
        if v.action == "expand" and len(fused) > top_k:
            try:
                from .recall_loop import AgenticRecallLoop
                loop = AgenticRecallLoop(self)
                return await loop.retrieve(
                    query,
                    top_k=top_k,
                    first_round_fused=fused,
                    first_round_ctx_state={
                        "ctx": ctx,
                        "nodes": nodes,
                        "src_docs": src_docs,
                        "cov_nodes": cov_nodes,
                        "doc_pages_map": doc_pages_map,
                        "pages_with_text": pages_with_text,
                    },
                    first_round_node_matches=node_matches,
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
        """Try multi-hop reasoning first, then Super-Tree, fallback to v2."""
        # Try multi-hop reasoning for decomposable queries
        if self.super_tree_index and hasattr(self.client, "db") and self.client.db:
            try:
                result = await self.multi_hop_reasoner.execute(
                    query, self, self.client.db, top_k=top_k
                )
                logging.info("[Router] Multi-hop hop_count=%d confidence=%s",
                            result.get("hop_count", 0), result.get("confidence"))
                return result
            except Exception as e:
                logging.warning("Multi-hop reasoning failed, falling back to Super-Tree: %s", e)

        # Fallback: direct Super-Tree search
        if self.super_tree_index:
            try:
                result = await self._search_super_tree(query, top_k)
                logging.info("[Router] Super-Tree confidence=%s docs=%d",
                            result.get("confidence"), len(result.get("matched_docs", [])))
                return result
            except Exception as e:
                logging.warning("Super-Tree search failed, falling back to v2: %s", e)
        result = await self._search_v2(query, top_k)
        logging.info("[Router] v2 confidence=%s docs=%d",
                    result.get("confidence"), len(result.get("matched_docs", [])))
        return result
