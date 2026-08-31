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
    # Act — tree search + context assembly。T32.2：两方法实现已整体移入
    # SuperTreeIndex（recall_nodes_for_doc / act_tree_search，检索引擎职责），
    # 此处保留薄委托兼容既有调用面（router 纯编排）。
    # ------------------------------------------------------------------
    async def _recall_nodes_for_doc(self, *a, **kw):
        """T32.2 已移入 SuperTreeIndex.recall_nodes_for_doc，保留委托兼容既有调用面。"""
        return await self.super_tree_index.recall_nodes_for_doc(*a, **kw)

    async def _act_tree_search(self, *a, **kw):
        """T32.2 已移入 SuperTreeIndex.act_tree_search（6-tuple 透传），保留委托兼容既有调用面。"""
        return await self.super_tree_index.act_tree_search(*a, **kw)

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

        # L2/L3: Act — tree search on selected documents (reuse SuperTreeIndex.act_tree_search, T32.2)
        doc_scores: Dict[str, float] = {}
        # [S6]#7 端到端：L1 选中理由经 conditional-kwarg 下传（有理由才传，行为不变）。
        act_kwargs = {"doc_scores_out": doc_scores}
        if l1_reasons:
            act_kwargs["l1_reasons"] = l1_reasons
        # [S8] 上下文组装排序同口径：证据束直通 act_tree_search（T9 已构建）。
        # 空束（无 db/构建失败/无命中）不传——act_tree_search 回退覆盖度分排序，
        # 与既有调用方（v2 路径）行为一致。
        if bundle:
            act_kwargs["evidence_bundle"] = bundle
        if evidence_ctx:
            act_kwargs["evidence_ctx"] = evidence_ctx
        try:
            ctx, nodes, src_docs, cov_nodes, doc_pages_map, pages_with_text = await self.super_tree_index.act_tree_search(
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
