"""Multi-hop loop reasoning with graph-guided navigation ([S8]).

Implements the reasoning-retrieval loop: query decomposability judgment →
per-hop navigation → entity extraction → graph-guided next hop →
multi-hop context aggregation → answer generation.
"""
import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from ..utils import llm_acompletion, llm_completion, extract_json, count_tokens

logger = logging.getLogger(__name__)

# Default max hops and token budget
_DEFAULT_MAX_HOPS = 3
_DEFAULT_TOKEN_BUDGET = 16000


class MultiHopReasoner:
    """Reasoning-retrieval loop that navigates the entity graph across hops."""

    def __init__(self, model: str, retrieve_model: str = None):
        self.model = model
        self.retrieve_model = retrieve_model

    @property
    def _llm_model(self) -> str:
        return self.retrieve_model or self.model

    async def execute(
        self,
        query: str,
        router: Any,
        db: Any,
        max_hops: int = _DEFAULT_MAX_HOPS,
        token_budget: int = _DEFAULT_TOKEN_BUDGET,
        top_k: int = 3,
    ) -> Dict:
        """Execute multi-hop reasoning for a query.

        Args:
            query: The user query.
            router: AgenticRouter instance (for _search_super_tree and
                super_tree_index.act_tree_search — T32.2 后 act 实现属检索引擎).
            db: PageIndexDB instance (for entity graph queries).
            max_hops: Maximum number of hops (default 3).
            token_budget: Total token budget across all hops (default 16000).
            top_k: Top-k documents per search (default 3).

        Returns:
            Dict with keys: query, answer, confidence, matched_docs, selected_nodes,
            pages, hop_count, hop_contexts.
        """
        # Step 1: Judge decomposability
        is_multi_hop = await self._judge_decomposable(query)

        if not is_multi_hop:
            result = await router._search_super_tree(query, top_k)
            result["hop_count"] = 1
            return result

        # Step 2: Multi-hop loop
        hop_contexts: List[str] = []
        all_matched_docs = []
        all_selected_nodes = []
        all_pages = []
        used_tokens = 0
        current_query = query
        visited_entities = set()

        # entity→document ids from the graph are DB integers; downstream recall
        # （recall_nodes_for_doc，T32.2 后属 SuperTreeIndex） is keyed by UUID, so resolve them here.
        id_mapper = getattr(getattr(router, "client", None), "_id_mapper", None)
        to_uuid = getattr(id_mapper, "to_uuid", None)

        for hop_idx in range(max_hops):
            logger.info("[MultiHop] hop %d/%d query=%r", hop_idx + 1, max_hops, current_query)

            # Navigate: use tree navigation for current sub-query
            hop_scores: Dict[str, float] = {}
            try:
                candidate_docs = self._get_candidate_docs(db, current_query, visited_entities, to_uuid)
                # T32.2: act 树搜索已移入 SuperTreeIndex（引擎公共 API）；无引擎的
                # 旧式调用方（mock/裸 router）回退薄委托面，保持可编译。
                engine = getattr(router, "super_tree_index", None)
                act_tree_search = (
                    engine.act_tree_search if engine is not None
                    else router._act_tree_search
                )
                ctx, nodes, src_docs, cov_nodes, doc_pages_map, pages_with_text = (
                    await act_tree_search(
                        current_query, candidate_docs, doc_scores_out=hop_scores
                    )
                )
            except Exception as e:
                logger.warning("[MultiHop] hop %d tree search failed: %s", hop_idx + 1, e)
                ctx = ""

            if not ctx:
                logger.info("[MultiHop] hop %d yielded empty context; stopping", hop_idx + 1)
                break

            # Check token budget
            ctx_tokens = count_tokens(ctx)
            if used_tokens + ctx_tokens > token_budget:
                logger.info("[MultiHop] token budget exceeded at hop %d; stopping", hop_idx + 1)
                # If we have no context yet, admit this one; otherwise stop
                if hop_contexts:
                    break

            hop_contexts.append(ctx)
            used_tokens += ctx_tokens
            all_selected_nodes.extend(nodes or [])
            all_pages.extend(pages_with_text or [])
            if doc_pages_map:
                # score = 节点召回覆盖度（act_tree_search 经 doc_scores_out 回填，
                # evidence-derived，(0,1]）；覆盖度缺失（防御场景）回退 1.0。
                seen_docs = {d["doc_id"] for d in all_matched_docs}
                all_matched_docs.extend(
                    {"doc_id": did, "score": round(float(hop_scores.get(did, 1.0)), 4)}
                    for did in doc_pages_map
                    if did not in seen_docs
                )

            # Extract: LLM extracts intermediate entities/facts
            extraction = await self._extract_intermediate(query, current_query, ctx)

            entities = extraction.get("entities", [])
            facts = extraction.get("facts", [])
            next_hint = extraction.get("next_hop_hint", "")

            # Record visited entities to avoid loops
            for ent in entities:
                visited_entities.add(ent.lower())

            # Early termination: no new info to follow
            if not next_hint and not entities:
                logger.info("[MultiHop] hop %d: no next hop hint; stopping", hop_idx + 1)
                break

            # Guide: use entity graph to find next entity/topic
            next_sub_query = await self._guide_next_hop(
                query=query,
                current_query=current_query,
                next_hint=next_hint,
                entities=entities,
                db=db,
                visited_entities=visited_entities,
            )

            if not next_sub_query:
                logger.info("[MultiHop] hop %d: graph provided no next hop; stopping", hop_idx + 1)
                break

            current_query = next_sub_query

        # Step 3: Generate final answer from aggregated multi-hop context
        aggregated_ctx = "\n\n---\n\n".join(hop_contexts)

        funcs = router._load_main_funcs()
        generate_answer = funcs.get("generate_answer")
        if generate_answer:
            answer = generate_answer(query, aggregated_ctx)
        else:
            answer = llm_completion(self._llm_model, self._build_answer_prompt(query, aggregated_ctx), thinking_disabled=True)

        answer = answer or "No answer generated."

        # Step 4: CRAG verification (same pattern as router tree-search paths)
        verifier = getattr(router, "verifier", None)
        verify_fn = getattr(verifier, "verify", None)
        if callable(verify_fn):
            v = await asyncio.to_thread(
                verify_fn, answer, aggregated_ctx, query,
                len(all_matched_docs), len(all_selected_nodes),
            )
            if v.action == "refuse":
                answer = "I don't know."
                confidence = "low"
            else:
                confidence = "high" if v.action == "answer" else "medium"
        else:
            confidence = "medium" if len(hop_contexts) >= 2 else "low"

        return {
            "query": query,
            "answer": answer,
            "confidence": confidence,
            "matched_docs": all_matched_docs,
            "selected_nodes": all_selected_nodes,
            "pages": all_pages,
            "hop_count": len(hop_contexts),
            "hop_contexts": hop_contexts,
        }

    async def _judge_decomposable(self, query: str) -> bool:
        """LLM judges whether a query is decomposable into multi-hop sub-queries."""
        prompt = (
            "你是一个查询分析专家。判断用户问题是否需要多跳推理才能回答。\n"
            "多跳推理意味着：需要先从一个来源获取信息，再用该信息去另一个来源查找，\n"
            "最终才能完整回答问题。\n\n"
            f"用户问题: {query}\n\n"
            "返回JSON格式:\n"
            '{"decomposable": true/false}\n'
            "如果不需要多跳，decomposable 为 false。\n"
            "直接返回JSON，不要其他内容。"
        )

        response = await llm_acompletion(self._llm_model, prompt, thinking_disabled=True)
        logger.info("[MultiHop] decomposable response=%r", (response or "")[:100])
        if not response:
            return False

        data = extract_json(response)
        if not isinstance(data, dict):
            return False

        return bool(data.get("decomposable", False))

    async def _extract_intermediate(
        self, original_query: str, current_sub_query: str, context: str
    ) -> Dict:
        """LLM extracts intermediate entities, facts, and next-hop hint from context."""
        prompt = (
            "你是一个信息提取专家。从以下上下文中提取中间实体、关键事实和下一跳提示。\n\n"
            f"原始问题: {original_query}\n"
            f"当前子查询: {current_sub_query}\n\n"
            f"上下文:\n{context[:3000]}\n\n"
            "返回JSON格式:\n"
            '{"entities": ["实体1", "实体2"], "facts": ["事实1"], "next_hop_hint": "需要进一步查询的实体或主题"}\n'
            "如果没有需要进一步查询的内容，next_hop_hint 为空字符串。\n"
            "直接返回JSON，不要其他内容。"
        )

        response = await llm_acompletion(self._llm_model, prompt, thinking_disabled=True)
        if not response:
            return {"entities": [], "facts": [], "next_hop_hint": ""}

        data = extract_json(response)
        if not isinstance(data, dict):
            return {"entities": [], "facts": [], "next_hop_hint": ""}

        return {
            "entities": data.get("entities", []) if isinstance(data.get("entities"), list) else [],
            "facts": data.get("facts", []) if isinstance(data.get("facts"), list) else [],
            "next_hop_hint": data.get("next_hop_hint", "") or "",
        }

    async def _guide_next_hop(
        self,
        query: str,
        current_query: str,
        next_hint: str,
        entities: List[str],
        db: Any,
        visited_entities: set,
    ) -> Optional[str]:
        """Use entity graph to guide the next hop sub-query.

        Searches for the next_hint entity, finds its relations, and returns
        a new sub-query targeting the related entity's documents.
        """
        if not db:
            return None

        search_term = next_hint or (entities[0] if entities else "")
        if not search_term:
            return None

        try:
            found_entities = db.search_entities(search_term, limit=5)
        except Exception as e:
            logger.warning("[MultiHop] entity search failed: %s", e)
            return None

        if not found_entities:
            return None

        # Find relations from the found entity
        best_next_entity = None
        best_confidence = 0.0

        for entity in found_entities:
            entity_id = entity.get("id")
            if not entity_id:
                continue
            entity_name = entity.get("name", "").lower()
            if entity_name in visited_entities:
                continue

            try:
                relations = db.get_entity_relations(entity_id)
            except Exception:
                relations = []

            for rel in relations:
                obj_name = rel.get("object_name", "")
                subj_name = rel.get("subject_name", "")
                conf = rel.get("confidence", 0.0)

                # Pick the entity on the other side of the relation
                candidate = None
                if subj_name.lower() == entity_name.lower() or search_term.lower() in subj_name.lower():
                    candidate = obj_name
                elif obj_name.lower() == entity_name.lower() or search_term.lower() in obj_name.lower():
                    candidate = subj_name

                if candidate and candidate.lower() not in visited_entities and conf > best_confidence:
                    best_next_entity = candidate
                    best_confidence = conf

            if best_next_entity:
                break

        if not best_next_entity:
            return None

        visited_entities.add(best_next_entity.lower())
        return f"{query} (相关主题: {best_next_entity})"

    def _get_candidate_docs(
        self,
        db: Any,
        query: str,
        visited_entities: set,
        to_uuid: Optional[Callable[[int], Optional[str]]] = None,
    ) -> List[str]:
        """Get candidate document UUIDs from entity graph for the current sub-query.

        Entity tables reference documents by DB integer id, but downstream tree
        navigation （recall_nodes_for_doc，T32.2 后属 SuperTreeIndex） is keyed by UUID. When a `to_uuid`
        mapper is provided, DB ids are converted at this outlet; unmapped ids
        are dropped (they could never be retrieved anyway). Without a mapper,
        fall back to the legacy str(db_id) behavior.
        """
        if not db:
            return []

        try:
            entities = db.search_entities(query, limit=5)
        except Exception:
            return []

        doc_ids = []
        for entity in entities:
            entity_id = entity.get("id")
            if not entity_id:
                continue
            try:
                docs = db.get_entity_documents(entity_id)
                for doc in docs:
                    doc_id = doc.get("id")
                    if not doc_id:
                        continue
                    if callable(to_uuid):
                        doc_id = to_uuid(doc_id)
                        if not doc_id:
                            continue
                    else:
                        doc_id = str(doc_id)
                    if doc_id not in doc_ids:
                        doc_ids.append(doc_id)
            except Exception:
                continue

        return doc_ids

    def _build_answer_prompt(self, query: str, context: str) -> str:
        """Build the final answer generation prompt."""
        return (
            "Answer the user's question based on the following context.\n"
            "If the answer is not in the context, say "
            '"I cannot find the answer in the provided context."\n\n'
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
