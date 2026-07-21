import asyncio
import json
import logging
from typing import List, Tuple, Dict

from .planner import RetrievalPlanner
from .strategies import MetadataStrategy, SemanticsStrategy, DescriptionStrategy
from .verifier import CRAGVerifier
from ..super_tree import SuperTreeIndex


class AgenticRouter:
    """Orchestrate Plan -> Route -> Act -> Verify."""

    def __init__(self, client, model: str, retrieve_model: str = None):
        self.client = client
        self.model = model
        self.retrieve_model = retrieve_model
        self.planner = RetrievalPlanner(model, retrieve_model)
        self.metadata_strategy = MetadataStrategy()
        self.semantics_strategy = None
        self.description_strategy = DescriptionStrategy(model, retrieve_model)
        self.verifier = CRAGVerifier(model, retrieve_model)
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
                    get_relevant_nodes,
                    build_context_with_budget,
                    generate_answer,
                    pages_from_nodes,
                    build_context_for_doc,
                )
                self._main_funcs = {
                    "get_relevant_nodes": get_relevant_nodes,
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
    ) -> Dict[str, List[Tuple[str, int]]]:
        tasks = {}
        tasks["metadata"] = asyncio.to_thread(
            self.metadata_strategy.search, query, docs_info
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
        if tasks:
            done = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for name, res in zip(tasks.keys(), done):
                if isinstance(res, Exception):
                    logging.warning("Strategy %s failed: %s", name, res)
                    results[name] = []
                else:
                    results[name] = res
        return results

    # ------------------------------------------------------------------
    # Act — tree search + context assembly (parallelized)
    # ------------------------------------------------------------------
    async def _recall_nodes_for_doc(self, query: str, doc_id: str):
        """Recall relevant nodes for a single document (runs in thread)."""
        funcs = self._load_main_funcs()
        get_relevant_nodes = funcs.get("get_relevant_nodes")
        pages_from_nodes = funcs.get("pages_from_nodes")

        if hasattr(self.client, "_ensure_doc_loaded"):
            self.client._ensure_doc_loaded(doc_id)
        doc = self.client.documents.get(doc_id)
        if not doc:
            return None
        structure = doc.get("structure", [])
        if not structure:
            return None

        tree_json = json.dumps(structure, ensure_ascii=False)
        node_ids = await asyncio.to_thread(get_relevant_nodes, query, tree_json)
        if not node_ids:
            return None

        from ..utils import create_node_mapping
        mapping = create_node_mapping(structure)
        selected = [mapping.get(nid) for nid in node_ids if nid in mapping]
        selected = [n for n in selected if n]
        if not selected:
            return None

        pages = pages_from_nodes(selected)
        if not pages:
            return None

        # Compute a simple relevance score based on number of matched nodes
        # (more matched nodes = higher relevance)
        relevance_score = len(selected) / max(len(structure), 1)

        return {
            "doc_id": doc_id,
            "doc": doc,
            "structure": structure,
            "selected": selected,
            "pages": pages,
            "relevance_score": relevance_score,
        }

    async def _act_tree_search(
        self, query: str, candidate_docs: List[str]
    ) -> Tuple[str, List[dict], int, int, Dict[str, List[int]], List[dict]]:
        funcs = self._load_main_funcs()
        pages_from_nodes = funcs.get("pages_from_nodes")
        if not pages_from_nodes:
            raise RuntimeError("main.py helpers not available")

        # Parallel node recall across documents
        recall_tasks = [
            self._recall_nodes_for_doc(query, doc_id)
            for doc_id in candidate_docs
        ]
        recall_results = await asyncio.gather(*recall_tasks, return_exceptions=True)

        # Filter out failures and sort by relevance score (descending)
        doc_results = []
        for r in recall_results:
            if isinstance(r, dict):
                doc_results.append(r)

        doc_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        contexts = []
        all_nodes = []
        source_docs = 0
        doc_pages_map: Dict[str, List[int]] = {}

        build_ctx = self._load_main_funcs().get("build_context_for_doc")

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
                contexts.append(context)
                all_nodes.extend(selected)
                source_docs += 1
                doc_pages_map[doc_id] = sorted(set(pages))

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
                    contexts.append("\n".join(entity_context_parts))
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
    # Super-Tree search
    # ------------------------------------------------------------------
    async def _search_super_tree(self, query: str, top_k: int = 3) -> Dict:
        """L0 prefilter → L1 Super-Tree selection → L2/L3 Act → Verify."""
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
            return {
                "query": query,
                "mode": "multi",
                "answer": "Super-Tree selection returned no documents.",
                "confidence": "low",
                "matched_docs": [],
                "selected_nodes": [],
                "pages": [],
            }

        # Build matched_docs with scores (all selected get score 1.0)
        matched = [{"doc_id": doc_id, "score": 1.0} for doc_id in selected_uuids]

        # L2/L3: Act — tree search on selected documents (reuse existing _act_tree_search)
        try:
            ctx, nodes, src_docs, cov_nodes, doc_pages_map, pages_with_text = await self._act_tree_search(
                query, selected_uuids
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
                "selected_nodes": [
                    {
                    "node_id": n.get("node_id"),
                    "title": n.get("title"),
                    "summary": n.get("summary", ""),
                    "text": n.get("text", ""),
                    "pages": list(range(n.get("start_index") or 0, (n.get("end_index") or 0) + 1)) if n.get("start_index") else [],
                }
                    for n in nodes
                ],
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
            "selected_nodes": [
                {
                    "node_id": n.get("node_id"),
                    "title": n.get("title"),
                    "summary": n.get("summary", ""),
                    "text": n.get("text", ""),
                    "pages": list(range(n.get("start_index") or 0, (n.get("end_index") or 0) + 1)) if n.get("start_index") else [],
                }
                for n in nodes
            ],
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
        results = await self._run_strategies(
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

        # Act
        try:
            ctx, nodes, src_docs, cov_nodes, doc_pages_map, pages_with_text = await self._act_tree_search(
                query, candidates
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
                "selected_nodes": [
                    {
                    "node_id": n.get("node_id"),
                    "title": n.get("title"),
                    "summary": n.get("summary", ""),
                    "text": n.get("text", ""),
                    "pages": list(range(n.get("start_index") or 0, (n.get("end_index") or 0) + 1)) if n.get("start_index") else [],
                }
                    for n in nodes
                ],
                "pages": [
                    {"doc_id": d, "pages": p}
                    for d, p in doc_pages_map.items()
                ],
            }

        # Expand on medium confidence
        if v.action == "expand" and len(fused) > top_k:
            expanded = [doc_id for doc_id, _ in fused[: top_k * 2]]
            try:
                ctx2, nodes2, src2, cov2, doc_pages_map2, pages_with_text2 = await self._act_tree_search(
                    query, expanded
                )
                if ctx2:
                    ans2 = generate_answer(query, ctx2)
                    v2 = await asyncio.to_thread(
                        self.verifier.verify, ans2, ctx2, query, src2, cov2
                    )
                    conf = "high" if v2.action == "answer" else "medium"
                    return {
                        "query": query,
                        "mode": "multi",
                        "answer": ans2,
                        "confidence": conf,
                        "matched_docs": [
                            {"doc_id": d, "score": round(s, 4)}
                            for d, s in fused[: top_k * 2]
                        ],
                        "selected_nodes": [
                            {
                    "node_id": n.get("node_id"),
                    "title": n.get("title"),
                    "summary": n.get("summary", ""),
                    "text": n.get("text", ""),
                    "pages": list(range(n.get("start_index") or 0, (n.get("end_index") or 0) + 1)) if n.get("start_index") else [],
                }
                            for n in nodes2
                        ],
                        "pages": [
                            {"doc_id": d, "pages": p}
                            for d, p in pages_with_text2.items()
                        ],
                    }
            except Exception as e:
                logging.warning("Expand search failed: %s", e)

        conf = "high" if v.action == "answer" else "medium"
        return {
            "query": query,
            "mode": "multi",
            "answer": answer,
            "confidence": conf,
            "matched_docs": matched,
            "selected_nodes": [
                {
                    "node_id": n.get("node_id"),
                    "title": n.get("title"),
                    "summary": n.get("summary", ""),
                    "text": n.get("text", ""),
                    "pages": list(range(n.get("start_index") or 0, (n.get("end_index") or 0) + 1)) if n.get("start_index") else [],
                }
                for n in nodes
            ],
            "pages": pages_with_text,
        }

    # ------------------------------------------------------------------
    # Public search
    # ------------------------------------------------------------------
    async def search(self, query: str, top_k: int = 3) -> Dict:
        """Try Super-Tree first, fallback to v2 on any failure."""
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
