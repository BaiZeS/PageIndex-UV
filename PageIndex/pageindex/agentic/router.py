import asyncio
import json
import logging
from typing import List, Tuple, Dict

from .planner import RetrievalPlanner
from .strategies import MetadataStrategy, SemanticsStrategy, DescriptionStrategy
from .verifier import CRAGVerifier


class AgenticRouter:
    """Orchestrate Plan -> Route -> Act -> Verify."""

    def __init__(self, client, model: str):
        self.client = client
        self.model = model
        self.planner = RetrievalPlanner(model)
        self.metadata_strategy = MetadataStrategy()
        self.semantics_strategy = None
        self.description_strategy = DescriptionStrategy(model)
        self.verifier = CRAGVerifier(model)
        self._main_funcs = None

        if hasattr(client, "closet_index") and client.closet_index:
            self.semantics_strategy = SemanticsStrategy(client.closet_index)

    # ------------------------------------------------------------------
    # Lazy import of main.py helpers (avoid circular deps at import time)
    # ------------------------------------------------------------------
    def _load_main_funcs(self):
        if self._main_funcs is None:
            try:
                import main
                self._main_funcs = {
                    "get_relevant_nodes": main.get_relevant_nodes,
                    "build_context_with_budget": main.build_context_with_budget,
                    "generate_answer": main.generate_answer,
                    "pages_from_nodes": main.pages_from_nodes,
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
    # Act — tree search + context assembly
    # ------------------------------------------------------------------
    async def _act_tree_search(
        self, query: str, candidate_docs: List[str]
    ) -> Tuple[str, List[dict], int, int, Dict[str, List[int]]]:
        funcs = self._load_main_funcs()
        get_relevant_nodes = funcs.get("get_relevant_nodes")
        pages_from_nodes = funcs.get("pages_from_nodes")
        if not all([get_relevant_nodes, pages_from_nodes]):
            raise RuntimeError("main.py helpers not available")

        contexts = []
        all_nodes = []
        source_docs = 0
        doc_pages_map: Dict[str, List[int]] = {}

        for doc_id in candidate_docs:
            if hasattr(self.client, "_ensure_doc_loaded"):
                self.client._ensure_doc_loaded(doc_id)
            doc = self.client.documents.get(doc_id)
            if not doc:
                continue
            structure = doc.get("structure", [])
            if not structure:
                continue

            tree_json = json.dumps(structure, ensure_ascii=False)
            node_ids = get_relevant_nodes(query, tree_json)
            if not node_ids:
                continue

            # Resolve nodes from in-memory structure
            from ..utils import create_node_mapping

            mapping = create_node_mapping(structure)
            selected = [
                mapping.get(nid) for nid in node_ids if nid in mapping
            ]
            selected = [n for n in selected if n]
            if not selected:
                continue

            pages = pages_from_nodes(selected)
            if not pages:
                continue

            # Build context from cached pages (PDF) or structure text (MD)
            ctx_parts = [f"\n=== Document: {doc.get('doc_name', '')} ===\n"]
            if doc.get("type") == "pdf" and doc.get("pages"):
                page_map = {p["page"]: p["content"] for p in doc["pages"]}
                for p in sorted(set(pages)):
                    text = page_map.get(p, "")
                    if text:
                        ctx_parts.append(f"\n--- Page {p} ---\n{text}")
            elif doc.get("type") == "md" and structure:
                # For MD, pull text from matching nodes
                for node in selected:
                    txt = node.get("text", "")
                    if txt:
                        ctx_parts.append(f"\n--- {node.get('title', '')} ---\n{txt}")

            if len(ctx_parts) > 1:
                contexts.append("".join(ctx_parts))
                all_nodes.extend(selected)
                source_docs += 1
                doc_pages_map[doc_id] = sorted(set(pages))

        return "\n\n".join(contexts), all_nodes, source_docs, len(all_nodes), doc_pages_map

    # ------------------------------------------------------------------
    # Public search
    # ------------------------------------------------------------------
    async def search(self, query: str, top_k: int = 3) -> Dict:
        # Plan
        plan = await self.planner.plan(query)

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
            ctx, nodes, src_docs, cov_nodes, doc_pages_map = await self._act_tree_search(
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
                    {"node_id": n.get("node_id"), "title": n.get("title")}
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
                ctx2, nodes2, src2, cov2, doc_pages_map2 = await self._act_tree_search(
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
                            {"node_id": n.get("node_id"), "title": n.get("title")}
                            for n in nodes2
                        ],
                        "pages": [
                            {"doc_id": d, "pages": p}
                            for d, p in doc_pages_map2.items()
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
                {"node_id": n.get("node_id"), "title": n.get("title")}
                for n in nodes
            ],
            "pages": [
                {"doc_id": d, "pages": p}
                for d, p in doc_pages_map.items()
            ],
        }
