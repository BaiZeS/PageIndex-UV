"""T6.3/T11 Agentic 多轮并发召回循环（spec [3.5]/[7.5]/[7.6]/[S8]）。

一次性固定 top_k 的召回在多相关数据集上被上限压死，且选错一批只能认命。
改用 agentic 多轮召回：每轮召回 → verifier 判停（复用 CRAG action）→
expand 时按 verifier 点名（VerifyResult.need）补召回，直到能作答或达到上限
（轮数/延迟/token 预算）。

设计要点（[S8] 扩召收敛后）：
- ① 轮 1 用调用方 top_k（轻量高精）；轮 ≥2 不再走融合序滑窗——改为只补
  verifier 点名（VerifyResult.need）的 doc_id 对象。删除 `_cut_candidates` 融合序
  滑窗与顺序回捞（回归 spec v3.2 [7.5]b "重新挑选必须 LLM 决策"原旨）；
- ② 轮内并发（router._act_tree_search 内部 asyncio.gather），轮间串行；RRF 只用作
  进入序（entry ordering），不引入新判据；
- ③ LLM 判停复用 CRAGVerifier 的 answer/expand/refuse action，不自造判据；
- ④ 增量去重：凡进入过某轮候选的文档都进 retrieved，后续轮不再召回；
- 上下文替换语义（[7.5]b）：每轮上下文在 max_context_tokens 预算内整体重建，
  绝不追加上一轮上下文；累积文档集合仅供 best_effort 使用；
- ⑤ 首轮即答快速路径：轮 1 的 answer 判定直接返回，不做任何预算/延迟记账；
- ⑥ 延迟/预算三道闸（[7.5]c）：开新轮前 elapsed + 预估 vs agentic_max_latency_ms；
  单轮 asyncio.wait_for(agentic_round_timeout_s)——超时轮降级为 best-effort
  （选择直接终止而非开下一轮：超时说明负载异常，重试大概率重蹈覆辙）；
  跨轮 token 总账（各轮上下文 + 答案 token 之和）超 agentic_max_total_tokens
  即提前终止进 best-effort；
- ⑦ need 为空/无有效 doc_id → stop_reason="no_target" 进 best_effort（[S8]）。
"""

import asyncio
import logging
import math
import time
from typing import Dict, List, Optional, Tuple


def _node_payload(nodes: List[dict]) -> List[dict]:
    """节点序列化——与 AgenticRouter 搜索响应的 selected_nodes 形状一致。

    [S10] span 表达：page 节点输出页区间（`pages`，UI 溯源沿用）；line 节点
    输出行区间（`start_line`/`end_line`），不再臆造页码。
    """
    out = []
    for n in nodes or []:
        kind = n.get("span_kind") or ("page" if n.get("start_index") is not None else "line")
        item = {
            "node_id": n.get("node_id"),
            "title": n.get("title"),
            "summary": n.get("summary", ""),
            "text": n.get("text", ""),
        }
        if kind == "page":
            item["pages"] = list(range(n.get("start_index") or 0, (n.get("end_index") or 0) + 1)) if n.get("start_index") else []
        else:
            item["start_line"] = n.get("line_num")
            item["end_line"] = n.get("end_line")
        out.append(item)
    return out


class AgenticRecallLoop:
    """多轮召回循环。复用 AgenticRouter 机制做轮内并发召回与接地。

    每轮产出形状：{"ctx", "nodes", "src_docs", "cov_nodes",
    "doc_pages_map", "pages_with_text"}（来自 router._act_tree_search）。
    返回形状与 router 搜索结果一致（query/mode/answer/confidence/matched_docs/
    selected_nodes/pages），仅附加 rounds_used / note 元数据（additive）。
    """

    DEFAULT_MAX_ROUNDS = 3
    DEFAULT_MAX_LATENCY_MS = 45000.0
    DEFAULT_ROUND_TIMEOUT_S = 30.0
    DEFAULT_MAX_TOTAL_TOKENS = 120000

    def __init__(self, router):
        self.router = router
        self._node_matches: Dict[str, List[dict]] = {}
        self._load_settings()

    def _load_settings(self):
        """从 config.yaml 读取 guard 参数；缺省/非法值逐字段回退类默认值。测试可直接覆写实例属性。"""
        cfg = None
        try:
            from ..utils import ConfigLoader
            cfg = ConfigLoader().load(None)
        except Exception as e:
            logging.warning("[AgenticLoop] config load failed, using defaults: %s", e)

        def _field(name, default, cast):
            raw = getattr(cfg, name, None)
            if raw is None:
                return default
            try:
                value = cast(raw)
            except (TypeError, ValueError):
                logging.warning(
                    "[AgenticLoop] invalid config %s=%r, fallback to default %r",
                    name, raw, default,
                )
                return default
            return value if value else default

        self.max_rounds = _field("agentic_max_rounds", self.DEFAULT_MAX_ROUNDS, int)
        self.max_latency_ms = _field("agentic_max_latency_ms", self.DEFAULT_MAX_LATENCY_MS, float)
        self.round_timeout_s = _field("agentic_round_timeout_s", self.DEFAULT_ROUND_TIMEOUT_S, float)
        self.max_total_tokens = _field("agentic_max_total_tokens", self.DEFAULT_MAX_TOTAL_TOKENS, int)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    async def retrieve(
        self,
        query: str,
        top_k: int = 3,
        first_round_fused: Optional[List[Tuple[str, float]]] = None,
        first_round_ctx_state: Optional[Dict] = None,
        first_round_node_matches: Optional[Dict[str, List[dict]]] = None,
        max_rounds: int = None,
        expand_need: Optional[List[dict]] = None,
    ) -> Dict:
        """多轮召回主循环。

        - first_round_fused: 调用方（_search_v2 expand 委派）已完成轮 1 时传入的
          RRF 融合序 [(doc_id, score)]；None 时循环自行跑 Plan→Route 打轮 1。
        - first_round_ctx_state: 调用方轮 1 的 Act 产出（同轮产出形状）；与
          first_round_fused 同时给出 ⇒ 轮 1 视为已完成，以其 fused[:top_k] 为
          排除种子，从轮 2 继续。
        - first_round_node_matches: 调用方 Route 阶段的内容策略节点命中信息
          （_run_strategies 的 node_matches）；给出时承接为本循环的节点匹配，
          供续接轮（≥2）的树搜索复用——否则续接轮以空 node_matches 召回，
          节点召回弱化。默认 None ⇒ 维持独立调用方行为（自行 Route 或空）。
        - expand_need: [S8] 轮 1 由调用方完成时的 verifier 点名（VerifyResult.need）；
          轮 ≥2 只补其中有效且未召回的 doc_id 对象。默认 None（独立模式：由本循环
          轮 1 verify 后置入）。
        """
        total_rounds = max(1, int(max_rounds or self.max_rounds))
        self._node_matches = first_round_node_matches if first_round_node_matches is not None else {}

        if first_round_fused is not None:
            fused = self._normalize_fused(first_round_fused)
            start_round = 2 if first_round_ctx_state is not None else 1
        else:
            fused = await self._route(query)
            start_round = 1
        score_map = {doc_id: score for doc_id, score in fused}

        # 轮 1 已完成时的排除种子：v2 轮 1 候选正是 fused[:top_k]
        seed: List[str] = []
        if first_round_ctx_state is not None:
            seed = [doc_id for doc_id, _ in fused[: max(0, int(top_k))]]

        retrieved = set(seed)            # ④ 增量去重：已召回（含失败轮候选）
        accumulated: List[str] = list(seed)   # best_effort 输入：成功接地的候选（轮序）
        acc_set = set(seed)
        last_state = first_round_ctx_state
        last_state_docs = set(seed) if first_round_ctx_state is not None else set()

        round_durations: List[float] = []
        tokens_used = 0
        last_completed = start_round - 1
        stop_reason = "rounds_exhausted"
        start_ts = time.monotonic()

        # [S8] 点名扩召：expand 轮候选 = 上一轮 verifier 的 need 点名对象。
        # 轮 1 由调用方完成时承接其 need（expand_need）；否则由本循环轮 1 verify 置入。
        pending_need = expand_need if first_round_ctx_state is not None else None

        for r in range(start_round, total_rounds + 1):
            # ⑤ 闸门只拦新开轮（轮 1 豁免——首轮即答快速路径）
            if r > 1:
                elapsed_ms = (time.monotonic() - start_ts) * 1000.0
                if elapsed_ms + self._estimate_round_ms(round_durations) >= self.max_latency_ms:
                    stop_reason = "latency_budget"
                    break
                if tokens_used > self.max_total_tokens:
                    stop_reason = "token_budget"
                    break

            if r == 1:
                candidates = [doc_id for doc_id, _ in fused[: max(0, int(top_k))]]
            else:
                candidates = self._named_candidates(pending_need, retrieved)

            # 无候选可召 → 终止进 best_effort。两种空窗：轮 1 融合池为空；
            # 轮 ≥2 无点名对象（need 空/无有效 doc_id/全已召回，[S8]）。
            if not candidates:
                stop_reason = "no_target"
                break

            round_start = time.monotonic()
            try:
                outcome = await asyncio.wait_for(
                    self._run_round(query, candidates, self._node_matches),
                    timeout=self.round_timeout_s,
                )
            except asyncio.TimeoutError:
                retrieved.update(candidates)
                logging.warning("[AgenticLoop] round %d timed out after %.1fs", r, self.round_timeout_s)
                stop_reason = "round_timeout"
                break
            except Exception as e:
                retrieved.update(candidates)
                logging.warning("[AgenticLoop] round %d failed: %s", r, e)
                stop_reason = "round_error"
                break

            round_durations.append((time.monotonic() - round_start) * 1000.0)
            retrieved.update(candidates)
            for doc_id in candidates:
                if doc_id not in acc_set:
                    acc_set.add(doc_id)
                    accumulated.append(doc_id)
            last_completed = r

            ctx = outcome.get("ctx") or ""
            tokens_used += self._count_tokens(ctx)
            if not ctx:
                # 本轮无接地上下文：不浪费答案/校验调用，直接进下一轮
                # （点名对象已进 retrieved，下一轮无新 need → no_target 终止）
                continue
            last_state = outcome
            last_state_docs = set(candidates)

            try:
                answer = await self._generate(query, ctx)
            except Exception as e:
                logging.warning("[AgenticLoop] generate failed: %s", e)
                stop_reason = "generator_error"
                break
            if answer is None:
                stop_reason = "no_answer_generator"
                break
            tokens_used += self._count_tokens(answer)

            try:
                v = await asyncio.to_thread(
                    self.router.verifier.verify, answer, ctx, query,
                    outcome.get("src_docs", 0), outcome.get("cov_nodes", 0),
                )
            except Exception as e:
                logging.warning("[AgenticLoop] verify failed: %s", e)
                stop_reason = "verifier_error"
                break
            if v.action == "answer":
                # ⑤ 轮 1 快速路径：直接返回，无预算/延迟记账开销
                return self._round_response(query, answer, "high", outcome, score_map, r)
            if v.action == "refuse":
                return self._refuse_response(query, outcome, score_map, r)
            # expand → 下一轮只补本次 need 点名对象（[S8]）
            pending_need = getattr(v, "need", None) or []

        logging.info(
            "[AgenticLoop] falling back to best-effort: reason=%s rounds=%d accumulated=%d",
            stop_reason, last_completed, len(accumulated),
        )
        return await self._best_effort(
            query, score_map, accumulated, last_state, last_state_docs,
            rounds_used=max(0, last_completed),
        )

    # ------------------------------------------------------------------
    # 轮机制
    # ------------------------------------------------------------------
    @staticmethod
    def _named_candidates(need, retrieved: set) -> List[str]:
        """[S8] 从 verifier need 点名清单提取 doc_id 子集（去重、保序、排除已召回）。

        node_id 条目无文档级对象，最小实现跳过；page 字段不参与文档级补召回。
        need 为空/非 list/无有效 doc_id → 空列表（调用方据此 no_target 终止）。
        """
        if not need:
            return []
        out: List[str] = []
        seen = set()
        for item in need:
            if not isinstance(item, dict):
                continue
            doc_id = item.get("doc_id")
            if not doc_id:
                continue
            # [Fix] verifier 的 _normalize_need 保留 doc_id 原样——LLM 可能回数字
            # （如 "doc_id": 5），若不强转字符串，int 永不命中 str 集合 `retrieved`
            # → 已召回文档被重复拉取，且 int 流入 _act_tree_search（List[str]）。
            doc_id = str(doc_id)
            if not doc_id:
                continue
            if doc_id in retrieved or doc_id in seen:
                continue
            seen.add(doc_id)
            out.append(doc_id)
        return out

    async def _run_round(self, query: str, candidates: List[str], node_matches=None) -> Dict:
        """轮内 Act：树搜索 + 预算内上下文构建（替换语义——每轮整体重建）。"""
        ctx, nodes, src_docs, cov_nodes, doc_pages_map, pages_with_text = (
            await self.router._act_tree_search(query, list(candidates), node_matches=node_matches)
        )
        return {
            "ctx": ctx,
            "nodes": nodes,
            "src_docs": src_docs,
            "cov_nodes": cov_nodes,
            "doc_pages_map": doc_pages_map,
            "pages_with_text": pages_with_text,
        }

    async def _route(self, query: str) -> List[Tuple[str, float]]:
        """独立模式轮 1 的 Route：Plan → 轮内并发召回 → RRF（仅进入序）。

        融合池只喂轮 1 的 fused[:top_k]；后续轮候选来自 verifier 点名（need），
        不再对融合池切窗（[S8] 删除滑窗）。
        """
        router = self.router
        try:
            docs_info = router._build_docs_info()
        except Exception as e:
            logging.warning("[AgenticLoop] docs info failed: %s", e)
            docs_info = []
        if not docs_info:
            return []
        try:
            plan = await router.planner.plan(query)
            weights = dict(getattr(plan, "weights", None) or {})
            queries = getattr(plan, "queries", None) or []
            route_query = queries[0] if queries else query
        except Exception as e:
            logging.warning("[AgenticLoop] planner failed, default weights: %s", e)
            weights = {"metadata": 0.15, "content": 0.35, "semantics": 0.3, "description": 0.2}
            route_query = query
        try:
            results, node_matches = await router._run_strategies(route_query, docs_info, weights)
        except Exception as e:
            logging.warning("[AgenticLoop] strategies failed: %s", e)
            return []
        self._node_matches = node_matches or {}
        return router._weighted_rrf(results, weights)

    # ------------------------------------------------------------------
    # best-effort（[7.6]）
    # ------------------------------------------------------------------
    async def _best_effort(
        self,
        query: str,
        score_map: Dict[str, float],
        accumulated: List[str],
        last_state: Optional[Dict],
        last_state_docs: set,
        rounds_used: int,
    ) -> Dict:
        """绝不全量喂入：先对累积池做一次接地再挑选（预算内 top-N）；
        confidence=low + 显式尽力作答标注；无任何证据命中 → 诚实拒答，不编造。"""
        if not accumulated:
            return {
                "query": query,
                "mode": "multi",
                "answer": "未在语料中找到相关证据，无法作答。",
                "confidence": "low",
                "matched_docs": [],
                "selected_nodes": [],
                "pages": [],
                "note": "诚实拒答：无任何证据命中。",
                "rounds_used": rounds_used,
            }

        # 快捷路径：最近一次接地状态已覆盖全部累积文档 → 不重复跑 Act
        outcome = None
        if last_state is not None and last_state.get("ctx") and set(accumulated) <= last_state_docs:
            outcome = last_state
        else:
            try:
                outcome = await asyncio.wait_for(
                    self._run_round(query, accumulated, self._node_matches),
                    timeout=self.round_timeout_s,
                )
            except Exception as e:
                logging.warning("[AgenticLoop] best-effort grounding failed: %s", e)
            if outcome is None or not outcome.get("ctx"):
                # 接地失败 → 退回最近有效状态；没有则诚实拒答
                if last_state is not None and last_state.get("ctx"):
                    outcome = last_state
                else:
                    return {
                        "query": query,
                        "mode": "multi",
                        "answer": "未能从语料中接地到有效证据，无法作答。",
                        "confidence": "low",
                        "matched_docs": self._matched(accumulated, score_map),
                        "selected_nodes": [],
                        "pages": [],
                        "note": "诚实拒答：召回了候选文档但接地失败。",
                        "rounds_used": rounds_used,
                    }

        try:
            answer = await self._generate(query, outcome["ctx"])
        except Exception as e:
            logging.warning("[AgenticLoop] best-effort generate failed: %s", e)
            answer = None
        if answer is None:
            answer = ""
        grounded_docs = list(outcome.get("doc_pages_map", {}).keys()) or list(accumulated)
        return {
            "query": query,
            "mode": "multi",
            "answer": answer,
            "confidence": "low",
            "matched_docs": self._matched(accumulated, score_map),
            "selected_nodes": _node_payload(outcome.get("nodes")),
            "pages": outcome.get("pages_with_text", []),
            "note": (
                "尽力作答：证据可能不充分，结论仅供参考。"
                f"引用来源：{', '.join(str(d) for d in grounded_docs)}"
            ),
            "rounds_used": rounds_used,
        }

    # ------------------------------------------------------------------
    # 响应构建
    # ------------------------------------------------------------------
    def _round_response(
        self, query: str, answer: str, confidence: str,
        outcome: Dict, score_map: Dict[str, float], rounds_used: int,
    ) -> Dict:
        docs = list(outcome.get("doc_pages_map", {}).keys())
        return {
            "query": query,
            "mode": "multi",
            "answer": answer,
            "confidence": confidence,
            "matched_docs": self._matched(docs, score_map),
            "selected_nodes": _node_payload(outcome.get("nodes")),
            "pages": outcome.get("pages_with_text", []),
            "rounds_used": rounds_used,
        }

    def _refuse_response(self, query: str, outcome: Dict, score_map: Dict[str, float], rounds_used: int) -> Dict:
        return self._round_response(query, "I don't know.", "low", outcome, score_map, rounds_used)

    @staticmethod
    def _matched(doc_ids: List[str], score_map: Dict[str, float]) -> List[Dict]:
        return [
            {"doc_id": d, "score": round(float(score_map.get(d, 0.0)), 4)}
            for d in sorted(doc_ids, key=lambda d: (-float(score_map.get(d, 0.0)), str(d)))
        ]

    # ------------------------------------------------------------------
    # 小工具
    # ------------------------------------------------------------------
    async def _generate(self, query: str, ctx: str):
        funcs = self.router._load_main_funcs()
        generate_answer = funcs.get("generate_answer") if isinstance(funcs, dict) else None
        if not generate_answer:
            return None
        return generate_answer(query, ctx)

    def _estimate_round_ms(self, durations: List[float]) -> float:
        """新轮耗时预估：有观测取均值；无观测按单轮超时的一半保守估计。"""
        if durations:
            return sum(durations) / len(durations)
        return self.round_timeout_s * 1000.0 / 2.0

    @staticmethod
    def _normalize_fused(first_round_fused) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        for item in first_round_fused or []:
            try:
                doc_id, score = item
                score = float(score)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            out.append((doc_id, score))
        return out

    @staticmethod
    def _count_tokens(text: str) -> int:
        if not text:
            return 0
        try:
            from ..utils import count_tokens
            return int(count_tokens(text))
        except Exception:
            return len(text) // 4
