"""语料树构建器（P1）——完全无向量管线。

实现设计文档 [S3]/[S3.1] 的语料统一结构树自动构建：

    无序文档集
      ① closet_tags（已有，ClosetIndex._extract_tags 抽取）
      ② 标签归一化：全库唯一标签 → LLM 合并同义 → 规范标签集
      ③ 规范标签 tag→docs 倒排 → 确定性 group-by 分组（不走 LLM）
      ④ LLM 递归生成上层结构（init/continue 模式）：定名+摘要+组织上下级
      ⑤ 组装层级 + 软归属（一篇文档可挂多簇，DAG，绝不硬删）
      ⑥ 增量：新文档标签先匹配已有规范集，不中再 LLM 单点裁定；
         簇卡界每次插入时评估（超上限→拆分）

细而不碎（[S3.1]）：簇大小双向卡界——过小→合并、过大→拆分；合并保护——
过相似的兄弟簇合并；均 LLM 语义裁定。合并裁定保守（LLM 不可用时不盲目合并：
自动造的树错了比没有树更糟）。全程无向量：分组是纯倒排 group-by，LLM 用于
标签归一、递归建树与合并/拆分裁定，每次调用输入有界。
"""
import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional

from .utils import llm_completion, extract_json

ROOT_TITLE = "知识库"
FALLBACK_TITLE = "未分类"
FALLBACK_WEIGHT = 0.3


class CorpusTreeBuilder:
    """索引期自动构建/增量维护语料级主题树（整库一棵树，无向量）。"""

    _CLUSTER_MIN = 10        # 过小→合并（防碎）；标定目标，实测可调
    _CLUSTER_MAX = 50        # 过大→拆分（防单簇塞爆 LLM 上下文）；标定目标
    _MAX_ROOT_FANOUT = 8     # ROOT 直接子节点超过该数时生成上层分组
    _GROUP_BATCH_SIZE = 8    # 上层结构 init/continue 每批簇数
    _MAX_UPPER_LEVELS = 3    # 上层层数安全上限（N 随语料量自适应，不写死）
    _MAX_BOUND_ROUNDS = 3    # 卡界处置迭代轮数上限

    def __init__(self, db, model: str, retrieve_model: str = None,
                 cluster_min: int = None, cluster_max: int = None):
        self.db = db
        self.model = model
        self.retrieve_model = retrieve_model
        self.cluster_min = cluster_min if cluster_min is not None else self._CLUSTER_MIN
        self.cluster_max = cluster_max if cluster_max is not None else self._CLUSTER_MAX

    @property
    def _llm_model(self):
        return self.retrieve_model or self.model

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def rebuild(self) -> dict:
        """全量构建：从全部已索引文档重建语料树。返回可检视树结构。"""
        docs = self.db.get_all_documents()
        doc_tags = {d["id"]: self.db.get_doc_tags(d["id"]) for d in docs}

        # ② 标签归一化（对"标签集合"的有界一次性 LLM 操作）
        all_tags = sorted({t["tag_text"] for tags in doc_tags.values() for t in tags})
        norm_map = self._normalize_tags(all_tags)
        self.db.set_corpus_tag_norm_map(norm_map)

        self.db.corpus_tree_clear()
        root_id = self.db.insert_corpus_tree_node(
            None, ROOT_TITLE, "语料库根节点", 0, kind="root")

        # ③ 规范标签 tag→docs 倒排 + 确定性 group-by（纯数据结构，不走 LLM）
        clusters: Dict[str, Dict[int, float]] = {}
        for doc_id, tags in doc_tags.items():
            for t in tags:
                canonical = norm_map.get(t["tag_text"], t["tag_text"])
                weights = clusters.setdefault(canonical, {})
                weights[doc_id] = max(weights.get(doc_id, 0.0), float(t["confidence"]))

        for canonical in sorted(clusters):
            node_id = self.db.insert_corpus_tree_node(
                root_id, canonical, "", 1, kind="cluster", tag=canonical)
            for doc_id, weight in sorted(clusters[canonical].items()):
                self.db.add_corpus_membership(doc_id, node_id, weight)

        # 无标签文档兜底簇，保证 100% 覆盖
        tagless = sorted(doc_id for doc_id, tags in doc_tags.items() if not tags)
        if tagless:
            node_id = self.db.insert_corpus_tree_node(
                root_id, FALLBACK_TITLE, "未提取到语义标签的文档", 1, kind="cluster")
            for doc_id in tagless:
                self.db.add_corpus_membership(doc_id, node_id, FALLBACK_WEIGHT)

        # [S3.1] 细而不碎：簇大小双向卡界（过大拆分、过小合并，留处置记录）
        self._enforce_size_bounds()
        # ④ 上层结构递归生成（簇数过多时才分批 init/continue）
        self._build_upper_structure(root_id)
        # [S3.1] 合并保护：过相似的兄弟簇合并（宁缺毋滥，LLM 失败不合并）
        self._merge_similar_siblings()
        return self.get_tree()

    def update_for_document(self, doc_id) -> None:
        """⑥ 增量更新：新文档 closet 标签落库后调用，就近挂簇。

        - 标签归一（增量版）：先匹配已有规范标签集；不中再由 LLM 单点裁定
          "并入已有规范标签 or 新开"——不重跑全库归一。
        - 簇卡界每次插入时评估：挂簇后超上限→立即拆分；低于下限可延迟
          到下次结构调整（rebuild 合并 pass 处理）。
        """
        root_id = self._ensure_root()
        rows = self.db.get_doc_tags(doc_id)
        if not rows:
            fallback_id = self._ensure_fallback_cluster(root_id)
            self.db.add_corpus_membership(doc_id, fallback_id, FALLBACK_WEIGHT)
            self._split_if_oversized(fallback_id)
            return

        norm_map = self.db.get_corpus_tag_norm_map()
        canonical_weights: Dict[str, float] = {}
        for r in rows:
            raw = r["tag_text"]
            conf = float(r["confidence"])
            if raw in norm_map:
                canonical = norm_map[raw]
            else:
                canonical = self._resolve_new_tag(raw)
                self.db.upsert_corpus_tag_norm(raw, canonical)
                norm_map[raw] = canonical
            canonical_weights[canonical] = max(canonical_weights.get(canonical, 0.0), conf)

        for canonical, weight in sorted(canonical_weights.items()):
            node_id = self._find_cluster_for_tag(root_id, canonical)
            self.db.add_corpus_membership(doc_id, node_id, weight)
            self._split_if_oversized(node_id)

    def get_tree(self) -> dict:
        """返回可检视语料树（嵌套 dict）；未建树返回 {}。"""
        nodes = self.db.get_corpus_tree_nodes()
        if not nodes:
            return {}
        entries = {}
        for n in nodes:
            entries[n["id"]] = {
                "node_id": n["id"], "parent_id": n["parent_id"],
                "title": n["title"], "summary": n["summary"] or "",
                "level": n["level"], "kind": n["kind"], "tag": n["tag"],
                "children": [], "docs": [],
            }
        doc_names = {d["id"]: d.get("pdf_name", "") for d in self.db.get_all_documents()}
        for doc_id, node_id, weight in self.db.get_all_corpus_memberships():
            entries[node_id]["docs"].append(
                {"doc_id": doc_id, "doc_name": doc_names.get(doc_id, ""), "weight": weight})
        root = None
        for n in nodes:
            entry = entries[n["id"]]
            entry["docs"].sort(key=lambda d: d["doc_id"])
            if n["parent_id"] is None:
                root = entry
            elif n["parent_id"] in entries:
                entries[n["parent_id"]]["children"].append(entry)
        for entry in entries.values():
            entry["children"].sort(key=lambda c: c["title"])
        return root or {}

    # ------------------------------------------------------------------
    # ② 标签归一化
    # ------------------------------------------------------------------

    def _normalize_tags(self, tags: List[str]) -> Dict[str, str]:
        """全库标签归一：LLM 合并同义 → raw_tag→canonical 映射。

        LLM 漏掉的标签保持自身为规范标签；LLM 失败退化为恒等映射。
        """
        mapping = {t: t for t in tags}
        if len(tags) <= 1:
            return mapping
        prompt = f"""你是一个语义标签归一化专家。以下是从语料库全部文档中抽取的语义标签集合，其中可能存在同义或近义标签（例如"风控"与"风险管理"同义）。
请将同义/近义标签合并，输出规范标签集。

标签全集：
{json.dumps(tags, ensure_ascii=False)}

要求：
1. 含义相同或高度相近的标签必须合并为同一个规范标签
2. 规范标签名选用其中最清晰、最通用的表述
3. 含义不同的标签保持各自独立，自成规范标签
4. 每个输入标签都必须归入一个规范标签，不得遗漏

返回JSON格式：{{"groups": [{{"canonical": "规范标签", "synonyms": ["原标签1", "原标签2"]}}, ...]}}
直接返回最终JSON结构，不要输出其他内容。"""
        data = self._llm_json(prompt)
        if not isinstance(data, dict):
            return mapping
        for group in data.get("groups", []):
            if not isinstance(group, dict):
                continue
            canonical = str(group.get("canonical", "")).strip()
            if not canonical:
                continue
            for syn in group.get("synonyms", []):
                syn = str(syn).strip()
                if syn in mapping:
                    mapping[syn] = canonical
        return mapping

    def _resolve_new_tag(self, raw_tag: str) -> str:
        """增量标签归一：单点 LLM 裁定"并入已有规范标签 or 新开"。"""
        canonical_tags = self.db.get_corpus_canonical_tags()
        if not canonical_tags:
            return raw_tag
        prompt = f"""你是一个语义标签归一化专家（标签归一化·增量单标签裁定）。已有规范标签集：
{json.dumps(canonical_tags, ensure_ascii=False)}

新文档产生了一个新标签："{raw_tag}"
请裁定：该标签应并入哪个已有规范标签（同义/近义），还是作为新的规范标签？

要求：
1. 与已有规范标签同义或高度近义时，并入该规范标签
2. 否则作为新的规范标签（可直接使用原标签名）

返回JSON格式：{{"canonical": "规范标签名"}}
直接返回最终JSON结构，不要输出其他内容。"""
        data = self._llm_json(prompt)
        if isinstance(data, dict):
            canonical = str(data.get("canonical", "")).strip()
            # 只接受已有规范标签（并入）；其他名字视为幻觉，退回原标签新开
            if canonical and canonical in canonical_tags:
                return canonical
        return raw_tag

    # ------------------------------------------------------------------
    # [S3.1] 簇大小双向卡界：过大拆分、过小合并
    # ------------------------------------------------------------------

    def _enforce_size_bounds(self) -> None:
        for _ in range(self._MAX_BOUND_ROUNDS):
            changed = False
            counts = self._node_doc_counts()
            for node in self.db.get_corpus_tree_nodes():
                size = counts.get(node["id"], 0)
                if size > self.cluster_max:
                    self._split_node(node, self.db.get_corpus_node_docs(node["id"]))
                    changed = True
            counts = self._node_doc_counts()
            nodes = self.db.get_corpus_tree_nodes()
            for node in nodes:
                size = counts.get(node["id"], 0)
                if 0 < size < self.cluster_min:
                    siblings = [
                        n for n in nodes
                        if n["parent_id"] == node["parent_id"]
                        and n["id"] != node["id"]
                        and counts.get(n["id"], 0) > 0
                    ]
                    if self._merge_small_cluster(node, siblings):
                        changed = True
            if not changed:
                break

    def _split_if_oversized(self, node_id: int) -> None:
        node = next((n for n in self.db.get_corpus_tree_nodes() if n["id"] == node_id), None)
        if node is None:
            return
        docs = self.db.get_corpus_node_docs(node_id)
        if len(docs) > self.cluster_max:
            self._split_node(node, docs)

    def _split_node(self, node: dict, docs: List[tuple]) -> None:
        """过大簇拆分：LLM 语义裁定；非法/失败时确定性均分兜底（不丢文档）。"""
        doc_ids = [did for did, _ in docs]
        weight_of = dict(docs)
        parts = self._llm_split_plan(node, doc_ids)
        if parts is None:
            parts = [
                {"title": f"{node['title']}·分组{i + 1}", "summary": node["summary"] or "",
                 "doc_ids": chunk}
                for i, chunk in enumerate(self._chunk_doc_ids(doc_ids))
            ]
        for part in parts:
            child_id = self.db.insert_corpus_tree_node(
                node["id"], part["title"], part["summary"],
                node["level"] + 1, kind="cluster", tag=node["tag"])
            for did in part["doc_ids"]:
                self.db.add_corpus_membership(did, child_id, weight_of.get(did, 1.0))
        self.db.delete_corpus_memberships_for_node(node["id"])
        self.db.insert_corpus_tree_event(
            node["id"], "split",
            json.dumps({"cluster": node["title"], "size": len(doc_ids),
                        "into": [p["title"] for p in parts]}, ensure_ascii=False))

    def _llm_split_plan(self, node: dict, doc_ids: List[int]) -> Optional[List[dict]]:
        docs_info = []
        for did in doc_ids:
            d = self.db.get_document_by_id(did) or {}
            docs_info.append({
                "id": did,
                "name": d.get("pdf_name", ""),
                "description": (d.get("doc_description") or "")[:100],
            })
        prompt = f"""你是一个主题聚类专家。知识库簇"{node['title']}"现有 {len(doc_ids)} 篇文档，超过上限 {self.cluster_max}，需要簇拆分。请将其拆分为更细的子簇。

文档列表（id、名称、描述）：
{json.dumps(docs_info, ensure_ascii=False)}

要求：
1. 每篇文档必须且只能归入一个子簇，不得遗漏
2. 按主题相似度拆分，每个子簇不超过 {self.cluster_max} 篇
3. 为每个子簇命名并给出一句话摘要

返回JSON格式：{{"clusters": [{{"title": "子簇名", "summary": "摘要", "doc_ids": [1, 2]}}, ...]}}
直接返回最终JSON结构，不要输出其他内容。"""
        data = self._llm_json(prompt)
        if not isinstance(data, dict):
            return None
        return self._validate_partition(data.get("clusters"), doc_ids)

    def _validate_partition(self, parts, all_doc_ids: List[int]) -> Optional[List[dict]]:
        """校验 LLM 拆分结果：全覆盖、无重复；仍超限的组进一步均分。"""
        if not isinstance(parts, list) or not parts:
            return None
        all_ids = set(all_doc_ids)
        seen = set()
        cleaned = []
        for part in parts:
            if not isinstance(part, dict):
                return None
            ids = []
            for x in part.get("doc_ids", []):
                try:
                    did = int(x)
                except (TypeError, ValueError):
                    return None
                if did not in all_ids or did in seen:
                    return None
                seen.add(did)
                ids.append(did)
            if not ids:
                continue
            cleaned.append({
                "title": str(part.get("title", "")).strip() or "子簇",
                "summary": str(part.get("summary", "")).strip(),
                "doc_ids": ids,
            })
        if seen != all_ids or not cleaned:
            return None
        bounded = []
        for part in cleaned:
            if len(part["doc_ids"]) <= self.cluster_max:
                bounded.append(part)
                continue
            for i, chunk in enumerate(self._chunk_doc_ids(part["doc_ids"])):
                bounded.append({
                    "title": part["title"] if i == 0 else f"{part['title']}·分组{i + 1}",
                    "summary": part["summary"],
                    "doc_ids": chunk,
                })
        return bounded

    def _chunk_doc_ids(self, doc_ids: List[int]) -> List[List[int]]:
        """确定性均分：切成每块 ≤ cluster_max。"""
        ids = sorted(doc_ids)
        n_parts = (len(ids) + self.cluster_max - 1) // self.cluster_max
        size = (len(ids) + n_parts - 1) // n_parts
        return [ids[i:i + size] for i in range(0, len(ids), size)]

    def _merge_small_cluster(self, node: dict, siblings: List[dict]) -> bool:
        """过小簇合并：LLM 语义裁定并入哪个兄弟簇。

        LLM 不可用或裁定无合适目标时不盲目合并（保留并记录）——
        自动造的树错了比没有树更糟。
        """
        docs = self.db.get_corpus_node_docs(node["id"])
        if not docs or len(docs) >= self.cluster_min:
            return False
        live_ids = {n["id"] for n in self.db.get_corpus_tree_nodes()}
        candidates = [s for s in siblings if s["id"] in live_ids]
        if not candidates:
            self.db.insert_corpus_tree_event(
                node["id"], "merge_skipped",
                json.dumps({"cluster": node["title"], "reason": "no_candidates"},
                           ensure_ascii=False))
            return False
        decided, target, reason = self._llm_merge_target(node, docs, candidates)
        if not decided:
            self.db.insert_corpus_tree_event(
                node["id"], "merge_skipped",
                json.dumps({"cluster": node["title"], "reason": reason},
                           ensure_ascii=False))
            return False
        if target is None:
            self.db.insert_corpus_tree_event(
                node["id"], "merge_skipped",
                json.dumps({"cluster": node["title"], "reason": "no_suitable_target"},
                           ensure_ascii=False))
            return False
        target_weights = dict(self.db.get_corpus_node_docs(target["id"]))
        for did, w in docs:
            self.db.add_corpus_membership(did, target["id"], max(w, target_weights.get(did, 0.0)))
        self.db.delete_corpus_memberships_for_node(node["id"])
        self.db.delete_corpus_tree_node(node["id"])
        self._remap_victim_tag(node, target)
        self.db.insert_corpus_tree_event(
            target["id"], "merge",
            json.dumps({"from": node["title"], "into": target["title"], "docs": len(docs)},
                       ensure_ascii=False))
        return True

    def _llm_merge_target(self, node: dict, docs: List[tuple], candidates: List[dict]):
        """返回 (decided, target, reason)。decided=False 时 reason 为处置原因：
        llm_unavailable（无响应/调用失败）或 invalid_response（响应格式非法）。"""
        counts = self._node_doc_counts()
        cand_info = [
            {"title": c["title"], "summary": c["summary"] or "",
             "doc_count": counts.get(c["id"], 0)}
            for c in candidates
        ]
        prompt = f"""你是一个主题聚类专家。知识库簇"{node['title']}"仅有 {len(docs)} 篇文档，低于下限 {self.cluster_min}，需要簇合并裁定。
候选合并目标簇：
{json.dumps(cand_info, ensure_ascii=False)}

请裁定：该簇应并入哪个候选簇（语义最相近者）？若无合适目标返回 null。

返回JSON格式：{{"target": "目标簇标题 或 null"}}
直接返回最终JSON结构，不要输出其他内容。"""
        data = self._llm_json(prompt)
        if data is None:
            return False, None, "llm_unavailable"
        if not isinstance(data, dict) or "target" not in data:
            return False, None, "invalid_response"
        title = data.get("target")
        if title is None:
            return True, None, None
        title = str(title).strip()
        for c in candidates:
            if c["title"] == title:
                return True, c, None
        return True, None, None

    # ------------------------------------------------------------------
    # [S3.1] 合并保护：过相似的兄弟簇合并（宁缺毋滥，全程无向量）
    # ------------------------------------------------------------------

    def _merge_similar_siblings(self) -> None:
        """合并保护：对每个兄弟簇集合各做一次"过相似簇对"LLM 裁定并合并。

        兄弟集合 = ROOT 的直接簇孩子 + 每个分组（group）的簇孩子。
        仅处理规模达标（≥ cluster_min）的簇；LLM 失败/无把握时一律不合并
        （自动造的树错了比没有树更糟）。
        """
        nodes = self.db.get_corpus_tree_nodes()
        root_id = next((n["id"] for n in nodes if n["kind"] == "root"), None)
        if root_id is None:
            return
        parent_ids = [root_id] + [n["id"] for n in nodes if n["kind"] == "group"]
        for parent_id in parent_ids:
            siblings = [
                n for n in self.db.get_corpus_tree_nodes()
                if n["parent_id"] == parent_id and n["kind"] == "cluster"
            ]
            self._merge_similar_set(siblings)

    def _merge_similar_set(self, siblings: List[dict]) -> None:
        counts = self._node_doc_counts()
        adequate = [s for s in siblings if counts.get(s["id"], 0) >= self.cluster_min]
        if len(adequate) < 2:
            return
        pairs = self._llm_similar_pairs(adequate)
        if not pairs:
            return
        merged_away = set()
        for target, victim in pairs:
            if target["id"] in merged_away or victim["id"] in merged_away:
                continue
            self._apply_similar_merge(victim, target)
            merged_away.add(victim["id"])

    def _apply_similar_merge(self, victim: dict, target: dict) -> None:
        """把 victim 的软归属迁入 target，删除 victim，保留 target 标题/摘要。"""
        docs = self.db.get_corpus_node_docs(victim["id"])
        target_weights = dict(self.db.get_corpus_node_docs(target["id"]))
        for did, w in docs:
            self.db.add_corpus_membership(
                did, target["id"], max(w, target_weights.get(did, 0.0)))
        self.db.delete_corpus_memberships_for_node(victim["id"])
        self.db.delete_corpus_tree_node(victim["id"])
        self._remap_victim_tag(victim, target)
        self.db.insert_corpus_tree_event(
            target["id"], "merge_similar",
            json.dumps({"from": victim["title"], "into": target["title"],
                        "docs": len(docs)}, ensure_ascii=False))
        # 合并发生在卡界 pass 之后：合并结果可能超限，必须复检（超则拆分）
        self._split_if_oversized(target["id"])

    def _remap_victim_tag(self, victim: dict, target: dict) -> None:
        """victim 规范标签改道至幸存簇——否则增量文档会复活被合并的簇。"""
        if victim.get("tag") and target.get("tag"):
            self.db.remap_corpus_tag_norm(victim["tag"], target["tag"])

    def _llm_similar_pairs(self, siblings: List[dict]) -> Optional[List[tuple]]:
        """一次有界 LLM 调用找出过相似簇对；失败/非法返回空（不合并）。

        返回 (target, victim) 对列表：保留规模较大者为目标（同规模取 id 小者）。
        """
        prompt = f"""你是一个主题聚类专家（合并保护·过相似兄弟簇合并）。以下是同一父节点下的一组主题簇，其中可能存在语义高度重复（近乎同义）的簇对。请找出这些过相似的簇对，以便合并。

簇列表：
{json.dumps(self._nodes_info(siblings), ensure_ascii=False)}

要求：
1. 只配对语义高度重复（近乎同义、可视为同一主题）的簇，宁缺毋滥
2. 仅部分相关或语义不同的簇不要配对
3. 每个簇最多出现在一个配对中
4. 没有过相似簇对时返回空列表

返回JSON格式：{{"pairs": [["簇标题A", "簇标题B"], ...]}}
直接返回最终JSON结构，不要输出其他内容。"""
        data = self._llm_json(prompt)
        if not isinstance(data, dict):
            return []
        raw_pairs = data.get("pairs")
        if not isinstance(raw_pairs, list):
            return []
        by_title = {s["title"]: s for s in siblings}
        counts = self._node_doc_counts()
        pairs = []
        used = set()
        for p in raw_pairs:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                continue
            a, b = str(p[0]).strip(), str(p[1]).strip()
            if a == b or a not in by_title or b not in by_title:
                continue
            if a in used or b in used:
                continue
            na, nb = by_title[a], by_title[b]
            ca, cb = counts.get(na["id"], 0), counts.get(nb["id"], 0)
            if ca < cb or (ca == cb and na["id"] > nb["id"]):
                na, nb = nb, na
            used.add(a)
            used.add(b)
            pairs.append((na, nb))
        return pairs

    # ------------------------------------------------------------------
    # ④ 上层结构递归生成（init/continue 模式）
    # ------------------------------------------------------------------

    def _build_upper_structure(self, root_id: int) -> None:
        for _ in range(self._MAX_UPPER_LEVELS):
            nodes = self.db.get_corpus_tree_nodes()
            children = [n for n in nodes if n["parent_id"] == root_id]
            if len(children) <= self._MAX_ROOT_FANOUT:
                break
            groups = self._group_nodes(children)
            if not groups:
                break
            nodes_by_parent = defaultdict(list)
            for n in nodes:
                nodes_by_parent[n["parent_id"]].append(n)
            for group in groups:
                group_id = self.db.insert_corpus_tree_node(
                    root_id, group["title"], group["summary"], 1, kind="group")
                for member in group["members"]:
                    self._relevel_subtree(nodes_by_parent, member["id"], 2)
                    self.db.update_corpus_tree_node(member["id"], parent_id=group_id)

    def _group_nodes(self, nodes: List[dict]) -> List[dict]:
        """分批 init/continue 生成上层分组；成员校验后返回。"""
        batches = [nodes[i:i + self._GROUP_BATCH_SIZE]
                   for i in range(0, len(nodes), self._GROUP_BATCH_SIZE)]
        structure = self._llm_group_init(batches[0])
        for batch in batches[1:]:
            structure = self._llm_group_continue(structure, batch)
        by_title = {n["title"]: n for n in nodes}
        used = set()
        groups = []
        for g in structure:
            if not isinstance(g, dict):
                continue
            title = str(g.get("title", "")).strip()
            if not title:
                continue
            members = []
            for name in g.get("members", []):
                member = by_title.get(str(name).strip())
                if member is not None and member["id"] not in used:
                    members.append(member)
                    used.add(member["id"])
            if len(members) >= 2:
                groups.append({"title": title,
                               "summary": str(g.get("summary", "")).strip(),
                               "members": members})
        return groups

    def _llm_group_init(self, batch: List[dict]) -> list:
        prompt = f"""你是一个知识库主题组织专家（上层结构生成）。以下是知识库的一批主题簇，请组织它们的上级结构：将语义相近的簇归入更粗的主题分组，并定名、撰写一句话摘要。没有合适归属的簇可不归组。

簇列表：
{json.dumps(self._nodes_info(batch), ensure_ascii=False)}

要求：
1. 每个分组至少包含 2 个簇
2. 一个簇最多归入一个分组
3. 宁缺毋滥：语义不相关不要强行归组

返回JSON格式：{{"groups": [{{"title": "分组名", "summary": "摘要", "members": ["簇标题1", "簇标题2"]}}, ...]}}
直接返回最终JSON结构，不要输出其他内容。"""
        data = self._llm_json(prompt)
        return data.get("groups", []) if isinstance(data, dict) else []

    def _llm_group_continue(self, structure: list, batch: List[dict]) -> list:
        existing = []
        for g in structure:
            if not isinstance(g, dict):
                continue
            members = [m["title"] if isinstance(m, dict) else str(m)
                       for m in g.get("members", [])]
            existing.append({"title": str(g.get("title", "")),
                             "summary": str(g.get("summary", "")),
                             "members": members})
        prompt = f"""你是一个知识库主题组织专家（上层结构生成）。已有前批的上层分组结构如下，请继续处理新一批簇：相关簇可并入已有分组，也可另立新组；没有合适归属的簇可不归组。不得改动前批结果，返回更新后的完整分组。

已有分组结构：
{json.dumps(existing, ensure_ascii=False)}

新一批簇：
{json.dumps(self._nodes_info(batch), ensure_ascii=False)}

要求：
1. 每个分组至少包含 2 个簇
2. 一个簇最多归入一个分组
3. 宁缺毋滥：语义不相关不要强行归组

返回JSON格式：{{"groups": [{{"title": "分组名", "summary": "摘要", "members": ["簇标题1", ...]}}, ...]}}
直接返回最终JSON结构，不要输出其他内容。"""
        data = self._llm_json(prompt)
        return data.get("groups", []) if isinstance(data, dict) else structure

    def _nodes_info(self, nodes: List[dict]) -> List[dict]:
        counts = self._node_doc_counts()
        return [{"title": n["title"], "summary": n["summary"] or "",
                 "doc_count": counts.get(n["id"], 0)} for n in nodes]

    def _relevel_subtree(self, nodes_by_parent: dict, node_id: int, level: int) -> None:
        self.db.update_corpus_tree_node(node_id, level=level)
        for child in nodes_by_parent.get(node_id, []):
            self._relevel_subtree(nodes_by_parent, child["id"], level + 1)

    # ------------------------------------------------------------------
    # ⑥ 增量辅助
    # ------------------------------------------------------------------

    def _ensure_root(self) -> int:
        for n in self.db.get_corpus_tree_nodes():
            if n["kind"] == "root":
                return n["id"]
        return self.db.insert_corpus_tree_node(None, ROOT_TITLE, "语料库根节点", 0, kind="root")

    def _ensure_fallback_cluster(self, root_id: int) -> int:
        for n in self.db.get_corpus_tree_nodes():
            if n["parent_id"] == root_id and n["title"] == FALLBACK_TITLE:
                return n["id"]
        return self.db.insert_corpus_tree_node(
            root_id, FALLBACK_TITLE, "未提取到语义标签的文档", 1, kind="cluster")

    def _find_cluster_for_tag(self, root_id: int, canonical: str) -> int:
        """规范标签→簇：命中则就近挂簇（多个候选取最小者均衡）；挂不进则 LLM 裁定新开/并入。"""
        nodes = self.db.get_corpus_tree_nodes()
        counts = self._node_doc_counts()
        tagged = [n for n in nodes if n["kind"] == "cluster" and n["tag"] == canonical]
        with_docs = [n for n in tagged if counts.get(n["id"], 0)]
        pool = with_docs or tagged
        if pool:
            return min(pool, key=lambda n: (counts.get(n["id"], 0), n["id"]))["id"]
        return self._adjudicate_new_cluster(root_id, canonical, nodes, counts)

    def _adjudicate_new_cluster(self, root_id: int, canonical: str,
                                nodes: List[dict], counts: dict) -> int:
        leaves = [n for n in nodes if n["kind"] == "cluster" and counts.get(n["id"], 0) > 0]
        if leaves:
            info = [{"title": n["title"], "summary": n["summary"] or "",
                     "doc_count": counts.get(n["id"], 0)} for n in leaves]
            prompt = f"""你是一个主题聚类专家（挂簇裁定）。新文档的规范标签"{canonical}"没有对应的主题簇。现有主题簇如下：
{json.dumps(info, ensure_ascii=False)}

请裁定：应为该标签新开一个簇，还是并入某个现有簇（语义高度相近时）？

返回JSON格式：{{"action": "create", "title": "新簇名", "summary": "一句话摘要"}} 或 {{"action": "attach", "target": "现有簇标题"}}
直接返回最终JSON结构，不要输出其他内容。"""
            data = self._llm_json(prompt)
            if isinstance(data, dict):
                action = data.get("action")
                if action == "attach":
                    title = str(data.get("target", "")).strip()
                    for n in leaves:
                        if n["title"] == title:
                            return n["id"]
                elif action == "create":
                    title = str(data.get("title", "")).strip() or canonical
                    summary = str(data.get("summary", "")).strip()
                    node_id = self.db.insert_corpus_tree_node(
                        root_id, title, summary, 1, kind="cluster", tag=canonical)
                    self.db.insert_corpus_tree_event(
                        node_id, "new_cluster",
                        json.dumps({"tag": canonical, "title": title}, ensure_ascii=False))
                    return node_id
        node_id = self.db.insert_corpus_tree_node(
            root_id, canonical, "", 1, kind="cluster", tag=canonical)
        self.db.insert_corpus_tree_event(
            node_id, "new_cluster",
            json.dumps({"tag": canonical, "title": canonical}, ensure_ascii=False))
        return node_id

    # ------------------------------------------------------------------
    # 通用辅助
    # ------------------------------------------------------------------

    def _node_doc_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = defaultdict(int)
        for _, node_id, _ in self.db.get_all_corpus_memberships():
            counts[node_id] += 1
        return counts

    def _llm_json(self, prompt: str):
        try:
            response = llm_completion(self._llm_model, prompt, thinking_disabled=True)
            if not response:
                return None
            return extract_json(response)
        except Exception as e:
            logging.warning("Corpus tree LLM call failed: %s", e)
            return None
