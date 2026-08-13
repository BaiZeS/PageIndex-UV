"""P2 统一检索增强原语 UnifiedNodeEnhancement（spec [3.2]/[1.1]/[1.2]/[7.3]-[7.7]）。

给定 (query, candidate_nodes)，对单文档节点与语料树分支一视同仁：
① 高召回 union 收窄（宁多勿漏）——四通道各自给出命中的节点，取并集；仍过宽才做保召回上限。
   P2.6 起另设正文内容通道：候选可选携带 `text` 字段，query token 命中正文即准入
   （存储签名被垃圾词淹没/缺失时的直接内容接地，[1.1] 召回优先）；无 `text` 字段的
   候选（如语料树簇节点）通道关闭，行为与之前完全一致。
② 证据组装——为每个候选节点打包接地证据（实体/关键词/标签命中 + 标题/摘要）。
③ LLM 精挑（唯一裁剪者；可变数量、宁缺毋滥）+ 候选池质疑信号
   → {"selected_ids": [...], "pool_concern": bool, "concern_reason": str}。
④ 被截候选进延迟池（deferred pool）可恢复，不做不可逆硬丢弃。

union/证据组装为纯 Python（查表 + 集合并，全程无 LLM）([7.3])；LLM 只在第③步精挑。
LLM 失效时不做启发式裁剪——放行证据（union 候选即选中），保证"LLM 唯一裁剪"语义 ([7.7])。
"""
import asyncio
import json
import logging

from ..utils import llm_completion, extract_json
from ..super_tree import KeywordIndex

# [1.2]③ 多信号加权：w_e > w_t > w_k，累加；多信号同时命中 > 单档
WEIGHT_ENTITY = 3.0
WEIGHT_TAG = 2.0
WEIGHT_KEYWORD = 1.0
# [1.2]③ 零值泛滥防护：全零时不按 score 收缩，按输入顺序取绝对上限（cap×该系数）兜底
ABSOLUTE_CEILING_MULTIPLIER = 2

# [7.4] 单节点证据封顶
EVIDENCE_MAX_ENTITIES_PER_NODE = 3
EVIDENCE_MAX_KEYWORDS_PER_NODE = 5
EVIDENCE_MAX_TAGS_PER_NODE = 2
# [7.4] 同一实体/关键词命中超过该阈值的节点数 → 一条全局注记代替逐节点重复
GLOBAL_NOTE_THRESHOLD = 3
# [3.2.2] 摘要呈现长度
SUMMARY_MAX_CHARS = 200

_DEFAULT_UNION_MAX_CANDIDATES = 80
_DEFAULT_EVIDENCE_MAX_CHARS = 6000

# [3.2.1] pool_concern 重选：union 上限放宽倍数。候选/签名不变，被截候选经
# union 自然回池，该参数只抬高上限让延迟池节点重新可被 LLM 精挑。
# 延迟分支与 force-all 全池分支共用该倍数：force-all 不放宽 cap 时，零信号候选
# 会在全量准入后按分降序垫底、再次被截——全池重选退化为 pass-1，判据①救不到
# 任何节点。单文档 _search_single / 多文档 _recall_nodes_for_doc 经
# retry_on_pool_concern 共享；语料树逐层 _navigate_level 只用延迟分支
# （簇节点无正文通道）。
POOL_CONCERN_RETRY_CAP_MULTIPLIER = 2


def _coerce_cap(value):
    """cap 强制：合法值钳到 ≥1；非数值/无效值 → None（调用方回退下一级，绝不抛出）。"""
    try:
        return max(1, int(value))
    except (TypeError, ValueError, OverflowError):
        return None


class UnifiedNodeEnhancement:
    """高召回 union 收窄 + 证据组装 + LLM 精挑（唯一裁剪者）。"""

    _GUIDANCE = (
        "实体和关键词匹配是语料事实，请优先依据它们与问题的语义关联程度判断，"
        "而非简单计数命中个数。"
    )
    _CONCERN_CRITERIA = (
        "候选池质疑（pool_concern）判断依据——仅当满足以下其一才置 true，判据之外一律 false：\n"
        "①查询里的关键概念（实体/核心关键词）没有命中任何候选的证据；\n"
        "②命中的候选数明显偏少（如仅 1 个）且证据偏弱；\n"
        "③选中节点间主题互斥/矛盾，疑似漏掉真正的分支。\n"
        "置 true 时 concern_reason 简述判据；否则 concern_reason 留空字符串。"
    )

    def __init__(self, model: str, retrieve_model: str = None):
        self.model = model
        self.retrieve_model = retrieve_model
        self.union_max_candidates = _DEFAULT_UNION_MAX_CANDIDATES
        self.evidence_max_chars = _DEFAULT_EVIDENCE_MAX_CHARS
        try:
            from ..utils import ConfigLoader
            cfg = ConfigLoader().load(None)
            self.union_max_candidates = int(getattr(cfg, "union_max_candidates", self.union_max_candidates))
            self.evidence_max_chars = int(getattr(cfg, "evidence_max_chars", self.evidence_max_chars))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 通道匹配（纯 Python，无 LLM；tokenizer 与索引期一致）
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> list:
        # KeywordIndex._tokenize 不依赖 self：复用索引期 jieba + 停用词过滤
        return KeywordIndex._tokenize(None, text or "")

    @staticmethod
    def _keyword_hits(query_tokens: list, keywords) -> list:
        """关键词通道：token 相等，或多字 token 双向包含。"""
        hits = []
        for kw in keywords or []:
            if not isinstance(kw, str) or not kw.strip():
                continue
            k = kw.strip().lower()
            for qt in query_tokens:
                if qt == k or (len(qt) >= 2 and len(k) >= 2 and (qt in k or k in qt)):
                    hits.append(kw.strip())
                    break
        return hits

    @staticmethod
    def _tag_hits(query_tokens: list, query_cf: str, tags) -> list:
        """标签通道：query token/整串子串 vs 节点标签（query_cf 为调用方提升的 query.casefold()）。"""
        hits = []
        for tag in tags or []:
            if not isinstance(tag, str) or not tag.strip():
                continue
            t = tag.strip()
            tl = t.casefold()
            if len(tl) >= 2 and tl in query_cf:
                hits.append(t)
                continue
            for qt in query_tokens:
                if qt == tl or (len(qt) >= 2 and len(tl) >= 2 and (qt in tl or tl in qt)):
                    hits.append(t)
                    break
        return hits

    @staticmethod
    def _content_hits(query_tokens: list, text) -> list:
        """正文内容通道（P2.6，[1.1] 召回优先）：多字 query token 子串命中候选正文。

        确定性、无 LLM、无需对正文重新分词（query token 已是 _tokenize 的
        len≥2 归一 token；正文按 casefold 子串匹配，兼容 CJK 与 ASCII 大小写）。
        存储签名被引用垃圾淹没/缺失时，正文是唯一可靠接地——命中即准入 union，
        命中词记入关键词证据。返回命中的 query token（保序去重）。
        防御性：text 非字符串/为空 → 空命中（通道关闭，绝不抛出）。
        已知取舍：子串匹配对嵌在更长词内部的短 token 可能误命中（如 "ai" 命中
        "retain"），召回优先 [1.1] 下接受该误报——准入宽于漏召。
        """
        if not isinstance(text, str) or not text or not query_tokens:
            return []
        t = text.casefold()
        hits = []
        seen = set()
        for qt in query_tokens:
            if len(qt) >= 2 and qt in t and qt not in seen:
                seen.add(qt)
                hits.append(qt)
        return hits

    @staticmethod
    def _entity_hits(query_entities, entities) -> list:
        """实体通道：query_entities 名 vs 节点实体名（casefold，子串放行）。"""
        hits = []
        qe_norm = [
            qe.strip().casefold()
            for qe in (query_entities or [])
            if isinstance(qe, str) and qe.strip()
        ]
        for ent in entities or []:
            if not isinstance(ent, dict):
                continue
            name = ent.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            nc = name.strip().casefold()
            for qc in qe_norm:
                if qc == nc or qc in nc or nc in qc:
                    hits.append({"name": name.strip(), "type": ent.get("type") or ""})
                    break
        return hits

    @staticmethod
    def _to_bool(val) -> bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ("true", "yes", "是", "1", "y")
        return bool(val)

    # ------------------------------------------------------------------
    # ② 证据组装（[3.2.2] 格式 + [7.4] 防过载）
    # ------------------------------------------------------------------
    def _assemble_evidence(self, union, node_signals, cand_by_id) -> str:
        matches = {}
        for nid in union:
            sig = node_signals[nid]
            matches[nid] = {
                "entities": list(sig["entities"]),
                "keywords": list(sig["keywords"]),
                "tags": list(sig["tags"]),
            }

        def keeper_of(node_ids):
            # 信号分最高者保留逐节点展开；平分取输入序靠前者
            return max(node_ids, key=lambda n: (node_signals[n]["score"], -node_signals[n]["pos"]))

        notes = []
        # 实体全局注记：同一实体命中 >GLOBAL_NOTE_THRESHOLD 个节点
        # （按 (key, nid) 去重——计数 distinct 节点：同 profile 的大小写变体不得重复计入）
        ent_spread = {}
        for nid in union:
            for ent in matches[nid]["entities"]:
                bucket = ent_spread.setdefault(str(ent["name"]).casefold(), [])
                if nid not in bucket:
                    bucket.append(nid)
        for key, node_ids in ent_spread.items():
            if len(node_ids) > GLOBAL_NOTE_THRESHOLD:
                keeper = keeper_of(node_ids)
                name = next(
                    (e["name"] for nid in union for e in matches[nid]["entities"]
                     if nid in node_ids and str(e["name"]).casefold() == key),
                    key,
                )
                notes.append(f"注：实体 {name} 命中于节点 {'、'.join(node_ids)}")
                for nid in node_ids:
                    if nid != keeper:
                        matches[nid]["entities"] = [
                            e for e in matches[nid]["entities"]
                            if str(e["name"]).casefold() != key
                        ]
        # 关键词全局注记（同上：按 (key, nid) 去重，重复关键词条目只计一次节点）
        kw_spread = {}
        for nid in union:
            for kw in matches[nid]["keywords"]:
                bucket = kw_spread.setdefault(str(kw).lower(), [])
                if nid not in bucket:
                    bucket.append(nid)
        for key, node_ids in kw_spread.items():
            if len(node_ids) > GLOBAL_NOTE_THRESHOLD:
                keeper = keeper_of(node_ids)
                notes.append(f"注：关键词 {key} 命中于节点 {'、'.join(node_ids)}")
                for nid in node_ids:
                    if nid != keeper:
                        matches[nid]["keywords"] = [
                            k for k in matches[nid]["keywords"] if str(k).lower() != key
                        ]

        # [7.4] 单节点封顶（只呈现命中项）
        for nid in union:
            m = matches[nid]
            m["entities"] = m["entities"][:EVIDENCE_MAX_ENTITIES_PER_NODE]
            m["keywords"] = m["keywords"][:EVIDENCE_MAX_KEYWORDS_PER_NODE]
            m["tags"] = m["tags"][:EVIDENCE_MAX_TAGS_PER_NODE]

        blocks = {}
        for nid in union:
            cand = cand_by_id[nid]
            title = str(cand.get("title") or "")
            summary = str(cand.get("summary") or "")[:SUMMARY_MAX_CHARS]
            m = matches[nid]
            lines = [f"候选节点 {nid}：", f"标题：{title}"]
            if m["entities"]:
                lines.append("实体匹配：" + "、".join(
                    f"{e['name']}（{e['type']}）" for e in m["entities"]
                ))
            if m["keywords"]:
                lines.append("关键词命中：" + "、".join(str(k) for k in m["keywords"]))
            if m["tags"]:
                lines.append("标签命中：" + "、".join(str(t) for t in m["tags"]))
            lines.append(f"摘要：{summary}")
            blocks[nid] = "\n".join(lines)

        # [7.4] 跨节点总预算：超限按证据强度（多信号分）升序退化弱候选为"标题+摘要"一行
        # 注：每节点一行、注记不动是有意的尽力下限——极小 evidence_max_chars 下仍可能超限，不再进一步裁剪。
        total = sum(len(b) for b in blocks.values()) + sum(len(n) for n in notes)
        if total > self.evidence_max_chars:
            degradables = sorted(
                union, key=lambda nid: (node_signals[nid]["score"], -node_signals[nid]["pos"])
            )
            for nid in degradables:
                if total <= self.evidence_max_chars:
                    break
                cand = cand_by_id[nid]
                one_line = (
                    f"候选节点 {nid}：标题：{cand.get('title') or ''}"
                    f"｜摘要：{str(cand.get('summary') or '')[:SUMMARY_MAX_CHARS]}"
                )
                total += len(one_line) - len(blocks[nid])
                blocks[nid] = one_line

        return "\n".join(notes + [blocks[nid] for nid in union])

    # ------------------------------------------------------------------
    # prompt 构建
    # ------------------------------------------------------------------
    @staticmethod
    def _positive_budget(v):
        """非正/非数值预算 → None（视为未给），避免渲染"最多选 0 个节点"之类无效条款。"""
        try:
            return v if float(v) > 0 else None
        except (TypeError, ValueError):
            return None

    def _build_budget_block(self, node_budget, token_budget) -> str:
        """[7.5]b 预算转 prompt 指令：预算只是约束，不替 LLM 决定谁相关。"""
        node_budget = self._positive_budget(node_budget)
        token_budget = self._positive_budget(token_budget)
        if node_budget is None and token_budget is None:
            return ""
        clauses = []
        if node_budget is not None:
            clauses.append(f"本轮最多选 {node_budget} 个节点")
        if token_budget is not None:
            clauses.append(f"所选节点正文合计约 {token_budget} token")
        clauses.append("超出预算优先选证据最充分的")
        if node_budget is not None:
            clauses.append(f"selected_ids 个数不得超过 {node_budget}")
        return "预算约束：" + "；".join(clauses) + "。预算只是约束，不替你决定谁相关。"

    def _build_prompt(self, query, evidence_text, node_budget, token_budget) -> str:
        budget_block = self._build_budget_block(node_budget, token_budget)
        budget_section = f"{budget_block}\n\n" if budget_block else ""
        return f"""你是检索增强专家。请基于语料证据，从候选节点中精选与查询真正相关的节点。宁缺毋滥：数量可变，只选相关的，不相关的一个都不选。

查询：{query}

候选节点证据：
{evidence_text}

判断指引：{self._GUIDANCE}证据只呈现命中项，未列全量签名。

{budget_section}{self._CONCERN_CRITERIA}

返回JSON格式: {{"selected_ids": [...], "pool_concern": bool, "concern_reason": str}}
selected_ids 只能取自上述候选节点的 node_id；直接返回JSON，不要其他内容。
"""

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    async def enhance_and_select(
        self,
        query,
        candidates,
        profiles,
        query_entities=None,
        node_budget=None,
        token_budget=None,
        max_candidates=None,
        force_all_candidates: bool = False,
    ) -> dict:
        """四通道高召回 union → 证据组装 → LLM 精挑。

        candidates: [{"node_id", "title", "summary", ("text")}]——`text` 为可选
                    正文内容通道字段（P2.6）：query token 命中正文即准入 union，
                    命中词记入关键词证据；缺省该字段的候选行为不变（如语料树簇节点）
        profiles:   {node_id: {"entities": [{"name","type"}], "keywords": [...], "tags": [...]}}
                    （缺失节点 → 空证据，仍可被选中，[7.7] 签名缺失退化）
        query_entities: 实体名字符串列表（由调用方解析，含别名）
        max_candidates: 本次调用的 union 上限覆盖（None → 实例配置的
                    union_max_candidates）；pool_concern 放宽重选时用 ([3.2.1])
        force_all_candidates: True → union 步骤全量候选准入（零信号直通），
                    pool_concern 且无被截候选时的全池重选用 ([3.2.1])；
                    弱候选由证据跨节点预算自然退化为一行标题+摘要 ([7.4])
        返回: {"selected_ids": [...], "pool_concern": bool,
               "concern_reason": str, "deferred": [node_ids]}
        """
        empty = {"selected_ids": [], "pool_concern": False, "concern_reason": "", "deferred": []}
        if not candidates:
            return empty
        profiles = profiles or {}
        query = query or ""

        # 归一候选：按 node_id 去重并记录输入序（确定性裁决）
        norm = []
        cand_by_id = {}
        for pos, cand in enumerate(candidates):
            if not isinstance(cand, dict):
                continue
            raw_id = cand.get("node_id")
            if raw_id is None or str(raw_id) == "" or str(raw_id) in cand_by_id:
                continue
            nid = str(raw_id)
            cand_by_id[nid] = cand
            norm.append((pos, nid))
        if not norm:
            return empty

        # ① union（纯查表/集合并，全程无 LLM，[7.3]）
        query_tokens = self._tokenize(query)
        query_cf = query.casefold()
        node_signals = {}
        union = []
        for pos, nid in norm:
            cand = cand_by_id[nid]
            raw_id = cand.get("node_id")
            try:
                prof = profiles.get(raw_id)
            except TypeError:
                # 不可哈希的 node_id（list/dict 等）：回退到归一后的 str(nid) 查表
                prof = None
            if prof is None:
                prof = profiles.get(nid)
            prof = prof if isinstance(prof, dict) else {}
            ents = self._entity_hits(query_entities, prof.get("entities"))
            kws = self._keyword_hits(query_tokens, prof.get("keywords"))
            tags = self._tag_hits(query_tokens, query_cf, prof.get("tags"))
            # 正文内容通道（P2.6）：命中词并入关键词证据（casefold 去重，
            # 与存储签名命中不重复计分/呈现）；候选无 `text` 字段则通道关闭。
            body_hits = self._content_hits(query_tokens, cand.get("text") or "")
            if body_hits:
                seen_kw = {str(k).casefold() for k in kws}
                kws = kws + [h for h in body_hits if str(h).casefold() not in seen_kw]
            score = (
                WEIGHT_ENTITY * len(ents)
                + WEIGHT_TAG * len(tags)
                + WEIGHT_KEYWORD * len(kws)
            )
            node_signals[nid] = {
                "entities": ents, "keywords": kws, "tags": tags,
                "score": score, "pos": pos,
            }
            if ents or kws or tags:
                union.append(nid)

        # 零信号：全量送 LLM（收窄纪律——保召回优先，[1.1]）。
        # force_all_candidates（pool_concern 空池全量重选）等价于同样的全量直通，
        # 弱候选由证据跨节点预算退化为一行标题+摘要，不重复实现裁剪逻辑 ([7.4])。
        if force_all_candidates or not union:
            union = [nid for _, nid in norm]

        # ①b union 防爆炸（[1.2]）
        # cap 解析链：显式覆盖 → 实例配置 → 类默认；任一级非法值回退下一级，绝不抛出
        deferred = []
        cap = _coerce_cap(max_candidates)
        if cap is None:
            cap = _coerce_cap(self.union_max_candidates)
        if cap is None:
            cap = _coerce_cap(_DEFAULT_UNION_MAX_CANDIDATES)
        if len(union) > cap:
            if all(node_signals[nid]["score"] == 0 for nid in union):
                # 零值泛滥防护：禁止按 score 收缩——输入顺序 + 绝对上限兜底
                ceiling = cap * ABSOLUTE_CEILING_MULTIPLIER
                deferred = union[ceiling:]
                union = union[:ceiling]
            else:
                # 多信号加权降序（稳定排序，平分按输入序），截断进延迟池可回捞
                ordered = sorted(
                    union, key=lambda nid: (-node_signals[nid]["score"], node_signals[nid]["pos"])
                )
                deferred = ordered[cap:]
                union = ordered[:cap]

        # ② 证据组装（只对 union 内节点，无冗余打分）
        evidence_text = self._assemble_evidence(union, node_signals, cand_by_id)

        # ③ LLM 精挑（唯一裁剪者）
        prompt = self._build_prompt(query, evidence_text, node_budget, token_budget)
        try:
            response = await asyncio.to_thread(
                llm_completion, self.retrieve_model or self.model, prompt,
                thinking_disabled=False,
            )
        except Exception as e:
            logging.warning("UnifiedNodeEnhancement LLM select failed: %s", e)
            response = ""

        data = extract_json(response) if isinstance(response, str) and response.strip() else {}
        if not isinstance(data, dict) or not isinstance(data.get("selected_ids"), list):
            # [7.7] 降级：不做启发式裁剪，放行证据（union 候选即选中）
            return {
                "selected_ids": list(union),
                "pool_concern": False,
                "concern_reason": "llm_unavailable",
                "deferred": deferred,
            }

        union_ids = set(union)
        selected = []
        seen = set()
        for item in data["selected_ids"]:
            sid = str(item)
            if sid in union_ids and sid not in seen:
                seen.add(sid)
                selected.append(sid)
        return {
            "selected_ids": selected,
            "pool_concern": self._to_bool(data.get("pool_concern", False)),
            "concern_reason": str(data.get("concern_reason") or ""),
            "deferred": deferred,
        }


def resolve_query_entities(db, query, limit=5) -> list:
    """查询实体解析（共享助手）：search_entities 命中实体的规范名 + 别名展开。

    供检索路径（T6.2 单文档 / T6.4 router）复用：把 query 命中的实体名与别名
    展平为字符串列表，交给 enhance_and_select 的实体通道。确定性：按搜索结果
    顺序去重（casefold 判重，保留首见写法）。防御性：db 调用异常、别名 JSON
    解析失败均不抛出——坏条目跳过，其余照常返回。
    """
    if db is None or not query or not str(query).strip():
        return []
    try:
        rows = db.search_entities(query, limit=limit)
    except Exception as e:
        logging.warning("resolve_query_entities: search_entities failed: %s", e)
        return []

    names = []
    seen = set()

    def _add(value):
        if not isinstance(value, str):
            return
        v = value.strip()
        if not v:
            return
        key = v.casefold()
        if key not in seen:
            seen.add(key)
            names.append(v)

    for row in rows or []:
        if not isinstance(row, dict):
            continue
        _add(row.get("name"))
        aliases = row.get("aliases")
        if isinstance(aliases, str):
            try:
                aliases = json.loads(aliases)
            except (TypeError, ValueError):
                aliases = []
        if isinstance(aliases, list):
            for alias in aliases:
                _add(alias)
    return names


def resolve_node_profiles(db, db_doc_id, mapping) -> dict:
    """逐节点证据签名解析（共享助手，[3.4]/[3.2.1]）。

    解析序：DB node_profiles 表优先（索引期权威签名；db_doc_id 必须是 documents
    表的整数 id——调用方经 _id_mapper.to_db 转换）；DB 无行时回退 structure 节点
    字典自带的 entities/keywords/tags 键（异步索引路径 / workspace JSON）；两者
    皆无 → 空 dict（enhance_and_select 对缺失签名优雅退化，[7.7]）。

    db: PageIndexDB 实例（None → 回退路径）
    db_doc_id: 文档整数 id（None → 回退路径）
    mapping: {node_id: node_dict}——create_node_mapping 的扁平结构映射

    防御性：get_node_profiles 异常不抛出——记日志并走回退路径。
    """
    if db is not None and db_doc_id is not None:
        rows = None
        try:
            rows = db.get_node_profiles(db_doc_id)
        except Exception as e:
            logging.warning(
                "resolve_node_profiles: get_node_profiles(%s) failed: %s", db_doc_id, e
            )
        if rows:
            return {
                p["node_id"]: {
                    "entities": p.get("entities") or [],
                    "keywords": p.get("keywords") or [],
                    "tags": p.get("tags") or [],
                }
                for p in rows if isinstance(p, dict) and p.get("node_id")
            }
    profiles = {}
    for nid, node in (mapping or {}).items():
        if not isinstance(node, dict):
            continue
        prof = {
            key: node.get(key)
            for key in ("entities", "keywords", "tags")
            if node.get(key)
        }
        if prof:
            profiles[nid] = prof
    return profiles


async def retry_on_pool_concern(enhancer, result, query, candidates, profiles,
                                query_entities=None) -> dict:
    """[3.2.1] pool_concern 重选共享助手：至多一次重试，二选一互斥分支，杜绝循环。

    ① 存在被截候选 → 放宽 union 上限重选（候选/签名不变，被截候选经 union
       自然回池，上限放宽只抬高 cap）。
    ② 无被截候选（候选池本就完整）→ 判据①"关键概念无命中"意味着 union 准入
       逻辑漏掉了相关节点 → force_all_candidates 全池直通重选一次，并同样放宽
       cap：不放宽则零信号候选在全量准入后按分降序垫底、再次被截，全池重选退化
       为 pass-1，救不到判据①要救的节点。

    无 pool_concern → 原样返回（零额外 LLM 调用）。重选后仍 pool_concern 也直接
    返回（接受结果，medium 语义由调用方决定）。候选/签名对象不做拷贝，与首选调用
    保持同一引用。语料树逐层 _navigate_level 只用延迟分支（簇节点无正文通道），
    不经本助手。
    """
    if not result["pool_concern"]:
        return result
    widened_cap = (
        max(1, int(enhancer.union_max_candidates))
        * POOL_CONCERN_RETRY_CAP_MULTIPLIER
    )
    if result["deferred"]:
        return await enhancer.enhance_and_select(
            query, candidates, profiles, query_entities=query_entities,
            max_candidates=widened_cap,
        )
    return await enhancer.enhance_and_select(
        query, candidates, profiles, query_entities=query_entities,
        max_candidates=widened_cap,
        force_all_candidates=True,
    )
