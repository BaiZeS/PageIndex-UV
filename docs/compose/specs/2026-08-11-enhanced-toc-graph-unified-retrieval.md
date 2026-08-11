# PageIndex-UV 整体架构优化：增强 TOC 图谱 + 统一检索增强

> 版本：v3.2（2026-08-11，修正收窄纪律 + 显式 pool_concern + 统一调用路径 + LLM 决策的 expand 重挑）
> 上游依据：`2026-08-04-overall-architecture-semantic-tree-navigation.md`（[S1]–[S13]）
> 本文档是对上游 spec 的**增量优化**，不推翻其骨架：LLM TOC 推理为核心、图谱辅助、四通道预筛。
> 决策记录（用户 2026-08-11 拍板）：①方案A（节点本地属性富集，树即图谱）②单/多文档统一增强 ③高召回 union + LLM 精挑为唯一裁剪 ④召回预算 = agentic 多轮并发召回（LLM 判停）。
> v2 更新（整体审查）：并入用户 5 项精化（见 [1.2]/[2.2]/[3.2.2]/[3.4.1]/[3.5]要点8–10）；澄清"图谱=召回信号源、非打分通道"（[3.3]）；P0 位置改以函数名定位；[3.4.1] 与代码现状对齐。
> v3 更新（实现细化）：新增 **[7] 实现细化与风险防控**，逐项回应用户 6 个工程问题（消歧轻量有效/标签稳定可控/union 效率/证据防过载/agentic 陷阱/best-effort 作答），并对齐现有代码（`disambiguate_entity`、`normalize_entities_batch`、`_extract_tags` 等已存在）。
> v3.1 追加拍板：**verifier 取偏严取向（召回优先）+ 多轮 token 预算 + 延迟保护（总耗时/单轮超时/首轮即答快速路径）**，见 [7.5]；[6] 对应开放项已关闭。
> v3.2 修正（回应 6 个纪律边界问题）：①通道 top_k 重定位为"召回成本上限"、区分精确/相似度通道（[1.2]①）；②超限收缩改"多信号加权"而非单档优先级（[1.2]③）；③被截候选进**延迟池**可恢复，不硬丢（[1.2]④）；④`pool_concern` 显式扩召信号（[3.2]、[3.5]要点7）；⑤新增 [3.2.1] 统一调用路径示例（语料树逐层也走 enhance_and_select，防走回老路）；⑥expand"重新挑选"必须 LLM 决策（[7.5]b）；⑦重排层默认不上维持（[1.1]）。

---

## [0] 设计命题与诊断

### [0.1] 实测痛点
正式检索基准召回仅 **0.12–0.33**（mldr_zh 0.33 / t2 0.16 / du 0.22），且单样本几乎**总是恰好命中 1 个相关文档、其余全丢**（t2 命中 1/6、1/8、1/5）。

### [0.2] 根因（诊断，已实证）
**基于语义树 TOC 的推理，主要依靠 LLM 自身能力。当查询概念超出 LLM 知识范围时，极易选错 TOC 分支，造成召回低、找不到正确答案。**

典型例证（浴血值案例）：
- 答案在节点1（"门派介绍"），但该节点摘要只写"详细介绍了12个门派的攻击特点"，**不含"浴血值"**。
- LLM 从"浴血值"推不出它归属"门派介绍"节点（参数化知识盲区）。
- LLM 改选含"帮会"字样的节点6 → 节点6无答案 → Recall=0。

**结论**：PageIndex TOC 分支选择赌的是"LLM 能从标题+摘要推出查询落点"。查询概念与分支标题的关联一旦不在 LLM 常识内，赌注必输。这是单文档与多文档**共同的**失败模式。

### [0.3] 优化立场（本方案核心）
用**语料接地（corpus grounding）**补上 LLM 知识盲区——把语料本身的事实（实体、关键词、标签、向量）作为**证据**注入 LLM 的分支决策，替代 LLM 的盲目猜测。

- LLM 猜不出"浴血值∈门派介绍"，但**关键词通道**能查出"浴血值出现在节点1"，**实体图谱**能查出"浴血值→相关→帮会系统"。
- 这些信号来自**索引的真实语料**，不是 LLM 脑补。

**这是"增强（augment）"而非"替代（replace）"：LLM 仍是唯一决策者，但它拿到的是有语料证据支撑的选择。**

---

## [1] 三大设计原则（承接上游，强化纪律）

| # | 原则 | 纪律 |
|---|------|------|
| 1 | **LLM TOC 推理是核心**（每层/每节点决策由 LLM 完成） | 不跳过 LLM；关键词/图谱/向量**不是**过滤器，是证据 |
| 2 | **图谱辅助推理**，把单文档优势扩展到多文档 | 图谱不是并列召回通道，是"快捷跳转 + 预筛加权 + 多跳引导"三件套 |
| 3 | **四通道预筛只做高召回收窄**，减轻 LLM 上下文压力 | 高召回 union（宁多勿漏），**LLM 精挑是唯一裁剪者**；禁止分数排序硬截 |

### [1.1] 收窄纪律（本方案明确的底线）
**核心**：LLM 是**最终决策者**（谁被选中）；四通道 union 只做**召回**（把候选捞全），不做**判决**。

**但收窄不可避免**：候选不可能全塞进 LLM 的上下文。因此纪律不是"禁止一切收窄"，而是收窄必须满足三个约束：
1. **保召回优先于省成本**：宁可多送给 LLM，也不能把相关候选悄悄丢光。
2. **收窄只是"谁能被 LLM 看到"的入场排序，不是"谁相关"的决策**——最终相关性判断永远在 LLM 手里（可变数量、宁缺毋滥）。
3. **被收窄挤出的候选必须可恢复**：进"延迟池"（deferred pool），由 LLM 的 `pool_concern` 信号或 agentic 扩召捞回（见 [1.2]④、[3.2.1]、[3.5]），**不做不可逆的硬丢弃**。

> 上游 audit 发现的漂移（死向量通道×0.2 参与排序、宽层硬截 top-20、RRF rank 语义反转）正是违反了这条纪律，本方案予以纠正。

**明确排除/降级的做法（源自最佳实践审视）**：
- **禁止**：分数排序后**不可逆地**硬丢候选、图谱/向量作为并列打分通道参与 RRF 融合、用关键词**过滤**节点——均违反收窄纪律。
- **RRF 降级**：RRF 融合保留但**仅用于 v2 兜底路径**，不进主路径（主路径是语义树导航 + agentic 召回）。
- **重排（cross-encoder / LLM-reranker）默认不上**：避免偏离"LLM 精挑是唯一裁剪者"，作为可选 P3。

### [1.2] 高召回 union 必须防"数量爆炸"（用户 2026-08-11 精化；v3.2 修正三条规则）
"宁多勿漏"不等于无限扩张。union 后候选过大会爆 token 预算、引入噪音。落地规则：

1. **通道 top_k 是"召回成本上限"，不是"筛选判决"**——并区分两类通道：
   - **精确匹配通道**（关键词/实体/标签，倒排索引）：天然是"命中即召回"，**不做 score 排序截断**，直接返回全部命中——不存在"未进 top_k 而丢失"的问题。
   - **相似度通道**（向量，可选）：用**相似度阈值 + 宽松上限**召回，而非纯 top-k 硬截；丢失由 union 冗余（同一候选可被多通道捞起）覆盖。
   - **任何通道的遗漏兜底**：靠 union 冗余 + 延迟池 + `pool_concern` + agentic 扩召四重补偿（见④）。
2. **安全上限**：默认上限设**宽松且可配**（如 60–100 节点，按语料规模标定），宁可多给。
3. **超限收缩用"多信号加权"而非"单档优先级"**（v3.2 修正）：超上限时按**多信号命中加权分**排序取 top——`score = w_e·实体命中数 + w_t·标签命中数 + w_k·关键词命中数`（`w_e > w_t > w_k`，但**累加**）。**同时命中实体+标签+关键词的节点 > 仅命中单一档的节点**，避免旧"实体>标签>关键词"单档规则把多信号节点压掉。这只决定"谁先进 LLM 视野"，**不是相关性决策**。
4. **被挤出的候选进"延迟池"，不硬丢（v3.2 修正）**：超限被截下的候选保留为 deferred 集合；由 LLM 的 `pool_concern=true` 信号或 agentic 扩召时**优先回捞**（衔接 [3.2.1]、[3.5]）。这保证"宁多勿漏"在机制上成立——相关候选不会因一次超限而永久丢失。

---

## [2] 目标架构：增强 TOC 图谱（Enhanced TOC Graph）

### [2.1] 方案A：节点本地属性富集（树即图谱）

**核心思想**：TOC 树与知识图谱不再是两套正交结构分别查询，而是把属性**融合进 TOC 节点**，让每个节点自带"图谱感知"的属性签名。查询期 LLM 选分支时，O(1) 取到节点本地的接地证据。

**索引期为每个 TOC 节点生成「属性签名」(node profile)**：

```
node_profile = {
  "node_id": ...,
  "title": ...,                    # 原有
  "summary": ...,                  # 原有（LLM 生成，命门）
  "entities":   [                  # 新增：节点文本抽取出的实体（entity_extractor 归属到节点）
      {"name":"浴血值","type":"concept","relations":["related_to:帮会系统"]}, ...
  ],
  "keywords":   ["浴血值","帮会","NPC", ...],   # 新增：显著词（TF-IDF/TextRank，top-K）
  "tags":       ["游戏","帮会活动", ...],        # 新增/复用：closet_tags
  "embedding":  <optional vector>,              # 可选；默认关以保持无向量
}
```

**落库**：
- 单文档：写进节点 JSON（`doc["structure"]`），随现有 `_save_doc`/`tree_json` 持久化。
- 语料树：新增 `node_profiles` 表（node_id → 实体/关键词/标签），供海量档树导航 O(1) 查询。

**收益**：
- "浴血值∈节点1"这一语料事实，成为节点1签名里的 `keyword`/`entity`，LLM 选分支时**一眼可见**，不再靠猜。
- 图谱加权从"查询期现场 join 子树"变为"索引期落库签名"，逐层查表 O(1)。

**最佳实践谱系**（本节方案A 对应外部成熟实践）：
- **GraphRAG / LightRAG**（KG 结构引导检索）：方案A"增强 TOC 图谱"与其同源；上游图谱三件套即其检索侧。
- **Contextual Retrieval**（Anthropic 2024：摘要注入文档/父级上下文）：直击"摘要丢失信息"命门，作为节点摘要增强落在 P2/P3（见 P3 项1）。
- **HyDE / 查询扩展**：planner 已具备，保留。

### [2.2] 实体消歧前置（图谱质量命门，P1 必须搞定）⭐用户 2026-08-11 强调
图谱增强的价值上限取决于实体一致性。上游 [S7.1] 已指出：`insert_entity` 按精确名字去重，"张三/小张"连不起来。
**若不消歧**：查询问"小张的身份"时，即使节点里有"张三"的实体签名，`matched_entities` 也会漏掉——整个属性签名的价值就打折。

**P1 必须补实体消歧/归一化**（对"实体集合"的有界 LLM 共指消解，与标签归一化同构），否则融合进 TOC 的图谱是噪声。落地要点：
- 用**轻量级、有界**的 LLM 对实体集合做共指消解（不必全量精消，控制成本）。
- **分期范围**：先只对**人名、机构名、专有名词**做归一（它们是查询里最常出现、也最易写异的），通用概念可后置。
- 与标签归一化复用同一"有界集合级 LLM 操作"模式；增量场景对齐标签归一的增量版（新实体先查已有规范集，不中再单点裁定）。

> **现有代码基础 + 落地细化**（blocking 预裁剪、批归一分块、别名累积等）见 **[7.1]**。

---

## [3] 统一检索增强机制（单/多文档共用）

### [3.1] 问题：当前两条路径各走各的
- `_search_single`（client.py）：LLM 选节点 + ad-hoc 关键词合并 + 按 `len(summary)` 重排。
- `_recall_nodes_for_doc`（router.py）+ `_navigate_level`（super_tree.py）：多文档/树导航路径，逻辑各异。
- 两者栽的是**同一个跟头**（LLM 知识盲区选错分支），却用两套 ad-hoc 实现，且 `_search_single` 的重排已偏离"纯 LLM 推理"。

### [3.2] 统一抽象 `UnifiedNodeEnhancement`
一个共享函数，给定 `(query, candidate_nodes)`，对**单文档节点**与**语料树分支**一视同仁：

```
def enhance_and_select(query, candidate_nodes, profiles):
    # ① 高召回 union 收窄（宁多勿漏）
    #    四通道各自给出命中的节点，取并集；仍过宽才做保召回上限
    union = union_recall(query, candidate_nodes, profiles, channels=[tag, keyword, vector, entity])

    # ② 证据组装（为每个候选节点打包接地证据）
    evidence = []
    for node in union:
        p = profiles[node.id]
        evidence.append({
            "node_id": node.id, "title": node.title, "summary": node.summary,
            "matched_keywords": [kw for kw in p.keywords if kw in query_tokens],
            "matched_entities": [e for e in p.entities if linked(e, query_entities)],  # 层级感知加权
            "tag_matches": ...,
        })

    # ③ LLM 精挑（唯一裁剪者；可变数量、宁缺毋滥）+ 候选池质疑信号
    return llm_select(query, evidence)
        # → {"selected_ids": [...], "pool_concern": bool, "concern_reason": str}
```

**要点**：
- **①只收窄不决策**：四通道信号决定"哪些节点值得让 LLM 看"，高召回 union 保证不漏。
- **②证据替代猜测**：把节点签名里的关键词/实体/标签命中显式摆给 LLM，让 LLM 用语料事实而非参数化知识判断。
- **③LLM 决策不变**：仍是可变数量精挑，宁缺毋滥；但输入被"接地"了。
- **④输出附带 `pool_concern`（v3.2 新增）**：LLM 在精挑时同时判断"当前候选池是否可能不够全"，返回 `pool_concern: true/false`。这是**显式信号**，使"扩召"不必靠答案含糊去间接推断（见 [3.5]）。

### [3.2.1] 统一调用路径必须显式化（用户 2026-08-11 精化）
`enhance_and_select` 是**单位无关**的——它对"文档级候选"和"节点级候选"都适用，这正是单/多文档统一的落点。**实现时必须防止语料树导航绕回老路**，逐层导航也要走这个函数：

```
# ── 单文档：对"文档内全部节点"做一次 enhance_and_select ──
client.search(query, 1 doc)
  → _search_single(query, doc_id)
      → enhance_and_select(query, all_nodes_of_doc, node_profiles)   # unit = 节点
      → build_context_for_doc(selected_nodes) → answer
        （若 pool_concern=true 且是单文档内子节点被截，放宽节点数上限重选）

# ── 多文档（Super-Tree 主路径）：先选文档，再对每篇文档做节点级 enhance_and_select ──
router.search(query) → _search_super_tree(query)
  → super_tree.prefilter(query)                      # 四通道 union 出候选文档（宽召回）
  → super_tree.select_documents(query, candidates)   # LLM 精挑候选文档（map-reduce）
  → for doc_id in selected: _recall_nodes_for_doc(query, doc_id)
        → enhance_and_select(query, nodes_of(doc_id), node_profiles)  # unit = 节点
        → build_context → answer → verifier

# ── 语料树逐层导航（海量档）：对"每一层的分支/节点"同样调 enhance_and_select ──
super_tree.navigate_tree(query)
  → for level in 逐层:
       _navigate_level(query, sibling_nodes_at_this_level, ...)
         → enhance_and_select(query, siblings, unit_profiles)  # unit = 该层分支节点
         → 只展开选中分支，下一层再 enhance_and_select
```

**约束**：三层调用都必须满足 [1.1] 收窄纪律——union 宽召回 + LLM 精挑 + 超限候选进延迟池，任何一层不得出现"分数硬截后直接丢弃"。

### [3.2.2] 证据呈现格式极其关键（用户 2026-08-11 精化）
把 `matched_keywords`/`matched_entities`/`tag_matches` 喂给 LLM 时，**不能罗列琐碎命中**——否则 LLM 会被"命中数量"带偏、忽略语义关联。规则：
1. **每类证据用简洁格式化模板**，每节点一块，形如：
   ```
   候选节点A：标题"帮会活动-浴血值获取"
     实体匹配：浴血值（概念，关联帮会系统）
     关键词命中：浴血值, 帮会
     标签命中：游戏, 帮会活动
     摘要：本节说明如何通过帮会活动积累浴血值...
   ```
2. **prompt 明确引导判断口径**，防止机械计数：
   > "实体和关键词匹配是**语料事实**，请优先依据它们与问题的**语义关联程度**判断，而非简单计数命中个数。"
3. 证据只呈现**命中项的摘要**（不罗列全量签名），控制每节点证据体积。

### [3.3] 图谱加权的正确落位（层级感知、作为证据）
- L0 预计算一次实体距离（BFS ≤3 跳），逐层查表（已实现，保留）。
- 权重 = 距离衰减 × 关系类型权重（causal 1.0 / part_of 0.8 / related_to 0.6 / other 0.4）。
- **该权重不作为排序裁剪分，而是作为 `entity_relation` 证据喂给 LLM**（distance/type 显式呈现），助 LLM 判断该分支与查询实体的亲疏。

**澄清"图谱是不是召回通道"（避免与原则2表面矛盾）**：
- **允许的**：实体在**文档级召回 union**里贡献"候选"（含查询实体的文档被捞进并集）——这正是三件套的 **①快捷跳转**（实体→文档直达），是**放宽召回**（宁多勿漏），不打分。
- **节点级**：entity_boost 作为**证据**加权呈现给 LLM（三件套 ②预筛加权），不参与硬性排序/裁剪。
- **禁止的**：实体/图谱产出的**分数参与 RRF 最终排序或硬裁剪**——那才是"图谱当打分通道"，违反收窄纪律。
- **一句话**：实体在召回阶段"把候选捞全"（union 的一路信号源），在决策阶段"给 LLM 摆证据"，但**永不独自排序/裁剪**。这与原则2"图谱是辅助、非并列打分通道"是一致的——"通道"指打分融合通道，非召回信号源。

### [3.4] 单文档路径如何"融合"而不"替代"
`_search_single` 重写为调用 `enhance_and_select`：
- LLM 仍是节点选择决策者（保持 PageIndex 单文档优势）。
- 关键词/实体签名作为证据注入（解决浴血值类盲区）。
- **移除** `len(summary)` 重排与"硬编码 score=1.0"，回归纯 LLM 推理 + 证据接地。

### [3.4.1] 与 PageIndex 的集成点必须"干净"（用户 2026-08-11 精化）
`enhance_and_select` 精挑后返回**多个节点**（可变数量）。集成风险：若下游只读其中一段，LLM 选了 2–3 个节点却只喂 1 个，又回到信息丢失老路。
> 现状核实：当前 `_search_single` 的 `build_context_for_doc` **已遍历全部选中节点**（MD 按节点 text / PDF 按 `pages_from_nodes` 的全部页），不是"只取一个节点"。因此重点是**验证该行为被保留**，并强化答案侧融合。
1. **确认多范围取数不丢段**：`enhance_and_select` 返回多节点后，上下文组装须全部覆盖（不被任何 top-1/单段逻辑截断）。
2. **答案合成跨段融合**：把多段证据一起交给答案 LLM，prompt 显式引导"综合多处证据作答"，而非拼接后只依赖首段。
3. **验收要点**：构造"答案分散在两个节点"的用例，确认两段证据都被读取并参与合成、且答案能融合两段。

### [3.5] Agentic 多轮并发召回（召回预算的核心机制）⭐用户 2026-08-11 定向

**动机**：一次性固定 top_k 的召回，在多相关数据集上被上限压死（t2/du 有 5–8 个相关文档，top_k=5 就漏），且选错一批就只能认命。改用 **agentic 多轮召回**：**每轮并行召回 → LLM 判断"证据是否足以准确作答" → 不足则放宽召回再来一轮**，直到能作答或到上限。

这正是把现有 CRAG verifier 的 `expand` 动作升级为**可用的、多轮的检索循环**（当前 `_search_v2` 的 expand 分支是坏的，本机制取代并修复它）。

**循环结构**：

```
async def agentic_retrieve(query, max_rounds=3):
    retrieved = set()                 # 已召回文档（增量去重）
    # 逐轮递增的召回配置：轮次越深，越宽、越放松
    round_cfg = {
        1: {"top_k": 5,   "channels": [tag, keyword, entity]},
        2: {"top_k": 10,  "channels": [tag, keyword, entity, vector]},
        3: {"top_k": 20,  "channels": all, "relax_threshold": True},
    }
    for round in 1..max_rounds:
        cfg = round_cfg[round]
        # ① 多并发召回：四通道并行 + 跨文档节点召回并行（asyncio.gather）
        candidates = await parallel_recall(query, cfg, exclude=retrieved)
        # ② 高召回 union + LLM 精挑（复用 [3.2] UnifiedNodeEnhancement）
        result = await enhance_and_select(query, candidates)
        selected, pool_concern = result.selected_ids, result.pool_concern
        retrieved |= selected.doc_ids
        # ③ 组装上下文（token 预算，累积已召回）
        context = build_context(selected, budget)
        # ④a 候选池质疑 —— 显式信号，直接强制再扩一轮（不依赖答案含糊）
        if pool_concern:  continue   # 回捞延迟池 + 放宽召回，进下一轮
        # ④b LLM 作答 + 验证（复用 CRAG verifier 的 action 信号）
        answer  = generate_answer(query, context)
        verdict = verifier.verify(answer, context, query, ...)   # answer|expand|refuse
        if verdict.action == "answer":  return answer, high_conf   # 置信足 → 停
        if verdict.action == "refuse":  return refuse              # 明确无 → 停
        # verdict.action == "expand" → 继续下一轮，放宽召回
    return best_effort_answer(retrieved)   # 到上限：用累积证据尽力作答
```

**设计要点**：
1. **逐轮放宽召回宽度**（top_k 5→10→20 + 放松阈值 + 逐轮开更多通道）。第一轮"宁缺毋滥"保精度；不足就放宽保召回——**precision-first → recall-expansion** 的自适应，直接治 top_k 上限。
2. **轮内并发、轮间串行**：轮内四通道/跨文档并行（`asyncio.gather`）；轮间因有反馈依赖串行。并发收益在轮内，成本可控在轮间。
3. **LLM 判停、复用既有判据**：用 CRAG verifier 的 `action`（answer/expand/refuse）驱动停/扩，**不自造新判据**；`sufficient` 信号直接复用。**verifier 取偏严取向（召回优先）+ 多轮预算 + 延迟保护，细则见 [7.5]。**
4. **增量召回 + 去重**：每轮 `exclude` 已召回文档，避免重复消耗 token 预算。
5. **与多跳（[S8]）的关系**：multi-hop 管"**换子查询目标**"（图谱引导下一跳），recall-loop 管"**同一目标查不够**"。两者可嵌套：multi-hop 的每一跳内部用 recall-loop。
6. **预算约束**：轮内组装上下文仍受 `max_context_tokens` 约束；轮数上限 `max_rounds` 兜底，防无限循环。
7. **（v3.2）`pool_concern` 显式扩召**：扩召有**两个**触发源——①`pool_concern=true`（LLM 精挑时直接判断候选池可能不够，**不经过答案生成/verifier**，立即回捞延迟池+放宽召回）②verifier 判 `expand`。前者是**直接判据**，后者是答案层的**间接判据**；两者互补，保证"候选被超限裁掉"不会只能靠答案含糊来兜（对应 [1.2]④ 的延迟池回捞）。

**延迟与成本控制**（用户 2026-08-11 精化）：多轮串行会拉高端到端延迟，须控：
8. **第一轮尽量轻量且高精**：大部分查询应在第 1 轮就 `answer` 停掉；第 1 轮用窄 top_k + 高置信验证，避免无谓多轮。
9. **top_k 梯度需匹配语料规模**：默认 5→10→20 是拍的；语料库很大时 20 可能不够，须按历史经验/实测调参（`round_cfg` 做成可配）。
10. **（P3 可选）投机式多轮流式**：当第 1 轮上下文已足够作答时即开始流式输出，后台异步预取第 2 轮作后备；若 verifier 判 `expand` 则无缝切换到更完整的答案。复杂度较高，作为 P3 延迟优化，不阻塞 P2 主流程。

**它替代/修复**：`_search_v2` 里坏掉的 `expand` 分支（`pages_with_text2.items()` AttributeError，审计出的缺陷之一）；并把 P3 的"Adaptive 召回预算"收敛为本机制。

> **多轮陷阱的工程细化**（verifier 判据、上下文 token 膨胀、best_effort 作答）见 **[7.5] / [7.6]**。

---

## [4] 分阶段实施计划

### P0 · 正确性 bug 修复（不改架构，纯止血）

**修复顺序与理由**（按"风险/解耦度"排，先止血后打通图谱通道；位置以函数名为准，行号以实施时实际为准）：

| 序 | Bug | 位置（函数） | 修复 | 顺序理由 |
|----|-----|------|------|---------|
| 1 | `_cluster_route_boost` 把 dict 当 int 比较 → **TypeError 崩溃**（`_score_nodes` 已返回 `Dict[int, Dict]`） | super_tree.py `_cluster_route_boost` | `s.get("total_score", 0.0)` 一行改 | 一行崩溃，零耦合，先止血 |
| 2 | ContentStrategy 命中数被当 rank 喂 RRF → **语义反转**（命中越多分越低） | router.py `_run_strategies` + `_weighted_rrf` | 命中数→真实 rank（rank+1） | 一行级，恢复 v2 融合语义 |
| 3 | 多跳 `_get_candidate_docs` 返回 **DB 整数 id**，`_recall_nodes_for_doc` 按 **UUID** 查 → 图谱跨文档检索整链失效；且**从不调 CRAG verifier** | multi_hop.py `_get_candidate_docs` + router.py `_recall_nodes_for_doc` | 返回 UUID（经 `_id_mapper` 转换）+ 补 CRAG 验证 | 打通图谱多文档扩展的核心通道，稍重，排第三 |
| 4 | `_prefilter_nodes` 的 `if not scores: return all` 保召回守卫**不可达** → 零信号宽层被硬截 top-20 | super_tree.py `_prefilter_nodes` | 最小修：按"是否有任何通道命中"判断，全零信号则全量返回 | 最小修后在 P2 被 union 收窄重写取代（先救急不重写） |

> 另：审计出的**第 5 个缺陷**——`_search_v2` expand 分支 `pages_with_text2.items()` AttributeError——**不单独立项修复**，由 P2 的 [3.5] agentic 多轮召回整体取代（正确机制本身就是修复）。

**验收**：47 个现有测试全绿 + 为 Bug1/3 各补一条回归测试（中档语料树不崩、多跳候选能解析为真实文档）。

### P1 · 方案A：索引期节点本地属性富集
1. `node_profiles`：实体（归属到节点）+ 显著关键词 + 标签 + 可选向量。
2. **实体消歧/归一化（命门，必做，见 [2.2]+[7.1]）**：复用现有 `disambiguate_entity`/`normalize_entities_batch`，补 blocking 预裁剪 + 批归一分块；先做人名/机构名/专名，通用概念后置；canonical 结果写入签名。
3. **标签稳定可控（见 [7.2]）**：LLM 抽象标签 temp=0 + 增量归一锚定 + fallback 降级到关键词层。
4. 单文档节点 JSON / `node_profiles` 表落库。
5. 验收：浴血值类查询，节点签名含正确 keyword/entity；"小张/张三"类查询能命中归一实体；同文档重索引标签幂等。

### P2 · 统一检索增强机制 + Agentic 多轮召回
1. 实现 `UnifiedNodeEnhancement.enhance_and_select`（高召回 union + 证据组装 + LLM 精挑），union 效率见 **[7.3]**（纯索引查表、无 LLM、签名 O(1)）。
2. **证据呈现格式（见 [3.2.2]+[7.4]）**：节点级格式化模板 + 单节点/跨节点证据封顶 + "语料事实、按语义关联判断而非计数"的 prompt 引导。
3. **union 防爆收窄（见 [1.2]）**：上限 50–80、实体优先、超限按信号类型排序、保留 LLM 质疑"候选池过窄"触发扩召的兜底。
4. 实现 `[3.5]` agentic 多轮召回循环（逐轮放宽 top_k + CRAG verifier 判停 + 增量去重），取代 v2 expand；top_k 梯度可配以匹配语料规模。**verifier 改判据为"context 是否支撑"+固定上下文预算 replace 而非叠加（见 [7.5]）**。
5. **best_effort_answer 接地再挑选 + 降置信 + 无证据拒答（见 [7.6]）**。
6. **PageIndex 多范围集成（见 [3.4.1]，必做）**：核实多范围取数不丢段 + 答案跨段融合；补"答案分散在两个节点"用例。
7. 单文档 `_search_single` 与多文档 `_recall_nodes_for_doc`/`_navigate_level` 全部接入 `enhance_and_select`。
8. 移除 `_search_single` 的 `len(summary)` 重排与硬编码 score。
9. 宽层收窄改为高召回 union（正式取代 Bug4 的临时修）。
10. 验收：pageindex-paper 三个数据集（mldr_zh/t2/du）召回对比，目标 >0.5。

### P3 · 最佳实践增强（可选，按增益取舍）
1. Contextual 摘要（节点摘要注入文档/父节点上下文）→ 直接治"摘要丢信息"。
2. 投机式多轮流式（[3.5] 要点9）：首轮即答 + 后台预取二轮作后备，verifier 判 expand 则无缝切换。
3. 可选重排层（默认关）。
4. （Adaptive 召回预算 / CRAG expand 已并入 P2 的 agentic 多轮召回，不再单列。）

---

## [5] 与上游 spec 的关系

- **保留**：[S3] 语料统一结构树骨架、[S5] 量级自适应、[S6] 语义树导航、[S7] 图谱三件套、[S9] 上下文预算。
- **强化**：[S6] 每层"预筛→加权→精挑"中的"预筛"从"四通道分数排序"明确为"**高召回 union**"；[S8] 循环推理泛化为 [3.5] agentic 多轮召回（multi-hop 管换目标、recall-loop 管查不够，可嵌套）。
- **新增**：[2] 增强 TOC 图谱（节点属性签名）、[3] 统一检索增强抽象、[3.5] agentic 多轮并发召回。
- **纠正**：死向量通道、宽层硬截、RRF 反转、中档崩溃、多跳 id 错位、expand 分支失效。

## [6] 开放问题
- `node_profiles` 的嵌入是否默认开启？（默认关以守"无向量"；开启需 `EMBEDDING_MODEL` 且走 [vector] 可选依赖）
- 重排层是否引入？（默认不，保持"LLM 精挑唯一裁剪"）
- [3.5] 中逐轮 `top_k` 梯度（5→10→20）的具体默认值与 `max_rounds`，需按实测召回/延迟标定（`max_latency`/token 总账阈值同理，见 [7.5]）。

> 已解决（2026-08-11 拍板）：verifier 采用**偏严取向（召回优先）+ 多轮预算 + 延迟保护**，含"首轮即答快速路径"——详见 [7.5]。

---

## [7] 实现细化与风险防控（用户 2026-08-11 六问逐项落地）

> 对齐现有代码后写就。多处"待补"其实**已有基础**：实体消歧已有 `entity_extractor.disambiguate_entity`（增量）+ `normalize_entities_batch`/`_normalize_entities_llm`（批归一）；标签抽取已有 `closet_index._extract_tags`（LLM 抽象标签）+ `_fallback_tags`（jieba 兜底）。本节聚焦"现有能力 + 补什么 + 怎么控"。

### [7.1] 实体消歧如何"轻量且有效"
**现状**：`disambiguate_entity` 对**每个新实体**喂"top-20 同类型已有实体"给 LLM 判合并；`normalize_entities_batch` 按类型对**全部名字一次性**喂 LLM 归一。问题：全量 prompt 随语料增长会爆上下文；全类型无差别消歧成本高。
**轻量且有效方案**：
1. **LLM 前置轻量候选裁剪（blocking）**：不把全部实体喂 LLM。先按 ①同 `entity_type` ②**字符/别名重叠信号**（字符 Jaccard、别名精确/前缀匹配、编辑距离）预筛出**疑似同簇**，仅当轻量信号命中才调 LLM 裁定。LLM 调用量从 O(全量) 降到 O(疑似对)。
2. **批归一分块（bounded chunk）**：`_normalize_entities_llm` 对大语料按类型分组 + **每块 ≤200 个名字**分批，块内归一致 + 块间用归一后的代表元再合一轮（map-reduce），杜绝单 prompt 爆上下文。
3. **类型分期**（对齐 [2.2]）：先做 `person`/`organization`/`project`（专名，查询最常出现、最易写异），`concept` 后置。
4. **保守合并**：不确定一律不合并（`should_merge=false`，现有行为保留）——**误合比不合更伤检索**（误合会把无关内容串进查询实体）。
5. **别名累积**：合并后把变体并入 canonical 实体的 `aliases`（复用 `merge_entities`/`merge_entity_aliases`），让后续 `小张→张三` 直接走别名命中、不再调 LLM。
6. **落地位置**：消歧结果写入 canonical 映射 → [2.1] `node_profiles.entities` 引用**归一后的 canonical 实体**（而非原始字面名），否则签名仍是噪声。

### [7.2] 标签（closet_tags）的稳定性和可控性
**现状**：`_extract_tags` LLM 抽 3–5 个抽象语义标签（conf≥0.5）；`_fallback_tags` 无 LLM 时用 jieba 词兜底（conf=0.3）。风险：LLM 抽取有 run-to-run 漂移；fallback 是原词而非抽象概念，混入语义漏斗会稀释；同义标签（容器编排/容器调度）不一致。
**稳定可控方案**：
1. **确定性抽取**：抽取固定 `temperature=0` + 固定 prompt + 固定取 K=5（`llm_completion` 已 temp=0），减少抖动。
2. **标签词表锚定（增量归一）**：复用 corpus_tree 的标签归一化（`corpus_tag_norm`）。新文档抽取后**先与已有 canonical 标签比对**——语义近似则复用既有 canonical 名，不新造；只有真新概念才新增。这与 [7.1] 实体消歧的增量模式同构，保证语料级标签集长期稳定。
3. **fallback 分层降级**：`_fallback_tags` 产物 `source="fallback"`、conf=0.3——**只进关键词层，不进语义标签漏斗**（语义通道只认 LLM 抽象标签），避免原词冒充概念稀释 union 精度。
4. **置信门槛**：语义通道只取 conf≥0.5（现有）；union 收窄时按 conf 加权优先级。
5. **可控/可审计**：标签列表可导出检视（对应 capstone 里"用户可见/可纠正"的 OPEN 问题）；重索引同一文档应产出相同标签（归一锚定 + temp=0 → 幂等）。

### [7.3] `enhance_and_select` 中四通道 union 的效率
**原则**：union 是**倒排索引查表 + 集合并**，**全程无 LLM**；LLM 只在第③步精挑。
1. **只传 node_id + 轻量信号**：每通道返回"命中的 node_id 集合（+命中类型）"，不搬全量签名；union = O(各通道命中数之和) 的集合并，廉价。
2. **签名 O(1) 查表**：[2.1] 把 profile 索引期落库（节点 JSON / `node_profiles` 表），union 后逐节点取签名是 O(1)，不现场重算。
3. **高置信短路**：若实体/关键词已精确命中一个**很小**的集合（如 ≤3 个节点），可跳过价值较低的通道，直接进证据组装——省无效扫描。
4. **union 后即套 [1.2] 上限**：集合并完立刻做 50–80 上限 + 信号优先级收缩，**不给下游喂爆炸候选**。
5. **无冗余打分**：只对 union 内的节点组装证据；union 外的节点不碰。

### [7.4] 证据呈现的 prompt 模板要防"证据过载"
**风险**：把全部命中堆给 LLM，既爆 token 又诱导"数命中个数"（已在 [3.2.2] 用引导语防，但还要控量）。
1. **单节点证据封顶**：每节点最多 top-K 实体(K≈2–3)、top-K 关键词(K≈3–5)、top-K 标签(K≈2)；只列命中的、不列全签名。
2. **跨节点证据总预算**：整块证据 token 设上限；超限则按节点证据强度（实体>标签>关键词）保留强候选的富证据，弱候选退化为"标题+摘要"一行。
3. **命中去重/全局化**：同一实体/关键词命中很多节点时，不逐节点重复，改为一条全局注记（"实体X 命中于节点 a、b、c"），正文只在最相关节点展开。
4. **富-简两级呈现**：候选多时，证据最充分的少数节点给完整证据块，其余给一行摘要；必要时先一次 LLM 粗筛 shortlist、再对 shortlist 喂富证据（仅候选很多时才用，控制额外调用）。
5. **与 [3.2.2] 引导语配合**：模板控量 + "按语义关联判断、非计数"共同防过载与带偏。

### [7.5] Agentic 多轮的潜在陷阱：Verifier 准确性 + 上下文 token 膨胀
**⭐ 用户 2026-08-11 拍板：verifier 用「偏严」取向——召回优先，但必须配多轮预算 + 延迟保护。**

**(a) Verifier 判断准确性（偏严 / 召回优先）**：
1. **改判据从"答案好不好"到"上下文是否支撑答案"**：verifier 不做自评（抗自证偏见），要求**从 context 引用支撑证据片段**——答案若引用不出 context 中的实锤，就不给 `answer`。
2. **偏严阈值**：上调触发 `expand` 的敏感度——宁可多一轮把证据补齐，也不给证据不足的 `answer`。即 `tau_high` 收紧（更难判 `answer`），让证据不充分的查询**继续 expand 扩召回**，从而**提召回**。
3. **校准 + verdict 分布监控**：偏严后 `expand` 触发率上升是**预期**；但须监控——若**几乎全是 expand**（永不收敛）说明判据过严或召回无效，要在 dev 集回归调 `tau_high/tau_low`，找到"召回优先但可收敛"的甜点。
4. **失败降级保留**：verifier LLM 调用失败时回退纯启发式 `s_ret` 判据（现有行为保留）。

**(b) 上下文累积 token 膨胀（多轮预算）**：
1. **固定上下文预算，expand ≠ 叠加**：每轮上下文受 `max_context_tokens` **固定预算**约束；expand 时用更宽候选**重新挑选**预算内最相关子集（**替换**低价值段，**不是**在旧上下文上继续堆）。否则轮轮累加必爆 token。
2. **expand 的"重新挑选"必须是 LLM 决策（v3.2 修正）**：重新挑选复用 `enhance_and_select`（LLM 精挑），**不得退化成纯规则筛选**（如按实体命中数截）——否则悄悄背离"LLM 是唯一裁剪者"。实现方式：把更宽候选喂给 LLM，并告知本轮可用的节点/字数预算，让 LLM 在预算内挑；预算只是约束条件，不替 LLM 决定谁相关。（回捞 [1.2]④ 延迟池的候选也纳入本轮 LLM 视野。）
3. **轮次 token 总账（多轮预算）**：记录多轮累计 LLM token，设总成本天花板；超即提前终止进 [7.6] best-effort，防"为召回不计成本"。
4. **轮数硬上限**：`max_rounds`（默认 2–3）兜底，绝不无限扩。

**(c) 延迟保护（偏严多轮的代价必须设闸）**：偏严 → 多轮更频繁 → 延迟显著增长，三道闸门：
1. **总耗时预算 `max_latency`**：从查询开始计时；若**已用时 + 预估下一轮耗时**将超预算，不再开新轮，直接进 [7.6] best-effort。
2. **单轮超时**：每轮召回+作答设超时；超时轮视为该轮失败，降级（跳过或 best-effort），不挂死。
3. **首轮即答快速路径**：第 1 轮若 verifier 即给 `answer`（大部分查询），直接返回，不进多轮——把多轮成本只留给"第 1 轮确实不足"的查询，避免全量拖慢。

### [7.6] 达到 max_rounds 后的 `best_effort_answer`
**风险**：此时已累积大量**质量混杂**节点，全塞给 LLM 会混乱/幻觉。
1. **绝不全量喂入**：`best_effort_answer` 先对累积节点池做**一次接地再挑选**（复用 `enhance_and_select` 的证据+精挑），在上下文预算内只取 top-N 最高相关节点。
2. **降置信 + 显式标注**：输出 `confidence="low"`（[3.5] 已有），并标注"尽力作答/证据不充分"；不让混合质量证据拼出貌似可信的错答案。
3. **无证据则诚实拒答**：若再挑选后**没有任何节点**含查询实体/关键词/标签命中 → 返回诚实拒答（"未在语料中找到相关证据"），**不编造**。
4. **答案须可溯源**：best-effort 答案要求注明引用的节点，避免混合证据被无声融合成"看似完整"的错误答案。
