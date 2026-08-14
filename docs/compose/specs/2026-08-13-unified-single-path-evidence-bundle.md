# 统一体·单链版：两阶段检索架构（证据束直通 + 裁定阶段扩大重试）

> 版本：v1.5（2026-08-13，讨论定稿 + 两轮独立评审闭环 + 统一 span locator + 系统级性能清单 + 孤儿代码终清）
> v1.5 变更：孤儿代码终清——`_build_docs_info`、strategies.py 四策略整模块、reasoning get_relevant_* 三函数入删除清单；multi_hop/planner 保留状态明确；vector 通道防重复计分；thinking 配置项。
> v1.4 变更：统一 span locator（MD 行号跨度，消除 page/type hack）；[S11] 补系统级性能清单（SQLite 连接复用/实体 N+1 批量/doc_summary 并行/PDF 单遍解析/liteparse 降级路径/jieba 可选）。
> v1.3 变更（复审 general-2 七项闭环）：`_build_super_tree` 入删除清单、doc_summary 迁移/兜底语义、config 漏键补齐、语料树删除边界、测试迁移计划、排序键措辞统一、reduce 阈值同源。
> 上游依据：`../../archive/2026-08-11-enhanced-toc-graph-unified-retrieval.md`（v3.3，已归档）
> v1.1 变更：独立评审（general-1）13 项问题全部闭环——keep 上限修复落位、verifier 上下文截断、matched_docs 分数口径、上下文排序、删除/保留与 config key 清单补全、P2 回滚策略、CTE tie-break、no-db 行为、评测口径钉死。
> v1.2 变更（用户拍板）：语料树不删除而**简化替代**——删除聚类层级构建，改产**文档级接地摘要**，作为 L1 候选呈现单元（结构树向上生长到 L1）。
> 本文档是对上游 spec 推理阶段的**架构收敛**：统一单/多文档与大小语料为一条路径、L0 证据束直通 L1、裁定阶段扩大重试。上游骨架（增强 TOC 图谱、四通道、enhance_and_select、verifier 偏严）保留，推理阶段的路径分支与滑窗机制被取代。
>
> 用户拍板记录（2026-08-13 讨论，按序）：
> ① 统一体方向 = 结构树为骨架、四通道为感知器、图谱为关联神经；
> ② L0 输出携带通道来源与图谱关联，直通 L1 prompt；
> ③ 决策契约 = **只直通不约束**（不设保底席位；评测数字是唯一裁判）；
> ④ 图谱遍历改 SQLite 递归 CTE；层间 reasoning trace（L1 选中理由下传）；
> ⑤ 统一单/多文档、大小语料同路径、无分支，架构只分索引阶段与推理阶段（推翻 2026-08-10 "单文档走多文档路径不可行" REJECTED 拍板——L1 候选=1 短路使成本≈0）；
> ⑥ CRAG 扩召重试从"作答后滑窗"改为"从 LLM 裁定阶段扩大重试"；
> ⑦ 最终 CRAG 不用滑窗，LLM 从补充候选清单自行点名需要补的文档（已到节点→page 粒度、范围小）。
>
> 已测事实（pageindex-paper，n=10）：BM25 R@5 = mldr_zh 0.80 / t2 0.61 / du 0.65；旧实现(ab24f1b) = 0.40/0.34/0.42；新实现修复前 = 0.20/0.35/0.50。L0 正文修复（1eb6cf3）后 e2e 复证：**L0 命中后全链路仍可 MISS——L1 选档 LLM 随机性独立丢档**。

---

## [S1] 问题与诊断

小规模测试集上全链路打不过 BM25。经代码实证，丢档点按因果排序：

1. **证据源错位（主因）**：L0 找到文档靠 doc_keywords 正文 BM25，但 L1 看到的证据是 `_doc_evidence_lines` 从 node_profiles（TF-IDF top-5，`node_keyword_topk: 5`）另起炉灶重算的——找到它的信号恰好是决策者看不到的信号；零命中文档连证据行都没有，纯靠标题+摘要被裁量。
2. **预算 pop 静默丢档**：`_holistic_select` 超 6000 token 时从末尾逐个 pop 文档（L0 分数序），无回捞；>10 候选 map-reduce 分组 map 阶段每组只留 3 篇。
3. **L1 prompt 自相矛盾**：同时写"基于标题摘要判断"、"证据命中是强信号但不因命中硬选"、"可少选甚至不选"——等于告诉 LLM 证据仅供参考、判不准就少选。
4. **判错无兜底**：L1 判错（选非空但错）无任何回退；仅判空走 `_content_fallback` 且 answer 为空串。
5. **`keep=5` 上限**：多相关数据集（5–8 篇相关）天花板即 5 篇。

结论：瓶颈在 LLM 裁定层（L1 选档 + 节点选择），不是 L0 召回、不是图谱性能。本 spec 针对裁定层重构。

## [S2] 设计总纲

架构只分两个阶段，推理阶段单链无分支：

- **索引阶段**（唯一写入端）：解析 → 结构树 → node_profiles（实体指纹/TF-IDF 关键词/标签）→ doc_keywords（正文 BM25）→ 实体图谱 → **文档级接地摘要**（见 [S3]）。全部异步并行。
- **推理阶段**（唯一入口）：证据束构建 → L1 文档裁定 → 节点裁定 → 上下文组装 → 作答 → CRAG 验证（扩大重试）。

统一体三部件：**结构树为骨架**（层级推理保留）、**四通道为感知器**（只产证据不产裁决）、**图谱为关联神经**（距离/关系类型作为证据注记）。

边界纪律（承接上游、保持不变）：
- LLM 仍是唯一裁剪者；四通道不做过滤、不打分裁决。
- **决策契约 = 只直通不约束**：证据束只进 prompt，不设保底席位、不设规则回退重排。数字不动时此契约重开（见 [S13]）。
- 不合并类：router（编排）与检索引擎（super_tree/enhance）职责边界保持。
- 重排层默认不上；关键词禁作过滤条件；降级路径（[7.7]）语义保留。

## [S3] 索引阶段

与现状一致的完整索引，变更一处：

- `entity_relations` 已有 subject_id/object_id/predicate 索引（db.py:216-218），无需迁移。
- 节点签名、doc_keywords、closet_tags、node_profiles 的生成逻辑不动。
- **语料树简化替代**（用户 2026-08-13 拍板）：删除聚类层级构建（LLM 聚类 + 簇命名——评审实证在统一单链下零推理消费者）；改产**文档级接地摘要 `doc_summary`**（索引期 LLM 生成：覆盖主要章节范围 + 关键实体/概念，复用并增强现有 `doc_description` 生成管线，三入口 page_index.py:1208 / page_index_md.py:436 / page_index_liteparse.py:101 均已具备）。**语义 = 结构树向上生长到 L1**：L1 的候选呈现单元 = 文档摘要节点，L2 单元 = 章节节点，L3 = 页面。
- **doc_summary 落库与兜底**：新增 `documents.doc_summary` 列（ALTER 迁移，复用现有 ALTER 范式 db.py:264-268）；不覆盖 doc_description（其消费者 closet_index 抽标签/entity_extractor/router 元数据/KB identity 均不受影响）；**已索引旧行 doc_summary 为空时 L1 呈现回退 doc_description**。
- `corpus_tag_norm` 标签锚定保留（写入者 ClosetIndex._anchor_tags 独立幸存）；语料树表结构保留（迁移安全）。
- **统一 span locator**：每个节点带统一跨度 `{"kind": "page"|"line", "start", "end"}`——PDF 沿用 start_index/end_index（kind=page）；MD 已有 `line_num`（page_index_md.py:332），build 时 `end_line` 已算出（:225-229），补存 end_line（kind=line，零额外解析成本）。doc_summary 的"章节范围"、L1→L2→L3 三级骨架均落在统一 span 之上。

## [S4] 推理阶段单链

```
search(query)
  ├─ ① 证据束构建（四通道原始命中 + 图谱 CTE，[S5][S9]）
  ├─ ② L1 文档裁定（[S6]）
  │     └─ pool_concern → 放宽候选池重裁定（deferred 回捞，至多 1 次）
  ├─ ③ 节点裁定 enhance_and_select（[S7]，含 L1 理由 trace）
  │     └─ pool_concern → 放宽节点池重裁定（至多 1 次）
  ├─ ④ 上下文组装（token 预算，不因预算静默丢候选证据摘要）
  ├─ ⑤ 作答
  └─ ⑥ CRAG 验证（[S8]）
        ├─ answer → 返回（高置信）
        ├─ expand → LLM 从补充候选清单点名（need + 理由）→ 只补点名对象
        └─ refuse → 诚实拒答
```

单链适配所有规模：候选=1 短路（单文档成本≈0）；规模差异内化为通道 top-k、union cap、分组参数——不再有 small/medium/massive 三档分支。

**locator 统一消费**：下游一律按 span kind 分派、不再按 doc type hack——`pages_from_nodes`→`spans_from_nodes`；上下文组装 PDF 走 page_map、MD 直接用节点 text（行切片解析期已完成）；`_node_payload`/UI 输出 span + 标题链。

## [S5] 证据束数据契约

`prefilter` 改造：从返回 `{db_id: float}` 改为返回证据束（查询级一次性构建，L1/L2/扩召共享同一实例）：

```
{
  db_id: {
    "channels": {
      "tag":     [{"text", "confidence"}],
      "keyword": [{"token", "field": "name|description|node_title|content", "bm25_score"}],
      "entity":  [{"name", "type", "confidence"}],
      "vector":  [...]  # 可选，仅真向量后端（SEARCH_BACKEND=hybrid/chroma）启用；
                        # keyword no-op 后端不进 vector 通道（防与 keyword 通道重复计分）
    },
    "graph": {
      "doc_entity_links": [{"entity", "distance", "relation_type", "weight"}]
    }
  }
}
```

要点：
- keyword 命中的 `field` 来源可追溯——正文 BM25 命中（content 字段）必须进证据块，封 [S1] 丢档点①。
- graph 关联由 CTE 一次算出（[S9]），单链全链共用。
- 证据束是 query 级缓存对象：query tokens、query entities、图谱距离一次计算，全链引用，消除各层重复解析（reasoning trace 的主机制）。
- 派生排序标量：证据束按通道命中合成一个文档级证据分（仅用于分组序与补充清单序，不参与裁定），**matched_docs 分数与上下文组装排序统一为：主键证据分（通道命中加权标量）、次键 L1 裁定序（rank）**（替代现行覆盖度分，见 [S6]#8、[S8]）。

## [S6] L1 文档裁定

`_holistic_select` 改造：

1. **候选=1 短路直通**（确定性，不调 LLM）。
2. **证据束直通 prompt**：候选呈现单元 = **文档级接地摘要（doc_summary）+ 证据块**（结构化、token 带来源字段）；替换 `_doc_evidence_lines` 的 node_profiles 重算与 `_build_super_tree` 的 top-nodes 结构（L1 呈现更轻，token 压力同步下降，缓解 [S1]② 预算 pop）。
3. **prompt 修正**（封 [S1] 丢档点③）：删除"基于标题摘要判断"的引导；证据段改为"命中是语料事实，优先依据证据与问题的语义关联判断"；保留"宁缺毋滥"但删除"可少选甚至不选"的判空授权句。
4. **预算超限先裁摘要不裁文档**（封 [S1] 丢档点②）：单次裁定内 token 超限时，优先截短摘要、再退化弱候选为一行证据摘要，**不静默 pop 文档**。
5. **分组 map-reduce 保留**（>10 候选），组内 keep 参数可配（跨组裁剪由分组机制承担，与 #4 的单次裁定内预算语义互不替代）；**reduce 阶段阈值与终选 keep 同源**（均取 `l1_select_keep`）。
6. **终选 keep 上限修复**（封 [S1] 丢档点⑤）：`selected[:keep]` 默认从 5 提为 **10（`l1_select_keep` 可配）**，对齐评测 R@10 口径；多相关场景天花板同步抬升。
7. **输出扩展**：返回 `selected_ids + reasons: {doc_id: 一句话选中理由}`（供 [S7] trace；理由缺失时留空，不阻塞）。
8. **matched_docs 分数口径**：= 证据分（通道命中加权标量），同分按 L1 裁定序排列；替代现行"节点覆盖度分"（router.py:302 / client.py:1268 的 selected/total——是评测排序噪声，R@5/R@10 是唯一裁判，排序分必须承载相关度语义）。

## [S7] 节点裁定 enhance_and_select

- 输入增加 `l1_reasons`（该文档的选中理由）：prompt 增加"上级选档依据（供参考，可推翻）"段，**明确标注为判断而非事实**——L2 以本层证据为准，理由不得替代证据（防锚定）。
- 证据来源切换为证据束（节点级命中从证据束/结构签名解析，正文内容通道 P2.6 保留）。
- `pool_concern` 重试保留现有语义（deferred 回捞 / force-all，至多 1 次），实现收敛到共享助手。
- **（P3 可选）节点理由下传**：选中的每个节点附带一句话理由，随上下文注记一并提供给作答与 verifier（evidence_quote 定位辅助）——需扩展 enhance 的 LLM 输出契约，作为 P3 项（不阻塞 P1/P2 主链）。

## [S8] CRAG 验证与扩大重试

- verifier 判据不变（偏严、上下文支撑 + evidence_quote），输出扩展：
  `{action: "answer"|"expand"|"refuse", need: [{"doc_id"|"node_id", "page"(可选), "reason"}], confidence}`
- **verifier 输入修正**：`context[:2000]` 硬截断（verifier.py:80）在多文档上下文下让 evidence_quote 只能引用首 2000 字、系统性误触 expand/refuse——改为可配 `verifier_context_chars`（默认 8000），且上下文按证据分排序组装（见下），保证支撑段可见。
- **expand = 裁定证据不足**，两个触发源收敛：
  1. 裁定层 `pool_concern`（L1/节点，作答前，至多各 1 次）；
  2. CRAG `expand`（作答后）→ verifier 从**补充候选清单**（deferred 池 + 证据束未选中文档的通道命中摘要）**LLM 点名**补召回，只补点名对象，重跑节点裁定 + 补上下文。
- **上下文组装排序**：主键证据分、次键 L1 裁定序（替代 router.py:356 的覆盖度分排序），与 matched_docs 同口径（[S5]）。
- **删除滑窗机制**：recall_loop 的 `_cut_candidates` 融合序滑窗与顺序回捞删除（回归 spec v3.2 [7.5]b "重新挑选必须 LLM 决策"原旨）。
- 三道闸保留：max_rounds / max_latency / token 总账；超限进 best_effort（接地再挑选 + 降置信 + 无证据拒答）。

## [S9] 图谱递归 CTE

- 新增 `db.py`：`get_entity_distances_cte(query_entity_ids, max_hop=3)`——`WITH RECURSIVE` 单次无向 BFS（subject/object 双向 join，索引已备）。
- 语义与 `_precompute_entity_distances`（super_tree.py:928-999）**逐项等价**：hop-min 距离确定性、距离衰减（0/1/2/3 → 1.0/0.7/0.4/0.2）× 关系类型权重（causal 1.0 / part_of 0.8 / related_to 0.6 / other 0.4）。
- **tie-break 钉死**：同 hop 多路径时取最大权重；再平取最小 entity_id（确定性）。P1 验收含与现行 BFS 输出的逐项对照测试——不一致处按 BFS 现行语义对齐或显式记录差异。
- SQLite 递归 CTE 跑到不动点、无深度上限，3 跳 BFS 无风险。
- 结果进证据束 `graph.doc_entity_links`；替换 Python 层 BFS，N+1 查询 → 1 次 SQL。
- 定位声明：CTE 是管道效率件，不是 recall 杠杆；跨查询物化留给索引期指纹（方向3 后续项，不在本 spec 范围）。

## [S10] 删除清单与保留清单

删除（P2 阶段执行，删除前以现有测试为回归网）：

| 删除项 | 位置 | 替代 |
|---|---|---|
| `_search_single` 分支 | client.py:1092-1094 | 单链（L1 候选=1 短路） |
| `_search_v2` + `_weighted_rrf` + RRF 融合池 | router.py:696-858, 114-129 | 单链 |
| tier 三档分支 + `navigate_tree` 推理路径 | super_tree.py:770-795, 1129+ | 单链 + 证据束容量参数 |
| multi_hop 前置门（LLM 判定先行） | router.py:866-873 | 多跳降为循环内子查询改写（P3）；图谱无命中回落主路径 |
| description 策略（LLM 判档）主路径 | strategies.py:143+ | L1 本身即判档；保留为显式兜底 |
| recall_loop 滑窗 `_cut_candidates` | recall_loop.py:264-313 | LLM 点名（[S8]） |
| `_content_fallback` answer 空串 | router.py:469-522, 586-607 | 证据束空时走单链 + 正常作答 |
| `_doc_evidence_lines` node_profiles 重算 | super_tree.py:510-609 | 证据束直通（[S6]） |
| `_build_super_tree`（top-nodes 结构 + 子节点计数 SQL，两调用方均删除后零消费者） | super_tree.py:1265+ | 删除；`get_top_level_nodes` 保留（KBIdentity/_backfill 使用） |
| `_hierarchy_boost` 软打分重排 | super_tree.py:732-768（调用点 783/794） | 删除——标签命中信息已进证据束，软打分与"只直通"契约冲突 |
| `_cluster_route_boost` | super_tree.py:1238+（调用点 792） | 随 tier 删除 |
| `_select_tag_sets` / `_score_candidates`（死代码，零调用） | super_tree.py:365+ / 422+ | 删除 |
| 语料树聚类层级构建（LLM 聚类+簇命名） | `CorpusTreeBuilder.rebuild`/`update_for_document`/`_build_upper_structure`/`_merge_similar_siblings`/`_enforce_size_bounds`/`_split_if_oversized`（corpus_tree.py，经 client.py:675/:916 触发） | 简化替代：文档级接地摘要 doc_summary（[S3]）；`resolve_new_tag`/`corpus_tag_norm` 读写保留 |
| 三处 page/type hack | router.py:294-298（`if not pages: if pdf`）、router.py:449-462（pages_with_text MD 空分支）、recall_loop.py:48（`_node_payload` start_index 判断） | 统一 span kind 分派（[S4]） |
| `_build_docs_info`（两消费者 `_search_v2`/`recall_loop._route` 均删后孤儿） | router.py:61-106 | 删除 |
| strategies.py 四策略整模块（Metadata/Semantics/Content/Description——`_run_strategies` 删除后零消费者；其语义已被证据束通道吸收，description 的兜底语义由 L1 零信号直通覆盖） | strategies.py 全文件 | 删除（节点级关键词上下文窗口不复用——enhance 正文内容通道已覆盖） |
| reasoning `get_relevant_nodes` / `get_relevant_pages`（零调用死代码）/ `get_relevant_documents_for_multidoc`（仅 DescriptionStrategy 消费） | reasoning.py:98/127/212 | 随 strategies.py 删除 |

保留：enhance_and_select 原语、verifier 偏严判据、三道闸、best_effort、[7.7] 降级语义、node_profiles/标签/实体索引期生成、KB identity、`corpus_tag_norm` 标签锚定、**planner（P3 HyDE 按需件）、multi_hop 组件（`_extract_intermediate`/`_guide_next_hop` 供 P3 循环内子查询改写复用；P2 仅断接前置门，不删文件）**。

**config 键迁移清单**：
- 删除：`rank_k`、`score_ratio`、`select_top_k`（仅死代码 `_score_candidates`/终选 keep 使用，前者删除、后者被 `l1_select_keep` 取代）、`hierarchy_boost_weight`、`scale_small_max_docs`、`scale_massive_min_docs`、`narrow_layer_max`、`entity_boost_weight`（随 tier/navigate 删除）、`max_top_nodes_per_doc`、`summary_max_len`（仅 top-nodes 结构消费）。
- 保留：`reason_group_size`、`reason_keep_per_group`、`l0_channel_topk`、`union_max_candidates`、`evidence_max_chars`、`agentic_*`（三道闸）、`node_keyword_topk`、`tau_*`；`max_super_tree_tokens` **重定义**为 L1 呈现预算（doc_summary + 证据块，[S6]#4 的 token 上限概念沿用此键）。
- 新增：`l1_select_keep`（默认 10）、`verifier_context_chars`（默认 8000）、`cte_max_hop`（默认 3）。

## [S11] 性能优化

1. **快速路径自然化**：单链默认无 multi_hop 判定、无 HyDE（两者均不前置调用）——简单词面查询不再付出前置 LLM 调用成本（原 6–9 次串行调用降至 4–6 次：L1 + 节点 + 作答 + verifier）。
2. **跨文档节点选择批处理（P3 可选）**：L1 选中多篇时合入一次 LLM 精挑调用（同 prompt 分节），超预算回退分篇。
3. **图谱 CTE**（[S9]）。
4. 延迟/轮数/token 三道闸沿用（[S8]）。
5. **SQLite 连接复用（P1，审查实证已实现）**：db.py:40-75 已用线程本地复用 + WAL + row_factory=Row（每线程一次建连）——P1 仅补回归测试固化现状，无代码变更。
6. **实体通道 N+1 批量（P1）**：`get_entity_documents` 每实体一查（super_tree.py:342-347）→ 单条 `WHERE entity_id IN (...)` 批量。
7. **doc_summary 并行（P2）**：进 `_enrich_document` 现有线程池，与标签/实体提取同批并发。
8. **PDF 单遍解析（P3）**：pymupdf 一次出 TOC + 页文本，砍 PyPDF2 二次打开（client.py:467-472, 729-732）；liteparse 仅作无书签/扫描件/乱码文本层的降级路径，同文档集评测对比节点划分质量后再决定是否扩大。
9. **jieba 可选加速（P3 配置项）**：正文索引分词可配 HMM=False 或 `jieba.enable_parallel`，评测验证词面精度后方启用。
10. **检索链 thinking 开关（P3 配置项）**：检索类 LLM 调用统一可配 thinking_disabled（现 closet_index 已关闭、其余开启，口径不一）；关闭后成本/延迟下降，评测验证裁定质量后方默认开启。

## [S12] 分阶段实施与验收

- **P1 证据束 + CTE + trace**：新机制落地，不动路径结构、不删旧路径。验收：证据束含正文命中（field=content）、CTE 结果与 BFS 逐项等价（含 tie-break 对照测试）、全套件以实施时 HEAD 实测为准（当前基线 614 passed/3 env）。
- **P2 路径统一 + BM25 合一 + 删除清单执行**：单链成为唯一入口；`_search_single`/`_search_v2`/tier/multi_hop 门/滑窗删除，语料树聚类构建简化为文档级接地摘要（[S3]），节点统一 span locator（[S3]/[S4]），page/type hack 消除（[S10]）。执行纪律：**按删除项分步提交，每步全套测试**；评测跑完后隔离旧结果再重启 runner（沿用 DashScope 挂起教训的三件套）；回滚 = git revert，不保留旧路径代码副本（A/B 对比对象是 harness 的 BM25/旧实现基线，无需在库内双路径）。**测试迁移计划**：删除 test_super_tree_tag_sets.py（整文件）、test_tree_navigation.py（navigate/_cluster_route_boost 已删）、test_corpus_tree.py 中 rebuild/update 相关用例（resolve_new_tag 用例保留）、test_super_tree.py 的 test_score_candidates_*/test_build_super_tree_*、strategies 相关单测（`_run_strategies` 语义已入证据束通道）；planner 单测保留（P3 HyDE）；单链语义新增/改写回归用例（候选=1 短路、LLM 点名扩召、证据束直通、MD span locator）。
- **P3 快速路径门 + 节点批处理 + 多跳循环内改写**：性能件。验收：延迟对比记录。
- **评测口径（P2 后，钉死）**：`SEARCH_BACKEND=keyword`（无向量），n=10 三数据集 Recall@k，对比 BM25（0.80/0.61/0.65 R@5）与旧实现（0.40/0.34/0.42）；沿用用户路线判据（mldr R@10 ≥0.5 收尾 / <0.4 需重开 [S13] 契约）。

## [S13] 边界与未决

- **决策契约**：只直通不约束。若 P2 评测数字不动（e4f919a 先例），重开契约讨论（保底席位/判空回退）。
- **no-db 模式**：单链依赖 db（doc_keywords/实体图 CTE）；无 db 的 workspace-only 多文档维持现状（"Router not available"），不在本 spec 范围。
- 方向6（骨架立即可搜 + 后台增强）撞 2026-08-10 "全量索引质量优先"拍板，未重拍，不在本 spec。
- 方向9（CRAG/用户反馈回流索引）降级为观测回流，未定稿，不在本 spec。
- 索引期实体指纹物化（方向3 后半）为后续项，不在本 spec。
- 快速路径门中"强词面命中"的判定阈值待 P3 实测标定。
- 多跳能力空窗期声明：P2 删除前置门至 P3 循环内改写落地之间，多跳类查询由单链 best-effort 作答——预期影响为正（现行前置门误路由是实测噪声源之一），评测需观察 t2/du 多跳样本。
- 海量语料语义路由声明：语料树层级路由移除后，海量场景靠通道 top-k（30/通道）+ union cap + 分组 + 文档级摘要承载 L1 缩窄；其语义路由质量未被评测过（评测仅 n=10 小语料）。若海量评测暴露短板，语料树层级（或文档摘要之上的合成层级）作为 P3 通道回归。
