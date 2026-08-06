# 语义树导航 · 实施计划（交接新对话）

> 日期：2026-08-04　状态：**P0-P4 全部交付**（2026-08-04 实施完成）
> **架构定稿**：`docs/compose/specs/2026-08-04-overall-architecture-semantic-tree-navigation.md`（[S1]-[S13]，唯一权威；取代更早的 4 份设计稿）。
> **实施前必读该设计文档**，本计划只是落地路线。

---

## 一、背景一句话

把 PageIndex"单文档先建树、再树上推理导航"推广到**海量多文档语料库**：语料库自动构建成一棵树（ROOT→[主题聚类0..N]→文档→节点），检索 = 在树上用语义理解逐层导航，图谱在实体/多跳处加速，规模决定导航档位。**全程无向量**（实证 PageIndex 单文档建树本就零向量）。

## 二、硬性约束（实施必须遵守）

- **D2 保持轻量**：不过度工程，每阶段最小可用。
- **D4 分工**：PageIndex-UV 只做代码/检索优化；海量/多跳 benchmark 归 pageindex-paper。
- **全程无向量**：建树（标签归一→确定性分组→LLM递归）与检索（标签/关键词预筛）都不用向量。
- **软归属不硬删**：聚类/路由绝不硬过滤删候选（H03 教训）。
- **推理挑选宁缺毋滥**：LLM 选档可变数量，不凑数（H02 教训）。

## 三、已完成（勿重复）

| 项 | commit | 内容 |
|:---|:---|:---|
| 三层重构 T9 选择层 | `8f9b659` | `_score_candidates`→map-reduce 推理选择（`_holistic_select`+`_select_documents_reasoning`） |
| 三层重构 T7 L0 | `ffda347` | 可配置通道召回 topk（`l0_channel_topk`） |
| 三层重构 T8 层级 | `ddbf136` | `_hierarchy_boost` 标签软路由 |
| **P0 上下文预算** | `a084ea9` | `_act_tree_search` 多文档上下文 token 预算（预算满即停）+ 测试 |
| **P1 语料树构建** | `f71d250` `8ee88f2` `0eb3f98` | CorpusTreeBuilder + 4 表存储 + 增量钩子 + 细而不碎（41 测试） |
| **P2 语义树导航** | `734577f` | 规模分档 + 逐层预筛/加权/精挑 + 渐进披露 + 并行展开 + P0 残留修复（38 测试） |
| **P3 图谱三件套** | `a450217` `454e258` `0a71cbc` | search_entities 分词化 + 实体消歧 + 三件套支持（34 测试） |
| **P4 循环推理** | `352582d` `4d00dc2` | MultiHopReasoner + router 集成 + 图谱引导逐跳（15 测试） |

## 四、已交付任务

### P1 语料树构建（无向量管线）✅
- **交付 commit**：`f71d250`（主体）+ `8ee88f2`（相似兄弟簇合并）+ `0eb3f98`（合并后尺寸重检 + 标签归一映射修正）
- **实现**：`pageindex_mutil/corpus_tree.py`（CorpusTreeBuilder，~720 行含 prompt），`db.py`（4 表 + ~15 方法），`client.py`（增量钩子），`tests/test_corpus_tree.py`（41 测试）
- **规格评审**：26/26 通过（[S3] + [S3.1]）
- **质量评审**：Yes
- **全套**：184 通过（P1 基线 143 + 41 新增）

### P2 语义树导航 + 量级自适应 ✅
- **交付 commit**：`734577f`
- **实现**：`super_tree.py`（+314 行：NodePrefilter / EntityBoost / SelectNodes / NavigateTree / 规模分档 / 集群路由），`router.py`（doc_id 去重 + P0 残留修复：首篇预算 + 实体上下文计入预算），`config.yaml`（分档阈值），`tests/test_tree_navigation.py`（38 测试）
- **规格评审**：19/19 通过（[S5] + [S6] + [S9]残留 + [S10]通道C）
- **质量评审**：Yes
- **全套**：219 通过

### P3 图谱三件套 ✅
- **交付 commit**：`a450217`（主体）+ `454e258`（消歧管线集成）+ `0a71cbc`（别名合并补全 + 查询 LIMIT）
- **实现**：`db.py`（search_entities 分词化 + get_entities_by_type + merge_entity_aliases），`entity_extractor.py`（disambiguate_entity），`client.py`（_resolve_entity 管线集成），`tests/test_entity_graph.py`（34 测试）
- **规格评审**：21/21 通过（[S7]）
- **质量评审**：Yes
- **全套**：254 通过

### P4 循环推理多跳 ✅
- **交付 commit**：`352582d`（主体）+ `4d00dc2`（matched_docs 填充 + 参数清理 + prompt 精简）
- **实现**：`pageindex_mutil/agentic/multi_hop.py`（MultiHopReasoner，315 行），`router.py`（search() 集成），`tests/test_multi_hop.py`（15 测试）
- **规格评审**：13/13 通过（[S8]）
- **质量评审**：Yes
- **全套**：269 通过
- **注意**：多跳效果验证需 pageindex-paper 先建多跳基准（D4 分工）

## 五、依赖关系

~~`P0(已完成) → P1 → P2 → P3 → P4`~~ — **全部交付完成**。最终全套 269 测试通过，0 失败。多跳效果验证待 pageindex-paper 建多跳基准。

## 六、验证与测试约定

- 单测用现有 `tests/` 的 `importlib.util.spec_from_file_location` stub 模式（见 `test_router.py`/`test_super_tree.py`/`test_corpus_tree.py`）。
- 每阶段跑 `uv run pytest tests/...`（排除需网络的 `test_search_backends`、并发的 `test_db_concurrency`）。
- 检索效果验证归 pageindex-paper（benchmark），PageIndex-UV 侧只做代码+单测。
- **最终状态**：269 测试通过（P0 基线 143 + P1 41 + P2 38 + P3 34 + P4 15），每个任务均通过独立规格评审 + 代码质量评审双重门控。