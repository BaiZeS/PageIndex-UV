# 语义树导航 · 实施计划（交接新对话）

> 日期：2026-08-04　状态：P0 已完成，P1-P4 待实施
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

## 四、待实施任务（按序）

### P1 语料树构建（无向量管线）——下一站
目标：索引期自动把语料建成树（设计文档 [S3]）。
管线（[S3]）：
1. 每篇索引抽 closet_tags（已有 `closet_index._extract_tags`）。
2. **标签归一化**：收集全库唯一标签→LLM 合并同义→规范标签集（修复 风控/风险管理 不一致）。【新增】
3. 规范标签建 tag→docs 倒排→确定性 group-by 分组（不走 LLM）。【新增】
4. LLM 递归生成上层结构（`generate_toc_init/continue` 模式，`page_index.py:528-616` 可借鉴：continue 528-560 / init 563-595 / 循环用法 `process_no_toc` 597-616）：定名+摘要+组织上下级+裁定合并/拆分。【新增】
5. 组装层级+软归属；增量更新。
- **细而不碎**（[S3.1]）：簇大小双向卡界（过小合并、过大拆分），合并由 LLM 语义裁定。
- **增量两点**（设计文档 [S3]）：①新文档标签先匹配已有规范集，不中再 LLM 单点裁定并入/新开（不重跑全库归一）；②簇卡界**每次插入时评估**（超上限→拆分，低下限→并入/上提），不做周期性全量重建。
- 新增存储：语料树表/字段（corpus_tree）。
- 验证（PageIndex-UV 侧单测级验收；召回@规模实证按 D4 归 pageindex-paper）：
  1. 产出可检视语料树，结构合理；
  2. 文档覆盖率 100%（每篇至少挂 1 簇，软归属可多挂）；
  3. 簇大小分布落在卡界区间（越界簇有合并/拆分处置记录）；
  4. 标签归一一致性（规范集内无同义标签并存，如 风控/风险管理）。

### P2 语义树导航 + 量级自适应
- 用树导航统合现有管线（设计文档 [S5]+[S6]）：每层=语义预筛(标签/关键词)+图谱加权+LLM精挑；渐进披露。
- 量级自适应档位（小直连/中单层/海量层级树）。
- 改造 `super_tree.py` prefilter/`_hierarchy_boost`/`select_documents`。
- 验证：longdoc 基准对比（pageindex-paper）。

### P3 图谱三件套
- **前置**：`db.search_entities` 从整串 LIKE 改为分词匹配（jieba 查询词 vs name/aliases）——现状多词查询几乎永不命中（设计文档 [S7.2]）。
- 实体消歧（对齐标签归一化思路，修复 张三/小张 不链接；现有 `_fuzzy_match` 仅单文档内，跨文档归一需新建）。【补 `entity_extractor`/`db.insert_entity`】
- 三件套（[S7.2]）：①实体快捷跳转 ②预筛信号加权 ③多跳导航。

### P4 循环推理多跳
- 推理-检索循环（[S8]）：可分解查询→逐跳导航+图谱引导下一跳。
- **需 pageindex-paper 先建多跳基准**才能验证。

## 五、依赖关系

`P0(已完成) → P1 → P2 → P3 → P4`；P3 依赖 P1/P2；P4 依赖 P2+P3+多跳基准。

## 六、验证与测试约定

- 单测用现有 `tests/` 的 `importlib.util.spec_from_file_location` stub 模式（见 `test_router.py`/`test_super_tree.py`）。
- 每阶段跑 `uv run pytest tests/...`（排除需网络的 `test_search_backends`、并发的 `test_db_concurrency`）。
- 检索效果验证归 pageindex-paper（benchmark），PageIndex-UV 侧只做代码+单测。