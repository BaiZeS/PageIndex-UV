# 文档归档（历史存档）

本目录存放**已被取代或已完成使命**的设计/规划文档，仅作历史存档，**不是当前架构依据**。

## 当前权威文档（不在本目录，以 `docs/README.md` 为准）

| 文档 | 状态 |
|:-----|:-----|
| `docs/compose/specs/2026-08-13-unified-single-path-evidence-bundle.md` | **当前架构 spec（v1.5）**——统一体·单链版，权威 |
| `docs/compose/plans/2026-08-13-unified-single-path-implementation.md` | 主实施计划（T1–T18） |
| `docs/architecture.md` | 已实现架构的用户向概览 |

## 取代关系（supersession chain）

```
2026-08-04 语义树导航 capstone
   └─► 2026-08-11 增强 TOC 图谱 + 统一检索增强（v3.3）
         └─► 2026-08-13 统一体·单链版（v1.5，当前权威）
```

| 归档文档 | 被谁取代 / 状态 | 说明 |
|:---------|:----------------|:-----|
| `2026-08-04-overall-architecture-semantic-tree-navigation.md` | 08-11 → 08-13 | 语义树导航 capstone 定稿（[S1]–[S13]），已实现后经两轮收敛 |
| `2026-08-11-enhanced-toc-graph-unified-retrieval.md` | 08-13 | 增强 TOC 图谱 + 统一检索增强（v3.3），推理阶段被 08-13 收敛，索引阶段/缺陷修复细节保留于此 |
| `2026-08-14-l0-evidence-bundle-unification.md` | （已完成，归档） | L0 证据束统一 + 节点级证据直通（T1–T5/T4）实施计划，已全部交付并推送 |
| `2026-08-04-semantic-tree-navigation-implementation.md` | （已完成，归档） | 语义树导航 P0–P4 实施路线，已全部交付 |
| `2026-08-04-corpus-tree-plan.md` | 语义树导航定稿（[S3]） | 语料树规划稿，内容并入定稿 [S3] |
| `2026-08-03-massive-rag-three-layer-architecture.md` | 语义树导航定稿 | 最早一版海量多文档三层架构稿 |
| `2026-08-02-q1-l1-rerank.md` | （已完成，归档） | Q1 L1 打分精排实施计划，已交付并被后续选档重构取代 |
| `rag-quality-optimization-options.md` | 语义树导航定稿 | Q1/Q2/Q3 质量优化早期评估，结论已并入定稿 |
| `rag-progressive-disclosure-options.md` | 语义树导航定稿 | 渐进披露三形态评估，"形态3 递归树导航"即定稿语义树导航 |

> 若你查阅的是"当前怎么做"，请以 `docs/README.md` 指向的权威文档为准；本目录内容反映的是历史推演过程。
