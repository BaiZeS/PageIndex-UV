# 文档归档（历史存档）

本目录存放**已被取代或已完成使命**的设计/规划文档，仅作历史存档，**不是当前架构依据**。

## 当前权威文档（不在本目录）

检索架构请以这两份为准：

| 文档 | 状态 |
|:-----|:-----|
| `docs/compose/specs/2026-08-04-overall-architecture-semantic-tree-navigation.md` | **总体设计定稿**（语义树导航，[S1]–[S13]，已实现） |
| `docs/compose/specs/2026-08-11-enhanced-toc-graph-unified-retrieval.md` | **增强 TOC 图谱 + 统一检索增强**（增量优化方案，讨论稿，规划中） |
| `docs/architecture.md` | 当前**已实现**架构的用户向概览 |

## 取代关系（supersession chain）

以下文档的结论已并入上述定稿，被明确取代，避免混淆故移入此处：

| 归档文档 | 被谁取代 | 说明 |
|:---------|:---------|:-----|
| `rag-quality-optimization-options.md` | 语义树导航定稿 | Q1/Q2/Q3 质量优化的早期评估，结论已并入定稿（Q1 打分精排后又被 `_select_documents_reasoning` 取代） |
| `rag-progressive-disclosure-options.md` | 语义树导航定稿 | 渐进披露三形态评估，其中"形态3 递归树导航"即定稿的语义树导航 |
| `2026-08-03-massive-rag-three-layer-architecture.md` | 语义树导航定稿 | 最早一版海量多文档三层架构稿 |
| `2026-08-04-corpus-tree-plan.md` | 语义树导航定稿（[S3]） | 语料树规划稿，内容并入定稿 [S3] |
| `2026-08-02-q1-l1-rerank.md` | （已完成，归档） | Q1 L1 打分精排的实施计划，已交付并被后续选档重构取代 |
| `2026-08-04-semantic-tree-navigation-implementation.md` | （已完成，归档） | 语义树导航 P0–P4 的实施路线，已全部交付 |

> 若你查阅的是"当前怎么做"，请勿引用本目录内容；它们反映的是历史推演过程。
