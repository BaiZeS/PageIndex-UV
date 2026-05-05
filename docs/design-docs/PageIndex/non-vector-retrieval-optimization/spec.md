# Feature: 非向量检索优化 (Agentic Multi-Strategy Router)

**作者**: AI + User
**日期**: 2026-05-05
**状态**: Approved

---

## 1. 背景 (Background)

同 v1 版本，见原始 spec 第 1 节。

核心问题新增：
- L1 检索仅依赖单一 LLM Description 策略，召回路径单一
- 串行 LLM 调用延迟高（~3s）
- 无答案置信度验证，无法区分"回答充分"和"证据不足"
- 无 query 扩展，原始 query 直接检索召回率受限

## 2. 目标 (Goals)

- 多文档检索延迟从串行 ~3s 降至并行 ~1.5s
- 三策略召回（Metadata + Semantics + Description）+ RRF 融合，召回率对标单文档基线
- CRAG 置信度验证，区分高/中/低置信度答案，低置信度时拒绝回答
- 零向量依赖，零新增重量级依赖

## 3. 需求细化 (Requirements)

### 3.1 功能性需求

- **FR1: Agentic 多策略路由**
  - Plan: Query 分类 + HyDE 假设答案生成 + 多 query 变体
  - Route: 并行执行三策略（Metadata/Semantics/Description）
  - Act: RRF 融合 → Tree 搜索 → 内容提取
  - Verify: CRAG 双打分 + 三阈值路由（回答/补充/拒绝）

- **FR2: Closet 语义标签索引（Semantics 策略）**
  - LLM 提取语义概念标签，jieba 分词后存入 SQLite 倒排表
  - 查询时 jieba 分词 → SQL 匹配 → 返回 Top-K 文档

- **FR3: Metadata 策略**
  - SQL LIKE 匹配文档名和描述
  - 零 LLM 开销

- **FR4: Description 策略**
  - LLM 判断文档描述与 query 相关性
  - 复用现有 `get_relevant_documents_for_multidoc()` 逻辑

- **FR5: RRF 融合**
  - 加权倒数排名融合，Semantics 策略权重可略高

- **FR6: 单文档兼容**
  - 单文档模式跳过 Plan/Route/Verify，直接 Tree 搜索

### 3.2 非功能性需求

- **NFR1: 延迟** — 多文档检索 < 2s（三策略并行后）
- **NFR2: 依赖** — 仅新增 `jieba`，无向量库/框架
- **NFR3: 兼容性** — `PageIndexClient` 现有接口不动，仅新增 `search()`
- **NFR4: 降级** — 任一策略失败不影响其他策略，全部失败可回退全库扫描

## 4. 设计方案 (Design)

### 4.1 方案概览

**Agentic 多策略路由**：Plan → Route → Act → Verify

```
query
  │
  ▼
┌─────────────┐    ┌─────────────────────────────────────────────┐
│ 1. Plan     │───→│ Query 分类 + HyDE 假设答案 + 3 query 变体    │
│ (RetrievalPlanner)│ 输出: queries[], weights{metadata, semantics, description}│
└─────────────┘    └─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Route — asyncio.gather() 并行三策略                        │
│                                                             │
│  S1 Metadata      ──→ SQL LIKE doc_name/description         │
│  S2 Semantics     ──→ jieba + closet_tags 匹配              │
│  S3 Description   ──→ LLM 判断文档描述相关性                  │
│                                                             │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────┐
│ 3. RRF 融合  │    score(doc) = Σ(w_i / (k + rank_i)), k=60
│ (加权)       │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Act — 对每个候选文档                                        │
│    get_relevant_nodes() → pages_from_nodes()                │
│    → build_context_with_budget() → generate_answer()        │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Verify — CRAG Confidence Routing                          │
│                                                             │
│  S_ret = f(context_tokens, source_docs, covered_nodes)      │
│  S_CoV = LLM 判断"答案置信度 0-1"                            │
│                                                             │
│  S_CoV ≥ 0.7  → 高置信度，直接返回答案                       │
│  0.4 ≤ S_CoV < 0.7 → 中置信度，expand search (×2 文档数)     │
│  S_CoV < 0.4  → 低置信度，返回 "I don't know."               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 组件设计

#### 4.2.1 文件结构

```
PageIndex/pageindex/
  agentic/
    __init__.py
    router.py       # AgenticRouter: orchestrate Plan→Route→Act→Verify
    planner.py      # RetrievalPlanner: query analysis, HyDE, variants
    strategies.py   # MetadataStrategy, SemanticsStrategy, DescriptionStrategy
    verifier.py     # CRAGVerifier: S_ret, S_CoV, threshold routing
  closet_index.py   # ClosetIndex: LLM tag extraction + jieba + SQLite
  client.py         # PageIndexClient: +db +router +closet +search()
```

#### 4.2.2 核心类

**AgenticRouter**
```python
class AgenticRouter:
    def __init__(self, client: PageIndexClient, model: str)
    async def search(self, query: str, top_k: int = 3) -> dict
```

**RetrievalPlanner**
```python
class RetrievalPlanner:
    def __init__(self, model: str)
    async def plan(self, query: str) -> PlanResult
    # PlanResult: queries[List[str]], weights[Dict[str,float]], query_type[str]
```

**CRAGVerifier**
```python
class CRAGVerifier:
    TAU_HIGH = 0.7
    TAU_LOW = 0.4
    def __init__(self, model: str)
    async def verify(self, answer: str, context: str, query: str) -> VerifyResult
    # VerifyResult: confidence[float], action[str] "answer"|"expand"|"refuse"
```

**ClosetIndex**（同 v1，见原始 spec 4.2.1）

#### 4.2.3 接口设计

**PageIndexClient 新增**
```python
async def search(self, query: str, top_k: int = 3) -> dict
```

返回值：
```python
{
    "query": str,
    "mode": "single" | "multi",
    "answer": str,
    "confidence": "high" | "medium" | "low" | "unknown",
    "matched_docs": [{"doc_id": str, "doc_name": str, "score": float}],
    "selected_nodes": [{"node_id": str, "title": str, "doc_name": str}],
    "pages": [{"doc_id": str, "pages": [int]}],
}
```

### 4.3 数据模型

同 v1：`closet_tags` 表定义不变（见原始 spec 4.2.3）。

### 4.4 错误处理与降级

| 场景 | 处理 |
|------|------|
| Plan LLM 失败 | 跳过 HyDE，原始 query 直接执行三策略 |
| 某策略失败 | 该策略返回空，RRF 权重归零 |
| 所有策略空 | 回退全库 Description 扫描 |
| Act Tree 搜索失败 | 跳过该文档，继续其他候选 |
| Verify LLM 失败 | 返回 Act 答案，confidence="unknown" |
| SQLite 未连接 | 跳过 Semantics，仅 Metadata + Description |
| 单文档 | 直接 L2 Tree 搜索，零 Plan/Route/Verify 开销 |
| Context 超预算 | 按 RRF 分数截断，优先保留高分文档 |

## 5. 变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `db.py` | 修改 | 新增/完善 `closet_tags` 表 + CRUD |
| `pyproject.toml` | 修改 | 新增 `jieba>=0.42` |
| `PageIndex/pageindex/closet_index.py` | 新建 | ClosetIndex 模块 |
| `PageIndex/pageindex/agentic/*.py` | 新建 | Router + Planner + Strategies + Verifier |
| `PageIndex/pageindex/client.py` | 修改 | 集成 SQLite + Router + ClosetIndex + search() |

## 6. 测试计划

1. **导入测试**：所有新模块可正常 import
2. **ClosetIndex**: add/search/remove 单测
3. **Planner**: query 分类 + HyDE + 变体生成
4. **Strategies**: 三策略各自返回结果
5. **Verifier**: 三阈值路由正确
6. **Router**: 端到端 search() 单文档 + 多文档
7. **降级测试**: 各失败场景降级路径

## 7. Changelog

| 日期 | 变更内容 | 作者 |
|------|----------|------|
| 2026-04-30 | v1: Closet + Tree 两层方案 | AI + User |
| 2026-05-05 | v2: 升级为 Agentic 多策略路由 | AI + User |
