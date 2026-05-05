# 实施任务清单

> 基于 spec.md v2 (Agentic Multi-Strategy Router)
> 核心原则: 先基础设施 → 再索引层 → 再策略层 → 再路由层 → 最后集成

## 依赖关系

```
Task 1 (db.py + pyproject.toml)
  ↓
Task 2 (closet_index.py)
  ↓
Task 3 (agentic/planner.py) ─┐
Task 4 (agentic/strategies.py)├→ Task 6 (agentic/router.py)
Task 5 (agentic/verifier.py) ─┘
  ↓
Task 7 (client.py 集成)
```

## 任务列表

### Task 1: db.py + pyproject.toml — 基础设施
- **文件**: `db.py`（修改）, `pyproject.toml`（修改）
- **依赖**: 无
- **说明**:
  - `ensure_schema()` 中加入 `closet_tags` 建表语句
  - 保留/简化 `ensure_closet_schema()`
  - 验证 `insert_closet_tags` / `delete_closet_tags` / `match_closet_tags` 正确性
  - `pyproject.toml` 新增 `"jieba>=0.42"` 依赖
- **验收**:
  - [x] `python3 -c "from db import PageIndexDB; db = PageIndexDB('/tmp/test.db'); print('OK')"` 成功
  - [x] `import jieba` 成功

### Task 2: closet_index.py — Closet 语义标签索引
- **文件**: `PageIndex/pageindex/closet_index.py`（新建）
- **依赖**: Task 1
- **说明**:
  - `Tag` 数据类 (`text`, `confidence`)
  - `ClosetIndex.__init__(db, model)` — 调用 `ensure_closet_schema()`
  - `_extract_tags()` — LLM 标签提取 prompt → JSON 解析 → 过滤
  - `add_document()` — 提取节点标题 → LLM 标签 → jieba 分词 → 写入 DB
  - `remove_document()` — 调 `db.delete_closet_tags()`
  - `search(query, top_k)` — jieba 分词 → 过滤停用词 → `match_closet_tags()`
  - `rebuild()` — 清空并重建（未来扩展）
- **验收**:
  - [x] `python3 -c "from pageindex.closet_index import Tag, ClosetIndex; print('OK')"` 成功

### Task 3: agentic/planner.py — Plan 阶段
- **文件**: `PageIndex/pageindex/agentic/planner.py`（新建）
- **依赖**: Task 2
- **说明**:
  - `PlanResult` 数据类 (`queries`, `weights`, `query_type`)
  - `RetrievalPlanner.__init__(model)`
  - `plan(query)` — LLM 分析 query 类型 → HyDE 假设答案 → 生成 3 query 变体 → 分配策略权重
- **验收**:
  - [x] planner 可 import，`plan()` 返回正确结构

### Task 4: agentic/strategies.py — Route 三策略
- **文件**: `PageIndex/pageindex/agentic/strategies.py`（新建）
- **依赖**: Task 2
- **说明**:
  - `MetadataStrategy` — SQL LIKE 匹配 doc_name / doc_description
  - `SemanticsStrategy` — 调用 `ClosetIndex.search()`
  - `DescriptionStrategy` — LLM 判断文档描述相关性（复用 `get_relevant_documents_for_multidoc` 逻辑）
  - 统一接口: `async def search(query, docs_info) -> list[(doc_id, rank)]`
- **验收**:
  - [x] 三策略各自可独立运行并返回排名结果

### Task 5: agentic/verifier.py — Verify 阶段
- **文件**: `PageIndex/pageindex/agentic/verifier.py`（新建）
- **依赖**: 无（纯 LLM 判断）
- **说明**:
  - `VerifyResult` 数据类 (`confidence`, `action`)
  - `CRAGVerifier.__init__(model)`
  - `_score_retrieval(context, query)` — S_ret 计算
  - `verify(answer, context, query)` — S_CoV LLM 判断 + 三阈值路由
  - `TAU_HIGH = 0.7`, `TAU_LOW = 0.4`
- **验收**:
  - [x] verifier 返回高/中/低置信度正确

### Task 6: agentic/router.py — Router 编排
- **文件**: `PageIndex/pageindex/agentic/router.py`（新建）
- **依赖**: Task 3, 4, 5
- **说明**:
  - `AgenticRouter.__init__(client, model)` — 持有 Planner/Strategies/Verifier
  - `_weighted_rrf()` — 加权 RRF 融合
  - `_act_tree_search()` — 对候选文档执行 Tree 搜索 + 内容提取
  - `async def search(query, top_k)` — 完整 Plan→Route→Act→Verify 链路
- **验收**:
  - [x] router 端到端运行不报错

### Task 7: client.py — 集成与 search() 接口
- **文件**: `PageIndex/pageindex/client.py`（修改）
- **依赖**: Task 6
- **说明**:
  - `__init__` 中集成 `PageIndexDB` + `AgenticRouter` + `ClosetIndex`
  - `index()` 完成后调用 `closet_index.add_document()`
  - 新增 `async def search(query, top_k)` — 单文档 shortcut / 多文档走 Router
  - 新增 `_ensure_db()` 辅助方法
- **验收**:
  - [x] `python3 -c "from pageindex import PageIndexClient; print('OK')"` 成功
  - [x] 现有接口行为不变
