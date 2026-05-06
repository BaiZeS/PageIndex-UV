# Feature: Super-Tree Retrieval v3

**作者**: AI + User
**日期**: 2026-05-06
**状态**: Approved

---

## 1. 背景 (Background)

v2 的 Agentic 多策略路由（Plan→Route→Act→Verify）虽然实现了三策略并行召回，但仍存在两个核心问题：

1. **上下文爆炸**：L2 节点召回时，若候选文档过多（>5），将所有文档的树结构塞进一个 prompt 会导致精度急剧下降。
2. **双重 LLM 筛选**：v2 在 Route 阶段用 LLM 做 Description 策略筛选，又在 Act 阶段用 LLM 做树推理，两次 LLM 调用均处理多文档上下文，成本高且互相干扰。

因此需要一层更轻量、更前置的文档筛选机制，在 L2 之前将候选集从"全库"压缩到"3-5 个最相关文档"。

## 2. 目标 (Goals)

- 多文档检索时，L2 阶段只处理 3-5 个精选文档，避免上下文爆炸
- 文档筛选阶段零向量依赖，纯结构化数据驱动
- L1 失败时自动降级到 v2 三策略路由，系统始终可用
- 总检索延迟不劣于 v2（~1.5s 内完成 L0+L1）

## 3. 需求细化 (Requirements)

### 3.1 功能性需求

- **FR1: L0 双通道预过滤**
  - Channel A: ClosetIndex 语义标签匹配（复用 v2）
  - Channel B: KeywordIndex jieba 倒排索引（新）
  - 两路结果按评分合并，最多保留 50 个候选文档

- **FR2: L1 Super-Tree 文档选择**
  - 为每个候选文档构建 mini-TOC（仅 depth=1 节点，最多 8 个）
  - 结合 KB Identity（知识库整体摘要）作为全局上下文
  - 单次 LLM 调用选出 3-5 个最相关文档
  - Prompt 总 token 预算 6000，超限时自动截断候选

- **FR3: KB Identity 懒加载**
  - 基于文档名和顶层章节标题，LLM 生成知识库整体摘要
  - 缓存到 SQLite `kb_identity` 表，文档增删时自动失效重建
  - LLM 失败时回退到文档列表拼接

- **FR4: 自动降级**
  - L1 任一阶段失败（预过滤空、LLM 选档失败、Act 阶段异常）时，自动捕获异常并回退到 v2 的 Plan→Route→Act→Verify 路径

- **FR5: 索引生命周期钩子**
  - 文档新增时自动触发关键词索引和语义标签索引
  - 文档删除时自动清理索引和 KB Identity 缓存

### 3.2 非功能性需求

- **NFR1: 延迟** — L0+L1 总延迟 < 1s（不含 L2/L3）
- **NFR2: 依赖** — 仅新增 `jieba`，无向量库/框架
- **NFR3: 兼容性** — `PageIndexClient` 接口不动，`search()` 行为透明升级
- **NFR4: 降级** — L1 失败不影响可用性，v2 路径兜底

## 4. 设计方案 (Design)

### 4.1 方案概览

Super-Tree Retrieval v3 在 v2 基础上前置了两层筛选：

```
User Question
    │
    ▼
┌─────────────────┐  L0: 双通道预过滤
│ 候选文档召回     │  → Channel A: ClosetIndex 语义标签
│ (SuperTreeIndex) │  → Channel B: KeywordIndex 倒排
└────────┬────────┘     最多 50 候选
         │
         ▼
┌─────────────────┐  L1: Super-Tree 文档选择
│ LLM 推理选档     │  → mini-TOC + KB Identity
│ (SuperTreeIndex) │  → 6000 tokens 预算
└────────┬────────┘     → 3-5 精选文档
         │
         ▼
┌─────────────────┐  L2/L3: 节点召回 + 上下文提取
│ (AgenticRouter)  │  → 逐文档独立树推理
│ 复用 v2 Act/Verify│  → 16K tokens 硬上限截断
└─────────────────┘
```

### 4.2 组件设计

#### 4.2.1 文件结构

```
PageIndex/pageindex/
  super_tree.py     # KeywordIndex + KBIdentity + SuperTreeIndex
  closet_index.py   # ClosetIndex（v2 已有，复用）
  agentic/          # v2 已有，复用
    router.py       # AgenticRouter: 优先 Super-Tree，失败回退 v2
  client.py         # PageIndexClient: 集成 SuperTreeIndex
  utils.py          # extract_json, count_tokens 等工具
```

#### 4.2.2 核心类

**KeywordIndex**
```python
class KeywordIndex:
    def __init__(self, db: PageIndexDB)
    def add_document(self, doc_id: int, doc_name: str, doc_description: str) -> None
    def remove_document(self, doc_id: int) -> None
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]
```

**KBIdentity**
```python
class KBIdentity:
    def __init__(self, db: PageIndexDB, model: str)
    def get_identity(self) -> str
    def invalidate(self) -> None
```

**SuperTreeIndex**
```python
class SuperTreeIndex:
    def __init__(self, db: PageIndexDB, model: str, client: PageIndexClient)
    def on_document_added(self, db_doc_id: int) -> None
    def on_document_removed(self, db_doc_id: int) -> None
    def prefilter(self, query: str) -> Dict[int, float]
    async def select_documents(self, query: str, candidate_db_ids: Dict[int, float]) -> List[str]
```

#### 4.2.3 接口设计

**PageIndexClient.search()** 行为不变，内部透明升级：
- 单文档：直接 L2 Tree 搜索
- 多文档：Router 优先尝试 Super-Tree (L0→L1→L2→L3)，失败回退 v2

返回值与 v2 保持一致：
```python
{
    "query": str,
    "mode": "single" | "multi",
    "answer": str,
    "confidence": "high" | "medium" | "low" | "unknown",
    "matched_docs": [{"doc_id": str, "score": float}],
    "selected_nodes": [{"node_id": str, "title": str}],
    "pages": [{"doc_id": str, "pages": [int]}],
}
```

### 4.3 数据模型

**新增表**：

```sql
-- 关键词倒排索引（Channel B）
CREATE TABLE doc_keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    keyword TEXT NOT NULL,
    field TEXT NOT NULL
);
CREATE INDEX idx_doc_keywords ON doc_keywords(keyword, doc_id);

-- 知识库整体摘要缓存
CREATE TABLE kb_identity (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    identity_text TEXT NOT NULL,
    doc_count INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

`closet_tags` 表复用 v2 定义，不变。

### 4.4 错误处理与降级

| 场景 | 处理 |
|------|------|
| L0 预过滤空结果 | Router 直接返回 "No relevant documents" |
| L1 LLM 调用失败 | 捕获异常，回退到 v2 三策略路由 |
| L1 返回空 doc_ids | Router 返回 "Super-Tree selection returned no documents" |
| L2/L3 失败 | 返回已选文档列表 + 错误提示，confidence="unknown" |
| KB Identity LLM 失败 | 回退到文档列表拼接作为上下文 |
| jieba 未安装 | KeywordIndex 不可用，仅依赖 ClosetIndex (Channel A) |

## 5. 变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `db.py` | 修改 | 新增 `doc_keywords`、`kb_identity` 表 + CRUD |
| `PageIndex/pageindex/super_tree.py` | 新建 | KeywordIndex + KBIdentity + SuperTreeIndex |
| `PageIndex/pageindex/agentic/router.py` | 修改 | 优先 Super-Tree，失败回退 v2 |
| `PageIndex/pageindex/client.py` | 修改 | 初始化 SuperTreeIndex，索引时触发关键词索引 |

## 6. 测试计划

1. **KeywordIndex**: add/search/remove 单测
2. **KBIdentity**: 缓存命中/缓存未命中/LLM 生成/失效重建
3. **SuperTreeIndex**: prefilter 空库/关键词匹配、select_documents 空结果/full path
4. **Router**: 优先 Super-Tree、Super-Tree 失败回退 v2、无 Super-Tree 时直接用 v2

## 7. Changelog

| 日期 | 变更内容 | 作者 |
|------|----------|------|
| 2026-04-30 | v1: Closet + Tree 两层方案 | AI + User |
| 2026-05-05 | v2: 升级为 Agentic 多策略路由 | AI + User |
| 2026-05-06 | v3: 前置 Super-Tree L0+L1 筛选，解决上下文爆炸 | AI + User |
