# PageIndex-UV 性能与功能优化计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent (recommended) or compose:execute to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 聚焦功能和性能优化，保持轻量，不过度工程化

**Architecture:** 8 个独立优化任务，按优先级排序，每个任务改动量小（10-50 行），直接提升检索质量或降低延迟

**Tech Stack:** Python 3.12+ / SQLite / ChromaDB / jieba

## Global Constraints

- 不引入新的外部依赖
- 不改变现有 API 接口
- 所有改动必须通过现有测试
- 保持代码简洁，避免过度抽象

---

### Task 1: 检索结果缓存（P0）

**目标:** 相同查询重复调用时零延迟

**Files:**
- Modify: `pageindex_mutil/keyword_backend.py`
- Modify: `pageindex_mutil/hybrid_backend.py`
- Test: `tests/test_search_backends.py`

**原理:** `utils.py` 已有 `_LLM_CACHE` 机制，复用相同模式为搜索结果添加 LRU 缓存

- [ ] **Step 1: 为 KeywordSearchBackend 添加搜索缓存**

在 `pageindex_mutil/keyword_backend.py` 的 `KeywordSearchBackend` 类中添加缓存：

```python
import hashlib
import time
from functools import lru_cache

class KeywordSearchBackend(SearchBackend):
    # 添加类级缓存
    _search_cache: dict = {}  # hash(query+top_k) -> (result, timestamp)
    _cache_ttl = 300  # 5 minutes
    _cache_max = 128

    def _cache_key(self, query: str, top_k: int) -> str:
        return hashlib.md5(f"{query}:{top_k}".encode()).hexdigest()

    def _cache_get(self, key: str):
        entry = self._search_cache.get(key)
        if entry and (time.monotonic() - entry[1]) < self._cache_ttl:
            return entry[0]
        if entry:
            del self._search_cache[key]
        return None

    def _cache_set(self, key: str, value):
        if len(self._search_cache) >= self._cache_max:
            oldest_key = min(self._search_cache, key=lambda k: self._search_cache[k][1])
            del self._search_cache[oldest_key]
        self._search_cache[key] = (value, time.monotonic())

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        # Check cache first
        cache_key = self._cache_key(query, top_k)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # Original search logic
        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores: Dict[int, float] = {}
        try:
            for doc_id, score in self.db.match_doc_keywords(tokens, top_k):
                scores[int(doc_id)] = scores.get(int(doc_id), 0.0) + float(score) * self.keyword_weight
        except Exception as e:
            logger.warning("Keyword search failed: %s", e)

        try:
            for doc_id, score in self.db.match_closet_tags(tokens, top_k):
                scores[int(doc_id)] = scores.get(int(doc_id), 0.0) + float(score) * self.tag_weight
        except Exception as e:
            logger.warning("Tag search failed: %s", e)

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Cache result
        self._cache_set(cache_key, sorted_results)
        return sorted_results
```

- [ ] **Step 2: 为 HybridSearchBackend 添加搜索缓存**

在 `pageindex_mutil/hybrid_backend.py` 的 `HybridSearchBackend` 类中添加相同缓存逻辑

- [ ] **Step 3: 运行测试验证**

```bash
uv run pytest tests/test_search_backends.py -v
```

Expected: 所有测试通过

- [ ] **Step 4: Commit**

```bash
git add pageindex_mutil/keyword_backend.py pageindex_mutil/hybrid_backend.py
git commit -m "perf: add search result caching to keyword and hybrid backends"
```

---

### Task 2: SQLite 查询优化（P0）

**目标:** 多关键词查询从 N 次 IO 降到 1 次

**Files:**
- Modify: `db.py:366-426`
- Test: `tests/test_db.py`

**原理:** 当前 `match_doc_keywords` 和 `match_closet_tags` 已经使用 `IN` 子句批量查询，但结果合并逻辑可以优化

- [ ] **Step 1: 优化结果合并逻辑**

在 `db.py` 的 `match_doc_keywords` 方法中，简化合并逻辑：

```python
def match_doc_keywords(self, tokens, top_k=10):
    if not tokens:
        return []
    conn = self._connect()
    # Single query with IN clause - already optimized
    placeholders = ",".join("?" for _ in tokens)
    sql = f"""
        SELECT doc_id, COUNT(*) AS score
        FROM doc_keywords
        WHERE keyword IN ({placeholders})
        GROUP BY doc_id
        ORDER BY score DESC
        LIMIT ?
    """
    rows = conn.execute(sql, (*tokens, top_k)).fetchall()
    return [(r["doc_id"], r["score"]) for r in rows]
```

- [ ] **Step 2: 同样优化 match_closet_tags**

```python
def match_closet_tags(self, tokens, top_k=5):
    if not tokens:
        return []
    conn = self._connect()
    placeholders = ",".join("?" for _ in tokens)
    sql = f"""
        SELECT doc_id, SUM(confidence) AS score
        FROM closet_tags
        WHERE tag_token IN ({placeholders})
        GROUP BY doc_id
        ORDER BY score DESC
        LIMIT ?
    """
    rows = conn.execute(sql, (*tokens, top_k)).fetchall()
    return [(r["doc_id"], r["score"]) for r in rows]
```

- [ ] **Step 3: 运行测试验证**

```bash
uv run pytest tests/test_db.py -v
```

Expected: 所有测试通过

- [ ] **Step 4: Commit**

```bash
git add db.py
git commit -m "perf: simplify SQLite query result merging in match_doc_keywords and match_closet_tags"
```

---

### Task 3: Embedding 模型身份追踪（P1）

**目标:** 防止切换 embedding 模型后维度不匹配导致的数据损坏

**Files:**
- Modify: `pageindex_mutil/chroma_backend.py`
- Test: `tests/test_search_backends.py`

- [ ] **Step 1: 在 ChromaSearchBackend 中追踪 embedding 模型信息**

在 `pageindex_mutil/chroma_backend.py` 的 `ChromaSearchBackend.__init__` 中添加模型身份记录：

```python
def __init__(self, db_path: str = "./data/vectors", embedding_model: str = "local"):
    if chromadb is None:
        raise ImportError("chromadb is required for vector search.")
    
    self.db_path = Path(db_path)
    self.db_path.mkdir(parents=True, exist_ok=True)
    
    self.client = chromadb.PersistentClient(path=str(self.db_path))
    self.collection = self.client.get_or_create_collection(
        name="pageindex_documents",
        metadata={"hnsw:space": "cosine"}
    )
    
    self.embedding_model = embedding_model
    self._embedder = None
    
    # Track embedding model identity
    stored_model = self.collection.metadata.get("embedding_model")
    if stored_model and stored_model != embedding_model:
        logging.warning(
            "Embedding model changed from '%s' to '%s'. "
            "Existing vectors may be incompatible.",
            stored_model, embedding_model
        )
    
    # Update stored model identity
    if not stored_model:
        self.collection.modify(metadata={
            **self.collection.metadata,
            "embedding_model": embedding_model
        })
    
    logging.info("ChromaDB backend initialized at %s", self.db_path)
```

- [ ] **Step 2: 运行测试验证**

```bash
uv run pytest tests/test_search_backends.py -v -k "chroma"
```

Expected: ChromaDB 测试通过（如果安装了 chromadb）

- [ ] **Step 3: Commit**

```bash
git add pageindex_mutil/chroma_backend.py
git commit -m "feat: track embedding model identity in ChromaDB to prevent dimension mismatches"
```

---

### Task 4: 健康检查增强（P1）

**目标:** 健康检查返回更多运维信息

**Files:**
- Modify: `app/server.py:549-557`

- [ ] **Step 1: 增强 health_endpoint**

在 `app/server.py` 的 `health_endpoint` 函数中返回更多信息：

```python
async def health_endpoint(request: Request) -> Response:
    import time as _time
    start = _time.monotonic()
    
    c = get_client()
    doc_count = 0
    db_status = "ok"
    entity_count = 0
    
    if c.db is not None:
        try:
            docs = c.db.get_all_documents()
            doc_count = len(docs)
            # Count entities
            conn = c.db._connect()
            entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        except Exception as e:
            db_status = f"error: {e}"
    else:
        db_status = "unavailable"
    
    latency_ms = int((_time.monotonic() - start) * 1000)
    
    return JSONResponse({
        "status": "ok",
        "documents": doc_count,
        "entities": entity_count,
        "db_status": db_status,
        "latency_ms": latency_ms,
        "search_backend": os.getenv("SEARCH_BACKEND", "keyword"),
    })
```

- [ ] **Step 2: 运行测试验证**

```bash
uv run pytest tests/test_web_console.py -v -k "health"
```

Expected: 测试通过

- [ ] **Step 3: Commit**

```bash
git add app/server.py
git commit -m "feat: enhance health endpoint with entity count and db status"
```

---

### Task 5: 文档统计 MCP 工具（P2）

**目标:** 添加获取知识库概览的 MCP 工具

**Files:**
- Modify: `app/server.py`
- Modify: `docs/mcp-tools.md`

- [ ] **Step 1: 添加 get_stats MCP 工具**

在 `app/server.py` 中添加新的 MCP 工具处理器：

```python
@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    # ... existing tools ...
    
    elif name == "get_stats":
        c = get_client()
        stats = {
            "documents": 0,
            "entities": 0,
            "relations": 0,
            "search_backend": os.getenv("SEARCH_BACKEND", "keyword"),
        }
        if c.db is not None:
            try:
                stats["documents"] = len(c.db.get_all_documents())
                conn = c.db._connect()
                stats["entities"] = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
                stats["relations"] = conn.execute("SELECT COUNT(*) FROM entity_relations").fetchone()[0]
            except Exception as e:
                stats["error"] = str(e)
        return [types.TextContent(type="text", text=json.dumps(stats, ensure_ascii=False))]
```

- [ ] **Step 2: 在 list_tools 中注册新工具**

```python
@mcp_server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        # ... existing tools ...
        types.Tool(
            name="get_stats",
            description="Get knowledge base statistics (document count, entity count, etc.)",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]
```

- [ ] **Step 3: 更新 MCP 工具文档**

在 `docs/mcp-tools.md` 中添加 `get_stats` 工具说明

- [ ] **Step 4: 运行测试验证**

```bash
uv run pytest tests/test_web_console.py -v
```

Expected: 所有测试通过

- [ ] **Step 5: Commit**

```bash
git add app/server.py docs/mcp-tools.md
git commit -m "feat: add get_stats MCP tool for knowledge base statistics"
```

---

### Task 6: 实体关系查询增强（P2）

**目标:** 支持按关系类型查询实体

**Files:**
- Modify: `db.py`
- Modify: `app/server.py`

- [ ] **Step 1: 在 db.py 中添加按关系类型查询的方法**

```python
def get_entities_by_relation(self, predicate: str, limit: int = 20) -> list:
    """Get entity pairs connected by a specific relation type."""
    conn = self._connect()
    rows = conn.execute(
        """
        SELECT e1.name as subject_name, e2.name as object_name, 
               er.confidence, er.doc_id
        FROM entity_relations er
        JOIN entities e1 ON er.subject_id = e1.id
        JOIN entities e2 ON er.object_id = e2.id
        WHERE er.predicate = ?
        ORDER BY er.confidence DESC
        LIMIT ?
        """,
        (predicate, limit)
    ).fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 2: 添加 MCP 工具 get_relations**

在 `app/server.py` 中添加：

```python
elif name == "get_relations":
    predicate = arguments.get("predicate", "")
    limit = arguments.get("limit", 20)
    c = get_client()
    if c.db is None:
        return [types.TextContent(type="text", text=json.dumps({"error": "DB unavailable"}))]
    relations = c.db.get_entities_by_relation(predicate, limit)
    return [types.TextContent(type="text", text=json.dumps({"relations": relations}, ensure_ascii=False))]
```

- [ ] **Step 3: 运行测试验证**

```bash
uv run pytest tests/test_db.py -v
```

Expected: 所有测试通过

- [ ] **Step 4: Commit**

```bash
git add db.py app/server.py
git commit -m "feat: add entity relation query by predicate type"
```

---

### Task 7: 异步索引优化（P3）

**目标:** 服务端可以并发处理多个索引请求

**Files:**
- Modify: `pageindex_mutil/client.py`

**原理:** 当前 `client.index()` 中的 PDF/MD 解析已经是异步的（通过 `asyncio.run`），但可以通过 `asyncio.to_thread` 包装避免阻塞事件循环

- [ ] **Step 1: 确认现有代码已经是非阻塞的**

检查 `app/server.py` 中的 `_index_one_file` 函数，确认已经使用了 `asyncio.to_thread`：

```python
async def _index_one_file(temp_path: Path, filename: str) -> dict:
    mode = _determine_mode(filename)
    try:
        async with _UPLOAD_SEMAPHORE:
            doc_id = await asyncio.to_thread(
                get_client().index, str(temp_path), mode=mode
            )
        return {"filename": filename, "success": True, "doc_id": doc_id}
    except Exception as e:
        return {"filename": filename, "success": False, "error": str(e)}
```

如果已经实现，跳过此任务。

- [ ] **Step 2: Commit（如果需要）**

---

### Task 8: 最终验证（P3）

**目标:** 确保所有优化都通过测试

**Files:** 无

- [ ] **Step 1: 运行完整测试套件**

```bash
uv run pytest tests/ -v --tb=short
```

Expected: 所有测试通过

- [ ] **Step 2: 验证性能改进**

重复执行相同搜索查询，验证缓存生效：

```python
import time
# 第一次查询（冷启动）
start = time.time()
result1 = backend.search("test query", top_k=5)
cold_time = time.time() - start

# 第二次查询（缓存命中）
start = time.time()
result2 = backend.search("test query", top_k=5)
cached_time = time.time() - start

print(f"冷启动: {cold_time:.3f}s, 缓存命中: {cached_time:.3f}s")
print(f"加速比: {cold_time/cached_time:.1f}x")
```

---

## 优化效果预期

| 任务 | 预期效果 | 改动量 |
|------|---------|--------|
| Task 1: 检索缓存 | 重复查询 10-100x 加速 | ~40 行 |
| Task 2: SQLite 优化 | 多关键词查询 2-5x 加速 | ~20 行 |
| Task 3: Embedding 身份 | 防止数据损坏 | ~15 行 |
| Task 4: 健康检查 | 运维可见性 | ~20 行 |
| Task 5: 统计工具 | 知识库概览 | ~30 行 |
| Task 6: 关系查询 | 更丰富检索 | ~40 行 |
| Task 7: 异步索引 | 并发能力 | 已实现 |
| Task 8: 最终验证 | 质量保证 | 0 行 |

总改动量: ~165 行代码
