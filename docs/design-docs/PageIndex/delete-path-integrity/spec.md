# W2 · 删除路径与数据完整性修复 — spec.md

> 阶段：clarify（§1-3 待确认）。仅含需求，不含技术设计。
> 来源：`architecture-review-2026-06/review-report.md` P0-2 / P1-8 / P1-10。依赖 W1（数据层并发安全）先落地。
> 用户确认范围：核心修复 + 现有孤儿数据清理 + 删除时清理磁盘文件。

## §1 背景

文档删除路径当前**完全失效**，且不维护索引一致性：

1. **删除静默失效（P0-2）**：`server.py:164` 调用 `c.db.delete_document(db_id)`，但 `db.py` 中**不存在该方法**；`except Exception`（`server.py:165`）静默吞掉 `AttributeError` 并返回 `{"success": True}`。`documents`/`nodes`/`pages` 表虽有 `ON DELETE CASCADE`，但从未执行 `DELETE FROM documents` → 级联不触发，孤儿行无限堆积，搜索准确率下降，数据保留/隐私风险。违反 super-tree-v3 FR5。
2. **删除不失效索引（P1-8）**：删除 handler 调 `closet_index.remove_document` 但**从不调** `super_tree_index.on_document_removed`（后者存在于 `super_tree.py:151`，会清关键词索引 + 失效 KB 身份）。添加路径（`client.py:172` 调 `on_document_added`）正确，删除路径不对称损坏。
3. **KBIdentity 存储未清洗（P1-10）**：`super_tree.py:103-106` 存 `response.strip()` 原样，不剥离 Markdown 代码块（对比 `select_documents@239` 用 `extract_json` 剥离）。LLM 若返回围栏文本则持久化并注入每个 L1 prompt。
4. **历史孤儿数据**：因删除长期失效，现有 `pageindex.db` 可能已堆积孤儿行。
5. **磁盘文件无清理**：上传的 PDF 永久驻留 `upload_dir`，删除文档时不移除。

## §2 目标

- 删除操作**真正删除**文档及其级联子表（nodes/pages/closet_tags/doc_keywords）。
- 删除时**失效 KB 索引与身份缓存**（FR5 合规）。
- **清洗** KBIdentity 存储，防止污染传播。
- **清理历史孤儿数据**（迁移脚本）。
- 删除时**移除上传的磁盘文件**。
- 为删除路径提供**集成测试**（当前仅 `test_super_tree.py:179` 测方法本身，未测 MCP 删除路径，故未捕获本缺陷）。

## §3 需求与范围

### 功能需求
- **FR1**：`PageIndexDB.delete_document(doc_id)` 执行 `DELETE FROM documents WHERE id=?`，依赖现有 `ON DELETE CASCADE` 外键清子表。
- **FR2**：`server.py` 删除路径调用 `super_tree_index.on_document_removed(db_id)`（失效关键词索引 + KB 身份）——满足 super-tree-v3 FR5。
- **FR3**：`KBIdentity` 存储前剥离 Markdown 代码围栏（与 `extract_json` 剥离逻辑一致或通用清洗）。
- **FR4**：删除文档时移除 `upload_dir` 中对应 PDF 文件（磁盘清理）。
- **FR5**：提供迁移脚本，清理现有 `pageindex.db` 中 `doc_id` 不再存在于 `documents` 的孤儿行（nodes/pages/closet_tags/doc_keywords）。

### 非功能需求
- **NFR1**：集成测试 插入→删除→断言 documents/nodes/pages/tags/keywords 行清除 + `kb_identity` 失效 + 磁盘文件移除。
- **NFR2**：删除幂等——删除不存在的 id 返回成功/无操作，不报错。
- **NFR3**：向后兼容——无 schema 变更（cascade 已存在）；孤儿清理为附加式迁移，可重复执行。
- **NFR4**：并发删除安全（依赖 W1 的数据层并发安全）。

### 范围
- **IN**：`db.delete_document`；`server.py` 删除路径接线；`KBIdentity` Markdown 剥离；磁盘文件清理；孤儿数据迁移脚本；删除集成测试。
- **OUT**：
  - 软删除（按现有 cascade 意图为**硬删除**）。
  - 超出 cascade 的 closet_index 标签清理（由 cascade 处理）。
  - 超出 W1 安全范围的并发删除竞争（由 W1 提供基础）。
  - `_build_super_tree` N+1 重写（W3）。

### 假设（ASSUMPTION，需确认）
- 硬删除（cascade）是预期语义（schema 已有 `ON DELETE CASCADE`）。
- `upload_dir` 路径可由文档记录的 `pdf_name`/`pdf_path` 确定性解析。
- 迁移脚本可独立运行（`python -m ...` 或一次性脚本），不要求在线热迁移。

---

> 🚨 **STOP — REQUIREMENTS_CONFIRMED**
> 待用户确认本 spec §1-3 后，由独立 agent 调用 quality-gate（threshold ≥70），通过后进入 system-design 阶段。

---

## §4 方案设计（System Design）

> 状态：DESIGN_DECISION 已用户确认（见各小节）。§4-8 为设计产出，待独立 agent 调 quality-gate（threshold ≥70）后进入 plan 阶段。

### 4.0 设计输入（已核实 — codegraph + Read）

- **`db.py:31-85`（ensure_schema）**：`documents` 表为父表；`nodes`/`pages`/`closet_tags`/`doc_keywords` 均声明 `doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE`。**cascade 已存在于 schema，无需新增**。`kb_identity` 表（单行 `CHECK (id=1)`）无外键，不受 cascade 影响，需显式 `invalidate`。
- **`db.py:14-19`（`_connect`）**：`sqlite3.connect(db_path)` + `PRAGMA foreign_keys = ON` —— **foreign_keys 每连接开启**，`DELETE FROM documents` 必触发级联（W1 线程本地连接层保证每连接独立 PRAGMA）。当前 `db.py` **无 `delete_document` 方法**（FR1 新增）。
- **`db.py` 既有按 doc_id 清理方法**：`delete_closet_tags(doc_id)`（268）、`delete_doc_keywords(doc_id)`（316）—— 仅在非 cascade 路径（索引层 `KeywordIndex.remove_document`/`ClosetIndex.remove_document`）使用，delete_document 不复用它们（cascade 即覆盖）。
- **`server.py:151-174`（delete handler）**：`c.db.delete_document(db_id)` 在 `try/except` 内（165 静默吞 `AttributeError`）；仅调 `c.closet_index.remove_document(db_id)`（170），**从不调 `c.super_tree_index.on_document_removed`**；返回 `{"success": True}`。内存态 `c.documents`（159）与 `c._uuid_to_db`（161）已清理。
- **`super_tree.py:143-153`**：`on_document_added`（add 路径，client.py:172 已正确调用）= `keyword_index.add_document` + `kb_identity.invalidate`；`on_document_removed`（**删除路径从未被调用**）= `keyword_index.remove_document` + `kb_identity.invalidate`。**删除路径不对称**是 P1-8 根因。
- **`super_tree.py:80-107`（`KBIdentity._generate_with_llm` → `set_kb_identity`）**：`self.db.set_kb_identity(response.strip(), len(docs))` 存原样 `response.strip()`，**不剥离 Markdown 围栏**。对比 `select_documents@239` 用 `extract_json` 剥离（用于 JSON 解析）。`_build_fallback@109` 存的是程序生成文本（无围栏），仅 `_generate_with_llm` 路径需清洗。
- **`utils.py:173-204`（`extract_json`）**：剥离逻辑 = 若含 ` ```json ` 则取首尾围栏间内容并 `.strip()`；否则整体 `.strip()`。**围栏剥离是通用子操作**，可抽为独立函数供 KBIdentity 复用（FR3 收敛为"剥离 Markdown 围栏"单一语义，实现取向见 §5.2）。
- **磁盘路径（关键架构事实）**：`server.py:368-381` 上传文件存 `workspace/uploads/{uuid4().hex}_{safe_name}`；`PageIndexClient.index(file_path=...)`（client.py:81）→ `db.insert_document(pdf_path=file_path, ...)`（client.py:159）**直接存绝对路径 `temp_path`**。即 **`documents.pdf_path` 列就是上传文件的绝对路径**，无需从 `pdf_name` 反解 `upload_dir`。`pdf_name` 列存的是 `doc_name`（`get_pdf_name`/sanitize_filename 产物，可能含 `.pdf`），与磁盘文件名（`{uuid}_{safe_name}`）**不同**，不可用于定位磁盘文件。
- **依赖 W1**：`delete_document` 用 `self._connect()`（线程本地连接层，W1 保证并发安全 + 每连接 `foreign_keys=ON` → cascade 生效）。

### 4.1 删除流程（端到端）

```
server.py delete_document handler
  │
  ├─ 1. 内存态清理：del c.documents[doc_id]；c._uuid_to_db.pop(doc_id) → db_id
  │
  ├─ 2. DB 级联删除：c.db.delete_document(db_id)        ← FR1（新增方法）
  │     └─ DELETE FROM documents WHERE id=?
  │        ├─ cascade → nodes / pages / closet_tags / doc_keywords 自动清除
  │        └─ kb_identity 不受影响（无 FK，步骤 3 显式失效）
  │
  ├─ 3. 索引失效：c.super_tree_index.on_document_removed(db_id)   ← FR2（新增接线）
  │     ├─ keyword_index.remove_document(db_id)  → delete_doc_keywords（幂等，cascade 已清则 0 行）
  │     └─ kb_identity.invalidate()              → DELETE FROM kb_identity WHERE id=1（下次 get_identity 重建）
  │
  ├─ 4. closet_index 清理：c.closet_index.remove_document(db_id)   ← 既有（保留，幂等）
  │     └─ delete_closet_tags（cascade 已清则 0 行，幂等无害）
  │
  └─ 5. 磁盘文件清理：os.remove(pdf_path)（try/except，FileNotFoundError 跳过）  ← FR4
        └─ pdf_path 取自删除前 get_document_by_id 的 pdf_path 列（绝对路径）
           或在步骤 2 前先取（步骤 2 后行已删）—— 见 §5.4 时序
```

**幂等性**：步骤 2 `DELETE FROM documents WHERE id=?` 对不存在 id 删 0 行（NFR2）；步骤 3/4 的 `delete_doc_keywords`/`delete_closet_tags` 对空集删 0 行；步骤 5 `FileNotFoundError` 被吞。故重复调用同一 `doc_id` 全程无报错（NFR2）。

### 4.2 KBIdentity 存储清洗（FR3）

**问题**：`_generate_with_llm@103-106` 存 `response.strip()`，若 LLM 返回 ` ```text\n摘要\n``` ` 围栏文本则原样持久化，污染后续每个 L1 prompt（`select_documents` 的 `[知识库概览]`）。

**收敛语义（quality-gate 建议吸收）**：FR3 收敛为**单一语义**——"剥离 Markdown 围栏"。**实现取向留设计阶段**：§5.2 给出两种实现取向（A: 抽 `strip_markdown_fence` 通用函数；B: 内联复用 `extract_json` 的围栏剥离片段），在 §5.2 明确并推荐 A。

**清洗点**：仅 `_generate_with_llm` 调用 `set_kb_identity` 前。`_build_fallback`（程序生成文本，无围栏）与 `invalidate`（DELETE，无文本）不需清洗。NFR1 增加"kb_identity 存储值不含围栏标记（` ``` `）"断言。

### 4.3 历史孤儿数据迁移（FR5 / NFR3）

**根因**：删除长期失效 → `pageindex.db` 可能已堆积 `doc_id` 不在 `documents` 的孤儿行（nodes/pages/closet_tags/doc_keywords）。

**迁移 SQL（幂等，可重复执行）**：
```sql
-- 对每张子表：
DELETE FROM <child_table>
WHERE doc_id NOT IN (SELECT id FROM documents);
-- 覆盖：nodes, pages, closet_tags, doc_keywords
```
- **幂等性**：`NOT IN` 子查询对已干净的库返回空集 → 删 0 行（NFR3）。
- **无 schema 变更**：纯 DML，附加式迁移（NFR3）。
- **`kb_identity`**：不含 `doc_id`，孤儿迁移不处理它；但历史脏值（围栏污染）由 §4.2 在下次 `_build` 时自然清洗，或迁移脚本可选 `DELETE FROM kb_identity` 触发重建（设计阶段建议可选，非必须）。

### 4.4 磁盘清理时序（FR4）

**关键时序约束**：`pdf_path` 必须在 `delete_document(db_id)` **之前**取出（步骤 2 后该行已级联删除，`get_document_by_id` 返回 None）。故磁盘清理时序：

```
doc = c.db.get_document_by_id(db_id)      # 删除前取 pdf_path
c.db.delete_document(db_id)              # 级联删行
c.super_tree_index.on_document_removed(db_id)
c.closet_index.remove_document(db_id)
if doc and doc.get("pdf_path"):
    try: os.remove(doc["pdf_path"])
    except FileNotFoundError: pass       # 幂等
    except OSError as e: logger.warning(...)  # 记录但不阻断成功
```

**安全护栏**：`os.remove` 前校验 `pdf_path` 解析后落在 `workspace/uploads/` 下（防误删库外文件）。见 §5.4 / §7。

---

## §5 接口设计（Interface Design）

### 5.1 `PageIndexDB.delete_document(doc_id)` —— FR1

**签名**（`db.py`，新增方法，与 `delete_closet_tags`/`delete_doc_keywords` 同层）：
```python
def delete_document(self, doc_id: int) -> None:
    """Delete a document and cascade-delete its child rows.

    Relies on existing ``ON DELETE CASCADE`` foreign keys on
    nodes/pages/closet_tags/doc_keywords (see ensure_schema).
    Idempotent: deleting a non-existent id deletes 0 rows (no error).
    """
    with self._connect() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
```
- **连接**：`self._connect()`（W1 线程本地层，每连接 `PRAGMA foreign_keys=ON` → cascade 生效）。
- **返回**：`None`（无显式 rowcount 检查；幂等语义由 SQL 0 行删除保证）。
- **不调用** `delete_closet_tags`/`delete_doc_keywords`：cascade 已覆盖，复用会重复（虽幂等无害，但冗余）。

### 5.2 KBIdentity 围栏剥离 —— FR3

**实现取向（设计阶段明确，推荐 A）**：

- **取向 A（推荐）**：在 `utils.py` 抽通用函数 `strip_markdown_fence(text: str) -> str`，复用 `extract_json` 的围栏剥离片段（` ```json `/` ``` ` 首尾截取 + `.strip()`），但**不做 JSON 解析**（KBIdentity 是纯文本摘要）。`_generate_with_llm` 调 `set_kb_identity(strip_markdown_fence(response), len(docs))`。
  - ✅ 语义单一（"剥离围栏"）；与 `extract_json` 共享剥离子逻辑但解耦（一个解析 JSON，一个不解析）；可单测。
- **取向 B（不推荐）**：在 `_generate_with_llm` 内联围栏剥离代码。重复 `extract_json` 的剥离片段，维护漂移风险。

**`strip_markdown_fence` 签名**（`utils.py`，新增）：
```python
def strip_markdown_fence(text: str) -> str:
    """Strip a single outermost Markdown code fence (```...```) if present.

    Unlike extract_json, does NOT attempt JSON parsing — returns the
    stripped text content. Idempotent on already-unfenced text.
    """
    if not text:
        return text
    s = text.strip()
    if s.startswith("```"):
        # drop the opening fence line (optionally ```lang)
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()
```
- **接线点**：`super_tree.py:103-106`，`set_kb_identity(response.strip(), ...)` → `set_kb_identity(strip_markdown_fence(response), ...)`。`_build_fallback` 不改（程序生成无围栏）。

### 5.3 `on_document_removed` 接线 —— FR2

**既有方法**（`super_tree.py:151-153`，无需修改签名）：
```python
def on_document_removed(self, db_doc_id: int) -> None:
    self.keyword_index.remove_document(db_doc_id)
    self.kb_identity.invalidate()
```
**接线点**（`server.py` delete handler，在 `closet_index.remove_document` **之后**补一行，见 §5.4 时序）：
```python
if c.super_tree_index and db_id is not None:
    try:
        c.super_tree_index.on_document_removed(db_id)
    except Exception as e:
        logger.warning("Failed to invalidate super-tree index: %s", e)
```
- **幂等**：`keyword_index.remove_document` → `delete_doc_keywords`（cascade 已清则 0 行）；`kb_identity.invalidate` → `DELETE FROM kb_identity WHERE id=1`（已删则 0 行）。
- **与添加路径对称**：`client.py:172`（add）已调 `on_document_added`；此接线使删除路径对称（修复 P1-8）。

### 5.4 `server.py` delete handler 改动（端到端接线）—— FR1/FR2/FR4

**改动点**（`server.py:151-174`）：
1. `delete_document(db_id)` 调用**前**先取 `doc = c.db.get_document_by_id(db_id)`（保存 `pdf_path`，供磁盘清理；行级联删除后取不到）。
2. 保留 `c.db.delete_document(db_id)`（FR1；现状已调用但方法不存在 → 修复即生效）。
3. 在 `c.closet_index.remove_document(db_id)` **后**补 `c.super_tree_index.on_document_removed(db_id)`（FR2）。
4. 末尾补磁盘清理（FR4）：`os.remove(doc["pdf_path"])` + try/except + `workspace/uploads` 安全护栏。

**改动后结构**（伪代码，精确编辑留 code 阶段）：
```python
elif name == "delete_document":
    doc_id = arguments.get("doc_id", "")
    if not doc_id: return error
    c = get_client()
    # 内存态
    if doc_id in c.documents: del c.documents[doc_id]
    db_id = c._uuid_to_db.pop(doc_id, None)
    if db_id is not None and c.db is not None:
        # 磁盘路径须在级联删除前取出
        doc = c.db.get_document_by_id(db_id)
        pdf_path = doc.get("pdf_path") if doc else None
        # DB 级联删除 (FR1)
        try: c.db.delete_document(db_id)
        except Exception as e: logger.warning("Failed to delete document from DB: %s", e)
        # 索引失效 (FR2)
        if c.closet_index:
            try: c.closet_index.remove_document(db_id)
            except Exception as e: logger.warning("Failed to delete closet tags: %s", e)
        if c.super_tree_index:
            try: c.super_tree_index.on_document_removed(db_id)
            except Exception as e: logger.warning("Failed to invalidate super-tree index: %s", e)
        # 磁盘清理 (FR4) — 安全护栏：仅删 workspace/uploads 下文件
        if pdf_path:
            _safe_remove_upload(pdf_path, c.workspace)
    return {"success": True, "doc_id": doc_id}
```

**`_safe_remove_upload(pdf_path, workspace)`**（`server.py` 新增辅助，或内联）：
```python
def _safe_remove_upload(pdf_path: str, workspace) -> None:
    try:
        upload_dir = (Path(workspace) / "uploads").resolve()
        target = Path(pdf_path).resolve()
        # 护栏：仅删 uploads 目录下文件，防误删库外路径
        if upload_dir not in target.parents:
            logger.warning("Skipping disk cleanup: %s not under uploads/", pdf_path)
            return
        os.remove(target)
    except FileNotFoundError:
        pass  # 幂等
    except OSError as e:
        logger.warning("Failed to remove upload file %s: %s", pdf_path, e)
```

### 5.5 孤儿迁移脚本入口 —— FR5

**入口**：`python -m pageindex_mutil.migrations.cleanup_orphans --db-path <path>`（或在 `migrations/` 下一次性脚本；设计阶段定入口形式，code 阶段实现）。
- **位置建议**：`pageindex_mutil/migrations/cleanup_orphans.py`（新目录 `migrations/`，与 W1 解耦）。
- **幂等**：对每张子表执行 `DELETE FROM <child> WHERE doc_id NOT IN (SELECT id FROM documents)`，覆盖 `nodes`/`pages`/`closet_tags`/`doc_keywords`。
- **可选**：`--purge-kb-identity` flag 触发 `DELETE FROM kb_identity`（强制下次重建，清洗历史围栏脏值；非默认）。

---

## §6 数据/配置变更（Data & Config Changes）

### 6.1 Schema 变更
**无**。`documents`/`nodes`/`pages`/`closet_tags`/`doc_keywords` 的 `ON DELETE CASCADE` 已存在于 `ensure_schema`（db.py:41/53/61/73）。`delete_document` 仅发 `DELETE FROM documents`，依赖既有 cascade。NFR3 满足。

### 6.2 孤儿迁移 SQL（FR5 / NFR3）
```sql
-- 幂等，可重复执行；对已干净的库删 0 行
DELETE FROM nodes        WHERE doc_id NOT IN (SELECT id FROM documents);
DELETE FROM pages        WHERE doc_id NOT IN (SELECT id FROM documents);
DELETE FROM closet_tags  WHERE doc_id NOT IN (SELECT id FROM documents);
DELETE FROM doc_keywords WHERE doc_id NOT IN (SELECT id FROM documents);
-- 可选（--purge-kb-identity）：DELETE FROM kb_identity;
```
- **事务**：单事务包裹 4 条 DELETE（原子性；部分失败回滚）。
- **大库锁/性能**：见 §7。

### 6.3 upload_dir 路径解析（FR4）
- **权威源**：`documents.pdf_path` 列（绝对路径，上传时 `server.py:378` 生成 `{uuid}_{safe_name}`，`client.py:159` 存入）。
- **`pdf_name` 不可用于磁盘定位**：`pdf_name` 列存的是 `doc_name`（`get_pdf_name`→`sanitize_filename` 产物），与磁盘文件名 `{uuid}_{safe_name}` 不同。
- **护栏**：`_safe_remove_upload` 用 `Path.resolve()` + `uploads` 父目录校验，防误删库外文件（§5.4）。

### 6.4 配置变更
**无**。不引入新 env/config。`WORKSPACE`（server.py:42）既有，`upload_dir = workspace/"uploads"` 既有。

---

## §7 风险与缓解（Risk & Mitigation）

| # | 风险 | 影响 | 缓解 | 回滚 |
|---|------|------|------|------|
| R1 | **孤儿迁移在大库上的锁/性能**：`DELETE FROM <child> WHERE doc_id NOT IN (...)` 子查询对大表全扫，可能长锁 `documents` + 子表，阻塞在线读写 | 在线服务超时 | (a) 离线执行（停服窗口）；(b) 分批 DELETE（`LIMIT N` 循环，SQLite 需 `DELETE ... WHERE rowid IN (SELECT rowid FROM <child> WHERE doc_id NOT IN (...) LIMIT 1000)`）；(c) 迁移前备份 `pageindex.db` | 迁移脚本不提交 schema，仅 DML；失败回滚 = 还原 DB 备份 |
| R2 | **磁盘路径解析错误 / 误删**：`pdf_path` 可能指向库外（历史数据路径漂移）或已不存在 | 误删用户文件 / 误报错 | `_safe_remove_upload` 护栏：`Path.resolve()` + 校验在 `workspace/uploads` 下；`FileNotFoundError` 吞；`OSError` 仅 warning | 护栏命中即 skip，不删 |
| R3 | **cascade 完整性依赖 `PRAGMA foreign_keys=ON`**：若某连接未开 PRAGMA，`DELETE FROM documents` 不触发级联 → 子表孤儿 | 孤儿数据残留 | W1 保证 `_connect` 每连接 `PRAGMA foreign_keys=ON`（db.py:17 已有）；delete_document 用 `self._connect()`（W1 线程本地层） | 依赖 W1；W1 未落地则 W2 不合并 |
| R4 | **并发删除竞争**：两请求同时删同一 `doc_id` | 重复 `os.remove`（第二次 FileNotFoundError，幂等）；`_uuid_to_db.pop` 竞争（Python GIL + dict pop 原子） | NFR4 依赖 W1 线程本地连接；步骤全程幂等（DELETE 0 行 / remove FileNotFoundError / invalidate 0 行） | 幂等设计本身即缓解 |
| R5 | **`on_document_removed` 抛异常中断后续磁盘清理** | 磁盘文件残留 | 每步独立 try/except（§5.4）；`on_document_removed` 异常仅 warning，不阻断磁盘清理 | DB 已删（级联生效），索引/磁盘失败仅 warning |
| R6 | **`strip_markdown_fence` 误剥合法内容**：KBIdentity 摘要若合法含 ` ``` `（罕见） | 摘要内容被截 | 仅剥**最外层**围栏（`startswith("```")`）；合法纯文本摘要不含围栏 | 回滚 `_generate_with_llm` 改动 |
| R7 | **`delete_document` 时序错误**：先 `delete_document` 再取 `pdf_path` → 行已删，取 None → 跳过磁盘清理 | 磁盘文件残留（静默） | §5.4 强制 `get_document_by_id` 在 `delete_document` **之前**；§8 NFR1 断言磁盘文件已删 | code review + 测试守卫 |

---

## §8 验收标准（Acceptance Criteria）

> 映射 §3 FR1-FR5 / NFR1-NFR4。每条给出可执行断言。

### 8.1 FR1 — `delete_document` 级联删除
- **AC1.1**：插入文档 → 插入 nodes/pages/closet_tags/doc_keywords → `delete_document(doc_id)` → 断言 `documents`/`nodes`/`pages`/`closet_tags`/`doc_keywords` 中该 `doc_id` 行数为 0。
- **AC1.2**：`delete_document` 对不存在 `doc_id` 不抛异常（删 0 行）—— NFR2。

### 8.2 FR2 — `on_document_removed` 接线
- **AC2.1**：插入文档（触发 `on_document_added` 建 keyword + kb_identity）→ 删除（MCP `delete_document` 路径）→ 断言 `doc_keywords` 中该 `doc_id` 行数为 0（`keyword_index.remove_document` 生效）。
- **AC2.2**：删除后 `kb_identity` 表为空（`invalidate` 生效）→ 下次 `get_identity` 重建。
- **AC2.3**：`grep -n "on_document_removed" server.py` 命中 delete handler（接线存在）。

### 8.3 FR3 — KBIdentity 围栏剥离
- **AC3.1**：mock LLM 返回 ` ```text\n某摘要\n``` ` → 调 `_generate_with_llm` → 断言 `kb_identity.identity_text` **不含** ` ``` `（围栏已剥）。
- **AC3.2**：mock LLM 返回纯文本 `某摘要`（无围栏）→ 断言 `identity_text == "某摘要"`（幂等，不误剥）。

### 8.4 FR4 — 磁盘文件清理
- **AC4.1**：索引 PDF（上传到 `workspace/uploads/`）→ 删除 → 断言 `pdf_path` 文件 `os.path.exists() == False`。
- **AC4.2**：`_safe_remove_upload` 对 `workspace/uploads/` 外路径不调用 `os.remove`（护栏）。

### 8.5 FR5 — 孤儿迁移幂等
- **AC5.1**：预置孤儿行（`nodes.doc_id` 指向不存在 `documents.id`）→ 运行迁移 → 断言孤儿行已删。
- **AC5.2**：再次运行迁移 → 删 0 行，无报错（幂等，NFR3）。

### 8.6 NFR1 — 集成测试（端到端）
- **AC6.1**：插入 → 删除（MCP 路径）→ 断言：(a) `documents`/`nodes`/`pages`/`closet_tags`/`doc_keywords` 行清除；(b) `kb_identity` 失效（表空）；(c) 磁盘 PDF 文件移除；(d) **`kb_identity` 存储值不含围栏标记**（quality-gate 建议吸收）。

### 8.7 NFR2 — 删除幂等
- **AC7.1**：删除不存在的 `doc_id` → 返回 `{"success": True}`，无异常。
- **AC7.2**：连续两次删同一 `doc_id` → 第二次无异常（`os.remove` FileNotFoundError 吞，DELETE 0 行）。

### 8.8 NFR3 — 向后兼容
- **AC8.1**：`grep -n "ALTER TABLE\|CREATE TABLE" db.py` 中无 W2 新增（无 schema 变更）。
- **AC8.2**：迁移脚本可重复执行（AC5.2）。

### 8.9 NFR4 — 并发删除安全
- **AC9.1**：依赖 W1 线程本地连接层；`delete_document` 用 `self._connect()`（W1 保证）。W1 测试覆盖并发；W2 不重复测，仅断言 `delete_document` 不自建连接（`grep "sqlite3.connect" db.py | grep delete_document` 为空）。

### 8.10 过程纪律（devkit 铁律）
- **Two-Agent Minimum**：W2 code 阶段由子代理（leaf executor）实现 `delete_document`/接线/迁移/测试，由**不同**子代理（independent verifier）独立跑 §8 守卫 → quality-gate。禁止自审自评。
- **TDD/L2**：`delete_document` 实现前先写失败测试（AC1.1 RED）→ 实现 → GREEN。
- **依赖 W1**：W2 合并前 W1 必须已落地（R3）；否则 cascade 不保证生效。

---

## 设计阶段完成检查
- [x] §4 删除流程（db.delete_document → cascade → on_document_removed 失效 → 磁盘清理）、KBIdentity 清洗、孤儿迁移流程
- [x] §5 接口设计：`delete_document` 签名、`on_document_removed` 接线点、`strip_markdown_fence` 清洗函数、迁移脚本入口、server.py delete handler 改动
- [x] §6 数据/配置变更：无 schema 变更（cascade 已存在）；孤儿迁移 SQL；upload_dir 路径解析（pdf_path 权威源）
- [x] §7 风险与缓解：7 条（孤儿迁移锁/性能、磁盘路径解析、cascade 完整性、并发删除、异常中断、围栏误剥、时序错误）
- [x] §8 验收标准：映射 §3 FR1-FR5 / NFR1-NFR4（含 NFR1 围栏标记断言；FR3 收敛单一剥离语义；幂等删除；迁移幂等）
> 设计完成。下一步：独立 agent 调 quality-gate（threshold ≥70）→ 通过后进入 plan 阶段（task-planning → tasks.md），W2 依赖 W1 先落地。
