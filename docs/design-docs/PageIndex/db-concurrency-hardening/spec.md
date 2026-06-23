# W1 · DB 并发安全与数据层加固 — spec.md

> 阶段：clarify（§1-3 待确认）。本文件仅含需求，**不含技术设计**（连接模型选择等在 system-design 阶段）。
> 来源：`docs/design-docs/PageIndex/architecture-review-2026-06/review-report.md` P0-1 / P0-3 / P1-13 / P2-15。

## §1 背景

`PageIndexDB`（`db.py`）是整个系统的数据层，被 `main.py`、`server.py`、`pageindex_mutil/{client,super_tree,closet_index}` 共享。当前实现存在三组相互叠加的缺陷：

1. **并发不安全（P0-1）**：`db.py:16` `sqlite3.connect(self.db_path)` 使用默认 `check_same_thread=True`，而连接被缓存为单实例并跨 `asyncio.to_thread` 共享——服务器 `_UPLOAD_SEMAPHORE=Semaphore(3)` 允许 3 个并发上传线程，`router._run_strategies` 多策略也跨线程访问同一连接。并发上传或 索引+搜索 会触发 `sqlite3.ProgrammingError` 或状态损坏，部分提交导致库不一致。
2. **无 WAL / pragma（P2-15）**：仅 `PRAGMA foreign_keys=ON`，默认 `journal_mode=DELETE`——写阻塞读，并发索引与搜索会出现 `database is locked`（默认 5s busy_timeout 甚至未设置）。
3. **缺索引（P0-3 / P1-13）**：`nodes`/`pages` 无专用 `doc_id` 索引，`nodes.parent_node_id` 完全无索引；`_build_super_tree` 的 per-node `COUNT(*) FROM nodes WHERE parent_node_id=?` 退化为全表扫描，直接危及 super-tree-v3 NFR1（L0+L1 < 1s）。

CLI 路径（`main._index_files_batch`）已通过"每线程独立 `PageIndexDB`"规避了线程安全问题，但服务器/路由路径未规避——修复须统一在数据层解决。

## §2 目标

- 并发操作**安全**：≥2 文件并发上传 + 并发搜索不崩溃、不损坏数据。
- 支持**读-写并发**：WAL 模式允许写期间并发读。
- 为 L1 热路径提供**索引**支撑，解除 NFR1 的架构性障碍（与 W3 的查询/缓存改造协同达成 < 1s）。
- 为 **W2 的 `delete_document`** 提供并发安全的数据层基础（W2 依赖本工作先落地）。
- 对现有 `pageindex.db` **透明**：索引/pragma 为附加式，无破坏性迁移。

## §3 需求与范围

### 功能需求
- **FR1**：`PageIndexDB` 连接在多线程 `asyncio.to_thread` 访问下安全（无 `ProgrammingError`、无损坏），覆盖并发上传 + 并发搜索。
- **FR2**：启用 `journal_mode=WAL`、`synchronous=NORMAL`、`busy_timeout`（具体值在设计中定），支持写期间并发读。
- **FR3**：新增索引 `idx_nodes_doc_id`、`idx_nodes_parent_node_id`、`idx_pages_doc_id`（`CREATE INDEX IF NOT EXISTS`，幂等）。

### 非功能需求
- **NFR1**：并发正确性——并发上传 + 并发搜索集成测试通过，无崩溃/损坏。
- **NFR2**：索引添加对现有数据透明（连接时 `CREATE INDEX IF NOT EXISTS`）。
- **NFR3**：不破坏 `PageIndexDB` 公开方法签名。
- **NFR4**：WAL/索引对磁盘开销可接受（sqlite WAL 文件 + 索引体积）。

### 范围
- **IN**：`db.py` 连接/pragma/索引；保持上层（`server.py`/`router.py`）调用方式不变（透明）。
- **OUT**（明确排除，留待后续）：
  - 连接模型选择（`check_same_thread=False`+写锁 vs 每线程连接 vs 每请求连接）= **system-design 阶段决策**。
  - `_build_super_tree` 的 N+1 查询重写（W3，本工作仅提供索引支撑）。
  - 检索路径 async LLM 改造（W3）。
  - 现有数据的内容迁移（索引/pragma 为附加式，无需迁移）。

### 假设（ASSUMPTION，需确认）
- 无已知线上部署实例需协调（本地 `pageindex.db`）。
- 并发目标沿用现有 `Semaphore(3)` 量级（设计中可调整）。

---

> 🚨 **STOP — REQUIREMENTS_CONFIRMED**
> 待用户确认本 spec §1-3 后，由独立 agent 调用 quality-gate（threshold ≥70），通过后进入 system-design 阶段。

---

## §4 系统设计 — 方案设计与权衡（Alternatives & Trade-offs）

### 4.0 设计输入（已核实代码）

- **`db.py:14-19` `_connect()`**：`sqlite3.connect(self.db_path)` 默认 `check_same_thread=True`，连接缓存为 `self._conn` 单例，仅设 `PRAGMA foreign_keys = ON`。根因：单连接被跨 `asyncio.to_thread` 的多 worker 线程共享 → 触发 `ProgrammingError`（线程不匹配）或交错访问损坏状态。
- **`server.py:295` `_UPLOAD_SEMAPHORE = asyncio.Semaphore(3)`** + `_index_one_file`（server.py:308-330）：`async with _UPLOAD_SEMAPHORE: doc_id = await asyncio.to_thread(get_client().index, ...)`。最多 3 个上传并发运行于线程池，共享 `get_client()` 返回的全局 `PageIndexClient` 的 `self.db`（单一 `PageIndexDB` 实例 → 单一缓存连接）。
- **`router.py:121-146` `_run_strategies`**：`asyncio.to_thread(self.metadata_strategy.search, ...)` / `semantics_strategy.search` / `description_strategy.search` 并行 gather。各策略经 `ClosetIndex`/`SuperTreeIndex` 间接读 `self.db`（`match_closet_tags`/`match_doc_keywords`/`get_all_documents` 等），与上传路径的写线程并发。
- **`server.py:437-459` `lifespan`**：全局 `client = PageIndexClient(...)`，shutdown 时 `client.close()` → `self.db.close()`。连接生命周期绑定进程。
- **`main.py:355-408` `_index_files_batch`**（CLI 规避路径）：`ThreadPoolExecutor(max_workers=3)`，每个 `_index_one` 内 `db = PageIndexDB(db_path)` 新建独立实例 + `db.close()`。证明"每线程独立连接"可行，但重复建连且无 WAL/索引/pragma。
- **表定义（`db.py:31-84`）**：`documents` / `nodes` / `pages` / `closet_tags` / `doc_keywords` / `kb_identity`。已有索引仅 `idx_closet_tags_token(doc_id, tag_token)`、`idx_doc_keywords(keyword, doc_id)`。`nodes`/`pages` 无专用索引；`nodes.parent_node_id` 完全无索引。

### 4.1 替代方案（连接模型）

#### 方案 A — `check_same_thread=False` + 全局写锁
单一缓存连接设 `check_same_thread=False`，所有写操作外加 `threading.Lock` 串行化，读不加锁。
- ✅ 改动最小（仅 `_connect` + 一把锁）。
- ❌ **不解决 WAL 缺失**（写仍阻塞读）；锁粒度粗，并发上传吞吐被锁串行化；读期间连接对象状态（如未提交事务）可能被写线程污染 → 隐性损坏风险。不满足 FR2。

#### 方案 B — 线程本地连接池 + WAL ⭐ 推荐（用户已选）
每个 `ThreadPoolExecutor` worker 线程持有 thread-local 连接（`threading.local` 或 `contextvars`），跨 `asyncio.to_thread` 调用复用同一连接；数据库级一次开启 WAL + `synchronous=NORMAL`；每连接设 `busy_timeout`；线程退出/进程关闭时清理连接。
- ✅ 消除"跨线程共享单连接"根因（每线程独占连接，无 `ProgrammingError`）；WAL 解锁读-写并发；连接复用避免每次 `to_thread` 重建连接开销；与 CLI 已验证的"每线程独立连接"思路一致，但改为 thread-local 复用而非每次新建。
- ✅ 透明：`PageIndexDB` 公开方法签名不变，上层 `server.py`/`router.py`/`main.py` 调用方式不变。
- ❌ 需管理 thread-local 连接生命周期（线程退出需清理，否则连接泄漏）；WAL 在 NFS 等网络文件系统不安全（本项目本地 `pageindex.db`，假设满足，见 §3 假设）。

#### 方案 C — 每请求连接（无池化）
每次方法调用 `_connect()` 新建连接、用完即 `close()`。
- ✅ 彻底无跨调用状态，最简单正确。
- ❌ 每次读/写都重建连接 + 重设 pragma + 重 `ensure_schema` 检查，高频查询（`_build_super_tree` per-node COUNT、检索多策略）性能退化明显；与 NFR1（super-tree NFR1 < 1s 架构支撑）冲突。不推荐。

### 4.2 推荐：方案 B（线程本地连接池 + WAL）

**根因匹配**：P0-1 的根因是"单一缓存连接被多线程共享"。方案 B 让每线程拥有独占连接，从结构上消除跨线程访问；WAL + `busy_timeout` 解决 P2-15 的写阻塞读；索引（§6）解决 P0-3/P1-13。三者叠加，一条数据层改动同时覆盖三组缺陷，且对上层透明。

**WAL 写者并发说明**：SQLite WAL 模式下**同一时刻只允许一个写者**（多写者会通过 `busy_timeout` 排队等待）。本设计并发目标为 ≥3 并发上传（对齐 `Semaphore(3)`）——这 3 个上传线程中，写操作（`insert_document`/`insert_nodes`/`insert_pages` 等）会经 `busy_timeout` 串行化排队，而非真正并行写。这是 SQLite 的固有约束（非本设计引入），WAL 的收益在于**读不被写阻塞**（搜索/检索可并发于上传）。设计中 `busy_timeout` 取值须 ≥ 单次写最坏时延 × 排队深度（见 §5.2）。

### 4.3 设计决策（DESIGN_DECISION — 已确认）
✅ **用户选择方案 B（线程本地连接池 + WAL）**。连接模型 = thread-local 连接 + WAL + `synchronous=NORMAL` + 每连接 `busy_timeout`。索引见 §6。

---

## §5 接口设计（Interface Design）

### 5.1 `PageIndexDB` 公开方法（不变，透明）
所有现有公开方法签名与语义保持不变，上层调用方无需改动：
- 写：`insert_document`、`update_document_tree`、`update_document_description`、`insert_nodes`、`insert_pages`、`insert_closet_tags`、`delete_closet_tags`、`insert_doc_keywords`、`delete_doc_keywords`、`set_kb_identity`。
- 读：`get_document_by_name`、`get_document_by_id`、`get_all_documents`、`get_node`、`get_nodes_by_ids`、`get_nodes_by_doc_id`、`get_top_level_nodes`、`get_pages_in_range`、`get_pages_by_numbers`、`match_closet_tags`、`match_doc_keywords`、`get_kb_identity`。
- 生命周期：`__init__`、`ensure_schema`、`close`。

> NFR3 约束：公开方法签名零变更。上层 `server.py`（`get_client().index` → `self.db.insert_document`...）、`router.py`（经 `ClosetIndex`/`SuperTreeIndex` 间接调 db）、`main.py`（`db.insert_nodes`...）调用方式均不变。

### 5.2 `_connect()` 行为变更（内部实现变更，签名不变）

**变更前**（`db.py:14-19`）：
```python
def _connect(self):
    if self._conn is None:
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.row_factory = sqlite3.Row
    return self._conn
```

**变更后（方案 B）**：
- `_connect()` 改为返回**当前线程的 thread-local 连接**。用 `threading.local()` 实例存储 per-thread 连接（`contextvars` 亦可行，但 `threading.local` 足够——`asyncio.to_thread` 底层就是 `ThreadPoolExecutor` 线程）。
- 连接创建时设置 pragma（见 §6.1）：`journal_mode=WAL`（db 级一次，但每连接设置幂等）、`synchronous=NORMAL`、`busy_timeout=<值>`、`foreign_keys=ON`、`row_factory=Row`。
- `busy_timeout` 取值：默认 SQLite 为 0（不等待即 `database is locked`）。设为 **5000ms**（5s）以覆盖：单次写通常 < 100ms，3 并发上传排队深度 ≤ 3，5s 足以吸收正常排队；超时则抛 `OperationalError`（可观测，优于静默损坏）。若实测写时延偏高可上调，但须文档化。
- `ensure_schema()` 在首次连接创建时调用（索引 DDL 幂等，见 §6.2），使每个新 thread-local 连接首次使用时即具备索引。
- `self._conn`（实例缓存单连接）废弃或仅用于主线程兼容（`main.py` CLI 单线程路径仍可用单实例，thread-local 退化为该线程的连接）。

**伪代码（示意，非最终实现）**：
```python
class PageIndexDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self._local = threading.local()
        self._tls_connections = []  # 跨线程登记，供 close() 遍历
        self._tls_lock = threading.Lock()
        self.ensure_schema()  # 首次建表 + 索引（幂等）

    def _connect(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA busy_timeout = 5000")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
            with self._tls_lock:
                self._tls_connections.append(conn)
            self._ensure_indexes(conn)  # 幂等 CREATE INDEX IF NOT EXISTS
        return conn

    def close(self):
        # 关闭当前线程 + 所有已登记的 thread-local 连接
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
        with self._tls_lock:
            for c in self._tls_connections:
                try: c.close()
                except Exception: pass
            self._tls_connections.clear()
```

### 5.3 索引创建接口
- 新增内部方法 `_ensure_indexes(conn)`（或并入 `ensure_schema` 的 `executescript`），执行 §6.2 的三条 `CREATE INDEX IF NOT EXISTS`。幂等，每连接首次创建时调用一次（或在 `ensure_schema` 的 `executescript` 中统一执行，因 schema DDL 本身幂等）。
- 无新增公开方法（索引创建为内部细节）。

### 5.4 上层影响（透明，无改动）
- `server.py`：`get_client().index` → `PageIndexClient.index` → `self.db.insert_document` 等，调用方式不变。`lifespan` 的 `client.close()` → `self.db.close()` 自动清理所有 thread-local 连接。
- `router.py`：`_run_strategies` 的 `asyncio.to_thread(...search...)` 不变，各策略经 `ClosetIndex`/`SuperTreeIndex` 读 db 透明复用 thread-local 连接。
- `main.py`：`_index_files_batch` 的"每线程 `PageIndexDB(db_path)`"仍可工作（每实例独立 thread-local，单连接），但可简化为共享单实例（thread-local 自动隔离）。本工作不强制改 main.py，保持现状以降风险。

---

## §6 数据/配置变更（Data & Configuration Changes）

### 6.1 SQLite Pragma（每连接设置，幂等）
| Pragma | 值 | 作用 | 级别 |
|--------|-----|------|------|
| `journal_mode` | `WAL` | 写不阻塞读（P2-15） | db 级一次生效，但每连接设置幂等 |
| `synchronous` | `NORMAL` | WAL 下安全且更快（FULL 太慢） | 每连接 |
| `busy_timeout` | `5000`（ms） | 写冲突时等待 5s 再抛 `database is locked` | 每连接 |
| `foreign_keys` | `ON` | 保留现有行为 | 每连接（现有已设） |
| `row_factory` | `sqlite3.Row` | 保留现有行为 | 每连接（现有已设） |

> `check_same_thread=False`：因 thread-local 保证每连接仅被所属线程使用，设 `False` 仅为消除 sqlite3 默认线程检查的误报（如 asyncio 线程池复用线程时的边界）。实际隔离由 thread-local 保证，非由 `check_same_thread` 保证。

### 6.2 索引 DDL（幂等，附加式，无破坏性 schema 变更）
```sql
CREATE INDEX IF NOT EXISTS idx_nodes_doc_id
    ON nodes(doc_id);

CREATE INDEX IF NOT EXISTS idx_nodes_parent_node_id
    ON nodes(parent_node_id);

CREATE INDEX IF NOT EXISTS idx_pages_doc_id
    ON pages(doc_id);
```
- **`idx_nodes_doc_id`**：加速 `get_nodes_by_doc_id`、`get_top_level_nodes`（`WHERE doc_id = ?`）、`insert_nodes`/`insert_pages` 的 `DELETE FROM nodes WHERE doc_id = ?` 前置查找。
- **`idx_nodes_parent_node_id`**：解除 P1-13 —— `_build_super_tree` per-node `COUNT(*) FROM nodes WHERE parent_node_id = ?` 退化为全表扫描的根因。索引后变为索引扫描。
- **`idx_pages_doc_id`**：加速 `get_pages_in_range`、`get_pages_by_numbers`（`WHERE doc_id = ? AND page_number ...`）。
- **创建时机**：`ensure_schema()` 的 `executescript` 内追加（与现有 `idx_closet_tags_token`、`idx_doc_keywords` 并列），或新增 `_ensure_indexes(conn)` 在 thread-local 连接首次创建时调用。二者皆幂等，选其一（推荐并入 `ensure_schema`，因 `executescript` 已含其他 `CREATE INDEX IF NOT EXISTS`）。

### 6.3 WAL 文件
- 启用 WAL 后，数据库目录新增 `pageindex.db-wal`（WAL 日志）与 `pageindex.db-shm`（共享内存索引）。
- **checkpoint**：SQLite 默认自动 checkpoint（`PRAGMA wal_autocheckpoint=1000`，每 1000 页提交一次）。本设计采用默认，不手动 checkpoint。
- **清理**：进程正常关闭时 SQLite 自动 checkpoint 并删除 `-wal`/`-shm`（或保留为空）。异常崩溃后下次打开自动恢复。
- **备份/部署注意**：备份 `pageindex.db` 时须一并备份 `-wal`/`-shm`（若有）。本工作无线上部署实例（§3 假设），本地开发无需特殊处理。

### 6.4 无破坏性 schema 变更
- 无 `ALTER TABLE`、无 `DROP`、无列变更。
- 所有 DDL 为 `CREATE ... IF NOT EXISTS`，对现有 `pageindex.db` 附加式生效，首次连接即创建索引（NFR2）。

---

## §7 风险与缓解（Risk & Mitigation）

| 风险 | 影响 | 缓解 | 回滚 |
|------|------|------|------|
| **R1 thread-local 连接泄漏**（线程退出未清理） | 连接句柄累积，`db close` 后悬挂连接写失败 | (a) `_tls_connections` 登记所有连接，`close()` 遍历关闭；(b) `ThreadPoolExecutor` 线程复用而非频繁退出，泄漏概率低；(c) 进程 shutdown 时 `lifespan` → `client.close()` → `db.close()` 兜底清理 | revert `_connect` 改动 |
| **R2 WAL 写者限制 vs 池大小配比** | SQLite WAL 单写者，3 并发上传写会排队，`busy_timeout` 超时则 `database is locked` | (a) `busy_timeout=5000ms` 覆盖正常排队；(b) 实测写时延，必要时上调或下调 `Semaphore`；(c) 写操作本身较短（单文档 insert 秒级内），3 排队可吸收；(d) 文档化"WAL 单写者"约束，避免误设过高并发 | 调整 `busy_timeout` 或 `Semaphore` 值 |
| **R3 连接泄漏检测缺失** | 无测试覆盖 thread-local 连接清理 | §8 验收新增连接计数测试（创建 N 线程 → 关闭 → 断言所有连接已 close） | 补测试 |
| **R4 现有库迁移**（`pageindex.db` 已有数据） | 索引创建锁库短暂阻塞；WAL 切换需无活跃连接 | (a) `CREATE INDEX IF NOT EXISTS` 幂等，首次连接自动建；(b) `journal_mode=WAL` 切换在无写事务时即时生效；(c) 本地库无线上实例，无停机协调 | 无需迁移（附加式） |
| **R5 WAL 文件磁盘开销** | `-wal`/`-shm` + 3 索引增加磁盘占用 | NFR4 量化阈值（见下）：WAL+索引体积 ≤ 现有 db 体积 1.5 倍；默认 auto-checkpoint 控制 WAL 增长 | 关闭 WAL + 删索引（revert） |
| **R6 NFS/网络文件系统** | WAL 在网络 FS 不安全（SQLite 已知限制） | 假设本地 FS（§3 A1）；若部署到网络卷须改 `journal_mode=DELETE` + 接受读写互斥 | 降级为方案 A |
| **R7 `check_same_thread=False` 误用** | 若 thread-local 隔离失效，`False` 会掩盖跨线程访问 | (a) thread-local 保证每连接单线程使用；(b) 集成测试覆盖并发场景验证无 `ProgrammingError` | 改回 `True` + 方案 C |
| **R8 部署迁移（阻塞假设）** | §3 假设"无线上部署实例"——**标注为阻塞假设**：若实际存在线上部署，须迁移策略（停机切换 WAL + 建索引、备份 `-wal`/`-shm`） | 用户确认假设前不进入实现；若部署存在，补充迁移 runbook | — |

### 7.1 NFR4 磁盘开销量化阈值
- **WAL 文件**：默认 auto-checkpoint（1000 页 ≈ 4MB），稳态 `-wal` 体积有界（通常 < 10MB）。
- **索引体积**：3 索引，`nodes`/`pages` 行数与文档页数/节点数成正比。典型文档（数百页、数百节点）单索引 < 1MB，3 索引合计 < 3MB。
- **阈值**：**WAL + 索引合计磁盘开销 ≤ 现有 `pageindex.db` 体积的 1.5 倍**（NFR4 可接受）。若现有 db 为 X，则新增开销 ≤ 0.5X。验收时实测并记录。

---

## §8 验收标准（Acceptance Criteria）

> 逐条映射 §3 FR/NFR 到可测试条件。每条须有对应测试或手动验证步骤。

### 8.1 FR 映射

| FR | 验收条件 | 验证方式 |
|----|---------|---------|
| **FR1**（多线程 `asyncio.to_thread` 安全） | 并发上传 ≥3 文件 + 并发搜索不崩溃、不损坏；无 `sqlite3.ProgrammingError` / `database is locked`（`busy_timeout` 内） | 集成测试：`ThreadPoolExecutor(max_workers=3)` 并发 `index` 3 个不同文件 + 并发 `search`；断言全部成功、db 一致性检查通过 |
| **FR1（并发度参数）** | 并发度 ≥3（对齐 `Semaphore(3)`） | 测试中显式 `max_workers=3` 并验证 3 线程并发执行；`_UPLOAD_SEMAPHORE` 值不变（3） |
| **FR2**（WAL + synchronous + busy_timeout） | `PRAGMA journal_mode` 返回 `wal`；`synchronous` 返回 `1`（NORMAL）；`busy_timeout` 返回 `5000` | 测试：新建 `PageIndexDB`，查 `PRAGMA journal_mode`/`synchronous`/`busy_timeout` 断言值 |
| **FR3**（3 索引存在） | `sqlite_master` 含 `idx_nodes_doc_id`、`idx_nodes_parent_node_id`、`idx_pages_doc_id` | 测试：`SELECT name FROM sqlite_master WHERE type='index'` 断言含 3 索引 |

### 8.2 NFR 映射

| NFR | 验收条件 | 验证方式 |
|-----|---------|---------|
| **NFR1**（并发正确性） | 并发上传 + 并发搜索集成测试通过，无崩溃/损坏；数据一致性校验（行数、外键完整）通过 | 集成测试 + `PRAGMA foreign_key_check` 断言无违反 |
| **NFR2**（索引对现有数据透明） | 对已有数据的 `pageindex.db` 打开，索引自动创建（`IF NOT EXISTS`），无数据丢失 | 测试：预置数据 → 新建 `PageIndexDB` → 查索引存在 → 查数据行数不变 |
| **NFR3**（公开方法签名不变） | `PageIndexDB` 公开方法签名 diff 为空；`server.py`/`router.py`/`main.py` 调用点无改动 | 静态检查：`git diff` 仅 `db.py`；上层文件无改动（或仅 import 无关） |
| **NFR4**（磁盘开销可接受） | WAL + 索引合计体积 ≤ 现有 db 体积 1.5 倍（§7.1 阈值） | 测试：建库 → 插入 N 文档 → 测 `pageindex.db` + `-wal` + `-shm` 体积 → 断言 ≤ 1.5 × 纯 db 体积 |

### 8.3 额外验收（连接生命周期）

| 项 | 验收条件 | 验证方式 |
|----|---------|---------|
| 连接清理 | N 线程各创建 thread-local 连接后 `close()`，所有连接已关闭（无句柄泄漏） | 测试：`ThreadPoolExecutor(max_workers=3)` 各线程 `_connect()` → 主线程 `db.close()` → 断言 `_tls_connections` 全部 close |
| 线程复用 | 同一线程多次 `to_thread` 调用复用同一连接（非每次新建） | 测试：同线程 `_connect()` 两次返回同一连接对象（`is` 相等） |

### 8.4 验证命令（基线 + 守卫）
```bash
# 基线：现有测试 GREEN
pytest -q tests/test_db.py tests/test_super_tree.py tests/test_client_integration.py

# 守卫：pragma 验证
python -c "from db import PageIndexDB; d=PageIndexDB('pageindex.db'); c=d._connect(); print(c.execute('PRAGMA journal_mode').fetchone()[0], c.execute('PRAGMA synchronous').fetchone()[0], c.execute('PRAGMA busy_timeout').fetchone()[0])"
# 期望：wal 1 5000

# 守卫：索引存在
python -c "import sqlite3; c=sqlite3.connect('pageindex.db'); print([r[0] for r in c.execute(\"SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'\").fetchall()])"
# 期望：含 idx_nodes_doc_id, idx_nodes_parent_node_id, idx_pages_doc_id（+ 现有 idx_closet_tags_token, idx_doc_keywords）

# 并发集成测试（新增）
pytest -q tests/test_db_concurrency.py -k "concurrent_upload_and_search"
```

### 8.5 过程纪律（devkit 铁律）
- **Two-Agent Minimum**：实现阶段由子代理（leaf executor）编码，由**不同**子代理（independent verifier）独立跑 §8 守卫——禁止自审自评。
- **TDD/L2**：先写失败测试（并发上传+搜索 → 当前崩溃 RED）→ 实现 _connect 改造 + WAL + 索引 → 测试转 GREEN。
- 实现产出 handoff.md（NEEDS_INDEPENDENT_VERIFICATION）→ verifier 独立跑 §8 守卫 → quality-gate。

---

## 设计阶段完成检查
- [x] §4 ≥2 替代方案 + 权衡（A/B/C），用户选 B
- [x] §5 接口设计（公开方法不变 + `_connect` 行为变更说明 + 索引创建接口）
- [x] §6 数据/配置变更（WAL pragma + 索引 DDL + WAL 文件 + 无破坏性 schema 变更）
- [x] §7 风险与缓解（含 R1-R8，含 R8 部署迁移阻塞假设标注）
- [x] §8 验收标准（逐条映射 FR1-FR3 / NFR1-NFR4 到可测试条件，含并发度参数 ≥3、NFR4 量化阈值 1.5×、连接生命周期验收）
- [x] quality-gate 建议吸收：NFR4 量化（§7.1）、FR1 并发度参数 ≥3（§8.1）、"无线上部署"标注为阻塞假设（R8）

> 设计完成。下一步：plan 阶段（task-planning → tasks.md），拆分实现任务（_connect 改造、WAL/pragma、索引、并发集成测试、连接生命周期测试）。
