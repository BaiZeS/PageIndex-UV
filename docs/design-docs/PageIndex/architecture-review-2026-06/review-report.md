# PageIndex-UV 全项目架构审查报告

| 项目 | PageIndex-UV |
|------|-------------|
| 审查日期 | 2026-06-23 |
| 当前分支 | `feat/unify-model-name-env` |
| 审查范围 | 整个项目当前架构（全代码库审计，非 diff 审查） |
| 审查深度 | 深度多维审计：架构 + 性能 + 健壮性/安全 + 工程规范 + 契约/需求符合度 |
| 产出形式 | 审查报告 + 优先级优化方案（不含代码改动） |
| 审查方法 | 5 个独立专项 reviewer 并行 → 3 个对抗性 critic refute-by-default → 综合裁决 |
| 验证结果 | 单源 P0/P1 命题 18/18 对抗确认（0 反驳）；关键问题多源交叉确认 |

> 本报告由编排器汇总 5 名 INDEPENDENT VERIFIER（reviewer）与 3 名 review-critic 的独立产出。所有 finding 均附 `file:line` 代码证据；对抗性验证逐条重读代码，默认尝试反驳。**本报告不修改任何代码**——所有优化项的落地须经完整代码工作流（spec→design→code→verify→review，见 AGENTS.md）。

---

## 1. 执行摘要

### 1.1 整体判定

PageIndex-UV 的**设计意图是先进的**（非向量检索 + Super-Tree L0/L1 两级选档 + Agentic Plan/Route/Act/Verify + CRAG 校验 + RRF 多策略融合），`project-refactor` 已成功消除旧 fork 嵌套与 `sys.path` hack，`feat/unify-model-name-env` 在 `utils.py`/`ConfigLoader` 路径上达成了模型名统一的**主目标**。

但**实现层存在系统性缺陷**，集中体现为五条根因线索，它们彼此耦合、相互放大：

1. **库反向依赖入口脚本**（`pageindex_mutil` 在运行期 `import main`）——这是可测试性缺失、库不自包含的直接根因，且与 refactor 的 G2/G3 目标矛盾。
2. **双检索管线无共享核心**——CLI（`answer_multidoc`）与 HTTP（`AgenticRouter`）各跑一套编排，行为发散（含 token 预算不一致）。
3. **数据层并发不安全**——sqlite 默认 `check_same_thread=True` + 单连接被 `asyncio.to_thread` 跨线程共享，并发上传/搜索会崩溃或损坏。
4. **删除路径完全失效**——`server.py:164` 调用不存在的 `db.delete_document()`，被 `except Exception` 静默吞掉，文档从不真正删除，且不失效 KB 索引/身份缓存。
5. **性能 NFR 架构性不可达**——L2 Act 串行 LLM + 同步 LLM 阻塞事件循环 + L1 N+1 + 无缓存，`<2s`/`<1s` 目标在非平凡语料下无法满足。

此外存在一批安全加固缺口（API key 非常量时间比较、上传无大小限制、prompt 注入、错误体信息泄露）与工程规范债务（`page_index.py` 1153 行上帝模块且零测试、`main.py` 65 处 print 无 logging、硬编码 `MODEL="gpt-4.1"` 残留）。

### 1.2 关键数字

| 维度 | 数量 | 说明 |
|------|------|------|
| P0（阻断/数据/安全） | 5 | 全部 CONFIRMED，2 条多源交叉确认 |
| P1（高优先） | 16 | 对抗验证 18 条中的 13 条 + 交叉确认 3 条 |
| P2（中/可维护性） | 20 | |
| P3（低） | 7 | |
| **合计** | **48** | 去重合并后 |

---

## 2. 审查方法与质量门

### 2.1 流程（devkit:code-review）

```
解析变更范围(=全项目) → 5 专项 reviewer 并行 → 对抗性 critic(refute-by-default) → Judge 综合 → 质量门
```

### 2.2 参与代理

| 角色 | 维度 | 状态 |
|------|------|------|
| reviewer-architecture | 架构与结构 | 12 findings |
| reviewer-performance | 性能 | 13 findings |
| reviewer-robustness | 健壮性与安全 | 12 findings |
| reviewer-standards | 工程规范与可维护性 | 9 findings |
| reviewer-contracts | 契约/信任链/需求符合度 | 12 findings |
| critic-perf | 对抗验证 A1–A7 | 7/7 CONFIRMED（A2 归因修正） |
| critic-security | 对抗验证 B1–B8 | 8/8 CONFIRMED（B1 严重度降级 + 新增信息泄露子项） |
| critic-arch | 对抗验证 C1–C3 | 3/3 CONFIRMED |

### 2.3 质量门结论

- **证据完整性**：100% finding 附 `file:line` + 代码片段，对抗验证逐条重读源码，未出现"仅凭 reviewer 转述"。
- **误报率**：18 条单源 P0/P1 命题 0 反驳；交叉确认问题 0 反驳。**未发现误报**。
- **遗漏补全**：critic 在对抗过程中**新增发现**（500 错误体信息泄露 CWE-209、`build_context_with_budget` 死注册、Semaphore 虚假安全、删除不失效身份的耦合放大），已并入报告。
- **严重度校准**：对抗验证对 2 条做了校准（见 §3 注），已采纳。

> 质量门由对抗性 critic（独立 agent）实质承担，符合 execution-discipline 的 Two-Agent Minimum 与"编排器不自审"原则。

---

## 3. Findings（按严重度）

> 标注说明：`[交叉确认×N]`=N 名独立 reviewer 同时发现；`[对抗确认]`=critic 重读源码确认；严重度经对抗校准的以 ★ 标注。

### 3.1 P0 — 阻断级（5）

#### P0-1 · SQLite 线程安全违规（并发损坏/崩溃）
- **位置**：`db.py:16`（`self._conn = sqlite3.connect(self.db_path)`，默认 `check_same_thread=True`）；`server.py:295`（`_UPLOAD_SEMAPHORE=Semaphore(3)`）、`server.py:316-317`（`asyncio.to_thread`）、`router.py:125-135`（`_run_strategies` 多策略 fan-out）
- **证据**：`PageIndexDB._connect` 缓存单连接；服务器全局 `client` 持单 `db` 实例；3 并发 `to_thread(index)` 与并行策略各自调用 `db.match_doc_keywords`/`match_closet_tags` → 同一连接跨线程使用 → `sqlite3.ProgrammingError` 或状态损坏。
- **影响**：任何 ≥2 文件并发批量上传、或并行 索引+搜索 → 随机崩溃 + 部分提交导致库不一致。
- **严重度**：P0 ｜ **[交叉确认×2：performance + robustness]**
- **置信度**：高

#### P0-2 · `delete_document` 调用不存在的方法（删除静默失效）
- **位置**：`server.py:164`（`c.db.delete_document(db_id)`）；`db.py` 无此方法（仅有 `delete_closet_tags`/`delete_doc_keywords`）
- **证据**：`except Exception`（`server.py:165`）静默吞掉 `AttributeError`，返回 `{"success": True}`。`documents`/`nodes`/`pages` 表虽有 `ON DELETE CASCADE`，但无 `DELETE FROM documents` 执行 → 级联从未触发。
- **影响**：MCP/HTTP 删除文档从不真正删除，孤儿行无限堆积；搜索准确率下降；数据保留/隐私风险。违反 super-tree-v3 FR5。
- **严重度**：P0 ｜ **[交叉确认×2：robustness + contracts]**
- **置信度**：高

#### P0-3 · `_build_super_tree` N+1 + 缺索引（直接违反 NFR1）
- **位置**：`super_tree.py:259-302`；`db.py`（`nodes` 表无 `parent_node_id` 索引）
- **证据**：循环 `doc_ids`，每个候选 re-fetch `get_all_documents()` 全表（`super_tree.py:264`）+ Python 层 `next()` 线性匹配；每个 top-node 执行 `COUNT(*) FROM nodes WHERE parent_node_id=?`（`super_tree.py:273-278`），`parent_node_id` 完全无索引 → 全表扫描。N×(全表+M×COUNT)。
- **影响**：直接违反 v3 NFR1（L0+L1 < 1s）；50 候选 × 8 节点 = 50 次全表 + 400 次 COUNT 扫描。
- **严重度**：P0 ｜ **[对抗确认 A1]**
- **置信度**：高

#### P0-4 · L2 Act 串行 LLM（延迟超 NFR）★ 归因修正
- **位置**：`router.py:151-216`（`_act_tree_search` 的 `for doc_id in candidate_docs` 循环无 `await`）；`main.py:212-230`→`main.py:144`（`get_relevant_nodes`→`_call_llm_json`→同步 `client.chat.completions.create`）
- **证据**：grep 确认 `router.py` 所有 `to_thread`/`gather` 均在 `_act_tree_search` 循环体**之外**；每个候选 doc 一次同步 LLM 往返。
- **影响**：3-5 docs × 0.5-2s = 1.5-10s 串行。
- **严重度校准**：spec NFR1 = "L0+L1 <1s（**不含 L2/L3**）"；`_act_tree_search` 属 L2/L3，故应归因到**项目顶层"多文档检索 <2s" NFR**，非 NFR1。串行事实 CONFIRMED。
- **严重度**：P0 ｜ **[对抗确认 A2]**
- **置信度**：高

#### P0-5 · 同步 LLM 阻塞事件循环（服务串行化）★ 精确边界
- **位置**：`main.py:134-163`（`_call_llm_json`）、`main.py:277-301`（`generate_answer`）；调用点 `router.py:176`、`router.py:294`、`router.py:441`（均未 `to_thread` 包裹）
- **证据**：两函数用同步 `OpenAI().chat.completions.create`，在 `async def` 路径中被直接调用。
- **严重度校准**：**并非所有 LLM 调用阻塞**——`select_documents`（`super_tree.py:234`）走 `await llm_acompletion`（异步，不阻塞）。真正阻塞的仅 `get_relevant_nodes` 与 `generate_answer` 两处同步函数被 async 路径直接调用的点。
- **影响**：每次多文档搜索期间整个 uvicorn 事件循环停摆（无其他请求/健康检查可处理）。
- **严重度**：P0 ｜ **[对抗确认 A3]**
- **置信度**：高

### 3.2 P1 — 高优先（16）

#### P1-1 · `import main` 反向依赖（库不自包含、不可独立测试的直接根因）
- **位置**：`router.py:35-47`（`_load_main_funcs` 内 `import main` 取 5 函数）、`strategies.py:56-60`（`DescriptionStrategy` 取 `get_relevant_documents_for_multidoc`）、`client.py:354-359`（`import main as _main`）
- **证据**：5 函数经 grep + CodeGraph 双重确认**仅定义于 `main.py`**（`get_relevant_nodes@212`/`pages_from_nodes@166`/`generate_answer@277`/`get_relevant_documents_for_multidoc@563`/`build_context_with_budget@586`）。`tests/test_router.py:154/196/221` 被迫 `patch.object(router,'_load_main_funcs',...)` 才能跑通。`client.py:354` 注释 `"project root must be on sys.path"` 自供违反 NFR2；`pyproject.toml` 无 `[project.scripts]` → `import main` 仅在 CWD=项目根时成立，可移植性归零。
- **影响**：库包反向依赖入口脚本；与 project-refactor G2/G3（单一自包含风格）矛盾；是 router/client 不可独立测试的直接根因。
- **严重度**：P1 ｜ **[对抗确认 C1]**
- **置信度**：高

#### P1-2 · 双检索管线无共享核心 + 行为发散
- **位置**：`main.py:464-560`（`answer_multidoc`，手搓 L1/L2/L3）vs `router.py:518-525`→`_search_super_tree`/`_search_v2`（Plan/Route/Act/Verify）
- **证据**：L1 不共享（`get_relevant_documents_for_multidoc` 仅被 `DescriptionStrategy` 用，`_search_super_tree` 从不调它）；**L3 行为发散**——`answer_multidoc` 经 `build_context_with_budget` 截断到 `MAX_CONTEXT_TOKENS=16000`，而 `_act_tree_search` 的 L3（`router.py:196-211`）从内存拼接 `ctx_parts` **无任何 token 预算**；`build_context_with_budget` 虽在 `_load_main_funcs` 注册（`router.py:41`）但 `_act_tree_search` 全程不调用 → **死注册**。
- **影响**：同一查询经 CLI vs HTTP 得到不同 context 窗口 → 不同答案；两套维护面。
- **严重度**：P1 ｜ **[对抗确认 C2]**
- **置信度**：高

#### P1-3 · `super_tree.py` 职责过载 + 裸 DELETE 绕过 db 层
- **位置**：`super_tree.py:12`（`KeywordIndex`）、`:52`（`KBIdentity`）、`:124`（`SuperTreeIndex`）；`super_tree.py:65-67`（`KBIdentity.invalidate` 裸 `DELETE FROM kb_identity WHERE id=1` 经 `db._connect()` 私有方法）
- **证据**：三类无共享基类/状态，仅 `SuperTreeIndex.__init__` 组合前两者（composition 非 cohesion）。`db.py` grep `kb_identity` 仅 `set_kb_identity@342`/`get_kb_identity@356`，**无 `clear_kb_identity`** → invalidate 完全绕过 db 抽象。invalidate 是热路径（文档增删触发）。
- **影响**：模糊 L0/L1 边界；db 层封装被业务类穿透。
- **严重度**：P1 ｜ **[对抗确认 C3]**
- **置信度**：高

#### P1-4 · API key 非常量时间比较（timing attack）
- **位置**：`server.py:60`（`if header_key != API_KEY:`）
- **证据**：`!=` 短路逐字节比较。修复：`hmac.compare_digest`。
- **影响**：CWE-208；远程利用受网络抖动限制，但属低成本加固缺口。
- **严重度**：P1 ｜ **[对抗确认 B2]**
- **置信度**：高

#### P1-5 · 上传无大小/数量限制（OOM DoS）
- **位置**：`server.py:379`（`content = await file_field.read()`）、`server.py:351-362`（`raw_files` 无上限）
- **证据**：无 Content-Length/单文件/总大小检查；`_UPLOAD_SEMAPHORE=Semaphore(3)` 限并发**不限内存** → 3 个大上传即 OOM（虚假安全）。
- **影响**：认证后单请求可致 OOM/磁盘填满。
- **严重度**：P1 ｜ **[对抗确认 B3]**
- **置信度**：高

#### P1-6 · 裸 `except:` 吞掉控制信号
- **位置**：`utils.py:199`（`extract_json` 回退 `except:`）、`page_index_md.py:7`（`try: from .utils import * except:`）
- **证据**：裸 except 捕获 `KeyboardInterrupt`/`SystemExit`/`MemoryError`，掩盖真实 bug（如 `SyntaxError` 表现为 JSON 解析失败返回 `{}`）。
- **影响**：隐藏 bug；`extract_json` 返回 `{}` 致下游 `KeyError`（见 P2-12）。
- **严重度**：P1 ｜ **[对抗确认 B4]**
- **置信度**：高

#### P1-7 · KBIdentity 缓存写入竞争（重复 LLM 调用）
- **位置**：`super_tree.py:59-115`（`get_identity`/`_build`/`invalidate`）
- **证据**：并发查询均观察 cache miss → 均调 LLM → `set_kb_identity`（last-write-wins）；`invalidate`（文档增删）与并发 writer 竞争。
- **影响**：浪费 LLM 成本；缓存抖动。
- **严重度**：P1 ｜ **[交叉确认×2：performance + robustness]**
- **置信度**：中高

#### P1-8 · 删除路径未失效 super_tree_index/kb_identity（违反 FR5）
- **位置**：`server.py:151-174`（调 `closet_index.remove_document` 但未调 `super_tree_index.on_document_removed`）
- **证据**：`on_document_removed`（`super_tree.py:151`）存在且有效（清关键词索引 + 失效身份），但删除 handler 从不调用。`on_document_added` 却在索引路径（`client.py:172`）被调用 → 添加/删除不对称损坏。
- **影响**：删除后关键词索引膨胀 + 过期 KB 身份继续喂给 L1 prompt（与 P1-10 耦合放大）。`tests/test_super_tree.py:179` 仅测方法本身，未测 MCP 删除路径 → 未捕获。
- **严重度**：P1 ｜ **[对抗确认 B7]** ｜ 与 P0-2 耦合
- **置信度**：高

#### P1-9 · Prompt 注入（不可信内容无分隔插值）
- **位置**：`main.py:213,234,281,565`、`super_tree.py:211`、`closet_index.py:54`、`planner.py:20`、`verifier.py:59`、`strategies.py:76`、`utils.py:653,697`（约 11 处）
- **证据**：抽查 8 处全部确认 f-string 直接插值 `{question}`/`{context}`/`{node['text']}` 等，无 XML/结构化分隔、无清洗。
- **影响**：恶意 PDF/用户查询可操纵检索与答案生成。`question`/`query` 为直接用户输入，是主注入向量。
- **严重度**：P1 ｜ **[对抗确认 B6]**
- **置信度**：高（风险存在；可利用性取决于 LLM）

#### P1-10 · KBIdentity 纯文本解析未强制（污染传播）
- **位置**：`super_tree.py:101,103-106`（prompt 要求纯文本，存 `response.strip()` 原样）
- **证据**：无 Markdown 代码块剥离（对比 `select_documents@239` 用 `extract_json` 剥离 ```json）。LLM 若返回 ```` ```text ... ``` ```` 则围栏持久化到 `kb_identity`，并注入每个 L1 选择 prompt。
- **影响**：身份污染传播到所有 L1 调用。
- **严重度**：P1 ｜ **[对抗确认 B5]** ｜ 与 P1-8 耦合
- **置信度**：中高

#### P1-11 · API 契约：无版本/OpenAPI/错误码不一致 + 信息泄露
- **位置**：`server.py:470-475`（路由 `/health`/`/upload`/`/sse`/`/messages/`，无 `/v1/`）、`server.py:477`（无 `openapi_url`）、`server.py:348`（`f"Failed to parse form: {e}"`）、`server.py:399`（`str(res)`）
- **证据**：400/403/500 不一致；500 类错误体泄露内部异常文本（CWE-209）。
- **影响**：无破坏性变更策略；客户端无法发现 schema；信息泄露。
- **严重度**：P1 ｜ **[对抗确认 B8 + 新增信息泄露子项]**
- **置信度**：中高

#### P1-12 · 硬编码 `MODEL="gpt-4.1"`（与当前分支目标矛盾）
- **位置**：`pageindex_mutil/page_index_md.py:312`（`if __name__ == "__main__":` 块）
- **证据**：`__main__` 演示路径直接硬编码，绕过 `MODEL_NAME`/`ConfigLoader`。运行 `page_index_md.py` 忽略环境变量。
- **影响**：运行时影响低（仅 `__main__`），但与 `feat/unify-model-name-env` 分支目标直接矛盾——分支自身留了反例。
- **严重度**：P1（原 standards 标 P0，因 `__main__` only 降级）｜ **[交叉确认×3：architecture + standards + contracts]**
- **置信度**：高

#### P1-13 · 缺 DB 索引（nodes/pages）
- **位置**：`db.py:39-77`（仅 `closet_tags`/`doc_keywords` 有索引）
- **证据**：`nodes`/`pages` 无专用 `doc_id` 索引；`parent_node_id` 完全无索引。热路径 `get_nodes_by_doc_id`/`get_top_level_nodes`/`get_pages_in_range`/`get_pages_by_numbers` 按 `doc_id` 过滤。
- **严重度校准**：`UNIQUE(doc_id,node_id)`/`UNIQUE(doc_id,page_number)` 复合约束的前导列 `doc_id` 可部分服务 `WHERE doc_id=?`，故"完全无索引"略夸张；但缺专用单列索引 + `parent_node_id` 无索引是事实。
- **严重度**：P1 ｜ **[对抗确认 A4]**
- **置信度**：高

#### P1-14 · `count_tokens` 每次重建 tiktoken encoder
- **位置**：`utils.py:93-100`
- **证据**：每次 `tiktoken.encoding_for_model(...)`，无模块级缓存。紧循环调用：`build_context_with_budget`（`main.py:594,598` 每页）、`select_documents` while 循环（`super_tree.py:203,209` 每次截断 3× + re-dump `tree_json`）。
- **严重度**：P1 ｜ **[对抗确认 A5]**
- **置信度**：高

#### P1-15 · `llm_completion` 重试固定 1s×10
- **位置**：`utils.py:103-132`（同步）、`135-156`（异步）
- **证据**：`max_retries=10`，`sleep(1)`/`asyncio.sleep(1)`，无退避/抖动，不区分 4xx/5xx/429 → 401 也重试 10 次共 ~10s。
- **影响**：429 卡 10s；同步版 `time.sleep` 若在 async 路径被调用则阻塞事件循环（与 P0-5 叠加）。
- **严重度**：P1 ｜ **[对抗确认 A6]**
- **置信度**：高

#### P1-16 · `select_documents` 无缓存
- **位置**：`super_tree.py:193-254`
- **证据**：每次 search 重建 super-tree（含 P0-3 的 N+1）+ 构造 prompt + LLM；无结果/物化缓存。调用链确认 `select_documents`←`_search_super_tree`←`AgenticRouter.search`，全路径无缓存层。
- **影响**：与 P0-3 叠加放大——每次查询重演 N+1。
- **严重度**：P1 ｜ **[对抗确认 A7]**
- **置信度**：高

### 3.3 P2 — 中优先（20）

| ID | 标题 | 位置 | 确认 |
|----|------|------|------|
| P2-1 | `page_index.py` 上帝模块（1153L/35函数/4子域，`page_index_main`/`page_index` 近重复） | `page_index.py` | 交叉×2 |
| P2-2 | `utils.py` grab-bag（812L，LLM client+config+tree helpers+text utils） | `utils.py` | arch |
| P2-3 | `db.py` 上帝对象（documents/nodes/pages + closet + keyword + kb_identity 四子系统） | `db.py:237-356` | arch |
| P2-4 | `page_index.py` vs `page_index_md.py` 树构建逻辑部分重复 | `page_index.py:1029`/`page_index_md.py:190` | arch |
| P2-5 | `AgenticRouter` 双策略 `_search_v2` vs `_search_super_tree` ~100 行重复 | `router.py:221,337` | arch |
| P2-6 | `logs/qa_20260213.jsonl`（25KB）被提交进 git（虽 `logs/` 已 gitignore，文件早于规则） | `logs/` | standards |
| P2-7 | `main.py` 65 处 `print()` 零 logging（`server.py` 13 logging/0 print，不一致） | `main.py` 全文 | standards |
| P2-8 | 类型注解缺失 ~86%；`page_index_main`/`page_index` 无注解无 docstring | `page_index.py:1066,1113` | standards |
| P2-9 | `page_index.py` 零测试（`page_index_main`/`page_index` 及 35 函数全无覆盖） | `tests/` | standards |
| P2-10 | `retrieve_model`/`RETRIEVE_MODEL_NAME` 死配置（加载存储但从不读取使用） | `client.py:60`、`config.yaml:24` | contracts |
| P2-11 | `_parse_pages` 接受无界范围（`"1-99999999999"` → 内存放大/OOM） | `retrieve.py:12-24` | robustness |
| P2-12 | `extract_json` 返回 `{}` → 下游无防护 `KeyError`（`['toc_detected']`/`['completed']`） | `page_index.py:122,140,158` | robustness |
| P2-13 | LLM prompt（含用户查询/文档内容）被写入错误日志（PII/外泄） | `utils.py:129,155` | 交叉×2 |
| P2-14 | `_build_multidoc_context`/`_build_docs_info`/`KBIdentity` 各自 N+1（per-doc re-fetch `get_all_documents`+`get_top_level_nodes`） | `main.py:437-461`/`router.py:52-93`/`super_tree.py:80-89` | perf |
| P2-15 | 无 WAL 模式 / pragma（默认 `journal_mode=DELETE`，写阻塞读，`database is locked`） | `db.py:14-19` | perf |
| P2-16 | `.env.example` 默认 DashScope vs `config.yaml`/P6 "标准 OpenAI 默认" 矛盾 | `.env.example:17`、`config.yaml:23` | 交叉×2 |
| P2-17 | 魔法数字散落（`THINNING_THRESHOLD=5000` 等）+ `main.py:70-72` 重新硬编码 `config.yaml` 默认值 | `page_index_md.py:313-316`、`main.py:70-72` | standards |
| P2-18 | `tasks.md` 陈旧（project-refactor 显示全 pending 但代码已完成；batch-upload 验收标准未勾选） | `project-refactor/tasks.md:212-234`、`batch-upload/tasks.md` | 交叉×2 |
| P2-19 | `pageindex_mutil` typo（应为 `pageindex_multi`，refactor 已接受为 debt） | 包名 | arch |
| P2-20 | UUID/int 双 ID 体系，映射在 3+ 处重复派生 | `client.py:72`、`server.py:129,265`、`router.py:58-64` | arch |

### 3.4 P3 — 低优先（7）

| ID | 标题 | 位置 | 确认 |
|----|------|------|------|
| P3-1 | `docs/REGISTRY.md` 缺失；设计文档无 status/version 头；`project-refactor/spec.md` 陈旧 | `docs/` | standards |
| P3-2 | `configure_llm` 导入时运行，无 key 时静默禁用 LLM 无诊断 | `utils.py:90` | robustness |
| P3-3 | `_resolve_upload_paths` glob 逃逸 `pdf_dir`（CLI 本地，意图模糊） | `main.py:318-352` | robustness |
| P3-4 | `list_documents`/`list_resources` O(docs×uuid) 线性扫描 | `server.py:124-138,261-277` | perf |
| P3-5 | jieba 惰性初始化，首查询冷启动延迟（无 warmup） | `super_tree.py:23` 等 | perf |
| P3-6 | `retrieve.py` 近孤儿模块（仅 `client.py` 引用） | `retrieve.py` | arch |
| P3-7 | `.env` 含线上密钥于工作树（已 gitignore 未追踪，但出现在审查输出中） | `.env` | operational |

### 3.5 优势（值得保持）

1. **SQL 完全参数化**——`db.py` 全部 `?` 占位符，动态 `IN(...)` 经 `placeholders=",".join("?"...)` 构造，无注入面。
2. **`/upload` 文件名遍历正确清洗**——`Path(filename).name` + UUID 前缀（`server.py:375,377`）。
3. **`main.py` 资源清理正确**——`log_f = open(...)` 在 `finally`（`main.py:879-881`）随 `db.close()` 关闭，覆盖所有退出路径；`_index_files_batch` 每线程独立 `PageIndexDB` 并 `close`。
4. **`pageindex_mutil/agentic/` 子包结构清晰**——Planner/Strategies/Verifier/Router 单一职责文件，策略模式可扩展。
5. **`project-refactor` 成功消除旧 fork 嵌套**——`PageIndex/` 目录消失、`sys.path` hack 移除、`tests/` 合并、`configure_llm` 单一源、`ConfigLoader` 环境覆盖（14 测试覆盖优先级/空白/空串边界）。

---

## 4. 优化方案（优先级工作流）

> 每个工作流（W）为一组可独立立项的优化项。**落地须经完整代码工作流**（`workflow-requirements-clarification`→`workflow-system-design`→`workflow-code-generation`→`workflow-code-review`，见 AGENTS.md），在 `docs/design-docs/PageIndex/<W>/` 下产出 `spec.md`+`tasks.md`。下表给出 **影响 / 工作量 / 风险 / 依赖** 以辅助排序。

### 4.1 工作流清单

#### W1 · 并发安全与数据层加固
- **覆盖**：P0-1、P2-15、P1-13、P0-3（索引部分）
- **内容**：`sqlite3.connect(check_same_thread=False)` + 写锁 或 每请求/每线程连接；加 `PRAGMA journal_mode=WAL`/`synchronous=NORMAL`/`busy_timeout=5000`；补 `idx_nodes_doc_id`/`idx_nodes_parent_node_id`/`idx_pages_doc_id`。
- **影响**：高（解除并发崩溃/损坏 + 解锁 NFR1） ｜ **工作量**：S-M ｜ **风险**：低（连接模型变更需并发测试） ｜ **依赖**：无（先行）
- **验收**：≥2 文件并发上传 + 并发搜索集成测试通过；NFR1 在 50 候选语料下达标。

#### W2 · 删除路径与数据完整性修复
- **覆盖**：P0-2、P1-8、P1-10
- **内容**：`db.py` 实现 `delete_document(doc_id)`（`DELETE FROM documents WHERE id=?`，级联清子表）；`server.py` 删除路径补 `super_tree_index.on_document_removed(db_id)`（失效关键词索引 + KB 身份）；`KBIdentity` 存储前剥离 Markdown 围栏。
- **影响**：高（修复静默数据保留 + FR5） ｜ **工作量**：S ｜ **风险**：低 ｜ **依赖**：W1（连接安全）更稳，但可并行
- **验收**：插入→删除集成测试：断言 documents/nodes/pages/tags/keywords 行清除 + `kb_identity` 失效；现有库孤儿数据迁移脚本。

#### W3 · 检索引擎异步化与性能
- **覆盖**：P0-4、P0-5、P1-14、P1-15、P1-16、P2-14、P1-7
- **内容**：`get_relevant_nodes`/`generate_answer` 在 async 路径用 `await asyncio.to_thread(...)` 包裹或迁移至 `llm_acompletion`；L2 Act 用 `asyncio.gather` 并行每文档召回；`count_tokens` 模块级缓存 encoder + `select_documents` while 循环外预 dump `tree_json`；重试改指数退避+抖动+4xx 不重试；`select_documents` 结果/super-tree 物化缓存（按候选集 key，增删失效）；KBIdentity 加 `asyncio.Lock` 单飞。
- **影响**：高（达成 <2s NFR + 解除事件循环阻塞） ｜ **工作量**：M-L ｜ **风险**：中（并发改造） ｜ **依赖**：W1（并发安全前提）
- **验收**：端到端多文档检索基准（当前缺失，须新增）<2s；事件循环阻塞检测通过。

#### W4 · 架构重构：自包含库与管线统一
- **覆盖**：P1-1、P1-2、P1-3、P2-1、P2-2、P2-3、P2-4、P2-5
- **内容**：
  - **W4a**：将 `get_relevant_nodes`/`pages_from_nodes`/`generate_answer`/`build_context_with_budget`/`get_relevant_documents_for_multidoc` 从 `main.py` 提取到 `pageindex_mutil/retrieval.py`；移除三处 `import main`；`main.py`/`server.py` 向下导入。
  - **W4b**：统一检索管线——`answer_multidoc` 改调 `PageIndexClient.search`（或提取共享 `RetrievalService`）；`build_context_with_budget` 接入 `_act_tree_search` 的 L3（消除无 token 预算发散 + 死注册）。
  - **W4c**：拆分 `super_tree.py`→`keyword_index.py`+`kb_identity.py`+`super_tree.py`；`db.py` 加 `clear_kb_identity()` 替裸 DELETE。
  - **W4d**（后续）：`page_index.py` 拆 `toc.py`/`page_offset.py`/`tree_builder.py`；`utils.py` 拆 `llm.py`/`config.py`/`tree_utils.py`/`text_utils.py`；`AgenticRouter` 双策略合一 + 提取共享 Act→Generate→Verify 尾。
- **影响**：高（解锁独立测试 + 行为一致 + 可维护性） ｜ **工作量**：L（W4d 最大） ｜ **风险**：中高（大重构需 GREEN 安全网，但 P2-9 测试缺失是障碍） ｜ **依赖**：W7 先补 `page_index.py` 测试为安全网
- **验收**：`AgenticRouter`/`PageIndexClient` 无 `import main` 可独立单测；CLI 与 HTTP 同查询行为一致（集成测试断言）。

#### W5 · 安全加固
- **覆盖**：P1-4、P1-5、P1-6、P1-9、P1-11、P2-11、P2-12、P2-13、P3-2
- **内容**：API key 用 `hmac.compare_digest`；上传加 Content-Length/文件数/单文件大小上限 + 流式落盘；`except:`→`except Exception:`/`except ImportError:`；prompt 用结构化分隔符标记不可信内容（`<user_query>`/`<document_page>`）并剥离冲突标记；`_parse_pages` 限范围（≤1000 页，`start>=1`）；`extract_json` 调用点改 `.get()`+默认；prompt 日志改截断/哈希 + debug 级；`/v1/` 前缀 + `openapi_url` + 统一错误信封 `{code,message,details}` + 500 不泄露内部异常；`configure_llm` 无 key 时 WARNING。
- **影响**：高（消除 DoS/注入/信息泄露） ｜ **工作量**：M ｜ **风险**：低-中 ｜ **依赖**：无（可与 W1-W3 并行）
- **验收**：安全测试（注入/超大/遍历）；OpenAPI 可发现；错误响应不泄露内部文本。

#### W6 · 模型配置统一收尾（完成当前分支）
- **覆盖**：P1-12、P2-10、P2-16、P2-17
- **内容**：`page_index_md.py:312` 改 `ConfigLoader().load(None).model`；决策 `retrieve_model`（接入或移除 env+config+accessor）；`.env.example` 默认与 P6 "标准 OpenAI 默认" 对齐或修正 P6 措辞；魔法数字入 `config.yaml` 或命名常量模块；`main.py:70-72` 改用 `ConfigLoader` 填默认。
- **影响**：中（完成分支契约 + 减少配置困惑） ｜ **工作量**：S ｜ **风险**：低 ｜ **依赖**：无（可与分支同 PR）
- **验收**：CI grep 守卫 `! grep -rn 'MODEL\s*=\s*"gpt-' pageindex_mutil/`；`retrieve_model` 行为明确。

#### W7 · 工程规范与测试覆盖
- **覆盖**：P2-6、P2-7、P2-8、P2-9、P3-1、P3-4、P3-5、P2-18、P3-3、P2-19、P2-20
- **内容**：`git rm --cached logs/qa_20260213.jsonl`；`main.py`/`page_index_md.py` print→logging；public API 类型注解 + `TypedDict` 结构；新增 `tests/test_page_index.py`（fixture PDF 覆盖 `page_index_main`）+ 纯函数单测；新增 `PageIndexClient.search` 与 `answer_multidoc` 集成测试；`list_documents` 用单一 `db_id↔uuid` 反向映射；jieba 启动 warmup；创建 `docs/REGISTRY.md` + 文档 status 头，标记 `project-refactor/spec.md` STALE/归档，更新 `tasks.md` 实际状态。
- **影响**：中-高（为 W4 重构提供安全网） ｜ **工作量**：M ｜ **风险**：低 ｜ **依赖**：W4a（提取后可独立单测）
- **验收**：`page_index.py` 有覆盖；W4 重构前后测试 GREEN。

### 4.2 路线图（按依赖与 ROI 排序）

```
阶段 0 · 热修（立即可做，独立 PR）
  └─ W2 删除路径完整性 (P0-2/FR5)         ──┐
  └─ W1 并发安全 (P0-1/WAL/索引)            ├─ 并行，无相互依赖
  └─ W6 模型配置收尾 (完成当前分支 P1-12)   ─┘

阶段 1 · 高 ROI（解锁测试 + 解锁 NFR）
  └─ W4a 提取 retrieval.py 移除 import main (P1-1)   ← W7 部分测试先行做安全网
  └─ W3 快赢: async LLM 包裹 (P0-5) + 并行 Act (P0-4)   ← 依赖 W1 完成
  └─ W5 安全加固 (可与上并行)

阶段 2 · 结构与一致性
  └─ W4b 统一管线 (P1-2)   ← 依赖 W4a
  └─ W4c 拆分 super_tree (P1-3)
  └─ W3 完整: 缓存/退避/tiktoken (P1-14/15/16) + KBIdentity 锁 (P1-7)

阶段 3 · 深度重构与规范
  └─ W4d 拆分 page_index.py/utils.py/双策略 (P2-1/2/5)   ← 依赖 W7 测试覆盖
  └─ W7 工程规范/REGISTRY/tasks 同步 (P2-6/7/8/9/18, P3-1/4/5)
```

### 4.3 优先级矩阵

| 工作流 | 影响 | 工作量 | 风险 | 优先级 | 阻塞 ship? |
|--------|------|--------|------|--------|-----------|
| W2 删除路径 | 高 | S | 低 | **P0 立即** | 是（数据完整性） |
| W1 并发安全 | 高 | S-M | 低 | **P0 立即** | 是（崩溃/损坏） |
| W6 配置收尾 | 中 | S | 低 | **P0 同分支** | 否（但完成分支契约） |
| W4a 提取 retrieval | 高 | M | 中 | P1 | 否（但解锁测试） |
| W3 async/并行 | 高 | M-L | 中 | P1 | 否（NFR） |
| W5 安全 | 高 | M | 低-中 | P1 | 否（安全债务） |
| W3 缓存/退避 | 中高 | M | 中 | P2 | 否 |
| W4b/c 统一管线+拆分 | 高 | M | 中 | P2 | 否 |
| W7 规范/测试 | 中高 | M | 低 | P2 | 否（重构安全网） |
| W4d 深度拆分 | 中 | L | 中高 | P3 | 否 |

---

## 5. 结论与建议

**最优先行动**（阶段 0，三条独立热修，建议立即立项并各自走代码工作流）：
1. **W2** — 修复 `delete_document`（P0-2）+ 删除失效（P1-8）：这是静默数据完整性破坏，且已有孤儿数据堆积。
2. **W1** — 修复 sqlite 并发安全（P0-1）+ WAL + 索引（P1-13/P0-3 索引部分）：解除并发崩溃并解锁 NFR1。
3. **W6** — 完成 `MODEL_NAME` 统一（P1-12）：与当前分支同 PR，闭合分支契约。

**结构性根因**（阶段 1-2）：`import main` 反向依赖（P1-1）是可测试性与库自包含的直接根因，应优先提取 `retrieval.py`，继而统一双管线（P1-2）。`page_index.py` 零测试（P2-9）是所有重构的安全网缺口，须先补测试。

**性能 NFR**（阶段 1-2）：`<2s`/`<1s` 目标当前架构性不可达（P0-3/4/5 + P1-16 叠加）。async LLM 包裹 + 并行 Act + 索引 + 缓存是达成路径，且当前**无任何性能基准**（应新增）。

**安全债务**（W5）：上传无限制、prompt 注入、错误体信息泄露、API key timing——均低成本修复，建议与阶段 0/1 并行。

---

## 附录 A · 交叉确认矩阵

| 问题 | reviewer 来源 | 确认方式 |
|------|--------------|---------|
| SQLite 线程安全 (P0-1) | performance + robustness | 交叉×2 |
| delete_document 不存在 (P0-2) | robustness + contracts | 交叉×2 |
| 硬编码 MODEL (P1-12) | architecture + standards + contracts | 交叉×3 |
| prompt 日志外泄 (P2-13) | robustness + contracts | 交叉×2 |
| page_index.py 上帝模块 (P2-1) | architecture + standards | 交叉×2 |
| KBIdentity 竞争/缓存 (P1-7/16) | performance + robustness + contracts | 交叉×3 |
| tasks.md 陈旧 (P2-18) | standards + contracts | 交叉×2 |
| .env.example 默认矛盾 (P2-16) | standards + contracts | 交叉×2 |

## 附录 B · 审查代理清单

- reviewers: `reviewer-architecture`、`reviewer-performance`、`reviewer-robustness`、`reviewer-standards`、`reviewer-contracts`
- critics: `critic-perf`（A1-A7）、`critic-security`（B1-B8）、`critic-arch`（C1-C3）
- 全程只读；零代码修改；符合 execution-discipline Two-Agent Minimum 与"编排器不自审"原则。

## 附录 C · 关键文件清单（绝对路径）

- `/home/cc-workspace/PageIndex-UV/db.py`
- `/home/cc-workspace/PageIndex-UV/server.py`
- `/home/cc-workspace/PageIndex-UV/main.py`
- `/home/cc-workspace/PageIndex-UV/pageindex_mutil/super_tree.py`
- `/home/cc-workspace/PageIndex-UV/pageindex_mutil/agentic/router.py`
- `/home/cc-workspace/PageIndex-UV/pageindex_mutil/agentic/{planner,strategies,verifier}.py`
- `/home/cc-workspace/PageIndex-UV/pageindex_mutil/client.py`
- `/home/cc-workspace/PageIndex-UV/pageindex_mutil/closet_index.py`
- `/home/cc-workspace/PageIndex-UV/pageindex_mutil/utils.py`
- `/home/cc-workspace/PageIndex-UV/pageindex_mutil/page_index.py`
- `/home/cc-workspace/PageIndex-UV/pageindex_mutil/page_index_md.py`
- `/home/cc-workspace/PageIndex-UV/pageindex_mutil/retrieve.py`
- `/home/cc-workspace/PageIndex-UV/docs/design-docs/PageIndex/{project-refactor,non-vector-retrieval-optimization,super-tree-retrieval-v3,batch-upload}/{spec,tasks}.md`
