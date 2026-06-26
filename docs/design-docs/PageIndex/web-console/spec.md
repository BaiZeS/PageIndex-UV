# Web 控制台（前端页面 + CLI/模型配置迁移）— spec.md（v1.1）

> 阶段：clarify（§1-3 已扩展 v1.1 范围；§4-8 待 system-design 阶段产出）。
> 来源：用户 `/devkit:main` → "给应用加一个简单的前端页面，把 CLI 功能搬上去，模型配置也搬上去，页面做好看点"。
> 当前分支：`main`（clean，v1.0 已于 5b30adc 合入）。v1.1 实现时应在特性分支 `feat/web-console-v1.1` 上进行。
> 版本历史：
> - v1.0（2026-06-24，commit 5b30adc）— FR1-FR8 已 shipped（文档列表/上传/问答/配置读写/静态托管/鉴权放行）。
> - v1.1（2026-06-26，本更新）— 扩展 FR9-FR12：上传进度条、文档删除、问答证据源、模型配置连通性测试。**强依赖 W2（delete-path-integrity）已合入**（db.delete_document / server delete handler / on_document_removed / strip_markdown_fence / disk cleanup / orphan migration / tests 已存在）。
> 用户确认决策：架构=复用 server.py；前端=单文件 HTML+CDN UI 库；模型名持久化=config.yaml+同步 .env；凭据持久化=configure_llm 运行时热切换+写 .env；鉴权=页面/静态放行+key 存 localStorage。
> v1.1 用户新增决策（2026-06-26）：4 个特性打包为同一 v1.1 模块（不新建 spec）；删除功能复用 W2 delete-path-integrity；auto-mode 启用（ALLOW_AUTO_MODE=true 来自 dispatch envelope）。

## §1 背景

PageIndex-UV 当前以两种形态对外：
- `main.py` — 交互式 REPL CLI，核心动作：`/add`（索引 PDF/MD）、`/list`（列文档）、`/doc <n>`（单文档问答）、默认输入（多文档问答）。
- `server.py` — Starlette 服务，端口 `3000`：`GET /health`、`POST /upload`（上传→索引）、`/sse`+`/messages/`（MCP）。启动时在 `lifespan` 创建全局 `PageIndexClient`，所有端点经 `get_client()` 共享。

模型配置已有一套统一收尾（W6）：`pageindex_mutil/utils.py` 的 `_resolve_llm_config` / `configure_llm(api_key, base_url)` 支持运行时重建 client；`ConfigLoader().load()` 从 `pageindex_mutil/config.yaml` 读 `model` / `retrieve_model`，env `MODEL_NAME` / `RETRIEVE_MODEL_NAME` 可覆盖。

**关键现状（已核实，2026-06-24 → 2026-06-26 v1.1 更新）**：
- 项目**零前端工具链**（无 `package.json` / `.vue` / node 构建）。
- 实际 `.env`：`MODEL_NAME=qwen3.7-plus`（已设置，**覆盖** config.yaml 的 `model: gpt-4.1-mini`）；`OPENAI_API_KEY` 已设；`OPENAI_BASE_URL=http://10.135.11.111/` 已设；`RETRIEVE_MODEL_NAME` 未设；`API_KEY=<见 .env，勿入库>` 已设（MCP HTTP 鉴权）。
- 故：单纯写 config.yaml 的 `model` **不会生效**（env 覆盖仍在）。
- v1.0 状态（5b30adc 已 ship）：`server.py:403` `search_endpoint` 已 enrich `matched_docs[].doc_name`；`server.py:436` `config_post_endpoint` 已实现四步幂等流程（运行时应用+os.environ 同步+备份+定点写+回滚）；`server.py:555` `upload_endpoint` 是**同步批量**（一次性索引所有文件后统一返回 results），前端只看到最终结果，看不到过程进度；前端 `web/static/app.js` 用 `el-upload :http-request=onUpload`，文件级成功/失败已展示，但**无进度条/无阶段状态**。
- v1.1 前置条件（已具备，2026-06-26 核实）：W2 `delete-path-integrity` 已合入（`db.py:337` `delete_document` 存在；`super_tree.py:154` `on_document_removed` 已被 server 删除路径调用；`utils.py:173` `strip_markdown_fence` 实现并被 KBIdentity 使用；删除路径已包含磁盘清理 + 孤儿迁移 + 集成测试）。F2 可直接基于现有 server 删除 handler 暴露 REST/MCP 入口。
- v1.1 缺口：(a) 上传无进度反馈，长任务体验差；(b) 删除能力只走 MCP `delete_document`，Web 控制台无对应入口；(c) 问答结果有 `matched_docs/selected_nodes/pages` 字段但前端只显示 `answer` + 文档名列表，未渲染证据链；(d) 模型配置无连通性自检，错误要等下次问答才暴露。

## §2 目标

**v1.1 目标**（覆盖 v1.0 §2 目标并扩展）：
- **一个 Web 控制台页面**，把 CLI 的核心动作（索引/列文档/问答）图形化，并提供模型配置 UI。
- **复用 server.py 单进程**：新增 REST 端点 + 静态托管，天然共享全局 client 与配置，零部署增量。
- **模型配置可热切换并持久化**：运行时即时生效（`configure_llm`），重启后保留（写 config.yaml + .env，保持二者一致）。
- **页面美观**：实现阶段加载 `frontend-design` skill 做排版/留白，避免默认模板感。
- **零前端构建**：不引入 node/Vite 构建链。
- **v1.1 新增**：上传过程可见（进度条 + 阶段状态）；文档删除（Web 入口，复用 W2 后端）；问答证据可视化（matched_docs/selected_nodes/pages 链式呈现）；模型连通性自检（保存前/后可选 ping）。

## §3 需求与范围

### 功能需求（FR）

**v1.0（已 ship，commit 5b30adc）**：
- **FR1 文档列表**：`GET /api/documents` 返回已索引文档（id / 名称 / 描述 / 页数或行数），等价 CLI `/list`。
- **FR2 索引（上传）**：复用现有 `POST /upload`；前端拖拽上传 + 逐文件结果展示。
- **FR3 问答**：`POST /api/search`（body `{query, top_k?}`）→ `await get_client().search(query, top_k)`，返回 answer/confidence/matched_docs/selected_nodes/pages；前端展示回答 + 命中引用。
- **FR4 配置读取**：`GET /api/config` 返回当前生效配置（model / retrieve_model / provider / base_url / key 掩码 / 文档数 / env 覆盖状态）。
- **FR5 配置写入（模型名）**：`POST /api/config` 传 `model` / `retrieve_model` → 运行时重建 client model + 定点写 config.yaml + 同步 `.env` MODEL_NAME/RETRIEVE_MODEL_NAME。
- **FR6 配置写入（凭据）**：`POST /api/config` 传 `api_key` / `base_url` → `configure_llm()` 运行时热切换 + 定点写 `.env`（OPENAI_API_KEY / OPENAI_BASE_URL）。
- **FR7 静态托管**：`GET /` 返回 `web/index.html`；`Mount("/static", StaticFiles(directory="web/static"))` 托管 `app.js` / `styles.css`。
- **FR8 鉴权放行**：`/` 与 `/static/*` 加入 `APIKeyMiddleware` 放行白名单（同 `/health`）；API 端点仍校验 `X-API-Key`。

**v1.1 新增**（FR9-FR12，本 spec 本次更新）：

- **FR9 上传进度条 + 滚动实时状态**：用户在大文件/多文件/慢索引场景下能看到过程进度而非"按下后转圈"。
  - 描述：上传过程中前端展示每文件的阶段状态（queued / uploading / indexing / succeeded / failed）与整体进度百分比；多文件时滚动日志区列出每文件的最新状态变更；支持取消正在上传的文件。
  - 现状（v1.0 缺口）：`/upload` 是同步批量端点（`server.py:555`），前端 `el-upload :http-request=onUpload`（`web/static/app.js`）只在响应返回后展示逐文件成功/失败，没有过程进度。
  - 验收信号（acceptance signal）：
    - 上传单个 50MB+ PDF 时，前端显示该文件从 0% → 100% 的进度条；
    - 多文件批量上传时，每个文件独立显示阶段状态（uploading / indexing / done）；
    - 整体有汇总进度条（`已完成数 / 总数`）；
    - 上传/索引失败的文件名以红色高亮 + 错误信息；
    - 用户可点击"取消"中止当前上传批次（已上传的字节流截断，正在索引的任务等其自然完成或失败，不强制 kill）。

- **FR10 文档删除（Web 入口）**：Web 控制台提供"删除文档"动作，等价 CLI 路径已具备删除能力（W2）。
  - 描述：用户点击文档表格行的"删除"按钮 → 弹出确认框（含文档名 + 不可恢复提示）→ 确认后调用 REST 端点执行删除 → 删除成功后从列表移除该文档；失败时显示错误。
  - 复用面（W2 已 ship）：
    - `db.delete_document(doc_id)`（db.py:337）— `DELETE FROM documents` 触发级联；
    - `server.py` delete handler（已被 `on_document_removed` 接线、`kb_identity.invalidate`、磁盘 `_safe_remove_upload` 清理）；
    - `super_tree.py:154` `on_document_removed`（keyword_index.remove_document + kb_identity.invalidate）；
    - `utils.py:173` `strip_markdown_fence`（KBIdentity 围栏剥离）；
    - 孤儿迁移脚本（幂等 DML，清理历史脏数据）；
    - 删除集成测试（端到端断言）。
  - 验收信号：
    - 文档表格行出现"删除"按钮（icon 或文字，hover 高亮）；
    - 点击删除弹出确认框（展示文档名 + "此操作不可恢复"提示）；
    - 确认后调用删除端点 → 后端走 W2 删除路径（DB 级联 + 索引失效 + 磁盘清理 + KBIdentity 失效）；
    - 删除成功后前端表格移除该行 + Toast 成功提示；
    - 删除失败（DB 错误 / 文件锁 / 权限）前端展示错误信息 + 行保留。

- **FR11 问答证据源模块**：问答结果页展示完整的证据链，而非仅 answer + 文档名列表。
  - 描述：当前 `POST /api/search` 已返回 `matched_docs` / `selected_nodes` / `pages` 字段（`server.py:403` `search_endpoint` 已 enrich `matched_docs[].doc_name`）；v1.1 前端新增"证据"模块：列出每个 matched_doc（含 doc_name / doc_description / pages 范围），对每个 selected_node 展示节点标题 + 节点路径 + 节点摘要片段 + 引用的 pages 列表（可点击预览但 v1.1 不实现全文预览）；答案下方显示"基于 N 篇文档 · 引用 M 个节点 · 涉及 X 个页码"。
  - 现状（v1.0 缺口）：`web/static/app.js` 问答区只显示 `<el-tag>{{ answer.confidence }}</el-tag>` + `<p>{{ answer.answer }}</p>` + `<ul><li v-for="d in answer.matched_docs">...`，未渲染 selected_nodes / pages，证据链不透明。
  - 验收信号：
    - 问答成功后下方出现"证据"折叠/展开区；
    - 每个 matched_doc 卡片展示：doc_name、doc_description（截断到 N 字符）、相关页码列表；
    - 每个 selected_node 展示：节点标题、节点路径（如 `Section 2.1`）、节点摘要片段（截断到 N 字符）；
    - 顶部汇总行显示"基于 N 篇文档 · 引用 M 个节点 · 涉及 X 个页码"；
    - 证据为空时（`matched_docs=[]`）显示占位"未找到相关证据"。

- **FR12 模型配置连通性测试**：用户在保存模型配置前可一键 ping 当前 model+api_key+base_url，验证端到端连通性。
  - 描述：模型配置 Tab 新增"测试连通性"按钮 → 调用新建端点对当前（或用户临时填写的）model/api_key/base_url 做最小 LLM 调用（如 `chat.completions.create(model=, messages=[{role:"user", content:"ping"}], max_tokens=8)`）→ 返回成功/失败 + 延迟（ms）+ 错误信息。前端展示：成功 → 绿色对勾 + 延迟数字（如 "OK · 312ms"）；失败 → 红色叉 + 错误原因（如 "401 Unauthorized" / "model not found"）。
  - 验收信号：
    - 模型配置 Tab 出现"测试连通性"按钮（独立于"应用"按钮）；
    - 点击调用连通性端点 → 前端显示 loading（"测试中…"）→ 返回后展示结果（成功/失败 + 延迟 + 错误）；
    - 用户可在表单未填完整时点击（端点读取表单当前值而非已持久化的值）；
    - 端点超时（如 10s）显示"超时"错误而非永久转圈；
    - 端点失败不影响现有"应用"按钮（POST /api/config 与 POST /api/config/test 解耦）。

### 非功能需求（NFR）

- **NFR1 零构建**：CDN 引入 Vue 3 + Element Plus，无 node 构建步骤。
- **NFR2 安全最小化**：GET 永不回显完整 key（掩码，仅露后 4 位）；写 `.env`/config.yaml 前必先备份（`.bak`）；落盘采用定点改键（`dotenv.set_key` / 行替换），绝不全量重写抹注释。
- **NFR3 不阻塞**：问答端点已 async（`await client.search`），前端 loading 态 + 超时/错误提示；连通性测试端点也需不阻塞（async 或线程池）。
- **NFR4 TDD**：后端端点 pytest + Starlette `TestClient`，先 RED 后 GREEN；v1.1 每个新端点/新 UI 模块都有对应测试。
- **NFR5 向后兼容**：不破坏现有 `/health` / `/upload` / MCP `/sse` / `/messages/`；FR10 删除路径不破坏 W2 既有 delete handler。
- **NFR6 v1.1 复用优先**：F2 必须复用 W2 `delete-path-integrity` 既有实现（db.delete_document + server handler + on_document_removed + strip_markdown_fence + 磁盘清理 + 孤儿迁移 + 测试），禁止重写或并行实现删除逻辑。F1/F3/F4 优先复用现有端点/字段，仅在前端或新增端点上扩展。
- **NFR7 v1.1 错误可见**：FR12 连通性测试失败、FR10 删除失败、FR9 索引失败，必须前端可见（Toast + 行内错误），不能静默。

### 范围

**v1.1 包含**（本 spec 本次更新）：
- FR9 上传进度条 + 阶段状态
- FR10 文档删除（Web 入口，复用 W2）
- FR11 问答证据源模块
- FR12 模型配置连通性测试

**v1.1 不包含**（v1.2+ 候选）：
- 单文档聚焦问答的独立入口（统一走多文档问答，v1.0 已确定）
- DashScope provider 一键切换（仍按 `.env.example` 手动改）
- config.yaml 调参项（魔法数字）的 UI 编辑
- 文档重命名/标签/分组
- 多用户/权限管理（当前鉴权只到 API Key 层级）
- 上传断点续传（v1.1 仅展示进度，不做分块上传）
- 全文预览（FR11 仅展示元数据 + 摘要片段，不渲染 PDF/MD 内容）
- 异步任务队列（FR9 进度条基于现有同步 /upload 端点，不引入 Celery/RQ）
- 删除前的导出/快照（v1.1 硬删除，FR10 不做软删除）
- 问答历史记录/收藏

### 假设（ASSUMPTION）

**v1.0 既有假设**：
- **A1**（✅ 已确认）：部署环境浏览器可访问外网 CDN（unpkg/jsdelivr）。故前端直接 CDN 引入 Vue 3 + Element Plus。
- **A2**：`.env` 与 `pageindex_mutil/config.yaml` 所在目录可写（落盘需要；运行时若只读，`persist:false` 仍可热切换）。
- **A3**（✅ 已确认）：模型名持久化"同步 .env"= config.yaml 与 .env MODEL_NAME/RETRIEVE_MODEL_NAME 写同一值。

**v1.1 新增假设**：
- **A4**（✅ 已确认）：W2 `delete-path-integrity` 已合入（commit e10df5f 验证：`db.delete_document` / `server.py` delete handler / `super_tree.on_document_removed` / `strip_markdown_fence` / 磁盘清理 / 孤儿迁移 / 集成测试 均存在）。FR10 仅需暴露 REST/MCP 入口 + 前端按钮 + 确认框。
- **A5**（✅ 已确认 + 由 design 阶段 **A-RESOLVED-5** 解析为 DD1-A 前端 XHR onprogress 模拟；详见 §4.3 Resolution Log）：FR9 上传进度条可通过现有 `/upload` 同步端点 + 前端"分段状态"模拟实现（不强制要求 SSE/流式后端）。即前端读取已索引文件数的"虚拟进度"，再叠加每文件实际 fetch onProgress（如 XHR `progress` 事件或 `fetch` 不支持则用 0% → 100% 快速跳变）。**真正的流式后端进度推送留 v1.2+**。
- **A6**（由 design 阶段 **A-RESOLVED-6** 解析为 DD4-A chat.completions ping；详见 §4.3 Resolution Log）：FR12 连通性测试使用 OpenAI 兼容 `chat.completions.create(messages=[{"role":"user","content":"ping"}], max_tokens=8)` 作为最小 ping；超时 10s；不走 `retrieve_model`（仅 ping 用户当前编辑的 `model`）。DashScope provider 当前 API 兼容 openai 客户端，故同一调用路径生效。
- **A7**（由 design 阶段 **A-RESOLVED-7** 解析为 CONFIRMED；详见 §4.3 Resolution Log）：FR12 端点读取表单**当前值**（未保存的草稿），不读取 `GET /api/config` 快照（用户可能改了表单还没点"应用"）。即端点 body 含 `model?` / `api_key?` / `base_url?`，缺省回退到当前生效值。
- **A8**（由 design 阶段 **A-RESOLVED-8** 解析为 DD2-A 单次 ElMessageBox.confirm；详见 §4.3 Resolution Log）：FR10 删除确认框文案为"删除文档 {doc_name}？此操作不可恢复"。二次确认（即先弹确认框 → 再要求输入文档名）v1.1 不做，留 v1.2+（除非用户明确要求）。
- **A9**（由 design 阶段 **A-RESOLVED-9** 解析为 DD3-A Accordion 默认折叠；详见 §4.3 Resolution Log）：FR11 证据模块默认折叠（accordion），用户点击展开；不为空时顶部显示 N 篇/M 个/X 个页码摘要。证据为空时显示占位文本。**视觉细节（折叠/标签页/抽屉）留 frontend-design 阶段打磨**。
- **A10**（由 design 阶段 **A-RESOLVED-10** 解析为 CONFIRMED 前端 XHR.abort()；详见 §4.3 Resolution Log）：FR9 取消按钮触发 `XMLHttpRequest.abort()`（如用 XHR）或忽略 fetch 响应（关闭 tab/不显示结果）。后端不感知取消，已上传字节的文件**继续索引**直至完成（不强制 kill）；用户取消的语义仅前端显示。
- **A11**（✅ 已确认）：v1.1 不引入新依赖、不引入新前端框架；FR9-FR12 均复用 Vue3 + ElementPlus（CDN）+ Starlette + pytest。

## §4 方案设计（System Design）

### 4.0 设计输入（已核实 — codegraph + Read + W2 spec）

| 来源 | 关键事实 | 对 v1.1 的影响 |
|------|---------|---------------|
| `server.py:383` `documents_endpoint` | 已返回 `{id, doc_name, doc_description, pdf_path}`。`doc_name` 来自 `pdf_name`。 | FR10 表格行可直接复用该字段。 |
| `server.py:403` `search_endpoint` | 已 `await client.search()` 并 **enrich** `matched_docs[].doc_name`（用 `c.documents` uuid→name 映射，getattr-guarded）。`result` 透传 `matched_docs` / `selected_nodes` / `pages` 字段。 | FR11 证据模块直接消费现有响应字段，**0 后端改动**。 |
| `server.py:436` `config_post_endpoint` | 四步幂等：runtime `configure_llm`/`_rebuild_client` + os.environ 同步 + 备份 + 定点写 + 回滚。已有 `persisted`/`applied` 字段。 | FR12 端点解耦于现有 `POST /api/config`（不动后者）；前端"测试连通性"按钮触发新端点。 |
| `server.py:555` `upload_endpoint` | **同步批量**：所有文件一次性索引完统一返回 `{"results": [...], "succeeded": N, "total": N}`，**无中间事件**。前端 `onUpload` 仅在响应返回后写一行结果。 | FR9 进度条 = **前端模拟**（无后端流式推送）；选 DD1-A（见 §4.1 DD1）。 |
| `server.py:151-174` `delete_document` MCP handler | 已修复（W2 commit `e10df5f`）：内存清理 → `db.delete_document`（cascade）→ `on_document_removed`（失效关键词+KBIdentity）→ 磁盘清理（`_safe_remove_upload`）。 | FR10 仅需**暴露 REST 端点**，复用 W2 实现；handler 内部调既有 MCP 删除路径的相同子步骤（§5.1）。 |
| `db.py:337` `delete_document(doc_id)` | `DELETE FROM documents WHERE id=?`，依赖既有 `ON DELETE CASCADE`；幂等（不存在 id 删 0 行）。 | FR10 REST handler 调此方法，NFR6 复用面明确。 |
| `pageindex_mutil/utils.py:61` `configure_llm(api_key=, base_url=)` | 重建全局 `_client = OpenAI(key, url)` / `_async_client = AsyncOpenAI(...)`。**同步 OpenAI 客户端**（非 AsyncOpenAI 调用 chat）。 | FR12 ping 必须使用**同步** `OpenAI().chat.completions.create(...)`，否则与 `configure_llm` 重建路径对称。 |
| `web/static/app.js:1-146` | 已 Vue3 + ElementPlus + `api()` 封装 + `loadDocs/onUpload/doSearch/saveConfig/loadConfig` + `matchedRows`（仅展示 doc_name）。 | FR9-FR12 仅扩展此文件：新增 `uploadProgress / deleteDoc / evidenceModule / testConnectivity` 逻辑；不改 fetch 封装。 |
| `web/static/app.js:62-77` `matchedRows` | 当前 `matched_docs` 渲染为 `<el-tag>` 列表，仅展示 `doc_name` + `score`。 | FR11 在此基础上扩展：嵌套 `selected_nodes` / `pages` / 摘要片段，**不重写** `matchedRows`。 |
| `docs/design-docs/PageIndex/delete-path-integrity/spec.md` §5.4 | W2 已写 `delete_document` handler（含 `_safe_remove_upload` 护栏）。 | FR10 直接复用，REST 入口挂在新路径 `/api/documents/{id}`。 |
| `docs/devkit/handoff-web-console-v1.1-clarify.md` | clarify 阶段已确认：W2 已 ship、A5/A11 已确认、A6-A10 暂留 design 决断；本 design 阶段已全部 RESOLVED（§4.3）。 | 本节负责 §4.2 DESIGN_DECISION + §4.3 解析。 |

### 4.1 替代方案表（≥2 per DD1-DD4）

#### DD1 — FR9 上传进度机制

| 取向 | 描述 | 优势 | 劣势 | 风险 |
|------|------|------|------|------|
| **A. 前端模拟（XHR onprogress + 阶段状态机）** | 改 `web/static/app.js` 的 `onUpload`：用 `XMLHttpRequest` 替换 `fetch`，绑 `upload.onprogress` 拿真实字节进度；用 `el-progress` 渲染 0-100%；文件大小 > 0 时进度精确。阶段状态机：`queued → uploading → indexing → done/failed`。`indexing` 阶段开始于 `onload` 触发、结束于响应 JSON 解析。**无后端改动**。 | 真实字节进度；0 后端风险；不动 `/upload`；保持现有契约 | 无法展示"pages 5/20"等解析阶段细粒度；多文件时整批同步等最后一份返回 | 取决于"indexing"阶段的视觉——前端只能以"灰色 + 旋转图标"占位 |
| **B. 后端流式（SSE/chunked HTTP）** | 改造 `upload_endpoint`：拆为 `/upload/start`（返回 upload_id） + `/upload/progress/{id}`（SSE）；每解析完一页推 `{upload_id, doc_id, page_done, page_total, stage}`。前端订阅 SSE。 | 真"解析页 5/20"展示；多文件并行进度清晰 | 改后端契约；前端需 SSE 客户端；服务器侧需维护 upload session 状态（in-memory map 或 DB）；与现有 `POST /upload` 同步契约冲突；前端复杂度陡增 | upload session 内存泄漏风险（异常崩溃未清理）；后端兼容性问题（MCP 客户端、CLI 测试可能复用 `POST /upload`） |
| **C. 轮询** | 上传时返回 `upload_id`；前端定时 `/api/upload/progress/{id}` 拉取进度；后端把进度写 in-memory dict 或 SQLite。 | 实现简单；后端可观测进度；前端简单 | 进度 laggy（轮询间隔 ≥ 1s）；占用后端资源；in-memory 状态进程重启丢失 | 与 B 同样的契约改动 + 额外的"状态过期"问题 |

**DESIGN_DECISION DD1 → A**（前端模拟）。理由：(1) v1.1 范围 = "可见进度"非"实时每页进度"；(2) 真字节进度（XHR `upload.onprogress`）已解决"按下后转圈"的主诉；(3) 0 后端改动 → 0 NFR5 风险（MCP/SSE 兼容）；(4) indexing 阶段用"灰色 indeterminate progress + 文件名 + spinner"视觉传达，不假装精确页码；(5) B/C 的契约改动成本与 v1.1 收益不匹配。**A-RESOLVED-5**（见 §4.3）。

#### DD2 — FR10 删除确认 UX

| 取向 | 描述 | 优势 | 劣势 | 风险 |
|------|------|------|------|------|
| **A. 单次确认（ElMessageBox.confirm）** | 弹框展示 "删除文档 '{doc_name}'?此操作不可恢复"，按钮 "确认删除"/"取消"。 | 1 click；符合 ElementPlus 习惯；开发最小 | 误点删除不可逆 | 删除即硬删，W2 已接受此语义 |
| **B. 双次确认（必须输入 doc_name）** | 弹框含输入框 "请输入文档名以确认"，用户输入完全匹配 `doc_name` 才启用确认按钮。 | 极大降低误删概率；用户必须"念出"doc_name | 2-3 步；表单交互复杂；移动端输入麻烦 | 误删概率 ≈ 0 的代价是 UX 摩擦，对单文档管理工具过重 |
| **C. 软确认（立即删 + Undo Toast）** | 点击立即调用删除 → 5s Toast 显示 "已删除 {doc_name} [撤销]"；5s 内点撤销回滚（重新插入）。 | 最佳 UX（Material Design 风格）；对误删兜底 | 复杂度陡增：需"删除是异步" + "撤销窗口" + "撤销需重新插入完整索引"（重索引成本高）；"撤销"语义不适用硬删 + cascade 场景 | 撤销 = 重建索引（秒级~分钟级），"软立即"语义不符 |

**DESIGN_DECISION DD2 → A**（单次确认）。理由：(1) 删除即硬删（W2 NFR 语义）；(2) W2 的 audit/磁盘清理已到位，错删可由"重建索引"恢复而非 UX 层；(3) v1.1 范围 = "Web 入口"+ "确认"，二次确认留 v1.2+；(4) A 已能满足 "NFR7 错误可见"——确认框文案明确"此操作不可恢复"。**A-RESOLVED-8**（见 §4.3）。

#### DD3 — FR11 证据模块布局

| 取向 | 描述 | 优势 | 劣势 | 风险 |
|------|------|------|------|------|
| **A. Accordion（默认折叠，按 doc 展开 → node → pages）** | 顶级 = "证据（N 篇文档）"折叠区，展开后每个 `matched_doc` 一个 `<el-collapse-item>`；展开某个 doc 后列出 `selected_nodes`（title + summary 截断 + pages list）。 | 与 ElementPlus `<el-collapse>` 原生契合；信息密度高；用户主动展开避免视觉过载；移动端友好 | 需 2 次点击才看到完整证据 | 无显著风险 |
| **B. Flat list（(doc, page, snippet) 三元组带 badge）** | 平铺所有证据三元组，每行展示 `doc_name` badge + `page 3` badge + 摘要截断；不带折叠。 | 一次看完；扫描友好 | 单文档多节点时过长；缺层级语义 | 文档多时一片混乱 |
| **C. 树视图（doc → node → page，hover 预览）** | `<el-tree>` 节点结构 + hover 显示 snippet popover。 | 强层级；适合大知识库 | 实现重；ElementPlus tree 配置繁；v1.1 范围外 | 过度工程 |

**DESIGN_DECISION DD3 → A**（Accordion 默认折叠）。理由：(1) 与 ElementPlus 习惯一致；(2) FR11 spec 明确"默认折叠，用户点击展开"；(3) 信息层级 = doc → node → page，与 FR11 验收信号一一对应；(4) FR11 spec 顶部"N 篇/M 个/X 个页码"摘要行独立于折叠区作头。**A-RESOLVED-9**（见 §4.3）。

#### DD4 — FR12 连通性测试实现

| 取向 | 描述 | 优势 | 劣势 | 风险 |
|------|------|------|------|------|
| **A. 轻量 chat.completions 调用** | `client.chat.completions.create(model=, messages=[{"role":"user","content":"ping"}], max_tokens=8)`，10s 超时，返回 `{ok, latency_ms, error?}`。 | 真测生成能力；语义清楚（"端到端可调用"）；超时阈值明确 | 需 token（极小：8 tokens）；L1 prompt 模型可能拒答"ping"（罕见，可捕获） | 单点失败模式相同于真实调用 |
| **B. GET /v1/models** | `GET {base_url}/models` → 验证 base_url+key 网络可达。 | 零 token；速度快；不依赖模型本身 | 不验证生成（key 仅有读权限、生成 quota 耗尽、model 名错误都暴露不出） | "测试通过但实际调用失败"是反信号 |
| **C. Combined（先 /models 后 /chat）** | 先 `/models`（快失败：网络/key）→ 通过后 `/chat.completions`（慢但真）。 | 准确 + 快反馈 | 2 次请求；UI loading 时间翻倍；冗余 | 复杂度提升；状态机分支增加 |

**DESIGN_DECISION DD4 → A**（chat.completions ping）。理由：(1) "连通性测试"语义 = "我能不能用这个配置问问题"，A 直接验证此语义；(2) B 假阳性太多（key 有效 ≠ 模型可用）；(3) C 复杂度收益不匹配（10s 内单 ping 已能覆盖大多数失败模式）；(4) max_tokens=8 + 10s 超时已含边界护栏。**A-RESOLVED-6**（见 §4.3）。

### 4.2 设计决策（DESIGN_DECISION）

| DD | 决策 | 替代品 | 选定方案 | 关键理由 |
|----|------|--------|---------|----------|
| **DD1** | FR9 上传进度机制 | A 前端模拟 / B 后端 SSE / C 轮询 | **A** | 0 后端改动；真字节进度（XHR onprogress）；NFR5 兼容；indexing 阶段用 indeterminate spinner 占位 |
| **DD2** | FR10 删除确认 UX | A 单次 / B 双次 / C Undo | **A** | W2 硬删语义；UX 摩擦最小；NFR7 错误可见已满足；二次确认留 v1.2+ |
| **DD3** | FR11 证据模块布局 | A Accordion / B Flat / C Tree | **A** | ElementPlus `<el-collapse>` 原生；与 FR11 spec "默认折叠"对齐；层级清晰 |
| **DD4** | FR12 连通性测试 | A chat ping / B /models / C combined | **A** | "我能用此配置问问题"语义直接验证；B 假阳性高；C 复杂度不匹配 |

### 4.3 Auto-Inferred Resolution Log（A5-A10）

| 假设 ID | 原状态（澄清阶段） | 解析后状态 | 解析依据 |
|---------|--------|------------|----------|
| **A5** (FR9 进度机制) | 前端 XHR onprogress 模拟 | **A-RESOLVED-5**：DESIGN_DECISION DD1 选 A；reason：0 后端改动 + 真字节进度；后端 SSE/流式推送留 v1.2+（spec §"v1.1 不包含"已列异步任务队列为 OUT）。 | §4.1 DD1 + §4.2 表 |
| **A6** (FR12 ping 实现) | openai 兼容 chat.completions "ping"，max_tokens=8，10s 超时，不走 retrieve_model | **A-RESOLVED-6**：DESIGN_DECISION DD4 选 A；reason：直接验证生成能力；10s 超时（spec 草拟值）经代码阶段微调（实际 Starlette `asyncio.wait_for(timeout=10.0)`）。DashScope provider 因走 OpenAI 兼容 client，自动覆盖。 | §4.1 DD4 + §5.1 `POST /api/config/test` |
| **A7** (FR12 端点读取表单值) | 端点 body 含 `model?`/`api_key?`/`base_url?`，缺省回退到当前生效值 | **A-RESOLVED-7**：CONFIRMED；reason：与 A6 绑定（test 用表单值而非已持久化值）→ §5.1 `POST /api/config/test` body schema 即此形式。 | §5.1 `POST /api/config/test` |
| **A8** (FR10 单次确认) | 单次确认 + "不可恢复" 提示 | **A-RESOLVED-8**：DESIGN_DECISION DD2 选 A；reason：硬删语义下二次确认过重；误删可由重建索引恢复。 | §4.1 DD2 + §5.3 `deleteDoc` |
| **A9** (FR11 证据模块 accordion) | 默认折叠 + 顶部 N/M/X 摘要 | **A-RESOLVED-9**：DESIGN_DECISION DD3 选 A；reason：ElementPlus `<el-collapse>` 原生契合；与 FR11 spec 描述一致。 | §4.1 DD3 + §5.3 `evidenceModule` |
| **A10** (FR9 取消 = 前端 only) | `XHR.abort()`；后端不感知，已上传字节继续索引 | **A-RESOLVED-10**：CONFIRMED；reason：与 DD1-A（前端模拟）一致——后端无进度事件可取消；客户端中断即视作"用户放弃查看"，后端按正常流程完成；前端 AbortController 在 onUpload 抛出后展示 "已取消"。 | §4.1 DD1 + §5.3 `uploadProgress.cancel` |

---

## §5 接口设计（Interface Design）

### 5.1 新增 REST 端点（v1.1）

#### 5.1.1 `DELETE /api/documents/{doc_id}` — FR10

| 项 | 内容 |
|----|------|
| 方法 | `DELETE` |
| 路径 | `/api/documents/{doc_id}` |
| 路径参数 | `doc_id: int`（DB 内 `documents.id`，非 uuid） |
| 鉴权 | `X-API-Key`（既有中间件，不放行白名单） |
| 成功响应 | `200 {"success": true, "doc_id": <int>}` |
| 错误响应 | `404 {"error": "Document not found"}`（若 doc_id 不存在且无级联删） / `500 {"error": "..."}`（DB/索引失败） |
| 实现要点 | 复用 W2 MCP `delete_document` handler 的子步骤：内存清理 → `c.db.delete_document(db_id)` → `c.super_tree_index.on_document_removed(db_id)` → `c.closet_index.remove_document(db_id)` → `_safe_remove_upload(pdf_path)`。把 W2 的内部代码抽为可复用函数 `delete_document_internal(db_id)`，MCP handler 与 REST handler 共享。 |
| 幂等性 | 重复删除同 id 返回 `{"success": true}`（DELETE 0 行 + FileNotFoundError 吞） |
| 测试 | `tests/test_web_console.py::test_delete_document_endpoint` + `tests/test_web_console.py::test_delete_document_idempotent` |

**DELETE handler 伪代码**（精确代码留 code 阶段）：
```python
async def document_delete_endpoint(request: Request) -> Response:
    doc_id_str = request.path_params.get("doc_id")
    try:
        doc_id = int(doc_id_str)
    except (TypeError, ValueError):
        return JSONResponse({"error": "doc_id must be integer"}, status_code=400)
    c = get_client()
    if c.db is None:
        return JSONResponse({"error": "DB unavailable"}, status_code=503)
    try:
        # Reuse W2 internal — extracted from the MCP delete_document handler.
        ok = delete_document_internal(c, doc_id)
        if not ok:
            return JSONResponse({"error": "Document not found"}, status_code=404)
    except Exception as e:
        logger.exception("document_delete_endpoint failed")
        return JSONResponse({"error": f"Delete failed: {e}"}, status_code=500)
    return JSONResponse({"success": True, "doc_id": doc_id})
```

**`delete_document_internal(client, db_id) -> bool`**（`server.py` 新增函数，W2 MCP handler 也调它）：
- 抽取自 W2 现有 MCP delete handler 的步骤序列。
- 返回 `bool`：True = 行存在并删除（含 0 行级联也返 True，幂等），False = DB 抛错（仅当 DB 不可用）。
- `_safe_remove_upload` 调用来自 W2。

#### 5.1.2 `POST /api/config/test` — FR12

| 项 | 内容 |
|----|------|
| 方法 | `POST` |
| 路径 | `/api/config/test`（独立于 `/api/config`，解耦） |
| 鉴权 | `X-API-Key` |
| 请求 body | `{model?: str, api_key?: str, base_url?: str}`（**全可选**，缺省回退当前生效值；A-RESOLVED-7） |
| 行为 | 用 body 参数（缺省回退）构造临时 `OpenAI(api_key, base_url)`，调 `chat.completions.create(model=, messages=[{"role":"user","content":"ping"}], max_tokens=8, timeout=10)` |
| 成功响应 | `200 {"ok": true, "latency_ms": 312, "model": <used>, "base_url": <used>}` |
| 失败响应 | `200 {"ok": false, "latency_ms": 0, "error": "401 Unauthorized"}` / `{"ok": false, "error": "Timeout after 10s"}` / `{"ok": false, "error": "model not found"}`（**永远 200**，错误信息在 body；前端用 `ok` 渲染红/绿） |
| 不影响状态 | 不改 `os.environ`、不写 `config.yaml`、不写 `.env`、不重建 client —— **纯只读 ping** |
| 测试 | `test_config_test_success`（mock openai 返回） / `test_config_test_timeout` / `test_config_test_401` / `test_config_test_fallback_to_active_config`（body 为空时用现有 `get_llm_config()`） |

**Handler 伪代码**：
```python
import time
from openai import OpenAI, APITimeoutError, AuthenticationError, NotFoundError

async def config_test_endpoint(request: Request) -> Response:
    try:
        body = await request.json() or {}
    except Exception:
        body = {}
    # A-RESOLVED-7: fall back to current values if field missing
    active_key, active_url = get_llm_config()
    api_key = body.get("api_key") or active_key
    base_url = body.get("base_url") or active_url
    # model: body overrides; else current ConfigLoader().load().model
    model = body.get("model") or ConfigLoader().load(None).model
    if not api_key or not model:
        return JSONResponse({"ok": False, "error": "Missing api_key or model"},
                            status_code=200)
    start = time.monotonic()
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        # A-RESOLVED-6: max_tokens=8, 10s timeout via run_in_executor
        await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=8,
                ),
            ),
            timeout=10.0,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        return JSONResponse({"ok": True, "latency_ms": latency_ms,
                             "model": model, "base_url": base_url})
    except asyncio.TimeoutError:
        return JSONResponse({"ok": False, "error": "Timeout after 10s",
                             "model": model})
    except AuthenticationError as e:
        return JSONResponse({"ok": False, "error": "401 Unauthorized: invalid api_key",
                             "model": model})
    except NotFoundError as e:
        return JSONResponse({"ok": False, "error": f"model not found: {model}",
                             "model": model})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}",
                             "model": model})
```

#### 5.1.3 路由注册（`server.py` `routes` 列表增量）

```python
Route("/api/documents/{doc_id}", endpoint=document_delete_endpoint, methods=["DELETE"]),
Route("/api/config/test", endpoint=config_test_endpoint, methods=["POST"]),
```

**新增 import**（如未存在）：`from openai import APITimeoutError, AuthenticationError, NotFoundError` / `import asyncio`（既有）。

#### 5.1.4 鉴权放行 / CORS

- 不放行：`/api/documents/{doc_id}` 与 `/api/config/test` 需 `X-API-Key`（与现有 `/api/*` 对称）。
- `CORSMiddleware` 已放行 `DELETE` / `OPTIONS`（`server.py:80-93` "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS"）—— **无需改动**。

### 5.2 既有端点契约确认（FR1-FR8 不变）

| 端点 | 方法 | 用途 | v1.1 影响 |
|------|------|------|----------|
| `/health` | GET | 状态 + 文档数 | 不变 |
| `/upload` | POST | 多文件批量上传 → 同步索引 | FR9 仅前端增强，**后端契约不变** |
| `/api/documents` | GET | 文档列表 | FR10 表格消费此端点，不变 |
| `/api/search` | POST | 多文档问答 | FR11 消费 `matched_docs`/`selected_nodes`/`pages`，不变 |
| `/api/config` | GET/POST | 配置读/写 | FR12 与此端点解耦（`POST /api/config/test` 不动此契约） |
| `/` | GET | 返回 `web/index.html` | 不变 |
| `/static/*` | GET | 静态资源 | 不变 |
| `/sse` + `/messages/` | GET/POST | MCP | 不变 |
| MCP `delete_document` tool | (MCP) | 现有删除工具 | FR10 复用其内部逻辑（抽函数），**MCP 工具契约不变** |

### 5.3 前端组件契约（v1.1 扩展 `web/static/app.js`）

| 组件 / 函数 | 职责 | Props / State | Events / 副作用 |
|-------------|------|---------------|----------------|
| `uploadProgress(file)` | FR9 单文件进度对象 | `{file, phase: 'queued'\|'uploading'\|'indexing'\|'done'\|'failed'\|'cancelled', pct: 0-100, xhr?: XMLHttpRequest}` | 注册到 `uploadList` 数组；`<el-progress>` 绑定 `pct` + `phase` |
| `uploadList` (ref) | 当前批次所有文件进度 | `UploadProgress[]` | 滚动日志区逐条 render；每个文件独立 row |
| `onUpload(opt)` (改写) | 上传处理（XHR 替代 fetch） | ElementPlus upload option | 创建 XHR、绑 `upload.onprogress` 推 `pct`、阶段切换、完成后 push result；支持 `abort()` |
| `deleteDoc(doc)` | FR10 表格行删除按钮 handler | `doc: {id, doc_name}` | `ElMessageBox.confirm` → `api("DELETE", /api/documents/{id})` → 成功移除行 + Toast；失败 Toast 错误并保留行 |
| `evidenceModule(answer)` | FR11 证据渲染函数 | `answer: {matched_docs, selected_nodes, pages}` | 顶部 N/M/X 摘要 + `<el-collapse>` 每 doc 展开 → `selected_nodes` 卡片（title + path + summary 截断 200 字 + page numbers） |
| `testConnectivity()` | FR12 模型配置 Tab 按钮 handler | `cfgForm` (model/api_key/base_url 当前值) | loading → `POST /api/config/test` body=cfgForm 非空字段 → 渲染 `ok?` 红/绿 + latency；按钮与 "应用" 独立 |

**uploadList 渲染结构**（ElementPlus）：
```
el-upload (drag, multiple, :show-file-list=false)
  └─ <el-button @click="cancelAll">取消全部</el-button> (仅 uploading/indexing 中显示)
  └─ <div v-for="up in uploadList">
        <el-progress :percentage="up.pct" :status="up.phase" :indeterminate="up.phase==='indexing'"></el-progress>
        <span>{{ up.file.name }} · {{ up.phase }}</span>
        <el-button v-if="up.phase==='uploading'" @click="up.xhr.abort()">取消</el-button>
     </div>
```

**evidenceModule 渲染结构**：
```
<div class="evidence">
  <div class="evidence-summary">基于 {{ N }} 篇文档 · 引用 {{ M }} 个节点 · 涉及 {{ X }} 个页码</div>
  <el-collapse v-model="expandedDocs">
    <el-collapse-item v-for="md in answer.matched_docs" :name="md.doc_id" :title="md.doc_name || md.doc_id">
      <div class="evidence-doc-desc">{{ md.doc_description || '(无描述)' }}</div>
      <div class="evidence-pages">页码: {{ md.pages?.join(', ') || '—' }}</div>
      <div v-for="node in selectedNodesForDoc(md.doc_id)" class="evidence-node">
        <h4>{{ node.title }} <span class="muted">{{ node.path }}</span></h4>
        <p>{{ truncate(node.summary, 200) }}</p>
        <div class="muted">pages: {{ node.pages.join(', ') }}</div>
      </div>
    </el-collapse-item>
  </el-collapse>
  <div v-if="!answer.matched_docs?.length" class="muted">未找到相关证据</div>
</div>
```

### 5.4 时序图

#### FR9 上传进度（含取消）

```
Browser                          server.py
  │                                  │
  │── onUpload(file) ───────────────►│
  │   create XHR                     │
  │── XHR POST /upload ──────────────►│
  │                                  │ (bytes flowing)
  │◄── upload.onprogress {75%} ──────│
  │   uploadList[i].pct = 75         │
  │   uploadList[i].phase='uploading'│
  │                                  │
  │  [用户点取消]                     │
  │── xhr.abort() ───────────────────►│ (server may ignore)
  │   uploadList[i].phase='cancelled'│
  │   (server 仍在索引此文件)         │
  │                                  │
  │   ────── 或正常完成 ──────         │
  │◄── upload.onload {response} ─────│
  │   uploadList[i].phase='indexing' │
  │   response = parse JSON          │
  │   uploadList[i].phase='done'|'failed'│
```

#### FR10 删除（REST + 复用 W2）

```
Browser                          server.py
  │                                  │
  │── ElMessageBox.confirm ─────────►│ (用户点 "确认删除")
  │── DELETE /api/documents/42 ─────►│
  │   X-API-Key: ...                 │
  │                                  │── document_delete_endpoint
  │                                  │   ├─ int(doc_id) = 42
  │                                  │   └─ delete_document_internal(c, 42):
  │                                  │       ├─ c.db.delete_document(42)  ← cascade
  │                                  │       ├─ c.super_tree_index.on_document_removed(42)
  │                                  │       ├─ c.closet_index.remove_document(42)
  │                                  │       └─ _safe_remove_upload(pdf_path)
  │◄── 200 {"success":true} ─────────│
  │   loadDocs() refresh            │
  │   ElMessage.success("已删除 ...") │
```

#### FR11 问答证据（无后端改动）

```
Browser                          server.py
  │                                  │
  │── POST /api/search ──────────────►│
  │                                  │── client.search(query, top_k)
  │◄── 200 {answer, confidence,     ──│
  │       matched_docs:[{doc_id,     │
  │                       doc_name,  │
  │                       pages:[…],  │
  │                       ...}],      │
  │       selected_nodes:[{...}],    │
  │       pages:[{page_number,       │
  │                snippet}]}        │
  │                                  │
  │   evidenceModule(answer)         │
  │   N = matched_docs.length        │
  │   M = selected_nodes.length      │
  │   X = unique pages count         │
  │   <el-collapse> per doc          │
```

#### FR12 连通性测试

```
Browser                          server.py
  │                                  │
  │── POST /api/config/test ────────►│
  │   {model:"qwen3.7-plus",         │
  │    api_key:"<用户当前填>"}        │
  │                                  │── config_test_endpoint
  │                                  │   ├─ 构造临时 OpenAI(key, url)
  │                                  │   └─ asyncio.wait_for(chat.completions.create(
  │                                  │       model, "ping", max_tokens=8), 10.0)
  │◄── 200 {ok:true,                 │
  │        latency_ms:312} ──────────│
  │   ElementPlus: 绿色对勾 "OK 312ms"│
  │                                  │
  │   ─── 或失败 ───                   │
  │◄── 200 {ok:false,                │
  │        error:"401 Unauthorized"} │
  │   ElementPlus: 红色叉 "401 ..."  │
```

### 5.5 MCP 复用面（FR10 与 W2）

W2 已 ship 的删除路径（`server.py` MCP `delete_document` handler）与 v1.1 REST `DELETE /api/documents/{doc_id}` **共享同一内部实现**：

```
delete_document_internal(client, db_id) -> bool  ← 抽取自 W2 MCP handler
  ├─ 内存清理 c.documents / c._uuid_to_db
  ├─ c.db.delete_document(db_id)         ← FR1 W2
  ├─ c.super_tree_index.on_document_removed(db_id)  ← FR2 W2
  ├─ c.closet_index.remove_document(db_id)         ← 既有
  └─ _safe_remove_upload(pdf_path, c.workspace)     ← FR4 W2
```

- MCP `delete_document` tool handler 改为调用 `delete_document_internal(c, db_id)`（参数适配：`arguments.get("doc_id")` 是 uuid → 先 `c._uuid_to_db.get(uuid)` 解析 db_id）。
- REST `document_delete_endpoint` 同样调用 `delete_document_internal(c, db_id)`（路径参数已是 db_id int）。
- v1.0 MCP 工具契约不变；v1.1 REST 端点是**新增入口**。

---

## §6 数据 / 配置变更（Data & Config Changes）

### 6.1 Schema 变更
**无**。FR10 复用 W2 cascade（`documents` → `nodes` / `pages` / `closet_tags` / `doc_keywords`）。FR9-FR12 均无 schema 影响。

### 6.2 配置文件变更
**无**。FR12 仅 ping，不修改 `config.yaml` / `.env`；FR9 进度条不引入新配置；FR11 无配置；FR10 无配置。

### 6.3 新增 / 修改文件

| 文件 | 状态 | v1.1 变更 |
|------|------|----------|
| `server.py` | 改 | 新增 `delete_document_internal()` 函数（从 W2 MCP handler 抽出）；新增 `document_delete_endpoint`；新增 `config_test_endpoint`；`routes` 列表加 2 个 Route；新增 openai exception 类的 import |
| `web/static/app.js` | 改 | `onUpload` 改 XHR 进度；新增 `uploadList`/`uploadProgress`/`cancelAll`；新增 `deleteDoc(doc)`；新增 `evidenceModule(answer)`；新增 `testConnectivity()`；模板新增证据 `<el-collapse>` + 上传进度 `<el-progress>` + 删除按钮 + 连通性按钮 |
| `web/index.html` | 改 | 文档 Tab 加 `<el-progress>` 行 + 删除按钮 + 取消按钮；问答 Tab 加 `<el-collapse>` 证据区；模型配置 Tab 加"测试连通性"按钮 |
| `web/static/styles.css` | 改 | `.evidence-*` / `.upload-row` 类样式（由 frontend-design skill 打磨） |
| `tests/test_web_console.py` | 改 | 新增 `test_delete_document_endpoint` / `test_delete_document_idempotent` / `test_config_test_success` / `test_config_test_timeout` / `test_config_test_401` / `test_config_test_fallback_to_active_config` |
| `docs/REGISTRY.md` | 改 | spec.md Last Modified → 2026-06-26（本日）；tasks.md 仍 STALE（待 task-planning 阶段更新） |

### 6.4 环境变量变更
**无**。FR12 ping 复用现有 `OPENAI_API_KEY` / `OPENAI_BASE_URL` 路径（与 v1.0 一致）。

### 6.5 磁盘文件变更
**无**。FR10 磁盘清理路径已由 W2 `_safe_remove_upload` 保护（`workspace/uploads/` 下校验）。

---

## §7 风险与缓解（Risk & Mitigation）

### 7.1 设计决策相关风险（per-DD）

| # | 风险 | 影响 | 缓解 | 回滚 |
|---|------|------|------|------|
| **R-DD1-1** | DD1-A 选 XHR，`upload.onprogress` 在浏览器 HTTPS 大文件下仍可能不准 | 进度条显示滞后实际 5-10% | `indexing` 阶段用 `indeterminate` 动画，遮盖字节进度不准；上限 99% 留 1% 给 indexing 阶段 | 改 fetch（无进度信息）但保持 indeterminate 占位 |
| **R-DD1-2** | 多文件上传时整批同步，最后一份才返回 | 视觉上前 N-1 个文件卡在 indexing 阶段 | 每个文件独立 row + 阶段状态；总进度条按 done 计数；明确这是"同步批量"已知约束 | 加 disable 按钮 + tooltip "批量上传请稍候" |
| **R-DD2-1** | 用户误删后无法前端撤销 | 误删只能重建索引 | 确认框文案明确"此操作不可恢复"；前端 Toast 显示"已删除" + 文档名；NFR7 错误可见；audit 由 W2 实现（无前端回退） | v1.2+ 可加 DD2-C Undo（成本高） |
| **R-DD3-1** | FR11 `matched_docs` 中 `pages`/`selected_nodes` 字段为 null/缺失 | `<el-collapse-item>` 报错 | `v-if` 守卫 + 默认占位（"—"）；`evidenceModule` 函数内部 `?.` + `\|\|` | 后端 enrich（已在 v1.0 ship） |
| **R-DD3-2** | 大知识库 N/M/X 摘要数字过大 | 摘要行视觉冗长 | `<span>{{ N }} 篇 / {{ M }} 节点 / {{ X }} 页</span>`，折叠时仅数字 | N/A |
| **R-DD4-1** | 测试用 `chat.completions.create` 触发实际 token 消耗 | 极小（max_tokens=8）；可忽略 | 设计阶段已确认；运维可监控 API quota | 改用 DD4-B（`/models`，但假阳性高） |
| **R-DD4-2** | 测试超时 10s 不够（大型模型冷启动） | UI 永久显示 "测试中..." | 10s 超时后返 `{"ok":false, "error":"Timeout after 10s"}`；前端红叉展示 | 把超时调高到 30s（折中） |
| **R-DD4-3** | 测试通过但实际配置失败（max_tokens=8 不暴露 quota 限制） | 误以为配置可用 | FR12 spec 已声明 "测试通过 ≠ 实际可用，请保存后用真实问答验证"；UI tooltip 提醒 | N/A |

### 7.2 FR 风险

| FR | 风险 | 缓解 |
|----|------|------|
| **FR9** | XHR 进度事件在某些反代/CDN 后被缓冲 | UI 文案提示"上传中，进度可能略有延迟"；超时 60s 后前端展示"上传超时，请重试" |
| **FR10** | REST 端点与 MCP 端点行为漂移 | `delete_document_internal` 单一入口（MCP + REST 共用）；AC 断言两端行为一致 |
| **FR10** | 删除前 `c.documents` / `c._uuid_to_db` 与 DB 不一致 | 取 `pdf_path` 必须**先**于 `delete_document`（W2 spec §5.4 时序已明） |
| **FR11** | `selected_nodes` 路径未在 v1.0 enrich | v1.1 不强求路径；展示 `title` + `summary` + `pages` 足够 |
| **FR11** | `matched_docs[].pages` 缺失 | `<div v-if="md.pages?.length">` 守卫；缺失显示 "—" |
| **FR12** | 测试端点被滥用作实际 ping | 与 `/api/config` 解耦（独立 endpoint）；测试不写入状态；key 掩码响应 |

### 7.3 跨切风险（v1.0 向后兼容 + W2 集成）

| # | 风险 | 缓解 |
|---|------|------|
| **R-X1** | v1.1 REST `DELETE /api/documents/{doc_id}` 暴露**整数 ID**，与 v1.0 MCP `delete_document` 工具用 UUID 不同 | REST 端点**直接用 DB int id**（前端 `docs[i].id` 已有），UUID 仅 MCP 使用；两端不互通文档明确 |
| **R-X2** | W2 MCP handler 重构（抽 `delete_document_internal`）破坏现有 MCP 行为 | 集成测试断言 W2 既有 AC 仍通过；`tests/test_delete_path_integrity.py` 不破坏 |
| **R-X3** | `POST /api/config/test` 与 `POST /api/config` 解耦不彻底 | test 端点**绝不调 `configure_llm` / 不写 env**；code review 强制检查 |
| **R-X4** | v1.1 引入新 import (`APITimeoutError` 等) 与 Starlette/uvicorn 版本冲突 | 已用 `openai` 库为现有依赖；新增 import 风险低 |
| **R-X5** | `asyncio.get_event_loop().run_in_executor` 在新 Python 版本 deprecated | code 阶段用 `asyncio.to_thread`（Python 3.9+）；spec 留口子 |

---

## §8 验收标准（Acceptance Criteria）

### 8.1 FR1-FR8（v1.0 既有，已 ship，不重复 AC）

v1.0 AC 见 `tasks.md`（v1.0 版）T1-T9 + spec §1-3 FR1-FR8 描述。**v1.1 不重测**，仅在 §8.6 回归测试中确保全绿。

### 8.2 FR9 — 上传进度条 + 滚动实时状态

- **AC9.1** 单文件 50MB+ PDF 上传：`<el-progress>` 从 0% 平滑涨到 99%（XHR `onprogress` 真实字节），最后一刻 `phase='indexing'` 用 indeterminate 动画，结束后 `phase='done'`。
- **AC9.2** 多文件批量上传：每个文件独立 row + 独立 `<el-progress>`；总进度条 `已完成数 / 总数`。
- **AC9.3** 失败文件红色高亮 + 错误信息（前端 `<ElNotification>` 已 ship，沿用）。
- **AC9.4** 取消按钮：在 `phase='uploading'` 时显示，点击触发 `xhr.abort()`，UI 展示 `phase='cancelled'`；已上传字节**继续索引**直至完成（不假装取消）。
- **AC9.5** 后端契约不变：`POST /upload` 仍返回 `{results, succeeded, total}`；前端不修改响应解析。

### 8.3 FR10 — 文档删除（Web 入口，复用 W2）

- **AC10.1** 文档表格行出现删除按钮（`<el-button type="danger" text @click="deleteDoc(doc)">删除</el-button>`）。
- **AC10.2** 点击删除 → `ElMessageBox.confirm` 文案："删除文档 {doc_name}？此操作不可恢复" + 确认/取消按钮。
- **AC10.3** 确认后调 `DELETE /api/documents/{id}` → 后端走 W2 删除路径（DB cascade + on_document_removed + closet_index + 磁盘清理）→ 返回 `{"success":true}` → 前端移除行 + Toast。
- **AC10.4** 删除失败（DB 错误/文件锁/权限）：Toast 红色错误信息，行保留在表格。
- **AC10.5** 删除幂等：连续 DELETE 同 id 两次，第二次仍返 `{"success":true}`（无异常）。
- **AC10.6** MCP 工具 `delete_document` 仍工作（行为不变）—— W2 集成测试断言。
- **AC10.7** `grep -n "delete_document_internal" server.py` 命中 ≥ 2 处（MCP handler + REST handler 共用同一函数）。

### 8.4 FR11 — 问答证据源模块

- **AC11.1** 问答成功后下方出现证据折叠区（`<el-collapse>` 默认全部折叠）。
- **AC11.2** 顶部摘要行："基于 N 篇文档 · 引用 M 个节点 · 涉及 X 个页码"（N/M/X 数字从 `matched_docs/selected_nodes/pages` 算）。
- **AC11.3** 每个 `matched_doc` 卡片展示：doc_name、doc_description（截断 200 字）、相关页码列表。
- **AC11.4** 展开某 doc 后列出 `selected_nodes`：节点 title + path + summary（截断 200 字）+ pages 列表。
- **AC11.5** 证据为空（`matched_docs=[]`）时显示 "未找到相关证据"。
- **AC11.6** `<el-collapse>` 移动端可用（不破坏布局）。
- **AC11.7** 不渲染 PDF/MD 全文内容（v1.1 不实现全文预览，spec "v1.1 不包含"已列）。

### 8.5 FR12 — 模型配置连通性测试

- **AC12.1** 模型配置 Tab 出现"测试连通性"按钮（独立于"应用"按钮）。
- **AC12.2** 点击 → loading 态（按钮 `loading=true`）→ 调用 `POST /api/config/test` body=当前表单非空字段。
- **AC12.3** 成功响应：`{"ok":true, "latency_ms":N}` → UI 绿色对勾 + "OK · {N}ms"。
- **AC12.4** 失败响应：`{"ok":false, "error":"..."}` → UI 红色叉 + 错误信息（401 / Timeout / model not found 等）。
- **AC12.5** 表单为空字段 → 端点回退到当前生效值（A-RESOLVED-7）：`GET /api/config` 拿 `model/api_key/base_url`，body 仅含用户修改过的字段。
- **AC12.6** 测试超时 10s：返 `{"ok":false, "error":"Timeout after 10s"}`；前端展示红叉 + "超时"。
- **AC12.7** 测试端点**不修改状态**：调前后 `os.environ` / `config.yaml` / `.env` 不变；不重建 client。
- **AC12.8** 测试端点**不阻塞** server：用 `asyncio.to_thread` + `asyncio.wait_for(timeout=10)`。

### 8.6 NFR 验收

- **NFR1（零构建）**：v1.1 不引入 `package.json` / node_modules / `.vue` / Vite 配置。
- **NFR2（安全最小化）**：测试端点响应**不包含完整 api_key**；写 `.env`/`config.yaml` 仅 `POST /api/config` 路径（已实现）；FR10 REST 不写任何配置。
- **NFR3（不阻塞）**：`POST /api/config/test` 用 `asyncio.to_thread` 跑同步 openai client，主事件循环不阻塞。
- **NFR4（TDD）**：
  - `test_delete_document_endpoint` 先 RED 后 GREEN；
  - `test_config_test_success/timeout/401/fallback` 先 RED 后 GREEN；
  - 已有 v1.0 测试（`tests/test_web_console.py` 14 个）保持全绿。
- **NFR5（向后兼容）**：`/health` / `/upload` / MCP `/sse`/`/messages/` 行为不变；MCP `delete_document` 工具行为不变；回归测试 `pytest -q` 全绿。
- **NFR6（v1.1 复用优先）**：FR10 调 `delete_document_internal`（从 W2 抽出）；FR11 后端 0 改动；FR12 ping 端点独立于 `POST /api/config`；FR9 仅前端 XHR 改写。
- **NFR7（错误可见）**：FR12 失败 → UI 红叉；FR10 失败 → Toast + 行保留；FR9 失败 → Notification + 行红字 + phase='failed'；FR11 证据为空 → 占位文本。

### 8.7 测试计划（TDD）

| 优先级 | 测试名 | FR | 类型 | 期望 |
|--------|--------|----|------|------|
| P0 | `test_delete_document_endpoint` | FR10 | backend TDD | RED: 404；GREEN: 200 success |
| P0 | `test_delete_document_idempotent` | FR10/NFR2 | backend TDD | 两次 DELETE 同 id 不抛异常 |
| P0 | `test_config_test_success` | FR12 | backend TDD (mock openai) | RED: 404；GREEN: 200 {ok:true, latency_ms>0} |
| P0 | `test_config_test_timeout` | FR12 | backend TDD (mock sleep 15s) | 200 {ok:false, error:"Timeout after 10s"} |
| P0 | `test_config_test_401` | FR12 | backend TDD (mock auth fail) | 200 {ok:false, error:"401 ..."} |
| P0 | `test_config_test_fallback_to_active_config` | FR12/A7 | backend TDD | body 空时用 `get_llm_config()` 现有值 |
| P1 | `test_delete_document_not_found` | FR10 | backend TDD | 404 |
| P1 | `test_search_endpoint_evidence_fields_unchanged` | FR11 | backend regression | `matched_docs/selected_nodes/pages` 字段保持 |
| P1 | v1.0 14 测试回归 | NFR5 | backend regression | 全绿 |
| P2 | browser-testing skill 手测：上传进度 UI / 删除按钮 / 证据折叠 / 连通性按钮 | FR9/10/11/12 | E2E | 截图 + console 无 error + network 200 |

### 8.8 Quality-gate 评分分解（self-reported，verifier 复核）

| 维度 | 权重 | self-score | 理由 |
|------|------|-----------|------|
| Spec 完整性（§1-8 覆盖所有 12 FR） | 25 | 23 | §1-3 v1.1 已有 + §4-8 本次新增；FR1-FR8 引用 v1.0 tasks.md AC |
| 设计决策（≥2 替代 + 选定 + 理由） | 20 | 19 | DD1-DD4 各 ≥3 替代；选定 + 理由明确 |
| 接口完整性（REST + 前端契约 + 时序图） | 20 | 19 | 5.1 REST + 5.3 前端 + 5.4 时序图 |
| 风险识别（per-DD + per-FR + 跨切） | 15 | 13 | 18 条风险项（R-DD1-1..2, R-DD2-1, R-DD3-1..2, R-DD4-1..3, FR9-12 risks, R-X1..5） |
| 验收标准可执行性（AC + 测试计划） | 10 | 9 | AC11-12 条 + 测试计划 9 项 |
| 假设解析（A5-A10 全部 RESOLVED） | 10 | 10 | §4.3 表 6 行全 RESOLVED |
| **总分（self-reported）** | 100 | **93** | 待 verifier 复核 |

---

## 设计阶段完成检查

- [x] §4 方案设计：DD1-DD4 各 ≥2 替代 + 选定 + 理由；A5-A10 解析为 A-RESOLVED-5..10。
- [x] §5 接口设计：2 个新增 REST 端点（DELETE /api/documents/{id}, POST /api/config/test）；前端 4 组件契约；4 时序图；MCP 复用面明确。
- [x] §6 数据/配置变更：0 schema 变更；0 配置变更；4 个文件改动 + 1 测试；0 新依赖。
- [x] §7 风险与缓解：5 设计决策风险 + 6 FR 风险 + 5 跨切风险 = 18 条。
- [x] §8 验收标准：FR9/10/11/12 各 5-8 条 AC + NFR 7 条 + 测试计划 9 项 + quality-gate self-score 93。
- [x] v1.0 NFR5 向后兼容、W2 NFR6 复用、NFR7 错误可见、NFR4 TDD 均纳入 §8 AC。

> 设计完成。下一步：独立 agent 调 quality-gate（threshold ≥70）→ 通过后进入 task-planning 阶段（tasks.md v1.1 扩展，v1.0 9 任务保留 + 新增 4 任务 = 13 任务）。

