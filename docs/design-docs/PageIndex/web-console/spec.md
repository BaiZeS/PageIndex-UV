# Web 控制台（前端页面 + CLI/模型配置迁移）— spec.md

> 阶段：design（§1-8 完整）。
> 来源：用户 `/devkit:main` → "给应用加一个简单的前端页面，把 CLI 功能搬上去，模型配置也搬上去，页面做好看点"。
> 当前分支：`main`（clean）。实现时应在特性分支 `feat/web-console` 上进行。
> 用户确认决策：架构=复用 server.py；前端=单文件 HTML+CDN UI 库；模型名持久化=config.yaml+同步 .env；凭据持久化=configure_llm 运行时热切换+写 .env；鉴权=页面/静态放行+key 存 localStorage。

## §1 背景

PageIndex-UV 当前以两种形态对外：
- `main.py` — 交互式 REPL CLI，核心动作：`/add`（索引 PDF/MD）、`/list`（列文档）、`/doc <n>`（单文档问答）、默认输入（多文档问答）。
- `server.py` — Starlette 服务，端口 `3000`：`GET /health`、`POST /upload`（上传→索引）、`/sse`+`/messages/`（MCP）。启动时在 `lifespan` 创建全局 `PageIndexClient`，所有端点经 `get_client()` 共享。

模型配置已有一套统一收尾（W6）：`pageindex_mutil/utils.py` 的 `_resolve_llm_config` / `configure_llm(api_key, base_url)` 支持运行时重建 client；`ConfigLoader().load()` 从 `pageindex_mutil/config.yaml` 读 `model` / `retrieve_model`，env `MODEL_NAME` / `RETRIEVE_MODEL_NAME` 可覆盖。

**关键现状（已核实，2026-06-24）**：
- 项目**零前端工具链**（无 `package.json` / `.vue` / node 构建）。
- 实际 `.env`：`MODEL_NAME=qwen3.7-plus`（已设置，**覆盖** config.yaml 的 `model: gpt-4.1-mini`）；`OPENAI_API_KEY` 已设；`OPENAI_BASE_URL=http://10.135.11.111/` 已设；`RETRIEVE_MODEL_NAME` 未设；`API_KEY=<见 .env，勿入库>` 已设（MCP HTTP 鉴权）。
- 故：单纯写 config.yaml 的 `model` **不会生效**（env 覆盖仍在）。

## §2 目标

- **一个 Web 控制台页面**，把 CLI 的核心动作（索引/列文档/问答）图形化，并提供模型配置 UI。
- **复用 server.py 单进程**：新增 REST 端点 + 静态托管，天然共享全局 client 与配置，零部署增量。
- **模型配置可热切换并持久化**：运行时即时生效（`configure_llm`），重启后保留（写 config.yaml + .env，保持二者一致）。
- **页面美观**：实现阶段加载 `frontend-design` skill 做排版/留白，避免默认模板感。
- **零前端构建**：不引入 node/Vite 构建链。

## §3 需求与范围

### 功能需求（FR）

- **FR1 文档列表**：`GET /api/documents` 返回已索引文档（id / 名称 / 描述 / 页数或行数），等价 CLI `/list`。
- **FR2 索引（上传）**：复用现有 `POST /upload`；前端拖拽上传 + 逐文件结果展示。
- **FR3 问答**：`POST /api/search`（body `{query, top_k?}`）→ `await get_client().search(query, top_k)`，返回 answer/confidence/matched_docs/selected_nodes/pages；前端展示回答 + 命中引用。
- **FR4 配置读取**：`GET /api/config` 返回当前生效配置（model / retrieve_model / provider / base_url / key 掩码 / 文档数 / env 覆盖状态）。
- **FR5 配置写入（模型名）**：`POST /api/config` 传 `model` / `retrieve_model` → 运行时重建 client model + 定点写 config.yaml + 同步 `.env` MODEL_NAME/RETRIEVE_MODEL_NAME。
- **FR6 配置写入（凭据）**：`POST /api/config` 传 `api_key` / `base_url` → `configure_llm()` 运行时热切换 + 定点写 `.env`（OPENAI_API_KEY / OPENAI_BASE_URL）。
- **FR7 静态托管**：`GET /` 返回 `web/index.html`；`Mount("/static", StaticFiles(directory="web/static"))` 托管 `app.js` / `styles.css`。
- **FR8 鉴权放行**：`/` 与 `/static/*` 加入 `APIKeyMiddleware` 放行白名单（同 `/health`）；API 端点仍校验 `X-API-Key`。

### 非功能需求（NFR）

- **NFR1 零构建**：CDN 引入 Vue 3 + Element Plus，无 node 构建步骤。
- **NFR2 安全最小化**：GET 永不回显完整 key（掩码，仅露后 4 位）；写 `.env`/config.yaml 前必先备份（`.bak`）；落盘采用定点改键（`dotenv.set_key` / 行替换），绝不全量重写抹注释。
- **NFR3 不阻塞**：问答端点已 async（`await client.search`），前端 loading 态 + 超时/错误提示。
- **NFR4 TDD**：后端端点 pytest + Starlette `TestClient`，先 RED 后 GREEN。
- **NFR5 向后兼容**：不破坏现有 `/health` / `/upload` / MCP `/sse` / `/messages/`。

### 范围

**包含**：§3 FR1–FR8、文档/问答/模型配置三页 UI。
**不包含**（v1.1）：文档删除、单文档聚焦问答的独立入口（统一走多文档问答）、DashScope provider 一键切换（仍按 `.env.example` 手动改）、config.yaml 调参项（魔法数字）的 UI 编辑。

### 假设（ASSUMPTION）

- **A1**（✅ 已确认）：部署环境浏览器可访问外网 CDN（unpkg/jsdelivr）。故前端直接 CDN 引入 Vue 3 + Element Plus。
- **A2**：`.env` 与 `pageindex_mutil/config.yaml` 所在目录可写（落盘需要；运行时若只读，`persist:false` 仍可热切换）。
- **A3**（✅ 已确认）：模型名持久化"同步 .env"= config.yaml 与 .env MODEL_NAME/RETRIEVE_MODEL_NAME 写同一值。

## §4 方案设计（System Design — Alternatives & Trade-offs）

### 4.0 设计输入（已核实，codegraph + Read）

- `server.py:516-522` routes 列表；`:483` lifespan 创建 `client = PageIndexClient(workspace, db_path)`；`:93` `get_client()`；`:60` `APIKeyMiddleware`（放行 `/health` + OPTIONS）。
- `pageindex_mutil/utils.py:38` `_resolve_llm_config`、`:61` `configure_llm(api_key, base_url)`、`:74` `get_llm_config()`、`:764` `ConfigLoader.load`（优先级：kwarg > env MODEL_NAME/RETRIEVE_MODEL_NAME > config.yaml）。
- `pageindex_mutil/client.py:81` `index(file_path, mode)`、`:298` `async search(query, top_k)`、`:269` get_document*。
- `db.get_all_documents()`（CLI `/list` 数据源）。
- `python-dotenv` 已安装，`set_key` / `get_key` / `dotenv_values` 可用。

### 4.1 替代方案（已评估并定夺）

| # | 方案 | 结论 |
|---|------|------|
| 集成 | 复用 server.py（新端点+静态托管）vs 新建 FastAPI 服务 | **选复用 server.py**：单进程共享 client/config，零部署增量。 |
| 前端 | 单文件 HTML+CDN vs Vue3 SPA+Vite vs 原生手写 | **选单文件 HTML+CDN(Vue3+ElementPlus)**：零构建、组件齐全、好看快。 |
| 模型名持久化 | config.yaml / .env / 双写同步 | **选双写同步**：仅 config.yaml 被 env 覆盖会失效，双写保证生效且一致。 |
| 凭据持久化 | 运行时切换 only / +写 .env | **选运行时切换+写 .env**（用户定夺）。 |
| .env 落盘机制 | 全量重写 / `dotenv.set_key` 定点 | **选定点 set_key**：保注释、控风险。 |
| config.yaml 落盘 | `yaml.dump` / 定点行替换 / ruamel | **选定点行替换**：保注释、零新依赖。 |

### 4.2 设计决策（DESIGN_DECISION — 已确认）

1. 三页式 UI（文档 / 问答 / 模型配置），Element Plus Tabs + 表格/表单。
2. 后端 4 个新端点（documents/search/config GET/config POST）+ 静态托管 + 鉴权放行。
3. 配置写入采用"先备份 → 定点改 → 运行时应用 → 校验回读"四步幂等流程。
4. 模型名双写：config.yaml 行替换 + .env `set_key`；二者同值。
5. 凭据（api_key/base_url）写入限定为 `OPENAI_API_KEY` / `OPENAI_BASE_URL` 两个键（当前活跃路径；DashScope 切换留 v1.1 手动）。

## §5 接口设计（Interface Design）

### 5.1 新增 REST 端点（server.py）

```
GET  /                  → 返回 web/index.html（只读）
MNT  /static            → StaticFiles(directory="web/static")
GET  /api/documents     → { documents: [ {id, doc_name, doc_description, pages|lines, type} ] }
POST /api/search        → body {query:str, top_k?:int} → client.search() 原结构
GET  /api/config        → 见 5.2
POST /api/config        → body 见 5.3
```

### 5.2 `GET /api/config` 响应

```json
{
  "model": "qwen3.7-plus",              // 生效值（env>yaml 解析后）
  "retrieve_model": "qwen3.7-plus",    // 生效值，未设则回退 model
  "model_yaml": "gpt-4.1-mini",        // config.yaml 原始 model 字段
  "retrieve_model_yaml": "gpt-4.1-mini",
  "provider": "openai",                // openai | dashscope | none；派生自 _resolve_llm_config（OPENAI_API_KEY 设→openai；仅 DASHSCOPE_API_KEY→dashscope；都无→none）
  "base_url": "http://10.135.11.111/",  // 派生自 get_llm_config()[1]
  "api_key_masked": "sk-xxxx****xxxx",
  "has_api_key": true,
  "document_count": 3,
  "env_overrides": { "MODEL_NAME": "qwen3.7-plus", "RETRIEVE_MODEL_NAME": null, "OPENAI_BASE_URL": "http://10.135.11.111/" }
}
```

### 5.3 `POST /api/config` 请求（字段均可选，传谁改谁）

```json
{
  "model": "...",          // → 运行时 client.model 重建 + 写 config.yaml(model) + 写 .env MODEL_NAME
  "retrieve_model": "...", // → 同上（retrieve_model / RETRIEVE_MODEL_NAME）
  "api_key": "...",        // → configure_llm(api_key=) 运行时 + 写 .env OPENAI_API_KEY
  "base_url": "...",       // → configure_llm(base_url=) 运行时 + 写 .env OPENAI_BASE_URL
  "persist": true          // 默认 true。false=仅运行时热切换，不落盘（快速试错）
}
```

**响应**：回读后的 `GET /api/config` 快照 + `{"applied": true, "persisted": true}`。

### 5.4 配置写入四步幂等流程（FR5/FR6 实现）

```
1. 备份：copy config.yaml → config.yaml.bak；copy .env → .env.bak（已存在则覆盖）
2. 定点改：
   - 模型名 → 行替换 config.yaml 的 ^model: / ^retrieve_model: 行（保留注释）
   - .env   → dotenv.set_key(".env", "MODEL_NAME", v)（set_key 保格式）
3. 运行时应用：
   - 凭据 → configure_llm(api_key=, base_url=)（重建 _client/_async_client）
   - 模型 → client.model = ConfigLoader().load({model:..}).model；client.retrieve_model = ...
4. 校验回读：重新 get_llm_config() / ConfigLoader().load() 验证生效；失败则回滚（恢复 .bak）。
```

### 5.5 鉴权（FR8）

- `APIKeyMiddleware` 放行列表新增 `/`、`/static`（前缀匹配，同 `/health` 写法）。
- `index.html` 内置 key 输入框，存 `localStorage["pageindex_api_key"]`；所有 fetch 注入 `X-API-Key`。
- `API_KEY` 未设 → 全放行（开发态）。

## §6 数据/配置变更（Data & Config Changes）

### 6.1 新增文件

```
web/index.html          # Vue3+ElementPlus(CDN) 单页入口，三 Tabs
web/static/app.js       # Vue app 逻辑（fetch 封装、文档/问答/配置）
web/static/styles.css   # 自定义样式（实现期 frontend-design 打磨）
docs/design-docs/PageIndex/web-console/{spec,tasks}.md
```

### 6.2 配置写入面（落盘即改）

- `pageindex_mutil/config.yaml`：`model:` / `retrieve_model:` 行（定点替换，注释保留）。
- `.env`：`MODEL_NAME` / `RETRIEVE_MODEL_NAME` / `OPENAI_API_KEY` / `OPENAI_BASE_URL`（`set_key` 定点）。
- 备份：`*.bak`（写前生成，写后保留以便回滚）。

### 6.3 server.py 改动面

- 导入 `StaticFiles`、`FileResponse`（或 `HTMLResponse`）。
- routes 新增 5 条（见 5.1）。
- `APIKeyMiddleware` 放行列表加 `/`、`/static`。
- 新增端点函数：`documents_endpoint` / `search_endpoint` / `config_get` / `config_post` + 静态 `index` 路由。
- 配置写回封装为新模块 `web_config.py`（备份/定点写/回读/回滚），供端点与测试复用。

## §7 风险与缓解（Risk & Mitigation）

### 7.1 写密钥到 `.env`（安全）

**风险**：UI 把 api_key 写磁盘，泄露/误改风险。
**缓解**：① 仅写 `OPENAI_API_KEY`/`OPENAI_BASE_URL`；② 写前 `.env.bak` 备份；③ `set_key` 定点不改他行；④ GET 永远掩码；⑤ 端点受 `API_KEY` 鉴权保护；⑥ 提供 `persist:false` 仅热切换不落盘选项。

### 7.2 模型名双写一致性

**风险**：config.yaml 与 .env 写不同步导致歧义。
**缓解**：同一值同事务写两处（§5.4 步骤 2）；写后回读校验；任一失败回滚 `.bak`。

### 7.3 CDN 不可达（内网）

**风险**：A1 不成立 → 页面空白。
**缓解**：CDN 加 `onerror` 回退到备用 CDN；仍不可用时 Element Plus 降级为原生控件（v1.1 可考虑 vendor 自托管）。在 README 注明网络要求。

### 7.4 鉴权 key 泄露于 localStorage

**风险**：XSS 可读 localStorage 中的 key。
**缓解**：key 本是服务端签发、同源使用；UI 不内联敏感数据；注明同源部署前提。

### 7.5 问答慢/超时

**风险**：agentic router 多轮 LLM 调用耗时长，前端挂起。
**缓解**：前端 loading + 端点 async 不阻塞；超时友好提示。

## §8 验收标准（Acceptance Criteria）

### 8.1 功能验收（映射 §3 FR）

- [ ] FR1 `GET /api/documents` 返回 db.get_all_documents() 数据。
- [ ] FR2 前端拖拽上传，复用 `/upload`，展示逐文件成功/失败。
- [ ] FR3 `POST /api/search` 返回 answer 等字段；前端展示。
- [ ] FR4 `GET /api/config` 返回 5.2 全字段，key 掩码。
- [ ] FR5 改 model → config.yaml + .env MODEL_NAME 同值；重启后生效；client.model 运行时更新。
- [ ] FR6 改 api_key/base_url → configure_llm 运行时 + .env 写入；后续问答用新凭据。
- [ ] FR7 `GET /` 出页面，`/static/app.js` 可加载。
- [ ] FR8 API_KEY 设时端点拒绝无 key 请求；未设时放行。

### 8.2 测试（NFR4 TDD，先 RED）

`tests/test_web_console.py`：
- `test_list_documents`、`test_search_endpoint`、`test_config_get_masks_key`、`test_config_post_model_dualwrite`、`test_config_post_credentials_env`、`test_config_post_persist_false_runtime_only`、`test_backup_created_before_write`、`test_rollback_on_failure`、`test_static_index_served`、`test_auth_skipped_for_health_and_static`。
- 全部用临时 workspace/.env/config.yaml（`tmp_path` fixture），不污染真实配置。

### 8.3 过程纪律（devkit 铁律）

- L2 TDD：每个端点先写失败测试，肉眼确认 RED，再实现。
- Two-Agent Minimum：编码由子代理（LEAF EXECUTOR）产出，独立验证子代理评审；主会话只编排。
- quality-gate：code ≥80 / review ≥70 / verify ≥75 方可 ship。
- 实现期加载 `frontend-design`、`browser-testing`、`std-python` skill。
