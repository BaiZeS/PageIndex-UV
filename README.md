# PageIndex-UV

基于 [PageIndex](https://github.com/VectifyAI/PageIndex) 思想的文档结构化索引与推理问答工具。采用**非向量化（Vectorless）、推理驱动（Reasoning-based）**方式处理长文档（PDF/Markdown），所有核心代码内联在仓库中。

## 核心特性

- **零向量依赖**：纯结构化数据（JSON 树 + jieba 分词 + SQLite），无需 Embedding 或向量数据库
- **多文档联合检索**：Super-Tree Retrieval v3 四层架构（L0 双通道预过滤 → L1 LLM 选档 → L2 逐文档节点推理 → L3 上下文提取）
- **SQLite 缓存加速**：节点元数据、页面文本、树结构持久化，避免重复解析 PDF
- **多格式支持**：PDF、Markdown、DOCX、PPTX、XLSX 等
- **MCP 标准化服务**：通过 Model Context Protocol 对外暴露 RAG 能力，支持 Claude Desktop、Cursor 等 Agent 工具

## 快速开始

### 环境准备

1. 安装 [uv](https://github.com/astral-sh/uv)
2. 配置环境变量：复制 `.env.example` 为 `.env`，填入 LLM API Key

```bash
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY 或 DASHSCOPE_API_KEY
```

3. 安装依赖

```bash
uv sync
```

### 交互式问答

```bash
uv run python -m app.main
```

支持命令：`/add <文件路径>` 索引文档、`/doc <编号>` 聚焦单文档、`/list` 查看列表。

### MCP 服务

```bash
uv run python -m app.server
```

服务默认监听 `0.0.0.0:3000`，提供 SSE 传输的 MCP 协议端点和 HTTP API。

## Web 控制台

服务启动后浏览器打开 `http://localhost:3000/`，提供图形化的文档管理、问答和模型配置界面。前端走 CDN，无需构建步骤。

## MCP 服务部署

### Docker Compose（推荐）

```bash
docker compose up -d
docker compose logs -f
```

数据通过 volume 持久化：`./data:/app/data`

### 本地开发

```bash
API_KEY=testkey uv run python server.py
```

### 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `API_KEY` | 是 | HTTP 请求认证密钥 |
| `OPENAI_API_KEY` | 是* | OpenAI API Key |
| `DASHSCOPE_API_KEY` | 是* | 阿里云 DashScope API Key（与 OpenAI 二选一） |
| `OPENAI_BASE_URL` | 否 | 自定义 LLM 端点 |
| `MODEL_NAME` | 否 | 模型名（默认 `gpt-4.1-mini`） |
| `HOST` | 否 | 监听地址（默认 `0.0.0.0`） |
| `PORT` | 否 | 监听端口（默认 `3000`） |
| `WORKSPACE` | 否 | 文档存储目录（默认 `./data/workspace`） |
| `DB_PATH` | 否 | SQLite 数据库路径（默认 `./data/index.db`） |
| `SEARCH_BACKEND` | 否 | 搜索模式：`hybrid`（向量+关键词）或 `chroma`（纯向量，默认 `hybrid`） |
| `VECTOR_DB_PATH` | 否 | ChromaDB 存储目录（默认 `./data/vectors`） |

> *至少配置一个 LLM API Key。优先级：显式参数 > `OPENAI_API_KEY` > `DASHSCOPE_API_KEY`。

### HTTP API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/upload` | POST | 上传文件并索引（multipart/form-data） |
| `/sse` | GET | MCP SSE 连接端点 |
| `/messages/` | POST | MCP JSON-RPC 消息通道 |
| `/api/documents` | GET | 文档列表 |
| `/api/documents/{id}` | DELETE | 删除文档 |
| `/api/search` | POST | RAG 检索 |
| `/api/config` | GET/POST | 模型配置 |
| `/api/config/test` | POST | 测试模型连接 |

认证：所有端点（除 `/health` 和 `/static/*`）需携带 `X-API-Key` 请求头。

### MCP 工具

| 工具 | 参数 | 说明 |
|------|------|------|
| `search` | `query`, `top_k` | RAG 检索，返回答案+置信度+来源 |
| `list_documents` | 无 | 列出已索引文档 |
| `get_document` | `doc_id` | 获取文档元数据和结构 |
| `delete_document` | `doc_id` | 删除文档及索引 |

详见 [docs/mcp-tools.md](docs/mcp-tools.md)。

### Claude Desktop 配置

将以下内容添加到 `claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "pageindex-uv": {
      "type": "sse",
      "url": "http://localhost:3000/sse",
      "headers": { "X-API-Key": "your-api-key" }
    }
  }
}
```

## 项目结构

```
.
├── pageindex_mutil/           # 核心库（PageIndex 框架）
│   ├── page_index.py          # PDF 结构化索引
│   ├── page_index_md.py       # Markdown 结构化索引
│   ├── page_index_liteparse.py # 多格式解析
│   ├── closet_index.py        # 语义标签索引
│   ├── super_tree.py          # Super-Tree 检索 v3
│   ├── client.py              # 统一入口（单/多文档检索）
│   ├── search_backend.py      # 搜索后端接口
│   ├── hybrid_backend.py      # 混合搜索后端（向量+关键词）
│   ├── chroma_backend.py      # ChromaDB 向量搜索后端
│   ├── entity_extractor.py    # 实体抽取
│   ├── reasoning.py           # 推理逻辑
│   ├── retrieve.py            # 单文档检索辅助
│   ├── utils.py               # 工具函数
│   ├── config.yaml            # 索引/检索参数配置
│   ├── migrations/            # 数据库迁移脚本
│   └── agentic/               # Agentic 多策略路由
├── app/                       # 应用入口
│   ├── __init__.py
│   ├── main.py                # 交互式问答入口
│   ├── server.py              # MCP 服务 + HTTP API
│   ├── config_utils.py        # 纯函数配置工具
│   └── config_service.py      # 运行时配置服务
├── db.py                      # SQLite 缓存层
├── web/                       # Web 控制台（零构建）
│   ├── index.html             # 入口 HTML
│   └── static/                # 静态资源（Vue 3 + Element Plus）
├── docs/                      # 项目文档
├── deploy/                    # 部署配置模板
├── tests/                     # 测试套件
├── .github/workflows/         # CI/CD
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

## 技术栈

Python 3.12+ / uv / SQLite / PyMuPDF / OpenAI SDK / MCP SDK / Starlette + uvicorn / jieba / tiktoken

## 文档

- [系统架构](docs/architecture.md)
- [MCP 工具参考](docs/mcp-tools.md)
- [Web 控制台设计系统](docs/web-console-design-system.md)
