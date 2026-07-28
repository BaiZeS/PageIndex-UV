# PageIndex-UV 项目全面审查与优化计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent (recommended) or compose:execute to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 全面检查项目所有内容，优化代码架构，逐级更新所有文档，清理脏数据和文档

**Architecture:** 本计划涵盖代码质量、文档一致性、测试修复、配置统一四个维度的优化

**Tech Stack:** Python 3.12+ / uv / SQLite / PyMuPDF / OpenAI SDK / MCP SDK / Starlette + uvicorn / jieba / tiktoken / ChromaDB

## Global Constraints

- 所有测试必须通过（当前 4 个失败需修复）
- 文档必须与代码实际行为一致
- 配置默认值必须在各处统一
- 不破坏现有 API 接口

---

## 审查发现汇总

### 1. 测试失败（4 个）

| 测试文件 | 测试名 | 失败原因 |
|---------|--------|---------|
| `tests/test_retrieve_model_wiring.py` | `test_call_llm_json_uses_retrieve_model_when_set` | `import main` 失败，应为 `app.main` |
| `tests/test_retrieve_model_wiring.py` | `test_call_llm_json_falls_back_to_model_when_retrieve_unset` | 同上 |
| `tests/test_retrieve_model_wiring.py` | `test_generate_answer_uses_retrieve_model_when_set` | 同上 |
| `tests/test_retrieve_model_wiring.py` | `test_generate_answer_falls_back_to_model_when_retrieve_unset` | 同上 |

**根因:** 测试代码第 172 行 `import main as main_mod` 应为 `import app.main as main_mod`

### 2. 文档不一致

| 文件 | 问题 | 修复方案 |
|------|------|---------|
| `README.md:66` | `uv run python server.py` 应为 `uv run python -m app.server` | 修正命令 |
| `docs/architecture.md:3` | 描述为 "non-vector, reasoning-based RAG" 但实际支持向量搜索 | 更新描述 |
| `docs/architecture.md` | 架构图未提及实体知识图谱和搜索后端 | 补充架构图 |
| `.dockerignore:33-34` | 引用不存在的路径 `docs/mempalace/` 和 `PageIndex/tests/results/` | 清理 |
| `.dockerignore:40` | `=2.8.0` 看起来是 typo | 删除 |

### 3. 配置默认值不一致

| 配置项 | config.yaml | .env.example | docker-compose.yml |
|--------|-------------|--------------|-------------------|
| `search_backend` | `hybrid` | `keyword` | `keyword` |
| `model` | `deepseek-v4-flash` | `gpt-4.1-mini` | `qwen-plus` |

**建议:** 统一为 `keyword`（默认无需向量依赖）和 `gpt-4.1-mini`（OpenAI 默认）

### 4. 代码质量问题

| 文件 | 问题 | 严重程度 |
|------|------|---------|
| `pageindex_mutil/` | 包名拼写错误 "mutil" 应为 "multi" | 低（改名影响大） |
| `pageindex_mutil/utils.py` | 878 行，职责过多 | 中 |
| `pageindex_mutil/page_index.py` | 1183 行，过于庞大 | 中 |
| `app/server.py` | 1063 行，过于庞大 | 中 |
| `chroma_backend.py:72-76` | TODO 注释未实现 Ollama/OpenAI embedding | 低 |

### 5. 缺失功能

| 功能 | 状态 | 说明 |
|------|------|------|
| `DocIdMapper` 类 | 架构文档提及但未在代码中找到 | 需确认是否已实现 |
| Ollama embedding 支持 | TODO | chroma_backend.py:72 |
| OpenAI embedding 支持 | TODO | chroma_backend.py:76 |

---

## 实施任务

### Task 1: 修复测试失败

**Covers:** 测试质量

**Files:**
- Modify: `tests/test_retrieve_model_wiring.py:172`

**Steps:**

- [ ] **Step 1: 修复 import 路径**

```python
# 将第 172 行从:
import main as main_mod
# 改为:
import app.main as main_mod
```

- [ ] **Step 2: 运行测试验证**

```bash
uv run pytest tests/test_retrieve_model_wiring.py -v
```

Expected: 所有测试通过

- [ ] **Step 3: Commit**

```bash
git add tests/test_retrieve_model_wiring.py
git commit -m "fix(tests): correct import path for main module in retrieve_model_wiring tests"
```

---

### Task 2: 修复 README.md 命令错误

**Covers:** 文档准确性

**Files:**
- Modify: `README.md:66`

**Steps:**

- [ ] **Step 1: 修正服务器启动命令**

将 `README.md:66` 从:
```
API_KEY=testkey uv run python server.py
```
改为:
```
API_KEY=testkey uv run python -m app.server
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(readme): fix server startup command to use module syntax"
```

---

### Task 3: 更新架构文档

**Covers:** 文档完整性

**Files:**
- Modify: `docs/architecture.md`

**Steps:**

- [ ] **Step 1: 更新项目描述**

将第 3 行从:
```
PageIndex-UV is a **non-vector, reasoning-based RAG** tool over long documents (PDF / Markdown).
```
改为:
```
PageIndex-UV is a **hybrid retrieval RAG** tool over long documents (PDF / Markdown / DOCX / PPTX / XLSX). It supports keyword, vector (ChromaDB), and hybrid search backends with reasoning-based document selection.
```

- [ ] **Step 2: 补充架构图中的搜索后端和实体图谱**

在架构图中添加:
- Search Backends (keyword / hybrid / chroma)
- Entity Knowledge Graph
- LiteParse multi-format parser

- [ ] **Step 3: Commit**

```bash
git add docs/architecture.md
git commit -m "docs(architecture): update description to reflect hybrid search and multi-format support"
```

---

### Task 4: 清理 .dockerignore

**Covers:** 部署配置清理

**Files:**
- Modify: `.dockerignore`

**Steps:**

- [ ] **Step 1: 删除无效路径和 typo**

删除:
- `docs/mempalace/` (不存在)
- `PageIndex/tests/results/` (不存在)
- `=2.8.0` (typo)

- [ ] **Step 2: Commit**

```bash
git add .dockerignore
git commit -m "chore(dockerignore): remove invalid paths and typo"
```

---

### Task 5: 统一配置默认值

**Covers:** 配置一致性

**Files:**
- Modify: `pageindex_mutil/config.yaml`
- Modify: `.env.example`
- Modify: `docker-compose.yml`

**Steps:**

- [ ] **Step 1: 统一 search_backend 默认值**

将 `config.yaml` 的 `search_backend` 从 `hybrid` 改为 `keyword`（与 .env.example 和 docker-compose.yml 一致，无需向量依赖）

- [ ] **Step 2: 统一 model 默认值**

将 `.env.example` 的 `MODEL_NAME` 从 `gpt-4.1-mini` 改为 `deepseek-v4-flash`（与 config.yaml 一致）

将 `docker-compose.yml` 的 `MODEL_NAME` 默认值从 `qwen-plus` 改为 `deepseek-v4-flash`

- [ ] **Step 3: 更新相关文档**

更新 `README.md` 中的默认模型说明

- [ ] **Step 4: Commit**

```bash
git add pageindex_mutil/config.yaml .env.example docker-compose.yml README.md
git commit -m "config: unify default model and search_backend across all config files"
```

---

### Task 6: 更新 MCP 工具文档

**Covers:** 文档完整性

**Files:**
- Modify: `docs/mcp-tools.md`

**Steps:**

- [ ] **Step 1: 验证工具数量**

确认 `mcp-tools.md` 中描述的工具与 `server.py` 中实际注册的工具一致

- [ ] **Step 2: 更新文档头部**

将 "The server exposes **8 tools**" 更新为实际数量

- [ ] **Step 3: Commit**

```bash
git add docs/mcp-tools.md
git commit -m "docs(mcp): ensure tool count matches actual implementation"
```

---

### Task 7: 运行完整测试套件验证

**Covers:** 质量保证

**Steps:**

- [ ] **Step 1: 运行所有测试**

```bash
uv run pytest tests/ -v --tb=short
```

Expected: 0 failures, 所有测试通过

- [ ] **Step 2: 检查测试覆盖率**

```bash
uv run pytest tests/ --tb=short -q
```

Expected: 所有测试通过，无 skipped（除非有明确的 skip 条件）

---

### Task 8: 最终验证和清理

**Covers:** 项目完整性

**Steps:**

- [ ] **Step 1: 验证项目结构**

```bash
# 检查所有 Python 文件语法
find . -name "*.py" -not -path "./.venv/*" -exec python -m py_compile {} \;
```

- [ ] **Step 2: 验证配置文件**

```bash
# 验证 YAML 语法
python -c "import yaml; yaml.safe_load(open('pageindex_mutil/config.yaml'))"
```

- [ ] **Step 3: 验证 Docker 构建**

```bash
docker build -t pageindex-uv-test .
```

- [ ] **Step 4: 最终 Commit**

```bash
git add -A
git commit -m "chore: final cleanup and verification"
```

---

## 优先级排序

| 优先级 | 任务 | 影响 |
|--------|------|------|
| P0 | Task 1: 修复测试失败 | 测试质量 |
| P0 | Task 7: 运行完整测试套件 | 质量保证 |
| P1 | Task 2: 修复 README 命令 | 用户体验 |
| P1 | Task 3: 更新架构文档 | 文档准确性 |
| P1 | Task 5: 统一配置默认值 | 配置一致性 |
| P2 | Task 4: 清理 .dockerignore | 部署配置 |
| P2 | Task 6: 更新 MCP 工具文档 | 文档完整性 |
| P3 | Task 8: 最终验证 | 项目完整性 |

---

## 执行建议

1. **先修复测试** (Task 1) - 确保代码质量基线
2. **再更新文档** (Task 2, 3, 6) - 保持文档与代码同步
3. **然后统一配置** (Task 5) - 减少用户困惑
4. **最后清理和验证** (Task 4, 7, 8) - 确保项目完整性

每个任务完成后运行测试验证，确保不引入新问题。
