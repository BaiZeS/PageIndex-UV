# 实施任务清单

> 由 spec.md 生成
> 任务总数: 2
> 核心原则: 两个入口独立并行——HTTP 端点和 CLI 命令各自实现批量上传，互不依赖

## 依赖关系总览

Task 1 (server.py 批量上传端点)

Task 2 (main.py /add 批量命令)

（两个任务无依赖，可并行实现）

## 变更影响概览

### 文件变更清单

| 文件 | 操作 | 涉及任务 | 说明 |
|------|------|---------|------|
| `server.py` | 修改 | Task 1 | upload_endpoint 支持多文件 + 并发限流 |
| `main.py` | 修改 | Task 2 | /add 命令支持 glob/目录 + 并发限流 |

### 受影响接口

| 接口 | 变更类型 | 调用方 | 涉及任务 |
|------|---------|--------|---------|
| `upload_endpoint()` | 行为扩展 | HTTP POST /upload | Task 1 |
| `/add` REPL command | 行为扩展 | CLI 交互 | Task 2 |

## 风险与假设

| # | 描述 | 影响任务 | 假设/处理 |
|---|------|---------|----------|
| 1 | `client.index()` 是同步阻塞调用 | Task 1, 2 | 在 async 环境中用 `asyncio.to_thread()` 包装，避免阻塞事件循环 |
| 2 | glob 展开依赖 shell | Task 2 | 在 Python 内用 `glob.glob()` 展开，不依赖 shell |

## 任务列表

### 任务 1: [x] server.py 批量上传端点
- 文件: `server.py`（修改）
- 依赖: 无
- spec 映射: 2.1 HTTP 批量上传
- 说明: 修改 upload_endpoint 支持一次接收多个文件，使用 Semaphore(3) 限制并发，每个文件独立索引，失败跳过。返回 JSON 数组包含每个文件的结果。
- context:
  - `server.py:297` — `upload_endpoint()` 现有实现，单文件处理逻辑
  - `server.py:335` — `get_client().index()` 调用点，需包装为异步
  - `PageIndex/pageindex/client.py:80` — `index()` 方法签名
- 验收标准:
  - [ ] 上传 2 个文件时返回包含 2 个结果的 JSON 数组
  - [ ] 并发数不超过 3（可通过日志或延迟验证）
  - [ ] 一个文件索引失败时其余文件仍返回成功结果
- 子任务:
  - [x] 1.1: 修改 upload_endpoint 提取多个 file 字段
  - [x] 1.2: 添加 `_index_with_semaphore()` 辅助函数（Semaphore(3) + to_thread）
  - [x] 1.3: 收集所有结果并返回 JSON 数组

### 任务 2: [x] main.py /add 批量命令
- 文件: `main.py`（修改）
- 依赖: 无
- spec 映射: 2.2 CLI 批量上传
- 说明: 修改 /add 命令处理逻辑，支持单个文件、glob 通配符、目录路径三种输入。目录和 glob 展开为文件列表后，使用 Semaphore(3) 并发索引。
- context:
  - `main.py:677` — `/add` 命令现有处理逻辑，单文件路径解析
  - `main.py:698` — `index_pdf()` 调用点
  - `main.py:498` — `index_pdf()` 签名和实现
- 验收标准:
  - [ ] `/add <单个文件>` 保持现有行为兼容
  - [ ] `/add <目录>` 递归索引目录下所有 PDF/Markdown
  - [ ] `/add "*.pdf"` 索引匹配的所有文件
  - [ ] 并发数不超过 3
  - [ ] 失败文件打印错误，成功文件打印 doc_id，最后汇总数量
- 子任务:
  - [x] 2.1: 提取 `_resolve_upload_paths()` 函数（区分文件/glob/目录）
  - [x] 2.2: 添加 `_index_files_batch()` 函数（ThreadPoolExecutor(max_workers=3) + 结果收集）
  - [x] 2.3: 修改 /add 命令入口调用批量函数

## Spec 覆盖映射

| Spec 章节 | 任务 | 说明 |
|-----------|------|------|
| 2.1 HTTP 批量上传 | Task 1 | server.py upload_endpoint |
| 2.2 CLI 批量上传 | Task 2 | main.py /add 命令 |
| 2.3 并发与限流 | Task 1, 2 | 两者均使用 Semaphore(3) |
