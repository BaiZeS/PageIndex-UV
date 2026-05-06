# Feature: 批量上传文档 (Batch Upload)

**作者**: AI + User
**日期**: 2026-05-06
**状态**: Quick Draft

---

## 1. 目标

将现有的单文件上传入口扩展为批量上传，支持 HTTP `/upload` 端点和 CLI `/add` 命令两种方式，并发限流 3，失败跳过继续。

## 2. 需求

### 2.1 HTTP 批量上传

- `/upload` 端点支持一次接收多个文件（multipart form-data 中多个 `file` 字段，或 `files[]` 数组）
- 使用 `asyncio.Semaphore(3)` 限制并发索引数
- 每个文件独立处理，失败时记录错误信息，继续处理剩余文件
- 返回 JSON 包含每个文件的结果：`{filename, success, doc_id?, error?}`

### 2.2 CLI 批量上传

- `/add <path>` 支持：
  - 单个文件（现有行为，保持兼容）
  - Glob 通配符（如 `/add *.pdf`）
  - 目录路径（递归查找 `.pdf`、`.md`、`.markdown`）
- 使用 `asyncio.Semaphore(3)` 限制并发索引数
- 失败跳过继续，最后汇总成功/失败数量

### 2.3 并发与限流

- 最大并发数：3（避免 LLM API 限流）
- 索引耗时较长（PDF 解析 + LLM 调用），并发可显著缩短总时间

## 3. 变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `server.py` | 修改 | `upload_endpoint` 支持多文件 + 并发限流 |
| `main.py` | 修改 | `/add` 命令支持 glob/目录 + 并发限流 |

## 4. 验收标准

1. `curl -F "file=@a.pdf" -F "file=@b.pdf" http://localhost:3000/upload` 能同时索引两个 PDF
2. `> /add PageIndex/tests/pdfs/` 能递归索引目录下所有 PDF
3. `> /add "*.pdf"` 能索引当前目录下所有 PDF
4. 并发索引时最多 3 个同时进行
5. 任一文件失败时其余文件继续处理
