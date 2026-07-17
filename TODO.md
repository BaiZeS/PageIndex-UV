# TODO / 项目进度

## 已完成 ✅

- [x] **重构 `main.py`：单文档 + 多文档问答入口**
  - 引入 `PageIndexDB` SQLite 缓存层，首次索引后跳过 PDF 重复解析
  - 统一 `_call_llm_json` 复用 LLM JSON 调用逻辑
  - 提取 `pages_from_nodes`、`_flatten_nodes`、`_extract_page_records` 等共享辅助函数
  - 预缓存 `tree_json_str`，避免多文档模式下重复序列化
  - 添加模式常量（`MODE_SINGLE`/`MODE_MULTI`、`REASONING_TREE`/`REASONING_TOC`）
  - 修复 `ensure_node_ids` 递归检查、`_fallback_to_toc` 参数精简、日志文件打开异常处理

- [x] **重构 `db.py`：SQLite 持久化层**
  - `insert_nodes` / `insert_pages` 改为接收调用方准备好的扁平记录列表，不再在持久化层处理树遍历或 PDF 打开
  - `insert_document` 使用 `INSERT ... ON CONFLICT ... RETURNING id` 单语句获取 ID
  - 消除读查询的隐式 `COMMIT`
  - 魔法数字替换为 `SQLITE_MAX_VARIABLE_NUMBER = 999`

- [x] **移除 `litellm`，改用原生 OpenAI SDK + `tiktoken`**
  - `utils.py` 的 `llm_completion` / `llm_acompletion` 改为 `openai.OpenAI` / `AsyncOpenAI`
  - `count_tokens` / `get_page_tokens` 改用 `tiktoken`
  - `client.py` 移除 `litellm/` 前缀注入
  - 模型名直接写服务商侧名称（如 `qwen-plus`），无需 provider 前缀

- [x] **删除废弃文件**
  - `main_tree.py.deprecated` 已删除（其多文档逻辑已内联到 `main.py`）

- [x] **更新 `README.md` 与 `.env`**
  - 反映内联 PageIndex 代码（非子模块）
  - `.env` 示例注明模型名直接使用服务商名称，无需前缀

## 待评估 / 开放项

- [ ] **PDF 目录树构建方式优化：当前纯 LLM 端到端提取 vs 2025 最佳实践差距**
  - **当前实现问题**：完全依赖 LLM 逐页检测目录 → 提取文本 → 转 JSON → 匹配页码 → 验证修复
    - 丢失 PDF 内置排版信息（字体、位置、bold/italic 标志）
    - 每页/每组都调用 LLM，成本高、延迟大
    - 对无目录 PDF 的层级推断纯靠 LLM 猜测，缺乏结构化信号
  - **最佳实践（HiPS 2025 / EMNLP 2024 / 混合架构）**：多模态混合流水线
    1. **优先提取 PDF 内置 Outline** (`doc.get_toc()`)：返回 `[[lvl, title, page], ...]`，是最高置信度种子。许多学术/技术 PDF 已内嵌完整书签树，可直接使用
    2. **字体特征生成标题候选**：PyMuPDF `span["size"]` + `span["font"]`（检测 Bold/Italic）+ `span["flags"]` + 行长度/位置。按字号排序分配层级，生成候选结构
    3. **LLM 角色从"生成"转为"精修"**：
       - 规则层生成高召回候选（内置 outline + 字体特征 + 编号模式如 1.1、1.2）
       - LLM 只做：歧义消解（同级标题 vs 正文）、缺失层级补全、验证修正
       - 参考 AssertCoder 架构：LayoutParser → LLM Semantic Classification
       - 参考 ESG Span Extraction：NER+Rules → Two-Stage LLM Refinement
    4. **保留现有验证循环**：`verify_toc` + `fix_incorrect_toc_with_retries` 是高质量保障机制，保留并复用到新流程
  - **具体可借鉴的工具/实现**：
    - **pdf-tocgen** (Krasjet)：PyMuPDF 基础上三件套 `pdfxmeta`(提取字体元数据) → `pdftocgen`(按 recipe 生成目录) → `pdftocio`(注入 PDF)。Recipe 示例按 `font.name` + `font.size` 定义 heading level
    - **PyMuPDF4LLM** (Artifex 官方)：PDF→Markdown 自动 heading 检测，基于字号层级推断 H1-H4
    - **docling-hierarchical-pdf**：先提取 PDF 内置 outline，若无则 fallback 到编号模式 + 字体大小推断层级
    - **pdfstruct**：font-aware heading detection，直接输出 H1-H6 树
  - **开源工具选型参考**：
    | 工具 | 优势 | 劣势 | 适用场景 |
    |------|------|------|----------|
    | PyMuPDF (当前在用) | 快、可控、纯 Python | 不自动推断层级 | 底层解析 |
    | pdf-tocgen | 字体元数据→目录，成熟 CLI | 需调 recipe | 批量处理已知格式 PDF |
    | Docling | IBM 开源，多模态 | 层级提取 flatten 严重 | 通用文档转换 |
    | GROBID | 学术 PDF 专用，CRF+DLM | 部署重(Java) | 学术论文 |
    | Marker | 高性能 Transformer 转 Markdown | 层级有时丢失 | 快速转换 |
    | Mineru | 商汤开源，多模态 | 新工具，稳定性待验证 | 复杂布局 |
  - **评估指标**：采用 TEDS (Tree Edit Distance Similarity) 评估目录树准确性，而非仅平面准确率
  - **参考**：
    - [HiPS arXiv](https://arxiv.org/html/2509.00909v1)
    - [EMNLP 2024 PDF-to-Tree](https://aclanthology.org/2024.findings-emnlp.628.pdf)
    - [Detect-Order-Construct](https://arxiv.org/html/2401.11874v2)
    - [pdf-tocgen GitHub](https://github.com/Krasjet/pdf.tocgen)
    - [PyMuPDF4LLM Blog](https://artifex.com/blog/introducing-pymupdf4llm-a-breakthrough-in-pdf-to-markdown-conversion-for-python-developers)
    - [PDF Data Extraction Benchmark 2025](https://procycons.com/en/blogs/pdf-data-extraction-benchmark/)

- [ ] **异构文档格式支持：当前仅支持 PDF，实际工况下需处理 DOC/DOCX/XLSX/MD 等多格式**
  - **当前限制**：`main.py` 仅通过 `fitz` (PyMuPDF) 处理 PDF，`PageIndexDB` 的 schema 也是围绕 PDF 的 page/node 模型设计的
  - **实际工况挑战**：
    - 格式混杂：doc/docx/xlsx/md/html 与 PDF 并存，甚至同一文档内嵌多种格式（如 PPTX 内嵌 Excel 图表）
    - 质量参差不齐：扫描件/图片 PDF、损坏文件、加密文档、手写批注、低分辨率传真件
    - 统一索引困难：不同格式的"页"概念不一致（Word 的节、Excel 的 sheet、Markdown 的 heading）
  - **业界最佳实践（2025-2026）**：统一文档智能层（Unified Document Intelligence Layer）
    - **核心架构**：格式感知路由 → 统一中间表示(Canonical Document Model) → 结构化提取 → 标准化索引
    - **路由模式**：根据文件类型和文档质量选择处理路径，而非单一工具
      ```
      输入 → 格式检测 → 质量评估 → 路由决策
        DOCX/PPTX → 原生解析 (python-docx / python-pptx) → 保留 heading 层级
        XLSX      → 结构化提取 (pandas/openpyxl) → sheet→table→row 层级
        MD        → frontmatter 解析 → heading 层级直接可用
        原生 PDF   → PyMuPDF 直接提取文本
        扫描 PDF   → OCR 路由 (PaddleOCR / Tesseract / VLM)
        混合质量   → 双模提取：文字区直接解析 + 图片区 OCR
      ```
    - **统一中间表示**：所有格式归一化为 `DoclingDocument` 或自研 Canonical Model
      ```json
      {
        "source": "s3://bucket/contract.docx",
        "format": "docx",
        "title": "...",
        "sections": [
          {"heading": "...", "level": 1, "content": "...", "tables": [...], "page/sheet": "..."}
        ],
        "metadata": {"author": "...", "created": "...", "ingestion_ts": "..."}
      }
      ```
  - **开源工具选型对比**：
    | 工具 | 覆盖格式 | 层级保留 | OCR | 部署复杂度 | 适用场景 |
    |------|----------|----------|-----|-----------|----------|
    | **Docling** (IBM/Linux基金会) | PDF/DOCX/PPTX/XLSX/MD/图片/LaTeX | 中（DocLayNet布局分析） | 内置VLM | 中等 | 首选统一方案，生态成熟，LangChain/LlamaIndex原生集成 |
    | **MMORE** (2025) | 文档/演示/表格/多媒体 | 高（>Docling 40%扫描PDF准确率） | 内置 | 中等 | 需要分布式处理（多节点K8s）、大规模扫描文档 |
    | **Unstructured.io** | PDF/DOCX/XLSX/HTML/图片/邮件 | 中 | 分区策略(auto/hi_res/ocr_only) | 高（服务化部署） | 复杂布局、需要大量预处理钩子 |
    | **Marker** | PDF | 低（转Markdown） | 无 | 低 | 快速PDF→Markdown，但层级易丢失 |
    | **python-docx + openpyxl + pymupdf** | 各格式独立 | 高（手动控制） | 无 | 最低 | 轻量级、可控、需自行实现统一层 |
  - **推荐策略**：
    1. **轻量起步**：用 `python-docx` / `openpyxl` / `markdown` 库分别解析，自行构建 Canonical Model，与现有 PageIndex 树结构对接
    2. **中期演进**：引入 **Docling** 作为统一解析层，将各种格式转为 Markdown/JSON 后接入现有 `page_index_md.py` 的 MD→Tree 流程
    3. **大规模生产**：若文档量>10K页/天或扫描件占比高，评估 **MMORE** 分布式模式或 **Unstructured** 服务化部署
  - **质量评估与降级机制**：
    - 文本提取后若字符数<阈值（如50字符），自动路由到 OCR/VLM
    - 文件损坏/加密 → 记录错误日志，不入索引，避免阻塞整个批次
    - 低置信度提取 → 标记待人工审核（HITL），而非直接丢弃
  - **增量索引**：基于文件 hash/mtime 判断是否需要重新解析，避免未变更文件重复处理
  - **参考**：
    - [Docling GitHub](https://github.com/docling-project/docling)
    - [MMORE arXiv](https://arxiv.org/html/2509.11937v1)
    - [Document Parsing Techniques Survey](https://arxiv.org/abs/2410.21169)
    - [Essential Document Parsing Tools 2025](https://ai.plainenglish.io/essential-document-parsing-tools-for-2025-786f55db86e3)
    - [MODE: Mixture of Document Experts](https://arxiv.org/html/2509.00100v1)

- [ ] **L1 文档筛选精度**：当前 L1 仅用 LLM 做轻量筛选（Top-3），文档数量多时精度可能下降。若未来文档超过 10 个，考虑引入向量/关键词预召回。
- [ ] **Token 预算调优**：`MAX_CONTEXT_TOKENS = 16000` 当前为硬编码，未来可按模型动态调整（如 GPT-4 可放宽到 32K）。
- [ ] **重新索引优化**：`db.py` 每次索引无条件 `DELETE + INSERT`，若 PDF 未变更则为冗余写入。未来可基于文件 mtime 或 hash 做增量判断。
- [ ] **`nodes` / `pages` 表索引**：超大规模文档下，可考虑在 `(doc_id, start_index)`、`(doc_id, page_number)` 上加索引以进一步加速查询。
