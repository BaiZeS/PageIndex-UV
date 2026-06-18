# Spec — 项目级重构：消除 fork 嵌套 + 包重命名

> 状态：**§1-3 产出中**（REQUIREMENTS_CONFIRMED Hard STOP — 待用户确认）
> 模块：PageIndex / feature：project-refactor
> 关联：本 spec 仅作目录结构与命名重构，**不改动任何索引/RAG 算法行为**。

---

## §1 背景（Background）

### 1.1 来源
本项目 fork 自上游 [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex)（面向长文档的非向量化、推理式结构索引与 RAG）。集成时以子项目形式内联，保留了上游的目录外壳，形成多层嵌套。

### 1.2 当前结构（重构前）
```
PageIndex-UV/
├── server.py              # Starlette + MCP 服务入口（HTTP/SSE + 文件上传）
├── main.py                # 交互式 CLI 入口（多文档问答）
├── db.py                  # PageIndexDB（SQLite 存储层）
├── tests/test_db.py
└── PageIndex/             # ① fork 外层目录（无 __init__.py，非正式包）
    ├── run_pageindex.py   #    上游原始 CLI（与 main.py 功能重叠）
    ├── pageindex/         # ② 真正的 Python 包（二级嵌套）
    │   ├── __init__.py    #    暴露 PageIndexClient / page_index / md_to_tree / get_document...
    │   ├── config.yaml    #    ConfigLoader 通过 Path(__file__).parent 自定位
    │   ├── client.py / page_index.py / utils.py
    │   ├── super_tree.py / closet_index.py / page_index_md.py / retrieve.py
    │   └── agentic/       # ③ 子包（三级嵌套）
    │       └── router.py / planner.py / strategies.py / verifier.py
    └── tests/             # 散落测试（7 个）
```

### 1.3 问题（为什么要重构）
1. **三重嵌套**：`PageIndex/pageindex/agentic/` —— 典型 fork 拼接残留。
2. **导入不一致（根因 = sys.path hack）**：
   - `server.py:19-20`：`sys.path.insert` 同时插入「仓库根」与 `PageIndex/`，再用 `from PageIndex.pageindex import PageIndexClient`。
   - `main.py:16`：`sys.path.append(.../PageIndex)`，再用裸 `from pageindex.utils import ...`、`from pageindex.page_index import page_index_main`。
   - `run_pageindex.py`：裸 `from pageindex import *`。
   - 同一个包有三种引用方式，强依赖运行时 sys.path 顺序，脆弱且不可移植。
3. **入口重复**：`run_pageindex.py`（上游 CLI）与顶层 `main.py` 功能重叠。
4. **测试分散**：`PageIndex/tests/` 与顶层 `tests/` 两处，无统一 pytest。

### 1.4 公开 API 面（重构须保持可用的符号，仅路径/包名变化，行为不变）
`PageIndexClient`、`page_index_main`、`page_index`、`extract_json`、`create_clean_structure_for_description`、`write_node_id`、`count_tokens`、`md_to_tree`、`ConfigLoader`、`get_document`、`get_document_structure`、`get_page_content`、`SuperTreeIndex`、`ClosetIndex`、`AgenticRouter`、`RetrievalPlanner`、`CRAGVerifier`、`MetadataStrategy`/`SemanticsStrategy`/`DescriptionStrategy`。

---

## §2 目标（Goals）

| ID | 目标 | 验收信号 |
|----|------|----------|
| G1 | **消除 `PageIndex/` 外壳**，包提到顶层 | `PageIndex/` 目录消失，模块直挂在顶层包下 |
| G2 | **重命名 import 包**：`pageindex` → `pageindex_mutil`（与上游 `pageindex` 解耦） | `import pageindex_mutil` 可用 |
| G3 | **统一导入**：单一自包含风格 `from pageindex_mutil import ...` / `from pageindex_mutil.agentic.router import ...` | 全仓无 `from pageindex...` / `from PageIndex...` 残留 |
| G4 | **删除 sys.path hack** | `server.py`/`main.py` 不再 `sys.path.insert/append(...PageIndex)` |
| G5 | **合并测试**到顶层 `tests/`，统一 pytest | 仓库根 `pytest` 通过 |
| G6 | **删除冗余 CLI** `run_pageindex.py` | 文件移除，CLI 入口收敛到 `main.py` |
| G7 | **更新文档路径引用**（保留上游署名） | README / docs/design-docs / AGENTS.md 中路径与新结构一致 |
| G8 | **更新部署/打包配置** | Dockerfile / docker-compose.yml / pyproject.toml 路径与包名一致 |
| G9 | **行为零回归** | 重构前后现有测试全绿；server 可导入；CLI `--help` 正常 |

### 2.1 重构后目标结构
```
PageIndex-UV/
├── server.py / main.py / db.py
├── pageindex_mutil/                 ← 顶层包（原 PageIndex/pageindex/）
│   ├── __init__.py
│   ├── config.yaml                  ← 随包迁移，ConfigLoader 自定位
│   ├── client.py / page_index.py / utils.py
│   ├── super_tree.py / closet_index.py / page_index_md.py / retrieve.py
│   └── agentic/                     ← 保留（内聚子域：路由/规划/策略/校验）
│       ├── __init__.py
│       ├── router.py / planner.py / strategies.py / verifier.py
├── tests/                           ← 合并（含原 PageIndex/tests/）
└── (docs / deploy / pyproject.toml ...)
```

---

## §3 需求与范围（Requirements & Scope）

### 3.1 功能需求（FR）
- **FR1（包迁移）**：`PageIndex/pageindex/` 整体平移为顶层 `pageindex_mutil/`，含 `config.yaml` 与 `agentic/` 子包；删除 `PageIndex/` 外壳目录。
- **FR2（入口导入）**：`server.py`、`main.py`、`db.py` 中所有 `from pageindex...` / `from PageIndex.pageindex...` 改为 `from pageindex_mutil...`。
- **FR3（包内导入）**：确认包内相对导入（`.`, `..`）在迁移后仍正确；`__init__.py` 的 re-export 列表保持不变（符号集合不变）。
- **FR4（删 sys.path hack）**：移除 `server.py:19-20` 与 `main.py:16` 的 `sys.path` 操作。
- **FR5（删冗余入口）**：删除 `PageIndex/run_pageindex.py`。
- **FR6（合并测试）**：`PageIndex/tests/` → 顶层 `tests/`；测试内 import 改为 `from pageindex_mutil...` 或相对被测包；`tests/test_db.py` 保持。
- **FR7（打包配置）**：`pyproject.toml` 的 `[project] name` 与 `[tool.uv.index]` 保留/对齐；如需声明包发现，补 `[tool.setuptools.packages]` 或等价项（uv 模式）。
- **FR8（部署配置）**：核对 `Dockerfile`（`COPY . .` + `pip install -e "."`）、`docker-compose.yml`、`deploy/` 在新结构下仍可构建运行。
- **FR9（文档）**：更新 README、`docs/design-docs/**`、AGENTS.md 中对 `PageIndex/` 路径与 `pageindex` 包名的引用；**保留对 VectifyAI/PageIndex 上游的来源/署名说明**。

### 3.2 非功能需求（NFR）
- **NFR1（零行为变更）**：不改动任何索引/RAG/校验算法逻辑，不动 DB schema，不动 config.yaml 配置项。
- **NFR2（导入自包含）**：无 `sys.path` hack、无对仓库外路径的隐式依赖。
- **NFR3（署名保留）**：文档保留上游归属（合规与致谢）。
- **NFR4（Python ≥3.12 / uv 工具链）**：依赖集合不变。

### 3.3 约束
- Python ≥3.12，uv 包管理（`uv.lock`、`[tool.uv.index]`）。
- 依赖集合（openai/pymupdf/pypdf2/python-dotenv/pyyaml/tiktoken/jieba/mcp/python-multipart）不变。
- 公开 API 符号集合不变（仅 import 路径/包名变化）。

### 3.4 范围外（Out of Scope）
- 任何算法/业务逻辑变更（索引、树构建、检索、校验策略）。
- 新增功能、新增 API。
- 数据库 schema 变更。
- 重命名 `PageIndexClient` 等对外类/函数符号（仅改 import 路径，不改类名）。

### 3.5 假设（ASSUMPTIONS — 需用户确认，见 Hard STOP）
- **A1（命名拼写）**：包名 **`pageindex_mutil`** 为用户指定值，按原样采用。⚠️ 若本意为 `pageindex_multi`（多文档语义），请在确认时纠正——该名会贯穿所有 import、pyproject 与入口，纠正成本在确认前最低。
- **A2（config.yaml 迁移）**：`config.yaml` 随包迁移到 `pageindex_mutil/config.yaml`；`ConfigLoader` 通过 `Path(__file__).parent / "config.yaml"` 自定位，迁移后无需改代码。（已核实：当前位于 `PageIndex/pageindex/config.yaml`，加载逻辑为相对路径。）
- **A3（包内相对导入）**：所有包内 cross-import 均为相对导入（`.`, `..`），目录平移不影响其正确性。（已核实 grep。）
- **A4（pyproject 包发现）**：当前未显式声明 `[tool.setuptools.packages]`，`pip install -e "."` 依赖自动发现；重命名后需确认安装后 `import pageindex_mutil` 可用，否则补充包发现声明。
- **A5（已删除的顶层 test_smoke.py）**：git 状态显示 `D test_smoke.py`（已删除），本重构不恢复它。

---

> ✅ **REQUIREMENTS_CONFIRMED（已确认）** — 用户确认：包名 `pageindex_mutil`（按原样）、目标结构（§2.1）、范围（§3 FR1-FR9）均 OK。进入设计阶段。

---

## §4 系统设计 — 替代方案与权衡（Alternatives & Trade-offs）

### 4.0 设计输入（已核实）
- 包内 cross-import 均为相对导入 → 目录平移零代码改动（仅入口/测试/打包需改绝对引用）。
- 测试导入风格**不统一**，是迁移的隐藏工作面：
  - `tests/test_db.py`：`from db import PageIndexDB`
  - `PageIndex/tests/test_planner_unit.py`：`from PageIndex.pageindex.agentic.planner import ...`
  - `PageIndex/tests/test_strategies_unit.py`：`from PageIndex.pageindex.agentic.strategies import ...`
  - `PageIndex/tests/test_verifier_unit.py`：`from PageIndex.pageindex.agentic.verifier import ...`
  - `PageIndex/tests/test_router.py`：**最复杂** —— `importlib.util` + `sys.path` 注入模拟 `pageindex.utils`/`closet_index` 以规避 PyPDF2 重依赖（`spec_from_file_location("pageindex.utils", ...)`）。迁移后这些 `sys.modules["pageindex.utils"]` 与路径须同步改为 `pageindex_mutil.*`。
- 打包：pyproject 无显式 `[tool.setuptools.packages]`，依赖 flat-layout 自动发现。

### 4.1 替代方案

#### 方案 A — 一次性大爆炸（Big-bang）
单次提交完成全部：`git mv` 整包改名 → 全局重写入口/测试/打包 import → 合并测试 → 改文档 → 单次 `pytest` 验证。
- ✅ 优点：diff 原子、一次评审、"无引用"目标一步到位。
- ❌ 缺点：超大 diff 不可二分（bisect）；若 `test_router.py` 的 importlib 模拟隐式断裂，定位困难；回滚须整体 revert。

#### 方案 B — 按关注点分阶段（Phased by concern）⭐ 推荐
拆成独立可验证阶段，每阶段单独提交且各自保持 GREEN：
1. **P1 结构平移**：`git mv PageIndex/pageindex pageindex_mutil`（保留 git 历史）；删空壳 `PageIndex/`；`config.yaml` 随包到达 `pageindex_mutil/config.yaml`。
2. **P2 入口与 sys.path 清理**：改 `server.py`/`main.py` import 为 `from pageindex_mutil...`；移除 `sys.path` hack。
3. **P3 测试合并与修复**：`git mv PageIndex/tests/* tests/`；统一测试 import；**重点重写 `test_router.py`** 的 importlib 模拟（`pageindex.utils` → `pageindex_mutil.utils` 等）。
4. **P4 打包/部署**：pyproject 包发现（flat 自动 / 显式声明）、Dockerfile/docker-compose 核对。
5. **P5 文档**：README / design-docs / AGENTS.md 路径与包名引用（保留上游署名）。
- ✅ 优点：每阶段可独立 `pytest` 验证 + 二分；风险局部化；符合 devkit TDD/verification-per-stage 铁律；回滚粒度细。
- ❌ 缺点：提交更多次；P1 后到 P2 前存在短暂"包已改名但入口未跟"的非 GREEN 中间态（可把 P1+P2 合并为一个原子提交消除）。

#### 方案 C — 兼容垫片（Compatibility shim）❌ 不推荐
保留薄 `pageindex/__init__.py` re-export 自 `pageindex_mutil`，过渡期双名可用。
- ❌ 与"不要引用了"目标直接冲突；徒增需后续删除的残留；纯内部项目无外部消费者，无收益。

### 4.2 推荐：方案 B（分阶段）
行为保持型重构的本质是"安全网驱动"：每阶段 GREEN 才能确认零回归。`test_router.py` 的 importlib 模拟是最易断裂点，分阶段使其在 P3 独立验证。P1+P2 可合并为单个原子提交（消除"非 GREEN 中间态"缺点），保留 P3/P4/P5 的细粒度。

### 4.3 设计决策（DESIGN_DECISION — 已确认）
✅ **用户选择方案 B（分阶段）**。P1+P2 合并为一个原子提交（消除中间非 GREEN 态），P3/P4/P5 各自独立提交。详见 §5。

---

## §5 迁移步骤（选定方案 B）

### P1+P2（原子提交：结构平移 + 入口/路径清理）
1. `git mv PageIndex/pageindex pageindex_mutil` —— 整包平移（含 `config.yaml`、`agentic/`、所有模块）。git 历史保留。
2. 改 `server.py`：`from PageIndex.pageindex import PageIndexClient` → `from pageindex_mutil import PageIndexClient`；**删除** `sys.path.insert(0, ...)` 两行（L19-20）。
3. 改 `main.py`：`from pageindex.utils import ...` → `from pageindex_mutil.utils import ...`；`from pageindex.page_index import page_index_main` → `from pageindex_mutil.page_index import page_index_main`；**删除** `sys.path.append(.../PageIndex)`（L16）。
4. 验证：`python -c "from pageindex_mutil import PageIndexClient"` 成功；`grep -rn "from PageIndex\|from pageindex\b\|sys.path" server.py main.py` 仅剩合法项。
> 此时 `PageIndex/` 仅剩 `run_pageindex.py` + `tests/`（待 P3 清理）。提交。

### P3（独立提交：测试合并 + test_router 重写 + 删冗余 CLI）
1. `git mv PageIndex/tests/test_planner_unit.py PageIndex/tests/test_strategies_unit.py PageIndex/tests/test_verifier_unit.py PageIndex/tests/test_router.py PageIndex/tests/test_super_tree.py PageIndex/tests/test_client_integration.py PageIndex/tests/test_docs_info.py tests/`。
2. 改测试 import：
   - `from PageIndex.pageindex.agentic.planner import ...` → `from pageindex_mutil.agentic.planner import ...`
   - `from PageIndex.pageindex.agentic.strategies import ...` → `from pageindex_mutil.agentic.strategies import ...`
   - `from PageIndex.pageindex.agentic.verifier import ...` → `from pageindex_mutil.agentic.verifier import ...`
3. **重写 `tests/test_router.py`**：`pageindex_path = .../"PageIndex"/"pageindex"` → `.../"pageindex_mutil"`；`spec_from_file_location("pageindex.utils", ...)` → `("pageindex_mutil.utils", ...)`；`sys.modules["pageindex.utils"]` → `sys.modules["pageindex_mutil.utils"]`；同改 `pageindex.closet_index` 等所有 `sys.modules["pageindex.*"]` 键与 spec name。删除上层 `sys.path.insert` 注入（改为依赖已安装/已平移的顶层包）。
4. 删除 `PageIndex/run_pageindex.py`；`rmdir PageIndex/tests PageIndex`（空目录清理）。
5. 验证：`pytest -q` 通过（含原 7 测试 + test_db）。

### P4（独立提交：打包/部署）
1. pyproject：补 `[build-system]`（setuptools/wheel）与 `[tool.setuptools.packages.find]` 显式 `include = ["pageindex_mutil*"]`（防止 flat 自动发现误收 tests/deploy/docs）。`[project] name = "pageindex-uv"`、`[tool.uv.index]` 保留。
2. Dockerfile：`COPY . .` + `pip install -e "."` 在新结构下成立；核对无 `PageIndex/` 硬编码路径。
3. docker-compose.yml / deploy/：核对工作目录与路径。
4. 验证：`uv pip install -e .`（或 `pip install -e .`）后 `python -c "import pageindex_mutil"` 在干净环境可用。

### P5（独立提交：文档）
1. README.md：结构图、命令、import 示例改为 `pageindex_mutil`；**保留** VectifyAI/PageIndex 上游署名。
2. docs/design-docs/**：移除/修正对 `PageIndex/` 路径的引用（保留历史设计文档语义，仅更新路径）。
3. AGENTS.md / CLAUDE.md：路径约定若提及 `PageIndex` 同步更新。
4. 验证：`grep -rn "PageIndex/pageindex\|from pageindex\b" docs/ *.md` 无残留旧引用。

## §6 打包设计（Package Discovery）
- **问题**：当前 pyproject 无 `[build-system]`、无 `[tool.setuptools.packages]` —— 这正是 `server.py`/`main.py` 用 sys.path hack 的根因（包未被"安装即用"）。
- **方案**：补显式声明，使 `import pageindex_mutil` 在安装后开箱可用：
  ```toml
  [build-system]
  requires = ["setuptools>=68", "wheel"]
  build-backend = "setuptools.build_meta"

  [tool.setuptools.packages.find]
  include = ["pageindex_mutil*"]
  ```
- `config.yaml` 作为 package data：补 `[tool.setuptools.package-data] pageindex_mutil = ["config.yaml"]`（保证 wheel 打入；editable 模式下 __file__ 自定位也成立）。

## §7 风险与回滚（Risk & Rollback）
| 风险 | 缓解 | 回滚 |
|------|------|------|
| `test_router.py` importlib 模拟断裂 | P3 专项重写 + 独立运行该测试 | `git revert` P3 提交 |
| `pageindex_mutil` 拼写不一致传播 | 全局 grep 守卫（§8）；用户已确认拼写 | 改名脚本批量替换 |
| 打包自动发现误收 stray 目录 | §6 显式 `include` 白名单 | 还原 pyproject |
| `config.yaml` 找不到 | ConfigLoader 用 `Path(__file__).parent` 自定位（已核实） | 确认 package-data 声明 |
| 行为回归（算法/DB） | 不动算法/逻辑/DB schema；逐阶段 GREEN | 按阶段 revert |
- **回滚策略**：每阶段独立提交 → `git log --oneline` 定位 → `git revert <sha>`。无需整体回滚。

## §8 验证策略（Verification）
### 8.1 基线（code 阶段最先执行）
重构前先跑 `pytest -q` 捕获**当前 GREEN 基线**（安全网）。若现状有失败用例，记录并告知用户——重构只保证"不引入新失败"。
### 8.2 每阶段验证门
- P1+P2：`python -c "from pageindex_mutil import PageIndexClient, page_index_main, md_to_tree"`；入口 grep 守卫。
- P3：`pytest -q tests/test_router.py` 单独绿；`pytest -q` 全绿。
- P4：干净 env `pip install -e .` + `python -c "import pageindex_mutil"`。
- P5：docs grep 守卫。
### 8.3 最终守卫
```bash
# 无旧引用残留
grep -rnE "from PageIndex|from pageindex\b|sys\.path\.(insert|append).*PageIndex" --include=*.py . | grep -v .venv
# 期望：空
python -c "import pageindex_mutil, server" 2>&1 | tail   # 入口可导入
python main.py --help                                      # CLI 入口
pytest -q                                                  # 全绿
```
### 8.4 过程纪律（devkit 铁律）
- **Two-Agent Minimum**：code 阶段由子代理（leaf executor）执行文件移动/编辑，由**不同**子代理（independent verifier）独立验证——禁止自审自评。
- **TDD/L2**：行为保持型重构，先建立 GREEN 基线（8.1），每步改动后确认仍 GREEN。
- 每个 phase 产出 handoff.md（NEEDS_INDEPENDENT_VERIFICATION）→ verifier 独立跑 §8 守卫 → quality-gate。

---

## §9 P6 — LLM 配置统一 + 标准 OpenAI 默认（功能变更，独立阶段/提交）

> 注：P0–P5 为行为保持型重构（NFR1 零行为变更）。**P6 是功能变更**，故作为独立阶段在重构稳定后执行，以免污染重构安全网。用户已确认目标="统一配置+标准 OpenAI 默认"、时机="重构后 P6 独立阶段"。

### 9.1 现状（已核实）
- `utils.py:24-28`：`_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")`；`_BASE_URL` 默认 `https://dashscope.aliyuncs.com/compatible-mode/v1`；**模块级**实例化 `_client = OpenAI(...)` / `_async_client`（导入即建、不可重配）。
- `main.py:24-40`：复制粘贴 `API_KEY`/`BASE_URL`/`MODEL_NAME` + `client = OpenAI(...)`（第二处真相）。
- `client.py:46-50`：第三种变体，含 `CHATGPT_API_KEY` 回退逻辑。
- `config.yaml`：`model: qwen-plus` / `retrieve_model: qwen-plus`。
- 调用本身已是 openai SDK 标准格式（`_client.chat.completions.create(model, messages, temperature)`）—— 故"标准 OpenAI 格式"的本质是**配置/默认值/可重配**，不是调用协议。

### 9.2 目标
| ID | 目标 |
|----|------|
| L1 | **单一配置源**：`utils.py` 作为 LLM 配置唯一真相（其余文件引用它，删除复制） |
| L2 | **标准 OpenAI 默认**：`base_url` 默认 `https://api.openai.com/v1`，去 DashScope 硬编码默认 |
| L3 | **API key 优先级**：`OPENAI_API_KEY` 为主；`DASHSCOPE_API_KEY` 作遗留回退（兼容现有部署，可后续移除） |
| L4 | **可重配/可注入**：新增 `configure_llm(api_key=None, base_url=None)` 重建 client（支持运行时切换/测试注入），取代"导入即建不可变" |
| L5 | **接口不变**：`llm_completion`/`llm_acompletion` 签名与 openai SDK 调用格式保持不变（避免破坏 27 处调用点） |

### 9.3 假设（P6 执行前 flag for review）
- **A6**：`base_url` 默认标准 OpenAI；`OPENAI_BASE_URL` env 可覆盖。
- **A7**：`OPENAI_API_KEY` 主 + `DASHSCOPE_API_KEY` 回退（保留以兼容现有 DashScope 部署，去留待用户在 P6 review 时定）。
- **A8**：`config.yaml` 的 `model` 默认值暂保持 `qwen-plus`（完全 env/config 可覆盖）；⚠️ 当 base_url 指向标准 OpenAI 时，须设 `MODEL_NAME` 为 OpenAI 模型——此不一致在 P6 review 与文档中提示。
- **A9**：新增 `configure_llm()` 重初始化函数；保留模块级懒初始化以兼容现有 import 时机。

### 9.4 验证（P6）
- 烟测：`python -c "from pageindex_mutil.utils import configure_llm, llm_completion; configure_llm('sk-test', 'https://api.openai.com/v1')"` 不报错。
- 配置单一性守卫：`grep -rnE "OpenAI\(|DASHSCOPE_API_KEY|OPENAI_BASE_URL" --include=*.py . | grep -v .venv` 仅命中 `utils.py`（main.py/client.py 不再自建 client）。
- 回归：`pytest -q` 全绿（仅默认配置值变化，行为对齐）。
- Two-Agent：P6 由子代理实现 → 不同 verifier 独立跑上述守卫。

---

## 设计阶段完成检查
- [x] §4 ≥2 替代方案 + 权衡（A/B/C），用户选 B
- [x] §5 选定方案迁移步骤（P1+P2 / P3 / P4 / P5）
- [x] §6 打包设计（含 build-system 修复 sys.path hack 根因）
- [x] §7 风险与回滚
- [x] §8 验证策略 + Two-Agent/TDD 纪律
- [x] §9 P6 LLM 配置统一（功能变更，独立阶段）—— 已澄清目标/时机
> 设计完成。下一步：plan 阶段（task-planning → tasks.md），含 P0–P5（重构）与 P6（LLM 优化）两轨。
