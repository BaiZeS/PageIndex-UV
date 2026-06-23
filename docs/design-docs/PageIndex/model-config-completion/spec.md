# W6 · 模型配置统一收尾 — spec.md

> 阶段：clarify（§1-3 待确认）。仅含需求，不含技术设计。
> 来源：`architecture-review-2026-06/review-report.md` P1-12 / P2-10 / P2-16 / P2-17。
> 当前分支：`feat/unify-model-name-env`（主目标已达成，本工作收尾残留）。
> 用户确认范围：retrieve_model 接入检索路径 + OpenAI 默认对齐 + 魔法数字移入 config.yaml。

## §1 背景

`feat/unify-model-name-env` 分支已在 `utils.py`/`ConfigLoader` 路径上统一了 `MODEL_NAME`（主目标达成，`tests/test_config.py` 14 测试覆盖优先级/空白/空串边界）。但残留四处不一致，使分支契约未闭合：

1. **硬编码模型名（P1-12）**：`pageindex_mutil/page_index_md.py:312` 在 `if __name__ == "__main__":` 块硬编码 `MODEL="gpt-4.1"`，绕过 `MODEL_NAME`/`ConfigLoader`。与分支目标直接矛盾。
2. **`retrieve_model` 死配置（P2-10）**：`RETRIEVE_MODEL_NAME` env + `config.yaml:retrieve_model` + `client.py:60` 被加载存储，但**从不被读取使用**（所有 LLM 调用用 `self.model`）；`_normalize_retrieve_model`（`client.py:32-36`）是恒等函数。声明的接口存在但未接线。
3. **`.env.example` 默认矛盾（P2-16）**：`.env.example` shipped 默认 DashScope 活跃（`DASHSCOPE_API_KEY` 取消注释），而 P6 提交说明称"默认为标准 OpenAI 端点"——新无 key 用户得到 OpenAI 端点 + qwen-plus 模型的不匹配。
4. **魔法数字散落（P2-17）**：`page_index_md.py:313-316`（`THINNING_THRESHOLD=5000` 等）不在 `config.yaml`；`main.py:70-72` 重新硬编码 `config.yaml` 已有的默认值（`toc_check_page_num=20` 等），使改 `config.yaml` 对 `main.py` 无效。

## §2 目标

- **零硬编码模型名**：移除 `MODEL="gpt-4.1"`，统一走 `ConfigLoader`。
- **`retrieve_model` 接线**（用户选择的新能力）：让检索路径的 LLM 调用实际使用 `retrieve_model`（设置时），未设置时回退 `model`——使该配置真正生效。
- **`.env.example` 对齐标准 OpenAI 默认**：与 P6 "标准 OpenAI 端点" 目标一致。
- **魔法数字可配**：移入 `config.yaml`，经 `ConfigLoader` 解析。
- **`main.py` 用 `ConfigLoader` 填默认**：消除重复硬编码。
- 保持 `MODEL_NAME` 解析链不变（14 测试保持 GREEN）。

## §3 需求与范围

### 功能需求
- **FR1**：移除 `page_index_md.py:312` 硬编码 `MODEL="gpt-4.1"`，改用 `ConfigLoader` 解析（与 `main.py:49` 一致）。
- **FR2**：`retrieve_model` 接入检索路径——检索 LLM 调用（`closet_index`/`super_tree` 等）使用 `retrieve_model`（设置时），回退 `model`。
- **FR3**：`.env.example` shipped 默认对齐标准 OpenAI（`OPENAI_API_KEY` 取消注释、`DASHSCOPE_API_KEY` 注释），与 P6 一致。
- **FR4**：魔法数字（`THINNING_THRESHOLD`、`SUMMARY_TOKEN_THRESHOLD`、`IF_THINNING`、`IF_SUMMARY`，及 `main.py:70-72` 的 `toc_check_page_num`/`max_page_num_each_node`/`max_token_num_each_node`）移入 `config.yaml`，经 `ConfigLoader` 解析。
- **FR5**：`main.py:70-72` 改用 `ConfigLoader().load()` 填默认，不再重复硬编码 `config.yaml` 值。

### 非功能需求
- **NFR1**：CI grep 守卫 `! grep -rn 'MODEL\s*=\s*"gpt-' pageindex_mutil/` 通过。
- **NFR2**：`tests/test_config.py`（14 测试）保持 GREEN；新增 `retrieve_model` 接线测试 + 魔法数字配置解析测试。
- **NFR3**：不回归 `MODEL_NAME` 解析链（优先级：caller 显式 > env > yaml）。
- **NFR4**：`retrieve_model` 未设置时行为与现状一致（回退 `model`）——向后兼容。
- **NFR5**：`config.yaml` 新增键以当前硬编码值为默认，保证现有部署行为不变。

### 范围
- **IN**：`page_index_md.py` `__main__`；`retrieve_model` 接线（`closet_index`/`super_tree` 检索 LLM 调用点）；`.env.example`；`config.yaml` + `ConfigLoader` 魔法数字；`main.py:70-72`；CI grep 守卫；相关测试。
- **OUT**：
  - 拆分 `utils.py`（W4d）。
  - 改变 `MODEL_NAME` 优先级链本身（保持不变）。
  - 其他配置重构。

### 假设（ASSUMPTION，需确认）
- `retrieve_model` 未设置时回退 `model`（向后兼容）。
- `ConfigLoader.load()` 扩展暴露新键，以当前硬编码值为默认。
- `.env.example` 对齐 OpenAI 后，`config.yaml` 的 `model:` 默认值是否需相应调整（DashScope qwen-plus vs OpenAI 模型）在设计中确认。

---

> 🚨 **STOP — REQUIREMENTS_CONFIRMED**
> 待用户确认本 spec §1-3 后，由独立 agent 调用 quality-gate（threshold ≥70），通过后进入 system-design 阶段。

---

## §4 方案设计（System Design — Alternatives & Trade-offs）

### 4.0 设计输入（已核实，codegraph + Read）

#### 4.0.1 `MODEL_NAME` 解析链现状（保持不变）
- `ConfigLoader.load()`（`pageindex_mutil/utils.py:744`）已是单一解析入口，优先级：caller 显式 kwarg > `MODEL_NAME`/`RETRIEVE_MODEL_NAME` env > `config.yaml` 默认。
- `main.py:49` `MODEL_NAME = ConfigLoader().load(None).model` —— 已统一走 `ConfigLoader`。
- `client.py:58-60` `PageIndexClient.__init__` —— 已走 `ConfigLoader().load(overrides)` 并 `_normalize_retrieve_model(opt.retrieve_model or self.model)`。
- `tests/test_config.py` 14 测试覆盖优先级 / 空白 / 空串边界，保持 GREEN 是硬约束。

#### 4.0.2 `retrieve_model` 死配置现状
- 加载链已存在：`config.yaml:retrieve_model` → `RETRIEVE_MODEL_NAME` env → `ConfigLoader.load()` → `PageIndexClient.retrieve_model`（`client.py:60`）。
- **读取链完全缺失**：所有 LLM 调用点均传 `self.model`（或 `MODEL_NAME`），`retrieve_model` 被加载存储后从不被读取。
- `_normalize_retrieve_model`（`client.py:32-36`）是恒等函数（`return model`），无实际归一化逻辑。
- 调用点 LLM 路径分两类（决定 `retrieve_model` 接入哪些点）：
  - **索引路径（Indexing path，构建树结构，应用 `model`）**：`page_index.py` 全部 16 处 `llm_completion`/`llm_acompletion`（行 39/67/119/138/156/168/179/196/218/267/295/324/485/535/569/754）、`utils.py:659` `generate_node_summary`、`utils.py:704` `generate_doc_description`、`page_index_md.py:282` `generate_doc_description`。这些在 `page_index()`/`md_to_tree()` 调用栈内，用户显式传 `model=`，属建索引，**不接 `retrieve_model`**。
  - **检索路径（Retrieval path，查询时定位文档/节点/生成答案）**：见 §5.2 清单。这些是 `retrieve_model` 的接入点。

#### 4.0.3 魔法数字散落现状
- `page_index_md.py:312-316`（`__main__`）：`MODEL="gpt-4.1"`、`IF_THINNING=False`、`THINNING_THRESHOLD=5000`、`SUMMARY_TOKEN_THRESHOLD=200`、`IF_SUMMARY=True`。
- `main.py:70-72`（`generate_structure`）：`toc_check_page_num=20`、`max_page_num_each_node=10`、`max_token_num_each_node=20000` —— 重复 `config.yaml` 已有的同名默认值（L25-27），改 `config.yaml` 对 `main.py` 无效。
- `config.yaml:25-27` 已有 `toc_check_page_num/max_page_num_each_node/max_token_num_each_node`，但 `main.py` 硬编码绕过。

#### 4.0.4 `.env.example` / `config.yaml` 默认值张力
- `.env.example:17` `DASHSCOPE_API_KEY` 取消注释、`:18` `OPENAI_API_KEY` 注释、`:21` `OPENAI_BASE_URL` 指向 DashScope —— shipped 默认 = DashScope 活跃。
- `config.yaml:23` `model: "qwen-plus"` —— 与 DashScope 端点匹配，但与 P6 "标准 OpenAI 默认" 目标矛盾。
- P6 已把 `utils.py` 的 `base_url` 默认改为标准 OpenAI（`_resolve_llm_config`），但 `.env.example` shipped 默认仍指向 DashScope —— 新无 key 用户得到 "OpenAI base_url 默认 + DashScope env 覆盖" 的不一致。
- **张力解决方案（用户确认）**：`.env.example` shipped 默认对齐 OpenAI（取消注释 `OPENAI_API_KEY`、注释 `DASHSCOPE_API_KEY`、`OPENAI_BASE_URL` 默认 OpenAI 或注释）；`config.yaml` `model:` 默认改为 OpenAI 模型（如 `gpt-4.1-mini`），与 `.env` OpenAI 默认一致。

### 4.1 替代方案（Alternatives）

#### 方案 A — `retrieve_model` 全局注入（包装层）
在 `utils.py` 新增 `retrieve_completion` / `retrieve_acompletion` 包装函数，内部用模块级全局 `RETRIEVE_MODEL` 变量（由 `configure_llm` 或独立 setter 设置），检索调用点改调包装函数。
- ✅ 优点：调用点改动小（仅换函数名）；`retrieve_model` 解析集中在 `utils.py`。
- ❌ 缺点：引入第二组全局状态 + 第二组 setter，与 P6 "单一配置源" 目标相悖；`retrieve_model` 已在 `ConfigLoader`/`PageIndexClient` 解析，再起模块级全局是第三处真相；包装函数破坏 `llm_completion`/`llm_acompletion` 签名稳定性目标（P6 L5）。

#### 方案 B — 调用点注入 `retrieve_model or model`（参数级回退）⭐ 推荐
在检索路径的每个 LLM 调用点，把 `self.model`（或 `MODEL_NAME`）替换为 `self.retrieve_model or self.model`（`PageIndexClient` 路径）/ `RETRIEVE_MODEL_NAME or MODEL_NAME`（`main.py` 路径，经 `ConfigLoader` 解析后即 `cfg.retrieve_model or cfg.model`）。`retrieve_model` 未设置时回退 `model`，行为与现状完全一致（向后兼容）。
- ✅ 优点：无新全局状态；复用已有 `ConfigLoader` 解析链（`retrieve_model` 已解析到 `PageIndexClient.retrieve_model`）；回退语义清晰；`llm_completion`/`llm_acompletion` 签名不变。
- ❌ 缺点：需在每个检索调用点显式改 `self.model` → `self.retrieve_model or self.model`（5 个类 × 1-2 处）；`main.py` 的 `_call_llm_json`/`generate_answer` 用模块级 `MODEL_NAME`，需新增 `RETRIEVE_MODEL_NAME` 解析。
- 实现细节：`PageIndexClient` 已存 `self.retrieve_model`（`client.py:60`，已做 `or self.model` 回退），故检索类只需在 `__init__` 接收 `retrieve_model` 并存 `self.retrieve_model`，调用点改用之。`AgenticRouter`/`SuperTreeIndex`/`ClosetIndex`/`DescriptionStrategy`/`CRAGVerifier`/`RetrievalPlanner` 当前 `__init__(model)` 只收 `model` —— 扩展为 `__init__(model, retrieve_model=None)`，`retrieve_model` 默认 None 时回退 `model`。

#### 方案 C — 仅 `PageIndexClient` 层注入，不改子组件
仅在 `PageIndexClient.__init__` 把 `retrieve_model` 传给子组件构造，但子组件调用点不改（仍用 `self.model`）。
- ❌ 缺点：`retrieve_model` 仍不被读取使用，等于没接；与 FR2 "检索 LLM 调用使用 `retrieve_model`" 直接矛盾。

### 4.2 推荐：方案 B（调用点注入 `retrieve_model or model`）
- 与 P6 "单一配置源（`ConfigLoader`）" 一致：`retrieve_model` 经 `ConfigLoader.load()` 解析，`PageIndexClient` 存 `self.retrieve_model`，传给检索子组件。
- 向后兼容：`retrieve_model` 未设置 / 空 → 回退 `model`（`or self.model` 语义），行为 = 现状。
- `main.py` 路径（`_call_llm_json`/`generate_answer`）：新增 `RETRIEVE_MODEL_NAME = ConfigLoader().load(None).retrieve_model`，调用点用 `RETRIEVE_MODEL_NAME or MODEL_NAME`。

### 4.3 设计决策（DESIGN_DECISION — 已确认）
✅ **用户确认**：方案 B（调用点注入 `retrieve_model or model`）。魔法数字入 `config.yaml` + `ConfigLoader.load()` 暴露；`.env.example` 对齐 OpenAI；`config.yaml` `model:` 默认改 OpenAI 模型。详见 §5。

---

## §5 接口设计（Interface Design）

### 5.1 `ConfigLoader.load()` 新增键（config.yaml 扩展）

`config.yaml` 新增 4 个键（默认值 = 当前硬编码值，保证向后兼容）：

| 键 | 默认值 | 来源（当前硬编码处） | 用途 |
|----|--------|----------------------|------|
| `if_thinning` | `false` | `page_index_md.py:313` `IF_THINNING=False` | `__main__` 薄化开关 |
| `thinning_threshold` | `5000` | `page_index_md.py:314` `THINNING_THRESHOLD=5000` | `__main__` 薄化阈值 |
| `summary_token_threshold` | `200` | `page_index_md.py:315` `SUMMARY_TOKEN_THRESHOLD=200` | `__main__` 摘要阈值 |
| `if_summary` | `true` | `page_index_md.py:316` `IF_SUMMARY=True` | `__main__` 摘要开关 |

> 注：`toc_check_page_num`/`max_page_num_each_node`/`max_token_num_each_node` 已在 `config.yaml:25-27`，无需新增，仅需让 `main.py:70-72` 读取之（§5.4）。

`_validate_keys` 自动接受新键（因新键在 `_default_dict` 中）。`load()` 返回的 `config(SimpleNamespace)` 自动暴露 `cfg.if_thinning` / `cfg.thinning_threshold` / `cfg.summary_token_threshold` / `cfg.if_summary`。

### 5.2 `retrieve_model` 注入点清单（检索路径 LLM 调用点，全枚举）

> 区分原则：**索引路径**（`page_index()`/`md_to_tree()` 调用栈内，建树用 `model`，**不接 `retrieve_model`**）vs **检索路径**（查询时定位文档/节点/生成/校验答案，**接 `retrieve_model or model`**）。

#### 5.2.1 `PageIndexClient` 路径（server / `client.py`）

`PageIndexClient.retrieve_model` 已在 `client.py:60` 解析（`_normalize_retrieve_model(opt.retrieve_model or self.model)`）。子组件构造扩展接收 `retrieve_model`：

| 文件:行 | 类 / 函数 | 当前调用 | 改为 |
|---------|-----------|----------|------|
| `pageindex_mutil/closet_index.py:72` | `ClosetIndex._extract_tags` | `llm_completion(self.model, prompt)` | `llm_completion(self.retrieve_model or self.model, prompt)` |
| `pageindex_mutil/super_tree.py:103` | `SuperTreeIndex`（L? 选文档，sync） | `llm_completion(self.model, prompt)` | `llm_completion(self.retrieve_model or self.model, prompt)` |
| `pageindex_mutil/super_tree.py:234` | `SuperTreeIndex.select_documents` | `await llm_acompletion(self.model, prompt)` | `await llm_acompletion(self.retrieve_model or self.model, prompt)` |
| `pageindex_mutil/agentic/planner.py:41` | `RetrievalPlanner.plan` | `await llm_acompletion(self.model, prompt)` | `await llm_acompletion(self.retrieve_model or self.model, prompt)` |
| `pageindex_mutil/agentic/strategies.py:87` | `DescriptionStrategy.search`（fallback 分支） | `llm_completion(self.model, prompt)` | `llm_completion(self.retrieve_model or self.model, prompt)` |
| `pageindex_mutil/agentic/verifier.py:78` | `CRAGVerifier.verify` | `llm_completion(self.model, prompt)` | `llm_completion(self.retrieve_model or self.model, prompt)` |

子组件 `__init__` 扩展（接收 `retrieve_model`，默认 None → 回退 `model`）：

| 文件:行 | 类 | 当前签名 | 改为 |
|---------|-----|-----------|------|
| `pageindex_mutil/closet_index.py:42` | `ClosetIndex.__init__` | `__init__(self, db, model: str)` | `__init__(self, db, model: str, retrieve_model: str = None)`；`self.retrieve_model = retrieve_model` |
| `pageindex_mutil/super_tree.py:132` | `SuperTreeIndex.__init__` | `__init__(self, db, model: str, client)` | `__init__(self, db, model: str, client, retrieve_model: str = None)`；`self.retrieve_model = retrieve_model` |
| `pageindex_mutil/agentic/planner.py:16` | `RetrievalPlanner.__init__` | `__init__(self, model: str)` | `__init__(self, model: str, retrieve_model: str = None)`；`self.retrieve_model = retrieve_model` |
| `pageindex_mutil/agentic/strategies.py:52` | `DescriptionStrategy.__init__` | `__init__(self, model: str)` | `__init__(self, model: str, retrieve_model: str = None)`；`self.retrieve_model = retrieve_model` |
| `pageindex_mutil/agentic/verifier.py:25` | `CRAGVerifier.__init__` | `__init__(self, model: str)` | `__init__(self, model: str, retrieve_model: str = None)`；`self.retrieve_model = retrieve_model` |
| `pageindex_mutil/agentic/router.py:15` | `AgenticRouter.__init__` | `__init__(self, client, model: str)` | `__init__(self, client, model: str, retrieve_model: str = None)`；传 `retrieve_model` 给 `RetrievalPlanner`/`DescriptionStrategy`/`CRAGVerifier` |

`client.py:76-79` `PageIndexClient.__init__` 传 `retrieve_model` 给子组件：
- `ClosetIndex(self.db, self.model)` → `ClosetIndex(self.db, self.model, self.retrieve_model)`
- `SuperTreeIndex(self.db, self.model, self)` → `SuperTreeIndex(self.db, self.model, self, self.retrieve_model)`
- `AgenticRouter(self, self.model)` → `AgenticRouter(self, self.model, self.retrieve_model)`

#### 5.2.2 `main.py` 路径（CLI）

`main.py` 用模块级 `client` + `MODEL_NAME`，不经 `PageIndexClient`。新增 `RETRIEVE_MODEL_NAME` 解析：

| 文件:行 | 函数 | 当前调用 | 改为 |
|---------|------|----------|------|
| `main.py:49` | 模块级 | `MODEL_NAME = ConfigLoader().load(None).model` | 新增下一行：`RETRIEVE_MODEL_NAME = ConfigLoader().load(None).retrieve_model` |
| `main.py:145` | `_call_llm_json` | `model=MODEL_NAME` | `model=RETRIEVE_MODEL_NAME or MODEL_NAME` |
| `main.py:292` | `generate_answer` | `model=MODEL_NAME` | `model=RETRIEVE_MODEL_NAME or MODEL_NAME` |

> `get_relevant_nodes`（被 `_call_llm_json` 包装）和 `get_relevant_documents_for_multidoc`（`DescriptionStrategy` 复用）属检索路径，经 `_call_llm_json` 自动接入。

#### 5.2.3 不接入 `retrieve_model` 的点（索引路径，保持 `model`）
- `page_index.py` 全部 16 处 `llm_completion`/`llm_acompletion`（行 39/67/119/138/156/168/179/196/218/267/295/324/485/535/569/754）—— `page_index_main`/`page_index` 建树调用栈内，用户显式传 `opt.model`。
- `utils.py:659` `generate_node_summary`、`utils.py:704` `generate_doc_description` —— 建索引时生成节点摘要/文档描述，属索引路径。
- `page_index_md.py:282` `generate_doc_description` —— `md_to_tree` 建索引路径。

### 5.3 `page_index_md.py` `__main__` 改动（FR1 + FR4）

当前（`page_index_md.py:312-324`）：
```python
MODEL="gpt-4.1"
IF_THINNING=False
THINNING_THRESHOLD=5000
SUMMARY_TOKEN_THRESHOLD=200
IF_SUMMARY=True
tree_structure = asyncio.run(md_to_tree(..., model=MODEL))
```

改为（走 `ConfigLoader`，与 `main.py:49` 一致）：
```python
from .utils import ConfigLoader  # 或顶部已 import
_cfg = ConfigLoader().load(None)
tree_structure = asyncio.run(md_to_tree(
    md_path=MD_PATH,
    if_thinning=_cfg.if_thinning,
    min_token_threshold=_cfg.thinning_threshold,
    if_add_node_summary='yes' if _cfg.if_summary else 'no',
    summary_token_threshold=_cfg.summary_token_threshold,
    model=_cfg.model))
```

> 注：`page_index_md.py` 作为 `pageindex_mutil` 包内模块，`__main__` 直接运行时需保证 `from .utils import ConfigLoader` 可解析（包内相对导入在 `python -m pageindex_mutil.page_index_md` 下成立；若直接 `python page_index_md.py` 则相对导入失败 —— 此为既有约束，本工作不改变运行方式，仅改配置来源）。`model=_cfg.model` 走 `ConfigLoader`（`MODEL_NAME` env > `config.yaml`），与 `main.py:49` 一致。

### 5.4 `main.py:70-72` 改动（FR5）

当前（`main.py:68-77` `generate_structure`）：
```python
opt = SimpleNamespace(
    model=MODEL_NAME,
    toc_check_page_num=20,
    max_page_num_each_node=10,
    max_token_num_each_node=20000,
    if_add_node_id='yes',
    ...
)
```

改为（用 `ConfigLoader().load()` 填默认，消除重复硬编码）：
```python
_cfg = ConfigLoader().load(None)
opt = SimpleNamespace(
    model=MODEL_NAME,
    toc_check_page_num=_cfg.toc_check_page_num,
    max_page_num_each_node=_cfg.max_page_num_each_node,
    max_token_num_each_node=_cfg.max_token_num_each_node,
    if_add_node_id='yes',
    ...
)
```

> `model=MODEL_NAME` 保留（`MODEL_NAME` 已是 `ConfigLoader().load(None).model`，§5.2.2）；`toc_check_page_num` 等改读 `_cfg`，使改 `config.yaml` 对 `main.py` 生效。`if_add_node_id`/`if_add_node_summary` 等保持现状（不在 W6 范围）。

### 5.5 `.env.example` 改动（FR3）

当前 shipped 默认 = DashScope 活跃。改为对齐标准 OpenAI：
- `:17` `DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx` → 注释：`# DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx`
- `:18` `# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx` → 取消注释：`OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx`
- `:21` `OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1` → 注释或改默认 OpenAI：`# OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`（让 `utils.py` 默认 `https://api.openai.com/v1` 生效）
- `:23-27` OpenAI provider alternative 块：注释块调整（主块已是 OpenAI，alternative 块改为 DashScope alternative）
- `:50` `MODEL_NAME=qwen-plus` → `MODEL_NAME=gpt-4.1-mini`（与 `config.yaml` `model:` 默认对齐）
- 顶部注释说明同步更新（"DashScope is the default provider" → "OpenAI is the default provider"）

### 5.6 `config.yaml` 改动（FR4 + OpenAI 对齐）

- `:23` `model: "qwen-plus"` → `model: "gpt-4.1-mini"`（与 `.env.example` OpenAI 默认一致）
- `:24` `retrieve_model: "qwen-plus"` → `retrieve_model: "gpt-4.1-mini"`（与 `model` 默认一致；未设置时回退 `model` 也成立）
- 新增 4 键（§5.1）：`if_thinning: false`、`thinning_threshold: 5000`、`summary_token_threshold: 200`、`if_summary: true`
- 顶部注释更新（"defaults below target DashScope/Qwen endpoint" → "defaults below target standard OpenAI endpoint"）

---

## §6 数据/配置变更（Data & Config Changes）

### 6.1 `config.yaml` 新键与默认值

```yaml
# W6 新增（默认值 = 当前硬编码值，向后兼容）
if_thinning: false
thinning_threshold: 5000
summary_token_threshold: 200
if_summary: true

# 默认值变更（OpenAI 对齐）
model: "gpt-4.1-mini"        # was "qwen-plus"
retrieve_model: "gpt-4.1-mini" # was "qwen-plus"
```

> `if_thinning`/`if_summary` 为布尔；`ConfigLoader` 用 `yaml.safe_load` 解析，YAML `false`/`true` → Python `bool`。`page_index_md.py` `__main__` 直接用 `cfg.if_thinning`（bool），`if_add_node_summary` 仍用 `'yes' if cfg.if_summary else 'no'` 字符串转换（`md_to_tree` 既有契约）。

### 6.2 `.env.example` shipped 默认对齐

| 行 | 变更前 | 变更后 |
|----|--------|--------|
| 17 | `DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx` | `# DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx` |
| 18 | `# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx` | `OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx` |
| 21 | `OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1` | `# OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`（注释，让 `utils.py` 默认 OpenAI 生效） |
| 50 | `MODEL_NAME=qwen-plus` | `MODEL_NAME=gpt-4.1-mini` |
| 顶部注释 | "DashScope is the default provider" | "OpenAI is the default provider; DashScope is the alternative" |

### 6.3 CI grep 守卫（NFR1）

项目当前无 `.github/workflows/`（已核实）。守卫以可执行脚本形式提供，供 CI 或本地 `make` 调用：

```bash
#!/usr/bin/env bash
# scripts/guard-no-hardcoded-model.sh — NFR1 grep 守卫
set -euo pipefail
# 不允许 pageindex_mutil/ 内出现硬编码 OpenAI 模型名（MODEL="gpt-..." 形式）
if grep -rnE 'MODEL\s*=\s*"gpt-' pageindex_mutil/ --include='*.py'; then
  echo "FAIL: hardcoded MODEL=\"gpt-...\" found in pageindex_mutil/ (must use ConfigLoader)"
  exit 1
fi
echo "OK: no hardcoded MODEL=\"gpt-...\" in pageindex_mutil/"
```

> 实现阶段决定落点（`.github/workflows/ci.yml` 新建 / `Makefile` target / `scripts/` 脚本）。守卫目标：`page_index_md.py:312` 的 `MODEL="gpt-4.1"` 被移除后不再回归。

---

## §7 风险与缓解（Risk & Mitigation）

| 风险 | 影响 | 缓解 | 回滚 |
|------|------|------|------|
| **R1 `retrieve_model` 回退正确性** | `retrieve_model` 未设置时检索调用点应用 `model`，行为须 = 现状 | `self.retrieve_model or self.model` 语义：`retrieve_model` 为 None/空 → 回退 `model`；`PageIndexClient.retrieve_model` 已在 `client.py:60` 做 `opt.retrieve_model or self.model` 回退，子组件再 `self.retrieve_model or self.model` 双重保险 | revert 接入提交，调用点恢复 `self.model` |
| **R2 `config.yaml` 向后兼容** | 新增键 / 改默认值可能破坏现有部署 | 新键默认值 = 当前硬编码值（`if_thinning: false` 等）；`model` 默认改 `gpt-4.1-mini` 是**有意行为变更**（OpenAI 对齐），但 `MODEL_NAME` env 可覆盖，现有 DashScope 部署设 `MODEL_NAME=qwen-plus` + `DASHSCOPE_API_KEY` 即恢复 | revert `config.yaml` 提交 |
| **R3 14 测试不回归** | `tests/test_config.py` 14 测试断言 `cfg.model == "qwen-plus"` / `cfg.retrieve_model == "qwen-plus"`（`:45-46` 等），改 `config.yaml` 默认为 `gpt-4.1-mini` 会使这些断言失败 | **测试需同步更新**：断言期望从 `"qwen-plus"` 改为 `"gpt-4.1-mini"`（`test_no_env_falls_back_to_yaml_default`、`test_whitespace_*_falls_back`、`test_empty_string_*_falls_back` 等涉及 yaml 默认值的断言）；env 覆盖类断言（`test_model_name_env_overrides_yaml_default` 等）不涉及默认值，无需改 | revert 测试更新提交 |
| **R4 OpenAI 模型默认与 endpoint 匹配** | `config.yaml` `model: gpt-4.1-mini` + `.env.example` `OPENAI_API_KEY` 活跃 → 标准 OpenAI 端点，模型名匹配 | shipped 默认自洽：OpenAI key + OpenAI base_url 默认 + OpenAI 模型名；DashScope 用户取消注释 `DASHSCOPE_API_KEY` + 设 `MODEL_NAME=qwen-plus` 即切回 | revert OpenAI 对齐提交 |
| **R5 `page_index_md.py` `__main__` 相对导入** | `from .utils import ConfigLoader` 在直接 `python page_index_md.py` 时失败（相对导入需 `-m`） | 既有约束（`page_index_md.py` 作为包内模块，`__main__` 块本就以包内运行为前提）；若需支持直接运行，用 `try/except ImportError` fallback 到 `from pageindex_mutil.utils import ConfigLoader`；实测当前 `__main__` 块未被任何测试/CI 调用（仅手动运行） | 保持 `MODEL="gpt-4.1"` 硬编码（不可接受，违反 FR1） |
| **R6 `retrieve_model` 接入测试覆盖** | 新接入的检索路径调用点无测试覆盖则无法验证回退正确性 | 新增接线测试：mock `llm_completion`/`llm_acompletion`，断言 `retrieve_model` 设置时检索调用点传 `retrieve_model`，未设置时传 `model`；`PageIndexClient` 构造时传 `retrieve_model="r-model"` → 子组件 `self.retrieve_model == "r-model"` | 测试失败 → 不合并 |

### 7.1 `.env` OpenAI 对齐与 `config.yaml` `model` 默认值张力（已解决）
- §3 ASSUMPTION 提出"`.env.example` 对齐 OpenAI 后，`config.yaml` 的 `model:` 默认值是否需相应调整"。
- **用户确认解决**：`config.yaml` `model:` 默认改为 OpenAI 模型（`gpt-4.1-mini`），与 `.env.example` OpenAI 默认一致。张力消除：shipped 默认 = OpenAI key + OpenAI base_url + OpenAI 模型名，三者自洽。DashScope 用户通过 env 覆盖切回。

---

## §8 验收标准（Acceptance Criteria）

### 8.1 映射 §3 FR/NFR

| 需求 | 验收信号 | 验证方式 |
|------|----------|----------|
| **FR1**（移除 `page_index_md.py:312` 硬编码 `MODEL="gpt-4.1"`） | `grep -rnE 'MODEL\s*=\s*"gpt-' pageindex_mutil/ --include='*.py'` 返回空；`page_index_md.py` `__main__` 走 `ConfigLoader` | grep 守卫（§6.3）+ 代码审查 |
| **FR2**（`retrieve_model` 接入检索路径） | §5.2 清单全部 6 个 `PageIndexClient` 路径调用点 + 2 个 `main.py` 路径调用点改为 `retrieve_model or model`；新增接线测试 GREEN | 接线测试 + 代码审查 |
| **FR3**（`.env.example` 对齐 OpenAI） | `OPENAI_API_KEY` 取消注释、`DASHSCOPE_API_KEY` 注释、`OPENAI_BASE_URL` 注释（让 utils 默认 OpenAI 生效）、`MODEL_NAME=gpt-4.1-mini` | `.env.example` diff 审查 |
| **FR4**（魔法数字入 `config.yaml`） | `config.yaml` 含 `if_thinning`/`thinning_threshold`/`summary_token_threshold`/`if_summary` 4 新键；`ConfigLoader().load(None)` 暴露之；`page_index_md.py` `__main__` 读取之 | 配置解析测试 + 代码审查 |
| **FR5**（`main.py:70-72` 用 `ConfigLoader` 填默认） | `main.py` `generate_structure` 中 `toc_check_page_num`/`max_page_num_each_node`/`max_token_num_each_node` 读自 `_cfg`，无硬编码 `20`/`10`/`20000` | grep `toc_check_page_num=20\|max_page_num_each_node=10\|max_token_num_each_node=20000` 在 `main.py` 返回空 + 代码审查 |
| **NFR1**（CI grep 守卫） | `scripts/guard-no-hardcoded-model.sh`（或等价 CI step）退出码 0 | 执行守卫脚本 |
| **NFR2**（14 测试 GREEN + 新增测试） | `pytest tests/test_config.py` 全绿（14 既有 + 新增 retrieve_model 接线 + 魔法数字解析测试）；既有断言期望同步更新（R3） | `pytest -q tests/test_config.py` |
| **NFR3**（`MODEL_NAME` 解析链不回归） | 优先级 caller显式 > env > yaml 保持；14 测试中优先级类断言 GREEN | `pytest -q tests/test_config.py`（优先级类测试） |
| **NFR4**（`retrieve_model` 未设置时回退 `model`） | `retrieve_model` 为 None/空时检索调用点传 `model`，行为 = 现状；接线测试覆盖回退路径 | 接线测试（未设 `retrieve_model` 分支） |
| **NFR5**（`config.yaml` 新键默认 = 当前硬编码值） | `if_thinning: false`/`thinning_threshold: 5000`/`summary_token_threshold: 200`/`if_summary: true` | `config.yaml` diff 审查 + 配置解析测试 |

### 8.2 最终守卫脚本

```bash
#!/usr/bin/env bash
set -euo pipefail

# NFR1: 无硬编码 MODEL="gpt-..." in pageindex_mutil/
if grep -rnE 'MODEL\s*=\s*"gpt-' pageindex_mutil/ --include='*.py'; then
  echo "FAIL: hardcoded MODEL"; exit 1
fi

# FR5: main.py 无重复硬编码 config.yaml 值
if grep -nE 'toc_check_page_num=20|max_page_num_each_node=10|max_token_num_each_node=20000' main.py; then
  echo "FAIL: main.py hardcoded config values"; exit 1
fi

# FR3: .env.example OpenAI 对齐
grep -q '^OPENAI_API_KEY=' .env.example || { echo "FAIL: OPENAI_API_KEY not uncommented"; exit 1; }
grep -q '^# DASHSCOPE_API_KEY=' .env.example || { echo "FAIL: DASHSCOPE_API_KEY not commented"; exit 1; }

# FR4: config.yaml 含新键
grep -q '^if_thinning:' pageindex_mutil/config.yaml || { echo "FAIL: if_thinning missing"; exit 1; }
grep -q '^thinning_threshold:' pageindex_mutil/config.yaml || { echo "FAIL: thinning_threshold missing"; exit 1; }

# OpenAI 默认一致
grep -q '^model: "gpt-4.1-mini"' pageindex_mutil/config.yaml || { echo "FAIL: model default not OpenAI"; exit 1; }
grep -q '^MODEL_NAME=gpt-4.1-mini' .env.example || { echo "FAIL: MODEL_NAME not OpenAI"; exit 1; }

# NFR2: 14 测试 + 新增测试 GREEN
pytest -q tests/test_config.py

echo "ALL GUARDS PASSED"
```

### 8.3 过程纪律（devkit 铁律）
- **Two-Agent Minimum**：W6 实现由子代理（leaf executor）执行文件编辑，由**不同**子代理（independent verifier）独立验证——禁止自审自评。
- **TDD/L2**：先写 `retrieve_model` 接线测试（RED：断言检索调用点传 `retrieve_model`）→ 实现 → 测试 GREEN。
- 实现阶段产出 handoff.md（NEEDS_INDEPENDENT_VERIFICATION）→ verifier 独立跑 §8.2 守卫 → quality-gate（threshold ≥70）。

---

## §9 实现提交拆分建议（Implementation Commit Split — 供 tasks.md 参考）

> 非强制，按 tasks.md 最终拆分。每提交独立可验证 + GREEN。

- **C1 config.yaml + ConfigLoader 扩展**：新增 4 键 + `model`/`retrieve_model` 默认改 OpenAI；同步更新 `tests/test_config.py` 断言期望（R3）。
- **C2 `.env.example` OpenAI 对齐**：取消注释 `OPENAI_API_KEY`、注释 `DASHSCOPE_API_KEY`、`MODEL_NAME=gpt-4.1-mini`、注释更新。
- **C3 `page_index_md.py` `__main__` 走 ConfigLoader**（FR1 + FR4 消费端）。
- **C4 `main.py:70-72` 用 `ConfigLoader` 填默认**（FR5）+ `main.py` `RETRIEVE_MODEL_NAME` 解析 + `_call_llm_json`/`generate_answer` 注入（FR2 main.py 路径）。
- **C5 `retrieve_model` 接入 `PageIndexClient` 路径**（FR2）：子组件 `__init__` 扩展 + `client.py` 传参 + 6 个调用点注入 + 接线测试。
- **C6 CI grep 守卫**（NFR1）：`scripts/guard-no-hardcoded-model.sh` 或 `.github/workflows/ci.yml`。

---

## 设计阶段完成检查
- [x] §4 ≥2 替代方案 + 权衡（A/B/C），用户选 B
- [x] §5 接口设计：`ConfigLoader.load()` 新键、`retrieve_model` 注入点清单（全枚举 8 调用点 file:line）、`page_index_md`/`main.py` 改动
- [x] §6 数据/配置变更：`config.yaml` 新键、`.env.example`、CI grep 守卫
- [x] §7 风险与缓解（6 项 + `.env`/`config.yaml` 张力已解决）
- [x] §8 验收标准：映射 §3 FR/NFR + 最终守卫脚本 + Two-Agent/TDD 纪律
> 设计完成。下一步：plan 阶段（task-planning → tasks.md）。
