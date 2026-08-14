# L0 证据束统一 + L2 复用 query 级共享物 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent (recommended) or compose:execute to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把当前偏离 spec [S5]/[S7] 的三层实现收敛到 spec 契约：`build_evidence_bundle`（证据束）成为唯一 L0 召回源（删除独立 `prefilter`），query tokens / query entities 一次计算、全链引用，L2 直接复用 L0 的 query 级共享物而非重新解析。

**Architecture:** L0 = `build_evidence_bundle` 返回 `(bundle, ctx)`（ctx 携带 query tokens + 展平后的 query entities），候选集从 `bundle.keys()` 按 `derive_evidence_score` 派生；L1 继续消费 bundle（不变）；L2 经 `_act_tree_search → _recall_nodes_for_doc → enhance_and_select` 透传 `ctx`，不再重复 tokenize / search_entities。删除 `super_tree.prefilter` 与其 HyDE 二次调用。

**Tech Stack:** Python 3.12+ / uv / SQLite / jieba。测试 `uv run pytest <path> -q`（禁 `--timeout`）。

## Global Constraints

- 测试用 `uv run pytest <path> -q`（pytest-timeout 未安装，**禁止 --timeout**）；全套件基线 530 passed / 3 env（3 个 TestChromaSearchBackend 无网既有失败，**不得归因新代码**）
- 提交风格 `refactor(P2): 描述`；`git add` 指定文件（禁 `git add -A`）；**NEVER `git config`**
- 决策契约：只直通不约束——不得引入保底席位/规则回退重排
- NFR4：每个新 LLM 调用点 `retrieve_model or self.model`（本重构不新增 LLM 调用点）
- 证据束 keyword 通道与 prefilter 通道 B 同源（`KeywordIndex.search` 即 `db.match_doc_keywords` 薄封装，super_tree.py:52-56）——**成员集合等价、不得另起数据源**；但候选**排序标量**从 prefilter 浮点分累加变为 `derive_evidence_score` 加权计数（score→count，[S5]/[S6]#8 本意）
- `resolve_query_entities` 的既有行为契约不变（其测试 test_unified_enhance.py:715-788 必须保持绿）

---

## File Structure

- **Modify** `pageindex_mutil/agentic/enhance.py` — 抽 `_flatten_entity_names`（T1）；`enhance_and_select` 增 `query_tokens` 参数（T3）
- **Modify** `pageindex_mutil/agentic/evidence.py` — `build_evidence_bundle` 返回 `(bundle, ctx)`（T1）
- **Modify** `pageindex_mutil/super_tree.py` — 删除 `prefilter`（T2）
- **Modify** `pageindex_mutil/agentic/router.py` — 解包 tuple（T1）；`_search_super_tree` 候选集从 bundle 派生 + 删 HyDE prefilter（T2）；`_act_tree_search`/`_recall_nodes_for_doc` 透传 ctx（T3）
- **Tests** — `test_evidence_bundle.py`（T1）、`test_unified_enhance.py`（T1/T3）、`test_super_tree.py`（T2）、`test_router.py`（T2）、`test_multi_doc_enhanced.py`（T1/T2/T3）、`test_multi_hop.py`（T2）、`test_client_integration.py`（T2）

---

### Task 1: 证据束返回 query 级共享物 `(bundle, ctx)` + 实体名展平抽取

**Covers:** [S5]（query tokens/entities 一次计算全链引用）

**Files:**
- Modify: `pageindex_mutil/agentic/enhance.py:502-545`（抽 `_flatten_entity_names`，`resolve_query_entities` 复用）
- Modify: `pageindex_mutil/agentic/evidence.py:25-153`（返回 tuple + ctx）
- Modify: `pageindex_mutil/agentic/router.py:447`（解包）
- Test: `tests/test_evidence_bundle.py`、`tests/test_multi_doc_enhanced.py:423`

**Interfaces:**
- Consumes: 现有 `db.search_entities` / `KeywordIndex._tokenize`
- Produces: `build_evidence_bundle(client, db, query, topk=30, max_hop=None) -> tuple[dict, dict]`：
  - `bundle`：原结构不变（`{db_id: {"channels": {...}, "graph": {...}}}`）
  - `ctx`：`{"tokens": list[str], "query_entities": list[str]}`（query 实体 = `_flatten_entity_names(search_entities(limit=topk))` 的展平名单，T3 由 L2 复用）
  - `_flatten_entity_names(rows) -> list[str]`：纯函数（casefold 判重，保留首见写法），供 `resolve_query_entities` 与 `build_evidence_bundle` 复用

- [ ] **Step 1: 写失败测试**

```python
# tests/test_evidence_bundle.py 追加（文件头部 import 区已有 build_evidence_bundle/derive_evidence_score）
def test_bundle_returns_query_ctx():
    """义务：build_evidence_bundle 返回 (bundle, ctx)，ctx 携带 query tokens 与展平实体名。"""
    from pageindex_mutil.agentic.evidence import build_evidence_bundle
    client, db = _make_client(tmp_path,
        [("A", "文档A", "", [("浴血", "content", 3)])], None, None)
    # 造一个含别名的实体，验证 ctx.query_entities 展平 name+aliases
    db.insert_entity("concept", "浴血值", ["浴血"])
    bundle, ctx = build_evidence_bundle(client, db, "浴血怎么获得", topk=30)
    assert isinstance(bundle, dict)
    assert "tokens" in ctx and isinstance(ctx["tokens"], list)
    assert "query_entities" in ctx and "浴血值" in ctx["query_entities"]
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_evidence_bundle.py::test_bundle_returns_query_ctx -q`
Expected: FAIL（`build_evidence_bundle` 返回 dict，解包 `bundle, ctx = ...` 抛 `ValueError: not enough values to unpack`）

- [ ] **Step 3: 实现**

`enhance.py`（在 `resolve_query_entities` 之前插入纯函数，并把 `resolve_query_entities` 主体换成复用）：

```python
def _flatten_entity_names(rows) -> list:
    """把 search_entities 结果行展平为「名字 + 别名」字符串列表（casefold 判重，
    保留首见写法）。纯函数，供 resolve_query_entities 与 build_evidence_bundle 复用
    （query 实体一次解析、全链引用，[S5]）。"""
    names, seen = [], set()

    def _add(value):
        if not isinstance(value, str):
            return
        v = value.strip()
        if not v:
            return
        key = v.casefold()
        if key not in seen:
            seen.add(key)
            names.append(v)

    for row in rows or []:
        if not isinstance(row, dict):
            continue
        _add(row.get("name"))
        aliases = row.get("aliases")
        if isinstance(aliases, str):
            try:
                aliases = json.loads(aliases)
            except (TypeError, ValueError):
                aliases = []
        if isinstance(aliases, list):
            for alias in aliases:
                _add(alias)
    return names


def resolve_query_entities(db, query, limit=5) -> list:
    """查询实体解析（共享助手）：search_entities 命中实体的规范名 + 别名展开。"""
    if db is None or not query or not str(query).strip():
        return []
    try:
        rows = db.search_entities(query, limit=limit)
    except Exception as e:
        logging.warning("resolve_query_entities: search_entities failed: %s", e)
        return []
    return _flatten_entity_names(rows)
```

`evidence.py`（实体通道取到 rows 后展平；末尾返回 tuple）：

```python
# 在 build_evidence_bundle 内、query_ids 计算之后（原 evidence.py:113 附近）插入：
    from .enhance import _flatten_entity_names
    query_entities = _flatten_entity_names(entities)
```

```python
# 函数末尾（原 evidence.py:149-153 的 `for db_id, e in bundle.items():` 去重之后），
# 把 `return bundle` 改为：
    return bundle, {"tokens": tokens, "query_entities": query_entities}
```

`router.py:447` 解包（T2 会用 ctx，本步先丢弃）：

```python
            bundle, _ = build_evidence_bundle(
                self.client, db, query,
                topk=getattr(cfg, "l0_channel_topk", 30),
            )
```

- [ ] **Step 4: 迁移既有调用点 + 运行确认通过**

`tests/test_evidence_bundle.py`：把 6 处 `bundle = build_evidence_bundle(...)` 全部改为 `bundle, ctx = build_evidence_bundle(...)`（行 38/72/89/108/124/156；行 183 是裸调用无赋值，不用改）。`tests/test_multi_doc_enhanced.py:423` 的 `return_value=bundle` 改为 `return_value=(bundle, {"tokens": [], "query_entities": []})`。

Run: `uv run pytest tests/test_evidence_bundle.py tests/test_unified_enhance.py tests/test_multi_doc_enhanced.py -q && uv run pytest tests/ -q`
Expected: PASS；全套件 530 passed/3 env 不回归

- [ ] **Step 5: 提交**

```bash
git add pageindex_mutil/agentic/enhance.py pageindex_mutil/agentic/evidence.py pageindex_mutil/agentic/router.py tests/test_evidence_bundle.py tests/test_multi_doc_enhanced.py
git commit -m "refactor(P2): 证据束返回 (bundle, ctx)——query tokens/实体名一次计算全链引用，抽 _flatten_entity_names"
```

---

### Task 2: 删除 prefilter，候选集从证据束派生

**Covers:** [S5]（prefilter 改造为证据束）、[S11]#1（HyDE 前置调用移除）

**Files:**
- Modify: `pageindex_mutil/super_tree.py:217-279`（删 `prefilter`）
- Modify: `pageindex_mutil/agentic/router.py:403-424, 437-455, 470-477`（`_search_super_tree` 候选源切换 + HyDE 移除 + ctx 透传槽位）
- Test: `tests/test_super_tree.py`、`tests/test_router.py`、`tests/test_multi_doc_enhanced.py`、`tests/test_multi_hop.py`、`tests/test_client_integration.py`

**Interfaces:**
- Consumes: `build_evidence_bundle`/`derive_evidence_score`（T1）
- Produces: `_search_super_tree` 不再调用 `super_tree_index.prefilter`；候选集 = `{db_id: derive_evidence_score(entry) for db_id, entry in bundle.items()}`；`_act_tree_search` 新增 `evidence_ctx` 形参（本任务只留槽位，T3 接线）

- [ ] **Step 1: 删除 prefilter + 写失败测试（候选集契约）**

`super_tree.py`：删除 `prefilter` 方法（:219-279）及 `:217` 的 "L0: Dual-channel prefilter" 注释块；`:136` 类 docstring 中 "L0 dual-channel prefilter" 改为 "L0 evidence bundle（见 agentic/evidence.py）"。

`tests/test_router.py` 新增（替换原 `test_prefilter_returns_empty` 语义）：

```python
class TestSearchSuperTree:
    @pytest.mark.asyncio
    async def test_empty_bundle_returns_graceful_empty(self, router):
        """L0 证据束空 → 候选集空 → 优雅空响应（替代 prefilter 空语义）。"""
        mock_st = MagicMock()
        mock_st.select_documents = AsyncMock(return_value=([], {}))
        router.super_tree_index = mock_st
        # client.db = None（mock_client fixture 已置 None）→ bundle 不构建，候选集空
        result = await router._search_super_tree("test query", top_k=3)
        assert result["answer"] == "No relevant documents found."
        assert result["confidence"] == "low"

    @pytest.mark.asyncio
    async def test_candidate_set_derived_from_bundle(self, router):
        """候选集 = bundle.keys() 派生（derive_evidence_score 作标量），不再走 prefilter。"""
        from pageindex_mutil.agentic import evidence as _ev
        mock_st = MagicMock()
        mock_st.select_documents = AsyncMock(return_value=([], {}))
        router.super_tree_index = mock_st
        router.client.db = MagicMock()
        router.client._id_mapper = None
        router.client._uuid_to_db = {"u1": 1, "u2": 2}
        bundle = {
            1: {"channels": {"keyword": [{"token": "a"}], "tag": [], "entity": [], "vector": []}, "graph": {}},
            2: {"channels": {"keyword": [], "tag": [{"text": "t"}], "entity": [], "vector": []}, "graph": {}},
        }
        with patch.object(_ev, "build_evidence_bundle", return_value=(bundle, {"tokens": ["a"], "query_entities": []})):
            result = await router._search_super_tree("test query", top_k=3)
        # 证据分：doc1=1*1 keyword=1.0，doc2=2*1 tag=2.0
        mock_st.select_documents.assert_awaited_once_with("test query", {1: 1.0, 2: 2.0}, evidence_bundle=bundle)
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_router.py::TestSearchSuperTree::test_candidate_set_derived_from_bundle -q`
Expected: FAIL（`_search_super_tree` 仍调 `prefilter`，`select_documents` 断言不符）

- [ ] **Step 3: 实现 `_search_super_tree` 改造**

替换 router.py `_search_super_tree` 顶部（:401-455 区间）为：

```python
        logging.info("[SuperTree] query=%r top_k=%d", query, top_k)

        # L0 = 证据束（[S5] prefilter 改造）：build_evidence_bundle 是唯一召回源，
        # query tokens/entities/图谱距离一次计算全链引用。候选集 = bundle.keys()
        # 派生（derive_evidence_score 作排序标量），替代独立 prefilter 四通道打分。
        # HyDE 前置调用随 prefilter 一并移除（[S11]#1 / P2 审查 FOLLOWUP④）。
        from .evidence import build_evidence_bundle, derive_evidence_score
        db = getattr(self.client, "db", None)
        bundle: dict = {}
        evidence_ctx = None
        if db is not None:
            try:
                from ..utils import ConfigLoader
                cfg = ConfigLoader().load(None)
                bundle, evidence_ctx = build_evidence_bundle(
                    self.client, db, query,
                    topk=getattr(cfg, "l0_channel_topk", 30),
                )
            except Exception as e:
                logging.warning("[SuperTree] evidence bundle build failed: %s", e)

        candidate_db_ids = {
            db_id: derive_evidence_score(entry)
            for db_id, entry in bundle.items()
        }
        logging.info("[SuperTree] L0 candidates=%d", len(candidate_db_ids))

        if not candidate_db_ids:
            return {
                "query": query,
                "mode": "multi",
                "answer": "No relevant documents found.",
                "confidence": "low",
                "matched_docs": [],
                "selected_nodes": [],
                "pages": [],
            }

        # L1: Super-Tree LLM selection——证据束直通（[S6]#2）。
        selected_uuids, l1_reasons = await self.super_tree_index.select_documents(
            query, candidate_db_ids, evidence_bundle=bundle)
```

并在 act_kwargs 组装处（:470-477）追加 ctx 槽位：

```python
        if evidence_ctx:
            act_kwargs["evidence_ctx"] = evidence_ctx
```

**同时给 `_act_tree_search` 补 `evidence_ctx` 形参**（本任务只留槽位、不消费；T3 才在方法体内提取 ctx_qe/ctx_qt 并下传。否则 `**act_kwargs` 传入会抛 TypeError）：

```python
    async def _act_tree_search(
        self, query: str, candidate_docs: List[str],
        node_matches: Dict[str, List[Dict]] = None,
        doc_scores_out: Dict[str, float] = None,
        l1_reasons: Dict[str, str] = None,
        evidence_bundle: Dict = None,
        evidence_ctx: Dict = None,
    ) -> Tuple[str, List[dict], int, int, Dict[str, List[int]], List[dict]]:
```

（删除原 :403-424 的 HyDE/prefilter 段，删除原 :440-452 重复的 bundle 构建段——已上移到 L0。）

- [ ] **Step 4: 迁移既有测试 + 运行确认通过**

- `tests/test_super_tree.py`：删除 `test_prefilter_empty_db`、`test_prefilter_with_keyword_match`（:187-200，prefilter 已不存在）。
- `tests/test_router.py`：其余 mock `mock_st.prefilter.return_value = {...}` / `mock_st.prefilter.side_effect = ...` 的用例（:113-263）——prefilter mock 全部删除；需要候选集的用例按 `test_candidate_set_derived_from_bundle` 模式 patch `_ev.build_evidence_bundle` 并给 `router.client.db = MagicMock()`。`test_super_tree_failure_returns_graceful_empty`（:260）的 `prefilter.side_effect` 改为 `select_documents = AsyncMock(side_effect=RuntimeError(...))` 或 `build_evidence_bundle.side_effect`。
- `tests/test_multi_doc_enhanced.py`：4 处 `mock_st.prefilter.return_value = {1: 1.0}`（:408/:442/:460/:708）改为 patch `_evidence_mod().build_evidence_bundle` 返回 `({1: {...}}, {"tokens": [], "query_entities": []})`（`_evidence_mod` 该文件已有 helper，:423 同款）。
- `tests/test_multi_hop.py:121`：`client.super_tree_index.prefilter.return_value = {}` 改为 `client.super_tree_index.select_documents = AsyncMock(return_value=([], {}))`（该测试后续 :122 已 mock select_documents，prefilter mock 直接删除即可）。
- `tests/test_client_integration.py`：`test_single_doc_lexical_miss_returns_prefilter_empty`（:244）的断言文案 "No relevant documents found in prefilter." → "No relevant documents found."；docstring "L0 prefilter 空" 改为 "L0 证据束空"。

Run: `uv run pytest tests/test_super_tree.py tests/test_router.py tests/test_multi_doc_enhanced.py tests/test_multi_hop.py tests/test_client_integration.py -q && uv run pytest tests/ -q`
Expected: PASS；全套件 530 passed/3 env 不回归（删 2 个 prefilter 用例、增 2 个候选集用例，计数基本持平）

- [ ] **Step 5: 提交**

```bash
git add pageindex_mutil/super_tree.py pageindex_mutil/agentic/router.py tests/test_super_tree.py tests/test_router.py tests/test_multi_doc_enhanced.py tests/test_multi_hop.py tests/test_client_integration.py
git commit -m "refactor(P2): 删除 prefilter——候选集从证据束派生，HyDE 前置调用移除（[S5]/[S11]#1）"
```

---

### Task 3: L2 复用 ctx（query_tokens / query_entities 直通 enhance_and_select）

**Covers:** [S7]（L2 证据来源切换为证据束/共享 query 物）

**Files:**
- Modify: `pageindex_mutil/agentic/enhance.py:345-398`（`enhance_and_select` 增 `query_tokens` 参数）
- Modify: `pageindex_mutil/agentic/router.py:195-236`（`_act_tree_search` 增 `evidence_ctx`）、`:53-142`（`_recall_nodes_for_doc` 增 `query_entities`/`query_tokens`）
- Test: `tests/test_multi_doc_enhanced.py`、`tests/test_unified_enhance.py`

**Interfaces:**
- Consumes: `ctx`（T1）、`evidence_ctx` 槽位（T2）
- Produces:
  - `_act_tree_search(..., evidence_ctx: dict = None)`（形参已在 T2 添加）：本任务在方法体内从 `evidence_ctx["query_entities"]`/`evidence_ctx["tokens"]` 提取并透传 `_recall_nodes_for_doc`
  - `_recall_nodes_for_doc(..., query_entities=None, query_tokens=None)`：None 时回退现有解析（`resolve_query_entities` / 内部 tokenize），非 None 时直接使用
  - `enhance_and_select(..., query_tokens=None)`：None 时 `self._tokenize(query)`，非 None 时复用

- [ ] **Step 1: 写失败测试**

```python
# tests/test_unified_enhance.py 追加
def test_enhance_reuses_provided_query_tokens(monkeypatch):
    """义务：query_tokens 非 None 时不再内部 tokenize（复用 L0 共享物）。"""
    from pageindex_mutil.agentic.enhance import UnifiedNodeEnhancement
    enh = UnifiedNodeEnhancement("m", retrieve_model="r")
    calls = {"n": 0}
    real = UnifiedNodeEnhancement._tokenize

    def spy(cls, text):
        calls["n"] += 1
        return real(cls, text)

    monkeypatch.setattr(UnifiedNodeEnhancement, "_tokenize", classmethod(spy))
    # 直接调 enhance_and_select（LLM 未调时走 union 空/全量路径不依赖真实 LLM）：
    import asyncio
    asyncio.run(enh.enhance_and_select(
        "q", [{"node_id": "n1", "title": "t", "summary": "s", "text": "浴血"}],
        {}, query_tokens=["浴血"]))
    assert calls["n"] == 0  # 提供 tokens 后零内部 tokenize
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_unified_enhance.py::test_enhance_reuses_provided_query_tokens -q`
Expected: FAIL（`enhance_and_select` 无 `query_tokens` 参数 → TypeError）

- [ ] **Step 3: 实现**

`enhance.py` — `enhance_and_select` 签名追加 `query_tokens=None`，tokenize 行改为：

```python
        query_tokens = query_tokens if query_tokens is not None else self._tokenize(query)
```

（替换原 `:398` 的 `query_tokens = self._tokenize(query)`。）

`router.py` — `_act_tree_search` 签名追加 `evidence_ctx: Dict = None`，在并发派发循环（:229-236）前提取：

```python
        ctx_qe = evidence_ctx.get("query_entities") if evidence_ctx else None
        ctx_qt = evidence_ctx.get("tokens") if evidence_ctx else None
```

并把 `call_kwargs` 组装追加：

```python
            if ctx_qe is not None:
                call_kwargs["query_entities"] = ctx_qe
            if ctx_qt is not None:
                call_kwargs["query_tokens"] = ctx_qt
```

`router.py` — `_recall_nodes_for_doc` 签名追加 `query_entities=None, query_tokens=None`；替换 `:111` 的：

```python
        query_entities = resolve_query_entities(db, query, limit=5) if db else []
```

为：

```python
        if query_entities is None:
            query_entities = resolve_query_entities(db, query, limit=5) if db else []
```

并在 `enhance_and_select` 调用（:140-142）追加 `query_tokens=query_tokens`（仅当非 None，用 conditional-kwarg 保旧签名测试替身兼容）：

```python
        call_kwargs = {}
        if l1_reasons:
            reason = l1_reasons.get(doc_id)
            if reason:
                call_kwargs["l1_reasons"] = {doc_id: reason}
        if query_tokens is not None:
            call_kwargs["query_tokens"] = query_tokens
        result = await enhancer.enhance_and_select(
            query, candidates, profiles, query_entities=query_entities, **call_kwargs
        )
```

- [ ] **Step 4: 迁移既有测试 + 运行确认通过**

- `tests/test_multi_doc_enhanced.py`：`test_matched_scores_are_evidence_score`（:402）在 `_search_super_tree` 调用后补一条断言——`_recall_nodes_for_doc` 收到的 `query_entities` 来自 ctx（用一个记录式 `enhance_and_select` 替身捕获实参，或复用 :296 的 `fake_select` 记录模式）。`fake_select` 若按位置参数接收，需兼容新增 `query_tokens` kwarg（用 `**kw` 吸收）。
- `tests/test_unified_chain.py`：全链真实跑（不 mock bundle），确认 `_index_doc` 写入的 doc_keywords 经 `match_doc_keywords` 命中（候选源切换后仍召回）——本任务只补跑确认，不改测试体。

Run: `uv run pytest tests/test_unified_enhance.py tests/test_multi_doc_enhanced.py tests/test_unified_chain.py -q && uv run pytest tests/ -q`
Expected: PASS；全套件 530 passed/3 env 不回归

- [ ] **Step 5: 提交**

```bash
git add pageindex_mutil/agentic/enhance.py pageindex_mutil/agentic/router.py tests/test_unified_enhance.py tests/test_multi_doc_enhanced.py
git commit -m "refactor(P2): L2 复用证据束 query 物——query_tokens/query_entities 直通，消除重复 tokenize/search_entities（[S7]）"
```

---

## 计划自审记录（按 compose:plan Self-Review 执行）

1. **Spec 覆盖**：[S5]（prefilter 改造=T2、query 物一次计算=T1）— [S7]（L2 证据来源切换=T3）— [S11]#1（HyDE 前置移除=T2）— [S6]（L1 消费 bundle 不变，无需任务）。无缺口。
2. **占位符**：无 TBD/TODO；每个 code step 均有完整代码或精确 diff 位置。
3. **类型一致性**：`build_evidence_bundle -> (bundle, ctx)`（T1 定义）→ `ctx["tokens"]/["query_entities"]`（T3 消费）形状一致；`evidence_ctx`（T2 留槽位）→ `_act_tree_search`（T3 接线）；`query_tokens`（T3 `enhance_and_select` 形参）→ `_recall_nodes_for_doc`（T3 透传）；`_flatten_entity_names`（T1 定义）→ `resolve_query_entities` + `build_evidence_bundle`（T1 复用）。
4. **同源已核实（成员集合等价）**：候选源从 `prefilter`（keyword 通道 = `KeywordIndex.search` = `db.match_doc_keywords`，super_tree.py:52-56）切到 bundle（keyword 通道同样 `db.match_doc_keywords`，evidence.py:51）——同源、非另起数据源。**注意排序标量变化**：候选集分数从 prefilter 浮点分累加变为 `derive_evidence_score` 加权计数（score→count），这是 [S5]/[S6]#8 本意，非"行为等价"。
5. **审查焦点提示**：①HyDE 移除是行为变化（[S11]#1 本就要求，且删后 planner 零生产消费者，multi_hop 走自身 `_judge_decomposable` 不依赖 planner），评审需确认可接受；②`ctx.query_entities` 用 bundle 的 `search_entities(limit=topk)` 全量展平，替代 L2 原 `limit=5`——实体名单变长（6×），属有意的"一次计算"统一，评审需确认实体通道匹配宽度可接受；③**遗留不一致（已知）**：`evidence_ctx=None` 时 `_recall_nodes_for_doc` 回退 `resolve_query_entities(limit=5)`，与 ctx 路径（30）宽度不一致——实施后观察实体通道噪声，若上升再统一两路径。

---

## 目标架构补盲区任务（T4/T5，用户 2026-08-14 指令"实现目标架构"，依赖 T1-T3 完成）

> 定位：T1-T3 收敛了 query 级共享物（消除重复计算）。但"证据链加强结构树、补 LLM 知识边界盲区"还差两块——①L2 正文命中只渲染裸 token、无上下文（判空根因）；②节点级实体（entity_mentions.node_id）被 bundle 压成 doc 级、L2 重读 node_profiles。T5 优先（直接补盲区），T4 次之。

### Task 5: L2 正文命中上下文直通（补知识边界盲区）

**Covers:** [S7]（证据来源切换为证据束，正文内容通道 P2.6 从"准入"补到"裁决呈现"）、[S1] 丢档点①（找到它的信号=决策者看到的信号）

**Files:**
- Modify: `pageindex_mutil/agentic/enhance.py`（`_content_hit_contexts` 新助手 + `enhance_and_select` 落上下文 + `_assemble_evidence` 渲染）
- Test: `tests/test_unified_enhance.py`

**Interfaces:**
- `_content_hit_contexts(query_tokens: list, text: str, window: int = 60, max_hits: int = 2) -> list[str]`：对前 `max_hits` 个命中 query token，取其在 `text` 首次出现的 ±window 字符窗口（casefold 定位，首尾加省略号）；无命中返回 []。
- `enhance_and_select`：内容通道命中时，`node_signals[nid]["content_contexts"] = _content_hit_contexts(query_tokens, text)`；`_assemble_evidence` 在节点块渲染 `正文命中: 「...上下文...」`（沿用 [7.4] 单节点封顶：≤2 条、每条 ≤120 字符）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_unified_enhance.py 追加
def test_content_hit_contexts_extracts_window():
    from pageindex_mutil.agentic.enhance import UnifiedNodeEnhancement
    text = "浴血值可以通过完成日常任务获得，是帮会系统中重要的成长数值。"
    ctxs = UnifiedNodeEnhancement._content_hit_contexts(["浴血"], text)
    assert ctxs and "浴血值" in ctxs[0]  # 命中处上下文，而非裸 token

def test_evidence_renders_content_context():
    """正文命中 → 节点证据块含命中上下文（非只裸 token），补 LLM 知识盲区。"""
    from pageindex_mutil.agentic.enhance import UnifiedNodeEnhancement
    enh = UnifiedNodeEnhancement("m", retrieve_model="r")
    text = "浴血值可以通过完成日常任务获得。"
    cand = {"node_id": "n1", "title": "t", "summary": "s", "text": text}
    # 直接构造 node_signals 模拟 content 通道命中后断言证据渲染含上下文
    signals = {"n1": {"entities": [], "keywords": [], "tags": [], "score": 0, "pos": 0,
                      "content_contexts": ["浴血值可以通过完成日常任务获得"]}}
    evidence = enh._assemble_evidence(["n1"], signals, {"n1": cand})
    assert "浴血值可以通过完成日常任务获得" in evidence
```

- [ ] **Step 2: 运行确认失败** — `uv run pytest tests/test_unified_enhance.py::test_content_hit_contexts_extracts_window -q`，Expected FAIL（无 `_content_hit_contexts`）。
- [ ] **Step 3: 实现**

```python
# enhance.py 新增 @staticmethod：
    @staticmethod
    def _content_hit_contexts(query_tokens, text, window=60, max_hits=2):
        """命中处上下文：query token 命中正文时取 ±window 字符窗口（casefold 定位，
        首尾省略号），供 L2 证据渲染补知识盲区。防御：非字符串/空 → []。"""
        if not isinstance(text, str) or not text or not query_tokens:
            return []
        t = text.casefold()
        out = []
        for qt in query_tokens:
            if len(out) >= max_hits:
                break
            if len(qt) < 2:
                continue
            i = t.find(qt)
            if i < 0:
                continue
            lo = max(0, i - window)
            hi = min(len(text), i + len(qt) + window)
            s = text[lo:hi]
            prefix = "…" if lo > 0 else ""
            suffix = "…" if hi < len(text) else ""
            out.append(f"{prefix}{s}{suffix}")
        return out
```

`enhance_and_select` 内容通道命中处（body_hits 之后、node_signals 组装处）落上下文：

```python
            content_contexts = self._content_hit_contexts(query_tokens, cand.get("text") or "") if body_hits else []
            node_signals[nid] = {
                "entities": ents, "keywords": kws, "tags": tags,
                "score": score, "pos": pos,
                "content_contexts": content_contexts,
            }
```

`_assemble_evidence` 节点块（:252-262 内）在关键词行后追加：

```python
            if m.get("content_contexts"):
                lines.append("正文命中: " + "；".join(
                    c[:120] for c in m["content_contexts"][:2]
                ))
```

（`matches[nid]` 透传须精确插在 `_assemble_evidence` 的 matches 构建循环（enhance.py:186-192，`matches[nid] = {"entities":..., "keywords":..., "tags":...}` 处）加 `m["content_contexts"] = list(sig.get("content_contexts") or [])`——**不是** blocks 渲染循环（:246）。全函数仅此一处 matches 构建，无遗漏透传点。）

> casefold 定位 note（评审）：`t.find(qt)` 中 `qt` 未 casefold，与既有 `_content_hits`（enhance.py:142 `qt in t`）同款取舍——query token 已 jieba 归一、CJK 无大小写，本语料安全；casefold 长度变化字符（ß→ss）与 `_content_hits` 同源同 miss/hit，无错位窗口风险。不单独改，保持与准入通道一致。

- [ ] **Step 4: 运行确认通过** — `uv run pytest tests/test_unified_enhance.py tests/test_multi_doc_enhanced.py tests/test_unified_chain.py -q && uv run pytest tests/ -q`，Expected: PASS；531 passed/3 env 不回归。
- [ ] **Step 5: 提交** — `git add pageindex_mutil/agentic/enhance.py tests/test_unified_enhance.py && git commit -m "feat(P2): L2 正文命中上下文直通——证据渲染命中处上下文补 LLM 知识盲区（[S7]/[S1]①）"`

---

### Task 4: 证据束 entity 通道携带 node_id（节点级实体直通 L2）

**Covers:** [S7]（L2 证据来源切换为证据束，实体路径落地）、[S5]（query 级缓存实体维度补全）

**Files:**
- Modify: `pageindex_mutil/agentic/evidence.py`（entity 通道 SELECT 加 `em.node_id`，实体条目带 node_id）
- Modify: `pageindex_mutil/agentic/router.py`（`_act_tree_search` 从 bundle 提取 doc 级 node-entity map 并透传；`_recall_nodes_for_doc` 消费）
- Modify: `pageindex_mutil/agentic/enhance.py`（`enhance_and_select` 增 `node_entities` 参数）
- Test: `tests/test_evidence_bundle.py`、`tests/test_multi_doc_enhanced.py`

**Interfaces:**
- bundle entity 条目增 `node_id` 字段（可空）：`{"name","type","confidence","node_id"}`（evidence.py:122-128 的 SELECT 加 `em.node_id`、append 加 `"node_id": r["node_id"]`）。
- **去重策略（Critical，评审钉死）**：bundle entity 通道去重 key 从 `name` 改为 `(name, node_id)`（evidence.py:154 `_dedup`）——保多节点归属，否则同名多节点实体被折叠成单条、node→entity map 丢失归属，T4 退化成不如 node_profiles（其本就完整逐节点归属）。**但** `derive_evidence_score`（证据分，evidence.py:15-22）与 `render_doc_evidence`（L1 渲染，:172）必须按 `name` 去重计数，否则多节点同实体重复抬高候选排序与 matched_docs（违 [7.4] 防过载）。加"同实体多节点"测试钉死两者口径。
- **时序 hoist（评审 note）**：`_act_tree_search` 聚合 node→entity map 必须在 recall 派发循环（router.py:248-261）**之前**完成 db_id→uuid 映射——`client._id_mapper`/`_uuid_to_db` 全程可用，提前重建 + 反向 db→uuid，不得依赖排序段（:278-284）才建立的映射。
- `enhance_and_select(..., node_entities: dict = None)`：`{node_id: [{"name","type","confidence"}]}`，非 None 时作为该文档节点级实体证据直接注入（替代 `resolve_node_profiles` 的实体部分；keywords/tags 仍走 profiles）。
- `_act_tree_search`：从 `evidence_bundle` 按 doc 聚合 `{node_id: [entity...]}`（db_id→uuid 映射后）透传给 `_recall_nodes_for_doc`；`_recall_nodes_for_doc` 增 `node_entities` 参数，非 None 时传给 `enhance_and_select`。

- [ ] **Step 1: 写失败测试**（`tests/test_evidence_bundle.py` 追加：构造 entity_mentions 带 node_id，断言 bundle entity 条目含 `node_id`；`tests/test_multi_doc_enhanced.py` 追加：`enhance_and_select` 收到 `node_entities` 后 `_entity_hits` 直接用注入的节点级实体）。
- [ ] **Step 2: 运行确认失败** — 断言 bundle 条目无 node_id / enhance_and_select 无 node_entities 参数。
- [ ] **Step 3: 实现**（按 Interfaces 逐条落地，最小改动）。
- [ ] **Step 4: 运行确认通过** — `uv run pytest tests/test_evidence_bundle.py tests/test_multi_doc_enhanced.py -q && uv run pytest tests/ -q`，Expected: PASS；531 passed/3 env 不回归。
- [ ] **Step 5: 提交** — `git add pageindex_mutil/agentic/evidence.py pageindex_mutil/agentic/router.py pageindex_mutil/agentic/enhance.py tests/test_evidence_bundle.py tests/test_multi_doc_enhanced.py && git commit -m "feat(P2): 节点级实体直通——entity 通道带 node_id，L2 复用替代 node_profiles 实体重读（[S7]/[S5]）"`

**边界**：BM25（`doc_keywords`）保持 doc 级不动；图谱关系边 `entity_relations` 实体级不落节点，不进本任务。启动前先评测 T1-T3 后的实体通道噪声与 R@k。
