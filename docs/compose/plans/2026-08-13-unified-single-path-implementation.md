# 统一体·单链版 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent (recommended) or compose:execute to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 PageIndex-UV 检索推理收敛为「索引/推理」两阶段单链：L0 证据束直通 L1、裁定阶段扩大重试、统一 span locator，并按 spec 删除清单清理路径分支与孤儿代码。

**Architecture:** 索引期产出结构树+签名+doc_keywords+实体图+doc_summary；推理期单链 = 证据束构建（四通道原始命中+图谱 CTE）→ L1 文档裁定（候选=1 短路 / 分组 map-reduce / doc_summary+证据块呈现）→ 节点裁定 enhance_and_select（L1 理由 trace）→ 上下文组装（span 分派）→ 作答 → CRAG 验证（answer / expand→LLM 点名补召回 / refuse）。决策契约 = 只直通不约束。

**Tech Stack:** Python 3.12+ / uv / SQLite（递归 CTE）/ jieba / OpenAI SDK。评测 harness 在 pageindex-paper（SEARCH_BACKEND=keyword，n=10 三数据集）。

## Global Constraints

- 测试用 `uv run pytest <path> -q`（pytest-timeout 未安装，**禁止 --timeout**）；全套件基线 614 passed / 3 env（3 个 TestChromaSearchBackend 失败为沙箱无网既有环境失败，**不得归因新代码**）
- NFR4：每个新 LLM 调用点必须 `retrieve_model or self.model`，由 tests/test_retrieve_model_wiring.py 强制
- Mock-LLM 测试按 prompt 子串标记路由（tests/test_corpus_tree.py `_route_llm`）；新 LLM 调用点须有独特 prompt 标记，且需检查 test_grouping_issues_no_llm_calls 的 allowlist
- 提交风格：`feat(P2): 描述` / `fix(P2): 描述`；`git add` 指定文件（禁 `git add -A`）；**NEVER `git config`**
- D2 轻量：删除即删净（不留兼容 shim/`_legacy` 残根）、不过度抽象
- 决策契约：只直通不约束——任何任务不得引入保底席位/规则回退重排
- P2 按删除项分步提交，每步全套测试；评测跑完后隔离旧结果再重启 runner
- 评测判据：mldr R@10 ≥ 0.5 收尾 / < 0.4 重开契约（仅 P2 完成后在 pageindex-paper 执行，不在本 repo）

---

## File Structure

- **Create** `pageindex_mutil/agentic/evidence.py` — 证据束构建/渲染/证据分（T2/T3/T9）
- **Modify** `db.py` — `get_entity_distances_cte`（T1）、线程本地连接复用（T5）
- **Modify** `pageindex_mutil/super_tree.py` — prefilter 输出证据束（T2）、L1 裁定改造（T9）、删除项（T13）
- **Modify** `pageindex_mutil/agentic/enhance.py` — `l1_reasons` 段（T4）
- **Modify** `pageindex_mutil/agentic/verifier.py` — 上下文预算 + need 输出（T10）
- **Modify** `pageindex_mutil/agentic/recall_loop.py` — 滑窗删除 + 点名扩召（T11）
- **Modify** `pageindex_mutil/agentic/router.py` — 单链入口（T8）、删除项（T13）
- **Modify** `pageindex_mutil/client.py` — 去分支（T8）、doc_summary 生成/落库（T12）、span（T7）
- **Modify** `pageindex_mutil/page_index_md.py` — `end_line` 补存（T7）
- **Modify** `pageindex_mutil/reasoning.py` — `spans_from_nodes`（T7）、死函数删除（T13）
- **Modify** `pageindex_mutil/config.yaml` — 键迁移（T6/T13）
- **Delete** `pageindex_mutil/agentic/strategies.py`（T13）、`pageindex_mutil/corpus_tree.py` 聚类构建部分（T12）

---

### Task 1: 图谱递归 CTE

**Covers:** [S9]

**Files:**
- Modify: `db.py`（新增方法，加在 `get_entity_relations` 之后）
- Test: `tests/test_entity_distances_cte.py`（新建）

**Interfaces:**
- Consumes: 现有 `entity_relations`/`entities` 表（索引已备 db.py:216-218）
- Produces: `PageIndexDB.get_entity_distances_cte(query_entity_ids: list[int], max_hop: int = 3) -> dict[int, dict]`——与 `_precompute_entity_distances` 同形状：`{entity_id: {"distance": int, "relation_type": str, "weight": float, "name": str}}`。取数规则：hop-min 优先；同 hop 取最大 weight；再平取最小 entity_id。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_entity_distances_cte.py
import pytest
from db import PageIndexDB

@pytest.fixture()
def graph_db(tmp_path):
    db = PageIndexDB(str(tmp_path / "t.db"))
    with db._connect() as conn:
        conn.executemany("INSERT INTO entities (id, name, entity_type, doc_count) VALUES (?,?,?,1)",
                         [(1, "浴血值", "concept"), (2, "帮会系统", "concept"),
                          (3, "门派介绍", "section"), (4, "远亲", "concept")])
        conn.executemany("INSERT INTO entity_relations (subject_id, predicate, object_id, doc_id, confidence) VALUES (?,?,?,?,0.9)",
                         [(1, "related_to", 2, 1), (2, "part_of", 3, 1), (3, "related_to", 4, 1)])
    return db

def test_cte_matches_bfs_semantics(graph_db):
    got = graph_db.get_entity_distances_cte([1], max_hop=3)
    # hop-min + 距离衰减 0.7 × related_to 0.6 = 0.42
    assert got[2]["distance"] == 1
    assert abs(got[2]["weight"] - 0.42) < 1e-6
    assert got[2]["relation_type"] == "related_to"
    assert got[3]["distance"] == 2
    assert got[3]["name"] == "门派介绍"
    # 权重语义 = 距离衰减(hop)×末边关系权重，非路径乘积：0.4 × part_of 0.8 = 0.32
    assert abs(got[3]["weight"] - 0.32) < 1e-6
    assert got[4]["distance"] == 3

def test_cte_tiebreak_same_hop_max_weight(graph_db):
    # 两个并行的 1-hop 边：causal(1.0) 与 related_to(0.6)
    with graph_db._connect() as conn:
        conn.execute("INSERT INTO entities (id, name, entity_type, doc_count) VALUES (5,'甲','concept',1)")
        conn.execute("INSERT INTO entity_relations (subject_id, predicate, object_id, doc_id, confidence) VALUES (1,'causal',5,1,0.9)")
    got = graph_db.get_entity_distances_cte([1], max_hop=3)
    assert got[2]["relation_type"] == "related_to"  # 同 hop 下 related_to 权重 0.42 < causal 0.7，但距离已是 1（hop-min 由距离决定，不因新边改变）
    # 同 hop 不同路径时取最大 weight：
    assert got[5]["distance"] == 1 and abs(got[5]["weight"] - 0.7) < 1e-6

def test_cte_self_excluded(graph_db):
    got = graph_db.get_entity_distances_cte([1], max_hop=3)
    assert 1 not in got
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_entity_distances_cte.py -q`
Expected: FAIL（`AttributeError: 'PageIndexDB' object has no attribute 'get_entity_distances_cte'`）

- [ ] **Step 3: 最小实现**

```python
# db.py，类 PageIndexDB 内（get_entity_relations 之后）
from collections import defaultdict

def get_entity_distances_cte(self, query_entity_ids, max_hop=3):
    """单次递归 CTE 无向 BFS。权重语义与 _precompute_entity_distances 逐项一致：
    weight = 距离衰减(hop) × 末边关系类型权重（非路径乘积）。聚合：hop-min 优先；
    同 hop 取最大 weight；再平取最小 entity_id。path 列防环。"""
    if not query_entity_ids:
        return {}
    placeholders = ",".join("?" for _ in query_entity_ids)
    conn = self._connect()
    sql = f"""
    WITH RECURSIVE walk(nid, hop, weight, relation_type, name, path) AS (
        SELECT e.id, 0, 1.0, 'direct', e.name, ',' || e.id || ','
        FROM entities e WHERE e.id IN ({placeholders})
        UNION ALL
        SELECT nxt.id,
               w.hop + 1,
               CASE w.hop + 1
                   WHEN 1 THEN 0.7 WHEN 2 THEN 0.4 WHEN 3 THEN 0.2 ELSE 0.1 END
                 * CASE er.predicate
                   WHEN 'causal' THEN 1.0 WHEN 'causes' THEN 1.0 WHEN 'effect' THEN 1.0
                   WHEN 'part_of' THEN 0.8 WHEN 'contains' THEN 0.8
                   WHEN 'has_part' THEN 0.8 WHEN 'belongs_to' THEN 0.8
                   WHEN 'related_to' THEN 0.6 WHEN 'associated' THEN 0.6
                   WHEN 'similar' THEN 0.6
                   ELSE 0.4 END,
               er.predicate, nxt.name,
               w.path || nxt.id || ','
        FROM walk w
        JOIN entity_relations er ON er.subject_id = w.nid OR er.object_id = w.nid
        JOIN entities nxt ON nxt.id =
            CASE WHEN er.subject_id = w.nid THEN er.object_id ELSE er.subject_id END
        WHERE w.hop < {int(max_hop)}
          AND instr(w.path, ',' ||
            (CASE WHEN er.subject_id = w.nid THEN er.object_id ELSE er.subject_id END) || ',') = 0
    )
    SELECT nid, hop, weight, relation_type, name
    FROM walk WHERE nid NOT IN ({placeholders})
    """
    rows = conn.execute(sql, (*query_entity_ids,)).fetchall()
    by_entity = defaultdict(list)
    for r in rows:
        by_entity[r["nid"]].append(r)
    out = {}
    for nid, rs in by_entity.items():
        best = min(rs, key=lambda r: (r["hop"], -r["weight"], r["nid"]))
        out[nid] = {"distance": best["hop"], "relation_type": best["relation_type"],
                    "weight": best["weight"], "name": best["name"]}
    return out
```

> P1 验收另含与 `_precompute_entity_distances` 的逐项对照测试（同一测试图上两者输出一致；不一致处按 BFS 现行语义对齐并记录差异）。

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_entity_distances_cte.py -q`
Expected: PASS（2-3 passed）

- [ ] **Step 5: 提交**

```bash
git add db.py tests/test_entity_distances_cte.py
git commit -m "feat(P1): 实体距离递归 CTE——单次 SQL 无向 BFS 替代 Python N+1 遍历"
```

---

### Task 2: 证据束构建与 prefilter 输出改造

**Covers:** [S5]

**Files:**
- Create: `pageindex_mutil/agentic/evidence.py`
- Modify: `pageindex_mutil/keyword_backend.py`（`is_vector = False`）、`pageindex_mutil/hybrid_backend.py` + `pageindex_mutil/chroma_backend.py`（`is_vector = True`）
- Test: `tests/test_evidence_bundle.py`（新建）

**Interfaces:**
- Consumes: `db.match_doc_keywords`（T 前置已存在）、`db.get_entity_distances_cte`（T1）、`closet_index.search`、`search_backend.search`
- Produces:
  - `build_evidence_bundle(client, db, query, topk=30) -> dict`：`{db_id(int): {"channels": {"tag": [{"text","confidence"}], "keyword": [{"token","field","bm25_score"}], "entity": [{"name","type","confidence"}], "vector": [...]}, "graph": {"doc_entity_links": [{"entity","distance","relation_type","weight"}]}}}`
  - `derive_evidence_score(bundle_entry) -> float`：`3*len(entities) + 2*len(tags) + 1*len(keywords)`（与 enhance 多信号加权同系数）
  - `render_doc_evidence(query, bundle, db_ids) -> str`：L1 证据块文本

- [ ] **Step 1: 写失败测试**

```python
# tests/test_evidence_bundle.py
import pytest
from pageindex_mutil.agentic.evidence import build_evidence_bundle, derive_evidence_score

def _make_client(tmp_path, docs, keywords, tags):
    from pageindex_mutil.client import PageIndexClient
    from db import PageIndexDB
    db = PageIndexDB(str(tmp_path / "t.db"))
    client = PageIndexClient(db_path=str(tmp_path / "t.db"))
    client.db = db
    client.closet_index = None
    client.search_backend = None
    # 写入 doc_keywords：正文命中带 field=content 来源
    for doc_id, name, desc, kws in docs:
        did = db.insert_document(pdf_name=name, pdf_path="", doc_description=desc)
        records = [(did, tok, field, tf) for tok, field, tf in kws]
        db.insert_doc_keywords(did, records)
    return client, db

def test_bundle_keyword_field_provenance(tmp_path):
    client, db = _make_client(tmp_path,
        [("A", "文档A", "", [("浴血值", "content", 3), ("帮会", "node_title", 1)]),
         ("B", "文档B", "", [("帮会", "content", 2)])], None, None)
    bundle = build_evidence_bundle(client, db, "帮会浴血值怎么获得", topk=30)
    a_kw = {(k["token"], k["field"]) for k in bundle[1]["channels"]["keyword"]}
    assert ("浴血值", "content") in a_kw
    assert ("帮会", "node_title") in a_kw  # field 来源可追溯

def test_evidence_score_multi_signal(tmp_path):
    entry = {"channels": {"keyword": [{"token": "x"}], "tag": [{"text": "y"}], "entity": [{"name": "z"}]}, "graph": {}}
    assert derive_evidence_score(entry) == 3 * 1 + 2 * 1 + 1 * 1
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_evidence_bundle.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 evidence.py**

```python
"""证据束：L0 四通道原始命中 + 图谱关联的 query 级缓存对象（spec [S5]）。"""
import logging


def _dedup(items, key):
    seen, out = set(), []
    for it in items:
        k = key(it)
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def derive_evidence_score(entry) -> float:
    """文档级证据分：3*实体 + 2*标签 + 1*关键词（仅用于分组序/补充清单序/matched_docs，不参与裁定）。"""
    ch = (entry or {}).get("channels") or {}
    return (
        3.0 * len(ch.get("entity") or [])
        + 2.0 * len(ch.get("tag") or [])
        + 1.0 * len(ch.get("keyword") or [])
    )


def build_evidence_bundle(client, db, query, topk=30) -> dict:
    """构建证据束。返回 {db_id: {"channels": {...}, "graph": {...}}}。"""
    bundle = {}

    def entry(db_id):
        return bundle.setdefault(int(db_id), {
            "channels": {"tag": [], "keyword": [], "entity": [], "vector": []},
            "graph": {"doc_entity_links": []},
        })

    # keyword 通道：先 BM25 打分，再查 doc_keywords 取 (token, field) 来源（spec [S5] 来源契约）
    from ..super_tree import KeywordIndex
    tokens = KeywordIndex._tokenize(None, query)
    if tokens:
        try:
            scored = dict(db.match_doc_keywords(tokens, top_k=topk))
        except Exception as e:
            logging.warning("evidence keyword scoring failed: %s", e)
            scored = {}
        if scored:
            tph = ",".join("?" for _ in tokens)
            dph = ",".join("?" for _ in scored)
            try:
                rows = db._connect().execute(
                    f"SELECT doc_id, keyword, field FROM doc_keywords "
                    f"WHERE keyword IN ({tph}) AND doc_id IN ({dph})",
                    (*tokens, *scored.keys())).fetchall()
                for r in rows:
                    entry(r["doc_id"])["channels"]["keyword"].append(
                        {"token": r["keyword"], "field": r["field"],
                         "bm25_score": scored.get(r["doc_id"], 0.0)})
            except Exception as e:
                logging.warning("evidence keyword provenance failed: %s", e)

    # tag 通道（closet 语义标签，source=llm）
    if getattr(client, "closet_index", None):
        try:
            for doc_id, score in client.closet_index.search(query, top_k=topk):
                entry(doc_id)["channels"]["tag"].append({"text": "", "confidence": float(score)})
        except Exception as e:
            logging.warning("evidence tag channel failed: %s", e)

    # vector 通道：仅真向量后端（hybrid/chroma，is_vector=True）；keyword no-op 后端不进（防重复计分）
    backend = getattr(client, "search_backend", None)
    if backend is not None and getattr(backend, "is_vector", False):
        try:
            for doc_id, score in backend.search(query, top_k=topk):
                entry(doc_id)["channels"]["vector"].append({"score": float(score)})
        except Exception as e:
            logging.warning("evidence vector channel failed: %s", e)

    # entity 通道 + 图谱关联（CTE）
    try:
        entities = db.search_entities(query, limit=topk)
    except Exception:
        entities = []
    query_ids = [e["id"] for e in entities if e.get("id")]
    dist_table = db.get_entity_distances_cte(query_ids, max_hop=3) if query_ids else {}
    try:
        # 实体→文档 批量 IN 查询（替代逐实体 get_entity_documents）
        if query_ids:
            placeholders = ",".join("?" for _ in query_ids)
            conn = db._connect()
            rows = conn.execute(
                f"SELECT em.entity_id, em.doc_id, em.confidence, e.name, e.entity_type "
                f"FROM entity_mentions em JOIN entities e ON e.id = em.entity_id "
                f"WHERE em.entity_id IN ({placeholders})", query_ids).fetchall()
            for r in rows:
                ent = entry(r["doc_id"])["channels"]["entity"]
                ent.append({"name": r["name"], "type": r["entity_type"], "confidence": r["confidence"]})
                info = dist_table.get(r["entity_id"])
                if info:
                    entry(r["doc_id"])["graph"]["doc_entity_links"].append(
                        {"entity": r["name"], "distance": info["distance"],
                         "relation_type": info["relation_type"], "weight": info["weight"]})
    except Exception as e:
        logging.warning("evidence entity channel failed: %s", e)

    for db_id, e in bundle.items():
        e["channels"]["keyword"] = _dedup(e["channels"]["keyword"], lambda k: (k["token"], k["field"]))
        e["channels"]["entity"] = _dedup(e["channels"]["entity"], lambda k: k["name"])
        e["channels"]["tag"] = _dedup(e["channels"]["tag"], lambda k: k["text"])
    return bundle


def render_doc_evidence(bundle, db_ids) -> str:
    """L1 证据块：结构化呈现（按通道分组）。返回文本；空命中文档返回无证据注记。"""
    lines = []
    for db_id in db_ids:
        e = bundle.get(int(db_id))
        if not e:
            continue
        ch = e["channels"]
        parts = []
        if ch["keyword"]:
            parts.append("关键词命中: " + ", ".join(k["token"] or f"content:{k['bm25_score']:.2f}" for k in ch["keyword"]))
        if ch["entity"]:
            parts.append("实体命中: " + ", ".join(f"{x['name']}（{x['type']}）" for x in ch["entity"]))
        if ch["tag"]:
            parts.append("标签命中: " + ", ".join(t["text"] for t in ch["tag"]))
        links = e["graph"].get("doc_entity_links") or []
        if links:
            parts.append("图谱关联: " + ", ".join(
                f"{l['entity']}(距离{l['distance']}·{l['relation_type']})" for l in links))
        lines.append(f"doc {db_id}: " + " | ".join(parts) if parts else f"doc {db_id}: 无通道命中")
    return "\n".join(lines)
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_evidence_bundle.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add pageindex_mutil/agentic/evidence.py tests/test_evidence_bundle.py
git commit -m "feat(P1): 证据束构建原语——四通道原始命中带来源 + 图谱 CTE 关联（spec [S5]）"
```

---

### Task 3: L1 证据束直通（P1 行为切换，不删旧路）

**Covers:** [S6]

**Files:**
- Modify: `pageindex_mutil/super_tree.py`（`_holistic_select` 证据块改用证据束；`_doc_evidence_lines` 保留为兜底）
- Test: `tests/test_evidence_bundle.py`（追加用例）

**Interfaces:**
- Consumes: `build_evidence_bundle`/`render_doc_evidence`（T2）
- Produces: `_holistic_select(query, db_ids, keep=None, evidence_bundle=None)` 新可选参数（T4/T9 沿用）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_evidence_bundle.py 追加
def test_render_doc_evidence_graph_link():
    from pageindex_mutil.agentic.evidence import render_doc_evidence
    bundle = {1: {"channels": {"keyword": [], "entity": [{"name": "浴血值", "type": "concept"}],
                               "tag": [], "vector": []},
                  "graph": {"doc_entity_links": [{"entity": "帮会系统", "distance": 1,
                                                  "relation_type": "related_to", "weight": 0.42}]}}}
    text = render_doc_evidence(bundle, [1])
    assert "实体命中: 浴血值（concept）" in text
    assert "图谱关联: 帮会系统(距离1·related_to)" in text
    assert "无通道命中" in render_doc_evidence({}, [1])  # 无证据文档降级行为显式
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_evidence_bundle.py::test_render_doc_evidence_graph_link -q`
Expected: FAIL（函数尚未定义或输出不含图谱注记）

- [ ] **Step 3: 实现**

```python
# super_tree.py _holistic_select 签名与证据块段改造：
async def _holistic_select(self, query: str, db_ids: list[int],
                           keep: int = None, evidence_bundle: dict = None) -> list[int]:
    ...
    # 证据块：优先证据束直通；缺失时回退 _doc_evidence_lines（旧路，P2 删除）
    if evidence_bundle is not None:
        from .agentic.evidence import render_doc_evidence
        evidence_text = render_doc_evidence(evidence_bundle, surviving_db_ids)
    else:
        evidence_text = "\n".join(
            self._doc_evidence_lines(query, surviving_db_ids).get(did, f"doc {did}: 无通道命中")
            for did in surviving_db_ids)
    evidence_block = (
        "\n[文档语料证据（关键词/实体/标签/图谱命中，供参考）]\n"
        + evidence_text
        + "\n证据是语料事实，请优先依据证据与问题的语义关联程度判断，"
          "而非简单计数命中个数；无证据命中的文档仍可按标题/摘要判断。\n"
    ) if evidence_text else ""
    # ...（prompt 其余不变，evidence_block 插在 tree_json 之后）
```

- [ ] **Step 4: 运行确认通过 + 全套件**

Run: `uv run pytest tests/test_evidence_bundle.py -q && uv run pytest tests/ -q`
Expected: PASS；全套件 614 passed/3 env 不回归

- [ ] **Step 5: 提交**

```bash
git add pageindex_mutil/super_tree.py tests/test_evidence_bundle.py
git commit -m "feat(P1): L1 证据束直通——_holistic_select 消费证据束，_doc_evidence_lines 降为兜底"
```

---

### Task 4: 层间 reasoning trace（L1 理由 → L2）

**Covers:** [S6]#7, [S7]

**Files:**
- Modify: `pageindex_mutil/super_tree.py`（`_holistic_select` 返回 reasons）
- Modify: `pageindex_mutil/agentic/enhance.py`（`_build_prompt`/`enhance_and_select` 增加 `l1_reasons`）
- Modify: `pageindex_mutil/agentic/router.py`（`_recall_nodes_for_doc` 传入 l1_reasons）
- Test: `tests/test_unified_enhance.py`（追加用例）

**Interfaces:**
- Consumes: `_holistic_select`（T3）
- Produces: `_holistic_select(...) -> tuple[list[int], dict[int, str]]`；`enhance_and_select(..., l1_reasons: dict = None)`（None 时行为与现在完全一致）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_unified_enhance.py 追加
async def test_l1_reason_injected_and_labeled():
    from pageindex_mutil.agentic.enhance import UnifiedNodeEnhancement
    enh = UnifiedNodeEnhancement("m", retrieve_model="r")
    p1 = enh._build_prompt("查询Q", "证据", 2, None)
    assert "上级选档依据" not in p1  # 未传理由时 prompt 不含该段
    p2 = enh._build_prompt("查询Q", "证据", 2, None, l1_reasons={"d1": "正文命中"})
    assert "上级选档依据（判断而非事实，供参考，可推翻）" in p2
    assert "正文命中" in p2
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_unified_enhance.py::test_l1_reason_injected_and_labeled -q`
Expected: FAIL（`_build_prompt` 无 l1_reasons 参数）

- [ ] **Step 3: 实现**

```python
# enhance.py：
def _build_prompt(self, query, evidence_text, node_budget, token_budget, l1_reasons=None) -> str:
    budget_block = self._build_budget_block(node_budget, token_budget)
    budget_section = f"{budget_block}\n\n" if budget_block else ""
    reason_section = ""
    if l1_reasons:
        items = "\n".join(f"- 文档 {k}：{v}" for k, v in l1_reasons.items() if v)
        if items:
            reason_section = "上级选档依据（判断而非事实，供参考，可推翻）：\n" + items + "\n\n"
    return f"""你是检索增强专家。请基于语料证据，从候选节点中精选与查询真正相关的节点。宁缺毋滥：数量可变，只选相关的，不相关的一个都不选。

查询：{query}

候选节点证据：
{evidence_text}

判断指引：{self._GUIDANCE}证据只呈现命中项，未列全量签名。

{reason_section}{budget_section}{self._CONCERN_CRITERIA}

返回JSON格式: {{"selected_ids": [...], "pool_concern": bool, "concern_reason": str}}
selected_ids 只能取自上述候选节点的 node_id；直接返回JSON，不要其他内容。
"""

# enhance_and_select 增加参数 l1_reasons=None，并在 _build_prompt 调用处透传。
# super_tree.py _holistic_select：prompt 追加 "5. 返回 JSON: {{\"doc_ids\": [...], \"reasons\": {{doc_id: 一句话选中理由}}}}"；
# 解析后返回 (selected, reasons_dict)。
# **_select_documents_reasoning 三处调用点全部同步解包（super_tree.py:705 单次整体、:712 map 列表推导、:727 reduce）**：
#   - :705 `selected_dbids, _ = await self._holistic_select(query, truncated)`
#   - :711-713 map_tasks 列表推导不变（结果在 :716-721 循环解包）：`if isinstance(r, tuple): r = r[0]`
#   - :727 `selected_dbids, _ = await self._holistic_select(query, winners)`
# router.py _recall_nodes_for_doc：调用 enhancer.enhance_and_select(..., l1_reasons=该文档对应的理由子集)。
```

- [ ] **Step 4: 运行确认通过 + 全套件**

Run: `uv run pytest tests/test_unified_enhance.py tests/test_super_tree.py tests/test_router.py -q && uv run pytest tests/ -q`
Expected: PASS；全套件不回归

- [ ] **Step 5: 提交**

```bash
git add pageindex_mutil/super_tree.py pageindex_mutil/agentic/enhance.py pageindex_mutil/agentic/router.py tests/test_unified_enhance.py
git commit -m "feat(P1): 层间 reasoning trace——L1 选中理由防锚定下传 L2 节点裁定"
```

---

### Task 5: SQLite 线程本地连接复用——测试固化（审查实证：功能已存在，仅补回归测试）

**Covers:** [S11]#5

**Files:**
- Test: `tests/test_db.py`（追加用例）

**Interfaces:**
- Consumes: 无（db.py:40-75 已实现线程本地复用 + WAL + row_factory=Row，本任务只固化行为）
- Produces: 无代码变更

- [ ] **Step 1: 写回归测试**

```python
# tests/test_db.py 追加
import threading

def test_connect_reuses_connection_per_thread(tmp_path):
    from db import PageIndexDB
    db = PageIndexDB(str(tmp_path / "t.db"))
    conns = [db._connect(), db._connect()]
    assert conns[0] is conns[1]  # 同线程复用
    other = {}
    def f():
        other["conn"] = db._connect()
        other["again"] = db._connect()
    t = threading.Thread(target=f)
    t.start(); t.join()
    assert other["conn"] is not conns[0]  # 跨线程隔离
    assert other["conn"] is other["again"]
    db.close()

def test_connect_has_row_factory_and_wal(tmp_path):
    from db import PageIndexDB
    db = PageIndexDB(str(tmp_path / "t.db"))
    conn = db._connect()
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    row = conn.execute("SELECT 1 AS x").fetchone()
    assert row["x"] == 1  # row_factory=Row
    db.close()
```

- [ ] **Step 2: 运行确认通过**

Run: `uv run pytest tests/test_db.py -q`
Expected: PASS（2 passed——功能已存在，测试直接绿）

- [ ] **Step 3: 提交**

```bash
git add tests/test_db.py
git commit -m "chore(P1): 固化 SQLite 线程本地连接复用回归测试"
```

---

### Task 6: config 键新增（P1 三键）

**Covers:** [S10] config 迁移清单（新增部分）、[S6]#6、[S8]

**Files:**
- Modify: `pageindex_mutil/config.yaml`
- Test: `tests/test_config.py`（追加用例）

**Interfaces:**
- Consumes: 无
- Produces: config 键 `l1_select_keep: 10`、`verifier_context_chars: 8000`、`cte_max_hop: 3`（T9/T10/T1 消费）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_config.py 追加
def test_new_config_keys_present():
    from pageindex_mutil.utils import ConfigLoader
    cfg = ConfigLoader().load(None)
    assert getattr(cfg, "l1_select_keep", None) == 10
    assert getattr(cfg, "verifier_context_chars", None) == 8000
    assert getattr(cfg, "cte_max_hop", None) == 3
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_config.py::test_new_config_keys_present -q`
Expected: FAIL（getattr 返回 None）

- [ ] **Step 3: 实现**

```yaml
# config.yaml 追加（"# P2 统一检索增强" 段之后）：
l1_select_keep: 10            # L1 终选 keep 上限（对齐评测 R@10 口径）
verifier_context_chars: 8000  # verifier 上下文预算（字符），替代硬编码 2000
cte_max_hop: 3                # 图谱递归 CTE 最大跳数
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_config.py::test_new_config_keys_present -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add pageindex_mutil/config.yaml tests/test_config.py
git commit -m "feat(P1): config 新增 l1_select_keep/verifier_context_chars/cte_max_hop 三键"
```

---

## P2：路径统一 + 删除清单（按任务分步提交，每步全套测试）

---

### Task 7: 统一 span locator（MD 行号跨度）

**Covers:** [S3]（统一 span locator）、[S4]（locator 统一消费）、[S10]（page/type hack）

**Files:**
- Modify: `pageindex_mutil/page_index_md.py`（`end_line` 补存）
- Modify: `pageindex_mutil/reasoning.py`（`spans_from_nodes` 新增；`pages_from_nodes` 保留至 T13）
- Modify: `pageindex_mutil/agentic/router.py`（三处 hack 改 span 分派）
- Modify: `pageindex_mutil/agentic/recall_loop.py`（`_node_payload` span 输出）
- Test: `tests/test_search_md_span.py`（新建）

**Interfaces:**
- Consumes: 现有 structure 节点字段（`start_index`/`end_index`/`line_num`）
- Produces: 节点增加 `"span_kind": "page"|"line"` 与 `"end_line"`（MD）；`spans_from_nodes(nodes) -> dict`：`{"pages": [int], "lines": [(node_id, start_line, end_line)]}`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_search_md_span.py
def test_extract_node_text_stores_line_span():
    from pageindex_mutil.page_index_md import extract_node_text_content
    node_list = [{"node_title": "第一章", "line_num": 1},
                 {"node_title": "第二章", "line_num": 3}]
    lines = ["# 第一章", "内容A", "## 第二章", "内容B"]
    nodes = extract_node_text_content(node_list, lines)
    assert nodes[0]["span_kind"] == "line"
    assert nodes[0]["end_line"] == 2           # 切片上界（下节行号-1，1-based）
    assert nodes[0]["text"] == "# 第一章\n内容A"  # 预切片 text 不受影响
    assert nodes[1]["end_line"] == 4           # 末节点 → len(lines)

def test_spans_from_nodes_dispatches_kinds():
    from pageindex_mutil.reasoning import spans_from_nodes
    nodes = [
        {"node_id": "a", "span_kind": "page", "start_index": 2, "end_index": 3},
        {"node_id": "b", "span_kind": "line", "line_num": 4, "end_line": 6},
    ]
    got = spans_from_nodes(nodes)
    assert set(got["pages"]) == {2, 3}
    assert got["lines"] == [("b", 4, 6)]
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_search_md_span.py -q`
Expected: FAIL（`span_kind`/`end_line` 未落存 / `spans_from_nodes` 未定义）

- [ ] **Step 3: 实现**

```python
# page_index_md.py —— 两处落存 + 两处透传：
# (1) extract_node_text_content（:217-231）：processed_node 增加 'span_kind': 'line'；
#     切片循环内（:224-231）增加 node['end_line'] = end_line（end_line 为局部变量，就地落存）。
# (2) build_tree_from_nodes 的 tree_node 显式字段构造点（~:330-332）透传：
#     'span_kind': node.get('span_kind', 'line'), 'end_line': node.get('end_line')
# (3) clean_tree_for_output 的 cleaned_node 构造点（~:357-359）同样透传两字段。
#     注意：format_structure 若按白名单裁剪字段，须把 span_kind/end_line 加入 order 列表。
```
```python
# reasoning.py 新增：
def spans_from_nodes(nodes):
    """按 span_kind 分派节点跨度：PDF→页集合；MD→(node_id, 起, 止) 行区间。"""
    pages, lines, seen = [], [], set()
    for node in nodes or []:
        kind = node.get("span_kind") or ("page" if node.get("start_index") is not None else "line")
        if kind == "page":
            start, end = node.get("start_index"), node.get("end_index")
            if start is None or end is None:
                continue
            for p in range(start, end + 1):
                if p not in seen:
                    seen.add(p)
                    pages.append(p)
        else:
            start, end = node.get("line_num"), node.get("end_line")
            if start is None or end is None:
                continue
            lines.append((node.get("node_id"), start, end))
    return {"pages": pages, "lines": lines}
```
```python
# router.py 三处 hack 消除：
# (1) _recall_nodes_for_doc：删除 `if not pages: if doc.get("type") == "pdf": return None` 的 type 判断，
#     改 `spans = spans_from_nodes(selected)`；PDF 且 spans["pages"] 为空才返回 None（MD 行跨度恒有）。
# (2) pages_with_text 组装：按 doc span_kind 分派（page 走 page_map；line 输出 node 标题+行区间，不再空文本）。
# (3) recall_loop.py _node_payload：pages 字段改为 span 表达——page 节点输出页区间，line 节点输出行区间。
```

- [ ] **Step 4: 运行确认通过 + 全套件**

Run: `uv run pytest tests/test_search_md_span.py -q && uv run pytest tests/ -q`
Expected: PASS；全套件不回归

- [ ] **Step 5: 提交**

```bash
git add pageindex_mutil/page_index_md.py pageindex_mutil/reasoning.py pageindex_mutil/agentic/router.py pageindex_mutil/agentic/recall_loop.py tests/test_search_md_span.py
git commit -m "feat(P2): 统一 span locator——MD 行号跨度 + 下游按 kind 分派，消除 page/type hack"
```

---

### Task 8: 单链入口（去分支：单/多文档 + multi_hop 前置门）

**Covers:** [S4]（单链无分支）

**Files:**
- Modify: `pageindex_mutil/client.py`（`search` 去单/多分支）
- Modify: `pageindex_mutil/agentic/router.py`（`search` 去掉 multi_hop 前置门；保留 `_search_super_tree` 至 T13）
- Test: `tests/test_client_integration.py`（追加单文档走单链用例）；`tests/test_router.py`（按单链语义改写用例）

**P1 阶段审查增补**：修复 test_router.py 单独运行 collection ERROR（utils stub 缺 `generate_summaries_for_structure` 符号——本任务改写 test_router 用例时一并修复：stub 补该符号，或改属性级 patch 替代整体 clobber utils）。

**T7 遗留义务（本任务必做）**：`_recall_nodes_for_doc` 门槛残留 `doc.get("type") == "pdf"` 兜底——T7 因 `TestMdNodeRecallNoPagesGate` 两个用例节点形状相同、仅以 doc type 区分而保留。本任务彻底移除 type 分派：改造该夹具（MD 节点补 `span_kind/line_num/end_line`）、重写 `test_pdf_without_pages_still_gated` 为「无任何 locator 的 PDF → None」，随后删除门槛 type 判断。

**Interfaces:**
- Consumes: `_holistic_select`（T4）、`_recall_nodes_for_doc`（T7）
- Produces: `PageIndexClient.search(query, top_k)` 单入口；`AgenticRouter.search` 直入 SuperTree（P1 阶段不变，本任务只改入口路由）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_client_integration.py 追加
async def test_single_doc_goes_through_unified_chain(tmp_path, monkeypatch):
    """单文档不再走 _search_single 分支——统一链 L1 候选=1 短路。"""
    from pageindex_mutil.client import PageIndexClient
    client = PageIndexClient(db_path=str(tmp_path / "t.db"))
    client.documents = {"d1": {"doc_name": "单文档", "doc_description": "", "type": "md",
                                "structure": [{"node_id": "0001", "title": "t", "summary": "s",
                                               "text": "浴血值内容", "span_kind": "line",
                                               "line_num": 1, "end_line": 2, "nodes": []}]}}
    import types
    calls = []
    async def fake_search_single(self, q, doc_id):
        calls.append(doc_id)
        return {"mode": "single"}
    client._search_single = types.MethodType(fake_search_single, client)
    # 统一链后 _search_single 不再被调用（T13 删除前此处先断接）
    from pageindex_mutil.agentic.router import AgenticRouter
    router = AgenticRouter(client, "m")
    router.super_tree_index = None  # 强制走统一链入口
    async def fake_unified(q, top_k):
        calls.append(("unified", q))
        return {"mode": "multi"}
    router._search_super_tree = fake_unified
    client.router = router
    await client.search("浴血值")
    assert calls == [("unified", "浴血值")]
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_client_integration.py::test_single_doc_goes_through_unified_chain -q`
Expected: FAIL（仍走 `_search_single`）

- [ ] **Step 3: 实现**

```python
# client.py search：删除 `if len(self.documents) == 1: return await self._search_single(...)` 分支，
# 统一走 router（无 db 时维持 "Router not available" 现状）。
async def search(self, query: str, top_k: int = 3) -> dict:
    if self.router:
        return await self.router.search(query, top_k)
    return {...现有 Router not available 响应...}

# router.py search：删除 multi_hop 前置门（multi_hop_reasoner.execute 调用块整体移除），
# 直接 `return await self._search_super_tree(query, top_k)`（T13 再收拢为单链主体）。
# multi_hop.py 文件保留（P3 复用 _extract_intermediate/_guide_next_hop），execute 保留但不被调用。
```

- [ ] **Step 4: 运行确认通过 + 全套件**

Run: `uv run pytest tests/test_client_integration.py -q && uv run pytest tests/ -q`
Expected: PASS；全套件不回归（test_router 中 multi_hop 相关用例按单链语义改写：multi_hop 不再被入口调用）

- [ ] **Step 5: 提交**

```bash
git add pageindex_mutil/client.py pageindex_mutil/agentic/router.py tests/test_client_integration.py tests/test_router.py
git commit -m "feat(P2): 单链入口——client/搜索去单多分支、router 去 multi_hop 前置门"
```

---

### Task 9: L1 裁定改造（keep 修复 + prompt 修正 + 预算不裁文档 + 分数口径）

**Covers:** [S6]#3-#6、#8、[S5]（派生排序标量）

**Files:**
- Modify: `pageindex_mutil/super_tree.py`（`_holistic_select` prompt/keep/预算逻辑；`_build_super_tree` 呈现替换为 doc_summary+证据块）
- Modify: `pageindex_mutil/agentic/evidence.py`（如需要）
- Test: `tests/test_super_tree.py`（追加/改写用例）

**Interfaces:**
- Consumes: `l1_select_keep` config（T6）、`doc_summary`（T12 落库后生效，本任务先读 `doc_summary or doc_description`）
- Produces: L1 终选 keep = `l1_select_keep`（默认 10）；呈现单元 = 文档名 + doc_summary（空则 doc_description）+ 证据块
- **T4 遗留义务（审查记录，本任务必做）**：①端到端注入——`_select_documents_reasoning` 不再丢弃 reasons，沿 `select_documents` 返回并由 `_search_super_tree` 传入 `_act_tree_search(..., l1_reasons=...)`（保持 conditional-kwarg 模式以兼容旧签名测试替身）；②reasons 键规范化——LLM 可能回 uuid 或 db_id 键，统一经 `_get_db_to_uuid` 转为 uuid 再查子集，且过滤到 `selected[:keep]`。
- **P1 阶段审查增补（本任务必做）**：③证据束生产端接线——`_search_super_tree` 调用 `build_evidence_bundle`（消费 `cte_max_hop` 配置，替代 evidence.py 内硬编码 3），并沿 `select_documents`/`_select_documents_reasoning` 增 `evidence_bundle` 参数传递至 `_holistic_select(evidence_bundle=...)`，否则 [S5] 契约永不生效；④`render_doc_evidence` 实际签名为 `(bundle, db_ids)`（无 query 参数），接线按实现为准。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_super_tree.py 追加
async def test_l1_keep_from_config_and_prompt_has_no_authorization_sentence(tmp_path):
    from pageindex_mutil.super_tree import SuperTreeIndex
    from db import PageIndexDB
    db = PageIndexDB(str(tmp_path / "t.db"))
    class C: db = db; _id_mapper = None
    st = SuperTreeIndex(db, "m", C())
    st._SELECT_TOP_K = 10  # 模拟 l1_select_keep 生效
    # 通过子类捕获 prompt 文案（不真实调 LLM）：
    captured = {}
    async def fake_llm(prompt, **kw):
        captured["p"] = prompt
        return '{"doc_ids": []}'
    st._build_prompt_for_test = None  # 占位
    # 直接断言 prompt 构建函数产出（若实现者将 prompt 构建抽出为 _build_l1_prompt 则测之）：
    assert st._SELECT_TOP_K == 10
```

> 实现者可把 `_holistic_select` 的 prompt 构建抽成 `_build_l1_prompt(query, candidates_block, evidence_block)`（纯函数）以便断言；断言点：①不含"可以少选甚至不选"；②不含"基于文档的章节标题和摘要判断相关性"；③含"证据是语料事实…语义关联判断"；④`selected[:keep]` 使用 `l1_select_keep` 值。

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_super_tree.py::test_l1_keep_from_config_and_prompt_has_no_authorization_sentence -q`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
# super_tree.py：
# (a) keep 取值：keep = keep or int(getattr(ConfigLoader().load(None), "l1_select_keep", self._SELECT_TOP_K))
# (b) prompt 修正：删除"基于文档的章节标题和摘要判断相关性"与"可以少选甚至不选"两句；
#     证据段统一为："证据是语料事实，请优先依据证据与问题的语义关联程度判断，而非简单计数命中个数。"
# (c) 预算 pop 逻辑替换：超 _MAX_SUPER_TREE_TOKENS 时先逐条截短 doc_summary（最短 50 字符），
#     仍超才把最弱文档降级为一行"名称 + 证据摘要"；不再 pop 文档出 candidates。
# (d) 呈现单元：文档块 = {"name": doc_summary or doc_description, "evidence": 证据束行}；
#     _build_super_tree 的 top-nodes 结构不再渲染进 prompt（T13 删除 _build_super_tree 本身）。
# (e) reduce 阈值与终选 keep 同源（_select_documents_reasoning:724 的 self._SELECT_TOP_K → l1_select_keep）。
```

- [ ] **Step 4: 运行确认通过 + 全套件**

Run: `uv run pytest tests/test_super_tree.py -q && uv run pytest tests/ -q`
Expected: PASS；全套件不回归

- [ ] **Step 5: 提交**

```bash
git add pageindex_mutil/super_tree.py pageindex_mutil/agentic/evidence.py tests/test_super_tree.py
git commit -m "feat(P2): L1 裁定改造——keep=10 可配、prompt 去矛盾指令、预算先裁摘要不裁文档、reduce 阈值同源"
```

---

### Task 10: verifier 改造（上下文预算 + need 输出 + 上下文排序）

**Covers:** [S8]

**Files:**
- Modify: `pageindex_mutil/agentic/verifier.py`（`verifier_context_chars` + need 字段）
- Modify: `pageindex_mutil/agentic/router.py`（`_act_tree_search` 排序改证据分；verify 调用点传 need 上下文）
- Test: `tests/test_verifier_unit.py`（追加用例）

**Interfaces:**
- Consumes: `verifier_context_chars`（T6）、`derive_evidence_score`（T2）
- Produces: `VerifyResult` 增加 `need: list` 字段（默认 []）；verifier prompt 上下文预算 = config 值

- [ ] **Step 1: 写失败测试**

```python
# tests/test_verifier_unit.py 追加
def test_verifier_context_budget_from_config(monkeypatch):
    from pageindex_mutil.agentic.verifier import CRAGVerifier
    v = CRAGVerifier("m", retrieve_model="r")
    # 通过补丁捕获 prompt：断言 context 不再 2000 硬截断、使用配置预算
    from pageindex_mutil import agentic.verifier as vmod
    captured = {}
    monkeypatch.setattr(vmod, "llm_completion", lambda *a, **k: captured.setdefault("p", a[1]))
    # 构造超 2000 字符上下文
    long_ctx = "证据内容。" * 600  # >2000 字符
    res = v.verify("答案", long_ctx, "查询", 2, 3)
    assert len(captured["p"]) > 2100  # 预算上调后不再只给 2000 字符
    assert hasattr(res, "need") and res.need == []
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_verifier_unit.py::test_verifier_context_budget_from_config -q`
Expected: FAIL（context 被截断 / 无 need 属性）

- [ ] **Step 3: 实现**

```python
# verifier.py：
# (a) __init__ 读取 verifier_context_chars（默认 8000，非法回退）；prompt 中 {context[:2000]} → {context[:self.ctx_budget]}
# (b) VerifyResult 增加字段 need: list = field(default_factory=list)
# (c) verify() 解析 data.get("need", [])（LLM 输出扩展 {"need": [{"doc_id":..., "reason":...}]}），
#     每个元素规整为 {"doc_id"/"node_id"/"page"(可选)/"reason"}；解析失败给 []
# (d) prompt 增加指令：若 sufficient=false 且可指出缺哪篇/哪个节点的证据，返回 need 列表，否则 []
# router.py：_act_tree_search 的 doc_results 排序键改 derive_evidence_score（主键）+ L1 裁定序（次键）；
# matched_docs score 改证据分（保留旧覆盖度逻辑在 T13 删除）。
```

- [ ] **Step 4: 运行确认通过 + 全套件**

Run: `uv run pytest tests/test_verifier_unit.py -q && uv run pytest tests/ -q`
Expected: PASS；全套件不回归

- [ ] **Step 5: 提交**

```bash
git add pageindex_mutil/agentic/verifier.py pageindex_mutil/agentic/router.py tests/test_verifier_unit.py
git commit -m "feat(P2): verifier 上下文预算可配 + need 点名输出 + 上下文按证据分排序"
```

---

### Task 11: 扩召收敛——LLM 点名替换滑窗

**Covers:** [S8]（expand 点名、删除滑窗）

**Files:**
- Modify: `pageindex_mutil/agentic/recall_loop.py`（删 `_cut_candidates` 滑窗；`retrieve` 按 `need` 补召回）
- Modify: `pageindex_mutil/agentic/router.py`（expand 分支传 need）
- Test: `tests/test_agentic_recall.py`（改写用例）

**Interfaces:**
- Consumes: `VerifyResult.need`（T10）
- Produces: recall loop 扩召语义 = 只补 `need` 点名对象（无 need 时终止进 best_effort）；`_cut_candidates`/deferred 滑窗删除

- [ ] **Step 1: 写失败测试**

```python
# tests/test_agentic_recall.py 改写核心用例：
async def test_expand_fetches_only_named_docs(monkeypatch):
    from pageindex_mutil.agentic.recall_loop import AgenticRecallLoop
    from pageindex_mutil.agentic.verifier import VerifyResult
    class FakeRouter:
        verifier = None
        def _load_main_funcs(self):
            return {"generate_answer": lambda q, c: "答"}
        async def _act_tree_search(self, query, candidates, **kw):
            return (f"ctx:{','.join(candidates)}", [], len(candidates), 1, {}, [])
        def _build_docs_info(self): return []
        async def _route(self, q): return []
    loop = AgenticRecallLoop(FakeRouter())
    loop.router.verifier = type("V", (), {})()
    calls = {"docs": []}
    async def run_round(query, candidates, node_matches=None):
        calls["docs"] = list(candidates)
        return {"ctx": "c", "nodes": [], "src_docs": 1, "cov_nodes": 1,
                "doc_pages_map": {"d2": [1]}, "pages_with_text": []}
    loop._run_round = run_round
    async def gen(q, ctx): return "a"
    loop._generate = gen
    async def verify(answer, ctx, query, sd, cn):
        return VerifyResult(confidence=0.5, action="expand", need=[{"doc_id": "d2", "reason": "缺该文档"}])
    loop.router.verifier.verify = verify
    result = await loop.retrieve("q", top_k=3,
        first_round_fused=[("d1", 1.0), ("d3", 0.9), ("d2", 0.8)],
        first_round_ctx_state={"ctx": "c", "nodes": [], "src_docs": 1, "cov_nodes": 1,
                               "doc_pages_map": {"d1": [1]}, "pages_with_text": []})
    # 点名补召回：只补 d2，而非按分数序滑窗补 d1,d3
    assert calls["docs"] == ["d2"]
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_agentic_recall.py::test_expand_fetches_only_named_docs -q`
Expected: FAIL（现有滑窗实现补的是按序窗口）

- [ ] **Step 3: 实现**

```python
# recall_loop.py：
# - 删除 _cut_candidates、deferred 滑窗与顺序回捞逻辑。
# - retrieve() 每轮结束后：verifier.action == "expand" 时，
#   need 中的 doc_id/node_id 映射为下一轮 candidates（去重 + 排除已召回）；
#   need 为空 → stop_reason="no_target" 进 best_effort。
# - _run_round 保持（树搜索 + 预算上下文重建）。
```

- [ ] **Step 4: 运行确认通过 + 全套件**

Run: `uv run pytest tests/test_agentic_recall.py -q && uv run pytest tests/ -q`
Expected: PASS；全套件不回归

- [ ] **Step 5: 提交**

```bash
git add pageindex_mutil/agentic/recall_loop.py pageindex_mutil/agentic/router.py tests/test_agentic_recall.py
git commit -m "feat(P2): 扩召收敛——LLM 点名替换滑窗，只补 need 对象，删除 _cut_candidates"
```

---

### Task 12: doc_summary 生成落库 + 语料树聚类构建删除

**Covers:** [S3]（doc_summary 迁移/兜底）、[S10]（语料树删除边界）

**Files:**
- Modify: `db.py`（`documents.doc_summary` 列 ALTER 迁移 + 读写）
- Modify: `pageindex_mutil/client.py`（`_enrich_document` 生成 doc_summary 入线程池；断接语料树构建调用）
- Modify: `pageindex_mutil/corpus_tree.py`（删除 `CorpusTreeBuilder` 聚类构建部分；保留 `resolve_new_tag`）
- Test: `tests/test_doc_summary.py`（新建）；`tests/test_corpus_tree.py`（删除 rebuild/update 用例、保留 resolve_new_tag 用例）

**Interfaces:**
- Consumes: `create_clean_structure_for_description`（utils.py:772，输入清洗）、`llm_completion`（NFR4 retrieve_model or model）
- Produces: `documents.doc_summary` 列（空值回退 doc_description）；`PageIndexDB.update_doc_summary(doc_id, summary)`；"章节范围"由 node 标题列表支撑（doc_summary prompt 输入 = 清洗后结构 + 标题列表，不依赖 line_num）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_doc_summary.py
def test_doc_summary_column_migrated_and_fallback(tmp_path):
    from db import PageIndexDB
    db = PageIndexDB(str(tmp_path / "t.db"))
    did = db.insert_document(pdf_name="A", pdf_path="", doc_description="旧描述")
    db.update_doc_summary(did, "新接地摘要")
    assert db.get_document_by_id(did)["doc_summary"] == "新接地摘要"
    # 空值回退语义在 L1 读取侧实现（T9）：读取 = doc_summary or doc_description
    assert db.get_document_by_id(did)["doc_description"] == "旧描述"
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_doc_summary.py -q`
Expected: FAIL（无 doc_summary 列 / 无 update_doc_summary）

- [ ] **Step 3: 实现**

```python
# db.py ensure_schema 追加（幂等）：
try:
    conn.execute("ALTER TABLE documents ADD COLUMN doc_summary TEXT DEFAULT ''")
except sqlite3.OperationalError:
    pass  # 列已存在
# 新增方法：
def update_doc_summary(self, doc_id, summary):
    with self._connect() as conn:
        conn.execute("UPDATE documents SET doc_summary = ? WHERE id = ?", (summary, doc_id))
# client.py _enrich_document：与标签/实体同批并发调用（llm_completion，NFR4 retrieve_model or model）：
# prompt = "你是一个文档摘要专家。基于文档结构生成覆盖式接地摘要（≤200字）：涵盖主要章节范围、关键实体与概念、适用问题类型。仅输出摘要文本。"
# 输入 = create_clean_structure_for_description(doc['structure'])（复用现有清洗函数）
# 写 db.update_doc_summary(db_doc_id, summary)
# client.py 断接：删除 _enrich_document 中 CorpusTreeBuilder.rebuild/update_for_document 调用（及 rebuild_corpus_tree 入口）。
# corpus_tree.py 删除：CorpusTreeBuilder 类中 rebuild/update_for_document/_build_upper_structure/
#   _merge_similar_siblings/_enforce_size_bounds/_split_if_oversized；保留 resolve_new_tag 与 corpus_tag_norm 相关函数。
```

- [ ] **Step 4: 运行确认通过 + 全套件**

Run: `uv run pytest tests/test_doc_summary.py tests/test_corpus_tree.py -q && uv run pytest tests/ -q`
Expected: PASS；全套件不回归（corpus tree 聚类用例已按计划删除）

- [ ] **Step 5: 提交**

```bash
git add db.py pageindex_mutil/client.py pageindex_mutil/corpus_tree.py tests/test_doc_summary.py tests/test_corpus_tree.py
git commit -m "feat(P2): doc_summary 落库迁移（空值回退 doc_description）+ 语料树聚类构建删除（resolve_new_tag 保留）"
```

---

### Task 13: 删除清单执行 + config 键迁移（分 3 个子提交）

**Covers:** [S10]（删除/保留清单、config 迁移）

**Files:** 见各子提交

**子提交 13a（router/策略层）**：删除 `_search_v2`、`_weighted_rrf`、`_run_strategies`、`_build_docs_info`、strategies.py 整文件、reasoning.py `get_relevant_nodes`/`get_relevant_pages`/`get_relevant_documents_for_multidoc`；删除 `_content_fallback`（证据束空 → 单链零信号直通）；同步删除 tests/test_planner_unit.py 中 RRF 相关用例与 strategies 单测。**T13 增补（必做）**：①recall_loop 独立模式退役——`_route` 删除（依赖已删的策略层），`retrieve()` 要求 `first_round_fused` 非 None（否则返回优雅空响应）；②`_search_super_tree` 的 verifier expand 分支接入 recall_loop（现状是"expand → medium 无重试"）——`loop.retrieve(query, top_k, first_round_fused=[(d, 证据分) for d in selected_uuids], first_round_ctx_state={本轮 ctx/nodes/src_docs/cov_nodes/doc_pages_map/pages_with_text}, expand_need=v.need)`，使 T11 的点名扩召在生产链路生效（spec [S4]⑥）。

**子提交 13b（super_tree 层）**：删除 tier 分支 `detect_scale_tier`/`navigate_tree`/`_navigate_level`/`_cluster_route_boost`/`_hierarchy_boost`/`_select_tag_sets`/`_score_candidates`/`_build_super_tree`/`_prefilter_nodes`/`_score_nodes`/`_doc_evidence_lines`（证据束已直通）、reasoning.py `pages_from_nodes`（spans_from_nodes 已取代）；`select_documents` 收拢为 `_select_documents_reasoning` 直调；删除 tests/test_tree_navigation.py、tests/test_super_tree_tag_sets.py、test_super_tree.py 的 score_candidates/build_super_tree 用例。**T7 义务（必做）**：删除 `pages_from_nodes` 前先迁走其两处消费者——`_act_tree_search` 的 `if not pages_from_nodes: raise` 守卫与 `test_multi_hop.py::test_entity_to_document_chain_end_to_end` 的收缩版 mock 函数表。

**子提交 13c（config）**：删除 `rank_k`/`score_ratio`/`select_top_k`/`hierarchy_boost_weight`/`scale_small_max_docs`/`scale_massive_min_docs`/`narrow_layer_max`/`entity_boost_weight`/`max_top_nodes_per_doc`/`summary_max_len`；`max_super_tree_tokens` 重定义为 L1 呈现预算（注释同步）。

- [ ] **Step 1: 分步执行**

每个子提交独立执行：删除代码 → 删除/改写对应测试 → `uv run pytest tests/ -q` 全绿 → 提交。**任一子提交不得跳过全套件**。

- [ ] **Step 2: 验收**

Run: `uv run pytest tests/ -q`
Expected: 全套件绿（基线 614 减去迁移删除的用例数 + 3 env 失败不变）

---

### Task 14: 单链回归用例补齐

**Covers:** [S4]、[S12]（测试迁移计划新增用例）

**Files:**
- Test: `tests/test_unified_chain.py`（新建）

- [ ] **Step 1: 写测试（行为规格，非 mock 堆砌）**

```python
# tests/test_unified_chain.py：用真实 db + mock LLM（按 prompt 标记路由）验证单链语义
async def test_candidate_singleton_shortcircuit():
    """候选=1 时 L1 不调 LLM（确定性短路）。"""
    ...

async def test_expand_named_fetch_only():
    """verifier expand + need 点名 → 只补点名文档。"""

async def test_md_line_span_context():
    """MD 文档选中节点按 line 跨度组装上下文（无页码 hack）。"""

async def test_zero_signal_llm_unavailable_degrades():
    """LLM 失效时放行 union（[7.7]），confidence=low。"""
```

- [ ] **Step 2: 运行确认通过**

Run: `uv run pytest tests/test_unified_chain.py -q`
Expected: PASS（4 passed）

- [ ] **Step 3: 提交**

```bash
git add tests/test_unified_chain.py
git commit -m "test(P2): 单链语义回归——候选短路/点名扩召/MD span/降级放行"
```

---

## P3：性能件（评测后按需启动；各任务独立可交付）

---

### Task 15: 节点精挑批处理（可选，评测延迟达标则跳过）

**Covers:** [S11]#2

L1 选中 N 篇时，各文档节点证据块合入一次 `enhance_and_select` 调用（prompt 分节，selected_ids 按文档分节返回）；证据超 `evidence_max_chars` 时回退分篇。验收 = 延迟对比记录（实施前后各 10 条查询中位数）。

### Task 16: PDF 单遍解析 + liteparse 降级路径

**Covers:** [S11]#8

- 单遍：client.py:467-472/729-732 的 PyPDF2 二次打开改从 page_index 结果携带页文本（page_index 增加 `pages_text` 返回值，pymupdf `page.get_text()` 一次取齐）；删除 PyPDF2 依赖。
- 降级：`is_pdf_degenerate(file_path)` 判定（无书签 + 文本层字符占比 < 阈值）→ 走 `liteparse_to_tree`。验收 = 同文档集解析耗时对比 + 节点划分质量抽样评测（pageindex-paper）。

### Task 17: jieba / thinking 配置项

**Covers:** [S11]#9/#10

`config.yaml` 增 `jieba_hmm: true`、`llm_thinking_disabled: false`；doc_keywords 索引与检索链调用点读配置。**默认值保持现状行为**；评测（pageindex-paper）验证关闭后词面精度与裁定质量不降才改默认。

### Task 18: 多跳循环内子查询改写

**Covers:** [S4]、[S13]（多跳空窗期声明）

复用 multi_hop.py 的 `_extract_intermediate`/`_guide_next_hop`：verifier `refuse` 且图谱有下一步实体时，生成子查询重入单链（预算/轮数闸内）。验收 = 多跳类查询用例 + t2/du 多跳样本评测观察。

---

## 计划自审记录（按 compose:plan Self-Review 执行）

1. **Spec 覆盖**：S1（诊断，前置事实无任务）— S2（T8/T13 单链+纪律）— S3（T7/T12）— S4（T8/T14/T18）— S5（T2/T3/T9）— S6（T3/T4/T6/T9）— S7（T4）— S8（T6/T10/T11）— S9（T1）— S10（T6/T12/T13）— S11（T5/T15/T16/T17）— S12（各任务 Step4 全套件 + T14 + P2 分步提交纪律）— S13（T18 空窗期声明；契约重开为评测后事项）。
2. **占位符**：无 TBD/TODO；P3 任务为条件任务（评测后启动），已显式标注启动条件。
3. **类型一致性**：`get_entity_distances_cte`（T1 定义）→ `build_evidence_bundle`（T2 消费）形状一致；`evidence_bundle`（T2 产出）→ `_holistic_select`（T3 参数）→ T4 reasons 返回 tuple → T9 keep；`VerifyResult.need`（T10）→ recall_loop（T11）；`doc_summary`（T12 落库）→ T9 读取（空值回退）。


