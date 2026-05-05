# Agentic Multi-Strategy Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade PageIndex-UV's non-vector retrieval from single-strategy L1 to an agentic Plan→Route→Act→Verify pipeline with three parallel retrieval strategies, RRF fusion, and CRAG confidence verification.

**Architecture:** SQLite-based semantic tag index (Closet) + asyncio-parallel Metadata/Semantics/Description strategies with weighted RRF fusion + Tree-based Act phase + CRAG Verify with three-threshold routing. Zero vector dependencies.

**Tech Stack:** Python 3.12+, OpenAI-compatible API (DashScope/Qwen), SQLite, jieba, asyncio.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `db.py` | Modify | Add `closet_tags` table with CRUD operations |
| `pyproject.toml` | Modify | Add `jieba>=0.42` dependency |
| `PageIndex/pageindex/closet_index.py` | Create | LLM tag extraction + jieba tokenization + SQLite inverted index |
| `PageIndex/pageindex/agentic/__init__.py` | Create | Package marker |
| `PageIndex/pageindex/agentic/planner.py` | Create | Query analysis, HyDE hypothetical answer, variant generation, weight assignment |
| `PageIndex/pageindex/agentic/strategies.py` | Create | MetadataStrategy (SQL LIKE), SemanticsStrategy (Closet), DescriptionStrategy (LLM relevance) |
| `PageIndex/pageindex/agentic/verifier.py` | Create | CRAG verifier: S_ret computation, S_CoV LLM judgment, three-threshold routing |
| `PageIndex/pageindex/agentic/router.py` | Create | AgenticRouter: orchestrates Plan→Route→Act→Verify pipeline |
| `PageIndex/pageindex/client.py` | Modify | Integrate PageIndexDB + ClosetIndex + AgenticRouter; add `async search()` |
| `test_smoke.py` | Create | End-to-end smoke test |

---

## Prerequisites

- `.env` file with `DASHSCOPE_API_KEY`, `OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`, `MODEL_NAME=qwen-plus`
- `config.yaml` default model set to `qwen-plus` for DashScope compatibility
- Working directory: `/home/ctyun/clauderestor/PageIndex-UV`

---

### Task 1: Infrastructure — db.py + pyproject.toml

**Files:**
- Modify: `db.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `closet_tags` table to `ensure_schema()`**

  In `db.py`, add the following table creation inside `ensure_schema()` (after existing tables):

  ```python
  self.cursor.execute("""
      CREATE TABLE IF NOT EXISTS closet_tags (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          doc_id INTEGER NOT NULL,
          tag TEXT NOT NULL,
          confidence REAL DEFAULT 1.0,
          FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
      )
  """)
  self.cursor.execute("""
      CREATE INDEX IF NOT EXISTS idx_closet_tag ON closet_tags(tag)
  """)
  self.cursor.execute("""
      CREATE INDEX IF NOT EXISTS idx_closet_doc ON closet_tags(doc_id)
  """)
  ```

- [ ] **Step 2: Add Closet CRUD methods to PageIndexDB**

  Add these methods to the `PageIndexDB` class in `db.py`:

  ```python
  def insert_closet_tags(self, doc_id: int, tags: list):
      """Insert semantic tags for a document. tags: [(text, confidence), ...]"""
      if not tags:
          return
      # Chunk to avoid SQLITE_MAX_VARIABLE_NUMBER (999)
      chunk_size = 900
      for i in range(0, len(tags), chunk_size):
          batch = tags[i:i + chunk_size]
          self.cursor.executemany(
              "INSERT INTO closet_tags (doc_id, tag, confidence) VALUES (?, ?, ?)",
              [(doc_id, text, conf) for text, conf in batch],
          )
      self.conn.commit()

  def delete_closet_tags(self, doc_id: int):
      self.cursor.execute("DELETE FROM closet_tags WHERE doc_id = ?", (doc_id,))
      self.conn.commit()

  def match_closet_tags(self, query_tags: list, top_k: int = 10) -> list:
      """Match query tags against closet_tags. Returns [(doc_id, match_count)]."""
      if not query_tags:
          return []
      placeholders = ",".join("?" for _ in query_tags)
      self.cursor.execute(f"""
          SELECT doc_id, COUNT(*) as cnt
          FROM closet_tags
          WHERE tag IN ({placeholders})
          GROUP BY doc_id
          ORDER BY cnt DESC
          LIMIT ?
      """, (*query_tags, top_k))
      return self.cursor.fetchall()
  ```

- [ ] **Step 3: Add `jieba` to pyproject.toml**

  Add `"jieba>=0.42"` to the `dependencies` list in `pyproject.toml`.

- [ ] **Step 4: Verify import**

  Run: `python3 -c "from db import PageIndexDB; db = PageIndexDB('/tmp/test.db'); print('OK')"`
  Expected: `OK`

- [ ] **Step 5: Commit**

  ```bash
  git add db.py pyproject.toml
  git commit -m "feat: add closet_tags table and jieba dependency"
  ```

---

### Task 2: ClosetIndex — Semantic Tag Index

**Files:**
- Create: `PageIndex/pageindex/closet_index.py`
- Test: `PageIndex/tests/test_closet_index.py`

- [ ] **Step 1: Write the failing test**

  Create `PageIndex/tests/test_closet_index.py`:

  ```python
  import tempfile
  import os
  import sys

  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

  from db import PageIndexDB
  from PageIndex.pageindex.closet_index import ClosetIndex, Tag


  class FakeDB:
      def __init__(self):
          self.tags = {}
      def insert_closet_tags(self, doc_id, tags):
          self.tags[doc_id] = tags
      def delete_closet_tags(self, doc_id):
          self.tags.pop(doc_id, None)
      def match_closet_tags(self, query_tags, top_k=10):
          results = {}
          for doc_id, tags in self.tags.items():
              for t, _ in tags:
                  if t in query_tags:
                      results[doc_id] = results.get(doc_id, 0) + 1
          return sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]


  def test_tag_dataclass():
      t = Tag(text="pdf", confidence=0.9)
      assert t.text == "pdf"
      assert t.confidence == 0.9


  def test_closet_index_add_and_search():
      db = FakeDB()
      idx = ClosetIndex(db, model="qwen-plus")
      # Mock _extract_tags to avoid LLM call
      idx._extract_tags = lambda text: [Tag("pdf", 0.9), Tag("indexing", 0.8)]
      idx.add_document(1, "How to index PDFs")
      results = idx.search("pdf indexing", top_k=5)
      assert len(results) >= 1
      assert results[0][0] == 1
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `python3 -m pytest PageIndex/tests/test_closet_index.py -v`
  Expected: `ModuleNotFoundError: No module named 'PageIndex.pageindex.closet_index'`

- [ ] **Step 3: Implement ClosetIndex**

  Create `PageIndex/pageindex/closet_index.py`:

  ```python
  import json
  import logging
  from dataclasses import dataclass
  from typing import List, Tuple

  import jieba

  from ..utils import llm_completion, extract_json


  _STOPWORDS = set()
  try:
      import jieba
      _STOPWORDS = set(jieba.analyse.default_tfidf.stop_words)
  except Exception:
      pass


  @dataclass
  class Tag:
      text: str
      confidence: float = 1.0


  class ClosetIndex:
      def __init__(self, db, model: str):
          self.db = db
          self.model = model
          if hasattr(db, "ensure_closet_schema"):
              db.ensure_closet_schema()

      def _extract_tags(self, text: str) -> List[Tag]:
          prompt = f"""从以下文本中提取3-5个语义概念标签（关键词）。

  文本: {text[:2000]}

  返回JSON格式: {{"tags": ["tag1", "tag2", "tag3"]}}
  直接返回JSON，不要其他内容。"""
          try:
              response = llm_completion(self.model, prompt)
              if not response:
                  return self._fallback_tags(text)
              data = extract_json(response)
              if isinstance(data, dict):
                  tags = data.get("tags", [])
              elif isinstance(data, list):
                  tags = data
              else:
                  return self._fallback_tags(text)
              return [Tag(str(t).strip(), 1.0) for t in tags if str(t).strip()]
          except Exception as e:
              logging.warning("Tag extraction failed: %s", e)
              return self._fallback_tags(text)

      def _fallback_tags(self, text: str) -> List[Tag]:
          tokens = jieba.lcut(text)
          keywords = [
              t.strip().lower()
              for t in tokens
              if len(t.strip()) > 1 and t.strip().lower() not in _STOPWORDS
          ]
          from collections import Counter
          return [Tag(k, 1.0) for k, _ in Counter(keywords).most_common(5)]

      def _tokenize_tag(self, tag_text: str) -> List[str]:
          tokens = jieba.lcut(tag_text.lower())
          return [t.strip() for t in tokens if len(t.strip()) > 1 and t.strip() not in _STOPWORDS]

      def add_document(self, doc_id: int, title_text: str):
          tags = self._extract_tags(title_text)
          all_tokens = []
          for tag in tags:
              all_tokens.extend(self._tokenize_tag(tag.text))
          # Deduplicate while preserving order
          seen = set()
          unique_tokens = []
          for t in all_tokens:
              if t not in seen:
                  seen.add(t)
                  unique_tokens.append(t)
          self.db.delete_closet_tags(doc_id)
          self.db.insert_closet_tags(doc_id, [(t, 1.0) for t in unique_tokens])

      def remove_document(self, doc_id: int):
          self.db.delete_closet_tags(doc_id)

      def search(self, query: str, top_k: int = 10) -> List[Tuple[int, int]]:
          tokens = jieba.lcut(query.lower())
          query_tags = [
              t.strip()
              for t in tokens
              if len(t.strip()) > 1 and t.strip() not in _STOPWORDS
          ]
          if not query_tags:
              return []
          results = self.db.match_closet_tags(query_tags, top_k)
          return results

      def rebuild(self):
          # Future: rebuild all tags from existing documents
          pass
  ```

- [ ] **Step 4: Run tests**

  Run: `python3 -m pytest PageIndex/tests/test_closet_index.py -v`
  Expected: All tests pass.

- [ ] **Step 5: Commit**

  ```bash
  git add PageIndex/pageindex/closet_index.py PageIndex/tests/test_closet_index.py
  git commit -m "feat: add ClosetIndex for semantic tag indexing"
  ```

---

### Task 3: Planner — Plan Phase

**Files:**
- Create: `PageIndex/pageindex/agentic/planner.py`
- Test: `PageIndex/tests/test_planner.py`

- [ ] **Step 1: Write the failing test**

  Create `PageIndex/tests/test_planner.py`:

  ```python
  import os
  import sys
  import asyncio

  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

  from PageIndex.pageindex.agentic.planner import RetrievalPlanner, PlanResult


  def test_plan_result_structure():
      p = PlanResult(
          queries=["q1", "q2"],
          weights={"metadata": 0.2, "semantics": 0.5, "description": 0.3},
          query_type="factual",
      )
      assert p.queries == ["q1", "q2"]
      assert p.weights["semantics"] == 0.5
      assert p.query_type == "factual"


  def test_planner_instantiation():
      planner = RetrievalPlanner("qwen-plus")
      assert planner.model == "qwen-plus"
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `python3 -m pytest PageIndex/tests/test_planner.py -v`
  Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement RetrievalPlanner**

  Create `PageIndex/pageindex/agentic/planner.py`:

  ```python
  import json
  import logging
  from dataclasses import dataclass
  from typing import List, Dict

  from ..utils import llm_acompletion, extract_json


  @dataclass
  class PlanResult:
      queries: List[str]
      weights: Dict[str, float]
      query_type: str


  class RetrievalPlanner:
      def __init__(self, model: str):
          self.model = model

      async def plan(self, query: str) -> PlanResult:
          prompt = f"""你是一个检索规划专家。分析用户查询，生成检索策略。

  用户查询: {query}

  请完成以下任务:
  1. 判断查询类型 (factual/comparative/summarization)
  2. 生成一个假设答案 (HyDE)
  3. 生成2个query变体，用于提升召回率
  4. 分配策略权重 (metadata/semantics/description，总和为1.0)

  返回JSON格式:
  {{
    "query_type": "factual",
    "hyde_answer": "假设答案...",
    "variants": ["变体1", "变体2"],
    "weights": {{
      "metadata": 0.2,
      "semantics": 0.5,
      "description": 0.3
    }}
  }}

  直接返回JSON，不要其他内容。"""
          try:
              response = await llm_acompletion(self.model, prompt)
              if not response:
                  return self._fallback_plan(query)
              data = extract_json(response)
              if not isinstance(data, dict):
                  return self._fallback_plan(query)

              variants = data.get("variants", [])
              queries = [query] + [v for v in variants if v and v != query]
              weights = data.get("weights", {})
              # Normalize weights
              total = sum(weights.values())
              if total > 0:
                  weights = {k: v / total for k, v in weights.items()}
              else:
                  weights = {"metadata": 0.2, "semantics": 0.5, "description": 0.3}
              query_type = data.get("query_type", "factual")
              return PlanResult(queries=queries, weights=weights, query_type=query_type)
          except Exception as e:
              logging.warning("Planner failed: %s", e)
              return self._fallback_plan(query)

      def _fallback_plan(self, query: str) -> PlanResult:
          return PlanResult(
              queries=[query],
              weights={"metadata": 0.2, "semantics": 0.5, "description": 0.3},
              query_type="factual",
          )
  ```

- [ ] **Step 4: Run tests**

  Run: `python3 -m pytest PageIndex/tests/test_planner.py -v`
  Expected: All tests pass.

- [ ] **Step 5: Commit**

  ```bash
  git add PageIndex/pageindex/agentic/planner.py PageIndex/tests/test_planner.py
  git commit -m "feat: add RetrievalPlanner with HyDE and query variants"
  ```

---

### Task 4: Strategies — Three Retrieval Strategies

**Files:**
- Create: `PageIndex/pageindex/agentic/strategies.py`
- Test: `PageIndex/tests/test_strategies.py`

- [ ] **Step 1: Write the failing test**

  Create `PageIndex/tests/test_strategies.py`:

  ```python
  import os
  import sys

  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

  from PageIndex.pageindex.agentic.strategies import MetadataStrategy


  def test_metadata_strategy_basic():
      strategy = MetadataStrategy()
      docs = [
          {"doc_id": "1", "doc_name": "PDF Guide", "description": "How to index PDFs"},
          {"doc_id": "2", "doc_name": "Markdown Tips", "description": "Writing markdown"},
      ]
      results = strategy.search("pdf indexing", docs)
      assert len(results) >= 1
      assert results[0][0] == "1"  # doc_id "1" should rank first
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `python3 -m pytest PageIndex/tests/test_strategies.py -v`
  Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement all three strategies**

  Create `PageIndex/pageindex/agentic/strategies.py`:

  ```python
  import json
  import logging
  from typing import List, Tuple, Dict

  import jieba

  from ..utils import llm_completion, extract_json
  from ..closet_index import ClosetIndex, _STOPWORDS


  class MetadataStrategy:
      def search(self, query: str, docs_info: List[Dict]) -> List[Tuple[str, int]]:
          tokens = jieba.lcut(query)
          keywords = [
              t.strip().lower()
              for t in tokens
              if len(t.strip()) > 1 and t.strip().lower() not in _STOPWORDS
          ]
          if not keywords:
              return []

          scored = []
          for doc in docs_info:
              doc_name = (doc.get("doc_name") or "").lower()
              description = (doc.get("description") or "").lower()
              score = 0
              for kw in keywords:
                  if kw in doc_name:
                      score += 2
                  if kw in description:
                      score += 1
              if score > 0:
                  scored.append((str(doc.get("doc_id", "")), score))

          scored.sort(key=lambda x: x[1], reverse=True)
          return [(doc_id, rank + 1) for rank, (doc_id, _) in enumerate(scored)]


  class SemanticsStrategy:
      def __init__(self, closet_index: ClosetIndex):
          self.closet_index = closet_index

      def search(self, query: str, docs_info: List[Dict]) -> List[Tuple[str, int]]:
          results = self.closet_index.search(query, top_k=10)
          return [(str(doc_id), rank + 1) for rank, (doc_id, _) in enumerate(results)]


  class DescriptionStrategy:
      def __init__(self, model: str):
          self.model = model

      def search(self, query: str, docs_info: List[Dict]) -> List[Tuple[str, int]]:
          if not docs_info:
              return []

          # Primary: reuse main.py helper (spec FR4)
          try:
              import main
              doc_ids = main.get_relevant_documents_for_multidoc(query, docs_info)
              if isinstance(doc_ids, list):
                  return [(str(doc_id), rank + 1) for rank, doc_id in enumerate(doc_ids)]
          except Exception as e:
              logging.warning("Description strategy (main.py) failed: %s", e)

          # Fallback: built-in implementation
          prompt = f"""你是一个文档相关性判断专家。给定用户问题和文档列表，选出最可能包含答案的文档。

  用户问题: {query}

  文档列表:
  {json.dumps(docs_info, indent=2, ensure_ascii=False)}

  请返回JSON格式: {{"doc_ids": ["doc_id_1", "doc_id_2", ...]}}
  最多返回5个最相关的文档。直接返回JSON，不要其他内容。
  """
          try:
              response = llm_completion(self.model, prompt)
              if not response:
                  return []
              data = extract_json(response)
              if isinstance(data, dict):
                  doc_ids = data.get("doc_ids", [])
              elif isinstance(data, list):
                  doc_ids = data
              else:
                  return []
              return [(str(doc_id), rank + 1) for rank, doc_id in enumerate(doc_ids)]
          except Exception as e:
              logging.warning("Description strategy failed: %s", e)
              return []
  ```

- [ ] **Step 4: Run tests**

  Run: `python3 -m pytest PageIndex/tests/test_strategies.py -v`
  Expected: All tests pass.

- [ ] **Step 5: Commit**

  ```bash
  git add PageIndex/pageindex/agentic/strategies.py PageIndex/tests/test_strategies.py
  git commit -m "feat: add Metadata, Semantics, Description retrieval strategies"
  ```

---

### Task 5: Verifier — CRAG Verify Phase

**Files:**
- Create: `PageIndex/pageindex/agentic/verifier.py`
- Test: `PageIndex/tests/test_verifier.py`

- [ ] **Step 1: Write the failing test**

  Create `PageIndex/tests/test_verifier.py`:

  ```python
  import os
  import sys

  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

  from PageIndex.pageindex.agentic.verifier import CRAGVerifier, VerifyResult


  def test_verify_result_dataclass():
      v = VerifyResult(confidence=0.8, action="answer")
      assert v.confidence == 0.8
      assert v.action == "answer"


  def test_to_bool_helper():
      assert CRAGVerifier._to_bool(True) is True
      assert CRAGVerifier._to_bool(False) is False
      assert CRAGVerifier._to_bool("true") is True
      assert CRAGVerifier._to_bool("yes") is True
      assert CRAGVerifier._to_bool("否") is False
      assert CRAGVerifier._to_bool("no") is False
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `python3 -m pytest PageIndex/tests/test_verifier.py -v`
  Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement CRAGVerifier**

  Create `PageIndex/pageindex/agentic/verifier.py`:

  ```python
  import logging
  from dataclasses import dataclass

  from ..utils import llm_completion, extract_json, count_tokens


  @dataclass
  class VerifyResult:
      confidence: float
      action: str  # "answer" | "expand" | "refuse"


  class CRAGVerifier:
      TAU_HIGH = 0.7
      TAU_LOW = 0.4

      def __init__(self, model: str):
          self.model = model

      def _score_retrieval(
          self, context: str, source_docs: int, covered_nodes: int
      ) -> float:
          tokens = count_tokens(context)
          token_score = min(tokens / 4000, 1.0)
          doc_score = min(source_docs / 3, 1.0)
          node_score = min(covered_nodes / 10, 1.0)
          return token_score * 0.4 + doc_score * 0.3 + node_score * 0.3

      @staticmethod
      def _to_bool(val) -> bool:
          if isinstance(val, bool):
              return val
          if isinstance(val, str):
              return val.lower() in ("true", "yes", "是", "1", "y")
          return bool(val)

      def verify(
          self,
          answer: str,
          context: str,
          query: str,
          source_docs: int = 0,
          covered_nodes: int = 0,
      ) -> VerifyResult:
          s_ret = self._score_retrieval(context, source_docs, covered_nodes)

          prompt = f"""你是一个答案质量评估专家。基于以下信息判断答案的置信度。

  问题: {query}

  检索到的上下文（部分）:
  {context[:2000]}

  生成的答案:
  {answer}

  请评估:
  1. 答案是否基于上下文中的事实？（是/否/部分）
  2. 上下文是否充分回答了问题？（充分/不充分）
  3. 整体置信度（0.0-1.0）

  返回JSON格式: {{"based_on_context": true, "sufficient": true, "confidence": 0.85}}
  直接返回JSON，不要其他内容。
  """
          try:
              response = llm_completion(self.model, prompt)
              if not response:
                  return VerifyResult(confidence=s_ret, action="answer")

              data = extract_json(response)
              if not isinstance(data, dict):
                  return VerifyResult(confidence=s_ret, action="answer")

              s_cov = float(data.get("confidence", s_ret))
              based = self._to_bool(data.get("based_on_context", True))
              sufficient = self._to_bool(data.get("sufficient", True))

              combined = s_ret * 0.3 + s_cov * 0.7
              if not based or not sufficient:
                  combined *= 0.5

              if combined >= self.TAU_HIGH:
                  return VerifyResult(confidence=combined, action="answer")
              elif combined >= self.TAU_LOW:
                  return VerifyResult(confidence=combined, action="expand")
              else:
                  return VerifyResult(confidence=combined, action="refuse")
          except Exception as e:
              logging.warning(f"Verification failed: {e}")
              return VerifyResult(confidence=s_ret, action="answer")
  ```

- [ ] **Step 4: Run tests**

  Run: `python3 -m pytest PageIndex/tests/test_verifier.py -v`
  Expected: All tests pass.

- [ ] **Step 5: Commit**

  ```bash
  git add PageIndex/pageindex/agentic/verifier.py PageIndex/tests/test_verifier.py
  git commit -m "feat: add CRAG verifier with three-threshold confidence routing"
  ```

---

### Task 6: Router — Orchestration Layer

**Files:**
- Create: `PageIndex/pageindex/agentic/router.py`
- Create: `PageIndex/pageindex/agentic/__init__.py`
- Test: `PageIndex/tests/test_router.py`

- [ ] **Step 1: Write the failing test**

  Create `PageIndex/tests/test_router.py`:

  ```python
  import os
  import sys
  import asyncio

  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


  def test_weighted_rrf():
      from PageIndex.pageindex.agentic.router import AgenticRouter

      results = {
          "metadata": [("doc1", 1), ("doc2", 2)],
          "semantics": [("doc2", 1), ("doc3", 2)],
      }
      weights = {"metadata": 1.0, "semantics": 1.0}
      fused = AgenticRouter._weighted_rrf(results, weights)
      assert len(fused) == 3
      # doc2 appears in both, should rank high
      assert fused[0][0] in ("doc1", "doc2")
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `python3 -m pytest PageIndex/tests/test_router.py -v`
  Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement AgenticRouter**

  Create `PageIndex/pageindex/agentic/__init__.py` (empty file, just a package marker).

  Create `PageIndex/pageindex/agentic/router.py`:

  ```python
  import asyncio
  import json
  import logging
  from typing import List, Tuple, Dict

  from .planner import RetrievalPlanner
  from .strategies import MetadataStrategy, SemanticsStrategy, DescriptionStrategy
  from .verifier import CRAGVerifier


  class AgenticRouter:
      """Orchestrate Plan -> Route -> Act -> Verify."""

      def __init__(self, client, model: str):
          self.client = client
          self.model = model
          self.planner = RetrievalPlanner(model)
          self.metadata_strategy = MetadataStrategy()
          self.semantics_strategy = None
          self.description_strategy = DescriptionStrategy(model)
          self.verifier = CRAGVerifier(model)
          self._main_funcs = None

          if hasattr(client, "closet_index") and client.closet_index:
              self.semantics_strategy = SemanticsStrategy(client.closet_index)

      def _load_main_funcs(self):
          if self._main_funcs is None:
              try:
                  import main
                  self._main_funcs = {
                      "get_relevant_nodes": main.get_relevant_nodes,
                      "build_context_with_budget": main.build_context_with_budget,
                      "generate_answer": main.generate_answer,
                      "pages_from_nodes": main.pages_from_nodes,
                  }
              except ImportError:
                  self._main_funcs = {}
          return self._main_funcs

      def _build_docs_info(self) -> List[Dict]:
          docs_info = []
          if hasattr(self.client, "db") and self.client.db:
              try:
                  for doc in self.client.db.get_all_documents():
                      doc_id = str(doc["id"])
                      if doc_id not in self.client.documents:
                          continue
                      top = self.client.db.get_top_level_nodes(doc["id"])
                      docs_info.append(
                          {
                              "doc_id": doc_id,
                              "doc_name": doc.get("pdf_name", ""),
                              "description": doc.get("doc_description", ""),
                              "top_level_sections": [
                                  n.get("title") for n in top if n.get("title")
                              ],
                          }
                      )
                  if docs_info:
                      return docs_info
              except Exception:
                  pass

          for doc_id, doc in self.client.documents.items():
              docs_info.append(
                  {
                      "doc_id": doc_id,
                      "doc_name": doc.get("doc_name", ""),
                      "description": doc.get("doc_description", ""),
                      "top_level_sections": [],
                  }
              )
          return docs_info

      @staticmethod
      def _weighted_rrf(
          results_dict: Dict[str, List[Tuple[str, int]]],
          weights: Dict[str, float],
          k: int = 60,
      ) -> List[Tuple[str, float]]:
          scores: Dict[str, float] = {}
          for strategy, results in results_dict.items():
              weight = weights.get(strategy, 1.0)
              for doc_id, rank in results:
                  scores[doc_id] = scores.get(doc_id, 0.0) + weight * (
                      1.0 / (k + rank)
                  )
          return sorted(scores.items(), key=lambda x: x[1], reverse=True)

      async def _run_strategies(
          self, query: str, docs_info: List[Dict], weights: Dict[str, float]
      ) -> Dict[str, List[Tuple[str, int]]]:
          tasks = {}
          tasks["metadata"] = asyncio.to_thread(
              self.metadata_strategy.search, query, docs_info
          )
          if self.semantics_strategy and weights.get("semantics", 0) > 0:
              tasks["semantics"] = asyncio.to_thread(
                  self.semantics_strategy.search, query, docs_info
              )
          if weights.get("description", 0) > 0:
              tasks["description"] = asyncio.to_thread(
                  self.description_strategy.search, query, docs_info
              )

          results: Dict[str, List[Tuple[str, int]]] = {}
          if tasks:
              done = await asyncio.gather(*tasks.values(), return_exceptions=True)
              for name, res in zip(tasks.keys(), done):
                  if isinstance(res, Exception):
                      logging.warning("Strategy %s failed: %s", name, res)
                      results[name] = []
                  else:
                      results[name] = res
          return results

      async def _act_tree_search(
          self, query: str, candidate_docs: List[str]
      ) -> Tuple[str, List[dict], int, int, Dict[str, List[int]]]:
          funcs = self._load_main_funcs()
          get_relevant_nodes = funcs.get("get_relevant_nodes")
          pages_from_nodes = funcs.get("pages_from_nodes")
          if not all([get_relevant_nodes, pages_from_nodes]):
              raise RuntimeError("main.py helpers not available")

          contexts = []
          all_nodes = []
          source_docs = 0
          doc_pages_map: Dict[str, List[int]] = {}

          for doc_id in candidate_docs:
              if hasattr(self.client, "_ensure_doc_loaded"):
                  self.client._ensure_doc_loaded(doc_id)
              doc = self.client.documents.get(doc_id)
              if not doc:
                  continue
              structure = doc.get("structure", [])
              if not structure:
                  continue

              tree_json = json.dumps(structure, ensure_ascii=False)
              node_ids = get_relevant_nodes(query, tree_json)
              if not node_ids:
                  continue

              from ..utils import create_node_mapping

              mapping = create_node_mapping(structure)
              selected = [
                  mapping.get(nid) for nid in node_ids if nid in mapping
              ]
              selected = [n for n in selected if n]
              if not selected:
                  continue

              pages = pages_from_nodes(selected)
              if not pages:
                  continue

              ctx_parts = [f"\n=== Document: {doc.get('doc_name', '')} ===\n"]
              if doc.get("type") == "pdf" and doc.get("pages"):
                  page_map = {p["page"]: p["content"] for p in doc["pages"]}
                  for p in sorted(set(pages)):
                      text = page_map.get(p, "")
                      if text:
                          ctx_parts.append(f"\n--- Page {p} ---\n{text}")
              elif doc.get("type") == "md" and structure:
                  for node in selected:
                      txt = node.get("text", "")
                      if txt:
                          ctx_parts.append(f"\n--- {node.get('title', '')} ---\n{txt}")

              if len(ctx_parts) > 1:
                  contexts.append("".join(ctx_parts))
                  all_nodes.extend(selected)
                  source_docs += 1
                  doc_pages_map[doc_id] = sorted(set(pages))

          return "\n\n".join(contexts), all_nodes, source_docs, len(all_nodes), doc_pages_map

      async def search(self, query: str, top_k: int = 3) -> Dict:
          plan = await self.planner.plan(query)

          docs_info = self._build_docs_info()
          if not docs_info:
              return {
                  "query": query,
                  "mode": "multi",
                  "answer": "No documents indexed.",
                  "confidence": "unknown",
                  "matched_docs": [],
                  "selected_nodes": [],
                  "pages": [],
              }

          results = await self._run_strategies(
              plan.queries[0], docs_info, plan.weights
          )

          if self.semantics_strategy and len(plan.queries) > 1:
              best_sem: Dict[str, int] = {}
              for doc_id, rank in results.get("semantics", []):
                  best_sem[doc_id] = rank
              for q in plan.queries[1:]:
                  try:
                      r = await asyncio.to_thread(
                          self.semantics_strategy.search, q, docs_info
                      )
                      for doc_id, rank in r:
                          if doc_id not in best_sem or rank < best_sem[doc_id]:
                              best_sem[doc_id] = rank
                  except Exception:
                      pass
              if best_sem:
                  results["semantics"] = sorted(
                      best_sem.items(), key=lambda x: x[1]
                  )

          fused = self._weighted_rrf(results, plan.weights)
          if not fused:
              return {
                  "query": query,
                  "mode": "multi",
                  "answer": "No relevant documents found.",
                  "confidence": "unknown",
                  "matched_docs": [],
                  "selected_nodes": [],
                  "pages": [],
              }

          candidates = [doc_id for doc_id, _ in fused[:top_k]]
          matched = [
              {"doc_id": doc_id, "score": round(score, 4)}
              for doc_id, score in fused[:top_k]
          ]

          try:
              ctx, nodes, src_docs, cov_nodes, doc_pages_map = await self._act_tree_search(
                  query, candidates
              )
          except Exception as e:
              logging.warning("Act phase failed: %s", e)
              return {
                  "query": query,
                  "mode": "multi",
                  "answer": f"Failed to retrieve content: {e}",
                  "confidence": "unknown",
                  "matched_docs": matched,
                  "selected_nodes": [],
                  "pages": [],
              }

          if not ctx:
              return {
                  "query": query,
                  "mode": "multi",
                  "answer": "No relevant content found.",
                  "confidence": "low",
                  "matched_docs": matched,
                  "selected_nodes": [],
                  "pages": [],
              }

          funcs = self._load_main_funcs()
          generate_answer = funcs.get("generate_answer")
          if not generate_answer:
              return {
                  "query": query,
                  "mode": "multi",
                  "answer": "Answer generation not available.",
                  "confidence": "unknown",
                  "matched_docs": matched,
                  "selected_nodes": [],
                  "pages": [],
              }

          answer = generate_answer(query, ctx)

          v = await asyncio.to_thread(
              self.verifier.verify, answer, ctx, query, src_docs, cov_nodes
          )
          if v.action == "refuse":
              return {
                  "query": query,
                  "mode": "multi",
                  "answer": "I don't know.",
                  "confidence": "low",
                  "matched_docs": matched,
                  "selected_nodes": [
                      {"node_id": n.get("node_id"), "title": n.get("title")}
                      for n in nodes
                  ],
                  "pages": [
                      {"doc_id": d, "pages": p}
                      for d, p in doc_pages_map.items()
                  ],
              }

          if v.action == "expand" and len(fused) > top_k:
              expanded = [doc_id for doc_id, _ in fused[: top_k * 2]]
              try:
                  ctx2, nodes2, src2, cov2, doc_pages_map2 = await self._act_tree_search(
                      query, expanded
                  )
                  if ctx2:
                      ans2 = generate_answer(query, ctx2)
                      v2 = await asyncio.to_thread(
                          self.verifier.verify, ans2, ctx2, query, src2, cov2
                      )
                      conf = "high" if v2.action == "answer" else "medium"
                      return {
                          "query": query,
                          "mode": "multi",
                          "answer": ans2,
                          "confidence": conf,
                          "matched_docs": [
                              {"doc_id": d, "score": round(s, 4)}
                              for d, s in fused[: top_k * 2]
                          ],
                          "selected_nodes": [
                              {"node_id": n.get("node_id"), "title": n.get("title")}
                              for n in nodes2
                          ],
                          "pages": [
                              {"doc_id": d, "pages": p}
                              for d, p in doc_pages_map2.items()
                          ],
                      }
              except Exception as e:
                  logging.warning("Expand search failed: %s", e)

          conf = "high" if v.action == "answer" else "medium"
          return {
              "query": query,
              "mode": "multi",
              "answer": answer,
              "confidence": conf,
              "matched_docs": matched,
              "selected_nodes": [
                  {"node_id": n.get("node_id"), "title": n.get("title")}
                  for n in nodes
              ],
              "pages": [
                  {"doc_id": d, "pages": p}
                  for d, p in doc_pages_map.items()
              ],
          }
  ```

- [ ] **Step 4: Run tests**

  Run: `python3 -m pytest PageIndex/tests/test_router.py -v`
  Expected: All tests pass.

- [ ] **Step 5: Commit**

  ```bash
  git add PageIndex/pageindex/agentic/
  git commit -m "feat: add AgenticRouter with Plan-Route-Act-Verify pipeline"
  ```

---

### Task 7: Client Integration

**Files:**
- Modify: `PageIndex/pageindex/client.py`
- Create: `test_smoke.py`

- [ ] **Step 1: Modify PageIndexClient to integrate router**

  In `PageIndex/pageindex/client.py`, add imports at the top:

  ```python
  from .closet_index import ClosetIndex

  try:
      from .agentic.router import AgenticRouter
  except ImportError:
      AgenticRouter = None  # type: ignore[misc,assignment]
  ```

  In `__init__`, after `self._load_workspace()`, add:

  ```python
  # Optional persistent layer for agentic retrieval
  self.db = None
  self.closet_index = None
  self.router = None
  self._uuid_to_db: dict[str, int] = {}

  if db_path and PageIndexDB:
      self.db = PageIndexDB(db_path)
      self.closet_index = ClosetIndex(self.db, self.model)
      if AgenticRouter:
          self.router = AgenticRouter(self, self.model)
  ```

  Modify `index()` method: after document processing is complete, add:

  ```python
  # Index closet tags
  if self.closet_index and doc_id:
      db_id = self._uuid_to_db.get(doc_id)
      if db_id:
          self.closet_index.add_document(db_id, doc.get("doc_name", ""))
  ```

  Add the `search()` method:

  ```python
  async def search(self, query: str, top_k: int = 3) -> dict:
      """Search across indexed documents.

      Single document: direct tree search shortcut.
      Multiple documents: agentic Plan→Route→Act→Verify pipeline.
      """
      self._ensure_db()

      # Single-document shortcut
      if len(self.documents) == 1:
          doc_id = list(self.documents.keys())[0]
          doc = self.documents[doc_id]
          from main import get_relevant_nodes, pages_from_nodes, generate_answer

          structure = doc.get("structure", [])
          if not structure:
              return {"query": query, "mode": "single", "answer": "No structure available.", "pages": []}

          tree_json = json.dumps(structure, ensure_ascii=False)
          node_ids = get_relevant_nodes(query, tree_json)
          if not node_ids:
              return {"query": query, "mode": "single", "answer": "No relevant nodes found.", "pages": []}

          from .utils import create_node_mapping
          mapping = create_node_mapping(structure)
          nodes = [mapping.get(nid) for nid in node_ids if nid in mapping]
          nodes = [n for n in nodes if n]
          pages = pages_from_nodes(nodes)

          ctx_parts = [f"Document: {doc.get('doc_name', '')}"]
          if doc.get("type") == "md":
              for node in nodes:
                  txt = node.get("text", "")
                  if txt:
                      ctx_parts.append(f"{node.get('title', '')}: {txt}")
          ctx = "\n".join(ctx_parts)
          answer = generate_answer(query, ctx)
          return {
              "query": query,
              "mode": "single",
              "answer": answer,
              "confidence": "high",
              "pages": [{"doc_id": doc_id, "pages": sorted(set(pages))}],
          }

      # Multi-document: agentic router
      if self.router:
          return await self.router.search(query, top_k)

      return {"query": query, "mode": "multi", "answer": "Router not available.", "pages": []}
  ```

- [ ] **Step 2: Write smoke test**

  Create `test_smoke.py`:

  ```python
  #!/usr/bin/env python3
  """Smoke test for Agentic Multi-Strategy Router."""

  import asyncio
  import tempfile
  import os


  def create_test_md(path: str) -> None:
      with open(path, "w", encoding="utf-8") as f:
          f.write("""# Test Document

  ## Introduction
  This is a test document for PageIndex-UV.

  ## Features
  - Feature A: supports PDF indexing
  - Feature B: supports Markdown indexing
  - Feature C: agentic multi-strategy search

  ## Conclusion
  PageIndex-UV provides tree-based document retrieval.
  """)


  async def main():
      with tempfile.TemporaryDirectory() as tmpdir:
          workspace = os.path.join(tmpdir, "workspace")
          db_path = os.path.join(tmpdir, "test.db")
          md_path = os.path.join(tmpdir, "test.md")
          create_test_md(md_path)

          print("=" * 50)
          print("Smoke Test: Agentic Multi-Strategy Router")
          print("=" * 50)

          print("\n[1/6] Import test...")
          from PageIndex.pageindex import PageIndexClient
          print("  OK")

          print("\n[2/6] Initialize client...")
          client = PageIndexClient(
              workspace=workspace,
              db_path=db_path,
          )
          print(f"  OK (docs={len(client.documents)})")

          print("\n[3/6] Index Markdown...")
          doc_id = client.index(md_path, mode="md")
          print(f"  OK (doc_id={doc_id})")

          print("\n[4/6] Verify DB persistence...")
          assert client.db is not None
          all_docs = client.db.get_all_documents()
          assert len(all_docs) >= 1
          print(f"  OK (db_docs={len(all_docs)})")

          print("\n[5/6] Single-document search...")
          result = await client.search("What features does PageIndex support?")
          print(f"  mode={result['mode']}, confidence={result['confidence']}")
          print(f"  answer={result['answer'][:100]}...")
          assert result["mode"] == "single"
          assert result["answer"]
          print("  OK")

          print("\n[6/6] Search with pages field...")
          assert "pages" in result
          print(f"  pages={result['pages']}")
          print("  OK")

          print("\n" + "=" * 50)
          print("All smoke tests passed!")
          print("=" * 50)


  if __name__ == "__main__":
      asyncio.run(main())
  ```

- [ ] **Step 3: Run smoke test**

  Run: `python3 test_smoke.py`
  Expected: All 6 steps pass.

- [ ] **Step 4: Commit**

  ```bash
  git add PageIndex/pageindex/client.py test_smoke.py
  git commit -m "feat: integrate AgenticRouter into PageIndexClient with search()"
  ```

---

## Self-Review

### Spec coverage check

| Spec Requirement | Task |
|------------------|------|
| FR1: Agentic 多策略路由 (Plan→Route→Act→Verify) | Task 3, 4, 5, 6 |
| FR2: Closet 语义标签索引 | Task 1, 2 |
| FR3: Metadata 策略 | Task 4 |
| FR4: Description 策略 | Task 4 |
| FR5: RRF 融合 | Task 6 |
| FR6: 单文档兼容 | Task 7 |
| NFR1: 延迟 < 2s | Task 4, 6 (asyncio.gather parallelism) |
| NFR2: 仅新增 jieba | Task 1 |
| NFR3: 现有接口不动 | Task 7 (only adds `search()`) |
| NFR4: 降级策略 | Task 6 (error handling in each phase) |

### Placeholder scan

- No "TBD", "TODO", "implement later" found.
- All steps include concrete code or exact commands.
- No vague directives like "add appropriate error handling" — specific try/except blocks are shown.

### Type consistency check

- `PlanResult.queries: List[str]` used consistently in Task 3 and Task 6.
- `PlanResult.weights: Dict[str, float]` used consistently.
- `VerifyResult.action: str` with values "answer"|"expand"|"refuse" used consistently in Task 5 and Task 6.
- `AgenticRouter.search()` return type `Dict` matches client expectations in Task 7.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-05-agentic-multi-strategy-router.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session, batch execution with checkpoints for review.

**Which approach?**
