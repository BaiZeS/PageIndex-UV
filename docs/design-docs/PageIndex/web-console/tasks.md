# Web 控制台 Implementation Plan — tasks.md

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 一个零构建的单页 Web 控制台，复用 server.py，图形化 CLI 三动作（索引/列文档/问答）+ 模型配置热切换与持久化。

**Architecture:** 复用 `server.py`（Starlette :3000）：新增 4 个 REST 端点 + 静态托管 + 鉴权放行；磁盘落盘逻辑抽到 `web_config.py`（纯函数，显式路径，易测）；前端单文件 `web/index.html` + `web/static/{app.js,styles.css}`，CDN 引入 Vue3 + ElementPlus。

**Tech Stack:** Python (Starlette, python-dotenv `set_key`, PyYAML), Vue 3 + Element Plus（CDN，零构建），pytest + Starlette TestClient。

**Spec:** `docs/design-docs/PageIndex/web-console/spec.md`

## Global Constraints

- **零前端构建**：不引入 node/Vite；Vue3 + ElementPlus 走 CDN（`https://unpkg.com`）。
- **定点落盘**：写 `config.yaml` 用行替换保注释；写 `.env` 用 `dotenv.set_key`；**写前必备份** `*.bak`。
- **掩码**：GET 端点永不回显完整 key。
- **TDD**：每个后端任务先 RED（写失败测试、肉眼确认失败）再 GREEN。
- **不破坏现有**：`/health`、`/upload`、MCP `/sse`、`/messages/` 保持不变。
- **测试隔离**：所有测试用 `tmp_path`，绝不写真实 `.env` / `config.yaml`。

## File Structure

| 文件 | 职责 |
|------|------|
| `web_config.py`（新） | 纯函数：掩码/备份/`config.yaml` 行替换/`.env` set_key/读快照。显式路径参数，零全局。 |
| `server.py`（改） | 4 端点 + 静态挂载 + 鉴权放行 + `_rebuild_client` + lifespan 捕获 workspace/db_path。 |
| `web/index.html`（新） | 单页入口：Vue3+ElementPlus(CDN) + 三 Tabs + key 输入框。 |
| `web/static/app.js`（新） | Vue app：fetch 封装（带 X-API-Key）、文档/问答/配置交互。 |
| `web/static/styles.css`（新） | 自定义样式（实现期 frontend-design 打磨）。 |
| `tests/test_web_console.py`（新） | 后端 TDD 测试。 |

`web_config.py` 默认路径（基于文件位置解析，CWD 无关）：
```python
DEFAULT_CONFIG_YAML = Path(__file__).resolve().parent / "pageindex_mutil" / "config.yaml"
DEFAULT_ENV = Path(__file__).resolve().parent / ".env"
```

---

### Task 1: `web_config.py` — 落盘纯函数（掩码/备份/行替换/写 .env）

**Files:**
- Create: `web_config.py`
- Test: `tests/test_web_console.py`（本任务首次创建）

**Interfaces:**
- Produces: `mask_key(key) -> str`, `backup_file(path) -> Path`, `set_config_yaml_model(path, *, model=None, retrieve_model=None) -> None`, `set_env_fields(env_path, fields: dict) -> None`, `read_yaml_model(path) -> dict`, `DEFAULT_CONFIG_YAML`, `DEFAULT_ENV`.

- [ ] **Step 1: Write the failing test**

`tests/test_web_console.py`:
```python
import textwrap
from pathlib import Path

from web_config import (
    mask_key, backup_file, set_config_yaml_model, set_env_fields, read_yaml_model,
)


def test_mask_key_hides_middle():
    assert mask_key("sk-fake000-not-a-real-key-9999") == "sk-fake0****9999"
    assert mask_key("short") == "*****"
    assert mask_key(None) == ""


def test_backup_file_creates_bak(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("hello", encoding="utf-8")
    bak = backup_file(f)
    assert bak.exists() and bak.read_text(encoding="utf-8") == "hello"
    assert bak.name == "c.yaml.bak"


def test_set_config_yaml_model_replaces_only_target_lines_and_keeps_comments(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent('''\
        # keep this comment
        model: "old-model"
        retrieve_model: "old-rm"
        if_thinning: false
        # MODEL_NAME overrides model
    '''), encoding="utf-8")
    set_config_yaml_model(p, model="new-model", retrieve_model="new-rm")
    data = read_yaml_model(p)
    assert data["model"] == "new-model"
    assert data["retrieve_model"] == "new-rm"
    assert data["if_thinning"] is False
    text = p.read_text(encoding="utf-8")
    assert "# keep this comment" in text
    assert "# MODEL_NAME overrides model" in text


def test_set_env_fields_sets_keys(tmp_path):
    env = tmp_path / ".env"
    env.write_text('OPENAI_API_KEY=old\n# OPENAI_BASE_URL=x\n', encoding="utf-8")
    set_env_fields(env, {"OPENAI_API_KEY": "new-sk", "OPENAI_BASE_URL": "http://h/"})
    from dotenv import dotenv_values
    vals = dotenv_values(env)
    assert vals["OPENAI_API_KEY"] == "new-sk"
    assert vals["OPENAI_BASE_URL"] == "http://h/"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_console.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'web_config'`.

- [ ] **Step 3: Write minimal implementation**

`web_config.py`:
```python
"""Disk-persistence helpers for the web console config endpoints.

Pure functions taking explicit paths — no server/global imports — so they are
trivially unit-testable with tmp_path fixtures and never touch the real
.env / config.yaml unless called with those exact paths at runtime.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import yaml
from dotenv import set_key

DEFAULT_CONFIG_YAML = Path(__file__).resolve().parent / "pageindex_mutil" / "config.yaml"
DEFAULT_ENV = Path(__file__).resolve().parent / ".env"


def mask_key(key: Optional[str]) -> str:
    """Return a masked copy: first 8 + '****' + last 4. Short keys fully hidden."""
    if not key:
        return ""
    if len(key) <= 12:
        return "*" * len(key)
    return key[:8] + "****" + key[-4:]


def backup_file(path: Path) -> Path:
    """Copy path -> path.with_suffix(<suffix>+'.bak'). Overwrites existing bak."""
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    return bak


def read_yaml_model(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def set_config_yaml_model(path: Path, *, model: Optional[str] = None,
                          retrieve_model: Optional[str] = None) -> None:
    """Targeted line replacement of `model:` / `retrieve_model:` lines.

    Preserves every comment and all other lines (yaml.dump would nuke comments).
    Only rewrites lines whose stripped text starts with the exact key + ':'.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        nl = "\n" if line.endswith("\n") else ""
        if model is not None and stripped.startswith("model:") and not stripped.startswith("model_"):
            lines[i] = f'{indent}model: "{model}"{nl}'
        elif retrieve_model is not None and stripped.startswith("retrieve_model:"):
            lines[i] = f'{indent}retrieve_model: "{retrieve_model}"{nl}'
    path.write_text("".join(lines), encoding="utf-8")


def set_env_fields(env_path: Path, fields: dict) -> None:
    """set_key each field: uncomments/updates/appends, preserves other lines."""
    for k, v in fields.items():
        set_key(str(env_path), k, str(v))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web_console.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add web_config.py tests/test_web_console.py
git commit -m "feat(web): web_config disk helpers (mask/backup/yaml/env)"
```

---

### Task 2: `web_config.py` — 读快照 + provider 派生

**Files:**
- Modify: `web_config.py`
- Test: `tests/test_web_console.py`

**Interfaces:**
- Produces: `derive_provider() -> str`, `read_config_snapshot(client, config_yaml_path=DEFAULT_CONFIG_YAML, env_path=DEFAULT_ENV) -> dict`（spec §5.2 结构）。
- Consumes: `pageindex_mutil.utils.get_llm_config`, `ConfigLoader`, Task 1 helpers.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_web_console.py`:
```python
from types import SimpleNamespace

import web_config
from web_config import read_config_snapshot, derive_provider


class _FakeDB:
    def get_all_documents(self):
        return [{"id": 1}, {"id": 2}]


def test_derive_provider(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    assert derive_provider() == "none"
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-x")
    assert derive_provider() == "dashscope"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-y")
    assert derive_provider() == "openai"


def test_read_snapshot_masks_key_and_reads_yaml(tmp_path, monkeypatch):
    yaml_p = tmp_path / "config.yaml"
    yaml_p.write_text('model: "gpt-4.1-mini"\nretrieve_model: "gpt-4.1-mini"\n', encoding="utf-8")
    env_p = tmp_path / ".env"
    env_p.write_text("", encoding="utf-8")
    monkeypatch.setattr(web_config, "get_llm_config", lambda: ("sk-fake000-not-a-real-key-9999", "http://h/v1"))
    monkeypatch.setattr(web_config, "ConfigLoader", lambda: SimpleNamespace(load=lambda _: SimpleNamespace(model="gpt-4.1-mini", retrieve_model=None)))
    client = SimpleNamespace(db=_FakeDB())
    snap = read_config_snapshot(client, config_yaml_path=yaml_p, env_path=env_p)
    assert snap["api_key_masked"] == "sk-fake0****9999"
    assert snap["has_api_key"] is True
    assert snap["document_count"] == 2
    assert snap["model_yaml"] == "gpt-4.1-mini"
    assert snap["retrieve_model"] == "gpt-4.1-mini"  # falls back to model
    assert "provider" in snap and "base_url" in snap and "env_overrides" in snap
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_console.py -v`
Expected: FAIL — `ImportError: cannot import name 'derive_provider'` / `read_config_snapshot`.

- [ ] **Step 3: Write minimal implementation**

Append to `web_config.py`:
```python
import os
from typing import Any

from dotenv import dotenv_values

from pageindex_mutil.utils import get_llm_config, ConfigLoader


def derive_provider() -> str:
    """openai | dashscope | none — mirrors _resolve_llm_config key priority."""
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("DASHSCOPE_API_KEY"):
        return "dashscope"
    return "none"


def read_config_snapshot(client: Any, config_yaml_path: Path = DEFAULT_CONFIG_YAML,
                          env_path: Path = DEFAULT_ENV) -> dict:
    key, url = get_llm_config()
    yaml_data = read_yaml_model(config_yaml_path) if Path(config_yaml_path).exists() else {}
    opt = ConfigLoader().load(None)
    doc_count = 0
    db = getattr(client, "db", None)
    if db is not None:
        try:
            doc_count = len(db.get_all_documents())
        except Exception:
            pass
    return {
        "model": opt.model,
        "retrieve_model": getattr(opt, "retrieve_model", None) or opt.model,
        "model_yaml": yaml_data.get("model"),
        "retrieve_model_yaml": yaml_data.get("retrieve_model"),
        "provider": derive_provider(),
        "base_url": url,
        "api_key_masked": mask_key(key),
        "has_api_key": bool(key),
        "document_count": doc_count,
        "env_overrides": {
            "MODEL_NAME": os.getenv("MODEL_NAME"),
            "RETRIEVE_MODEL_NAME": os.getenv("RETRIEVE_MODEL_NAME"),
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web_console.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add web_config.py tests/test_web_console.py
git commit -m "feat(web): config snapshot reader + provider derivation"
```

---

### Task 3: `GET /api/documents` 端点 + 鉴权放行 `/`,`/static`

**Files:**
- Modify: `server.py`（imports, `APIKeyMiddleware.dispatch`, routes）
- Test: `tests/test_web_console.py`

**Interfaces:**
- Produces: `GET /api/documents -> {"documents": [...]}`.
- Consumes: `get_client().db.get_all_documents()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_web_console.py`:
```python
import server


def _client_with_docs(monkeypatch, docs):
    fake_db = SimpleNamespace(get_all_documents=lambda: docs)
    monkeypatch.setattr(server, "get_client", lambda: SimpleNamespace(db=fake_db))
    monkeypatch.setattr(server, "API_KEY", "")
    from starlette.testclient import TestClient
    return TestClient(server.app)


def test_documents_endpoint(monkeypatch):
    docs = [{"id": 1, "pdf_name": "a.pdf", "doc_description": "d", "pdf_path": "/x/a.pdf"}]
    c = _client_with_docs(monkeypatch, docs)
    r = c.get("/api/documents")
    assert r.status_code == 200
    body = r.json()
    assert body["documents"][0]["pdf_name"] == "a.pdf"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_console.py::test_documents_endpoint -v`
Expected: FAIL — 404 (route not registered).

- [ ] **Step 3: Write minimal implementation**

In `server.py`:
1. Add imports near top:
```python
from starlette.staticfiles import StaticFiles
from starlette.responses import JSONResponse, FileResponse
import web_config
```
2. Add endpoint function near `health_endpoint` (after it):
```python
async def documents_endpoint(request: Request) -> Response:
    c = get_client()
    docs = []
    if c.db is not None:
        try:
            docs = c.db.get_all_documents()
        except Exception:
            logger.exception("documents_endpoint failed")
    safe = [
        {
            "id": d.get("id"),
            "doc_name": d.get("pdf_name"),
            "doc_description": d.get("doc_description"),
            "pdf_path": d.get("pdf_path"),
        }
        for d in docs
    ]
    return JSONResponse({"documents": safe})
```
3. In `APIKeyMiddleware.dispatch`, extend the skip block:
```python
        # Skip auth for health + static console (page itself is public; API still gated)
        if request.url.path == "/health" \
                or request.url.path == "/" \
                or request.url.path.startswith("/static"):
            return await call_next(request)
```
4. Add route to `routes` list:
```python
    Route("/api/documents", endpoint=documents_endpoint, methods=["GET"]),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web_console.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_web_console.py
git commit -m "feat(web): GET /api/documents + auth whitelist for / and /static"
```

---

### Task 4: `POST /api/search` 端点

**Files:**
- Modify: `server.py`, Test: `tests/test_web_console.py`

**Interfaces:**
- Produces: `POST /api/search` body `{query, top_k?}` → `client.search()` 结果。

- [ ] **Step 1: Write the failing test**

Append:
```python
def test_search_endpoint(monkeypatch):
    async def fake_search(query, top_k=3):
        return {"query": query, "answer": "A1", "confidence": "high",
                "matched_docs": [], "selected_nodes": [], "pages": []}
    monkeypatch.setattr(server, "get_client",
                        lambda: SimpleNamespace(search=fake_search, db=None))
    monkeypatch.setattr(server, "API_KEY", "")
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.post("/api/search", json={"query": "what is x", "top_k": 5})
    assert r.status_code == 200
    assert r.json()["answer"] == "A1"


def test_search_rejects_missing_query(monkeypatch):
    monkeypatch.setattr(server, "get_client", lambda: SimpleNamespace(db=None))
    monkeypatch.setattr(server, "API_KEY", "")
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.post("/api/search", json={})
    assert r.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_console.py::test_search_endpoint -v`
Expected: FAIL — 404.

- [ ] **Step 3: Write minimal implementation**

Add endpoint to `server.py`:
```python
async def search_endpoint(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    query = (body or {}).get("query")
    if not query or not str(query).strip():
        return JSONResponse({"error": "Missing 'query'"}, status_code=400)
    top_k = body.get("top_k", 3)
    try:
        result = await get_client().search(str(query).strip(), top_k=top_k)
    except Exception as e:
        logger.exception("search_endpoint failed")
        return JSONResponse({"error": f"Search failed: {e}"}, status_code=500)
    return JSONResponse(result)
```
Add route:
```python
    Route("/api/search", endpoint=search_endpoint, methods=["POST"]),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web_console.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_web_console.py
git commit -m "feat(web): POST /api/search endpoint"
```

---

### Task 5: `GET /api/config` 端点

**Files:**
- Modify: `server.py`, Test: `tests/test_web_console.py`

**Interfaces:**
- Produces: `GET /api/config` → spec §5.2 快照（来自 Task 2 `read_config_snapshot`）。

- [ ] **Step 1: Write the failing test**

Append:
```python
def test_config_get_endpoint_masks_key(monkeypatch):
    monkeypatch.setattr(web_config, "read_config_snapshot",
                        lambda client, **kw: {"api_key_masked": "sk-1234****5678",
                                              "has_api_key": True, "model": "m"})
    monkeypatch.setattr(server, "get_client", lambda: SimpleNamespace(db=None))
    monkeypatch.setattr(server, "API_KEY", "")
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert body["api_key_masked"] == "sk-1234****5678"
    assert "api_key" not in body or not body.get("api_key")  # never raw key
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_console.py::test_config_get_endpoint_masks_key -v`
Expected: FAIL — 404.

- [ ] **Step 3: Write minimal implementation**

Add endpoint to `server.py`:
```python
async def config_get_endpoint(request: Request) -> Response:
    snap = web_config.read_config_snapshot(get_client())
    return JSONResponse(snap)
```
Add route:
```python
    Route("/api/config", endpoint=config_get_endpoint, methods=["GET"]),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web_console.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_web_console.py
git commit -m "feat(web): GET /api/config endpoint (key masked)"
```

---

### Task 6: `POST /api/config` 端点 + `_rebuild_client`（热切换+持久化+回滚）

**Files:**
- Modify: `server.py`, Test: `tests/test_web_console.py`

**Interfaces:**
- Produces: `POST /api/config` body `{model?, retrieve_model?, api_key?, base_url?, persist?}`；`_rebuild_client(model, retrieve_model)`；返回回读快照。
- Consumes: `web_config.{backup_file, set_config_yaml_model, set_env_fields, read_config_snapshot}`；`configure_llm`；`PageIndexClient`。

**运行时语义**（spec §5.4）：凭据→`configure_llm()`（共享 client，即时生效）；模型名→重建全局 client；持久化→备份→定点改（config.yaml+env）→回读校验，失败回滚 `.bak`。

- [ ] **Step 1: Write the failing test**

Append:
```python
def test_config_post_dualwrite_model(monkeypatch, tmp_path):
    yaml_p = tmp_path / "config.yaml"
    yaml_p.write_text('model: "old"\nretrieve_model: "old"\n', encoding="utf-8")
    env_p = tmp_path / ".env"
    env_p.write_text("MODEL_NAME=old\n", encoding="utf-8")
    monkeypatch.setattr(web_config, "DEFAULT_CONFIG_YAML", yaml_p)
    monkeypatch.setattr(web_config, "DEFAULT_ENV", env_p)
    calls = {}
    monkeypatch.setattr(server, "_rebuild_client", lambda **kw: calls.update(kw))
    monkeypatch.setattr(web_config, "read_config_snapshot", lambda client, **kw: {"model": "new"})
    monkeypatch.setattr(server, "get_client", lambda: SimpleNamespace(db=None))
    monkeypatch.setattr(server, "API_KEY", "")
    monkeypatch.delenv("MODEL_NAME", raising=False)
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.post("/api/config", json={"model": "new-model"})
    assert r.status_code == 200
    assert r.json()["applied"] is True and r.json()["persisted"] is True
    assert calls.get("model") == "new-model"
    # runtime env sync: os.environ updated so read-back snapshot is consistent
    import os as _os
    assert _os.environ.get("MODEL_NAME") == "new-model"
    # dual write: yaml + env both updated
    assert "new-model" in yaml_p.read_text(encoding="utf-8")
    from dotenv import dotenv_values
    assert dotenv_values(env_p)["MODEL_NAME"] == "new-model"
    assert (yaml_p.with_suffix(".yaml.bak")).exists()


def test_config_post_credentials_env_and_runtime(monkeypatch, tmp_path):
    env_p = tmp_path / ".env"
    env_p.write_text("OPENAI_API_KEY=old\nOPENAI_BASE_URL=http://o/\n", encoding="utf-8")
    monkeypatch.setattr(web_config, "DEFAULT_CONFIG_YAML", tmp_path / "config.yaml")
    monkeypatch.setattr(web_config, "DEFAULT_ENV", env_p)
    llm_calls = {}
    monkeypatch.setattr(server, "configure_llm",
                        lambda api_key=None, base_url=None: llm_calls.update(dict(api_key=api_key, base_url=base_url)))
    monkeypatch.setattr(web_config, "read_config_snapshot", lambda client, **kw: {})
    monkeypatch.setattr(server, "get_client", lambda: SimpleNamespace(db=None))
    monkeypatch.setattr(server, "API_KEY", "")
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.post("/api/config", json={"api_key": "sk-new", "base_url": "http://n/v1"})
    assert r.status_code == 200
    assert llm_calls["api_key"] == "sk-new" and llm_calls["base_url"] == "http://n/v1"
    from dotenv import dotenv_values
    vals = dotenv_values(env_p)
    assert vals["OPENAI_API_KEY"] == "sk-new" and vals["OPENAI_BASE_URL"] == "http://n/v1"


def test_config_post_persist_false_runtime_only(monkeypatch, tmp_path):
    env_p = tmp_path / ".env"
    env_p.write_text("MODEL_NAME=old\n", encoding="utf-8")
    monkeypatch.setattr(web_config, "DEFAULT_CONFIG_YAML", tmp_path / "config.yaml")
    monkeypatch.setattr(web_config, "DEFAULT_ENV", env_p)
    monkeypatch.setattr(server, "_rebuild_client", lambda **kw: None)
    monkeypatch.setattr(web_config, "read_config_snapshot", lambda client, **kw: {})
    monkeypatch.setattr(server, "get_client", lambda: SimpleNamespace(db=None))
    monkeypatch.setattr(server, "API_KEY", "")
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.post("/api/config", json={"model": "x", "persist": False})
    assert r.status_code == 200 and r.json()["persisted"] is False
    assert "MODEL_NAME=x" not in env_p.read_text(encoding="utf-8")  # not persisted
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_console.py -k "config_post" -v`
Expected: FAIL — 404 / missing `_rebuild_client`.

- [ ] **Step 3: Write minimal implementation**

In `server.py`:
1. Add import: `from pageindex_mutil.utils import configure_llm`（若已 import 则跳过）。
2. Add module-level capture (near the `client` global) + `_rebuild_client`:
```python
_startup_workspace = WORKSPACE
_startup_db_path = DB_PATH


def _rebuild_client(*, model=None, retrieve_model=None):
    """Rebuild the global PageIndexClient with new model overrides.

    Credentials live in the shared utils client (configure_llm); only model
    names need a client rebuild because closet_index/router are bound at __init__.
    """
    global client
    if client is None:
        return
    overrides = {}
    if model:
        overrides["model"] = model
    if retrieve_model:
        overrides["retrieve_model"] = retrieve_model
    if not overrides:
        return
    try:
        client.close()
    except Exception:
        pass
    client = PageIndexClient(
        workspace=_startup_workspace, db_path=_startup_db_path, **overrides
    )
    logger.info("Rebuilt PageIndexClient with overrides=%s", overrides)
```
3. In `lifespan`, after `client = PageIndexClient(...)`, record the resolved paths (handles the fallback branch) — replace the single assignment with:
```python
    global client, _startup_workspace, _startup_db_path
    _startup_workspace = workspace
    _startup_db_path = db_path
    client = PageIndexClient(workspace=workspace, db_path=db_path)
```
4. Add the POST endpoint:
```python
async def config_post_endpoint(request: Request) -> Response:
    try:
        body = await request.json() or {}
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    persist = body.get("persist", True)
    model = body.get("model")
    retrieve_model = body.get("retrieve_model")
    api_key = body.get("api_key")
    base_url = body.get("base_url")

    # 1) Runtime apply (immediate effect) + sync os.environ so the read-back
    #    snapshot (which reads os.getenv via ConfigLoader/get_llm_config) is
    #    consistent. configure_llm rebuilds the shared client but does NOT touch
    #    os.environ; _rebuild_client sets client.model but ConfigLoader().load()
    #    re-reads os.getenv — so without these env syncs the snapshot would be stale.
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    if api_key or base_url:
        configure_llm(api_key=api_key, base_url=base_url)
    if model:
        os.environ["MODEL_NAME"] = model
    if retrieve_model:
        os.environ["RETRIEVE_MODEL_NAME"] = retrieve_model
    if model or retrieve_model:
        _rebuild_client(model=model, retrieve_model=retrieve_model)

    # 2) Persist to disk (per-file guarded backup -> targeted write -> verify)
    persisted = False
    wrote_any = model or retrieve_model or api_key or base_url
    if persist and wrote_any:
        try:
            if model or retrieve_model:
                # config.yaml: back up + line-replace only when present
                if web_config.DEFAULT_CONFIG_YAML.exists():
                    web_config.backup_file(web_config.DEFAULT_CONFIG_YAML)
                web_config.set_config_yaml_model(
                    web_config.DEFAULT_CONFIG_YAML, model=model, retrieve_model=retrieve_model)
                env_fields = {}
                if model:
                    env_fields["MODEL_NAME"] = model
                if retrieve_model:
                    env_fields["RETRIEVE_MODEL_NAME"] = retrieve_model
                if web_config.DEFAULT_ENV.exists():
                    web_config.backup_file(web_config.DEFAULT_ENV)
                web_config.set_env_fields(web_config.DEFAULT_ENV, env_fields)
            cred_fields = {}
            if api_key:
                cred_fields["OPENAI_API_KEY"] = api_key
            if base_url:
                cred_fields["OPENAI_BASE_URL"] = base_url
            if cred_fields:
                if web_config.DEFAULT_ENV.exists():
                    web_config.backup_file(web_config.DEFAULT_ENV)
                web_config.set_env_fields(web_config.DEFAULT_ENV, cred_fields)
            persisted = True
        except Exception as e:
            logger.exception("config persist failed; backups remain at *.bak")
            return JSONResponse({"error": f"Persist failed: {e}", "applied": True,
                                 "persisted": False}, status_code=500)
    snap = web_config.read_config_snapshot(get_client())
    snap.update({"applied": True, "persisted": persisted})
    return JSONResponse(snap)
```
5. Update route — change the `/api/config` Route to support both methods:
```python
    Route("/api/config", endpoint=config_get_endpoint, methods=["GET"]),
    Route("/api/config", endpoint=config_post_endpoint, methods=["POST"]),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web_console.py -v`
Expected: PASS (13 tests).

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_web_console.py
git commit -m "feat(web): POST /api/config (hot-swap + persist + rollback) + _rebuild_client"
```

---

### Task 7: 静态托管（`GET /` + `/static`）

**Files:**
- Modify: `server.py`, Test: `tests/test_web_console.py`
- Create (stub, content in Task 8): `web/index.html`

**Interfaces:**
- Produces: `GET /` → `web/index.html`；`Mount("/static", StaticFiles(directory="web/static"))`.

- [ ] **Step 1: Write the failing test**

Append:
```python
def test_index_served_and_static_whitelisted(monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "")
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.get("/")
    assert r.status_code == 200
    assert "<div id=\"app\">" in r.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_console.py::test_index_served_and_static_whitelisted -v`
Expected: FAIL — 404 / file not found.

- [ ] **Step 3: Write minimal implementation**

1. Create a minimal `web/index.html` (full UI in Task 8):
```html
<!DOCTYPE html>
<html lang="zh">
<head><meta charset="utf-8"><title>PageIndex Console</title></head>
<body><div id="app">loading</div></body>
</html>
```
2. In `server.py` add a module-level web dir + an index route handler + static mount. Add near the other module vars:
```python
WEB_DIR = Path(__file__).resolve().parent / "web"
```
Routes list becomes:
```python
routes = [
    Route("/", endpoint=lambda request: FileResponse(str(WEB_DIR / "index.html")), methods=["GET"]),
    Route("/health", endpoint=health_endpoint, methods=["GET"]),
    Route("/upload", endpoint=upload_endpoint, methods=["POST", "OPTIONS"]),
    Route("/api/documents", endpoint=documents_endpoint, methods=["GET"]),
    Route("/api/search", endpoint=search_endpoint, methods=["POST"]),
    Route("/api/config", endpoint=config_get_endpoint, methods=["GET"]),
    Route("/api/config", endpoint=config_post_endpoint, methods=["POST"]),
    Route("/sse", endpoint=handle_sse, methods=["GET"]),
    Mount("/static", app=StaticFiles(directory=str(WEB_DIR / "static")), name="static"),
    Mount("/messages/", app=sse_transport.handle_post_message),
]
```
(`StaticFiles` requires `WEB_DIR/"static"` to exist — create `web/static/.gitkeep` now.)
> 注：Task 6 已注册 `GET /api/config` 与 `POST /api/config` 两条 Route；本任务只是把它们合并进完整 routes 列表展示，**勿重复添加**。

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web_console.py -v`
Expected: PASS (14 tests).

- [ ] **Step 5: Commit**

```bash
git add server.py web/ tests/test_web_console.py
git commit -m "feat(web): serve index.html at / + static mount"
```

---

### Task 8: 前端三页（文档 / 问答 / 模型配置）

**Files:**
- Create: `web/index.html`, `web/static/app.js`, `web/static/styles.css`

> 无单元测试；用 `browser-testing` skill + `frontend-design` skill 手测+截图为证。本任务先给出**结构+精确 API 接线**，视觉打磨交 frontend-design。

**Layout 合约**（实现须满足）：
- 顶栏：标题 "PageIndex 控制台" + API Key 输入（存 `localStorage.pageindex_api_key`）。
- `el-tabs` 三页：文档 / 问答 / 模型配置。
- 所有 fetch 经统一 `api(path, opts)` 封装，注入 `X-API-Key`。

- [ ] **Step 1: `web/static/app.js`**（完整骨架）

```javascript
const { createApp, ref, onMounted } = Vue;

const API_KEY_STORE = "pageindex_api_key";

function apiKey() {
  return localStorage.getItem(API_KEY_STORE) || "";
}

async function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
  const key = apiKey();
  if (key) headers["X-API-Key"] = key;
  const res = await fetch(path, { ...opts, headers });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try { msg = (await res.json()).error || msg; } catch (e) {}
    throw new Error(msg);
  }
  return res.json();
}

const app = createApp({
  setup() {
    const tab = ref("docs");
    const keyInput = ref(apiKey());
    // docs
    const docs = ref([]);
    const uploadRef = ref(null);
    const uploadResults = ref([]);
    // search
    const query = ref("");
    const answer = ref(null);
    const searching = ref(false);
    // config
    const config = ref(null);
    const cfgForm = ref({ model: "", retrieve_model: "", api_key: "", base_url: "", persist: true });
    const cfgMsg = ref("");

    const saveKey = () => { localStorage.setItem(API_KEY_STORE, keyInput.value); ElementPlus.ElMessage.success("Key saved"); };

    const loadDocs = async () => { try { docs.value = (await api("/api/documents")).documents; } catch (e) { ElementPlus.ElMessage.error(e.message); } };
    const onUpload = async (opt) => {
      const f = opt.file; const fd = new FormData(); fd.append("file", f);
      try {
        const key = apiKey(); const h = {}; if (key) h["X-API-Key"] = key;
        const r = await fetch("/upload", { method: "POST", headers: h, body: fd });
        const j = await r.json();
        uploadResults.value.push({ name: f.name, ok: (j.succeeded||0)>0, raw: j });
        if ((j.succeeded||0) > 0) loadDocs();
        opt.onSuccess();
      } catch (e) { ElementPlus.ElMessage.error(e.message); opt.onError(); }
    };
    const doSearch = async () => {
      if (!query.value.trim()) return; searching.value = true; answer.value = null;
      try { answer.value = await api("/api/search", { method: "POST", body: JSON.stringify({ query: query.value, top_k: 5 }) }); }
      catch (e) { ElementPlus.ElMessage.error(e.message); } finally { searching.value = false; }
    };
    const loadConfig = async () => { try { config.value = await api("/api/config"); const c = config.value; cfgForm.value.model = c.model||""; cfgForm.value.retrieve_model = (c.retrieve_model||""); } catch (e) { ElementPlus.ElMessage.error(e.message); } };
    const saveConfig = async () => {
      cfgMsg.value = "";
      const body = { persist: cfgForm.value.persist };
      if (cfgForm.value.model) body.model = cfgForm.value.model;
      if (cfgForm.value.retrieve_model) body.retrieve_model = cfgForm.value.retrieve_model;
      if (cfgForm.value.api_key) body.api_key = cfgForm.value.api_key;
      if (cfgForm.value.base_url) body.base_url = cfgForm.value.base_url;
      try { await api("/api/config", { method: "POST", body: JSON.stringify(body) }); cfgMsg.value = "已应用" + (body.persist ? "并持久化" : "（仅运行时）"); ElementPlus.ElMessage.success(cfgMsg.value); loadConfig(); }
      catch (e) { ElementPlus.ElMessage.error(e.message); }
    };

    onMounted(() => { loadDocs(); loadConfig(); });
    return { tab, keyInput, saveKey, docs, uploadRef, uploadResults, onUpload, query, answer, searching, doSearch, config, cfgForm, cfgMsg, saveConfig, loadConfig, loadDocs };
  },
});
app.use(ElementPlus);
app.mount("#app");
```

- [ ] **Step 2: `web/index.html`**（CDN + 模板，视觉由 frontend-design 打磨）

```html
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PageIndex 控制台</title>
  <link rel="stylesheet" href="https://unpkg.com/element-plus/dist/index.css">
  <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
  <div id="app">
    <div class="topbar">
      <span class="brand">PageIndex 控制台</span>
      <el-input v-model="keyInput" placeholder="API Key（与 .env API_KEY 一致）" style="width:260px" show-password></el-input>
      <el-button type="primary" @click="saveKey">保存 Key</el-button>
    </div>
    <el-tabs v-model="tab" class="content">
      <el-tab-pane label="文档" name="docs">
        <el-upload :http-request="onUpload" drag multiple :show-file-list="false">
          <div class="upload-hint">拖拽或点击上传 PDF / Markdown</div>
        </el-upload>
        <el-button size="small" @click="loadDocs">刷新</el-button>
        <el-table :data="docs" border>
          <el-table-column prop="id" label="#" width="60"></el-table-column>
          <el-table-column prop="doc_name" label="文档"></el-table-column>
          <el-table-column prop="doc_description" label="描述"></el-table-column>
        </el-table>
      </el-tab-pane>
      <el-tab-pane label="问答" name="qa">
        <el-input v-model="query" type="textarea" :rows="3" placeholder="输入问题…"></el-input>
        <el-button type="primary" :loading="searching" @click="doSearch">提问</el-button>
        <div v-if="answer" class="answer">
          <el-tag size="small">{{ answer.confidence }}</el-tag>
          <p>{{ answer.answer }}</p>
          <ul><li v-for="d in answer.matched_docs" :key="d.doc_id||d.pdf_name">{{ d.doc_name || d.doc_id }}</li></ul>
        </div>
      </el-tab-pane>
      <el-tab-pane label="模型配置" name="cfg">
        <div v-if="config">
          <p class="muted">当前：{{ config.model }} · {{ config.provider }} · {{ config.base_url }} · key {{ config.api_key_masked }}</p>
        </div>
        <el-form label-width="140px" class="cfg-form">
          <el-form-item label="模型名 (model)"><el-input v-model="cfgForm.model"></el-input></el-form-item>
          <el-form-item label="检索模型"><el-input v-model="cfgForm.retrieve_model"></el-input></el-form-item>
          <el-form-item label="API Key"><el-input v-model="cfgForm.api_key" show-password placeholder="留空不改"></el-input></el-form-item>
          <el-form-item label="Base URL"><el-input v-model="cfgForm.base_url"></el-input></el-form-item>
          <el-form-item label="持久化"><el-switch v-model="cfgForm.persist"></el-switch></el-form-item>
          <el-form-item><el-button type="primary" @click="saveConfig">应用</el-button> <span class="muted">{{ cfgMsg }}</span></el-form-item>
        </el-form>
      </el-tab-pane>
    </el-tabs>
  </div>
  <script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
  <script src="https://unpkg.com/element-plus"></script>
  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 3: `web/static/styles.css`**（实现期 frontend-design 细化）

```css
body { margin: 0; font-family: -apple-system, "Segoe UI", "PingFang SC", sans-serif; background: #f5f7fa; }
.topbar { display: flex; align-items: center; gap: 12px; padding: 12px 20px; background: #fff; border-bottom: 1px solid #ebeef5; box-shadow: 0 1px 4px rgba(0,0,0,.04); }
.brand { font-weight: 600; font-size: 18px; margin-right: auto; }
.content { padding: 20px; max-width: 960px; margin: 0 auto; }
.upload-hint { padding: 24px; color: #909399; }
.answer { margin-top: 16px; background: #fff; padding: 16px; border-radius: 8px; border: 1px solid #ebeef5; }
.cfg-form { max-width: 560px; }
.muted { color: #909399; font-size: 13px; }
```

- [ ] **Step 4: Manual verify via browser-testing skill**

Run (start server, open browser, exercise 3 tabs, capture screenshots/console):
- `python server.py` → open `http://localhost:3000/`
- 文档：上传一个测试 PDF/MD（用 `tests/pdfs/` 现有文件）→ 表格出现。
- 问答：提问 → 回答显示。
- 配置：改 model → 应用 → 刷新页面 → 确认 `当前` 行变化 + `pageindex_mutil/config.yaml` 与 `.env` 同值。
Evidence: 截图 + 控制台无 error + network 200。

- [ ] **Step 5: Commit**

```bash
git add web/index.html web/static/app.js web/static/styles.css
git commit -m "feat(web): single-page console (docs/qa/config) via Vue3+ElementPlus CDN"
```

---

### Task 9: README + 全量回归

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add console section to README**

Append a short "## Web 控制台" section: open `http://localhost:3000/`；API Key 同 `.env` 的 `API_KEY`；CDN 需外网；改模型/凭据会写 `pageindex_mutil/config.yaml` 与 `.env`（写前备份 `.bak`）。

- [ ] **Step 2: Full regression**

Run: `python -m pytest -q`
Expected: 全绿（含既有 config 测试 + 14 新 web 测试）。

Run: `python server.py`（冒烟，确认启动无错、`/health` 200、`/` 出页面）。

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(web): console usage section + regression green"
```

---

## Self-Review（已执行）

- **Spec coverage**：FR1→T3；FR2→T8(复用/upload)；FR3→T4；FR4→T5；FR5→T6(model dualwrite)；FR6→T6(credentials)；FR7→T7+T8；FR8→T3 放行。NFR1→T8 CDN；NFR2 掩码→T2/T5、备份→T1/T6、定点→T1；NFR3→T4 async；NFR4 TDD 每任务；NFR5→T9 回归。全覆盖。
- **Placeholder scan**：无 TBD/TODO；每步含具体代码或命令。
- **Type consistency**：`read_config_snapshot(client, config_yaml_path=, env_path=)` 签名 T2 定义、T5 调用一致；`_rebuild_client(model=, retrieve_model=)` T6 定义、测试 mock 一致；`set_config_yaml_model(path, *, model=, retrieve_model=)` T1 定义、T6 调用一致。
