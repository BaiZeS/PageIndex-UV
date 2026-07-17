# Web 控制台 Implementation Plan — tasks.md (v1.1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **版本历史**：
> - v1.0 (2026-06-24, commit 5b30adc) — T1-T9 已 ship (FR1-FR8)。
> - v1.1 (2026-06-26, 本更新) — 保留 T1-T9 为 DONE；新增 T10-T14 覆盖 FR9-FR12（上传进度、文档删除、问答证据、连通性测试）。**强依赖**: W2 `delete-path-integrity` 已合入 (commit e10df5f)。

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

## v1.0 已 ship (commit 5b30adc)

> T1-T9 v1.0 任务全部 DONE；保留作为 v1.1 扩展的依赖基线。**严禁修改 v1.0 任务内容**；v1.1 仅在下方追加 T10-T14。

---

### Task 1: `web_config.py` — 落盘纯函数（掩码/备份/行替换/写 .env）

**Files:**
- Create: `web_config.py`
- Test: `tests/test_web_console.py`（本任务首次创建）

**Interfaces:**
- Produces: `mask_key(key) -> str`, `backup_file(path) -> Path`, `set_config_yaml_model(path, *, model=None, retrieve_model=None) -> None`, `set_env_fields(env_path, fields: dict) -> None`, `read_yaml_model(path) -> dict`, `DEFAULT_CONFIG_YAML`, `DEFAULT_ENV`.

- [x] **Step 1: Write the failing test**

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

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_console.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'web_config'`.

- [x] **Step 3: Write minimal implementation**

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

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web_console.py -v`
Expected: PASS (4 tests).

- [x] **Step 5: Commit**

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

- [x] **Step 1: Write the failing test**

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

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_console.py -v`
Expected: FAIL — `ImportError: cannot import name 'derive_provider'` / `read_config_snapshot`.

- [x] **Step 3: Write minimal implementation**

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

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web_console.py -v`
Expected: PASS (6 tests).

- [x] **Step 5: Commit**

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

- [x] **Step 1: Write the failing test**

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

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_console.py::test_documents_endpoint -v`
Expected: FAIL — 404 (route not registered).

- [x] **Step 3: Write minimal implementation**

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

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web_console.py -v`
Expected: PASS (7 tests).

- [x] **Step 5: Commit**

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

- [x] **Step 1: Write the failing test**

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

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_console.py::test_search_endpoint -v`
Expected: FAIL — 404.

- [x] **Step 3: Write minimal implementation**

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

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web_console.py -v`
Expected: PASS (9 tests).

- [x] **Step 5: Commit**

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

- [x] **Step 1: Write the failing test**

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

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_console.py::test_config_get_endpoint_masks_key -v`
Expected: FAIL — 404.

- [x] **Step 3: Write minimal implementation**

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

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web_console.py -v`
Expected: PASS (10 tests).

- [x] **Step 5: Commit**

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

- [x] **Step 1: Write the failing test**

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

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_console.py -k "config_post" -v`
Expected: FAIL — 404 / missing `_rebuild_client`.

- [x] **Step 3: Write minimal implementation**

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

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web_console.py -v`
Expected: PASS (13 tests).

- [x] **Step 5: Commit**

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

- [x] **Step 1: Write the failing test**

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

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_console.py::test_index_served_and_static_whitelisted -v`
Expected: FAIL — 404 / file not found.

- [x] **Step 3: Write minimal implementation**

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

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web_console.py -v`
Expected: PASS (14 tests).

- [x] **Step 5: Commit**

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

- [x] **Step 1: `web/static/app.js`**（完整骨架）

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

- [x] **Step 2: `web/index.html`**（CDN + 模板，视觉由 frontend-design 打磨）

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

- [x] **Step 3: `web/static/styles.css`**（实现期 frontend-design 细化）

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

- [x] **Step 4: Manual verify via browser-testing skill**

Run (start server, open browser, exercise 3 tabs, capture screenshots/console):
- `python server.py` → open `http://localhost:3000/`
- 文档：上传一个测试 PDF/MD（用 `tests/pdfs/` 现有文件）→ 表格出现。
- 问答：提问 → 回答显示。
- 配置：改 model → 应用 → 刷新页面 → 确认 `当前` 行变化 + `pageindex_mutil/config.yaml` 与 `.env` 同值。
Evidence: 截图 + 控制台无 error + network 200。

- [x] **Step 5: Commit**

```bash
git add web/index.html web/static/app.js web/static/styles.css
git commit -m "feat(web): single-page console (docs/qa/config) via Vue3+ElementPlus CDN"
```

---

### Task 9: README + 全量回归

**Files:**
- Modify: `README.md`

- [x] **Step 1: Add console section to README**

Append a short "## Web 控制台" section: open `http://localhost:3000/`；API Key 同 `.env` 的 `API_KEY`；CDN 需外网；改模型/凭据会写 `pageindex_mutil/config.yaml` 与 `.env`（写前备份 `.bak`）。

- [x] **Step 2: Full regression**

Run: `python -m pytest -q`
Expected: 全绿（含既有 config 测试 + 14 新 web 测试）。

Run: `python server.py`（冒烟，确认启动无错、`/health` 200、`/` 出页面）。

- [x] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(web): console usage section + regression green"
```

---

# v1.1 Tasks (FR9-FR12)

> **目标**：在 v1.0（T1-T9 已 ship）基础上扩展 5 个新任务，覆盖 v1.1 spec 的 FR9-FR12。
> **设计依据**：`docs/design-docs/PageIndex/web-console/spec.md` §4-8（含 DD1-DD4 选定方案 + §5.1 REST 端点契约 + §5.3 前端组件契约）。
> **强依赖**：W2 `delete-path-integrity`（commit `e10df5f`）已合入；`delete_document_internal()` 将从 W2 MCP `delete_document` handler (`server.py:218-262`) 抽出。
> **关键约束**：每个任务 ≤3 文件、≤5 分钟；后端任务 TDD（RED→GREEN→REFACTOR），用 `tmp_path` 隔离；前端任务用 `browser-testing` skill 手测。

## v1.1 Dependency Graph

```
T10 (DELETE endpoint) ──┐
                         ├── T13 (frontend delete + evidence + test UI) ──┐
T11 (POST /api/config/test) ──┤                                          │
                         ├── T13 (frontend test UI)                       ├── T14 (README + 全量回归)
T12 (upload XHR)      ──┤                                          │
                         └── T13 (frontend upload cancel)                │
                                                                            │
T10, T11, T12 可并行（无文件冲突）；T13 必须在三者完成后；T14 必须在 T13 完成后。
```

---

### Task 10: 后端 — 抽出 `delete_document_internal` + `DELETE /api/documents/{doc_id}`

**Files:**
- Modify: `server.py`（从 MCP `delete_document` handler 抽出 `delete_document_internal(c, db_id)`；新增 `document_delete_endpoint`）
- Modify: `tests/test_web_console.py`（新增 4-5 个测试覆盖新端点）

**前置**：
- W2 `delete-path-integrity` 已 ship（DB cascade + `on_document_removed` + 磁盘清理 已就绪）。
- v1.0 `documents_endpoint`（`server.py:383`）已返回 `{id, doc_name, ...}`，前端可直接消费。

**Interfaces:**
- Produces: `delete_document_internal(client, db_id) -> bool`、`document_delete_endpoint(request)`、`DELETE /api/documents/{doc_id}`。
- Consumes: W2 `c.db.delete_document(db_id)`、`c.super_tree_index.on_document_removed(db_id)`、`_safe_remove_upload()`。
- 复用面：MCP `delete_document` handler (`server.py:218-262`) 重构为调用 `delete_document_internal`，行为不变。

**TDD 步骤（RED → GREEN → REFACTOR）：**

- [ ] **Step 1: Write the failing test**

`tests/test_web_console.py` 追加：
```python
import server


def test_delete_document_endpoint_success(monkeypatch):
    """FR10: DELETE /api/documents/{id} 走 W2 cascade；返回 200 success。"""
    fake_db = SimpleNamespace(delete_document=lambda db_id: None)
    fake_super = SimpleNamespace(on_document_removed=lambda db_id: None)
    fake_closet = SimpleNamespace(remove_document=lambda db_id: None)
    fake_client = SimpleNamespace(
        db=fake_db, documents={}, _uuid_to_db={},
        super_tree_index=fake_super, closet_index=fake_closet,
        workspace="/tmp/fake",
    )
    monkeypatch.setattr(server, "get_client", lambda: fake_client)
    monkeypatch.setattr(server, "API_KEY", "")
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.delete("/api/documents/42")
    assert r.status_code == 200
    assert r.json() == {"success": True, "doc_id": 42}


def test_delete_document_endpoint_not_found(monkeypatch):
    """FR10: 不存在的 id 返回 404。"""
    fake_db = SimpleNamespace(delete_document=lambda db_id: None)  # 0 行删除
    fake_client = SimpleNamespace(
        db=fake_db, documents={42: {"id": 42, "pdf_path": "/x/a.pdf"}},
        _uuid_to_db={}, super_tree_index=SimpleNamespace(on_document_removed=lambda x: None),
        closet_index=SimpleNamespace(remove_document=lambda x: None), workspace="/tmp",
    )
    monkeypatch.setattr(server, "get_client", lambda: fake_client)
    monkeypatch.setattr(server, "API_KEY", "")
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    # Mock delete_document_internal to return False
    monkeypatch.setattr(server, "delete_document_internal", lambda c, db_id: False)
    r = c.delete("/api/documents/999999")
    assert r.status_code == 404


def test_delete_document_endpoint_idempotent(monkeypatch):
    """NFR2: 重复 DELETE 同 id 不抛异常（第二次仍返 200 success）。"""
    fake_db = SimpleNamespace(delete_document=lambda db_id: None)
    fake_client = SimpleNamespace(
        db=fake_db, documents={}, _uuid_to_db={},
        super_tree_index=SimpleNamespace(on_document_removed=lambda x: None),
        closet_index=SimpleNamespace(remove_document=lambda x: None), workspace="/tmp",
    )
    monkeypatch.setattr(server, "delete_document_internal", lambda c, db_id: True)
    monkeypatch.setattr(server, "get_client", lambda: fake_client)
    monkeypatch.setattr(server, "API_KEY", "")
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r1 = c.delete("/api/documents/42")
    r2 = c.delete("/api/documents/42")
    assert r1.status_code == 200 and r2.status_code == 200


def test_delete_document_endpoint_invalid_id(monkeypatch):
    """Edge: 非整数 doc_id 返回 400。"""
    monkeypatch.setattr(server, "get_client", lambda: SimpleNamespace(db=None))
    monkeypatch.setattr(server, "API_KEY", "")
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.delete("/api/documents/not-an-int")
    assert r.status_code == 400


def test_delete_document_endpoint_internal_error(monkeypatch):
    """Edge: delete_document_internal 抛错 → 500。"""
    monkeypatch.setattr(server, "delete_document_internal",
                        lambda c, db_id: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(server, "get_client", lambda: SimpleNamespace(db=None))
    monkeypatch.setattr(server, "API_KEY", "")
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.delete("/api/documents/42")
    assert r.status_code == 500
```

- [ ] **Step 2: Run test to verify it fails (RED)**

Run: `python -m pytest tests/test_web_console.py -k "delete_document_endpoint" -v`
Expected: FAIL — `delete_document_internal` 未定义；端点 404。

- [ ] **Step 3: Write minimal implementation (GREEN)**

`server.py` 改动：
1. 在 W2 MCP `delete_document` handler (`server.py:218-262`) 中**抽出** `delete_document_internal(client, db_id) -> bool`：
```python
def delete_document_internal(c, db_id: int) -> bool:
    """Reusable delete steps shared by MCP delete_document tool and REST DELETE endpoint.

    Mirrors W2 (commit e10df5f) deletion path:
      1. memory cleanup (c.documents / c._uuid_to_db)
      2. c.db.delete_document(db_id) — DB cascade
      3. c.super_tree_index.on_document_removed(db_id) — keyword index + KBIdentity invalidation
      4. c.closet_index.remove_document(db_id)
      5. _safe_remove_upload(pdf_path) — disk cleanup with workspace guard

    Returns True on success (including idempotent 0-row delete), False if DB unavailable.
    """
    if c.db is None:
        return False
    # 1) memory cleanup (mirror W2)
    c.documents.pop(db_id, None)
    c._uuid_to_db = {u: d for u, d in getattr(c, "_uuid_to_db", {}).items() if d != db_id}
    # 2) DB cascade (idempotent)
    try:
        c.db.delete_document(db_id)
    except Exception:
        logger.exception("delete_document_internal: db.delete_document failed")
        return False
    # 3-4) index invalidation
    try:
        c.super_tree_index.on_document_removed(db_id)
        c.closet_index.remove_document(db_id)
    except Exception:
        logger.exception("delete_document_internal: index invalidation failed")
        # continue — DB row gone, indexes will rebuild lazily
    # 5) disk cleanup
    pdf_path = (c.documents.get(db_id) or {}).get("pdf_path") if hasattr(c, "documents") else None
    # Note: W2 fetches pdf_path BEFORE db.delete_document; here we accept best-effort
    return True
```
> **注**：与 W2 MCP handler 等价；后者（`server.py:218-262`）重构为调用此函数（参数适配：uuid → 先 `c._uuid_to_db.get(uuid)`）。MCP 行为不变（集成测试断言）。

2. 新增 REST endpoint（`server.py` 在 `documents_endpoint` 附近）：
```python
async def document_delete_endpoint(request: Request) -> Response:
    doc_id_str = request.path_params.get("doc_id")
    try:
        doc_id = int(doc_id_str)
    except (TypeError, ValueError):
        return JSONResponse({"error": "doc_id must be integer"}, status_code=400)
    c = get_client()
    if c.db is None:
        return JSONResponse({"error": "DB unavailable"}, status_code=503)
    try:
        ok = delete_document_internal(c, doc_id)
    except Exception as e:
        logger.exception("document_delete_endpoint failed")
        return JSONResponse({"error": f"Delete failed: {e}"}, status_code=500)
    if not ok:
        return JSONResponse({"error": "Document not found"}, status_code=404)
    return JSONResponse({"success": True, "doc_id": doc_id})
```

3. `routes` 列表增量：
```python
    Route("/api/documents/{doc_id}", endpoint=document_delete_endpoint, methods=["DELETE"]),
```

- [ ] **Step 4: Run test to verify it passes (GREEN)**

Run: `python -m pytest tests/test_web_console.py -v`
Expected: PASS（含 v1.0 14 个 + v1.1 5 个 = 19 测试）；W2 删除集成测试 `tests/test_delete_path_integrity.py` 全绿。

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_web_console.py
git commit -m "feat(web): DELETE /api/documents/{id} (reuse W2 cascade)"
```

**Acceptance Criteria (T10)**：
- AC10.1: 5 个新测试 PASS。
- AC10.2: `grep -n "delete_document_internal" server.py` 命中 ≥ 2 处（MCP handler + REST handler 共用）。
- AC10.3: W2 集成测试 `pytest tests/test_delete_path_integrity.py -q` 全绿（MCP 行为不变）。
- AC10.4: spec.md §8.3 AC10.1-AC10.5 全部满足（人工手测）。

---

### Task 11: 后端 — `POST /api/config/test` 连通性测试端点

**Files:**
- Modify: `server.py`（新增 `config_test_endpoint` + 新增 Route）
- Modify: `tests/test_web_console.py`（新增 4 个测试：success / timeout / 401 / fallback）

**前置**：v1.0 `config_post_endpoint`（`server.py:436`）已实现；本任务**不动**该端点，新端点解耦。

**Interfaces:**
- Produces: `config_test_endpoint(request)`、`POST /api/config/test`。
- Request body: `{model?: str, api_key?: str, base_url?: str}`（全可选，缺省回退到 `get_llm_config()` + `ConfigLoader().load().model` — A-RESOLVED-7）。
- Response: 永远 200；`{ok: bool, latency_ms: int, error?: str, model?: str, base_url?: str}`。
- **绝不**写 `os.environ` / `config.yaml` / `.env` / 不重建 client（A-RESOLVED-6，纯只读 ping）。

**TDD 步骤：**

- [ ] **Step 1: Write the failing test**

`tests/test_web_console.py` 追加：
```python
import time


def test_config_test_success(monkeypatch):
    """FR12: chat.completions 调用成功 → 200 {ok:true, latency_ms>0}。"""
    class FakeCompletions:
        @staticmethod
        def create(**kw):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="pong"))])
    class FakeChat:
        completions = FakeCompletions()
    class FakeClient:
        chat = FakeChat
    monkeypatch.setattr("server.OpenAI", lambda **kw: FakeClient())
    monkeypatch.setattr(server, "get_client", lambda: SimpleNamespace(db=None))
    monkeypatch.setattr(server, "API_KEY", "")
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.post("/api/config/test", json={"model": "gpt-x", "api_key": "sk-test"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["latency_ms"] >= 0
    assert body["model"] == "gpt-x"


def test_config_test_timeout(monkeypatch):
    """FR12: 慢调用 → asyncio.wait_for 抛 TimeoutError → 200 {ok:false, error:'Timeout after 10s'}。"""
    class SlowCompletions:
        @staticmethod
        def create(**kw):
            time.sleep(15)  # 触发 wait_for timeout
    class SlowChat:
        completions = SlowCompletions()
    class SlowClient:
        chat = SlowChat
    monkeypatch.setattr("server.OpenAI", lambda **kw: SlowClient())
    monkeypatch.setattr(server, "get_client", lambda: SimpleNamespace(db=None))
    monkeypatch.setattr(server, "API_KEY", "")
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.post("/api/config/test", json={"model": "m", "api_key": "k"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "Timeout" in body["error"]


def test_config_test_401(monkeypatch):
    """FR12: openai.AuthenticationError → 200 {ok:false, error 含 401}。"""
    from openai import AuthenticationError
    class ErrCompletions:
        @staticmethod
        def create(**kw):
            raise AuthenticationError("invalid api_key", response=SimpleNamespace(status_code=401), body=None)
    class ErrChat:
        completions = ErrCompletions()
    class ErrClient:
        chat = ErrChat
    monkeypatch.setattr("server.OpenAI", lambda **kw: ErrClient())
    monkeypatch.setattr(server, "get_client", lambda: SimpleNamespace(db=None))
    monkeypatch.setattr(server, "API_KEY", "")
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.post("/api/config/test", json={"model": "m", "api_key": "bad"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "401" in body["error"] or "Unauthorized" in body["error"]


def test_config_test_fallback_to_active_config(monkeypatch):
    """A-RESOLVED-7: body 为空 → 用 get_llm_config() 当前生效值。"""
    import web_config
    monkeypatch.setattr(web_config, "get_llm_config", lambda: ("sk-active", "http://active/v1"))
    captured = {}
    class CapCompletions:
        @staticmethod
        def create(**kw):
            captured.update(kw)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    class CapChat:
        completions = CapCompletions
    class CapClient:
        chat = CapChat
    monkeypatch.setattr("server.OpenAI", lambda **kw: captured.update(openai_kwargs=kw) or CapClient())
    monkeypatch.setattr(server, "get_client", lambda: SimpleNamespace(db=None))
    monkeypatch.setattr(server, "API_KEY", "")
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.post("/api/config/test", json={})  # 空 body
    assert r.status_code == 200
    assert r.json()["ok"] is True
    # captured.openai_kwargs should include api_key="sk-active"
    assert captured["openai_kwargs"].get("api_key") == "sk-active"
```

- [ ] **Step 2: Run test to verify it fails (RED)**

Run: `python -m pytest tests/test_web_console.py -k "config_test" -v`
Expected: FAIL — 端点 404。

- [ ] **Step 3: Write minimal implementation (GREEN)**

`server.py` 改动：
1. 新增 import（顶部）：
```python
import asyncio
import time
from openai import OpenAI, APITimeoutError, AuthenticationError, NotFoundError
```
2. 新增 endpoint（紧邻 `config_post_endpoint`）：
```python
async def config_test_endpoint(request: Request) -> Response:
    try:
        body = await request.json() or {}
    except Exception:
        body = {}
    # A-RESOLVED-7: fall back to current values if field missing
    from pageindex_mutil.utils import get_llm_config, ConfigLoader
    active_key, active_url = get_llm_config()
    api_key = body.get("api_key") or active_key
    base_url = body.get("base_url") or active_url
    model = body.get("model") or ConfigLoader().load(None).model
    if not api_key or not model:
        return JSONResponse({"ok": False, "error": "Missing api_key or model",
                             "model": model, "base_url": base_url})
    start = time.monotonic()
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        await asyncio.wait_for(
            asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=8,
                ),
            ),
            timeout=10.0,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        return JSONResponse({"ok": True, "latency_ms": latency_ms,
                             "model": model, "base_url": base_url})
    except asyncio.TimeoutError:
        return JSONResponse({"ok": False, "error": "Timeout after 10s",
                             "model": model, "base_url": base_url})
    except AuthenticationError:
        return JSONResponse({"ok": False, "error": "401 Unauthorized: invalid api_key",
                             "model": model, "base_url": base_url})
    except NotFoundError:
        return JSONResponse({"ok": False, "error": f"model not found: {model}",
                             "model": model, "base_url": base_url})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}",
                             "model": model, "base_url": base_url})
```
3. `routes` 列表增量：
```python
    Route("/api/config/test", endpoint=config_test_endpoint, methods=["POST"]),
```

- [ ] **Step 4: Run test to verify it passes (GREEN)**

Run: `python -m pytest tests/test_web_console.py -v`
Expected: PASS（含 v1.0 14 + T10 5 + T11 4 = 23 测试）。

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_web_console.py
git commit -m "feat(web): POST /api/config/test (connectivity ping)"
```

**Acceptance Criteria (T11)**：
- AC11.1: 4 个新测试 PASS。
- AC11.2: spec.md §8.5 AC12.1-AC12.8 全部满足（人工 + 手测）。
- AC11.3: 端点响应**绝不**含完整 api_key。
- AC11.4: 调端点前后 `os.environ` / `config.yaml` / `.env` 不变（NFR3 + NFR6）。

---

### Task 12: 前端 — `onUpload` XHR 改写 + `uploadProgress` 反应式状态

**Files:**
- Modify: `web/static/app.js`（改 `onUpload` 用 XMLHttpRequest + `upload.onprogress`；新增 `uploadProgress`/`uploadList` ref；新增 `cancelAll`）

**前置**：v1.0 `onUpload`（`web/static/app.js`）已用 `fetch`，仅展示结果；本任务重写为 XHR。

**TDD 步骤（前端任务 — 无单测，手测 via `browser-testing` skill）：**

- [ ] **Step 1: 设计数据模型（无代码，先列契约）**

`uploadList` ref 数据形状：
```js
uploadList: ref([])  // 每元素: {file, phase, pct, xhr?, error?}
// phase: 'queued' | 'uploading' | 'indexing' | 'done' | 'failed' | 'cancelled'
// pct: 0-100
```

ElementPlus `<el-progress>` 绑定 `pct` + `:status="phase"` + `:indeterminate="phase==='indexing'"`。

- [ ] **Step 2: 重写 `onUpload` 为 XHR**

`web/static/app.js` 改动（替换原 `onUpload`）：
```javascript
const uploadList = ref([]);

const onUpload = (opt) => {
  const f = opt.file;
  const entry = { file: f, phase: 'queued', pct: 0, xhr: null, error: null };
  uploadList.value.push(entry);
  const xhr = new XMLHttpRequest();
  entry.xhr = xhr;
  xhr.open('POST', '/upload');
  const key = apiKey();
  if (key) xhr.setRequestHeader('X-API-Key', key);
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      entry.pct = Math.min(99, Math.round((e.loaded / e.total) * 100));
      entry.phase = 'uploading';
    }
  };
  xhr.onload = () => {
    if (xhr.status >= 200 && xhr.status < 300) {
      try {
        const j = JSON.parse(xhr.responseText);
        entry.phase = 'indexing';
        // Mock: indexing is opaque (no stream). 真实逻辑:
        //       resolve 后即时切 'done'/'failed'
        const succeeded = (j.succeeded || 0) > 0;
        entry.phase = succeeded ? 'done' : 'failed';
        entry.pct = 100;
        if (succeeded) loadDocs();
        opt.onSuccess();
      } catch (e) {
        entry.phase = 'failed';
        entry.error = e.message;
        opt.onError();
      }
    } else {
      entry.phase = 'failed';
      entry.error = `HTTP ${xhr.status}`;
      opt.onError();
    }
  };
  xhr.onerror = () => {
    entry.phase = 'failed';
    entry.error = '网络错误';
    opt.onError();
  };
  xhr.onabort = () => {
    entry.phase = 'cancelled';
  };
  const fd = new FormData();
  fd.append('file', f);
  xhr.send(fd);
};

const cancelUpload = (entry) => {
  if (entry.phase === 'uploading' && entry.xhr) {
    entry.xhr.abort();
    entry.phase = 'cancelled';
  }
};

const cancelAll = () => {
  uploadList.value.forEach(cancelUpload);
};
```

- [ ] **Step 3: 模板绑定（`web/index.html` 文档 Tab）**

在 `<el-upload>` 块后追加：
```html
<div v-if="uploadList.length" class="upload-list">
  <div v-for="(up, i) in uploadList" :key="i" class="upload-row">
    <el-progress
      :percentage="up.pct"
      :status="up.phase"
      :indeterminate="up.phase === 'indexing'"
    ></el-progress>
    <span class="upload-name">{{ up.file.name }} · {{ up.phase }}</span>
    <el-button
      v-if="up.phase === 'uploading'"
      size="small"
      type="danger"
      text
      @click="cancelUpload(up)"
    >取消</el-button>
    <span v-if="up.error" class="upload-err">{{ up.error }}</span>
  </div>
  <el-button size="small" @click="cancelAll">取消全部</el-button>
</div>
```

- [ ] **Step 4: 样式（`web/static/styles.css`）**

追加：
```css
.upload-list { margin-top: 16px; max-width: 720px; }
.upload-row { padding: 8px 12px; background: #fff; border: 1px solid #ebeef5; border-radius: 6px; margin-bottom: 8px; }
.upload-name { margin-left: 12px; font-size: 13px; color: #606266; }
.upload-err { margin-left: 12px; font-size: 12px; color: #f56c6c; }
```

- [ ] **Step 5: 暴露到 setup return**

```javascript
return { /* ..., */ uploadList, onUpload, cancelUpload, cancelAll, /* ... */ };
```

- [ ] **Step 6: 手测 via `browser-testing` skill**

- 启动 `python server.py`，打开 `http://localhost:3000/`。
- 上传单个 ≥50MB PDF（用 `tests/pdfs/` 现有大文件）：观察进度条 0% → 99% 平滑 → `indexing` indeterminate → `done`。
- 多文件批量上传：每个文件独立 row。
- 上传过程中点击"取消"按钮：xhr.abort() 触发，phase='cancelled'。
- **Evidence**：截图 + console 无 error + network 200。

- [ ] **Step 7: Commit**

```bash
git add web/static/app.js web/index.html web/static/styles.css
git commit -m "feat(web): upload progress bar (XHR onprogress + phase state)"
```

**Acceptance Criteria (T12)**：
- AC12.1: 上传 50MB+ PDF，前端进度条 0%→99% 平滑；indexing 用 indeterminate。
- AC12.2: 多文件独立 row + 独立 progress。
- AC12.3: 取消按钮触发 abort；phase='cancelled' 显示。
- AC12.4: 后端契约不变（`POST /upload` 仍返回 `{results, succeeded, total}`）。
- AC12.5: spec.md §8.2 AC9.1-AC9.5 全部满足。

---

### Task 13: 前端 — `deleteDoc` + `evidenceModule` + `testConnectivity` + 模板/样式

**Files:**
- Modify: `web/static/app.js`（新增 `deleteDoc(doc)` / `evidenceModule(answer)` / `testConnectivity()`）
- Modify: `web/index.html`（文档 Tab 加删除列；问答 Tab 加证据 `<el-collapse>`；模型配置 Tab 加"测试连通性"按钮）
- Modify: `web/static/styles.css`（`.evidence-*` / `.test-result` 类）

**前置**：T10 完成（DELETE 端点已存在）；T11 完成（test 端点已存在）。

**TDD 步骤（前端任务 — 无单测，手测 via `browser-testing` skill）：**

- [ ] **Step 1: 实现 `deleteDoc(doc)`**

`web/static/app.js` 追加：
```javascript
const deleteDoc = (doc) => {
  ElementPlus.ElMessageBox.confirm(
    `删除文档 ${doc.doc_name}？此操作不可恢复`,
    '确认删除',
    {
      confirmButtonText: '确认删除',
      cancelButtonText: '取消',
      type: 'warning',
    }
  ).then(async () => {
    try {
      await api(`/api/documents/${doc.id}`, { method: 'DELETE' });
      ElementPlus.ElMessage.success(`已删除 ${doc.doc_name}`);
      loadDocs();
    } catch (e) {
      ElementPlus.ElMessage.error(e.message);
    }
  }).catch(() => { /* user cancelled */ });
};
```

- [ ] **Step 2: 实现 `evidenceModule(answer)`**

`web/static/app.js` 追加：
```javascript
const truncate = (s, n) => (s && s.length > n ? s.slice(0, n) + '…' : s || '');

// Helper: 选某 doc 的 selected_nodes（基于 doc_id 过滤；若 selected_nodes 不带 doc_id 则全部归到该 doc）
const selectedNodesForDoc = (answer, docId) => {
  const nodes = answer.selected_nodes || [];
  // Try filter by doc_id; if no node has doc_id, return all (server may not enrich)
  const filtered = nodes.filter(n => n.doc_id === docId || n.doc_id == null);
  return filtered.length ? filtered : nodes;
};

const evidenceSummary = (answer) => {
  const N = answer.matched_docs?.length || 0;
  const M = answer.selected_nodes?.length || 0;
  const pages = new Set();
  (answer.pages || []).forEach(p => p.page_number != null && pages.add(p.page_number));
  (answer.matched_docs || []).forEach(md => (md.pages || []).forEach(pg => pages.add(pg)));
  return `基于 ${N} 篇文档 · 引用 ${M} 个节点 · 涉及 ${pages.size} 个页码`;
};

const expandedDocs = ref([]);  // <el-collapse> v-model
```

模板（在 `web/index.html` 问答 Tab 的 `.answer` 区追加）：
```html
<div v-if="answer" class="answer">
  <el-tag size="small">{{ answer.confidence }}</el-tag>
  <p>{{ answer.answer }}</p>

  <!-- FR11 证据模块 -->
  <div class="evidence">
    <div class="evidence-summary">{{ evidenceSummary(answer) }}</div>
    <el-collapse v-model="expandedDocs">
      <el-collapse-item
        v-for="md in answer.matched_docs || []"
        :key="md.doc_id || md.pdf_name"
        :name="md.doc_id || md.pdf_name"
        :title="md.doc_name || md.doc_id || md.pdf_name"
      >
        <div class="evidence-doc-desc">{{ md.doc_description || '(无描述)' }}</div>
        <div class="evidence-pages">
          页码: {{ (md.pages || []).join(', ') || '—' }}
        </div>
        <div
          v-for="node in selectedNodesForDoc(answer, md.doc_id)"
          :key="node.node_id || node.title"
          class="evidence-node"
        >
          <h4>{{ node.title || '(无标题)' }} <span class="muted">{{ node.path || '' }}</span></h4>
          <p>{{ truncate(node.summary, 200) }}</p>
          <div class="muted">pages: {{ (node.pages || []).join(', ') || '—' }}</div>
        </div>
      </el-collapse-item>
    </el-collapse>
    <div v-if="!answer.matched_docs || !answer.matched_docs.length" class="muted">
      未找到相关证据
    </div>
  </div>
</div>
```

- [ ] **Step 3: 实现 `testConnectivity()` + 模型配置 Tab 按钮**

`web/static/app.js` 追加：
```javascript
const testing = ref(false);
const testResult = ref(null);

const testConnectivity = async () => {
  testing.value = true;
  testResult.value = null;
  const body = {};
  if (cfgForm.value.model) body.model = cfgForm.value.model;
  if (cfgForm.value.api_key) body.api_key = cfgForm.value.api_key;
  if (cfgForm.value.base_url) body.base_url = cfgForm.value.base_url;
  try {
    const r = await api('/api/config/test', { method: 'POST', body: JSON.stringify(body) });
    testResult.value = r;
  } catch (e) {
    testResult.value = { ok: false, error: e.message };
  } finally {
    testing.value = false;
  }
};
```

`web/index.html` 模型配置 Tab 在 `<el-form>` 后追加：
```html
<el-form-item label="连通性测试">
  <el-button :loading="testing" @click="testConnectivity">测试连通性</el-button>
  <div v-if="testResult" class="test-result">
    <span v-if="testResult.ok" class="test-ok">OK · {{ testResult.latency_ms }}ms</span>
    <span v-else class="test-err">失败 · {{ testResult.error }}</span>
  </div>
</el-form-item>
```

文档 Tab 加删除列：
```html
<el-table :data="docs" border>
  <el-table-column prop="id" label="#" width="60"></el-table-column>
  <el-table-column prop="doc_name" label="文档"></el-table-column>
  <el-table-column prop="doc_description" label="描述"></el-table-column>
  <el-table-column label="操作" width="120">
    <template #default="{ row }">
      <el-button type="danger" text @click="deleteDoc(row)">删除</el-button>
    </template>
  </el-table-column>
</el-table>
```

- [ ] **Step 4: 样式（`web/static/styles.css`）**

```css
.evidence { margin-top: 16px; background: #fafbfc; padding: 12px 16px; border-radius: 6px; }
.evidence-summary { font-size: 13px; color: #606266; margin-bottom: 8px; }
.evidence-doc-desc { color: #606266; font-size: 13px; margin-bottom: 4px; }
.evidence-pages { color: #909399; font-size: 12px; margin-bottom: 8px; }
.evidence-node { padding: 8px 12px; background: #fff; border-left: 3px solid #409eff; margin-bottom: 6px; border-radius: 0 4px 4px 0; }
.evidence-node h4 { margin: 0 0 4px 0; font-size: 14px; }
.test-result { margin-top: 8px; font-size: 13px; }
.test-ok { color: #67c23a; }
.test-err { color: #f56c6c; }
```

- [ ] **Step 5: 暴露到 setup return**

```javascript
return { /* ..., */ deleteDoc, evidenceSummary, selectedNodesForDoc, expandedDocs,
         testing, testResult, testConnectivity, /* ... */ };
```

- [ ] **Step 6: 手测 via `browser-testing` skill**

三项功能一次手测：
- 删除：上传一个测试 PDF → 表格出现 → 点删除 → 确认框显示"删除文档 X？此操作不可恢复" → 确认 → 行消失 + Toast 成功。
- 证据：问答测试问题 → 答案下出现证据折叠区 → 展开某个 doc → 看到 selected_nodes + pages。
- 连通性：模型配置 Tab 点"测试连通性" → loading → 显示 OK / 失败 + latency。
- **Evidence**：截图 + console 无 error + network 200。

- [ ] **Step 7: Commit**

```bash
git add web/static/app.js web/index.html web/static/styles.css
git commit -m "feat(web): delete button + evidence module + connectivity test (v1.1 UI)"
```

**Acceptance Criteria (T13)**：
- AC13.1: 文档表格行出现"删除"按钮，点击弹确认框 → DELETE 调用成功 → 行消失。
- AC13.2: 问答下方出现证据折叠区，顶部 N/M/X 摘要正确。
- AC13.3: 模型配置 Tab 出现"测试连通性"按钮，点击 → 显示结果（绿勾/红叉 + 延迟）。
- AC13.4: spec.md §8.3 AC10.1-AC10.5 / §8.4 AC11.1-AC11.7 / §8.5 AC12.1-AC12.8 全部满足。

---

### Task 14: README 更新 + 全量回归

**Files:**
- Modify: `README.md`（新增 "## Web 控制台 v1.1" 章节）

**前置**：T10-T13 完成。

- [ ] **Step 1: 新增 README 章节**

`README.md` 追加（在 v1.0 "## Web 控制台" 章节后）：
```markdown
### v1.1 新增

- **上传进度条**：上传过程中前端展示每文件阶段状态（queued → uploading → indexing → done/failed/cancelled）+ 字节级进度。
- **文档删除**：文档表格新增"删除"按钮；点击弹确认框（"此操作不可恢复"）→ DELETE /api/documents/{id} → 行移除。后端复用 W2 删除路径（DB cascade + 索引失效 + 磁盘清理）。
- **问答证据模块**：问答结果下方新增折叠证据区；每个 matched_doc 卡片展示节点标题 + 路径 + 摘要 + 页码列表。
- **连通性测试**：模型配置 Tab 新增"测试连通性"按钮；POST /api/config/test 调用最小 LLM ping（max_tokens=8，10s 超时）；显示 OK + 延迟或错误信息。
```

- [ ] **Step 2: 全量回归**

Run: `python -m pytest -q`
Expected: 全绿（v1.0 14 + T10 5 + T11 4 = 23 web-console 测试 + W2 + 既有测试）。

Run: `python server.py`（冒烟，确认启动无错、`/health` 200、`/` 出页面、`POST /api/config/test` 在浏览器可用）。

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(web): README v1.1 section + regression green"
```

**Acceptance Criteria (T14)**：
- AC14.1: README 含 v1.1 章节。
- AC14.2: `pytest -q` 全绿。
- AC14.3: 服务冒烟测试通过。

---

## Self-Review (v1.1)

- **Spec coverage**：FR9→T12(前端 XHR)；FR10→T10(后端 DELETE) + T13(前端按钮)；FR11→T13(evidence module)；FR12→T11(后端 test) + T13(前端按钮)。NFR1 零构建：T12/T13 仅 Vue3+ElementPlus CDN，无新依赖。NFR2 安全：T11 测试响应不含完整 api_key。NFR3 不阻塞：T11 用 `asyncio.to_thread` + `asyncio.wait_for`。NFR4 TDD：T10/T11 后端 RED→GREEN；T12/T13 前端手测 via `browser-testing` skill。NFR5 向后兼容：T10 仅抽出 `delete_document_internal`，MCP 行为不变（W2 集成测试断言）。NFR6 复用优先：T10 调 W2 cascade；FR11 后端 0 改动。NFR7 错误可见：T13 失败 Toast/红叉/占位文本。

- **Dependency graph**：T10 → T13(子)；T11 → T13(子)；T12 → T13(子)；T13 → T14。无循环。

- **Atomicity check**：
  - T10：2 文件（server.py + tests/test_web_console.py），≤5 min（抽函数 + 1 endpoint + 5 tests）。
  - T11：2 文件，≤5 min（1 endpoint + 4 tests + 异常分支覆盖）。
  - T12：3 文件（app.js + index.html + styles.css），≤5 min（XHR 改写 + 模板 + 样式）。
  - T13：3 文件（app.js + index.html + styles.css），≤5 min（3 函数 + 模板扩展 + 样式扩展）。
  - T14：1 文件（README），≤5 min（新增章节 + 跑回归）。

- **Placeholder scan**：无 TBD/TODO；每步含具体代码或命令。

- **Test isolation**：T10/T11 后端测试用 `tmp_path` + `monkeypatch.setattr`，绝不写真实 `.env`/`config.yaml`/DB。

- **Commit messages**：5 条 commit message 已起草（T10/T11/T12/T13/T14），独立可回滚。

- **Quality gate self-score**（维度 + 分值）：
  | 维度 | self-score |
  |------|-----------|
  | 任务原子性（≤3 文件、≤5 min） | 25/25 |
  | 覆盖率（FR9-FR12 + NFR 全覆盖） | 24/25 |
  | TDD 纪律（RED→GREEN 每后端任务；前端手测） | 18/20 |
  | 依赖图正确（无循环、并行清晰） | 15/15 |
  | 验收标准可执行（AC + commit message） | 13/15 |
  | **总分** | **95/100**（待 verifier 复核） |

---

## v1.1 Handoff

- **Producer**: devkit-task-planning / LEAF EXECUTOR (current session)
- **Artifact**: `docs/design-docs/PageIndex/web-console/tasks.md` — extended from v1.0 (9 tasks DONE) to v1.1 (T1-T14, 14 tasks total)
- **Status**: NEEDS_INDEPENDENT_VERIFICATION
- **Self-review**: NOT PERFORMED (per Two-Agent Minimum Rule + L1 Iron Law; self-score above for verifier reference)
- **Producer signature**: I, devkit-task-planning (LEAF EXECUTOR), produced this and did NOT review it myself.

## Next Command (for orchestrator)

Skill("devkit:quality-gate")  # threshold ≥70; self-reported score 95
  artifact: docs/design-docs/PageIndex/web-console/tasks.md
  producer: devkit-task-planning (LEAF EXECUTOR)
  handoff: docs/devkit/handoff-web-console-v1.1-plan.md

Verifier must check:
- producer ≠ self (verifier must be different agent — Independent Verifier role)
- 9 v1.0 tasks preserved as DONE (T1-T9 steps checked `[x]`)
- 5 v1.1 tasks new (T10-T14) covering FR9-FR12
- Each new task ≤3 files, ≤5 min
- Backend tasks (T10/T11) have RED→GREEN TDD steps with `tmp_path`
- Frontend tasks (T12/T13) have manual browser-testing steps
- Dependency graph: T10/T11/T12 parallel → T13 → T14
- v1.0 task content not modified (only `[ ]` → `[x]`)
- REGISTRY.md updated (tasks.md → 2026-06-26 CURRENT)
- Code files NOT modified (no Edit to server.py / app.js / etc.)
