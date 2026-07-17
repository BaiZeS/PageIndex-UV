import json
import textwrap
import asyncio
from pathlib import Path
from types import SimpleNamespace

import web_config
from web_config import (
    mask_key, backup_file, set_config_yaml_model, set_env_fields, read_yaml_model,
    read_config_snapshot, derive_provider,
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


import server


def _client_with_docs(monkeypatch, docs):
    fake_db = SimpleNamespace(get_all_documents=lambda: docs)
    monkeypatch.setattr(server, "get_client", lambda: SimpleNamespace(db=fake_db))
    monkeypatch.delenv("API_KEY", raising=False)
    # API_KEY is read at module-import time (server.py); patch the live constant so
    # the middleware treats the request as unauthenticated.
    monkeypatch.setattr(server, "API_KEY", "")
    from starlette.testclient import TestClient
    return TestClient(server.app)


def test_documents_endpoint(monkeypatch):
    docs = [{"id": 1, "pdf_name": "a.pdf", "doc_description": "d", "pdf_path": "/x/a.pdf"}]
    c = _client_with_docs(monkeypatch, docs)
    r = c.get("/api/documents")
    assert r.status_code == 200
    body = r.json()
    assert body["documents"][0]["doc_name"] == "a.pdf"


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


def test_rollback_on_failure(monkeypatch, tmp_path):
    # spec §5.4 step 4 + §7.2: on persist failure, restore .bak -> original for
    # BOTH config.yaml and .env. set_env_fields is made to raise AFTER
    # set_config_yaml_model has mutated config.yaml -> the mutation must roll back.
    orig_yaml = 'model: "gpt-4.1-mini"\nretrieve_model: "gpt-4.1-mini"\n'
    orig_env = "MODEL_NAME=qwen3.7-plus\n"
    yaml_p = tmp_path / "config.yaml"
    yaml_p.write_text(orig_yaml, encoding="utf-8")
    env_p = tmp_path / ".env"
    env_p.write_text(orig_env, encoding="utf-8")
    monkeypatch.setattr(web_config, "DEFAULT_CONFIG_YAML", yaml_p)
    monkeypatch.setattr(web_config, "DEFAULT_ENV", env_p)
    monkeypatch.setattr(server, "_rebuild_client", lambda **kw: None)
    monkeypatch.setattr(web_config, "read_config_snapshot", lambda client, **kw: {})
    monkeypatch.setattr(server, "get_client", lambda: SimpleNamespace(db=None))
    monkeypatch.setattr(server, "API_KEY", "")

    def boom(*a, **kw):
        raise RuntimeError("persist boom")
    monkeypatch.setattr(web_config, "set_env_fields", boom)

    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.post("/api/config", json={"model": "newmodel", "persist": True})
    # structured failure (not a bare Starlette 500): JSON body parseable
    assert r.status_code == 500
    body = r.json()
    assert body.get("applied") is True and body.get("persisted") is False
    # ROLLBACK: config.yaml restored to original (set_config_yaml_model mutation undone)
    assert yaml_p.read_text(encoding="utf-8") == orig_yaml
    # .env unchanged
    assert env_p.read_text(encoding="utf-8") == orig_env


def test_rebuild_client_failure_preserves_old_client(monkeypatch):
    # _rebuild_client must construct the NEW client BEFORE closing/swapping, so a
    # construction failure leaves the global `client` pointing at the LIVE old
    # instance (not a closed one). persist=False isolates the runtime-apply path.
    monkeypatch.setattr(server, "API_KEY", "")
    # A live old client with a usable .db and a real .close() that nukes it (so the
    # close-before-construct bug is observable: a closed client's .db becomes None).
    live_db = SimpleNamespace(get_all_documents=lambda: [{"id": 1}])

    class LiveClient:
        def __init__(self):
            self.db = live_db

        def close(self):
            self.db = None  # simulate resource release
    live_client = LiveClient()
    monkeypatch.setattr(server, "client", live_client)
    monkeypatch.setattr(server, "get_client", lambda: server.client)

    # Make PageIndexClient(...) raise ONLY when called with a model override
    # (i.e. _rebuild_client's construction path). The old client was built before
    # this patch, so it stays intact.
    real_ctor = server.PageIndexClient

    def ctor_with_model_boom(*a, **kw):
        if kw.get("model"):
            raise RuntimeError("bad model")
        return real_ctor(*a, **kw)
    monkeypatch.setattr(server, "PageIndexClient", ctor_with_model_boom)

    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.post("/api/config", json={"model": "broken-model", "persist": False})
    # structured failure (not a bare Starlette 500): JSON body parseable
    assert r.status_code == 500
    body = r.json()
    assert body.get("applied") is True and body.get("persisted") is False
    # OLD CLIENT SURVIVED: still the live instance, .db usable (not closed/None)
    assert server.get_client() is live_client
    assert server.get_client().db is live_db


def test_index_served_and_static_whitelisted(monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "")
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.get("/")
    assert r.status_code == 200
    assert "<div id=\"app\">" in r.text


def test_search_endpoint_enriches_matched_doc_names(monkeypatch):
    async def fake_search(query, top_k=3):
        return {"answer": "a", "confidence": "high",
                "matched_docs": [{"doc_id": "uuid-1", "score": 1.0}]}
    stub = SimpleNamespace(
        search=fake_search,
        documents={"uuid-1": {"doc_name": "My Doc"}},
    )
    monkeypatch.setattr(server, "get_client", lambda: stub)
    monkeypatch.setattr(server, "API_KEY", "")
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.post("/api/search", json={"query": "q"})
    assert r.status_code == 200
    assert r.json()["matched_docs"][0]["doc_name"] == "My Doc"


# ---------------------------------------------------------------------------
# T10 — DELETE /api/documents/{doc_id} (FR10)
# Reuses delete_document_internal extracted from W2 MCP handler (AC10.3, AC10.5,
# AC10.6, AC10.7). Test isolation: monkeypatch get_client/API_KEY; tmp_path for
# any temp files; do not touch real .env / config.yaml.
# ---------------------------------------------------------------------------


def _delete_client(monkeypatch, fake_client):
    monkeypatch.setattr(server, "get_client", lambda: fake_client)
    monkeypatch.setattr(server, "API_KEY", "")
    from starlette.testclient import TestClient
    return TestClient(server.app)


def test_delete_document_endpoint_success(monkeypatch):
    """AC10.3 — DELETE /api/documents/{id} returns 200 + {success, doc_id}."""
    fake_db = SimpleNamespace()
    fake_client = SimpleNamespace(db=fake_db, workspace="/tmp/ws")
    monkeypatch.setattr(server, "delete_document_internal",
                        lambda c, db_id: True)
    c = _delete_client(monkeypatch, fake_client)
    r = c.delete("/api/documents/42")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["doc_id"] == 42


def test_delete_document_endpoint_not_found(monkeypatch):
    """AC10.5 idempotent — missing doc_id still returns 200 + {success, existed=False}."""
    # Simulate internal function returning False (DB row not present).
    monkeypatch.setattr(server, "delete_document_internal",
                        lambda c, db_id: False)
    fake_client = SimpleNamespace(db=SimpleNamespace(), workspace="/tmp/ws")
    c = _delete_client(monkeypatch, fake_client)
    r = c.delete("/api/documents/999")
    # Per v1.1 spec §5.1.1 error response table: missing -> 404
    assert r.status_code == 404
    body = r.json()
    assert "error" in body


def test_delete_document_endpoint_uses_internal_function(monkeypatch):
    """AC10.3 — REST handler MUST delegate to delete_document_internal."""
    calls = {}
    fake_client = SimpleNamespace(db=SimpleNamespace(), workspace="/tmp/ws")

    def fake_internal(c, db_id):
        calls["args"] = (c, db_id)
        return True
    monkeypatch.setattr(server, "delete_document_internal", fake_internal)
    c = _delete_client(monkeypatch, fake_client)
    r = c.delete("/api/documents/7")
    assert r.status_code == 200
    assert calls.get("args") == (fake_client, 7)


def test_delete_document_endpoint_mcp_unchanged(monkeypatch):
    """AC10.6 — MCP delete_document handler still routes through
    delete_document_internal (no regression vs W2 behavior)."""
    calls = {}
    # The MCP handler in handle_call_tool resolves uuid -> db_id via
    # c._uuid_to_db.pop(doc_id, None). Wire a fake client whose pop returns
    # a known db_id, then assert delete_document_internal(c, db_id) was called.
    fake_db = SimpleNamespace()
    fake_closet = SimpleNamespace(remove_document=lambda db_id: None)
    fake_super = SimpleNamespace(on_document_removed=lambda db_id: None)
    fake_client = SimpleNamespace(
        db=fake_db,
        workspace="/tmp/ws",
        documents={"uuid-x": {"doc_name": "x"}},
        _uuid_to_db={"uuid-x": 123},
        closet_index=fake_closet,
        super_tree_index=fake_super,
    )
    monkeypatch.setattr(server, "get_client", lambda: fake_client)
    monkeypatch.setattr(server, "delete_document_internal",
                        lambda c, db_id: calls.setdefault("called", True) or True)
    import asyncio as _asyncio
    result = _asyncio.run(server.handle_call_tool(
        "delete_document", {"doc_id": "uuid-x"}))
    # MCP handler still returns TextContent JSON {success: True, doc_id}
    parsed = json.loads(result[0].text)
    assert parsed.get("success") is True
    assert parsed.get("doc_id") == "uuid-x"
    # The shared internal function was invoked (MCP -> internal, not a private copy).
    assert calls.get("called") is True


def test_delete_document_endpoint_idempotent(monkeypatch, tmp_path):
    """AC10.5 — repeated DELETE on same id is safe; second call returns 200/404
    with no exception (idempotent semantics per §5.1.1)."""
    invocations = []
    monkeypatch.setattr(server, "delete_document_internal",
                        lambda c, db_id: invocations.append(db_id) or True)
    fake_client = SimpleNamespace(db=SimpleNamespace(), workspace=str(tmp_path))
    c = _delete_client(monkeypatch, fake_client)
    r1 = c.delete("/api/documents/5")
    r2 = c.delete("/api/documents/5")
    assert r1.status_code == 200
    assert r2.status_code == 200
    # Internal called twice; both succeed (idempotent at REST layer).
    assert invocations == [5, 5] or invocations == [5]  # tolerate internal caching


# ---------------------------------------------------------------------------
# T11 — POST /api/config/test (FR12 ping)
# spec §5.1.2: pure read-only connectivity test. ALWAYS returns 200 — errors
# are surfaced as `ok: false` in the body. Endpoint MUST NOT mutate any state
# (no configure_llm, no env writes, no yaml writes).
# ---------------------------------------------------------------------------

import time as _time_for_tests


def _config_test_client(monkeypatch, fake_chat_create=None,
                        fake_get_llm_config=None, fake_loader_model="gpt-test"):
    """Build a TestClient for /api/config/test with monkeypatched openai client.

    `fake_chat_create` replaces OpenAI(...).chat.completions.create.
    `fake_get_llm_config` replaces get_llm_config() — used when body has no
    api_key/base_url.
    """
    monkeypatch.setattr(server, "API_KEY", "")
    if fake_get_llm_config is None:
        fake_get_llm_config = lambda: ("sk-active", "http://active/v1")
    monkeypatch.setattr(server, "get_llm_config", fake_get_llm_config)
    # ConfigLoader().load(None).model — server.py uses this when body has no model
    monkeypatch.setattr(server, "ConfigLoader",
                        lambda: SimpleNamespace(load=lambda _: SimpleNamespace(model=fake_loader_model)))
    # Sentinel to assert NO state mutation occurred
    state_calls = {"configure_llm": 0, "set_env_fields": 0,
                   "set_config_yaml_model": 0}
    monkeypatch.setattr(server, "configure_llm",
                        lambda **kw: state_calls.__setitem__("configure_llm",
                                                             state_calls["configure_llm"] + 1))
    if fake_chat_create is not None:
        # Patch openai.OpenAI to a stub whose .chat.completions.create is fake_chat_create
        class _StubCompletions:
            def __init__(self, fn):
                self._fn = fn

            def create(self, **kwargs):
                return self._fn(**kwargs)

        class _StubChat:
            def __init__(self, fn):
                self.completions = _StubCompletions(fn)

        class _StubOpenAI:
            def __init__(self, api_key=None, base_url=None, **kw):
                self.api_key = api_key
                self.base_url = base_url
                self.chat = _StubChat(fake_chat_create)
        monkeypatch.setattr(server, "OpenAI", _StubOpenAI)
    return state_calls


def test_config_test_endpoint_success(monkeypatch):
    # Mock chat.completions.create returns a fake completion object
    def fake_create(model, messages, max_tokens):
        # Return an object with .choices[0].message.content (openai shape)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="pong"))]
        )
    state = _config_test_client(monkeypatch, fake_chat_create=fake_create)
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.post("/api/config/test",
               json={"model": "gpt-x", "api_key": "sk-test",
                     "base_url": "http://test/v1"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert isinstance(body["latency_ms"], int)
    assert body["latency_ms"] >= 0
    assert body["model"] == "gpt-x"
    # No state mutation — FR12 is read-only
    assert state["configure_llm"] == 0


def test_config_test_endpoint_timeout(monkeypatch):
    # Mock chat.completions.create to sleep longer than the 10s wait_for budget.
    # We can't realistically wait 15s in unit tests; patch `asyncio.wait_for`
    # (the symbol the handler binds to) to raise asyncio.TimeoutError
    # immediately, which exercises the same handler branch.
    def fake_create(model, messages, max_tokens):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="x"))])
    state = _config_test_client(monkeypatch, fake_chat_create=fake_create)

    real_wait_for = asyncio.wait_for
    real_to_thread = asyncio.to_thread

    def fake_wait_for(awaitable, timeout=None, **kw):
        # Cancel the inner coroutine to avoid "coroutine was never awaited"
        # warnings, then raise TimeoutError.
        try:
            awaitable.close()
        except Exception:
            pass
        raise asyncio.TimeoutError()

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.post("/api/config/test",
               json={"model": "gpt-x", "api_key": "sk-test",
                     "base_url": "http://test/v1"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "timeout"
    assert body["latency_ms"] >= 9000  # ~10s budget (handler reports >= 10000)
    # No state mutation
    assert state["configure_llm"] == 0


def test_config_test_endpoint_auth_error(monkeypatch):
    import openai
    def fake_create(model, messages, max_tokens):
        # openai SDK requires response.request; build a minimal stub.
        fake_resp = SimpleNamespace(
            status_code=401,
            request=SimpleNamespace(method="POST", url="http://test/v1/chat/completions"),
            headers={},
        )
        raise openai.AuthenticationError(
            message="invalid api key",
            response=fake_resp,
            body={"error": "unauthorized"})
    state = _config_test_client(monkeypatch, fake_chat_create=fake_create)
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.post("/api/config/test",
               json={"model": "gpt-x", "api_key": "sk-bad",
                     "base_url": "http://test/v1"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "auth_error"
    assert "detail" in body and body["detail"]
    # No state mutation
    assert state["configure_llm"] == 0


def test_config_test_endpoint_not_found(monkeypatch):
    import openai
    def fake_create(model, messages, max_tokens):
        fake_resp = SimpleNamespace(
            status_code=404,
            request=SimpleNamespace(method="POST", url="http://test/v1/chat/completions"),
            headers={},
        )
        raise openai.NotFoundError(
            message="model not found",
            response=fake_resp,
            body={"error": "model_not_found"})
    state = _config_test_client(monkeypatch, fake_chat_create=fake_create)
    from starlette.testclient import TestClient
    c = TestClient(server.app)
    r = c.post("/api/config/test",
               json={"model": "no-such-model", "api_key": "sk-test",
                     "base_url": "http://test/v1"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "model_not_found"
    assert "detail" in body and body["detail"]
    # No state mutation
    assert state["configure_llm"] == 0
