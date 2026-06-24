import textwrap
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
