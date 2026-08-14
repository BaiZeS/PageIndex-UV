"""P1 [S5] 证据束构建原语测试 —— 四通道原始命中带来源 + 图谱 CTE 关联。"""

import asyncio
import sys

import pytest

from pageindex_mutil.agentic.evidence import (
    build_evidence_bundle,
    derive_evidence_score,
    render_doc_evidence,
)


def _make_client(tmp_path, docs, keywords, tags):
    from pageindex_mutil.client import PageIndexClient
    from db import PageIndexDB
    db = PageIndexDB(str(tmp_path / "t.db"))
    # search_backend="keyword" 保持 vectorless（不加载 embedding 模型）。
    client = PageIndexClient(db_path=str(tmp_path / "t.db"), search_backend="keyword")
    client.db = db
    client.closet_index = None
    client.search_backend = None
    for doc_id, name, desc, kws in docs:
        did = db.insert_document(pdf_name=name, pdf_path="", doc_description=desc)
        records = [(did, tok, field, tf) for tok, field, tf in kws]
        db.insert_doc_keywords(did, records)
    return client, db


def test_bundle_keyword_field_provenance(tmp_path):
    # 注：jieba 会把 "浴血值" 切成 "浴血"+"值"（"值" 单字被丢弃），故用
    # jieba 稳定 token "浴血"，语义不变——正文(content)命中与标题(node_title)
    # 命中的 field 来源均可追溯。
    client, db = _make_client(tmp_path,
        [("A", "文档A", "", [("浴血", "content", 3), ("帮会", "node_title", 1)]),
         ("B", "文档B", "", [("帮会", "content", 2)])], None, None)
    bundle, ctx = build_evidence_bundle(client, db, "帮会浴血怎么获得", topk=30)
    a_kw = {(k["token"], k["field"]) for k in bundle[1]["channels"]["keyword"]}
    assert ("浴血", "content") in a_kw
    assert ("帮会", "node_title") in a_kw  # field 来源可追溯


def test_evidence_score_multi_signal(tmp_path):
    entry = {"channels": {"keyword": [{"token": "x"}], "tag": [{"text": "y"}], "entity": [{"name": "z"}]}, "graph": {}}
    assert derive_evidence_score(entry) == 3 * 1 + 2 * 1 + 1 * 1


def _graph_client(tmp_path):
    from pageindex_mutil.client import PageIndexClient
    from db import PageIndexDB
    db = PageIndexDB(str(tmp_path / "t.db"))
    client = PageIndexClient(db_path=str(tmp_path / "t.db"), search_backend="keyword")
    client.db = db
    client.closet_index = None
    client.search_backend = None
    return client, db


def test_graph_channel_links_neighbor_entities(tmp_path):
    # 图谱通道语义：CTE 邻居实体（非 query 实体）→ 提及它们的文档。
    # query 实体走 entity 通道不变；graph 通道只挂邻居实体的 doc_entity_links。
    client, db = _graph_client(tmp_path)
    doc_a = db.insert_document(pdf_name="A", pdf_path="", doc_description="")
    doc_b = db.insert_document(pdf_name="B", pdf_path="", doc_description="")
    e1 = db.insert_entity("concept", "浴血", [])
    e2 = db.insert_entity("section", "门派介绍", [])
    db.insert_entity_relation(e1, "related_to", e2)
    db.insert_entity_mention(e1, doc_a)
    db.insert_entity_mention(e2, doc_b)

    bundle, ctx = build_evidence_bundle(client, db, "浴血", topk=30)

    # entity 通道：query 实体 浴血 → doc A
    assert any(x["name"] == "浴血" for x in bundle[doc_a]["channels"]["entity"])
    # graph 通道：邻居实体 门派介绍（距离1·related_to·0.42）→ doc B
    assert bundle[doc_b]["graph"]["doc_entity_links"] == [
        {"entity": "门派介绍", "distance": 1, "relation_type": "related_to", "weight": 0.42}
    ]


def test_render_shows_field_and_missing_docs(tmp_path):
    client, db = _make_client(
        tmp_path,
        [("A", "文档A", "", [("浴血", "content", 3)])],
        None,
        None,
    )
    bundle, ctx = build_evidence_bundle(client, db, "浴血", topk=30)
    text = render_doc_evidence(bundle, [1])
    assert "浴血(content)" in text

    # 缺席 db_id（不在 bundle 中）须显式注记，而非静默跳过
    missing = render_doc_evidence({}, [999])
    assert "doc 999: 无通道命中" in missing


def test_tag_text_populated(tmp_path):
    client, db = _graph_client(tmp_path)
    did = db.insert_document(pdf_name="A", pdf_path="", doc_description="")
    db.insert_closet_tags(did, [(did, "帮会活动", "帮会 活动", 0.9, "llm")])

    class _StubCloset:
        def search(self, query, top_k=10):
            return [(did, 0.9)]

    client.closet_index = _StubCloset()
    bundle, ctx = build_evidence_bundle(client, db, "帮会活动", topk=30)
    assert bundle[did]["channels"]["tag"] == [{"text": "帮会活动", "confidence": 0.9}]


def test_holistic_select_uses_evidence_bundle(tmp_path, monkeypatch):
    """evidence_bundle 传入时 prompt 证据块来自证据束（field 来源可见）。"""
    client, db = _make_client(
        tmp_path,
        [("A", "文档A", "", [("浴血", "content", 3)]),
         ("B", "文档B", "", [])],
        None,
        None,
    )
    from pageindex_mutil.super_tree import SuperTreeIndex
    from pageindex_mutil.agentic.evidence import build_evidence_bundle
    st = SuperTreeIndex(db, "m", client)
    bundle, ctx = build_evidence_bundle(client, db, "浴血", topk=30)
    captured = {}

    async def fake_llm(model, prompt, **kw):
        captured["p"] = prompt
        return '{"doc_ids": []}'

    # 打补丁必须落在类真实所在的模块对象上：test_super_tree.py 导入期会用 stub
    # 模块 clobber sys.modules["pageindex_mutil.super_tree"]，字符串路径经
    # monkeypatch.resolve 走的是 pageindex_mutil 包属性（可能是另一个模块对象），
    # 合并运行时会打空导致 fake_llm 不被调用。故按 __module__ 反查 sys.modules。
    st_module = sys.modules[SuperTreeIndex.__module__]
    monkeypatch.setattr(st_module, "llm_acompletion", fake_llm)
    # KBIdentity 会同步调 llm_completion；屏蔽以隔离本用例（只验证证据源切换）。
    monkeypatch.setattr(st_module, "llm_completion", lambda *a, **k: "测试知识库")
    asyncio.run(st._holistic_select("浴血", [1, 2], evidence_bundle=bundle))
    assert "浴血(content)" in captured["p"]


def test_bundle_reaches_holistic_select_via_select_documents(tmp_path, monkeypatch):
    """义务 A：evidence_bundle 经 select_documents → _select_documents_reasoning →
    _holistic_select 生产链路到达 prompt（证据束渲染行 field 来源可见）。"""
    client, db = _make_client(
        tmp_path,
        [("A", "文档A", "", [("浴血", "content", 3)]),
         ("B", "文档B", "", [])],
        None,
        None,
    )
    from pageindex_mutil.super_tree import SuperTreeIndex
    from pageindex_mutil.agentic.evidence import build_evidence_bundle
    st = SuperTreeIndex(db, "m", client)
    bundle, ctx = build_evidence_bundle(client, db, "浴血", topk=30)
    captured = {}

    async def fake_llm(model, prompt, **kw):
        captured["p"] = prompt
        return '{"doc_ids": []}'

    st_module = sys.modules[SuperTreeIndex.__module__]
    monkeypatch.setattr(st_module, "llm_acompletion", fake_llm)
    monkeypatch.setattr(st_module, "llm_completion", lambda *a, **k: "测试知识库")
    asyncio.run(st.select_documents("浴血", {1: 1.0, 2: 1.0}, evidence_bundle=bundle))
    assert "浴血(content)" in captured["p"]


def test_bundle_max_hop_param_consumed(tmp_path, monkeypatch):
    """义务 A：build_evidence_bundle 接受 max_hop 可选参数并透传给 CTE（替代硬编码 3）。"""
    client, db = _graph_client(tmp_path)
    doc = db.insert_document(pdf_name="A", pdf_path="", doc_description="")
    e1 = db.insert_entity("concept", "浴血", [])
    db.insert_entity_mention(e1, doc)
    captured = {}

    def fake_cte(query_ids, max_hop=3):
        captured["max_hop"] = max_hop
        return {}

    monkeypatch.setattr(db, "get_entity_distances_cte", fake_cte)
    build_evidence_bundle(client, db, "浴血", topk=30, max_hop=5)
    assert captured["max_hop"] == 5


def test_bundle_returns_query_ctx(tmp_path):
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


def test_entity_channel_carries_node_id_and_name_node_dedup(tmp_path):
    """[S7]/[S5] entity 通道携带 node_id；(name, node_id) 去重保多节点同实体归属；
    derive_evidence_score/render_doc_evidence 按 name 去重（同名只计一次/只列一次）。"""
    client, db = _graph_client(tmp_path)
    doc = db.insert_document(pdf_name="A", pdf_path="", doc_description="")
    e1 = db.insert_entity("concept", "浴血", [])
    # 同一实体在同一文档的两个节点各提及一次（同 doc，不同 node_id）
    db.insert_entity_mention(e1, doc, node_id="n1")
    db.insert_entity_mention(e1, doc, node_id="n2")

    bundle, ctx = build_evidence_bundle(client, db, "浴血", topk=30)

    ents = bundle[doc]["channels"]["entity"]
    # 每条都携带 node_id
    assert all("node_id" in x for x in ents)
    # (name, node_id) 去重：同名不同 node 各成一条（保多节点同实体归属）
    assert {(x["name"], x["node_id"]) for x in ents} == {("浴血", "n1"), ("浴血", "n2")}
    # derive_evidence_score：同名只计一次 → 3.0（而非 3.0*2）
    assert derive_evidence_score(bundle[doc]) == 3.0
    # render_doc_evidence：同名只列一次
    text = render_doc_evidence(bundle, [doc])
    assert text.count("浴血") == 1
