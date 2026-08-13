"""P1 [S5] 证据束构建原语测试 —— 四通道原始命中带来源 + 图谱 CTE 关联。"""

import pytest

from pageindex_mutil.agentic.evidence import build_evidence_bundle, derive_evidence_score


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
    bundle = build_evidence_bundle(client, db, "帮会浴血怎么获得", topk=30)
    a_kw = {(k["token"], k["field"]) for k in bundle[1]["channels"]["keyword"]}
    assert ("浴血", "content") in a_kw
    assert ("帮会", "node_title") in a_kw  # field 来源可追溯


def test_evidence_score_multi_signal(tmp_path):
    entry = {"channels": {"keyword": [{"token": "x"}], "tag": [{"text": "y"}], "entity": [{"name": "z"}]}, "graph": {}}
    assert derive_evidence_score(entry) == 3 * 1 + 2 * 1 + 1 * 1
