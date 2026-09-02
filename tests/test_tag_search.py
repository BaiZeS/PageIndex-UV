"""P1 [Channel A] closet 语义标签查询期检索测试 —— 子串语义修复（A0「tag 恒零命中」）。

真实索引产物形状：`closet_index._tokenize_tag` 把 `tag_token` 落成**空格连接的多词串**
（如 `'游戏 世界观 门派 体系'`，抽样 97.9% 多 token）。旧查询链路
`ClosetIndex.search → db.match_closet_tags` 以 `tag_token IN (单个查询词)` **整串精确
匹配** → 真实数据上恒零命中；渲染侧 `UnifiedNodeEnhancement._tag_hits` 却是子串语义。
本文件钉住：

1. 多词串行**能命中**（score = 命中 tag confidence 之和、provenance 带文本、topk 截断）；
2. 旧的等值精确匹配在该形状上确实零命中（证明修的是真缺陷，且不再走恒零路径）；
3. 搜索侧与渲染侧 `_tag_hits` 判定逐标签一致（标准严格对齐）；
4. 每查询**一次 SELECT**（消除 matched_docs × get_doc_tags 的 N+1）；
5. `build_evidence_bundle` tag 通道真实回填 + `no_tag` 消融 patch 位（tag_search.search_tags）生效。
"""

import pytest

from db import PageIndexDB
from pageindex_mutil.agentic.tag_search import search_tags

pytest.importorskip("jieba")  # 生产词项口径来自 jieba 分词


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

def _tokens(query):
    """生产口径词项：与 evidence.ctx["tokens"]、渲染侧 _tag_hits 同一份。"""
    from pageindex_mutil.super_tree import KeywordIndex
    return KeywordIndex._tokenize(None, query)


def _doc(db, name, tags):
    """tags: [(tag_text, tag_token, confidence, source)]，直插多词串真实形状。"""
    did = db.insert_document(pdf_name=name, pdf_path="", doc_description="")
    db.insert_closet_tags(did, [(did, t, tok, c, src) for t, tok, c, src in tags])
    return did


# ---------------------------------------------------------------------------
# 1. 多词串 tag_token 命中（主修）
# ---------------------------------------------------------------------------

def test_multiword_tag_token_hits_single_query_token(tmp_path):
    db = PageIndexDB(str(tmp_path / "t.db"))
    hit = _doc(db, "文档A", [("游戏世界观门派体系", "游戏 世界观 门派 体系", 0.9, "llm")])
    _doc(db, "文档B", [("电子表格自动化", "电子表格 自动化", 0.95, "llm")])

    res = search_tags(None, db, ["游戏"], topk=10, query="游戏世界观门派体系是怎么设定的")

    assert [r[0] for r in res] == [hit]                      # 命中的是含该词的 doc
    doc_id, score, provenance = res[0]
    assert doc_id == hit
    assert score == pytest.approx(0.9)                       # score = 命中 tag confidence 之和
    assert provenance == [("游戏世界观门派体系", 0.9)]          # tag 文本随查带出
    db.close()


def test_score_sums_all_hit_tags_of_doc(tmp_path):
    db = PageIndexDB(str(tmp_path / "t.db"))
    did = _doc(db, "文档A", [
        ("游戏世界观设定", "游戏 世界观 设定", 0.9, "llm"),
        ("游戏数值成长曲线", "游戏 数值 成长 曲线", 0.8, "llm"),
        ("电子表格自动化", "电子表格 自动化", 0.95, "llm"),   # 不相关，不计
    ])
    res = search_tags(None, db, _tokens("游戏的门派与数值成长"), topk=10, query="游戏的门派与数值成长")
    assert res[0][0] == did
    assert res[0][1] == pytest.approx(1.7)
    assert len(res[0][2]) == 2
    db.close()


def test_topk_truncation_and_scoreordering(tmp_path):
    db = PageIndexDB(str(tmp_path / "t.db"))
    low = _doc(db, "弱", [("游戏周边玩法", "游戏 周边 玩法", 0.4, "llm")])
    high = _doc(db, "强", [("游戏世界观架构", "游戏 世界观 架构", 0.95, "llm"),
                           ("游戏社交生态", "游戏 社交 生态", 0.9, "llm")])
    mid = _doc(db, "中", [("游戏生命周期管理", "游戏 生命周期 管理", 0.7, "llm")])

    full = search_tags(None, db, ["游戏"], topk=10, query="游戏")
    assert [r[0] for r in full] == [high, mid, low]          # score 降序
    assert [round(r[1], 2) for r in full] == [1.85, 0.7, 0.4]

    capped = search_tags(None, db, ["游戏"], topk=2, query="游戏")
    assert [r[0] for r in capped] == [high, mid]             # topk 截断
    db.close()


def test_case_insensitive_matching(tmp_path):
    db = PageIndexDB(str(tmp_path / "t.db"))
    did = _doc(db, "DocA", [("Spreadsheet Automation", "Spreadsheet Automation", 0.9, "llm")])
    res = search_tags(None, db, ["spreadsheet"], topk=5, query="SPREADSHEET automation")
    assert [r[0] for r in res] == [did]
    res2 = search_tags(None, db, ["automation"], topk=5, query="automation")
    assert [r[0] for r in res2] == [did]
    db.close()


def test_source_llm_only_fallback_rows_excluded(tmp_path):
    """[7.2] 分层口径不变：jieba 兜底原词（source='fallback'）不入语义通道。"""
    db = PageIndexDB(str(tmp_path / "t.db"))
    _doc(db, "文档A", [("游戏", "游戏", 0.3, "fallback")])
    assert search_tags(None, db, ["游戏"], topk=5, query="游戏") == []
    db.close()


# ---------------------------------------------------------------------------
# 2. 旧路径（等值精确匹配）在该形状上确实恒零命中 → 不再走
# ---------------------------------------------------------------------------

def test_legacy_exact_in_list_path_is_vacuous_on_real_shape(tmp_path):
    from pageindex_mutil.closet_index import ClosetIndex

    db = PageIndexDB(str(tmp_path / "t.db"))
    did = _doc(db, "文档A", [("游戏世界观门派体系", "游戏 世界观 门派 体系", 0.9, "llm")])

    # 索引期形状事实：tag_token 是多词串，与任何单个查询词都不等值。
    rows = db._connect().execute(
        "SELECT tag_token FROM closet_tags WHERE source='llm'").fetchall()
    assert any(" " in r[0] for r in rows)

    # 旧检索语义：等值匹配 → 零命中（缺陷本身）。
    assert db.match_closet_tags(["游戏"], top_k=5, source="llm") == []
    assert ClosetIndex(db, "m").search("游戏的门派体系", top_k=5) == []

    # 新检索语义：同一标签行可命中。
    assert [r[0] for r in search_tags(None, db, ["游戏"], topk=5, query="游戏的门派体系")] == [did]
    db.close()


# ---------------------------------------------------------------------------
# 3. 与渲染侧 _tag_hits 标准严格一致
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag_text", [
    "游戏世界观门派体系",   # 词项是标签子串 → 命中
    "电子表格自动化",       # 完全无关 → 不命中
    "帮会",                 # 标签是词项子串（反向包含）
    "浴血",                 # 与词项等值
    "X",                    # 单字：_tag_hits 的 len>=2 守卫 → 不命中
    "",                     # 空标签 → 不命中
])
def test_search_side_parity_with_render_side_tag_hits(tmp_path, tag_text):
    from pageindex_mutil.agentic.enhance import UnifiedNodeEnhancement

    db = PageIndexDB(str(tmp_path / "t.db"))
    query = "游戏帮会浴血获得"
    tokens = _tokens(query)
    _doc(db, "文档A", [(tag_text, tag_text, 0.9, "llm")])

    rendered = UnifiedNodeEnhancement._tag_hits(tokens, query.casefold(), [tag_text])
    hit_docs = {r[0] for r in search_tags(None, db, tokens, topk=10, query=query)}

    assert (len(hit_docs) == 1) == bool(rendered), (tag_text, tokens, rendered, hit_docs)
    db.close()


# ---------------------------------------------------------------------------
# 4. 一次 SELECT（无 N+1）
# ---------------------------------------------------------------------------

def test_single_select_per_query_no_n_plus_one(tmp_path):
    class _CountingDB:
        def __init__(self, db):
            self._db = db
            self.calls = 0

        def _connect(self):
            db = self._db

            class _Conn:
                def execute(_self, sql, params=()):
                    self.calls += 1
                    return db._connect().execute(sql, params)

            return _Conn()

    db = PageIndexDB(str(tmp_path / "t.db"))
    for i in range(6):
        _doc(db, f"文档{i}", [(f"游戏体系{i}", f"游戏 体系 {i}", 0.6 + i / 100, "llm")])
    wrapped = _CountingDB(db)

    res = search_tags(None, wrapped, ["游戏"], topk=10, query="游戏")

    assert len(res) == 6
    assert wrapped.calls == 1, f"每查询只允许一次 SELECT，实际 {wrapped.calls}"
    db.close()


# ---------------------------------------------------------------------------
# 5. 边界输入
# ---------------------------------------------------------------------------

def test_empty_inputs_return_empty(tmp_path):
    db = PageIndexDB(str(tmp_path / "t.db"))
    _doc(db, "文档A", [("游戏世界观", "游戏 世界观", 0.9, "llm")])
    assert search_tags(None, db, [], topk=10, query="") == []
    assert search_tags(None, db, ["  ", ""], topk=10, query="") == []
    assert search_tags(None, None, ["游戏"], topk=10, query="游戏") == []   # 无 db 句柄
    db.close()


def test_empty_workspace_returns_empty(tmp_path):
    db = PageIndexDB(str(tmp_path / "empty.db"))
    assert search_tags(None, db, ["游戏"], topk=10, query="游戏") == []
    db.close()


# ---------------------------------------------------------------------------
# 6. 证据束接线（bundle schema 不变）+ 消融 patch 位
# ---------------------------------------------------------------------------

def _client(db):
    from pageindex_mutil.client import PageIndexClient
    client = PageIndexClient(db_path=str(db.db_path), search_backend="keyword")
    client.db = db
    client.search_backend = None
    return client


def test_bundle_tag_channel_populated_with_text(tmp_path):
    from pageindex_mutil.agentic.evidence import build_evidence_bundle, derive_evidence_score

    db = PageIndexDB(str(tmp_path / "t.db"))
    client = _client(db)
    assert client.closet_index is not None          # 通道开关（closet 可用）
    did = _doc(db, "文档A", [("游戏世界观门派体系", "游戏 世界观 门派 体系", 0.9, "llm")])

    query = "游戏的门派体系怎么设定"
    bundle, ctx = build_evidence_bundle(client, db, query, topk=30)

    assert bundle[did]["channels"]["tag"] == [{"text": "游戏世界观门派体系", "confidence": 0.9}]
    assert set(bundle[did]["channels"]) == {"tag", "keyword", "entity", "vector"}  # schema 不动
    assert derive_evidence_score(bundle[did]) >= 2.0                               # tag 权重生效
    assert "游戏" in ctx["tokens"]                                                 # 两侧同一份词项
    db.close()


def test_bundle_tag_channel_ablated_via_tag_search_patch(tmp_path, monkeypatch):
    """消融 harness no_tag 的新 patch 位：tag_search.search_tags 置空即为关通道 A。"""
    from pageindex_mutil.agentic import tag_search
    from pageindex_mutil.agentic.evidence import build_evidence_bundle

    db = PageIndexDB(str(tmp_path / "t.db"))
    client = _client(db)
    _doc(db, "文档A", [("游戏世界观门派体系", "游戏 世界观 门派 体系", 0.9, "llm")])

    def _disabled(*a, **kw):
        return []

    monkeypatch.setattr(tag_search, "search_tags", _disabled)
    bundle, ctx = build_evidence_bundle(client, db, "游戏的门派体系怎么设定", topk=30)
    assert all(not e["channels"]["tag"] for e in bundle.values())
    db.close()
