"""T5.2 / P1.2-entity + T5.4 / P1.4-keywords: node profile signatures.

Scope:
- Deterministic entity→TOC-node attribution (fills entity_mentions.node_id).
- node_profiles table round-trip (upsert/get accessors).
- Canonical entity names in profiles (post merge / post batch-normalization).
- "entities" / "keywords" keys attached to doc["structure"] node dicts.
- Node-level salient keywords: TF-IDF top-K per node, no LLM (P1.4).

All LLM calls mocked. No real LLM, no vectors, no new LLM call sites.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# NOTE: test_router.py pre-seeds sys.modules with stub pageindex_mutil.* modules,
# and test_retrieve_model_wiring.py purges + re-imports them at collection time
# (creating fresh module objects). Purge stubs, import the REAL modules once
# here, and hold module/class references — all patching uses patch.object on
# these references so patches always target the same module object the classes
# under test use, regardless of later sys.modules mutation by other test files.
for _mod in list(sys.modules):
    if _mod == "pageindex_mutil" or _mod.startswith("pageindex_mutil."):
        del sys.modules[_mod]

import pageindex_mutil.client as client_mod
import pageindex_mutil.entity_extractor as entity_extractor_mod
from db import PageIndexDB
from pageindex_mutil.client import PageIndexClient
from pageindex_mutil.entity_extractor import Entity


@pytest.fixture
def db(tmp_path):
    d = PageIndexDB(str(tmp_path / "test.db"))
    yield d
    d.close()


@pytest.fixture
def client_factory(tmp_path):
    """Create PageIndexClient instances with a temp DB (+ optional workspace)."""
    sys.modules["PyPDF2"] = MagicMock()

    def _make(workspace=True):
        return PageIndexClient(
            db_path=str(tmp_path / "test.db"),
            workspace=str(tmp_path / "ws") if workspace else None,
            search_backend="keyword",
        )

    return _make


def _two_node_structure():
    """Node A mentions 浴血值 (text+summary), node B does not."""
    return [
        {
            "node_id": "node-a",
            "title": "章节A",
            "text": "本节点介绍浴血值机制，浴血值是核心概念。",
            "summary": "浴血值说明",
            "level": 1,
            "nodes": [],
        },
        {
            "node_id": "node-b",
            "title": "章节B",
            "text": "其他机制说明，不含目标词。",
            "summary": "其他内容",
            "level": 1,
            "nodes": [],
        },
    ]


def _mock_extract_single(entity_name="浴血值", entity_type="concept", aliases=None):
    """Mock entity_extractor.extract_from_document returning one entity."""
    return MagicMock(
        return_value=(
            [Entity(name=entity_name, entity_type=entity_type,
                    aliases=aliases or [], confidence=0.9)],
            [],
            [],
        )
    )


def _patch_md(structure, doc_name="doc.md", description="test doc"):
    """Patch md_to_tree (async → auto AsyncMock) returning a fixed result."""
    return patch.object(
        client_mod,
        "md_to_tree",
        return_value={
            "doc_name": doc_name,
            "doc_description": description,
            "line_count": 5,
            "structure": structure,
        },
    )


# ===========================================================================
# (1) node_profiles table round-trip (DB accessors)
# ===========================================================================


class TestNodeProfilesTable:
    def test_upsert_get_roundtrip(self, db):
        """upsert + get returns entities/keywords/tags JSON correctly."""
        doc_id = db.insert_document("a.md", "/tmp/a.md")
        db.upsert_node_profiles(doc_id, [
            {"node_id": "n1",
             "entities": [{"name": "浴血值", "type": "concept"}],
             "tags": ["游戏机制"]},
            {"node_id": "n2", "entities": [], "tags": ["游戏机制"]},
        ])
        got = db.get_node_profiles(doc_id)
        assert len(got) == 2
        by_node = {p["node_id"]: p for p in got}
        assert by_node["n1"]["entities"] == [{"name": "浴血值", "type": "concept"}]
        assert by_node["n1"]["keywords"] == []
        assert by_node["n1"]["tags"] == ["游戏机制"]
        assert by_node["n2"]["entities"] == []

    def test_reindex_replaces_not_duplicates(self, db):
        """Re-indexing a doc must REPLACE its profiles (incl. stale nodes)."""
        doc_id = db.insert_document("a.md", "/tmp/a.md")
        db.upsert_node_profiles(doc_id, [
            {"node_id": "n1", "entities": [{"name": "X", "type": "concept"}]},
            {"node_id": "n2", "entities": [{"name": "Y", "type": "concept"}]},
        ])
        db.upsert_node_profiles(doc_id, [
            {"node_id": "n3", "entities": [{"name": "Z", "type": "concept"}]},
        ])
        got = db.get_node_profiles(doc_id)
        assert [p["node_id"] for p in got] == ["n3"]

    def test_profiles_cascade_with_document(self, db):
        """Deleting a document removes its node_profiles rows (FK CASCADE)."""
        doc_id = db.insert_document("a.md", "/tmp/a.md")
        db.upsert_node_profiles(doc_id, [
            {"node_id": "n1", "entities": [{"name": "X", "type": "concept"}]},
        ])
        db.delete_document(doc_id)
        assert db.get_node_profiles(doc_id) == []

    def test_get_profiles_empty_doc(self, db):
        doc_id = db.insert_document("a.md", "/tmp/a.md")
        assert db.get_node_profiles(doc_id) == []


# ===========================================================================
# (2) Deterministic attribution matcher (no LLM)
# ===========================================================================


class TestDeterministicMatcher:
    def test_match_by_text_case_insensitive_and_alias(self, client_factory):
        client = client_factory(workspace=False)
        try:
            structure = [
                {"node_id": "n1", "title": "T1", "text": "The ABC system"},
                {"node_id": "n2", "title": "T2", "text": "unrelated"},
                {"node_id": "n3", "title": "T3", "text": "uses abc-v2"},
            ]
            ids = client._match_nodes_for_entity("ABC", ["abc-v2"], structure)
            assert ids == ["n1", "n3"]
        finally:
            client.close()

    def test_match_title_counts(self, client_factory):
        client = client_factory(workspace=False)
        try:
            structure = [{"node_id": "n1", "title": "浴血值详解", "text": "nothing here"}]
            assert client._match_nodes_for_entity("浴血值", [], structure) == ["n1"]
        finally:
            client.close()

    def test_match_summary_fallback_when_no_text(self, client_factory):
        """PDF-style node without 'text' falls back to summary."""
        client = client_factory(workspace=False)
        try:
            structure = [
                {"node_id": "n1", "title": "T", "summary": "提到浴血值"},
                {"node_id": "n2", "title": "T2", "text": "with 浴血值", "summary": "no"},
            ]
            ids = client._match_nodes_for_entity("浴血值", [], structure)
            assert ids == ["n1", "n2"]
        finally:
            client.close()

    def test_match_cap_at_20_nodes(self, client_factory):
        client = client_factory(workspace=False)
        try:
            structure = [
                {"node_id": f"n{i}", "title": f"T{i}", "text": "浴血值"}
                for i in range(25)
            ]
            ids = client._match_nodes_for_entity("浴血值", [], structure)
            assert ids == [f"n{i}" for i in range(20)]
        finally:
            client.close()

    def test_no_match_returns_empty(self, client_factory):
        client = client_factory(workspace=False)
        try:
            structure = [{"node_id": "n1", "title": "T", "text": "nothing"}]
            assert client._match_nodes_for_entity("浴血值", ["别名"], structure) == []
        finally:
            client.close()


# ===========================================================================
# (3) Single-doc index flow: attribution + profiles + structure JSON
# ===========================================================================


def _prepare_client_for_index(client):
    """Mock every LLM-touching component except entity logic under test."""
    client.super_tree_index.on_document_added = MagicMock()
    client.closet_index.add_document = MagicMock()
    client.corpus_tree.update_for_document = MagicMock()
    client.search_backend.index_document = MagicMock()


class TestSingleDocAttribution:
    def test_mentions_attributed_to_matched_node_only(self, client_factory, tmp_path):
        """Entity text only in node A → mention row for node A (node_id set),
        none for node B, and NO doc-level (NULL) row when matches exist."""
        client = client_factory()
        try:
            _prepare_client_for_index(client)
            client.entity_extractor.extract_from_document = _mock_extract_single()

            md_path = tmp_path / "doc.md"
            md_path.write_text("# Test\n\ncontent\n", encoding="utf-8")
            with _patch_md(_two_node_structure()):
                client.index(str(md_path), mode="md")

            db_doc = client.db.get_document_by_name("doc.md")
            mentions = client.db.get_entity_mentions_by_doc(db_doc["id"])
            assert len(mentions) == 1
            assert mentions[0]["node_id"] == "node-a"
            assert mentions[0]["entity_name"] == "浴血值"
        finally:
            client.close()

    def test_unmatched_entity_keeps_doc_level_mention(self, client_factory, tmp_path):
        """Entity appearing in no node text → single doc-level row (node_id NULL)."""
        client = client_factory()
        try:
            _prepare_client_for_index(client)
            client.entity_extractor.extract_from_document = _mock_extract_single(
                entity_name="完全不相关的词")

            md_path = tmp_path / "doc.md"
            md_path.write_text("# Test\n\ncontent\n", encoding="utf-8")
            with _patch_md(_two_node_structure()):
                client.index(str(md_path), mode="md")

            db_doc = client.db.get_document_by_name("doc.md")
            mentions = client.db.get_entity_mentions_by_doc(db_doc["id"])
            assert len(mentions) == 1
            assert mentions[0]["node_id"] is None
        finally:
            client.close()

    def test_node_profiles_written_to_table(self, client_factory, tmp_path):
        """node_profiles table holds the matched node's canonical entity."""
        client = client_factory()
        try:
            _prepare_client_for_index(client)
            client.entity_extractor.extract_from_document = _mock_extract_single()

            md_path = tmp_path / "doc.md"
            md_path.write_text("# Test\n\ncontent\n", encoding="utf-8")
            with _patch_md(_two_node_structure()):
                client.index(str(md_path), mode="md")

            db_doc = client.db.get_document_by_name("doc.md")
            profiles = client.db.get_node_profiles(db_doc["id"])
            by_node = {p["node_id"]: p for p in profiles}
            assert "node-a" in by_node
            assert {"name": "浴血值", "type": "concept"} in by_node["node-a"]["entities"]
            # node B has no entities → either absent or empty
            assert by_node.get("node-b", {}).get("entities", []) == []
        finally:
            client.close()

    def test_profiles_tags_reuse_doc_closet_tags(self, client_factory, tmp_path):
        """Profile tags reuse the doc-level closet_tags."""
        client = client_factory()
        try:
            _prepare_client_for_index(client)

            def fake_add_document(db_doc_id, doc_name, doc_description, structure):
                client.db.insert_closet_tags(db_doc_id, [
                    (db_doc_id, "游戏机制", "游戏", 0.9, "llm"),
                    (db_doc_id, "数值设计", "数值", 0.8, "llm"),
                ])

            client.closet_index.add_document = MagicMock(side_effect=fake_add_document)
            client.entity_extractor.extract_from_document = _mock_extract_single()

            md_path = tmp_path / "doc.md"
            md_path.write_text("# Test\n\ncontent\n", encoding="utf-8")
            with _patch_md(_two_node_structure()):
                client.index(str(md_path), mode="md")

            db_doc = client.db.get_document_by_name("doc.md")
            profiles = client.db.get_node_profiles(db_doc["id"])
            by_node = {p["node_id"]: p for p in profiles}
            assert by_node["node-a"]["tags"] == ["游戏机制", "数值设计"]
        finally:
            client.close()

    def test_reindex_replaces_profiles(self, client_factory, tmp_path):
        """Indexing the same doc twice must replace profiles, not duplicate."""
        client = client_factory()
        try:
            _prepare_client_for_index(client)
            client.entity_extractor.extract_from_document = _mock_extract_single()

            md_path = tmp_path / "doc.md"
            md_path.write_text("# Test\n\ncontent\n", encoding="utf-8")
            with _patch_md(_two_node_structure()):
                client.index(str(md_path), mode="md")
            with _patch_md(_two_node_structure()):
                client.index(str(md_path), mode="md")

            db_doc = client.db.get_document_by_name("doc.md")
            profiles = client.db.get_node_profiles(db_doc["id"])
            node_a_rows = [p for p in profiles if p["node_id"] == "node-a"]
            assert len(node_a_rows) == 1
            mentions = client.db.get_entity_mentions_by_doc(db_doc["id"])
            assert len([m for m in mentions if m["node_id"] == "node-a"]) == 1
        finally:
            client.close()


class TestStructureJsonEntities:
    def test_workspace_structure_contains_entities_key(self, client_factory, tmp_path):
        """After single-doc index, doc['structure'] node dicts carry 'entities'."""
        client = client_factory()
        try:
            _prepare_client_for_index(client)
            client.entity_extractor.extract_from_document = _mock_extract_single()

            md_path = tmp_path / "doc.md"
            md_path.write_text("# Test\n\ncontent\n", encoding="utf-8")
            with _patch_md(_two_node_structure()):
                doc_uuid = client.index(str(md_path), mode="md")

            data = json.loads((client.workspace / f"{doc_uuid}.json").read_text())
            nodes = {n["node_id"]: n for n in data["structure"]}
            assert {"name": "浴血值", "type": "concept"} in nodes["node-a"]["entities"]
            assert nodes["node-b"]["entities"] == []
        finally:
            client.close()


# ===========================================================================
# (4) Canonical entity references in profiles
# ===========================================================================


class TestCanonicalProfiles:
    def test_profile_shows_canonical_after_merge(self, client_factory):
        """After merge_entities(小张→张三), rebuilt profiles reference 张三."""
        client = client_factory(workspace=False)
        try:
            db = client.db
            doc_id = db.insert_document("doc1", "/tmp/doc1.pdf")
            zs = db.insert_entity("person", "张三")
            xz = db.insert_entity("person", "小张")
            db.insert_entity_mention(xz, doc_id, node_id="n1", confidence=0.9)

            client._write_node_profiles(doc_id)
            ents = db.get_node_profiles(doc_id)[0]["entities"]
            assert ents == [{"name": "小张", "type": "person"}]

            db.merge_entities(zs, xz)
            client._write_node_profiles(doc_id)
            profiles = db.get_node_profiles(doc_id)
            assert [e["name"] for e in profiles[0]["entities"]] == ["张三"]
        finally:
            client.close()

    def test_profiles_from_merge_canonicalize_all_mentions(self, client_factory):
        """Two synonym entities on different nodes → both profiles canonical."""
        client = client_factory(workspace=False)
        try:
            db = client.db
            doc_id = db.insert_document("doc1", "/tmp/doc1.pdf")
            zs = db.insert_entity("person", "张三")
            xz = db.insert_entity("person", "小张")
            db.insert_entity_mention(zs, doc_id, node_id="n1", confidence=0.9)
            db.insert_entity_mention(xz, doc_id, node_id="n2", confidence=0.8)

            db.merge_entities(zs, xz)
            client._write_node_profiles(doc_id)
            profiles = {p["node_id"]: p for p in db.get_node_profiles(doc_id)}
            assert [e["name"] for e in profiles["n1"]["entities"]] == ["张三"]
            assert [e["name"] for e in profiles["n2"]["entities"]] == ["张三"]
        finally:
            client.close()


# ===========================================================================
# (5) Batch path: profiles written AFTER normalization → canonical names
# ===========================================================================


class TestBatchProfiles:
    def test_batch_profiles_canonical_after_normalization(
        self, client_factory, tmp_path
    ):
        """index_batch: doc1 extracts 小张, doc2 extracts 张先生; batch
        normalization merges 张先生→小张; node_profiles must show canonical 小张
        on BOTH docs, and mentions carry node_id."""
        client = client_factory(workspace=False)
        try:
            client.super_tree_index.on_document_added = MagicMock()
            client.closet_index.add_document = MagicMock()
            client.corpus_tree.rebuild = MagicMock(return_value={})
            client.search_backend.index_document = MagicMock()

            def fake_extract(doc_name, doc_description, structure):
                if "doc1" in doc_name:
                    return ([Entity(name="小张", entity_type="person",
                                    aliases=[], confidence=0.9)], [], [])
                return ([Entity(name="张先生", entity_type="person",
                                aliases=[], confidence=0.85)], [], [])

            client.entity_extractor.extract_from_document = MagicMock(
                side_effect=fake_extract)

            def fake_md_to_tree(md_path=None, **kwargs):
                if "doc1" in (md_path or ""):
                    return {
                        "doc_name": "doc1.md",
                        "doc_description": "d1",
                        "line_count": 3,
                        "structure": [{
                            "node_id": "d1-n1", "title": "人物",
                            "text": "小张负责风控", "summary": "小张", "level": 1,
                        }],
                    }
                return {
                    "doc_name": "doc2.md",
                    "doc_description": "d2",
                    "line_count": 3,
                    "structure": [{
                        "node_id": "d2-n1", "title": "人物",
                        "text": "张先生负责风控", "summary": "张先生", "level": 1,
                    }],
                }

            # Mock the batch-normalization LLM: merge 张先生 into 小张
            norm_response = json.dumps({
                "groups": [{"canonical": "小张", "synonyms": ["小张", "张先生"]}]
            })

            paths = []
            for name in ("doc1.md", "doc2.md"):
                p = tmp_path / name
                p.write_text(f"# {name}\n\ncontent\n", encoding="utf-8")
                paths.append(str(p))

            with patch.object(client_mod, "md_to_tree",
                              side_effect=fake_md_to_tree), \
                 patch.object(entity_extractor_mod, "llm_completion",
                              return_value=norm_response):
                client.index_batch(paths, mode="md")

            db = client.db
            d1 = db.get_document_by_name("doc1.md")
            d2 = db.get_document_by_name("doc2.md")

            # Mentions are node-attributed
            m1 = db.get_entity_mentions_by_doc(d1["id"])
            m2 = db.get_entity_mentions_by_doc(d2["id"])
            assert [m["node_id"] for m in m1] == ["d1-n1"]
            assert [m["node_id"] for m in m2] == ["d2-n1"]

            # Profiles reference the CANONICAL entity on both docs
            p1 = {p["node_id"]: p for p in db.get_node_profiles(d1["id"])}
            p2 = {p["node_id"]: p for p in db.get_node_profiles(d2["id"])}
            assert {"name": "小张", "type": "person"} in p1["d1-n1"]["entities"]
            assert {"name": "小张", "type": "person"} in p2["d2-n1"]["entities"]
            assert all(e["name"] != "张先生"
                       for p in (p1, p2) for prof in p.values()
                       for e in prof["entities"])
        finally:
            client.close()


# ===========================================================================
# (6) Node-level keywords: TF-IDF top-K, no LLM (P1.4)
# ===========================================================================


def _kw_helper(structure, topk=5):
    """Access the pure keyword-computation helper via the real module ref."""
    return client_mod._compute_node_keywords(structure, topk)


class TestKeywordComputation:
    """Pure TF-IDF computation (no DB, no client, no LLM)."""

    def test_stopwords_and_short_tokens_excluded(self):
        """Stopwords, len<2 tokens and single-char CJK noise never surface."""
        st = [{"node_id": "n1", "title": "的",
               "text": "一个 没有 the a is 我 有 值"}]
        assert _kw_helper(st) == {"n1": []}

    def test_empty_or_textless_node_empty_keywords(self):
        st = [
            {"node_id": "n1", "title": "", "text": ""},
            {"node_id": "n2"},  # PDF-ish: no title/text/summary at all
        ]
        assert _kw_helper(st) == {"n1": [], "n2": []}

    def test_summary_fallback_when_no_text(self):
        """Same convention as entity attribution: text or fall back to summary."""
        st = [{"node_id": "n1", "title": "T", "summary": "浴血值机制 浴血值机制"}]
        assert "浴血" in _kw_helper(st)["n1"]

    def test_topk_cap_respected(self):
        st = [{"node_id": "n1", "title": "",
               "text": "攻击力 防御力 魔法值 暴击率 闪避率 命中率 抗性 治疗量"}]
        kw = _kw_helper(st, topk=3)
        assert len(kw["n1"]) == 3
        assert len(_kw_helper(st, topk=0)["n1"]) == 0

    def test_idf_universal_term_below_rare_term(self):
        """N=3 controlled fixture: 系统 in EVERY node (df=N, lowest idf),
        引擎 only in node 1 (df=1). Equal tf → rare term must rank first."""
        st = [
            {"node_id": "n1", "text": "系统 引擎"},
            {"node_id": "n2", "text": "系统 网络"},
            {"node_id": "n3", "text": "系统 存储"},
        ]
        kw = _kw_helper(st, topk=1)
        assert kw["n1"] == ["引擎"]

    def test_empty_structure(self):
        assert _kw_helper(None) == {}
        assert _kw_helper([]) == {}

    def test_deterministic(self):
        st = _two_node_structure()
        assert _kw_helper(st) == _kw_helper(st)


class TestSingleDocKeywords:
    """Single-doc index flow: keywords in node_profiles table + structure JSON."""

    def test_salient_keyword_attributed_to_correct_node(self, client_factory, tmp_path):
        """浴血值 recurs in node A only → keyword present in A's profile,
        absent from B's. jieba splits 浴血值 → 浴血 + 值 (单字噪声被过滤)."""
        client = client_factory()
        try:
            _prepare_client_for_index(client)
            client.entity_extractor.extract_from_document = _mock_extract_single()

            md_path = tmp_path / "doc.md"
            md_path.write_text("# Test\n\ncontent\n", encoding="utf-8")
            with _patch_md(_two_node_structure()):
                client.index(str(md_path), mode="md")

            db_doc = client.db.get_document_by_name("doc.md")
            by_node = {p["node_id"]: p
                       for p in client.db.get_node_profiles(db_doc["id"])}
            assert "浴血" in by_node["node-a"]["keywords"]
            assert "浴血" not in by_node.get("node-b", {}).get("keywords", [])
            assert len(by_node["node-a"]["keywords"]) <= client._node_keyword_topk
            # entity signature (P1.2) still intact alongside keywords
            assert {"name": "浴血值", "type": "concept"} in by_node["node-a"]["entities"]
        finally:
            client.close()

    def test_reindex_idempotent_and_deterministic(self, client_factory, tmp_path):
        """Indexing the same doc twice: identical keyword sets, row count stable."""
        client = client_factory()
        try:
            _prepare_client_for_index(client)
            client.entity_extractor.extract_from_document = _mock_extract_single()

            md_path = tmp_path / "doc.md"
            md_path.write_text("# Test\n\ncontent\n", encoding="utf-8")
            with _patch_md(_two_node_structure()):
                client.index(str(md_path), mode="md")
            db_doc = client.db.get_document_by_name("doc.md")
            first = client.db.get_node_profiles(db_doc["id"])

            with _patch_md(_two_node_structure()):
                client.index(str(md_path), mode="md")
            second = client.db.get_node_profiles(db_doc["id"])

            assert len(second) == len(first)
            assert first == second  # ORDER BY node_id → comparable as-is
        finally:
            client.close()

    def test_topk_override_shrinks_keywords(self, client_factory, tmp_path):
        """topk=1 → node A's single keyword is 浴血 (tf=3/10 dominates)."""
        client = client_factory()
        try:
            _prepare_client_for_index(client)
            client.entity_extractor.extract_from_document = _mock_extract_single()
            client._node_keyword_topk = 1

            md_path = tmp_path / "doc.md"
            md_path.write_text("# Test\n\ncontent\n", encoding="utf-8")
            with _patch_md(_two_node_structure()):
                client.index(str(md_path), mode="md")

            db_doc = client.db.get_document_by_name("doc.md")
            profiles = client.db.get_node_profiles(db_doc["id"])
            assert all(len(p["keywords"]) <= 1 for p in profiles)
            by_node = {p["node_id"]: p for p in profiles}
            assert by_node["node-a"]["keywords"] == ["浴血"]
        finally:
            client.close()

    def test_default_topk_from_config(self, client_factory):
        """config.yaml ships node_keyword_topk: 5 and the client picks it up."""
        client = client_factory(workspace=False)
        try:
            assert client._node_keyword_topk == 5
        finally:
            client.close()

    def test_structure_json_carries_keywords(self, client_factory, tmp_path):
        """Sync path: doc['structure'] node dicts carry the 'keywords' key."""
        client = client_factory()
        try:
            _prepare_client_for_index(client)
            client.entity_extractor.extract_from_document = _mock_extract_single()

            md_path = tmp_path / "doc.md"
            md_path.write_text("# Test\n\ncontent\n", encoding="utf-8")
            with _patch_md(_two_node_structure()):
                doc_uuid = client.index(str(md_path), mode="md")

            data = json.loads((client.workspace / f"{doc_uuid}.json").read_text())
            nodes = {n["node_id"]: n for n in data["structure"]}
            assert "keywords" in nodes["node-a"]
            assert "浴血" in nodes["node-a"]["keywords"]
            assert "浴血" not in nodes["node-b"]["keywords"]
        finally:
            client.close()
