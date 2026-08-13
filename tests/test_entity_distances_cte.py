"""P1 [S9] 图谱递归 CTE 测试 —— 单次 SQL 无向 BFS 替代 Python N+1 遍历。

验收覆盖：
1. 与 _precompute_entity_distances 逐项等价：hop-min 优先、距离衰减
   (0/1/2/3 -> 1.0/0.7/0.4/0.2) × 末边关系类型权重 (causal 1.0 /
   part_of 0.8 / related_to 0.6 / other 0.4)；
2. 权重语义 = 距离衰减(hop) × 末边关系权重，非路径乘积；
3. 自身（query entity）从结果中排除。

注：entity_relations.doc_id 有 FOREIGN KEY 指向 documents(id)，且
_connect() 开启 foreign_keys=ON。fixture 不插入 documents 行，故
entity_relations 的 INSERT 省略 doc_id（NULL），测试语义不受影响。
"""
import pytest

from db import PageIndexDB


@pytest.fixture()
def graph_db(tmp_path):
    db = PageIndexDB(str(tmp_path / "t.db"))
    with db._connect() as conn:
        conn.executemany(
            "INSERT INTO entities (id, name, entity_type, doc_count) VALUES (?,?,?,1)",
            [(1, "浴血值", "concept"), (2, "帮会系统", "concept"),
             (3, "门派介绍", "section"), (4, "远亲", "concept")],
        )
        # doc_id 省略（NULL）以避开 documents 外键约束；confidence 保留 0.9。
        conn.executemany(
            "INSERT INTO entity_relations (subject_id, predicate, object_id, confidence) VALUES (?,?,?,0.9)",
            [(1, "related_to", 2), (2, "part_of", 3), (3, "related_to", 4)],
        )
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


def test_cte_self_excluded(graph_db):
    got = graph_db.get_entity_distances_cte([1], max_hop=3)
    assert 1 not in got


def test_cte_matches_bfs_implementation(graph_db):
    """与 _precompute_entity_distances 的权重语义逐项对齐：
    直接引用 super_tree 的距离衰减/关系权重常量，手工重算期望值并断言。
    """
    from pageindex_mutil.super_tree import SuperTreeIndex

    decay = SuperTreeIndex._DISTANCE_DECAY
    rw = SuperTreeIndex._RELATION_TYPE_WEIGHTS

    got = graph_db.get_entity_distances_cte([1], max_hop=3)
    # 链：1 -related_to-> 2 -part_of-> 3 -related_to-> 4
    assert abs(got[2]["weight"] - decay[1] * rw["related_to"]) < 1e-6
    assert abs(got[3]["weight"] - decay[2] * rw["part_of"]) < 1e-6
    assert abs(got[4]["weight"] - decay[3] * rw["related_to"]) < 1e-6
    assert got[2]["distance"] == 1
    assert got[3]["distance"] == 2
    assert got[4]["distance"] == 3


def test_cte_tiebreak_same_hop_max_weight(graph_db):
    # 两个并行的 1-hop 边：causal(1.0) 与 related_to(0.6)——同 hop 取最大权重
    with graph_db._connect() as conn:
        conn.execute("INSERT INTO entities (id, name, entity_type, doc_count) VALUES (5,'甲','concept',1)")
        conn.execute("INSERT INTO entity_relations (subject_id, predicate, object_id, doc_id, confidence) VALUES (1,'causal',5,NULL,0.9)")
    got = graph_db.get_entity_distances_cte([1], max_hop=3)
    assert got[2]["distance"] == 1
    assert abs(got[2]["weight"] - 0.42) < 1e-6  # 0.7×related_to 0.6
    assert got[5]["distance"] == 1
    assert abs(got[5]["weight"] - 0.7) < 1e-6   # 0.7×causal 1.0


def test_cte_covers_all_predicates_and_defaults(graph_db):
    """全部谓词/别名/默认权重与 super_tree 常量程序化对照（防 SQL 内硬编码漂移）。"""
    from pageindex_mutil.super_tree import SuperTreeIndex
    decay = SuperTreeIndex._DISTANCE_DECAY
    rel_w = SuperTreeIndex._RELATION_TYPE_WEIGHTS
    cases = [
        ("causal", rel_w["causal"]), ("causes", rel_w["causes"]), ("effect", rel_w["effect"]),
        ("part_of", rel_w["part_of"]), ("contains", rel_w["contains"]),
        ("has_part", rel_w["has_part"]), ("belongs_to", rel_w["belongs_to"]),
        ("related_to", rel_w["related_to"]), ("associated", rel_w["associated"]),
        ("similar", rel_w["similar"]), ("unknown_pred", rel_w["_default"]),
    ]
    base = 100
    with graph_db._connect() as conn:
        for i, (pred, w) in enumerate(cases, 1):
            conn.execute("INSERT INTO entities (id, name, entity_type, doc_count) VALUES (?,?,?,1)",
                         (base + i, f"N{i}", "concept"))
            conn.execute("INSERT INTO entity_relations (subject_id, predicate, object_id, doc_id, confidence) VALUES (1,?,?,NULL,0.9)",
                         (pred, base + i))
    got = graph_db.get_entity_distances_cte([1], max_hop=3)
    for i, (pred, w) in enumerate(cases, 1):
        expected = decay[1] * w  # hop 1 衰减 0.7
        assert abs(got[base + i]["weight"] - expected) < 1e-6, f"predicate {pred}"
