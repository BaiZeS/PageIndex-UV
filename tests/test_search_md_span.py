"""T7 统一 span locator 测试（spec [S3]/[S4]）。

覆盖：
1. extract_node_text_content 落存 span_kind="line" 与 end_line（1-based 切片上界）；
2. spans_from_nodes 按 span_kind 分派：page→页集合；line→(node_id, 起, 止) 行区间。

全部纯函数，无 LLM 调用。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# T32.1：防御性 purge 退役——桩创建者已迁移真实 import，purge 反成新污染源。
from pageindex_mutil.page_index_md import extract_node_text_content
from pageindex_mutil.reasoning import spans_from_nodes


def test_extract_node_text_stores_line_span():
    node_list = [{"node_title": "第一章", "line_num": 1},
                 {"node_title": "第二章", "line_num": 3}]
    lines = ["# 第一章", "内容A", "## 第二章", "内容B"]
    nodes = extract_node_text_content(node_list, lines)
    assert nodes[0]["span_kind"] == "line"
    assert nodes[0]["end_line"] == 2           # 切片上界（下节行号-1，1-based）
    assert nodes[0]["text"] == "# 第一章\n内容A"  # 预切片 text 不受影响
    assert nodes[1]["end_line"] == 4           # 末节点 → len(lines)


def test_spans_from_nodes_dispatches_kinds():
    nodes = [
        {"node_id": "a", "span_kind": "page", "start_index": 2, "end_index": 3},
        {"node_id": "b", "span_kind": "line", "line_num": 4, "end_line": 6},
    ]
    got = spans_from_nodes(nodes)
    assert set(got["pages"]) == {2, 3}
    assert got["lines"] == [("b", 4, 6)]
