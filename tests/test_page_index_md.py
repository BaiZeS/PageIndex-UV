import importlib.util
import sys
from pathlib import Path

# Preload pageindex_mutil.utils minimal stub so page_index_md's imports resolve.
_mutil = Path(__file__).parent.parent / "pageindex_mutil"
sys.path.insert(0, str(_mutil))

utils_spec = importlib.util.spec_from_file_location("pageindex_mutil.utils", _mutil / "utils.py")
utils_mod = importlib.util.module_from_spec(utils_spec)
sys.modules["pageindex_mutil.utils"] = utils_mod
utils_mod.count_tokens = lambda text, model=None: len(text or "") // 4
# page_index_md 模块级 import 需要这些名字；仅 stub 掉真实实现，避免拉 heavy deps。
utils_mod.structure_to_list = lambda s: []
utils_mod.write_node_id = lambda *a, **k: None
utils_mod.format_structure = lambda s, **k: s
utils_mod.create_clean_structure_for_description = lambda s: s
utils_mod.ConfigLoader = lambda *a, **k: None
utils_mod.print_json = lambda *a, **k: None
utils_mod.print_toc = lambda *a, **k: None
utils_mod.generate_node_summary = lambda *a, **k: None
utils_mod.generate_doc_description = lambda *a, **k: None

spec = importlib.util.spec_from_file_location("pageindex_mutil.page_index_md", _mutil / "page_index_md.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["pageindex_mutil.page_index_md"] = mod
spec.loader.exec_module(mod)

extract_nodes_from_markdown = mod.extract_nodes_from_markdown


def test_headers_still_detected():
    md = "# Chapter One\ncontent\n## Section A\nmore content\n"
    nodes, _ = extract_nodes_from_markdown(md)
    titles = [n["node_title"] for n in nodes]
    assert "Chapter One" in titles
    assert "Section A" in titles


def test_bold_standalone_line_as_heading():
    """阶段2 -- 无 # 结构时，独立粗体行应被识别为 level 标题（对齐原版）。"""
    md = "**Section One**\ncontent here\n**Section Two**\nmore content\n"
    nodes, _ = extract_nodes_from_markdown(md)
    titles = [n["node_title"] for n in nodes]
    assert titles == ["Section One", "Section Two"]


def test_bold_in_paragraph_ignored():
    """阶段2 -- 行内粗体不应误判为标题。"""
    md = "This is **emphasized** text inline, not a heading.\n"
    nodes, _ = extract_nodes_from_markdown(md)
    assert nodes == []