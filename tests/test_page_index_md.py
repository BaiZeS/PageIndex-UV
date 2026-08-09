import importlib.util
import json
import sys
from pathlib import Path

import pytest

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
utils_mod.llm_completion = lambda *a, **k: None
utils_mod.extract_json = lambda text: json.loads(text) if text else None

spec = importlib.util.spec_from_file_location("pageindex_mutil.page_index_md", _mutil / "page_index_md.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["pageindex_mutil.page_index_md"] = mod
spec.loader.exec_module(mod)

extract_nodes_from_markdown = mod.extract_nodes_from_markdown
_normalize_line_breaks = mod._normalize_line_breaks


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


def test_semantic_sections_empty_text():
    """阶段2 -- 空文本返回空列表，不调 LLM。"""
    assert mod.semantic_sections_from_markdown("") == []
    assert mod.semantic_sections_from_markdown("   \n\n  ") == []


def test_semantic_sections_parses_llm_output():
    """阶段2 -- LLM 返回章节 JSON 时解析为 {title, line_num} 列表。"""
    from unittest.mock import patch
    mock_out = json.dumps([
        {"title": "引言", "line_num": 1},
        {"title": "方法", "line_num": 5},
        {"title": "结论", "line_num": 20},
    ])
    with patch.object(mod, "llm_completion", return_value=mock_out) as m, \
         patch.object(mod, "extract_json", side_effect=json.loads):
        secs = mod.semantic_sections_from_markdown("无标题文本\n" * 30)
        assert secs == [
            {"title": "引言", "line_num": 1},
            {"title": "方法", "line_num": 5},
            {"title": "结论", "line_num": 20},
        ]
        m.assert_called_once()


def test_semantic_sections_llm_failure_returns_empty():
    """阶段2 -- LLM 失败/返回非法时返回空列表（不崩溃）。"""
    from unittest.mock import patch
    with patch.object(mod, "llm_completion", return_value="not json"):
        assert mod.semantic_sections_from_markdown("some text") == []


@pytest.mark.asyncio
async def test_md_to_tree_falls_back_to_semantic_sections(tmp_path):
    """阶段2 -- 无标题(空树) MD 走语义切分分支，生成非空树。"""
    from unittest.mock import patch
    md = tmp_path / "notes.md"
    md.write_text("第一段内容\n更多内容\n\n第二段内容\n结束内容\n", encoding="utf-8")

    async def _noop(structure, **kwargs):
        return structure

    with patch.object(mod, "semantic_sections_from_markdown",
                      return_value=[{"title": "第一章", "line_num": 1},
                                    {"title": "第二章", "line_num": 4}]) as m, \
         patch.object(mod, "generate_summaries_for_structure_md", new=_noop):
        result = await mod.md_to_tree(
            str(md), if_add_node_summary='yes', summary_token_threshold=200,
            if_add_node_id='yes', model=None,
        )
        m.assert_called_once()
        titles = [n.get("title") for n in result["structure"]]
        assert titles == ["第一章", "第二章"]


@pytest.mark.asyncio
async def test_summaries_concurrency_limited():
    """阶段4 -- 摘要生成用信号量限流，避免并发风暴；结果仍正确汇总。"""
    from unittest.mock import patch
    import asyncio

    structure = [
        {"title": chr(65 + i), "text": "x" * 2000, "nodes": []}
        for i in range(6)
    ]
    active = 0
    max_active = 0

    async def fake_summary(node, model=None):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return f"summary:{node['title']}"

    with patch.object(mod, "generate_node_summary", new=fake_summary), \
             patch.object(mod, "structure_to_list", side_effect=lambda s: (
                 [s] if isinstance(s, dict) else s
             )):
        await mod.generate_summaries_for_structure_md(
            structure, summary_token_threshold=200, model=None
        )
    # 并发被限制在信号量上限内（当前实现无限制会并发到 6）
    assert max_active <= 2
    # 结果仍正确
    assert all(n["summary"] == f"summary:{n['title']}" for n in structure)


# ---------------------------------------------------------------------------
# _normalize_line_breaks tests
# ---------------------------------------------------------------------------

def test_normalize_single_line_chinese_sentences():
    """单行中文按句号拆分为多行。"""
    result = _normalize_line_breaks("段落一。段落二。段落三。")
    lines = [ln for ln in result.split('\n') if ln.strip()]
    assert len(lines) == 3
    assert lines[0] == "段落一。"
    assert lines[1] == "段落二。"
    assert lines[2] == "段落三。"


def test_normalize_single_line_english_sentences():
    """单行英文按句号拆分为多行。"""
    result = _normalize_line_breaks("First sentence. Second sentence. Third sentence.")
    lines = [ln for ln in result.split('\n') if ln.strip()]
    assert len(lines) == 3


def test_normalize_preserves_well_structured():
    """已有 >=3 非空行的 markdown 不做处理。"""
    md = "# A\ncontent\n# B\nmore\n# C\nend\n"
    assert _normalize_line_breaks(md) == md


def test_normalize_two_line_content():
    """只有 2 行的内容也会被归一化。"""
    result = _normalize_line_breaks("第一段。第二段。")
    lines = [ln for ln in result.split('\n') if ln.strip()]
    assert len(lines) >= 2


def test_normalize_list_markers():
    """单行内嵌列表标记被拆分。"""
    result = _normalize_line_breaks("前置内容 - 项目一 - 项目二 - 项目三")
    lines = [ln for ln in result.split('\n') if ln.strip()]
    assert len(lines) >= 2


@pytest.mark.asyncio
async def test_single_line_md_gets_nonempty_node_text(tmp_path):
    """单行无换行 markdown 经过 md_to_tree 后节点 text 不为空。"""
    from unittest.mock import patch
    md = tmp_path / "single.md"
    md.write_text("这是第一段内容。这是第二段内容。这是第三段内容。", encoding="utf-8")

    async def _noop(structure, **kwargs):
        return structure

    with patch.object(mod, "semantic_sections_from_markdown",
                      return_value=[{"title": "章节一", "line_num": 1},
                                    {"title": "章节二", "line_num": 2},
                                    {"title": "章节三", "line_num": 3}]) as m, \
         patch.object(mod, "generate_summaries_for_structure_md", new=_noop):
        result = await mod.md_to_tree(
            str(md), if_add_node_summary='yes', summary_token_threshold=200,
            if_add_node_id='yes', model=None,
        )
        m.assert_called_once()
        # Every node should have non-empty text
        for node in result["structure"]:
            assert node.get("text", "").strip(), f"Node '{node['title']}' has empty text"