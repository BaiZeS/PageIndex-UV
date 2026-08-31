import importlib.util
import json
import sys
from pathlib import Path

import pytest

# T32.1 真实 import（包 __init__ PEP 562 惰性化）：utils 桩 + page_index_md
# spec 加载退役——被测纯函数（extract_nodes_from_markdown 等）不依赖桩行为。
import pageindex_mutil.page_index_md as mod

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
        # T32.1：历史 utils 桩的 identity format_structure 掩盖了缺
        # if_add_node_text='yes'——真实实现按 'no' 剥 text（生产链 client.index
        # 硬编码 'yes'）。补齐参数以匹配被测语义。
        result = await mod.md_to_tree(
            str(md), if_add_node_summary='yes', summary_token_threshold=200,
            if_add_node_id='yes', if_add_node_text='yes', model=None,
        )
        m.assert_called_once()
        titles = [n.get("title") for n in result["structure"]]
        assert titles == ["第一章", "第二章"]


@pytest.mark.asyncio
async def test_summaries_batch_mode():
    """摘要生成使用批量调用，减少 LLM round-trip。"""
    from unittest.mock import patch, AsyncMock

    structure = [
        {"title": chr(65 + i), "text": "x" * 2000, "nodes": []}
        for i in range(6)
    ]
    batch_call_count = 0

    async def fake_batch(nodes_with_text, model=None):
        nonlocal batch_call_count
        batch_call_count += 1
        return [f"batch:{t}" for _, t, _ in nodes_with_text]

    with patch.object(mod, "_batch_generate_summaries", side_effect=fake_batch), \
         patch.object(mod, "structure_to_list", side_effect=lambda s: (
             [s] if isinstance(s, dict) else s
         )):
        await mod.generate_summaries_for_structure_md(
            structure, summary_token_threshold=200, model=None
        )
    # 6 nodes / batch_size=3 = 2 batch calls (not 6 individual calls)
    assert batch_call_count == 2
    assert all(n["summary"] == f"batch:{n['title']}" for n in structure)


@pytest.mark.asyncio
async def test_summaries_batch_fallback_on_failure():
    """批量调用失败时回退到单节点调用。"""
    from unittest.mock import patch

    structure = [
        {"title": "A", "text": "x" * 2000, "nodes": []},
        {"title": "B", "text": "x" * 2000, "nodes": []},
    ]
    fallback_count = 0

    async def fake_batch(*a, **k):
        return None  # simulate failure

    async def fake_summary(node, model=None):
        nonlocal fallback_count
        fallback_count += 1
        return f"single:{node['title']}"

    with patch.object(mod, "_batch_generate_summaries", side_effect=fake_batch), \
         patch.object(mod, "generate_node_summary", side_effect=fake_summary), \
         patch.object(mod, "structure_to_list", side_effect=lambda s: (
             [s] if isinstance(s, dict) else s
         )):
        await mod.generate_summaries_for_structure_md(
            structure, summary_token_threshold=200, model=None
        )
    # Fallback triggered for both nodes
    assert fallback_count == 2
    assert all(n["summary"] == f"single:{n['title']}" for n in structure)


@pytest.mark.asyncio
async def test_summaries_short_text_no_llm():
    """短文本节点直接使用 text 作为摘要，不调 LLM。"""
    from unittest.mock import patch

    structure = [
        {"title": "A", "text": "short", "nodes": []},  # < threshold
    ]

    async def fake_batch(*a, **k):
        raise AssertionError("Should not be called for short text")

    with patch.object(mod, "_batch_generate_summaries", side_effect=fake_batch), \
         patch.object(mod, "structure_to_list", side_effect=lambda s: (
             [s] if isinstance(s, dict) else s
         )):
        await mod.generate_summaries_for_structure_md(
            structure, summary_token_threshold=200, model=None
        )
    assert structure[0]["summary"] == "short"


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
        # T32.1：历史 utils 桩的 identity format_structure 掩盖了缺
        # if_add_node_text='yes'——真实实现按 'no' 剥 text（生产链 client.index
        # 硬编码 'yes'）。补齐参数以匹配被测语义。
        result = await mod.md_to_tree(
            str(md), if_add_node_summary='yes', summary_token_threshold=200,
            if_add_node_id='yes', if_add_node_text='yes', model=None,
        )
        m.assert_called_once()
        # Every node should have non-empty text
        for node in result["structure"]:
            assert node.get("text", "").strip(), f"Node '{node['title']}' has empty text"