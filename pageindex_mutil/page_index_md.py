import asyncio
import json
import re
import os
from .utils import (
    count_tokens,
    structure_to_list,
    write_node_id,
    format_structure,
    create_clean_structure_for_description,
    ConfigLoader,
    print_json,
    print_toc,
    generate_node_summary,
    generate_doc_description,
    llm_completion,
    llm_acompletion,
    extract_json,
)

# Batch size for summary generation — how many nodes per LLM call.
_BATCH_SIZE = 3


async def _batch_generate_summaries(nodes_with_text, model=None):
    """Generate summaries for a batch of nodes in a single LLM call.

    Each entry in *nodes_with_text* is ``(index, title, text)``.
    Returns a list of summary strings aligned with the input order.
    Falls back to per-node calls on parse failure.
    """
    sections = []
    for idx, title, text in nodes_with_text:
        sections.append(f"[{idx}] {title}\n{text}")
    prompt = (
        "你是一个文档摘要专家。请为以下每个章节生成摘要，概括该章节涵盖的主要内容。"
        "摘要应完整反映章节的核心信息，包括重要的名称、术语和关键概念。\n\n"
        + "\n\n".join(sections)
        + "\n\n请以 JSON 数组格式返回，每个元素对应一个章节的摘要，顺序保持一致。\n"
        '格式：["摘要1", "摘要2", ...]\n'
        "直接返回 JSON 数组，不要输出其他内容。"
    )
    response = await llm_acompletion(model, prompt)
    if not response:
        return None
    result = extract_json(response)
    if isinstance(result, list) and len(result) == len(nodes_with_text):
        return [str(s) for s in result]
    return None


async def generate_summaries_for_structure_md(structure, summary_token_threshold, model=None):
    nodes = structure_to_list(structure)

    # Phase 1: short nodes use text directly (no LLM call needed).
    results = [None] * len(nodes)
    long_nodes = []  # (original_index, title, text)
    for i, node in enumerate(nodes):
        node_text = node.get('text', '')
        if count_tokens(node_text, model=model) < summary_token_threshold:
            results[i] = node_text
        else:
            long_nodes.append((i, node.get('title', ''), node_text))

    # Phase 2: batch long nodes into groups of _BATCH_SIZE.
    if long_nodes:
        for batch_start in range(0, len(long_nodes), _BATCH_SIZE):
            batch = long_nodes[batch_start:batch_start + _BATCH_SIZE]
            batch_summaries = await _batch_generate_summaries(batch, model=model)
            if batch_summaries:
                for (orig_idx, _, _), summary in zip(batch, batch_summaries):
                    results[orig_idx] = summary
            else:
                # Fallback: per-node calls for this batch
                for orig_idx, title, text in batch:
                    node_obj = {'title': title, 'text': text}
                    results[orig_idx] = await generate_node_summary(node_obj, model=model)

    # Phase 3: assign summaries to nodes.
    for node, summary in zip(nodes, results):
        if not node.get('nodes'):
            node['summary'] = summary
        else:
            node['prefix_summary'] = summary
    return structure


def semantic_sections_from_markdown(markdown_content, model=None):
    """阶段2：对无标题结构的 markdown 做 LLM 语义章节切分。

    返回 [{title, line_num}] 列表（按文档顺序），供 build_tree_from_nodes 建树。
    与 PageIndex 原版 process_no_toc 的 generate_toc_init 同构：无显式结构时，
    由 LLM 从正文提取语义章节边界，使短文档也长成树。失败返回空列表。
    """
    if not markdown_content or not markdown_content.strip():
        return []

    prompt = (
        "你是一个文档结构分析专家。给定一段没有标题的 markdown 文本，"
        "请将其切分为若干个语义连贯的章节，并给出每个章节的起始行号。\n\n"
        "文本：\n" + markdown_content[:8000] + "\n\n"
        "要求：\n"
        "1. 按语义主题切分（如 3-8 个章节），每章节给出简短标题。\n"
        "2. line_num 为该章节在文本中的起始行号（从 1 开始，按 \\n 分行）。\n"
        "3. 章节按文档顺序排列。\n\n"
        "返回JSON格式：\n"
        '[{"title": "章节标题", "line_num": 1}, ...]\n'
        "直接返回最终JSON数组，不要输出其他内容。"
    )
    try:
        response = llm_completion(model, prompt, thinking_disabled=True)
        if not response:
            return []
        data = extract_json(response)
        if not isinstance(data, list):
            return []
        sections = []
        for item in data:
            if isinstance(item, dict) and item.get("title"):
                try:
                    line_num = int(item.get("line_num", 1))
                except (TypeError, ValueError):
                    line_num = 1
                sections.append({"title": item["title"].strip(), "line_num": line_num})
        return sections
    except Exception:
        return []


def _normalize_line_breaks(markdown_content):
    """Split single-line or low-line content at sentence/paragraph boundaries.

    When markdown_content has very few lines (e.g. a single-line string with
    no ``\\n``), the ``line_num`` system has no granularity and downstream
    line-based slicing produces empty text.  This function normalizes such
    content by inserting ``\\n`` at sentence and paragraph boundaries so that
    ``extract_nodes_from_markdown`` and ``semantic_sections_from_markdown``
    can work with meaningful line numbers.

    Only activates when the content has fewer than 3 non-empty lines to avoid
    touching well-structured markdown.
    """
    non_empty = [ln for ln in markdown_content.split('\n') if ln.strip()]
    if len(non_empty) >= 3:
        return markdown_content

    text = markdown_content
    # Paragraph boundaries: two or more consecutive newlines (already multi-line,
    # but may be collapsed into one long line by the caller).
    text = re.sub(r'\n{2,}', '\n', text)

    # Split at Chinese/English sentence-ending punctuation.
    # Handles both "A。B" (no space) and "A. B" (with space).
    # Keeps the delimiter attached to the preceding sentence.
    text = re.sub(r'([。！？])(?=\S)', r'\1\n', text)
    text = re.sub(r'([.!?])(\s+)', r'\1\n', text)

    # Split at list markers that may be jammed together on one line.
    text = re.sub(r'(\S)(\s*[-*]\s+)', r'\1\n\2', text)
    text = re.sub(r'(\S)(\s*\d+[.、]\s+)', r'\1\n\2', text)

    # Split at markdown horizontal rules (---, ***, ___).
    text = re.sub(r'(\S)(\s*)([-*_]{3,})', r'\1\n\3', text)

    return text


def extract_nodes_from_markdown(markdown_content):
    header_pattern = r'^(#{1,6})\s+(.+)$'
    bold_heading_pattern = r'^\*\*(.+?)\*\*\s*$'
    code_block_pattern = r'^```'
    node_list = []

    lines = markdown_content.split('\n')
    in_code_block = False
    
    for line_num, line in enumerate(lines, 1):
        stripped_line = line.strip()
        
        # Check for code block delimiters (triple backticks)
        if re.match(code_block_pattern, stripped_line):
            in_code_block = not in_code_block
            continue
        
        # Skip empty lines
        if not stripped_line:
            continue
        
        # Only look for headers when not inside a code block
        if not in_code_block:
            match = re.match(header_pattern, stripped_line)
            if match:
                title = match.group(2).strip()
                node_list.append({'node_title': title, 'line_num': line_num})
                continue

            # 阶段2：无 # 结构时，独立粗体行视为一级标题（对齐原版 PageIndex）
            bold_match = re.match(bold_heading_pattern, stripped_line)
            if bold_match:
                title = bold_match.group(1).strip()
                if title:
                    node_list.append({'node_title': title, 'line_num': line_num})

    return node_list, lines


def extract_node_text_content(node_list, markdown_lines):    
    all_nodes = []
    for node in node_list:
        line_content = markdown_lines[node['line_num'] - 1]
        header_match = re.match(r'^(#{1,6})', line_content)
        
        if header_match is None:
            print(f"Warning: Line {node['line_num']} does not contain a valid header: '{line_content}'")
            continue
            
        processed_node = {
            'title': node['node_title'],
            'line_num': node['line_num'],
            'level': len(header_match.group(1)),
            'span_kind': 'line',
        }
        all_nodes.append(processed_node)
    
    for i, node in enumerate(all_nodes):
        start_line = node['line_num'] - 1 
        if i + 1 < len(all_nodes):
            end_line = all_nodes[i + 1]['line_num'] - 1 
        else:
            end_line = len(markdown_lines)
        
        # end_line 为切片上界（1-based 含末行）；解析期就地落存，零额外解析成本。
        node['end_line'] = end_line
        node['text'] = '\n'.join(markdown_lines[start_line:end_line]).strip()    
    return all_nodes


def _find_all_children(parent_index, parent_level, node_list):
    """Find all direct and indirect children of a parent node."""
    children_indices = []
    for i in range(parent_index + 1, len(node_list)):
        current_level = node_list[i]['level']
        if current_level <= parent_level:
            break
        children_indices.append(i)
    return children_indices


def update_node_list_with_text_token_count(node_list, model=None):

    # Make a copy to avoid modifying the original
    result_list = node_list.copy()
    
    # Process nodes from end to beginning to ensure children are processed before parents
    for i in range(len(result_list) - 1, -1, -1):
        current_node = result_list[i]
        current_level = current_node['level']
        
        # Get all children of this node
        children_indices = _find_all_children(i, current_level, result_list)
        
        # Start with the node's own text
        node_text = current_node.get('text', '')
        total_text = node_text
        
        # Add all children's text
        for child_index in children_indices:
            child_text = result_list[child_index].get('text', '')
            if child_text:
                total_text += '\n' + child_text
        
        # Calculate token count for combined text
        result_list[i]['text_token_count'] = count_tokens(total_text, model=model)
    
    return result_list


def tree_thinning_for_index(node_list, min_node_token=None, model=None):
    result_list = node_list.copy()
    nodes_to_remove = set()
    
    for i in range(len(result_list) - 1, -1, -1):
        if i in nodes_to_remove:
            continue
            
        current_node = result_list[i]
        current_level = current_node['level']
        
        total_tokens = current_node.get('text_token_count', 0)
        
        if total_tokens < min_node_token:
            children_indices = _find_all_children(i, current_level, result_list)
            
            children_texts = []
            for child_index in sorted(children_indices):
                if child_index not in nodes_to_remove:
                    child_text = result_list[child_index].get('text', '')
                    if child_text.strip():
                        children_texts.append(child_text)
                    nodes_to_remove.add(child_index)
            
            if children_texts:
                parent_text = current_node.get('text', '')
                merged_text = parent_text
                for child_text in children_texts:
                    if merged_text and not merged_text.endswith('\n'):
                        merged_text += '\n\n'
                    merged_text += child_text
                
                result_list[i]['text'] = merged_text
                
                result_list[i]['text_token_count'] = count_tokens(merged_text, model=model)
    
    for index in sorted(nodes_to_remove, reverse=True):
        result_list.pop(index)
    
    return result_list


def build_tree_from_nodes(node_list):
    if not node_list:
        return []
    
    stack = []
    root_nodes = []
    node_counter = 1
    
    for node in node_list:
        current_level = node['level']
        
        tree_node = {
            'title': node['title'],
            'node_id': str(node_counter).zfill(4),
            'text': node['text'],
            'line_num': node['line_num'],
            'span_kind': node.get('span_kind', 'line'),
            'end_line': node.get('end_line'),
            'nodes': []
        }
        node_counter += 1
        
        while stack and stack[-1][1] >= current_level:
            stack.pop()
        
        if not stack:
            root_nodes.append(tree_node)
        else:
            parent_node, parent_level = stack[-1]
            parent_node['nodes'].append(tree_node)
        
        stack.append((tree_node, current_level))
    
    return root_nodes


def clean_tree_for_output(tree_nodes):
    cleaned_nodes = []
    
    for node in tree_nodes:
        cleaned_node = {
            'title': node['title'],
            'node_id': node['node_id'],
            'text': node['text'],
            'line_num': node['line_num'],
            'span_kind': node.get('span_kind', 'line'),
            'end_line': node.get('end_line')
        }
        
        if node['nodes']:
            cleaned_node['nodes'] = clean_tree_for_output(node['nodes'])
        
        cleaned_nodes.append(cleaned_node)
    
    return cleaned_nodes


async def md_to_tree(md_path, if_thinning=False, min_token_threshold=None, if_add_node_summary='no', summary_token_threshold=None, model=None, if_add_doc_description='no', if_add_node_text='no', if_add_node_id='yes'):
    with open(md_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    markdown_content = _normalize_line_breaks(markdown_content)
    line_count = markdown_content.count('\n') + 1

    print(f"Extracting nodes from markdown...")
    node_list, markdown_lines = extract_nodes_from_markdown(markdown_content)

    print(f"Extracting text content from nodes...")
    nodes_with_content = extract_node_text_content(node_list, markdown_lines)
    
    if if_thinning:
        nodes_with_content = update_node_list_with_text_token_count(nodes_with_content, model=model)
        print(f"Thinning nodes...")
        nodes_with_content = tree_thinning_for_index(nodes_with_content, min_token_threshold, model=model)
    
    print(f"Building tree from nodes...")
    tree_structure = build_tree_from_nodes(nodes_with_content)

    # 阶段2：无标题/无标记结构时（空树），用 LLM 语义章节切分重建树，
    # 使短、无结构文档也能长成树参与 Super-Tree 检索。
    if not tree_structure:
        sections = semantic_sections_from_markdown(markdown_content, model=model)
        if sections:
            # Sort sections by line_num to compute text intervals
            sections_sorted = sorted(sections, key=lambda s: s.get("line_num", 1))
            semantic_nodes = []
            for i, s in enumerate(sections_sorted):
                start = s.get("line_num", 1) - 1  # 0-indexed
                if i + 1 < len(sections_sorted):
                    end = sections_sorted[i + 1].get("line_num", 1) - 1
                else:
                    end = len(markdown_lines)
                text = '\n'.join(markdown_lines[start:end]).strip()
                # Fallback to full content if interval slicing yields nothing
                if not text:
                    text = markdown_content.strip()
                semantic_nodes.append({
                    "title": s["title"],
                    "line_num": s.get("line_num", 1),
                    "level": 1,
                    "text": text,
                    "span_kind": "line",
                    "end_line": end,
                })
            tree_structure = build_tree_from_nodes(semantic_nodes)

    if if_add_node_id == 'yes':
        write_node_id(tree_structure)

    print(f"Formatting tree structure...")
    
    if if_add_node_summary == 'yes':
        # Always include text for summary generation
        tree_structure = format_structure(tree_structure, order = ['title', 'node_id', 'line_num', 'span_kind', 'end_line', 'summary', 'prefix_summary', 'text', 'nodes'])
        
        print(f"Generating summaries for each node...")
        tree_structure = await generate_summaries_for_structure_md(tree_structure, summary_token_threshold=summary_token_threshold, model=model)
        
        if if_add_node_text == 'no':
            # Remove text after summary generation if not requested
            tree_structure = format_structure(tree_structure, order = ['title', 'node_id', 'line_num', 'span_kind', 'end_line', 'summary', 'prefix_summary', 'nodes'])
        
        if if_add_doc_description == 'yes':
            print(f"Generating document description...")
            # Create a clean structure without unnecessary fields for description generation
            clean_structure = create_clean_structure_for_description(tree_structure)
            doc_description = generate_doc_description(clean_structure, model=model)
            return {
                'doc_name': os.path.splitext(os.path.basename(md_path))[0],
                'doc_description': doc_description,
                'line_count': line_count,
                'structure': tree_structure,
            }
    else:
        # No summaries needed, format based on text preference
        if if_add_node_text == 'yes':
            tree_structure = format_structure(tree_structure, order = ['title', 'node_id', 'line_num', 'span_kind', 'end_line', 'summary', 'prefix_summary', 'text', 'nodes'])
        else:
            tree_structure = format_structure(tree_structure, order = ['title', 'node_id', 'line_num', 'span_kind', 'end_line', 'summary', 'prefix_summary', 'nodes'])
    
    return {
        'doc_name': os.path.splitext(os.path.basename(md_path))[0],
        'line_count': line_count,
        'structure': tree_structure,
    }


if __name__ == "__main__":
    import os
    import json

    # Deterministic relative import (quality-gate improvement #5): __main__ has
    # no test/CI coverage, so use a try/except fallback to cover both
    # `python -m pageindex_mutil.page_index_md` (relative import works) and a
    # direct `python pageindex_mutil/page_index_md.py` invocation (falls back to
    # the absolute package import).
    try:
        from .utils import ConfigLoader
    except ImportError:
        from pageindex_mutil.utils import ConfigLoader

    # MD_NAME = 'Detect-Order-Construct'
    MD_NAME = 'cognitive-load'
    MD_PATH = os.path.join(os.path.dirname(__file__), '..', 'examples/documents/', f'{MD_NAME}.md')


    _cfg = ConfigLoader().load(None)
    tree_structure = asyncio.run(md_to_tree(
        md_path=MD_PATH,
        if_thinning=_cfg.if_thinning,
        min_token_threshold=_cfg.thinning_threshold,
        if_add_node_summary='yes' if _cfg.if_summary else 'no',
        summary_token_threshold=_cfg.summary_token_threshold,
        model=_cfg.model))
    
    print('\n' + '='*60)
    print('TREE STRUCTURE')
    print('='*60)
    print_json(tree_structure)

    print('\n' + '='*60)
    print('TABLE OF CONTENTS')
    print('='*60)
    print_toc(tree_structure['structure'])

    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', f'{MD_NAME}_structure.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tree_structure, f, indent=2, ensure_ascii=False)
    
    print(f"\nTree structure saved to: {output_path}")