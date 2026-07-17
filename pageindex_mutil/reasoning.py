"""Shared reasoning logic for RAG retrieval.

Extracted from main.py to break the circular dependency between
main.py (CLI) and pageindex_mutil (library). Both main.py and
client.py/router.py import from this module instead of importing main.py.
"""

import json

from .utils import (
    extract_json,
    count_tokens,
    get_llm_client,
    get_llm_config,
    ConfigLoader,
)

# Multi-doc context budget (configurable via config.yaml)
_cfg = ConfigLoader().load(None)
MAX_CONTEXT_TOKENS = getattr(_cfg, "max_context_tokens", 16000)

# Resolve model names (same precedence as before: env > config.yaml)
MODEL_NAME = _cfg.model
RETRIEVE_MODEL_NAME = _cfg.retrieve_model


def _call_llm_json(prompt, extract_key=None, expect_list=False):
    """Generic LLM JSON caller.

    If extract_key is set, pulls that key from a dict result.
    If expect_list is True, expects the result to be a list directly.
    """
    client = get_llm_client()
    if not client:
        return []
    try:
        response = client.chat.completions.create(
            model=RETRIEVE_MODEL_NAME or MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content
        if not content:
            return []
        result = extract_json(content)
        if expect_list and isinstance(result, list):
            return result
        if extract_key and isinstance(result, dict) and isinstance(result.get(extract_key), list):
            return [str(x) for x in result[extract_key] if x is not None]
        return []
    except Exception:
        return []


def pages_from_nodes(nodes):
    """Extract unique page numbers from a list of nodes."""
    seen = set()
    pages = []
    for node in nodes:
        start = node.get('start_index')
        end = node.get('end_index')
        if start is None or end is None:
            continue
        for p in range(start, end + 1):
            if p not in seen:
                seen.add(p)
                pages.append(p)
    return pages


def get_relevant_nodes(question, tree_json_str):
    """Find relevant node IDs for a question using LLM."""
    prompt = f"""
        You are given a question and a tree structure of a document.
        Each node contains a node id, node title, and a corresponding summary.
        Your task is to find all nodes that are likely to contain the answer to the question.

        Question: {question}

        Document tree structure:
        {tree_json_str}

        Please reply in the following JSON format:
        {{
            "thinking": "<Your thinking process on which nodes are relevant to the question>",
            "node_list": ["node_id_1", "node_id_2", "..."]
        }}
        Directly return the final JSON structure. Do not output anything else.
        """
    return _call_llm_json(prompt, extract_key='node_list')


def get_relevant_pages(question, toc_text):
    """Find relevant page numbers from TOC using LLM."""
    prompt = f"""
        You are an intelligent assistant.
        I have a document with the following Table of Contents (TOC), which may include summaries for each section:

        {toc_text}

        The user has asked: "{question}"

        Based on the TOC and summaries, which pages are most likely to contain the answer?
        Reasoning about which section covers the topic.
        Then return ONLY a JSON list of physical page numbers.
        Format: [page_num1, page_num2, ...]
        Example: [5, 6, 7]

        If you are unsure, select the most relevant sections' pages.
        """
    pages = _call_llm_json(prompt, expect_list=True)
    return [int(p) for p in pages if isinstance(p, (int, str)) and str(p).isdigit()]


def extract_text_from_db(db, doc_id, pages):
    """Extract text content from DB for given pages."""
    rows = db.get_pages_by_numbers(doc_id, pages)
    parts = []
    for page_num, content in rows:
        parts.append(f"\n--- Page {page_num} ---\n")
        parts.append(content)
    return "".join(parts)


def generate_answer(question, context):
    """Generate an answer using the LLM given a question and context."""
    client = get_llm_client()
    if not client:
        return "Error: OpenAI client not initialized."

    prompt = f"""
        Answer the user's question based on the following context.
        If the answer is not in the context, say "I cannot find the answer in the provided context."

        Context:
        {context}

        Question: {question}
        """
    try:
        response = client.chat.completions.create(
            model=RETRIEVE_MODEL_NAME or MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {e}"


def build_context_with_budget(db, doc_id, pages, doc_name, remaining_tokens):
    """Extract page text from DB while respecting a token budget."""
    if remaining_tokens <= 0:
        return "", 0, True

    rows = db.get_pages_by_numbers(doc_id, sorted(set(pages)))
    parts = [f"\n=== Document: {doc_name} ===\n"]
    truncated = False
    used = count_tokens(parts[0])

    for page_num, content in rows:
        page_text = f"\n--- Page {page_num} ---\n{content}"
        page_tokens = count_tokens(page_text)
        if used + page_tokens > remaining_tokens:
            truncated = True
            break
        parts.append(page_text)
        used += page_tokens

    return "".join(parts), used, truncated


def get_relevant_documents_for_multidoc(question, docs_info):
    """Select top-K relevant documents from a list of doc metadata."""
    prompt = f"""
You are given a user question and a list of documents.
Your task is to identify which documents are most likely to contain the answer.
Select up to 3 most relevant documents.

User Question: {question}

Documents:
{json.dumps(docs_info, indent=2, ensure_ascii=False)}

Please reply in the following JSON format:
{{
    "thinking": "<brief reasoning>",
    "doc_ids": ["doc_id_1", "doc_id_2", ...]
}}
Directly return the final JSON structure. Do not output anything else.
If no documents seem relevant, return an empty list.
"""
    return _call_llm_json(prompt, extract_key='doc_ids')
