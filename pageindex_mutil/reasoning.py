"""Shared reasoning logic for RAG retrieval.

Extracted from main.py to break the circular dependency between
main.py (CLI) and pageindex_mutil (library). Both main.py and
client.py/router.py import from this module instead of importing main.py.
"""

from .utils import (
    extract_json,
    count_tokens,
    get_llm_client,
    get_llm_config,
    ConfigLoader,
)

# Lazy-loaded config values (avoid module-level side effects)
_cfg_cache = None


def _get_config():
    global _cfg_cache
    if _cfg_cache is None:
        _cfg_cache = ConfigLoader().load(None)
    return _cfg_cache


def _get_max_context_tokens():
    cfg = _get_config()
    return getattr(cfg, "max_context_tokens", 16000)


def _get_model_name():
    cfg = _get_config()
    return cfg.model


def _get_retrieve_model_name():
    cfg = _get_config()
    return cfg.retrieve_model


# Module-level aliases expected by app/main.py
MAX_CONTEXT_TOKENS = _get_max_context_tokens()
MODEL_NAME = _get_model_name()
RETRIEVE_MODEL_NAME = _get_retrieve_model_name()


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
            model=_get_retrieve_model_name() or _get_model_name(),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            extra_body={"thinking": {"type": "disabled"}},
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


def spans_from_nodes(nodes):
    """按 span_kind 分派节点跨度：PDF→页集合；MD→(node_id, 起, 止) 行区间。

    统一 span locator（[S3]/[S4]）：下游不再按 doc type hack，改按节点 span_kind
    分派——page 节点输出页区间（去重）；line 节点输出 (node_id, start_line,
    end_line) 行区间（1-based 含末行）。缺 span_kind 时按 start_index 是否存
    在兜底判定（page/line），兼容解析期尚未落存 span_kind 的存量节点。
    """
    pages, lines, seen = [], [], set()
    for node in nodes or []:
        kind = node.get("span_kind") or ("page" if node.get("start_index") is not None else "line")
        if kind == "page":
            start, end = node.get("start_index"), node.get("end_index")
            if start is None or end is None:
                continue
            for p in range(start, end + 1):
                if p not in seen:
                    seen.add(p)
                    pages.append(p)
        else:
            start, end = node.get("line_num"), node.get("end_line")
            if start is None or end is None:
                continue
            lines.append((node.get("node_id"), start, end))
    return {"pages": pages, "lines": lines}


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
        如证据分布在多个段落，请综合多处证据作答。

        Context:
        {context}

        Question: {question}
        """
    try:
        response = client.chat.completions.create(
            model=_get_retrieve_model_name() or _get_model_name(),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            extra_body={"thinking": {"type": "disabled"}},
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


def build_context_for_doc(doc, selected_nodes, pages):
    """Build context string for a single document from selected nodes and pages.

    Shared logic used by both single-doc and multi-doc search paths.
    Returns (context_string, pages_used).
    """
    ctx_parts = [f"\n=== Document: {doc.get('doc_name', '')} ===\n"]
    if doc.get("type") == "pdf" and doc.get("pages"):
        page_map = {p["page"]: p["content"] for p in doc["pages"]}
        for p in sorted(set(pages)):
            text = page_map.get(p, "")
            if text:
                ctx_parts.append(f"\n--- Page {p} ---\n{text}")
    elif doc.get("type") == "md":
        for node in selected_nodes:
            txt = node.get("text", "")
            if txt:
                ctx_parts.append(f"\n--- {node.get('title', '')} ---\n{txt}")

    context = "".join(ctx_parts) if len(ctx_parts) > 1 else ""
    return context
