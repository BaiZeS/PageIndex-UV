import os
import sys
import json
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from types import SimpleNamespace
from datetime import datetime
import fitz

from db import PageIndexDB

# Add PageIndex to path to allow imports
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, "PageIndex"))

from pageindex.utils import extract_json, create_clean_structure_for_description, write_node_id, count_tokens

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-plus")

# Ensure environment variables are set for the PageIndex library
if API_KEY:
    os.environ["CHATGPT_API_KEY"] = API_KEY
if BASE_URL:
    os.environ["OPENAI_BASE_URL"] = BASE_URL

if not API_KEY:
    print("Warning: API Key not found. Please set DASHSCOPE_API_KEY or OPENAI_API_KEY environment variable.")

try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

# Multi-doc context budget
MAX_CONTEXT_TOKENS = 16000

REASONING_TREE = 'tree'
REASONING_TOC = 'toc_fallback'


def generate_structure(pdf_path, json_path):
    print(f"Structure file not found. Generating index for {pdf_path}...")
    print("This may take a few minutes...")

    try:
        from pageindex.page_index import page_index_main

        opt = SimpleNamespace(
            model=MODEL_NAME,
            toc_check_page_num=20,
            max_page_num_each_node=10,
            max_token_num_each_node=20000,
            if_add_node_id='yes',
            if_add_node_summary='yes',
            if_add_doc_description='yes',
            if_add_node_text='no'
        )

        toc_with_page_number = page_index_main(pdf_path, opt)

        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)

        print(f"Structure generated and saved to: {json_path}")
        return toc_with_page_number
    except Exception as e:
        print(f"Error generating structure: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_structure(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_toc(nodes, indent=0):
    parts = []
    for node in nodes:
        prefix = "  " * indent
        start = node.get('start_index', '?')
        end = node.get('end_index', '?')
        title = node.get('title', 'Untitled')
        summary = node.get('summary', '').strip()

        parts.append(f"{prefix}- {title} (Pages: {start}-{end})")
        if summary:
            parts.append(f"{prefix}  Summary: {summary}")
        parts.append("")

        if node.get('nodes'):
            parts.extend(format_toc(node['nodes'], indent + 1))
    return "\n".join(parts)


def _any_node_id(nodes):
    for node in nodes:
        if node.get('node_id') is not None:
            return True
        if node.get('nodes') and _any_node_id(node['nodes']):
            return True
    return False


def ensure_node_ids(structure):
    if _any_node_id(structure):
        return structure
    write_node_id(structure)
    return structure


def _call_llm_json(prompt, extract_key=None, expect_list=False):
    """Generic LLM JSON caller.

    If extract_key is set, pulls that key from a dict result.
    If expect_list is True, expects the result to be a list directly.
    """
    if not client:
        print("OpenAI client not initialized.")
        return []
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
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
    except Exception as e:
        print(f"Error in LLM call: {e}")
        return []


def pages_from_nodes(nodes):
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


def _flatten_nodes(structure, doc_id, parent_node_id=None, out=None):
    if out is None:
        out = []
    for node in structure:
        node_id = node.get("node_id")
        out.append((
            doc_id,
            node_id,
            node.get("title"),
            node.get("summary"),
            node.get("start_index"),
            node.get("end_index"),
            parent_node_id,
        ))
        children = node.get("nodes")
        if children:
            _flatten_nodes(children, doc_id, parent_node_id=node_id, out=out)
    return out


def _extract_page_records(pdf_path, doc_id):
    doc = fitz.open(pdf_path)
    records = []
    for idx in range(len(doc)):
        page_num = idx + 1
        content = doc[idx].get_text()
        records.append((doc_id, page_num, content))
    doc.close()
    return records


def get_relevant_nodes(question, tree_json_str):
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
    rows = db.get_pages_by_numbers(doc_id, pages)
    parts = []
    for page_num, content in rows:
        parts.append(f"\n--- Page {page_num} ---\n")
        parts.append(content)
    return "".join(parts)


def ensure_logs_dir(base_dir):
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def write_qa_log(log_f, record):
    try:
        log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_f.flush()
    except Exception as e:
        print(f"Failed to write log: {e}")


def generate_answer(question, context):
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
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {e}"


def list_pdfs(directory):
    return sorted([
        entry.name for entry in os.scandir(directory)
        if entry.is_file() and entry.name.lower().endswith('.pdf')
    ])


# ---------------------------------------------------------------------------
# Batch upload helpers for /add command
# ---------------------------------------------------------------------------

_ALLOWED_EXTENSIONS = ('.pdf', '.md', '.markdown')


def _resolve_upload_paths(raw_path, pdf_dir):
    """Resolve a raw path into a list of file paths.

    Supports:
      - Single file
      - Glob pattern (contains * or ?)
      - Directory (recursively collects .pdf, .md, .markdown)
    """
    # Try pdf_dir resolution for relative bare paths first
    if not os.path.isabs(raw_path) and not os.path.exists(raw_path):
        candidate = os.path.join(pdf_dir, raw_path)
        if os.path.exists(candidate):
            raw_path = candidate

    if os.path.isdir(raw_path):
        files = []
        for root, _dirs, filenames in os.walk(raw_path):
            for name in filenames:
                if name.lower().endswith(_ALLOWED_EXTENSIONS):
                    files.append(os.path.join(root, name))
        return sorted(files)

    if '*' in raw_path or '?' in raw_path:
        matched = glob.glob(raw_path)
        return sorted([
            f for f in matched
            if os.path.isfile(f) and f.lower().endswith(_ALLOWED_EXTENSIONS)
        ])

    # Single file: validate existence and extension
    if not os.path.exists(raw_path):
        return []
    if not raw_path.lower().endswith(_ALLOWED_EXTENSIONS):
        return []
    return [raw_path]


def _index_files_batch(db_path, file_paths, results_dir):
    """Index a list of files concurrently with max 3 workers.

    Each worker creates its own DB connection to avoid SQLite thread-safety issues.
    Prints per-file progress and a final summary.
    Returns True if at least one file was newly indexed successfully.
    """
    total = len(file_paths)
    print(f"Indexing {total} file(s)...")

    def _index_one(path):
        filename = os.path.basename(path)
        json_filename = os.path.splitext(filename)[0] + "_structure.json"
        json_path = os.path.join(results_dir, json_filename)

        # Create a fresh DB connection per thread to avoid SQLite thread checks
        db = PageIndexDB(db_path)
        doc = db.get_document_by_name(filename)
        if doc:
            db.close()
            return filename, 'skipped', None

        try:
            doc = index_pdf(db, path, json_path, filename)
            db.close()
            if doc:
                return filename, 'ok', doc['id']
            else:
                return filename, 'error', 'index_pdf returned None'
        except Exception as exc:
            db.close()
            return filename, 'error', str(exc)

    succeeded = 0
    failed = 0
    skipped = 0
    any_new = False

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_index_one, path) for path in file_paths]
        for idx, future in enumerate(as_completed(futures), start=1):
            filename, status, detail = future.result()
            if status == 'ok':
                print(f"[{idx}/{total}] {filename}... OK (doc_id={detail})")
                succeeded += 1
                any_new = True
            elif status == 'skipped':
                print(f"[{idx}/{total}] {filename}... SKIPPED (already indexed)")
                skipped += 1
            else:
                print(f"[{idx}/{total}] {filename}... ERROR: {detail}")
                failed += 1

    print(f"Done: {succeeded} succeeded, {failed} failed, {skipped} skipped.")
    return any_new


def _fallback_to_toc(question, toc_text, db, doc_id, log_f, selected_pdf_name, mode, node_ids):
    pages = get_relevant_pages(question, toc_text)
    if not pages:
        print("Could not find relevant pages based on TOC.")
        write_qa_log(log_f, {
            "timestamp": datetime.now().isoformat(),
            "document": selected_pdf_name,
            "question": question,
            "mode": mode,
            "node_ids": node_ids,
            "pages": [],
            "answer": "",
            "status": "no_pages"
        })
        return None, None
    print(f"Identified relevant pages: {pages}")
    print("Reading content...")
    context = extract_text_from_db(db, doc_id, pages)
    return pages, context


# ---------------------------------------------------------------------------
# Multi-document retrieval helpers
# ---------------------------------------------------------------------------

def _build_multidoc_context(db):
    """Build lightweight doc info list and tree JSON cache from DB."""
    all_docs = db.get_all_documents()
    docs_info = []
    doc_lookup = {}
    doc_tree_json = {}
    for doc in all_docs:
        doc_id = doc['id']
        doc_lookup[doc_id] = doc
        top_nodes = db.get_top_level_nodes(doc_id)
        top_sections = [n.get('title') for n in top_nodes if n.get('title')]
        description = doc.get('doc_description') or ""
        if not description and top_nodes:
            description = " ".join([n.get('summary', '') for n in top_nodes[:3] if n.get('summary')])
        docs_info.append({
            "doc_id": str(doc_id),
            "doc_name": doc['pdf_name'],
            "description": description,
            "top_level_sections": top_sections
        })
        tree_json = doc.get('tree_json')
        if tree_json:
            tree = json.loads(tree_json)
            doc_tree_json[doc_id] = json.dumps(tree, indent=2, ensure_ascii=False)
    return docs_info, doc_lookup, doc_tree_json


def answer_multidoc(question, db, docs_info, doc_lookup, doc_tree_json, log_f):
    """Execute one multi-document Q&A. Returns (answer, status_dict) or (None, None)."""
    # L1: Document selection
    print("\nThinking (Selecting relevant documents)...")
    relevant_doc_ids = get_relevant_documents_for_multidoc(question, docs_info)
    if not relevant_doc_ids:
        print("No relevant documents found for this question.")
        return None, {"status": "no_docs"}

    relevant_docs = [doc_lookup[int(did)] for did in relevant_doc_ids if int(did) in doc_lookup]
    if not relevant_docs:
        print("No valid documents found after filtering.")
        return None, {"status": "no_valid_docs"}

    print(f"Selected {len(relevant_docs)} document(s): " + ", ".join([d['pdf_name'] for d in relevant_docs]))

    # L2: Per-document node recall
    all_selected_nodes = []
    doc_nodes_map = {}
    for doc in relevant_docs:
        doc_id = doc['id']
        tree_json_str = doc_tree_json.get(doc_id)
        if not tree_json_str:
            print(f"Skipping {doc['pdf_name']}: no cached tree structure.")
            continue

        print(f"Thinking (Nodes in {doc['pdf_name']})...")
        node_ids = get_relevant_nodes(question, tree_json_str)
        if not node_ids:
            print(f"  No relevant nodes in {doc['pdf_name']}")
            continue

        node_ids = node_ids[:5]
        selected_nodes = db.get_nodes_by_ids(doc_id, node_ids)
        if selected_nodes:
            doc_nodes_map[doc_id] = selected_nodes
            all_selected_nodes.extend(selected_nodes)
            print(f"  Found {len(selected_nodes)} node(s) in {doc['pdf_name']}")

    # L3: Context extraction with token budget
    context_parts = []
    remaining_tokens = MAX_CONTEXT_TOKENS
    overall_truncated = False
    context_log = []

    for doc in relevant_docs:
        doc_id = doc['id']
        selected_nodes = doc_nodes_map.get(doc_id, [])
        if not selected_nodes:
            continue

        pages = pages_from_nodes(selected_nodes)
        seen = set(pages)

        if not pages:
            continue

        ctx_text, used, truncated = build_context_with_budget(
            db, doc_id, pages, doc['pdf_name'], remaining_tokens
        )
        if ctx_text:
            context_parts.append(ctx_text)
            remaining_tokens -= used
            context_log.append({
                "doc_id": doc_id,
                "doc_name": doc['pdf_name'],
                "pages": sorted(seen),
                "tokens": used,
                "truncated": truncated
            })
        if truncated:
            overall_truncated = True

    if not context_parts:
        print("No relevant context found in any document.")
        return None, {"status": "no_context"}

    full_context = "\n".join(context_parts)
    if overall_truncated:
        full_context += "\n\n[Note: Some content was truncated to fit context budget.]"

    print("Generating answer...")
    answer = generate_answer(question, full_context)

    write_qa_log(log_f, {
        "timestamp": datetime.now().isoformat(),
        "mode": "multidoc",
        "question": question,
        "relevant_docs": [d['pdf_name'] for d in relevant_docs],
        "relevant_doc_ids": relevant_doc_ids,
        "truncated": overall_truncated,
        "context_details": context_log,
        "answer": answer,
        "status": "ok"
    })

    return answer, {"status": "ok", "truncated": overall_truncated}


def get_relevant_documents_for_multidoc(question, docs_info):
    """L1: Select top-K relevant documents from a list of doc metadata."""
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


# ---------------------------------------------------------------------------
# Unified helpers
# ---------------------------------------------------------------------------

def index_pdf(db, pdf_path, json_path, selected_pdf_name):
    """Index a PDF into the database. Returns doc dict or None."""
    if not os.path.exists(json_path):
        data = generate_structure(pdf_path, json_path)
        if not data:
            print("Failed to generate structure.")
            return None
    else:
        print(f"Loading structure from {json_path}...")
        try:
            data = load_structure(json_path)
        except Exception as e:
            print(f"Error loading structure JSON: {e}")
            return None

    structure = data.get('structure', [])
    if not structure:
        print("Structure is empty.")
        return None
    ensure_node_ids(structure)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    doc_description = data.get('doc_description', '')
    doc_id = db.insert_document(selected_pdf_name, pdf_path, doc_description=doc_description)
    db.insert_nodes(doc_id, _flatten_nodes(structure, doc_id))
    db.insert_pages(doc_id, _extract_page_records(pdf_path, doc_id))

    tree_for_reasoning = create_clean_structure_for_description(structure)
    db.update_document_tree(doc_id, json.dumps(tree_for_reasoning, ensure_ascii=False))
    return db.get_document_by_name(selected_pdf_name)


def answer_single_doc(question, db, doc_id, tree_json_str, toc_text, selected_pdf_name, log_f):
    """Execute one single-document Q&A. Returns answer string or None."""
    print("\nThinking (Reasoning over tree)...")
    mode = REASONING_TREE
    selected_nodes = []
    node_ids = get_relevant_nodes(question, tree_json_str)
    if not node_ids:
        print("Node reasoning failed. Falling back to TOC pages.")
        mode = REASONING_TOC
        pages, context = _fallback_to_toc(
            question, toc_text, db, doc_id, log_f,
            selected_pdf_name, mode, []
        )
        if context is None:
            return None
    else:
        selected_nodes = db.get_nodes_by_ids(doc_id, node_ids)
        if not selected_nodes:
            print("No valid nodes found. Falling back to TOC pages.")
            mode = REASONING_TOC
            pages, context = _fallback_to_toc(
                question, toc_text, db, doc_id, log_f,
                selected_pdf_name, mode, node_ids
            )
            if context is None:
                return None
        else:
            print("Identified relevant nodes:")
            for node in selected_nodes:
                print(f"- {node.get('title', 'Untitled')} (Pages: {node.get('start_index', '?')}-{node.get('end_index', '?')})")
            pages = pages_from_nodes(selected_nodes)
            print("Reading content...")
            context = extract_text_from_db(db, doc_id, pages)

    print("Generating answer...")
    answer = generate_answer(question, context)
    write_qa_log(log_f, {
        "timestamp": datetime.now().isoformat(),
        "document": selected_pdf_name,
        "question": question,
        "mode": mode,
        "node_ids": node_ids,
        "nodes": [
            {
                "node_id": node.get("node_id"),
                "title": node.get("title"),
                "start_index": node.get("start_index"),
                "end_index": node.get("end_index")
            } for node in selected_nodes
        ],
        "pages": pages,
        "context": context,
        "answer": answer,
        "status": "ok"
    })
    return answer


def _print_help():
    print("""
Commands:
  /add <pdf_path>   Index a new PDF document
  /doc <number>    Focus on a single document (type '..' to return)
  /list             List all indexed documents
  /help             Show this help message
  /quit  or  q      Exit the program

Any other input is treated as a question and answered using multi-document RAG.
""")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(base_dir, "PageIndex", "tests", "pdfs")
    results_dir = os.path.join(base_dir, "PageIndex", "tests", "results")
    logs_dir = ensure_logs_dir(base_dir)
    db_path = os.path.join(base_dir, "pageindex.db")

    os.makedirs(results_dir, exist_ok=True)

    db = PageIndexDB(db_path)

    # Print welcome and cached docs
    all_docs = db.get_all_documents()
    print("\n" + "=" * 50)
    print("PageIndex-UV Multi-Document Q&A")
    print(f"Powered by {MODEL_NAME}")
    print("=" * 50)
    if all_docs:
        print(f"Cached documents ({len(all_docs)}):")
        for i, doc in enumerate(all_docs, 1):
            desc = doc.get('doc_description') or ''
            print(f"  {i}. {doc['pdf_name']}" + (f" ({desc[:50]}...)" if desc else ""))
    else:
        print("No cached documents found.")
        print("Use `/add <pdf_path>` to index a document first.")
    print("-" * 50)

    # Open multi-doc log (always used for main loop)
    date_tag = datetime.now().strftime("%Y%m%d")
    log_path = os.path.join(logs_dir, f"qa_multidoc_{date_tag}.jsonl")
    try:
        log_f = open(log_path, "a", encoding="utf-8")
    except OSError as e:
        print(f"Error opening log file {log_path}: {e}")
        db.close()
        return

    # Pre-build multi-doc context if docs exist
    docs_info, doc_lookup, doc_tree_json = _build_multidoc_context(db) if all_docs else ([], {}, {})

    try:
        while True:
            try:
                user_input = input("\n> ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Command dispatch
            if user_input.lower() in ('q', 'quit', 'exit', '/quit'):
                break

            if user_input == '/help':
                _print_help()
                continue

            if user_input == '/list':
                all_docs = db.get_all_documents()
                if not all_docs:
                    print("No cached documents.")
                else:
                    print(f"\nCached documents ({len(all_docs)}):")
                    for i, doc in enumerate(all_docs, 1):
                        desc = doc.get('doc_description') or ''
                        print(f"  {i}. {doc['pdf_name']}" + (f" ({desc[:50]}...)" if desc else ""))
                    docs_info, doc_lookup, doc_tree_json = _build_multidoc_context(db)
                continue

            if user_input.startswith('/add '):
                raw_path = user_input[5:].strip()
                file_paths = _resolve_upload_paths(raw_path, pdf_dir)

                if not file_paths:
                    print(f"No matching files found: {raw_path}")
                    continue

                any_new = _index_files_batch(db_path, file_paths, results_dir)
                if any_new:
                    docs_info, doc_lookup, doc_tree_json = _build_multidoc_context(db)
                continue

            if user_input.startswith('/doc '):
                parts = user_input.split(None, 1)
                if len(parts) < 2:
                    print("Usage: /doc <document_number>")
                    continue
                try:
                    doc_idx = int(parts[1]) - 1
                except ValueError:
                    print("Please provide a valid document number.")
                    continue

                all_docs = db.get_all_documents()
                if not all_docs or doc_idx < 0 or doc_idx >= len(all_docs):
                    print("Invalid document number.")
                    continue

                doc = all_docs[doc_idx]
                doc_id = doc['id']
                selected_pdf_name = doc['pdf_name']
                pdf_path = doc.get('pdf_path', os.path.join(pdf_dir, selected_pdf_name))
                json_filename = os.path.splitext(selected_pdf_name)[0] + "_structure.json"
                json_path = os.path.join(results_dir, json_filename)

                # Ensure tree structure is available
                tree_json = doc.get('tree_json')
                if tree_json:
                    tree_for_reasoning = json.loads(tree_json)
                else:
                    if os.path.exists(json_path):
                        data = load_structure(json_path)
                        structure = data.get('structure', [])
                        tree_for_reasoning = create_clean_structure_for_description(structure)
                        db.update_document_tree(doc_id, json.dumps(tree_for_reasoning, ensure_ascii=False))
                    else:
                        print(f"Structure file not found for {selected_pdf_name}. Re-indexing...")
                        doc = index_pdf(db, pdf_path, json_path, selected_pdf_name)
                        if not doc:
                            continue
                        tree_for_reasoning = json.loads(doc['tree_json'])

                toc_text = format_toc(tree_for_reasoning)
                tree_json_str = json.dumps(tree_for_reasoning, indent=2, ensure_ascii=False)

                print("\n" + "=" * 50)
                print(f"Focusing on: {selected_pdf_name}")
                print("Type '..' to return to multi-document mode.")
                print("=" * 50)

                while True:
                    try:
                        q = input("  > ").strip()
                    except EOFError:
                        break
                    if q in ('..', '...'):
                        break
                    if q.lower() in ('q', 'quit', 'exit'):
                        return
                    if not q:
                        continue

                    answer = answer_single_doc(q, db, doc_id, tree_json_str, toc_text, selected_pdf_name, log_f)
                    if answer is not None:
                        print(f"\n  Answer:\n  {answer}\n  {'-' * 46}")
                continue

            # Default: treat as multi-document question
            if not all_docs:
                print("No documents indexed. Use `/add <pdf_path>` first.")
                continue

            answer, _ = answer_multidoc(user_input, db, docs_info, doc_lookup, doc_tree_json, log_f)
            if answer is not None:
                print("\nAnswer:")
                print(answer)
                print("-" * 50)
    finally:
        log_f.close()
        db.close()


if __name__ == "__main__":
    main()
