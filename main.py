import os
import sys
import json
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

MODE_SINGLE = 's'
MODE_MULTI = 'm'
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


def select_pdf(pdf_dir):
    pdfs = list_pdfs(pdf_dir)
    if not pdfs:
        print(f"No PDF files found in {pdf_dir}")
        return None

    print("\nAvailable Documents:")
    for i, pdf in enumerate(pdfs, 1):
        print(f"{i}. {pdf}")

    while True:
        try:
            choice = input("\nSelect a document by number (or 'q' to quit): ").strip()
            if choice.lower() in ('q', 'quit', 'exit'):
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(pdfs):
                return pdfs[idx]
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


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


def run_multidoc_mode(db, logs_dir, log_f):
    """Multi-document Q&A loop using L1 -> L2 -> L3 layered retrieval."""
    all_docs = db.get_all_documents()
    if not all_docs:
        print("No cached documents found. Please process at least one PDF in single-doc mode first.")
        return

    print("\n" + "=" * 50)
    print("Welcome to PageIndex Multi-Doc Q&A")
    print(f"Loaded {len(all_docs)} cached document(s):")
    for doc in all_docs:
        desc = doc.get('doc_description') or ''
        print(f"- {doc['pdf_name']}" + (f" ({desc[:60]}...)" if desc else ""))
    print("=" * 50)

    # Pre-build a lightweight doc info list for L1 and cache tree JSON strings
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

    while True:
        try:
            question = input("\nAsk a question (or 'q' to quit): ").strip()
        except EOFError:
            break

        if question.lower() in ('q', 'quit', 'exit'):
            break
        if not question:
            continue

        # L1: Document selection
        print("\nThinking (Selecting relevant documents)...")
        relevant_doc_ids = get_relevant_documents_for_multidoc(question, docs_info)
        if not relevant_doc_ids:
            print("No relevant documents found for this question.")
            continue

        relevant_docs = [doc_lookup[int(did)] for did in relevant_doc_ids if int(did) in doc_lookup]
        if not relevant_docs:
            print("No valid documents found after filtering.")
            continue

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

            # Limit to top 5 nodes per doc
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
            continue

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

        print("\nAnswer:")
        print(answer)
        print("-" * 50)


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

    # Mode selection
    print("\nSelect mode:")
    print("  (s) Single-document Q&A")
    print("  (m) Multi-document Q&A")
    mode_choice = input("Mode [s/m]: ").strip().lower()
    if mode_choice == MODE_MULTI:
        date_tag = datetime.now().strftime("%Y%m%d")
        log_path = os.path.join(logs_dir, f"qa_multidoc_{date_tag}.jsonl")
        try:
            log_f = open(log_path, "a", encoding="utf-8")
        except OSError as e:
            print(f"Error opening log file {log_path}: {e}")
            db.close()
            return
        try:
            run_multidoc_mode(db, logs_dir, log_f)
        finally:
            log_f.close()
            db.close()
        return

    # Single-document mode (original behavior)
    selected_pdf_name = select_pdf(pdf_dir)
    if not selected_pdf_name:
        db.close()
        return

    pdf_path = os.path.join(pdf_dir, selected_pdf_name)
    json_filename = os.path.splitext(selected_pdf_name)[0] + "_structure.json"
    json_path = os.path.join(results_dir, json_filename)

    print(f"\nSelected: {selected_pdf_name}")

    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        db.close()
        return

    doc = db.get_document_by_name(selected_pdf_name)

    if not doc:
        if not os.path.exists(json_path):
            data = generate_structure(pdf_path, json_path)
            if not data:
                print("Failed to generate structure. Exiting.")
                db.close()
                return
        else:
            print(f"Loading structure from {json_path}...")
            try:
                data = load_structure(json_path)
            except Exception as e:
                print(f"Error loading structure JSON: {e}")
                db.close()
                return

        structure = data.get('structure', [])
        if not structure:
            print("Structure is empty.")
            db.close()
            return
        ensure_node_ids(structure)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        doc_description = data.get('doc_description', '')
        doc_id = db.insert_document(selected_pdf_name, pdf_path, doc_description=doc_description)
        db.insert_nodes(doc_id, _flatten_nodes(structure, doc_id))
        db.insert_pages(doc_id, _extract_page_records(pdf_path, doc_id))

        tree_for_reasoning = create_clean_structure_for_description(structure)
        db.update_document_tree(doc_id, json.dumps(tree_for_reasoning, ensure_ascii=False))
        doc = db.get_document_by_name(selected_pdf_name)
    else:
        doc_id = doc['id']
        tree_json = doc.get('tree_json')
        if tree_json:
            tree_for_reasoning = json.loads(tree_json)
        else:
            print(f"Loading structure from {json_path}...")
            try:
                data = load_structure(json_path)
            except Exception as e:
                print(f"Error loading structure JSON: {e}")
                db.close()
                return
            structure = data.get('structure', [])
            tree_for_reasoning = create_clean_structure_for_description(structure)

    # Build TOC text from tree for fallback logic
    toc_text = format_toc(tree_for_reasoning)

    tree_json_str = json.dumps(tree_for_reasoning, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("Welcome to PageIndex Q&A (Powered by Qwen-Plus)")
    print(f"Document: {selected_pdf_name}")
    print("=" * 50)

    date_tag = datetime.now().strftime("%Y%m%d")
    log_path = os.path.join(logs_dir, f"qa_{date_tag}.jsonl")
    try:
        log_f = open(log_path, "a", encoding="utf-8")
    except OSError as e:
        print(f"Error opening log file {log_path}: {e}")
        db.close()
        return

    try:
        while True:
            try:
                question = input("\nAsk a question (or 'q' to quit): ").strip()
            except EOFError:
                break

            if question.lower() in ('q', 'quit', 'exit'):
                break

            if not question:
                continue

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
                    continue
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
                        continue
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

            print("\nAnswer:")
            print(answer)
            print("-" * 50)
    finally:
        log_f.close()
        db.close()


if __name__ == "__main__":
    main()
