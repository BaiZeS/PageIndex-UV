import os
import sys
import json
import fitz  # pymupdf
from openai import OpenAI
from dotenv import load_dotenv
from types import SimpleNamespace

# Add PageIndex to path to allow imports
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, "PageIndex"))

# Load environment variables
load_dotenv()

# Configuration
# Try to get API Key from DASHSCOPE_API_KEY first, then OPENAI_API_KEY
API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
# Qwen/DashScope compatible base URL
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = "qwen-plus"

# Ensure environment variables are set for the PageIndex library
if API_KEY:
    os.environ["CHATGPT_API_KEY"] = API_KEY
    # The library uses os.getenv("CHATGPT_API_KEY")
if BASE_URL:
    os.environ["OPENAI_BASE_URL"] = BASE_URL
    # Standard OpenAI client in library will respect this

if not API_KEY:
    print("Warning: API Key not found. Please set DASHSCOPE_API_KEY or OPENAI_API_KEY environment variable.")

try:
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

def generate_structure(pdf_path, json_path):
    print(f"Structure file not found. Generating index for {pdf_path}...")
    print("This may take a few minutes...")
    
    try:
        from pageindex.page_index import page_index_main
        
        # Configure options matching defaults in run_pageindex.py
        opt = SimpleNamespace(
            model=MODEL_NAME,
            toc_check_page_num=20,
            max_page_num_each_node=10,
            max_token_num_each_node=20000,
            if_add_node_id='yes',
            if_add_node_summary='yes',
            if_add_doc_description='no',
            if_add_node_text='no'
        )
        
        # Process the PDF
        toc_with_page_number = page_index_main(pdf_path, opt)
        
        # Save results
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)
            
        print(f"Structure generated and saved to: {json_path}")
        return True
    except Exception as e:
        print(f"Error generating structure: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_structure(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_toc(nodes, indent=0):
    toc_text = ""
    for node in nodes:
        prefix = "  " * indent
        # Use start_index/end_index as page range
        start = node.get('start_index', '?')
        end = node.get('end_index', '?')
        title = node.get('title', 'Untitled')
        summary = node.get('summary', '').strip()
        
        toc_text += f"{prefix}- {title} (Pages: {start}-{end})"
        if summary:
            toc_text += f"\n{prefix}  Summary: {summary}"
        toc_text += "\n"
        
        if 'nodes' in node and node['nodes']:
            toc_text += format_toc(node['nodes'], indent + 1)
    return toc_text

def get_relevant_pages(question, toc_text):
    if not client:
        print("OpenAI client not initialized.")
        return []

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

        # Simple cleanup to ensure JSON
        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        # Clean up any non-json text that might remain (e.g. "Here is the list: [1, 2]")
        start_idx = content.find('[')
        end_idx = content.rfind(']')
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx+1]

        pages = json.loads(content)
        if isinstance(pages, list):
            # Ensure integers
            return [int(p) for p in pages if isinstance(p, (int, str)) and str(p).isdigit()]
        return []
    except Exception as e:
        print(f"Error getting relevant pages: {e}")
        return []

def extract_text_from_pdf(pdf_path, pages):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in pages:
            # PDF pages are 0-indexed, but our structure uses 1-indexed
            idx = page_num - 1
            if 0 <= idx < len(doc):
                page = doc[idx]
                text += f"\n--- Page {page_num} ---\n"
                text += page.get_text()
            else:
                print(f"Warning: Page {page_num} is out of range.")
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

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
    pdfs = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    return sorted(pdfs)

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

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(base_dir, "PageIndex", "tests", "pdfs")
    results_dir = os.path.join(base_dir, "PageIndex", "tests", "results")
    
    # Ensure directories exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Select PDF
    selected_pdf_name = select_pdf(pdf_dir)
    if not selected_pdf_name:
        return

    pdf_path = os.path.join(pdf_dir, selected_pdf_name)
    json_filename = os.path.splitext(selected_pdf_name)[0] + "_structure.json"
    json_path = os.path.join(results_dir, json_filename)

    print(f"\nSelected: {selected_pdf_name}")
    
    # Check if PDF exists (sanity check)
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return

    # Check if JSON exists, if not generate it
    if not os.path.exists(json_path):
        success = generate_structure(pdf_path, json_path)
        if not success:
            print("Failed to generate structure. Exiting.")
            return

    # Load Structure
    print(f"Loading structure from {json_path}...")
    try:
        data = load_structure(json_path)
    except Exception as e:
        print(f"Error loading structure JSON: {e}")
        return

    toc_text = format_toc(data.get('structure', []))
    
    print("\n" + "="*50)
    print("Welcome to PageIndex Q&A (Powered by Qwen-Plus)")
    print(f"Document: {data.get('doc_name', 'Unknown')}")
    print("="*50)
    
    # Interactive Loop
    while True:
        try:
            question = input("\nAsk a question (or 'q' to quit): ").strip()
        except EOFError:
            break
            
        if question.lower() in ('q', 'quit', 'exit'):
            break
        
        if not question:
            continue

        print("\nThinking (Retrieving relevant pages)...")
        pages = get_relevant_pages(question, toc_text)
        print(f"Identified relevant pages: {pages}")
        
        if not pages:
            print("Could not find relevant pages based on TOC.")
            continue
            
        print("Reading content...")
        context = extract_text_from_pdf(pdf_path, pages)
        
        print("Generating answer...")
        answer = generate_answer(question, context)
        
        print("\nAnswer:")
        print(answer)
        print("-" * 50)

if __name__ == "__main__":
    main()
