#!/usr/bin/env python3
"""Smoke test for Agentic Multi-Strategy Router."""

import asyncio
import tempfile
import os


def create_test_md(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("""# Test Document

## Introduction
This is a test document for PageIndex-UV.

## Features
- Feature A: supports PDF indexing
- Feature B: supports Markdown indexing
- Feature C: agentic multi-strategy search

## Conclusion
PageIndex-UV provides tree-based document retrieval.
""")


async def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = os.path.join(tmpdir, "workspace")
        db_path = os.path.join(tmpdir, "test.db")
        md_path = os.path.join(tmpdir, "test.md")
        create_test_md(md_path)

        print("=" * 50)
        print("Smoke Test: Agentic Multi-Strategy Router")
        print("=" * 50)

        # 1. Import
        print("\n[1/6] Import test...")
        from PageIndex.pageindex import PageIndexClient
        print("  OK")

        # 2. Initialize client with workspace + db
        print("\n[2/6] Initialize client...")
        client = PageIndexClient(
            workspace=workspace,
            db_path=db_path,
        )
        print(f"  OK (docs={len(client.documents)})")

        # 3. Index Markdown
        print("\n[3/6] Index Markdown...")
        doc_id = client.index(md_path, mode="md")
        print(f"  OK (doc_id={doc_id})")

        # 4. Verify db persistence
        print("\n[4/6] Verify DB persistence...")
        assert client.db is not None
        all_docs = client.db.get_all_documents()
        assert len(all_docs) >= 1
        print(f"  OK (db_docs={len(all_docs)})")

        # 5. Single-document search
        print("\n[5/6] Single-document search...")
        result = await client.search("What features does PageIndex support?")
        print(f"  mode={result['mode']}, confidence={result['confidence']}")
        print(f"  answer={result['answer'][:100]}...")
        assert result["mode"] == "single"
        assert result["answer"]
        print("  OK")

        # 6. Multi-document search (with only 1 doc, falls back to single-doc shortcut)
        print("\n[6/6] Search with pages field...")
        assert "pages" in result
        print(f"  pages={result['pages']}")
        print("  OK")

        print("\n" + "=" * 50)
        print("All smoke tests passed!")
        print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
