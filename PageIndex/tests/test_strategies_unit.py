"""Unit tests for strategy failure fallback and edge cases."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from PageIndex.pageindex.agentic.strategies import MetadataStrategy


def test_metadata_strategy_empty_query():
    s = MetadataStrategy()
    result = s.search("", [])
    assert result == []


def test_metadata_strategy_no_docs():
    s = MetadataStrategy()
    result = s.search("pdf indexing", [])
    assert result == []


def test_metadata_strategy_scoring():
    s = MetadataStrategy()
    docs = [
        {"doc_id": "1", "doc_name": "PDF Guide", "description": "How to index PDFs"},
        {"doc_id": "2", "doc_name": "Markdown Tips", "description": "Writing markdown"},
        {"doc_id": "3", "doc_name": "Other Doc", "description": "Unrelated"},
    ]
    result = s.search("pdf index", docs)
    # jieba.lcut("pdf index") → ["pdf", " ", "index"]
    # doc 1: "pdf" in name (+2), "index" in desc (+1) = 3
    # doc 2: no match for "pdf" or "index"
    # doc 3: no match
    assert len(result) == 1
    assert result[0] == ("1", 1)


def test_metadata_strategy_keyword_in_both():
    s = MetadataStrategy()
    docs = [
        {"doc_id": "1", "doc_name": "pdf", "description": "about pdf"},
    ]
    result = s.search("pdf", docs)
    assert len(result) == 1
    # +2 for name, +1 for description = score 3 → rank 1
    assert result[0] == ("1", 1)
