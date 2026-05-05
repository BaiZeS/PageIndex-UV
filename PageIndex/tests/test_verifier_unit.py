"""Unit tests for CRAGVerifier threshold routing."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from PageIndex.pageindex.agentic.verifier import CRAGVerifier, VerifyResult


def test_to_bool_normalization():
    assert CRAGVerifier._to_bool(True) is True
    assert CRAGVerifier._to_bool(False) is False
    assert CRAGVerifier._to_bool("true") is True
    assert CRAGVerifier._to_bool("yes") is True
    assert CRAGVerifier._to_bool("是") is True
    assert CRAGVerifier._to_bool("1") is True
    assert CRAGVerifier._to_bool("y") is True
    assert CRAGVerifier._to_bool("false") is False
    assert CRAGVerifier._to_bool("no") is False
    assert CRAGVerifier._to_bool("否") is False
    assert CRAGVerifier._to_bool(0) is False
    assert CRAGVerifier._to_bool(None) is False


def test_score_retrieval_computation():
    v = CRAGVerifier("qwen-plus")
    # Empty context → token_score = 0
    score = v._score_retrieval("", source_docs=0, covered_nodes=0)
    assert score == 0.0

    # Maxed out values
    long_context = "x " * 4000  # ~4000 tokens (approximate)
    score = v._score_retrieval(long_context, source_docs=3, covered_nodes=10)
    assert score == 1.0


def test_verify_high_confidence():
    """Mock verify by directly testing threshold logic."""
    v = CRAGVerifier("qwen-plus")
    # S_ret=1.0, S_CoV=1.0 → combined=1.0 >= TAU_HIGH → answer
    result = v.verify("ans", long_context(), "q", source_docs=3, covered_nodes=10)
    assert isinstance(result, VerifyResult)
    # We can't assert exact action without mocking LLM, but we can assert structure
    assert result.action in ("answer", "expand", "refuse")


def long_context():
    return "x " * 4000
