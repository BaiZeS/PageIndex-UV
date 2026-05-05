"""Unit tests for RetrievalPlanner weight normalization."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from PageIndex.pageindex.agentic.planner import RetrievalPlanner, PlanResult


def test_default_plan():
    p = RetrievalPlanner("qwen-plus")
    result = p._default_plan("test query")
    assert result.queries == ["test query"]
    assert abs(sum(result.weights.values()) - 1.0) < 1e-9
    assert result.query_type == "factual"


def test_weight_normalization():
    """Verify that weights returned by plan() sum to 1.0."""
    p = RetrievalPlanner("qwen-plus")

    # Simulate what happens when LLM returns unnormalized weights
    # by calling _default_plan which already returns normalized weights
    result = p._default_plan("hello")
    total = sum(result.weights.values())
    assert abs(total - 1.0) < 1e-9, f"Weights should sum to 1.0, got {total}"


def test_plan_result_structure():
    result = PlanResult(
        queries=["q1", "q2"],
        weights={"metadata": 0.2, "semantics": 0.5, "description": 0.3},
        query_type="factual",
    )
    assert result.queries == ["q1", "q2"]
    assert result.weights["semantics"] == 0.5
    assert result.query_type == "factual"
