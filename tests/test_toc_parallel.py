"""Tests for parallel TOC detection with early stop (T12).

Verifies that find_toc_pages uses ThreadPoolExecutor for the first 10 pages,
stops early when a TOC page is found, and falls back to sequential for
remaining pages.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pageindex_mutil.page_index import find_toc_pages


def _make_opt(toc_check_page_num=20, model=None):
    return SimpleNamespace(toc_check_page_num=toc_check_page_num, model=model)


def _make_page_list(n):
    return [(f"PAGE[{i}]",) for i in range(n)]


def _page_num(content):
    """Extract page number from PAGE[N] format."""
    return int(content.split("[")[1].split("]")[0])


class TestTocParallelDetection:
    """Parallel TOC detection finds TOC page correctly."""

    def test_parallel_finds_toc_on_page_2(self):
        """TOC on page 2 — detected via parallel path, sequential scan finds end."""
        def detect(content, model=None):
            return "yes" if _page_num(content) == 2 else "no"

        page_list = _make_page_list(10)
        result = find_toc_pages(0, page_list, _make_opt(), _detector=detect)

        assert result == [2]


class TestTocEarlyStop:
    """Early stop: not all pages checked when TOC is in first 10."""

    def test_early_stop_skips_pages_beyond_10(self):
        """When TOC is on page 3, pages beyond first batch not checked."""
        call_pages = []

        def track_calls(content, model=None):
            call_pages.append(_page_num(content))
            return "yes" if _page_num(content) == 3 else "no"

        page_list = _make_page_list(20)
        result = find_toc_pages(0, page_list, _make_opt(toc_check_page_num=20), _detector=track_calls)

        assert 3 in result
        # Pages 10-19 must NOT have been checked
        for p in range(11, 20):
            assert p not in call_pages, f"Page {p} should not have been checked"


class TestTocFallbackSequential:
    """Fallback: when no TOC in first 10 pages, sequential check continues."""

    def test_fallback_sequential_finds_toc_at_page_12(self):
        """TOC not in first 10 pages — sequential fallback finds it at page 12."""
        call_pages = []

        def track_calls(content, model=None):
            call_pages.append(_page_num(content))
            return "yes" if _page_num(content) == 12 else "no"

        page_list = _make_page_list(20)
        result = find_toc_pages(0, page_list, _make_opt(toc_check_page_num=20), _detector=track_calls)

        assert 12 in result
        assert 12 in call_pages

    def test_no_toc_returns_empty(self):
        """No TOC anywhere — returns empty list."""
        def detect(content, model=None):
            return "no"

        page_list = _make_page_list(5)
        result = find_toc_pages(0, page_list, _make_opt(toc_check_page_num=20), _detector=detect)

        assert result == []

    def test_short_pdf_fewer_than_10_pages(self):
        """PDF with fewer than 10 pages — all submitted in parallel."""
        def detect(content, model=None):
            return "yes" if _page_num(content) == 1 else "no"

        page_list = _make_page_list(3)
        result = find_toc_pages(0, page_list, _make_opt(), _detector=detect)

        assert 1 in result

    def test_multitoc_spanning_pages(self):
        """TOC spanning pages 1-3 — all three reported."""
        def detect(content, model=None):
            return "yes" if _page_num(content) in [1, 2, 3] else "no"

        page_list = _make_page_list(10)
        result = find_toc_pages(0, page_list, _make_opt(), _detector=detect)

        assert result == [1, 2, 3]
