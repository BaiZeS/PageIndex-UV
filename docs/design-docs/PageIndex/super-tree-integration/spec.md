# Feature: Super-Tree Retrieval Integration

**Status**: Quick Draft
**Date**: 2026-05-06

---

## 1. Background

v2's L1 three-strategy parallel router (Metadata + Semantics + Description) works but has redundant LLM calls. The new Super-Tree architecture replaces L1 with a single LLM reasoning pass over a condensed "super tree" of candidate documents.

## 2. Goals

- Replace v2 L1 parallel strategies with Super-Tree LLM selection when available
- Maintain v2 as fallback on any Super-Tree failure
- Zero breaking changes to existing PageIndexClient API

## 3. Design

### 3.1 Architecture

```
query
  │
  ▼
┌─────────────────────────────────────────┐
│ L0: Dual-channel prefilter              │
│   Channel A: ClosetIndex tag matching   │
│   Channel B: KeywordIndex jieba inverted│
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ L1: SuperTreeIndex.select_documents()   │
│   Build mini-TOC + KB Identity context  │
│   Single LLM call selects 3-5 docs      │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ L2/L3: Act (reuse existing _act_tree_search)
│   Tree search on selected documents     │
│   → generate_answer → Verify            │
└─────────────────────────────────────────┘
```

### 3.2 Fallback

If `super_tree_index` is None or any step raises, fall back to `_search_v2()` (the existing Plan→Route→Act→Verify pipeline).

## 4. Changes

| File | Action | Description |
|------|--------|-------------|
| `PageIndex/pageindex/agentic/router.py` | Modify | Add Super-Tree path, extract v2 to `_search_v2` |
| `PageIndex/pageindex/client.py` | Modify | Init `SuperTreeIndex`, call `on_document_added` in `index()` |
| `PageIndex/tests/test_router.py` | Create | Test Super-Tree routing and fallback |
| `PageIndex/tests/test_client_integration.py` | Create | Test client integration |

## 5. Acceptance

- `test_router.py` passes
- `test_client_integration.py` passes
- `test_super_tree.py` still passes
