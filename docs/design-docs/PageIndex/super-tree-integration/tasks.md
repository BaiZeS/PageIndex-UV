# Super-Tree Integration Tasks

## Task 1: Modify router.py
- Add `SuperTreeIndex` import
- Extract existing `search()` body to `_search_v2()`
- Add `_search_super_tree()` method
- Update `search()` to try Super-Tree first, fallback to v2

## Task 2: Modify client.py
- Add `SuperTreeIndex` import
- Initialize `super_tree_index` in `__init__`
- Call `on_document_added` during `index()`

## Task 3: Create test_router.py
- Test `_weighted_rrf`
- Test `_search_super_tree` uses prefilter and select_documents
- Test search() routing and fallback

## Task 4: Create test_client_integration.py
- Test super_tree_index initialization
- Test on_document_added called during index

## Task 5: Run all tests and commit
- test_router.py, test_client_integration.py, test_super_tree.py
- Commit with message: `feat: integrate Super-Tree into AgenticRouter and PageIndexClient`
