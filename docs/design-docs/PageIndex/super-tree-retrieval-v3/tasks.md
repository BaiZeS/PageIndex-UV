# Super-Tree Retrieval v3 — Tasks

## Task 1: Database Schema

- [x] Add `doc_keywords` table with keyword + field columns
- [x] Add `kb_identity` table with identity_text + doc_count columns
- [x] Add CRUD methods: insert_doc_keywords, delete_doc_keywords, match_doc_keywords
- [x] Add CRUD methods: get_kb_identity, set_kb_identity, get_document_by_id

## Task 2: KeywordIndex

- [x] Implement jieba tokenization with stopword filtering
- [x] Implement add_document / remove_document / search
- [x] Unit tests

## Task 3: KBIdentity

- [x] Implement lazy generation with LLM
- [x] Implement fallback (doc name list concatenation)
- [x] Implement invalidate lifecycle
- [x] Unit tests

## Task 4: SuperTreeIndex

- [x] Implement L0 prefilter (ClosetIndex + KeywordIndex dual channel)
- [x] Implement L1 select_documents (mini-TOC + KB Identity + LLM)
- [x] Implement _build_super_tree with child-count sorting
- [x] Implement token budget truncation
- [x] Implement lifecycle hooks (on_document_added / on_document_removed)
- [x] Implement backfill on init
- [x] Unit tests

## Task 5: Router Integration

- [x] Add _search_super_tree to AgenticRouter
- [x] Add _search_v2 (extract original v2 search)
- [x] search() tries Super-Tree first, catches Exception, falls back to v2
- [x] Unit tests

## Task 6: Client Integration

- [x] Initialize SuperTreeIndex in PageIndexClient.__init__
- [x] Call super_tree_index.on_document_added() during index()
- [x] Remove duplicate closet_index.add_document() call

## Task 7: Documentation Sync

- [x] Sync README.md with v3 architecture (L0→L1→L2→L3)
- [x] Write v3 spec.md and tasks.md
- [x] Commit and push
