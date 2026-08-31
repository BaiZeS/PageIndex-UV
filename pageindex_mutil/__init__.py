"""PageIndex-UV package.

PEP 562 lazy attribute access (T32.1): the historical eager imports pulled
PyPDF2/openai/tiktoken (via .client → .page_index) into EVERY import of
``pageindex_mutil`` — including tests that only exercise ``agentic.enhance``
or ``super_tree``. That is what forced ~21 test files into runtime
``sys.modules`` stub-seeding (stubs clobbering each other, ordering
fragility). With lazy attributes, ``import pageindex_mutil`` is now
dependency-free; heavy submodules load on first attribute access, and tests
import the real modules directly.
"""
from importlib import import_module

__all__ = [
    "page_index_main",
    "page_index",
    "md_to_tree",
    "get_document",
    "get_document_structure",
    "get_page_content",
    "PageIndexClient",
]

_LAZY = {
    "page_index_main": (".page_index", "page_index_main"),
    "page_index": (".page_index", "page_index"),
    "md_to_tree": (".page_index_md", "md_to_tree"),
    "get_document": (".retrieve", "get_document"),
    "get_document_structure": (".retrieve", "get_document_structure"),
    "get_page_content": (".retrieve", "get_page_content"),
    "PageIndexClient": (".client", "PageIndexClient"),
}


def __getattr__(name):
    try:
        rel_mod, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = import_module(rel_mod, __package__)
    return getattr(module, attr)


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))
