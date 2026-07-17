"""LiteParse-based multi-format document parser.

Converts DOCX/PPTX/XLSX/ODT/images/HTML/etc. to Markdown via LiteParse,
then feeds the Markdown into the existing md_to_tree() pipeline.
"""

import logging
import os
import tempfile

from .utils import ConfigLoader

logger = logging.getLogger(__name__)

# Extensions handled by LiteParse (not PDF/MD which have dedicated parsers)
LITEPARSE_EXTENSIONS = frozenset({
    '.docx', '.doc',
    '.pptx', '.ppt',
    '.xlsx', '.xls',
    '.odt', '.ods', '.odp',
    '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp',
    '.html', '.htm',
    '.txt', '.csv', '.rtf',
})


def is_liteparse_format(file_path: str) -> bool:
    """Check if a file should be parsed via LiteParse."""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in LITEPARSE_EXTENSIONS


def _get_liteparse_config():
    """Read LiteParse settings from config.yaml."""
    try:
        cfg = ConfigLoader().load(None)
        return {
            'ocr_enabled': getattr(cfg, 'liteparse_ocr_enabled', True),
            'ocr_language': getattr(cfg, 'liteparse_ocr_language', 'eng'),
            'dpi': getattr(cfg, 'liteparse_dpi', 150),
        }
    except Exception:
        return {'ocr_enabled': True, 'ocr_language': 'eng', 'dpi': 150}


def parse_to_markdown(file_path: str) -> str:
    """Parse a file to Markdown text using LiteParse."""
    from liteparse import LiteParse

    config = _get_liteparse_config()
    parser = LiteParse(
        output_format="markdown",
        ocr_enabled=config['ocr_enabled'],
        ocr_language=config['ocr_language'],
        dpi=config['dpi'],
        image_mode="placeholder",
        extract_links=True,
    )

    result = parser.parse(file_path)
    return result.text or ""


async def liteparse_to_tree(file_path: str, model=None, **kwargs):
    """Parse a multi-format file to a document tree.

    Uses LiteParse to convert to Markdown, then feeds into md_to_tree().

    Returns dict with keys: doc_name, doc_description, structure, page_count/line_count.
    """
    from .page_index_md import md_to_tree

    logger.info("LiteParse: converting %s to Markdown...", file_path)

    # Parse to Markdown
    markdown_text = parse_to_markdown(file_path)
    if not markdown_text.strip():
        logger.warning("LiteParse produced empty output for %s", file_path)
        return {
            'doc_name': os.path.splitext(os.path.basename(file_path))[0],
            'doc_description': '',
            'line_count': 0,
            'structure': [],
        }

    # Write to temp file for md_to_tree
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.md', encoding='utf-8', delete=False
    ) as tmp:
        tmp.write(markdown_text)
        tmp_path = tmp.name

    try:
        # Feed into existing MD pipeline
        result = await md_to_tree(
            md_path=tmp_path,
            if_thinning=kwargs.get('if_thinning', False),
            if_add_node_summary=kwargs.get('if_add_node_summary', 'yes'),
            summary_token_threshold=kwargs.get('summary_token_threshold', 200),
            model=model,
            if_add_doc_description=kwargs.get('if_add_doc_description', 'yes'),
            if_add_node_text=kwargs.get('if_add_node_text', 'yes'),
            if_add_node_id=kwargs.get('if_add_node_id', 'yes'),
        )

        # Override doc_name with original filename
        result['doc_name'] = os.path.splitext(os.path.basename(file_path))[0]
        return result
    finally:
        os.unlink(tmp_path)
