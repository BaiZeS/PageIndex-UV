"""Shared test fixtures for PageIndex-UV."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def tmp_db():
    """Create a temporary SQLite database for testing."""
    from db import PageIndexDB
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = PageIndexDB(path)
    yield db
    db.close()
    os.unlink(path)


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_pdf(project_root):
    """Return path to a test PDF if available, else skip."""
    pdf_dir = project_root / "tests" / "pdfs"
    if not pdf_dir.exists():
        pytest.skip("tests/pdfs/ directory not found")
    pdfs = list(pdf_dir.glob("*.pdf"))
    if not pdfs:
        pytest.skip("No PDF files in tests/pdfs/")
    return str(pdfs[0])
