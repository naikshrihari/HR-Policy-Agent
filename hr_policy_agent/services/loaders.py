"""Document loaders — extract plain text from .txt / .md / .pdf / .docx files.

Kept dependency-light: PDFs use ``pypdf`` and Word files use ``docx2txt`` (both
pure-Python, no torch).  Missing a parser raises a clear error telling you what to
install.
"""

from __future__ import annotations

import os
from typing import List, Tuple

SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf", ".docx")


def load_file_text(path: str) -> str:
    """Return the extracted text of a single document file."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".docx":
        return _load_docx(path)
    raise ValueError(f"Unsupported file type: {path}")


def _load_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Reading PDFs needs 'pypdf'. Install it with:  pip install '.[docs]'  (or  pip install pypdf)"
        ) from exc
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


def _load_docx(path: str) -> str:
    try:
        import docx2txt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Reading .docx needs 'docx2txt'. Install it with:  pip install '.[docs]'  (or  pip install docx2txt)"
        ) from exc
    return docx2txt.process(path) or ""


def iter_corpus_files(corpus_dir: str) -> List[Tuple[str, str]]:
    """Return ``[(absolute_path, filename), ...]`` for supported files under ``corpus_dir``."""
    found = []
    for root, _dirs, files in os.walk(corpus_dir):
        for fn in sorted(files):
            if fn.lower().endswith(SUPPORTED_EXTENSIONS):
                found.append((os.path.join(root, fn), fn))
    return found
