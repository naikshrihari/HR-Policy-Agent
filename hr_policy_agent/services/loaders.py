"""Document loaders — extract plain text from .txt / .md / .pdf / .docx files.

Kept dependency-light: PDFs use ``pypdf`` and Word files use ``docx2txt`` (both
pure-Python, no torch).  Missing a parser raises a clear error telling you what to
install.
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

# pypdf logs a noisy "Ignoring wrong pointing object …" warning for minor structural
# quirks in many PDFs; extraction still works, so keep the console clean.
logging.getLogger("pypdf").setLevel(logging.ERROR)

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
            pages.append(clean_pdf_text(text))
    return "\n\n".join(pages)


import re as _re

# Standalone page-furniture lines (bare page numbers, or "SECTION NAME 53").
_PAGE_FURNITURE = _re.compile(r"^\s*(?:\d{1,4}|[A-Z][A-Z &/'-]{2,60}\s+\d{1,4})\s*$")


def clean_pdf_text(text: str) -> str:
    """Turn line-wrapped PDF text into flowing paragraphs.

    PDF extractors emit one ``\\n`` per visual line, so a paragraph arrives as many short
    lines and real paragraph breaks are lost. Without this, the chunker treats a whole
    page as one blob and hard-splits it mid-word ("si|gns", "thr|ough"), which wrecks
    both readability and embedding quality. This de-hyphenates across line breaks, drops
    bare page-number / running-header lines, and joins wrapped lines back into paragraphs
    (a blank line, or a line that ends a sentence, starts a new paragraph).
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _re.sub(r"-\n(?=\w)", "", text)  # de-hyphenate words split across lines

    paragraphs = []
    buf = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            if buf:
                paragraphs.append(" ".join(buf))
                buf = []
            continue
        if _PAGE_FURNITURE.match(line):
            continue
        buf.append(line)
        # A line ending a sentence closes the paragraph.
        if line.endswith((".", "!", "?", ":")):
            paragraphs.append(" ".join(buf))
            buf = []
    if buf:
        paragraphs.append(" ".join(buf))

    cleaned = "\n\n".join(p for p in (_re.sub(r"\s+", " ", p).strip() for p in paragraphs) if p)
    return cleaned


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
