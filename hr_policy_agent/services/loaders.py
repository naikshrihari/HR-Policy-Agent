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

    from ..config import get_settings
    settings = get_settings()
    ocr_enabled = settings.ocr_enabled
    ocr_state = {"ok": ocr_enabled, "doc": None}  # lazily opened fitz doc; disabled on failure

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # A page with little extractable text is likely image-based (scanned page or a
        # graphic with the text baked in) -> OCR it if enabled.
        if ocr_state["ok"] and len(text.strip()) < settings.ocr_min_chars:
            ocr_text = _ocr_pdf_page(path, i, settings, ocr_state)
            if len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text
        if text.strip():
            pages.append(clean_pdf_text(text))
    if ocr_state.get("doc") is not None:
        try:
            ocr_state["doc"].close()
        except Exception:
            pass
    return "\n\n".join(pages)


_OCR_WARNED = False


def _ocr_pdf_page(path: str, page_index: int, settings, ocr_state) -> str:
    """OCR a single PDF page (rendered via PyMuPDF) with Tesseract.

    Returns "" and disables OCR for the rest of the run if the OCR stack isn't
    available (PyMuPDF/pytesseract not installed, or the Tesseract binary missing),
    printing a single clear message so ingest still completes on the text it has.
    """
    global _OCR_WARNED
    try:
        import io

        import fitz  # PyMuPDF — renders pages to images without an external poppler
        import pytesseract
        from PIL import Image
    except ImportError:
        if not _OCR_WARNED:
            _OCR_WARNED = True
            print("  [ocr] OCR is enabled but its packages are missing — install with: "
                  "pip install '.[ocr]'  (and the Tesseract binary). Skipping OCR.")
        ocr_state["ok"] = False
        return ""

    try:
        if ocr_state["doc"] is None:
            ocr_state["doc"] = fitz.open(path)
        page = ocr_state["doc"][page_index]
        pix = page.get_pixmap(dpi=settings.ocr_dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, lang=settings.ocr_language) or ""
    except pytesseract.TesseractNotFoundError:
        if not _OCR_WARNED:
            _OCR_WARNED = True
            print("  [ocr] Tesseract binary not found. Install Tesseract-OCR and ensure it is "
                  "on PATH (Windows: the UB Mannheim installer). Skipping OCR.")
        ocr_state["ok"] = False
        return ""
    except Exception as exc:  # noqa: BLE001 - never let OCR abort ingest
        if not _OCR_WARNED:
            _OCR_WARNED = True
            print(f"  [ocr] OCR failed ({exc}); continuing with extracted text only.")
        ocr_state["ok"] = False
        return ""


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
